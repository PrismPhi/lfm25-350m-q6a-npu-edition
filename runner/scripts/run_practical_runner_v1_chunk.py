#!/usr/bin/env python3
"""Practical Runner V1 P6: ctx2048 chunk prefill + slim-cache decode EPContexts."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import probe_p4_chunk_prefill_handoff as handoff


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ROOT = Path(
    os.environ.get("LFM2_5_STATE_DIR")
    or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "lfm2.5-350m-q6a-qcs6490-qnn-npu"
)
DEFAULT_LOG_ROOT = STATE_ROOT / "logs"
DEFAULT_P6_CHUNK_CONTEXT = (
    STATE_ROOT / "contexts" / "chunk" / "chunk_epcontext.onnx"
)
DEFAULT_P6_DECODE_CONTEXT = (
    STATE_ROOT / "contexts" / "decode" / "decode_epcontext.onnx"
)
DEFAULT_P6_CHUNK = 16
DEFAULT_P6_TOTAL_LEN = 2048


def build_contexts(v0, args, chunk_sess, decode_sess):
    chunk_input_specs, _ = v0.load_io_shapes(args.chunk_context)
    decode_input_specs, _ = v0.load_io_shapes(args.decode_context)
    decode_rope_inputs = v0.discover_rope_inputs(args.decode_context)
    cos_cache, sin_cache = v0.load_rope_cache(args.official_model)
    embedding_state, embedding_info = handoff.load_embedding_state_for_runner(v0, args)
    return {
        "tokenizer": v0.Tokenizer.from_file(str(args.tokenizer)),
        "embedding_state": embedding_state,
        "embedding_info": embedding_info,
        "chunk_input_specs": chunk_input_specs,
        "decode_input_specs": decode_input_specs,
        "chunk_output_names": [out.name for out in chunk_sess.get_outputs()],
        "decode_output_names": [out.name for out in decode_sess.get_outputs()],
        "decode_rope_context": {
            "rope_inputs": decode_rope_inputs,
            "cos_cache": cos_cache,
            "sin_cache": sin_cache,
        },
    }


def run_prompt(v0, chunk_sess, decode_sess, contexts, prompt: str, args, name: str):
    task = {"name": name, "prompt": prompt, "max_new_tokens": args.max_new_tokens}
    return handoff.run_task(v0, chunk_sess, decode_sess, contexts, task, args)


def finish_profiles(chunk_sess, decode_sess):
    profiles = []
    if chunk_sess is not None:
        profiles.append(handoff.finish_profile(chunk_sess, "chunk"))
    if decode_sess is not None:
        profiles.append(handoff.finish_profile(decode_sess, "decode"))
    return profiles


def write_summary(log_dir: Path, result: dict):
    lines = [
        "# Practical Runner V1 P6 Slim Cache",
        "",
        f"- ok: `{result.get('ok')}`",
        f"- v1_complete: `{result.get('v1_complete')}`",
        f"- mode: `{result.get('mode')}`",
        f"- share_ep_contexts: `{result.get('share_ep_contexts_requested')}`",
        f"- qnn_only_by_profile: `{result.get('qnn_only_by_profile')}`",
        f"- chunk_session_create_s: `{(result.get('chunk_session') or {}).get('session_create_s')}`",
        f"- decode_session_create_s: `{(result.get('decode_session') or {}).get('session_create_s')}`",
        f"- peak_hwm_mb: `{((result.get('proc_final') or {}).get('VmHWM_mb'))}`",
        "",
        "## Generations",
        "",
    ]
    for item in result.get("generations", []):
        lines.extend(
            [
                f"### {item.get('name')}",
                "",
                f"- prompt: `{item.get('prompt')}`",
                f"- prompt_tokens: `{item.get('prompt_token_count')}`",
                f"- generated_tokens: `{len(item.get('generated_token_ids') or [])}`",
                f"- generated_text: `{item.get('generated_text')}`",
                f"- stream_matches_decode: `{item.get('stream_matches_decode')}`",
                f"- prefill_speed: `{item.get('prefill_speed')}`",
                f"- decode_speed: `{item.get('decode_speed')}`",
                f"- ttft_s_excluding_session_create: `{item.get('ttft_s_excluding_session_create')}`",
                f"- logical_cache_length_after_prefill: `{item.get('logical_cache_length_after_prefill')}`",
                f"- logical_cache_length_after_decode: `{item.get('logical_cache_length_after_decode')}`",
                "",
            ]
        )
    if result.get("error"):
        lines.extend(["## Error", "", "```", result.get("traceback") or result.get("error"), "```", ""])
    lines.extend(
        [
            "## Files",
            "",
            f"- result_json: `{log_dir / 'result.json'}`",
            f"- summary_md: `{log_dir / 'summary.md'}`",
        ]
    )
    (log_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_arg_parser():
    parser = argparse.ArgumentParser(description="Practical Runner V1 P6 ctx2048 chunk/slim-decode runner.")
    parser.add_argument("--prompt")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--system-prompt")
    parser.add_argument("--no-template", action="store_true")
    parser.add_argument("--chunk-context", type=Path, default=DEFAULT_P6_CHUNK_CONTEXT)
    parser.add_argument("--decode-context", type=Path, default=DEFAULT_P6_DECODE_CONTEXT)
    parser.add_argument("--tokenizer", type=Path, default=handoff.DEFAULT_TOKENIZER)
    parser.add_argument("--rope-cache", dest="official_model", type=Path, default=handoff.DEFAULT_OFFICIAL_MODEL)
    parser.add_argument("--model-root", type=Path, default=handoff.DEFAULT_MODEL_ROOT)
    parser.add_argument("--v0-runner-dir", type=Path, default=handoff.DEFAULT_V0_RUNNER_DIR)
    parser.add_argument("--embedding-backend", choices=["official_raw", "official_int8_rowwise"], default="official_int8_rowwise")
    parser.add_argument("--embedding-int8-dir", type=Path, default=handoff.DEFAULT_EMBEDDING_INT8_DIR)
    parser.add_argument("--build-embedding-int8-if-missing", action="store_true")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--timestamp")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_P6_TOTAL_LEN)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-token-id", action="append", type=int, default=[])
    parser.add_argument("--chunk", type=int, default=DEFAULT_P6_CHUNK)
    parser.add_argument("--total-len", type=int, default=DEFAULT_P6_TOTAL_LEN)
    parser.add_argument("--mask-value", type=float, default=handoff.DEFAULT_MASK_VALUE)
    parser.add_argument("--rope-theta", type=float, default=handoff.DEFAULT_ROPE_THETA)
    parser.add_argument("--share-ep-contexts", action="store_true")
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    return parser


def main() -> int:
    args = make_arg_parser().parse_args()
    if not args.interactive and not args.prompt:
        raise SystemExit("pass --prompt TEXT or --interactive")

    timestamp = args.timestamp or time.strftime("%Y%m%d_%H%M%S_chunk_v1")
    log_dir = args.log_root / f"practical_runner_v1_chunk_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": timestamp,
        "mode": "practical_runner_v1_p6_slim_cache",
        "v1_complete": True,
        "chunk_context": str(args.chunk_context),
        "decode_context": str(args.decode_context),
        "share_ep_contexts_requested": bool(args.share_ep_contexts),
        "sampling": {
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "greedy": bool(args.greedy),
        },
        "streaming": {"enabled": bool(args.stream), "stdout_token_stream": True},
        "no_sudo_or_root": True,
        "power_before": handoff.power_snapshot(),
        "thermal_before": handoff.thermal_snapshot(),
        "rss_start_mb": handoff.rss_mb(),
        "proc_start": handoff.proc_status(),
        "generations": [],
        "ok": False,
    }
    chunk_sess = None
    decode_sess = None
    try:
        v0 = handoff.import_v0(args.v0_runner_dir)
        chunk_sess, chunk_info = handoff.make_session(args.chunk_context, log_dir, "chunk", args.share_ep_contexts)
        result["chunk_session"] = chunk_info
        result["after_chunk_load"] = {"rss_mb": handoff.rss_mb(), "proc": handoff.proc_status()}
        decode_sess, decode_info = handoff.make_session(args.decode_context, log_dir, "decode", args.share_ep_contexts)
        result["decode_session"] = decode_info
        result["after_decode_load_both_alive"] = {"rss_mb": handoff.rss_mb(), "proc": handoff.proc_status()}
        contexts = build_contexts(v0, args, chunk_sess, decode_sess)
        result["embedding_info"] = contexts.get("embedding_info")
        result["after_embedding_load"] = {"rss_mb": handoff.rss_mb(), "proc": handoff.proc_status()}

        if args.interactive:
            print("Practical Runner V1 chunk REPL ready. Ctrl-D to exit.", flush=True)
            for idx, line in enumerate(sys.stdin):
                prompt = line.strip()
                if not prompt:
                    continue
                result["generations"].append(run_prompt(v0, chunk_sess, decode_sess, contexts, prompt, args, f"repl_{idx}"))
        else:
            result["generations"].append(run_prompt(v0, chunk_sess, decode_sess, contexts, args.prompt, args, "prompt"))

        result["after_generation"] = {"rss_mb": handoff.rss_mb(), "proc": handoff.proc_status()}
    except Exception:
        result["error"] = traceback.format_exc()
        result["error_type"] = "Exception"
    finally:
        profiles = finish_profiles(chunk_sess, decode_sess)
        result["profiles"] = profiles
        result["qnn_only_by_profile"] = handoff.profile_qnn_only(profiles)
        chunk_required = any(
            item.get("prefill_strategy") != "decode_only_initial_partial"
            for item in result.get("generations", [])
        )
        result["chunk_graph_executed"] = bool(chunk_required)
        result["power_after"] = handoff.power_snapshot()
        result["thermal_after"] = handoff.thermal_snapshot()
        result["rss_final_mb"] = handoff.rss_mb()
        result["proc_final"] = handoff.proc_status()
        result["ok"] = bool(
            result.get("generations")
            and all(item.get("stream_matches_decode") for item in result.get("generations", []))
            and result["qnn_only_by_profile"].get("decode")
            and ((not chunk_required) or result["qnn_only_by_profile"].get("chunk"))
            and not result.get("error")
        )
        del decode_sess
        del chunk_sess
        gc.collect()
        (log_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        write_summary(log_dir, result)

    print(json.dumps({"ok": result.get("ok"), "log_dir": str(log_dir), "summary": str(log_dir / "summary.md")}, ensure_ascii=False), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
