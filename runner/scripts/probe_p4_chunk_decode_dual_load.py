#!/usr/bin/env python3
import argparse
import gc
import glob
import json
import os
import resource
import time
import traceback
from pathlib import Path

import onnxruntime as ort
import numpy as np

from probe_p4_patha2_full_chunk_graph import build_feeds
from probe_p4_patha2_chunk_l2_attention import provider_counts_from_profile, qnn_register


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ROOT = Path(
    os.environ.get(
        "LFM25_STATE_DIR",
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        / "lfm25-350m-q6a-npu-edition",
    )
)
DEFAULT_CHUNK_CONTEXT = STATE_ROOT / "contexts" / "chunk" / "chunk_epcontext.onnx"
DEFAULT_DECODE_CONTEXT = STATE_ROOT / "contexts" / "decode" / "decode_epcontext.onnx"
DEFAULT_MASK_VALUE = -64.0
DEFAULT_ROPE_THETA = 1_000_000.0
DEFAULT_TOKENIZER = STATE_ROOT / "models" / "tokenizer" / "tokenizer.json"
DEFAULT_OFFICIAL_MODEL = STATE_ROOT / "models" / "host" / "rope_cache.npz"
DEFAULT_V0_RUNNER_DIR = SCRIPT_DIR
PROMPT = "The capital of Japan is"


class RunnerArgs:
    pass


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def proc_status() -> dict:
    out = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if key in {"VmRSS", "VmHWM", "VmSize"}:
                parts = value.split()
                if parts and parts[0].isdigit():
                    out[f"{key}_kb"] = int(parts[0])
                    out[f"{key}_mb"] = int(parts[0]) / 1024.0
    except Exception as exc:
        out["error"] = str(exc)
    return out


def read_number(path: str):
    try:
        return float(Path(path).read_text(encoding="utf-8").strip())
    except Exception:
        return None


def thermal_snapshot() -> dict:
    zones = []
    for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        temp = read_number(str(Path(path) / "temp"))
        if temp is None:
            continue
        typ = ""
        try:
            typ = (Path(path) / "type").read_text(encoding="utf-8").strip()
        except Exception:
            pass
        zones.append({"path": path, "type": typ, "temp_c": temp / 1000.0 if temp > 1000 else temp})
    return {"zones": zones, "max_zone": max(zones, key=lambda item: item["temp_c"], default=None)}


def power_snapshot() -> dict:
    readings = {}
    for pattern in [
        "/sys/class/power_supply/*/power_now",
        "/sys/class/power_supply/*/current_now",
        "/sys/class/power_supply/*/voltage_now",
        "/sys/class/hwmon/hwmon*/power*_input",
    ]:
        for path in sorted(glob.glob(pattern)):
            value = read_number(path)
            if value is not None:
                readings[path] = value
    return {
        "readings": readings,
        "direct_power_available": any("power" in Path(path).name for path in readings),
        "note": "Raw world-readable sysfs only; no sudo/root and no external power meter.",
    }


def make_session(model_path: Path, log_dir: Path, label: str, share_ep_contexts: bool):
    register_status, qdevs, provider_options = qnn_register()
    if not qdevs:
        raise RuntimeError("No QNN OrtEpDevice after provider registration")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_profiling = True
    so.profile_file_prefix = str(log_dir / f"{label}_profile")
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if share_ep_contexts:
        so.add_session_config_entry("ep.share_ep_contexts", "1")
    so.add_provider_for_devices(qdevs, provider_options)

    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model_path), sess_options=so, enable_fallback=False)
    create_s = time.perf_counter() - t0
    return sess, {
        "register_status": register_status,
        "provider_options": provider_options,
        "session_create_s": create_s,
        "session_providers": list(sess.get_providers()),
        "share_ep_contexts": share_ep_contexts,
    }


def finish_profile(sess, label: str) -> dict:
    try:
        profile = sess.end_profiling()
        return {
            "label": label,
            "profile": str(profile),
            "provider_counts": provider_counts_from_profile(profile),
        }
    except Exception as exc:
        return {"label": label, "profile_error": repr(exc)}


def run_chunk_once(sess, args) -> dict:
    feeds = build_feeds(sess, args.chunk, args.total_len - args.chunk, args.total_len, args.seed, args.mask_value, args.rope_theta)
    t0 = time.perf_counter()
    outputs = sess.run(None, feeds)
    run_s = time.perf_counter() - t0
    return {
        "run_s": run_s,
        "prefill_tok_s": args.chunk / run_s if run_s else None,
        "output_count": len(outputs),
    }


def make_runner_args(args):
    runner_args = RunnerArgs()
    runner_args.model = args.decode_context
    runner_args.tokenizer = args.tokenizer
    runner_args.official_model = args.official_model
    runner_args.embedding_backend = "official_raw"
    runner_args.max_new_tokens = args.decode_max_new_tokens
    runner_args.max_prompt_tokens = 96
    runner_args.temperature = 0.0
    runner_args.top_k = 10
    runner_args.top_p = 1.0
    runner_args.greedy = True
    runner_args.seed = args.seed
    runner_args.stop_token_id = []
    runner_args.tail_mask_value = -64.0
    runner_args.disable_rope_feed = False
    runner_args.disable_tail_mask_feed = False
    return runner_args


def run_decode_prompt(sess, args) -> dict:
    import sys

    sys.path.insert(0, str(args.v0_runner_dir))
    import run_practical_hybrid_prompt_v0 as v0

    input_specs, _ = v0.load_io_shapes(args.decode_context)
    rope_inputs = v0.discover_rope_inputs(args.decode_context)
    rope_context = None
    if rope_inputs:
        cos_cache, sin_cache = v0.load_rope_cache(args.official_model)
        rope_context = {"rope_inputs": rope_inputs, "cos_cache": cos_cache, "sin_cache": sin_cache}
    tokenizer = v0.Tokenizer.from_file(str(args.tokenizer))
    embedding_state = v0.load_embedding_state_for_backend("official_raw", args.official_model)
    runner_args = make_runner_args(args)
    rng = np.random.default_rng(args.seed)
    output_names = [out.name for out in sess.get_outputs()]
    t0 = time.perf_counter()
    item = v0.run_one_prompt(
        "dual_load_decode_prompt",
        PROMPT,
        sess,
        output_names,
        input_specs,
        embedding_state,
        tokenizer,
        runner_args,
        rng,
        rope_context,
    )
    wall_s = time.perf_counter() - t0
    return {"prompt": PROMPT, "wall_s": wall_s, "result": item}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default=time.strftime("%Y%m%d_%H%M%S_dual_load"))
    parser.add_argument("--log-root", type=Path, default=STATE_ROOT / "logs")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--chunk-context", type=Path, default=DEFAULT_CHUNK_CONTEXT)
    parser.add_argument("--decode-context", type=Path, default=DEFAULT_DECODE_CONTEXT)
    parser.add_argument("--share-ep-contexts", action="store_true")
    parser.add_argument("--run-chunk-once", action="store_true")
    parser.add_argument("--run-decode-prompt", action="store_true")
    parser.add_argument("--decode-max-new-tokens", type=int, default=4)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--official-model", type=Path, default=DEFAULT_OFFICIAL_MODEL)
    parser.add_argument("--v0-runner-dir", type=Path, default=DEFAULT_V0_RUNNER_DIR)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--total-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--mask-value", type=float, default=DEFAULT_MASK_VALUE)
    parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
    args = parser.parse_args()

    log_dir = args.log_dir or (args.log_root / f"p4_chunk_decode_dual_load_{args.timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": args.timestamp,
        "mode": "p4_chunk_decode_dual_load",
        "chunk_context": str(args.chunk_context),
        "decode_context": str(args.decode_context),
        "share_ep_contexts_requested": args.share_ep_contexts,
        "run_chunk_once": args.run_chunk_once,
        "v1_complete": False,
        "no_sudo_or_root": True,
        "power_before": power_snapshot(),
        "thermal_before": thermal_snapshot(),
        "rss_start_mb": rss_mb(),
        "proc_start": proc_status(),
    }
    chunk_sess = None
    decode_sess = None
    try:
        chunk_sess, chunk_info = make_session(args.chunk_context, log_dir, "chunk", args.share_ep_contexts)
        result["chunk_session"] = chunk_info
        result["after_chunk_load"] = {"rss_mb": rss_mb(), "proc": proc_status()}
        if args.run_chunk_once:
            result["chunk_run"] = run_chunk_once(chunk_sess, args)
            result["after_chunk_run"] = {"rss_mb": rss_mb(), "proc": proc_status()}

        decode_sess, decode_info = make_session(args.decode_context, log_dir, "decode", args.share_ep_contexts)
        result["decode_session"] = decode_info
        result["after_decode_load_both_alive"] = {"rss_mb": rss_mb(), "proc": proc_status()}
        if args.run_decode_prompt:
            result["decode_prompt"] = run_decode_prompt(decode_sess, args)
            result["after_decode_prompt_both_alive"] = {"rss_mb": rss_mb(), "proc": proc_status()}
        result["ok"] = True
    except Exception:
        result["ok"] = False
        result["error"] = traceback.format_exc()
    finally:
        profiles = []
        if chunk_sess is not None:
            profiles.append(finish_profile(chunk_sess, "chunk"))
        if decode_sess is not None:
            profiles.append(finish_profile(decode_sess, "decode"))
        result["profiles"] = profiles
        result["power_after"] = power_snapshot()
        result["thermal_after"] = thermal_snapshot()
        result["rss_final_mb"] = rss_mb()
        result["proc_final"] = proc_status()
        del decode_sess
        del chunk_sess
        gc.collect()
        (log_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        lines = [
            "# P4 Chunk + Decode Dual Load Probe",
            "",
            f"- share_ep_contexts_requested: `{args.share_ep_contexts}`",
            f"- ok: `{result.get('ok')}`",
            f"- chunk_session_create_s: `{(result.get('chunk_session') or {}).get('session_create_s')}`",
            f"- decode_session_create_s: `{(result.get('decode_session') or {}).get('session_create_s')}`",
            f"- after_decode_load_both_alive: `{result.get('after_decode_load_both_alive')}`",
            f"- after_decode_prompt_both_alive: `{result.get('after_decode_prompt_both_alive')}`",
            f"- decode_prompt_text: `{((result.get('decode_prompt') or {}).get('result') or {}).get('generated_text')}`",
            f"- profiles: `{result.get('profiles')}`",
            f"- error: `{(result.get('error') or '')[:300]}`",
            "",
        ]
        (log_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ok": result.get("ok"), "log_dir": str(log_dir), "result_json": str(log_dir / "result.json")}, ensure_ascii=False), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
