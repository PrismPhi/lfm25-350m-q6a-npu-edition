#!/usr/bin/env python3
"""Prompt -> chunk prefill EPContext -> decode EPContext handoff probe.

This is a P4 integration probe, not a V1 completion declaration. It keeps the
model compute on QNN EPContexts with CPU-side tokenization, embedding lookup,
sampling, and cache bookkeeping.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from probe_p4_chunk_decode_dual_load import (
    DEFAULT_CHUNK_CONTEXT,
    DEFAULT_DECODE_CONTEXT,
    DEFAULT_OFFICIAL_MODEL,
    DEFAULT_TOKENIZER,
    DEFAULT_V0_RUNNER_DIR,
    finish_profile,
    make_session,
    power_snapshot,
    proc_status,
    rss_mb,
    thermal_snapshot,
)
from probe_p4_patha2_full_chunk_graph import (
    ATTN_LAYERS,
    CONV_LAYERS,
    DEFAULT_MASK_VALUE,
    DEFAULT_ROPE_THETA,
    DEFAULT_TOTAL_LEN,
    HIDDEN,
    HEADS,
    HEAD_DIM,
    KV_HEADS,
    PAST_CONV,
)


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ROOT = Path(
    os.environ.get(
        "LFM25_STATE_DIR",
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        / "lfm25-350m-q6a-npu-edition",
    )
)
DEFAULT_LOG_ROOT = STATE_ROOT / "logs"
DEFAULT_MODEL_ROOT = STATE_ROOT / "models" / "tokenizer"
DEFAULT_EMBEDDING_INT8_DIR = STATE_ROOT / "models" / "host" / "embedding_int8_rowwise"
DEFAULT_CHUNK = 16
DEFAULT_CHUNK_PAST = 112
VOCAB = 65536

LONG_MEMO = """Summarize the following product operations memo in three concise bullet points.

Memo:
The inventory application team is preparing a reliability release for store managers.
During the last batch import, several stores saw a delay before stock counts appeared
on handheld devices. The backend eventually applied all updates, but managers could
not tell whether the import was still running or had failed. The mobile client also
lost one saved draft when the network dropped during a basement stock check. The
support team wants clearer status messages, retry after a failed save, and an audit
history page that can be filtered by product, store, staff member, and date. The
engineering team already improved the average API response time to 180 ms, but peak
latency still reaches 800 ms during regional uploads. The next sprint should focus on
incremental sync, visible import progress, retryable draft saves, and history search.
CSV export and permission settings must keep working exactly as before.
"""

BENCHMARK_TASKS = [
    {
        "name": "en_capital",
        "prompt": "The capital of Japan is",
        "max_new_tokens": 32,
        "expect": "Tokyo",
    },
    {
        "name": "json_capital",
        "prompt": 'Return JSON only with exactly these keys: "capital" and "country". The country is Japan.',
        "max_new_tokens": 64,
        "json": True,
        "expect": "capital=Tokyo,country=Japan",
    },
    {
        "name": "practical_email",
        "prompt": (
            "Write a short polite email in English to reschedule a meeting from "
            "Tuesday afternoon to Wednesday morning. Keep it under 70 words."
        ),
        "max_new_tokens": 96,
        "expect": "short reschedule email",
    },
    {
        "name": "long_memo_summary",
        "prompt": LONG_MEMO,
        "max_new_tokens": 128,
        "long": True,
        "expect": "three bullet summary",
    },
]


class GenArgs:
    pass


def shape_to_list(shape):
    return [int(x) if isinstance(x, (int, np.integer)) else 1 for x in shape]


def import_v0(v0_runner_dir: Path):
    if str(v0_runner_dir) not in sys.path:
        sys.path.insert(0, str(v0_runner_dir))
    import run_practical_hybrid_prompt_v0 as v0

    return v0


def make_internal_args(args, max_new_tokens: int):
    out = GenArgs()
    out.max_new_tokens = int(max_new_tokens)
    out.max_prompt_tokens = int(args.max_prompt_tokens)
    out.temperature = float(args.temperature)
    out.top_k = int(args.top_k)
    out.top_p = float(args.top_p)
    out.greedy = bool(args.greedy)
    out.seed = int(args.seed)
    out.stop_token_id = list(args.stop_token_id or [])
    out.tail_mask_value = float(args.mask_value)
    out.disable_rope_feed = False
    out.disable_tail_mask_feed = False
    out.chunk = int(args.chunk)
    out.total_len = int(args.total_len)
    out.mask_value = float(args.mask_value)
    out.rope_theta = float(args.rope_theta)
    out.repetition_penalty = float(getattr(args, "repetition_penalty", 1.0))
    out.stream = bool(getattr(args, "stream", False))
    return out


def embedding_int8_paths(root: Path):
    return {
        "root": Path(root),
        "q": Path(root) / "model_embed_tokens_weight_rowwise_int8.npy",
        "scale": Path(root) / "model_embed_tokens_weight_rowwise_scale.npy",
        "metadata": Path(root) / "metadata.json",
    }


def build_official_int8_embedding(v0, official_model: Path, out_dir: Path):
    paths = embedding_int8_paths(out_dir)
    out_dir = paths["root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if paths["q"].exists() and paths["scale"].exists() and paths["metadata"].exists():
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))

    state = v0.load_embedding_state_for_backend("official_raw", official_model)
    weight = np.asarray(state["official_raw"], dtype=np.float32)
    if tuple(weight.shape) != (VOCAB, HIDDEN):
        raise RuntimeError(f"unexpected official embedding shape for int8 build: {weight.shape}")
    max_abs = np.max(np.abs(weight), axis=1).astype(np.float32)
    scale = np.maximum(max_abs / np.float32(127.0), np.float32(1.0e-8)).astype(np.float32)
    q = np.rint(weight / scale[:, None]).clip(-127, 127).astype(np.int8)
    np.save(paths["q"], q)
    np.save(paths["scale"], scale)
    sample_ids = np.asarray([0, 1, 2, 6, 730, 1098, 19444, 63301], dtype=np.int64)
    sample = weight[sample_ids]
    recon = q[sample_ids].astype(np.float32) * scale[sample_ids, None]
    diff = sample - recon
    metadata = {
        "format": "official_embed_tokens_rowwise_symmetric_int8",
        "official_model": str(official_model),
        "q_path": str(paths["q"]),
        "scale_path": str(paths["scale"]),
        "shape": [int(x) for x in q.shape],
        "q_dtype": "int8",
        "scale_dtype": "float32",
        "size_bytes_q": int(paths["q"].stat().st_size),
        "size_bytes_scale": int(paths["scale"].stat().st_size),
        "sample_ids": [int(x) for x in sample_ids],
        "sample_max_abs_error": float(np.max(np.abs(diff))),
        "sample_mean_abs_error": float(np.mean(np.abs(diff))),
        "no_sudo_or_root": True,
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_embedding_state_for_runner(v0, args):
    backend = getattr(args, "embedding_backend", "official_raw")
    if backend == "official_raw":
        return v0.load_embedding_state_for_backend("official_raw", args.official_model), {
            "backend": "official_raw",
            "storage": "float32_resident",
        }
    if backend == "official_int8_rowwise":
        paths = embedding_int8_paths(args.embedding_int8_dir)
        if not (paths["q"].exists() and paths["scale"].exists() and paths["metadata"].exists()):
            if getattr(args, "build_embedding_int8_if_missing", False):
                build_official_int8_embedding(v0, args.official_model, args.embedding_int8_dir)
            else:
                raise FileNotFoundError(f"missing int8 embedding artifacts under {args.embedding_int8_dir}")
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        q = np.load(paths["q"], mmap_mode="r")
        scale = np.load(paths["scale"], mmap_mode="r")
        if tuple(q.shape) != (VOCAB, HIDDEN) or q.dtype != np.int8:
            raise RuntimeError(f"unexpected int8 embedding q array: shape={q.shape} dtype={q.dtype}")
        if tuple(scale.shape) != (VOCAB,) or scale.dtype != np.float32:
            raise RuntimeError(f"unexpected int8 embedding scale array: shape={scale.shape} dtype={scale.dtype}")
        return {
            "official_int8_rowwise": {
                "q": q,
                "scale": scale,
                "metadata": metadata,
            }
        }, {
            "backend": "official_int8_rowwise",
            "storage": "int8_mmap_plus_rowwise_float32_scale",
            "metadata": metadata,
        }
    raise ValueError(f"unknown embedding backend: {backend}")


def embedding_backend_name(embedding_state):
    if "official_int8_rowwise" in embedding_state:
        return "official_int8_rowwise"
    if "official_raw" in embedding_state:
        return "official_raw"
    if "weight" in embedding_state:
        return "practical_rmsnorm"
    return "unknown"


def embedding_vector(embedding_state, token_id: int):
    if "official_int8_rowwise" in embedding_state:
        item = embedding_state["official_int8_rowwise"]
        q = item["q"]
        scale = item["scale"]
        if token_id < 0 or token_id >= q.shape[0]:
            raise ValueError(f"token id {token_id} outside int8 embedding range")
        return (np.asarray(q[int(token_id)], dtype=np.float32) * np.float32(scale[int(token_id)])).astype(np.float32, copy=False)
    if "official_raw" in embedding_state:
        weight = embedding_state["official_raw"]
        if token_id < 0 or token_id >= weight.shape[0]:
            raise ValueError(f"token id {token_id} outside official embedding range")
        return np.asarray(weight[int(token_id)], dtype=np.float32)
    raise RuntimeError(f"embedding backend {embedding_backend_name(embedding_state)} requires v0.set_hidden path")


def set_hidden(v0, feed, embedding_state, token_id: int):
    if "official_int8_rowwise" in embedding_state or "official_raw" in embedding_state:
        feed["x"] = embedding_vector(embedding_state, int(token_id)).reshape(1, 1, HIDDEN)
        return
    v0.set_hidden(feed, embedding_state, int(token_id))


def apply_repetition_penalty(logits, token_ids, penalty: float):
    if not penalty or float(penalty) == 1.0:
        return np.asarray(logits)
    out = np.array(logits, dtype=np.float64, copy=True).reshape(-1)
    for token_id in set(int(x) for x in token_ids if 0 <= int(x) < out.shape[0]):
        if out[token_id] < 0:
            out[token_id] *= float(penalty)
        else:
            out[token_id] /= float(penalty)
    return out


def choose_next(v0, logits, tokenizer_vocab_size, args, rng, history_ids):
    adjusted = apply_repetition_penalty(logits, history_ids, getattr(args, "repetition_penalty", 1.0))
    return v0.choose_next(adjusted, tokenizer_vocab_size, args, rng)


def manual_chatml_template(prompt: str, system_prompt: str | None):
    parts = ["<|startoftext|>"]
    if system_prompt:
        parts.append("<|im_start|>system\n")
        parts.append(system_prompt)
        parts.append("<|im_end|>\n")
    parts.append("<|im_start|>user\n")
    parts.append(prompt)
    parts.append("<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def apply_chat_template(prompt: str, model_root: Path, system_prompt: str | None, use_template: bool):
    if not use_template:
        return prompt, {"mode": "raw", "template_applied": False}
    manual = manual_chatml_template(prompt, system_prompt)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return templated, {
            "mode": "official_chat_template",
            "template_applied": True,
            "model_root": str(model_root),
            "system_prompt_present": bool(system_prompt),
        }
    except Exception as exc:
        return manual, {
            "mode": "official_chat_template_manual_no_jinja",
            "template_applied": True,
            "manual_renderer": "simple_system_user_assistant_path_from_tokenizer_config",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_root": str(model_root),
            "system_prompt_present": bool(system_prompt),
        }


def weighted_speed(token_count: int, run_s):
    vals = [float(x) for x in run_s]
    total = sum(vals)
    return {
        "tokens": int(token_count),
        "runs": len(vals),
        "total_s": total,
        "min_s": min(vals) if vals else None,
        "avg_s": statistics.fmean(vals) if vals else None,
        "max_s": max(vals) if vals else None,
        "tok_per_s": (float(token_count) / total) if total > 0 else None,
    }


def rope_freq(theta: float):
    return np.asarray([theta ** (-i / (HEAD_DIM // 2)) for i in range(HEAD_DIM // 2)], dtype=np.float32)


def make_rope_rows(row_positions, heads: int, theta: float):
    positions = np.asarray(row_positions, dtype=np.float32).reshape(1, -1, 1)
    angles = positions * rope_freq(theta).reshape(1, 1, HEAD_DIM // 2)
    angles = np.concatenate([angles, angles], axis=-1).astype(np.float32)
    cos = np.tile(np.cos(angles).reshape(1, 1, positions.shape[1], HEAD_DIM), (1, heads, 1, 1))
    sin = np.tile(np.sin(angles).reshape(1, 1, positions.shape[1], HEAD_DIM), (1, heads, 1, 1))
    return cos.astype(np.float32), sin.astype(np.float32)


def make_chunk_tail_mask(
    chunk: int,
    past_len: int,
    total_len: int,
    past_valid: int,
    row_start: int,
    valid_tokens: int,
    mask_value: float,
):
    mask = np.full((1, HEADS, chunk, total_len), float(mask_value), dtype=np.float32)
    past_valid = max(0, min(int(past_valid), past_len))
    past_start = past_len - past_valid
    current_start = total_len - int(valid_tokens)
    for j in range(int(valid_tokens)):
        row = int(row_start) + j
        if past_valid > 0:
            mask[:, :, row, past_start:past_len] = 0.0
        mask[:, :, row, current_start : current_start + j + 1] = 0.0
    return mask


def make_zero_feed(input_specs):
    feed = {}
    for name, spec in input_specs.items():
        feed[name] = np.zeros(spec["shape"], dtype=spec["dtype"])
    return feed


def set_chunk_x(feed, embedding_state, token_ids, chunk: int):
    valid = len(token_ids)
    row_start = chunk - valid
    x = np.zeros((1, chunk, HIDDEN), dtype=np.float32)
    for offset, token_id in enumerate(token_ids):
        tid = int(token_id)
        x[0, row_start + offset, :] = embedding_vector(embedding_state, tid)
    feed["x"] = x
    return row_start


def update_chunk_cache_from_outputs(feed, output_names, outputs):
    by_name = {name: np.asarray(value) for name, value in zip(output_names, outputs)}
    for layer in CONV_LAYERS:
        key = f"past_conv{layer}"
        out = by_name.get(f"l{layer}_present_conv")
        if key in feed and out is not None:
            feed[key] = out.astype(np.float32, copy=False)
    for layer in ATTN_LAYERS:
        for kind in ["k", "v"]:
            key = f"past_{kind}{layer}"
            out = by_name.get(f"l{layer}attn_present_{kind}")
            if key in feed and out is not None:
                feed[key] = out[:, :, -feed[key].shape[2] :, :].astype(np.float32, copy=False)
    return by_name


def update_decode_cache_from_outputs(feed, output_names, outputs):
    """Update decode caches from either full present-KV or new-token-KV outputs."""
    by_name = {name: np.asarray(value) for name, value in zip(output_names, outputs)}
    for layer in CONV_LAYERS:
        key = f"past_conv{layer}"
        out = by_name.get(f"l{layer}_present_conv")
        if key in feed and out is not None:
            feed[key] = out.astype(np.float32, copy=False)
    for layer in ATTN_LAYERS:
        for kind in ["k", "v"]:
            key = f"past_{kind}{layer}"
            if key not in feed:
                continue
            present = by_name.get(f"l{layer}attn_present_{kind}")
            if present is not None:
                feed[key] = present[:, :, -feed[key].shape[2] :, :].astype(np.float32, copy=False)
                continue
            new = by_name.get(f"l{layer}attn_new_{kind}")
            if new is None:
                continue
            previous = np.asarray(feed[key], dtype=np.float32)
            new = np.asarray(new, dtype=np.float32)
            capacity = previous.shape[2]
            new_tokens = min(capacity, new.shape[2])
            if new_tokens >= capacity:
                feed[key] = new[:, :, -capacity:, :].astype(np.float32, copy=False)
            else:
                feed[key] = np.concatenate(
                    [previous[:, :, new_tokens:, :], new[:, :, -new_tokens:, :]], axis=2
                ).astype(np.float32, copy=False)
    return by_name


def compact_attention_past(previous_feed, by_name, logical_before: int, valid_tokens: int, capacity: int):
    overrides = {}
    valid_tokens = int(valid_tokens)
    logical_before = int(logical_before)
    capacity = int(capacity)
    for layer in ATTN_LAYERS:
        for kind in ["k", "v"]:
            key = f"past_{kind}{layer}"
            out_name = f"l{layer}attn_present_{kind}"
            new_name = f"l{layer}attn_new_{kind}"
            if key not in previous_feed or (out_name not in by_name and new_name not in by_name):
                continue
            previous = np.asarray(previous_feed[key], dtype=np.float32)
            if out_name in by_name:
                present = np.asarray(by_name[out_name], dtype=np.float32)
                new_actual = present[:, :, -valid_tokens:, :] if valid_tokens > 0 else present[:, :, 0:0, :]
            else:
                new = np.asarray(by_name[new_name], dtype=np.float32)
                new_actual = new[:, :, -valid_tokens:, :] if valid_tokens > 0 else new[:, :, 0:0, :]
            prior_keep = min(logical_before, previous.shape[2], max(0, capacity - valid_tokens))
            prior_actual = previous[:, :, -prior_keep:, :] if prior_keep > 0 else previous[:, :, 0:0, :]
            prefix = capacity - prior_keep - valid_tokens
            if prefix < 0:
                raise ValueError(f"negative compact prefix for {key}: capacity={capacity} prior={prior_keep} valid={valid_tokens}")
            zeros = np.zeros((previous.shape[0], previous.shape[1], prefix, previous.shape[3]), dtype=np.float32)
            overrides[key] = np.concatenate([zeros, prior_actual, new_actual], axis=2).astype(np.float32, copy=False)
    return overrides


def compact_conv_past(previous_feed, by_name, logical_before: int, valid_tokens: int):
    overrides = {}
    valid_tokens = int(valid_tokens)
    logical_before = int(logical_before)
    capacity = PAST_CONV
    for layer in CONV_LAYERS:
        key = f"past_conv{layer}"
        out_name = f"l{layer}_present_conv"
        if key not in previous_feed or out_name not in by_name:
            continue
        previous = np.asarray(previous_feed[key], dtype=np.float32)
        present = np.asarray(by_name[out_name], dtype=np.float32)
        actual_new = min(valid_tokens, capacity)
        new_actual = present[:, :, -actual_new:] if actual_new > 0 else present[:, :, 0:0]
        prior_keep = min(logical_before, previous.shape[2], max(0, capacity - actual_new))
        prior_actual = previous[:, :, -prior_keep:] if prior_keep > 0 else previous[:, :, 0:0]
        prefix = capacity - prior_keep - actual_new
        zeros = np.zeros((previous.shape[0], previous.shape[1], prefix), dtype=np.float32)
        overrides[key] = np.concatenate([zeros, prior_actual, new_actual], axis=2).astype(np.float32, copy=False)
    return overrides


def apply_cache_overrides(feed, overrides):
    for key, value in (overrides or {}).items():
        if key in feed:
            if tuple(feed[key].shape) != tuple(value.shape):
                raise ValueError(f"cache override shape mismatch for {key}: {value.shape} vs {feed[key].shape}")
            feed[key] = value.astype(np.float32, copy=False)


def strip_json_fence(text: str) -> str:
    s = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else s


def json_check(text: str):
    stripped = strip_json_fence(text)
    try:
        parsed = json.loads(stripped)
    except Exception as exc:
        return {"syntax_valid": False, "schema_valid": False, "error": str(exc), "stripped": stripped[:300]}
    schema_valid = isinstance(parsed, dict) and set(parsed.keys()) == {"capital", "country"}
    if schema_valid:
        schema_valid = str(parsed.get("capital", "")).lower() == "tokyo" and str(parsed.get("country", "")).lower() == "japan"
    return {"syntax_valid": True, "schema_valid": bool(schema_valid), "parsed": parsed}


def run_chunk_prefill(
    chunk_sess,
    chunk_input_specs,
    chunk_output_names,
    ids,
    embedding_state,
    args,
):
    feed = make_zero_feed(chunk_input_specs)
    chunk = int(args.chunk)
    total_len = int(args.total_len)
    past_len = total_len - chunk
    if past_len < 0:
        raise ValueError(f"invalid chunk/total_len: chunk={chunk} total_len={total_len}")
    attn_past_lens = {
        int(spec["shape"][2])
        for name, spec in chunk_input_specs.items()
        if (name.startswith("past_k") or name.startswith("past_v")) and len(spec.get("shape", [])) >= 3
    }
    if len(attn_past_lens) > 1:
        raise ValueError(f"mixed chunk attention past lengths are not supported: {sorted(attn_past_lens)}")
    if attn_past_lens and next(iter(attn_past_lens)) != past_len:
        raise ValueError(
            f"chunk context past_len mismatch: args imply {past_len}, "
            f"context has {next(iter(attn_past_lens))}"
        )
    mask_total_lens = {
        int(spec["shape"][3])
        for name, spec in chunk_input_specs.items()
        if name.endswith("attn_tail_mask") and len(spec.get("shape", [])) >= 4
    }
    if len(mask_total_lens) > 1:
        raise ValueError(f"mixed chunk tail-mask lengths are not supported: {sorted(mask_total_lens)}")
    if mask_total_lens and next(iter(mask_total_lens)) != total_len:
        raise ValueError(
            f"chunk context total_len mismatch: args imply {total_len}, "
            f"context has {next(iter(mask_total_lens))}"
        )

    logical = 0
    run_s = []
    chunk_records = []
    last_outputs = None
    last_by_name = None
    decode_cache_overrides = {}
    prefill_t0 = time.perf_counter()
    groups = [ids[i : i + chunk] for i in range(0, len(ids), chunk)]
    for index, group in enumerate(groups):
        valid = len(group)
        previous_feed = {key: np.asarray(value).copy() for key, value in feed.items() if key.startswith("past_")}
        row_start = set_chunk_x(feed, embedding_state, group, chunk)
        row_positions = np.zeros((chunk,), dtype=np.float32)
        row_positions[row_start:] = np.arange(logical, logical + valid, dtype=np.float32)
        past_valid = min(logical, past_len)
        mask = make_chunk_tail_mask(
            chunk,
            past_len,
            total_len,
            past_valid,
            row_start,
            valid,
            args.mask_value,
        )
        for layer in ATTN_LAYERS:
            q_cos, q_sin = make_rope_rows(row_positions, HEADS, args.rope_theta)
            k_cos, k_sin = make_rope_rows(row_positions, KV_HEADS, args.rope_theta)
            feed[f"l{layer}attn_q_rope_cos_dq"] = q_cos
            feed[f"l{layer}attn_q_rope_sin_dq"] = q_sin
            feed[f"l{layer}attn_k_rope_cos_dq"] = k_cos
            feed[f"l{layer}attn_k_rope_sin_dq"] = k_sin
            feed[f"l{layer}attn_tail_mask"] = mask
        t0 = time.perf_counter()
        last_outputs = chunk_sess.run(None, feed)
        elapsed = time.perf_counter() - t0
        run_s.append(elapsed)
        last_by_name = update_chunk_cache_from_outputs(feed, chunk_output_names, last_outputs)
        chunk_overrides = compact_attention_past(previous_feed, last_by_name, logical, valid, past_len)
        chunk_overrides.update(compact_conv_past(previous_feed, last_by_name, logical, valid))
        apply_cache_overrides(feed, chunk_overrides)
        decode_cache_overrides = compact_attention_past(previous_feed, last_by_name, logical, valid, total_len - 1)
        decode_cache_overrides.update(compact_conv_past(previous_feed, last_by_name, logical, valid))
        chunk_records.append(
            {
                "index": index,
                "valid_tokens": valid,
                "row_start": row_start,
                "logical_start": logical,
                "logical_end": logical + valid,
                "past_valid_before": past_valid,
                "run_s": elapsed,
            }
        )
        logical += valid

    if last_by_name is None or "logits" not in last_by_name:
        raise RuntimeError("chunk graph did not return logits")
    return {
        "chunk_feed_after": feed,
        "last_outputs": last_outputs,
        "last_logits": np.asarray(last_by_name["logits"]),
        "logical_cache_length": logical,
        "prefill_wall_s": time.perf_counter() - prefill_t0,
        "prefill_run_s": run_s,
        "chunk_records": chunk_records,
        "prefill_speed": weighted_speed(len(ids), run_s),
        "decode_cache_overrides": decode_cache_overrides,
    }


def prepare_decode_feed_from_chunk(v0, decode_input_specs, decode_output_names, chunk_output_names, chunk_outputs, cache_overrides=None):
    feed = v0.make_initial_feed(decode_input_specs)
    v0.update_cache_from_outputs(feed, chunk_output_names, chunk_outputs)
    apply_cache_overrides(feed, cache_overrides)
    return feed


def run_decode_prefill(v0, decode_sess, contexts, ids, args, strategy: str):
    feed = v0.make_initial_feed(contexts["decode_input_specs"])
    logical = 0
    last_logits = None
    run_s = []
    records = []
    t_wall = time.perf_counter()
    for index, token_id in enumerate(ids):
        set_hidden(v0, feed, contexts["embedding_state"], int(token_id))
        v0.apply_position_feeds(feed, contexts["decode_input_specs"], logical, args, contexts["decode_rope_context"])
        t0 = time.perf_counter()
        outputs = decode_sess.run(None, feed)
        elapsed = time.perf_counter() - t0
        run_s.append(elapsed)
        logical += 1
        last_logits = np.asarray(outputs[0])
        update_decode_cache_from_outputs(feed, contexts["decode_output_names"], outputs)
        records.append({"index": index, "token_id": int(token_id), "logical_after": logical, "run_s": elapsed})
    if last_logits is None:
        raise RuntimeError("decode prefill received no tokens")
    return {
        "decode_feed_after_prefill": feed,
        "last_logits": last_logits,
        "logical_cache_length": logical,
        "prefill_wall_s": time.perf_counter() - t_wall,
        "prefill_run_s": run_s,
        "prefill_speed": weighted_speed(len(ids), run_s),
        "chunk_records": [],
        "prefill_strategy": strategy,
        "decode_prefill_token_count": int(len(ids)),
        "decode_prefill_records": records,
        "tail_decode_token_count": int(len(ids)) if strategy.endswith("tail") else 0,
        "tail_decode_run_s": run_s if strategy.endswith("tail") else [],
    }


def run_prefill_with_decode_tail(
    v0,
    chunk_sess,
    decode_sess,
    contexts,
    ids,
    args,
):
    """Run full 16-token chunks, then use decode prefill for any non-initial tail.

    Left-padding a final partial chunk is valid for attention masks, but it inserts
    pad rows into the time axis seen by conv layers. For prompts longer than one
    chunk, the final partial tail is therefore prefetched through the one-token
    decode graph to preserve conv/cache continuity.
    """
    chunk = int(args.chunk)
    if len(ids) < chunk:
        return run_decode_prefill(v0, decode_sess, contexts, ids, args, "decode_only_initial_partial")

    if len(ids) % chunk == 0:
        prefill = run_chunk_prefill(
            chunk_sess,
            contexts["chunk_input_specs"],
            contexts["chunk_output_names"],
            ids,
            contexts["embedding_state"],
            args,
        )
        prefill["decode_feed_after_prefill"] = prepare_decode_feed_from_chunk(
            v0,
            contexts["decode_input_specs"],
            contexts["decode_output_names"],
            contexts["chunk_output_names"],
            prefill["last_outputs"],
            prefill.get("decode_cache_overrides"),
        )
        prefill["prefill_strategy"] = "chunk_only"
        prefill["tail_decode_token_count"] = 0
        prefill["tail_decode_run_s"] = []
        return prefill

    full_len = (len(ids) // chunk) * chunk
    full_ids = ids[:full_len]
    tail_ids = ids[full_len:]
    t_wall = time.perf_counter()
    prefill = run_chunk_prefill(
        chunk_sess,
        contexts["chunk_input_specs"],
        contexts["chunk_output_names"],
        full_ids,
        contexts["embedding_state"],
        args,
    )
    feed = prepare_decode_feed_from_chunk(
        v0,
        contexts["decode_input_specs"],
        contexts["decode_output_names"],
        contexts["chunk_output_names"],
        prefill["last_outputs"],
        prefill.get("decode_cache_overrides"),
    )
    logical = int(prefill["logical_cache_length"])
    last_logits = np.asarray(prefill["last_logits"])
    tail_run_s = []
    tail_records = []
    for offset, token_id in enumerate(tail_ids):
        set_hidden(v0, feed, contexts["embedding_state"], int(token_id))
        v0.apply_position_feeds(feed, contexts["decode_input_specs"], logical, args, contexts["decode_rope_context"])
        t0 = time.perf_counter()
        outputs = decode_sess.run(None, feed)
        elapsed = time.perf_counter() - t0
        tail_run_s.append(elapsed)
        logical += 1
        last_logits = np.asarray(outputs[0])
        update_decode_cache_from_outputs(feed, contexts["decode_output_names"], outputs)
        tail_records.append(
            {
                "tail_index": offset,
                "token_id": int(token_id),
                "logical_after": logical,
                "run_s": elapsed,
            }
        )

    combined_run_s = list(prefill.get("prefill_run_s") or []) + tail_run_s
    prefill.update(
        {
            "last_logits": last_logits,
            "logical_cache_length": logical,
            "prefill_wall_s": time.perf_counter() - t_wall,
            "prefill_run_s": combined_run_s,
            "prefill_speed": weighted_speed(len(ids), combined_run_s),
            "decode_feed_after_prefill": feed,
            "prefill_strategy": "full_chunks_plus_decode_tail",
            "chunk_prefill_token_count": int(full_len),
            "tail_decode_token_count": int(len(tail_ids)),
            "tail_decode_run_s": tail_run_s,
            "tail_records": tail_records,
        }
    )
    return prefill


def decode_generate(
    v0,
    decode_sess,
    decode_feed,
    decode_input_specs,
    decode_output_names,
    last_logits,
    logical_cache_length: int,
    ids,
    embedding_state,
    tokenizer,
    args,
    task,
    rope_context,
):
    token_vocab_size = tokenizer.get_vocab_size()
    rng = np.random.default_rng(args.seed)
    stop_ids = set(v0.stop_token_ids(tokenizer, args.stop_token_id))
    generated = []
    decode_run_s = []
    decode_steps = []
    streamed_text = ""
    stream_events = []
    current_logits = np.asarray(last_logits)
    stopped = False
    stream_t0 = time.perf_counter()
    for step in range(int(task["max_new_tokens"])):
        next_id, candidates, policy = choose_next(v0, current_logits, token_vocab_size, args, rng, ids + generated)
        generated.append(int(next_id))
        token_text = tokenizer.decode([int(next_id)]) if int(next_id) < token_vocab_size else ""
        streamed_text += token_text
        now = time.perf_counter()
        stream_events.append(
            {
                "step": step,
                "token_id": int(next_id),
                "text": token_text,
                "elapsed_s": float(now - stream_t0),
            }
        )
        decode_steps.append(
            {
                "step": step,
                "selected_token_id": int(next_id),
                "text": token_text,
                "selection_policy": policy,
                "top_k_before": candidates,
            }
        )
        if getattr(args, "stream", False):
            print(token_text, end="", flush=True)
        if next_id in stop_ids:
            stopped = True
            break
        set_hidden(v0, decode_feed, embedding_state, next_id)
        v0.apply_position_feeds(decode_feed, decode_input_specs, logical_cache_length, args, rope_context)
        t0 = time.perf_counter()
        outputs = decode_sess.run(None, decode_feed)
        elapsed = time.perf_counter() - t0
        decode_run_s.append(elapsed)
        logical_cache_length += 1
        current_logits = np.asarray(outputs[0])
        update_decode_cache_from_outputs(decode_feed, decode_output_names, outputs)
    if getattr(args, "stream", False):
        print("", flush=True)

    generated_text = tokenizer.decode([int(x) for x in generated if int(x) < token_vocab_size])
    return {
        "generated_token_ids": generated,
        "generated_text": generated_text,
        "streamed_text": streamed_text,
        "stream_matches_decode": generated_text == streamed_text,
        "stopped_on_eos_or_stop": stopped,
        "decode_run_s": decode_run_s,
        "decode_steps": decode_steps,
        "stream_events": stream_events,
        "decode_speed": weighted_speed(len(decode_run_s), decode_run_s),
        "logical_cache_length_after_decode": logical_cache_length,
        "cache_stats_after_decode": v0.cache_stats(decode_feed, logical_cache_length),
        "final_top_k": v0.top_k(current_logits, args.top_k, token_vocab_size),
    }


def run_task(v0, chunk_sess, decode_sess, contexts, task, args):
    tokenizer = contexts["tokenizer"]
    embedding_state = contexts["embedding_state"]
    token_vocab_size = tokenizer.get_vocab_size()
    model_prompt, template_info = apply_chat_template(task["prompt"], args.model_root, args.system_prompt, not args.no_template)
    ids = tokenizer.encode(model_prompt).ids
    if not ids:
        raise RuntimeError(f"tokenizer returned no ids for {task['name']}")
    if len(ids) > args.max_prompt_tokens:
        ids = ids[-args.max_prompt_tokens :]
    task_args = make_internal_args(args, int(task["max_new_tokens"]))
    t0 = time.perf_counter()
    prefill = run_prefill_with_decode_tail(v0, chunk_sess, decode_sess, contexts, ids, task_args)
    ttft_s = time.perf_counter() - t0
    decode_feed = prefill["decode_feed_after_prefill"]
    cache_after_prefill = v0.cache_stats(decode_feed, prefill["logical_cache_length"])
    generation = decode_generate(
        v0,
        decode_sess,
        decode_feed,
        contexts["decode_input_specs"],
        contexts["decode_output_names"],
        prefill["last_logits"],
        prefill["logical_cache_length"],
        ids,
        embedding_state,
        tokenizer,
        task_args,
        task,
        contexts["decode_rope_context"],
    )
    out = {
        "name": task["name"],
        "prompt": task["prompt"],
        "model_prompt": model_prompt,
        "template": template_info,
        "prompt_token_ids": [int(x) for x in ids],
        "prompt_token_count": len(ids),
        "prompt_detokenized": tokenizer.decode([int(x) for x in ids if int(x) < token_vocab_size]),
        "max_new_tokens": int(task["max_new_tokens"]),
        "expect": task.get("expect"),
        "embedding_backend": contexts.get("embedding_info", {}).get("backend"),
        "chunk_records": prefill["chunk_records"],
        "prefill_strategy": prefill.get("prefill_strategy"),
        "tail_decode_token_count": prefill.get("tail_decode_token_count", 0),
        "tail_decode_run_s": prefill.get("tail_decode_run_s", []),
        "tail_records": prefill.get("tail_records", []),
        "prefill_run_s": prefill["prefill_run_s"],
        "prefill_speed": prefill["prefill_speed"],
        "prefill_wall_s": prefill["prefill_wall_s"],
        "ttft_s_excluding_session_create": ttft_s,
        "cache_stats_after_prefill": cache_after_prefill,
        "logical_cache_length_after_prefill": prefill["logical_cache_length"],
        "prefill_top_k": v0.top_k(prefill["last_logits"], args.top_k, token_vocab_size),
    }
    out.update(generation)
    if task.get("json"):
        out["json_check"] = json_check(out["generated_text"])
    return out


def run_sequential_reference(v0, decode_sess, contexts, prompt: str, args):
    task_args = make_internal_args(args, args.reference_max_new_tokens)
    rng = np.random.default_rng(args.seed)
    model_prompt, template_info = apply_chat_template(prompt, args.model_root, args.system_prompt, not args.no_template)
    ref = v0.run_one_prompt(
        "sequential_decode_reference",
        model_prompt,
        decode_sess,
        contexts["decode_output_names"],
        contexts["decode_input_specs"],
        contexts["embedding_state"],
        contexts["tokenizer"],
        task_args,
        rng,
        contexts["decode_rope_context"],
    )
    ref["user_prompt"] = prompt
    ref["model_prompt"] = model_prompt
    ref["template"] = template_info
    return ref


def profile_qnn_only(profile_items):
    out = {}
    for item in profile_items:
        counts = item.get("provider_counts") or {}
        out[item.get("label", "unknown")] = bool(counts.get("QNNExecutionProvider", 0) > 0 and counts.get("CPUExecutionProvider", 0) == 0)
    return out


def write_summary(log_dir: Path, result: dict):
    lines = [
        "# P4 Chunk Prefill Handoff Probe",
        "",
        f"- ok: `{result.get('ok')}`",
        f"- v1_complete: `{result.get('v1_complete')}`",
        f"- share_ep_contexts: `{result.get('share_ep_contexts_requested')}`",
        f"- embedding_backend: `{(result.get('embedding_info') or {}).get('backend')}`",
        f"- chunk_session_create_s: `{(result.get('chunk_session') or {}).get('session_create_s')}`",
        f"- decode_session_create_s: `{(result.get('decode_session') or {}).get('session_create_s')}`",
        f"- qnn_only_by_profile: `{result.get('qnn_only_by_profile')}`",
        f"- peak_hwm_mb: `{((result.get('proc_final') or {}).get('VmHWM_mb'))}`",
        "",
        "## Tasks",
        "",
    ]
    for item in result.get("task_results", []):
        lines.extend(
            [
                f"### {item.get('name')}",
                "",
                f"- prompt_tokens: `{item.get('prompt_token_count')}`",
                f"- template: `{item.get('template')}`",
                f"- generated_tokens: `{len(item.get('generated_token_ids') or [])}`",
                f"- prefill_tok_s: `{(item.get('prefill_speed') or {}).get('tok_per_s')}`",
                f"- decode_tok_s: `{(item.get('decode_speed') or {}).get('tok_per_s')}`",
                f"- ttft_s_excluding_session_create: `{item.get('ttft_s_excluding_session_create')}`",
                f"- generated_text: `{item.get('generated_text')}`",
                f"- json_check: `{item.get('json_check')}`",
                "",
            ]
        )
    if result.get("sequential_reference"):
        ref = result["sequential_reference"]
        lines.extend(
            [
                "## Sequential Decode Reference",
                "",
                f"- generated_text: `{ref.get('generated_text')}`",
                f"- prefill_speed: `{ref.get('prefill_speed')}`",
                f"- decode_speed: `{ref.get('decode_speed')}`",
                "",
            ]
        )
    if result.get("error"):
        lines.extend(["## Error", "", "```", result.get("error", "")[:4000], "```", ""])
    lines.extend(
        [
            "## Files",
            "",
            f"- result_json: `{log_dir / 'result.json'}`",
            f"- summary_md: `{log_dir / 'summary.md'}`",
        ]
    )
    (log_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="P4 chunk prefill -> decode EPContext handoff probe.")
    parser.add_argument("--timestamp", default=time.strftime("%Y%m%d_%H%M%S_chunk_handoff"))
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--chunk-context", type=Path, default=DEFAULT_CHUNK_CONTEXT)
    parser.add_argument("--decode-context", type=Path, default=DEFAULT_DECODE_CONTEXT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--official-model", type=Path, default=DEFAULT_OFFICIAL_MODEL)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--v0-runner-dir", type=Path, default=DEFAULT_V0_RUNNER_DIR)
    parser.add_argument("--embedding-backend", choices=["official_raw", "official_int8_rowwise"], default="official_raw")
    parser.add_argument("--embedding-int8-dir", type=Path, default=DEFAULT_EMBEDDING_INT8_DIR)
    parser.add_argument("--build-embedding-int8-if-missing", action="store_true")
    parser.add_argument("--prompt", default="The capital of Japan is")
    parser.add_argument("--system-prompt")
    parser.add_argument("--no-template", action="store_true")
    parser.add_argument("--task-set", choices=["single", "benchmark"], default="single")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--reference-max-new-tokens", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--stop-token-id", action="append", type=int, default=[])
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--total-len", type=int, default=DEFAULT_TOTAL_LEN)
    parser.add_argument("--mask-value", type=float, default=DEFAULT_MASK_VALUE)
    parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
    parser.add_argument("--share-ep-contexts", action="store_true")
    parser.add_argument("--run-sequential-reference", action="store_true")
    args = parser.parse_args()

    log_dir = args.log_dir or (args.log_root / f"p4_chunk_prefill_handoff_{args.timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": args.timestamp,
        "mode": "p4_chunk_prefill_handoff",
        "v1_complete": False,
        "chunk_context": str(args.chunk_context),
        "decode_context": str(args.decode_context),
        "share_ep_contexts_requested": bool(args.share_ep_contexts),
        "no_sudo_or_root": True,
        "power_before": power_snapshot(),
        "thermal_before": thermal_snapshot(),
        "rss_start_mb": rss_mb(),
        "proc_start": proc_status(),
    }
    chunk_sess = None
    decode_sess = None
    try:
        v0 = import_v0(args.v0_runner_dir)
        chunk_sess, chunk_info = make_session(args.chunk_context, log_dir, "chunk", args.share_ep_contexts)
        result["chunk_session"] = chunk_info
        result["after_chunk_load"] = {"rss_mb": rss_mb(), "proc": proc_status()}
        decode_sess, decode_info = make_session(args.decode_context, log_dir, "decode", args.share_ep_contexts)
        result["decode_session"] = decode_info
        result["after_decode_load_both_alive"] = {"rss_mb": rss_mb(), "proc": proc_status()}

        chunk_input_specs, _ = v0.load_io_shapes(args.chunk_context)
        decode_input_specs, _ = v0.load_io_shapes(args.decode_context)
        decode_rope_inputs = v0.discover_rope_inputs(args.decode_context)
        cos_cache, sin_cache = v0.load_rope_cache(args.official_model)
        embedding_state, embedding_info = load_embedding_state_for_runner(v0, args)
        result["embedding_info"] = embedding_info
        result["after_embedding_load"] = {"rss_mb": rss_mb(), "proc": proc_status()}
        contexts = {
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
        tasks = (
            BENCHMARK_TASKS
            if args.task_set == "benchmark"
            else [{"name": "single", "prompt": args.prompt, "max_new_tokens": args.max_new_tokens}]
        )
        result["task_results"] = [run_task(v0, chunk_sess, decode_sess, contexts, task, args) for task in tasks]
        result["after_tasks_both_alive"] = {"rss_mb": rss_mb(), "proc": proc_status()}
        if args.run_sequential_reference:
            result["sequential_reference"] = run_sequential_reference(v0, decode_sess, contexts, args.prompt, args)
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
        result["qnn_only_by_profile"] = profile_qnn_only(profiles)
        result["power_after"] = power_snapshot()
        result["thermal_after"] = thermal_snapshot()
        result["rss_final_mb"] = rss_mb()
        result["proc_final"] = proc_status()
        del decode_sess
        del chunk_sess
        gc.collect()
        (log_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        write_summary(log_dir, result)
    print(json.dumps({"ok": result.get("ok"), "log_dir": str(log_dir), "result_json": str(log_dir / "result.json")}, ensure_ascii=False), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
