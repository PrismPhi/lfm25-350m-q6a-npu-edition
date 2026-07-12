#!/usr/bin/env python3
"""Localhost-only OpenAI-style API backed by the ctx2048 QNN runner.

OpenWebUI compatibility: passive tool metadata is accepted for normal chat,
but this runner never executes tools or emits tool calls.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ROOT = Path(
    os.environ.get("LFM2_5_STATE_DIR")
    or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "lfm2.5-350m-q6a-qcs6490-qnn-npu"
)
MODEL_ID = "lfm2.5-350m-qnn-ctx2048"
DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS = 1024
DEFAULT_PROFILE = "chat"
SAMPLING_PROFILES = {
    "chat": {
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "repetition_last_n": 64,
    },
    "extraction": {
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "repetition_last_n": -1,
    },
}
DEFAULT_CHUNK_CONTEXT = STATE_ROOT / "contexts" / "chunk" / "chunk_epcontext.onnx"
DEFAULT_DECODE_CONTEXT = STATE_ROOT / "contexts" / "decode" / "decode_epcontext.onnx"


def import_v1_helpers():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import probe_p4_chunk_prefill_handoff as handoff
    import run_practical_runner_v1_chunk as runner

    return handoff, runner


def api_error(message: str, code: str, status: int = HTTPStatus.BAD_REQUEST, param: str | None = None) -> tuple[int, dict]:
    return int(status), {"error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}}


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("message content must be a string or a list of text parts")
    parts = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text" or not isinstance(item.get("text"), str):
            raise ValueError("only text content parts are supported")
        parts.append(item["text"])
    return "".join(parts)


def decode_visible_text(tokenizer, token_ids: list[int]) -> str:
    """Decode the complete token sequence before producing a stream delta.

    Byte-level BPE tokens can represent an incomplete UTF-8 sequence alone.
    Decoding each token separately turns those fragments into U+FFFD, even
    though the combined sequence decodes into a valid character.
    """
    return tokenizer.decode([int(token_id) for token_id in token_ids]).replace("\ufffd", "")


def render_chatml(messages: list[dict]) -> str:
    if not messages:
        raise ValueError("messages must not be empty")
    normalized = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = item.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("roles must be system, user, or assistant")
        normalized.append({"role": role, "content": content_to_text(item.get("content", ""))})
    if normalized[-1]["role"] != "user":
        raise ValueError("the final message must have role user")
    output = ["<|startoftext|>"]
    if normalized[0]["role"] == "system":
        system = normalized.pop(0)["content"]
        if system:
            output.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    last_assistant = max((index for index, item in enumerate(normalized) if item["role"] == "assistant"), default=-1)
    for index, item in enumerate(normalized):
        content = item["content"]
        if item["role"] == "assistant" and index != last_assistant and "</think>" in content:
            content = content.split("</think>")[-1].strip()
        output.append(f"<|im_start|>{item['role']}\n{content}<|im_end|>\n")
    output.append("<|im_start|>assistant\n")
    return "".join(output)


class GenerationArgs:
    pass


def make_generation_args(request: dict, total_len: int, chunk: int, default_profile: str) -> GenerationArgs:
    args = GenerationArgs()
    profile_name = request.get("profile", request.get("qnn_profile", default_profile))
    if not isinstance(profile_name, str) or profile_name not in SAMPLING_PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(sorted(SAMPLING_PROFILES))}")
    profile = SAMPLING_PROFILES[profile_name]
    temperature = request.get("temperature", profile["temperature"])
    top_p = request.get("top_p", profile["top_p"])
    top_k = request.get("top_k", profile["top_k"])
    penalty = request.get("repetition_penalty", profile["repetition_penalty"])
    repetition_last_n = request.get("repetition_last_n", profile["repetition_last_n"])
    min_new_tokens = request.get("min_new_tokens", 0)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        raise ValueError("temperature must be a number in [0, 2]")
    if not isinstance(top_p, (int, float)) or top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be a number in (0, 1]")
    if not isinstance(top_k, int) or top_k < 1 or top_k > 4096:
        raise ValueError("top_k must be an integer in [1, 4096]")
    if not isinstance(penalty, (int, float)) or penalty < 0.1 or penalty > 2:
        raise ValueError("repetition_penalty must be a number in [0.1, 2]")
    if not isinstance(repetition_last_n, int) or repetition_last_n < -1 or repetition_last_n > total_len:
        raise ValueError(f"repetition_last_n must be an integer in [-1, {total_len}]")
    if not isinstance(min_new_tokens, int) or min_new_tokens < 0 or min_new_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"min_new_tokens must be an integer in [0, {MAX_NEW_TOKENS}]")
    max_tokens = request.get("max_tokens", DEFAULT_MAX_NEW_TOKENS)
    if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"max_tokens must be an integer in [1, {MAX_NEW_TOKENS}]")
    args.max_new_tokens = int(max_tokens)
    args.max_prompt_tokens = int(total_len)
    args.temperature = float(temperature)
    args.top_k = int(top_k)
    args.top_p = float(top_p)
    args.repetition_penalty = float(penalty)
    args.repetition_last_n = int(repetition_last_n)
    args.min_new_tokens = min(int(min_new_tokens), int(max_tokens))
    args.profile = profile_name
    args.greedy = float(temperature) == 0.0
    args.seed = int(request.get("seed", 2468))
    args.stop_token_id = []
    args.tail_mask_value = -64.0
    args.disable_rope_feed = False
    args.disable_tail_mask_feed = False
    args.chunk = int(chunk)
    args.total_len = int(total_len)
    args.mask_value = -64.0
    args.rope_theta = 1_000_000.0
    args.stream = False
    return args


def normalize_stop(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value[:4]
    raise ValueError("stop must be a string or an array of strings")


def normalize_logit_bias(value, tokenizer_vocab_size: int) -> dict[int, float]:
    """Validate the standard OpenAI logit_bias object without any default bias."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("logit_bias must be an object mapping token ids to bias values")
    normalized = {}
    for raw_token_id, raw_bias in value.items():
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("logit_bias keys must be integer token ids") from exc
        if isinstance(raw_bias, bool) or not isinstance(raw_bias, (int, float)) or not np.isfinite(raw_bias):
            raise ValueError("logit_bias values must be finite numbers")
        bias = float(raw_bias)
        if not -100.0 <= bias <= 100.0:
            raise ValueError("logit_bias values must be in [-100, 100]")
        if token_id < 0 or token_id >= tokenizer_vocab_size:
            raise ValueError(f"logit_bias token id {token_id} is outside tokenizer vocabulary")
        normalized[token_id] = bias
    return normalized


def apply_logit_bias(logits, logit_bias: dict[int, float]):
    if not logit_bias:
        return logits
    adjusted = np.asarray(logits, dtype=np.float32).copy()
    for token_id, bias in logit_bias.items():
        adjusted[token_id] += np.float32(bias)
    return adjusted


def logit_bias_metadata(logit_bias: dict[int, float], stop_ids: set[int], json_mode: bool) -> dict:
    eos_entries = {str(token_id): float(logit_bias[token_id]) for token_id in sorted(stop_ids) if token_id in logit_bias}
    return {
        "applied": bool(logit_bias),
        "entries": {str(token_id): float(logit_bias[token_id]) for token_id in sorted(logit_bias)},
        "entry_count": len(logit_bias),
        "eos_bias_explicit": bool(eos_entries),
        "eos_entries": eos_entries,
        "default_unbiased": not bool(logit_bias),
        "json_forced_structure_unbiased": bool(json_mode),
    }


def choose_profiled_next(handoff, v0, logits, tokenizer_vocab_size: int, args: GenerationArgs, rng, history_ids: list[int], stop_ids: set[int], step: int):
    """Apply profile sampling, explicit logit_bias, then optional min-new EOS suppression."""
    if args.repetition_last_n > 0:
        history_ids = history_ids[-args.repetition_last_n :]
    # EOS may occur in the rendered ChatML prompt. Penalizing it changes the
    # EOS/non-EOS balance for reasons unrelated to repetition in the answer.
    penalty_history = [token_id for token_id in history_ids if token_id not in stop_ids]
    adjusted = handoff.apply_repetition_penalty(logits, penalty_history, args.repetition_penalty)
    adjusted = apply_logit_bias(adjusted, args.logit_bias)
    eos_suppressed = step < args.min_new_tokens
    if eos_suppressed:
        adjusted = np.asarray(adjusted, dtype=np.float32).copy()
        for token_id in stop_ids:
            if 0 <= token_id < adjusted.size:
                adjusted[token_id] = -np.inf
    next_id, candidates, policy = v0.choose_next(adjusted, tokenizer_vocab_size, args, rng)
    return int(next_id), candidates, policy, eos_suppressed


def choose_json_value_token(v0, logits, tokenizer, has_value: bool, value_length: int, candidate_pool: int, logit_bias: dict[int, float]):
    candidates = v0.top_k(apply_logit_bias(logits, logit_bias), candidate_pool, tokenizer.get_vocab_size())
    for rank, candidate in enumerate(candidates):
        token_id = int(candidate["token_id"])
        text = tokenizer.decode([token_id]) if token_id < tokenizer.get_vocab_size() else ""
        if not text:
            continue
        if text == '"' and has_value:
            return token_id, text, True, value_length, rank
        if all(ord(char) >= 32 and char not in {'\\', '"'} for char in text) and value_length + len(text) <= 96:
            return token_id, text, False, value_length + len(text), rank
    raise RuntimeError("no JSON-value-compatible token in top {} logits".format(candidate_pool))


class QNNEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.log_dir = args.log_root / f"openai_server_{args.timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.request_records: list[dict] = []
        self.ready = False
        self.handoff, self.runner = import_v1_helpers()
        self.v0 = self.handoff.import_v0(args.v0_runner_dir)
        self.chunk_sess, self.chunk_session = self.handoff.make_session(args.chunk_context, self.log_dir, "chunk", False)
        self.after_chunk_load = {"proc": self.handoff.proc_status(), "rss_mb": self.handoff.rss_mb()}
        self.decode_sess, self.decode_session = self.handoff.make_session(args.decode_context, self.log_dir, "decode", False)
        self.after_decode_load = {"proc": self.handoff.proc_status(), "rss_mb": self.handoff.rss_mb()}
        runtime_args = argparse.Namespace(
            chunk_context=args.chunk_context,
            decode_context=args.decode_context,
            official_model=args.rope_cache,
            tokenizer=args.tokenizer,
            embedding_backend="official_int8_rowwise",
            embedding_int8_dir=args.embedding_int8_dir,
            build_embedding_int8_if_missing=False,
        )
        self.contexts = self.runner.build_contexts(self.v0, runtime_args, self.chunk_sess, self.decode_sess)
        self.tokenizer = self.contexts["tokenizer"]
        self.ready = True
        self.started_at = time.time()
        self.write_ready()

    def write_ready(self) -> None:
        data = {
            "ready": self.ready,
            "model": MODEL_ID,
            "host": self.args.host,
            "port": self.args.port,
            "default_profile": self.args.default_profile,
            "fallback_disabled": True,
            "embedding_backend": self.contexts["embedding_info"].get("backend"),
            "chunk_session": self.chunk_session,
            "decode_session": self.decode_session,
            "after_decode_load": self.after_decode_load,
        }
        (self.log_dir / "ready.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def health(self) -> dict:
        return {
            "status": "ok" if self.ready else "not_ready",
            "ready": self.ready,
            "model": MODEL_ID,
            "backend": "QNNExecutionProvider / HTP",
            "fallback_disabled": True,
            "queue_serialized": True,
            "default_profile": self.args.default_profile,
            "requests_completed": len(self.request_records),
            "uptime_s": round(time.time() - self.started_at, 3),
        }

    def _decode(self, prefill: dict, ids: list[int], gen_args: GenerationArgs, stop_strings: list[str], json_mode: bool, on_token):
        feed = prefill["decode_feed_after_prefill"]
        current_logits = np.asarray(prefill["last_logits"])
        logical = int(prefill["logical_cache_length"])
        generated_ids, generated_text, emitted_text, events, decode_run_s = [], "", "", [], []
        stream_started = False
        token_vocab_size = self.tokenizer.get_vocab_size()
        rng = np.random.default_rng(gen_args.seed)
        stop_ids = set(self.v0.stop_token_ids(self.tokenizer, gen_args.stop_token_id))
        json_prefix_ids = [int(item) for item in self.tokenizer.encode('{"answer":"').ids] if json_mode else []
        json_suffix_ids = [int(item) for item in self.tokenizer.encode("}").ids] if json_mode else []
        json_phase = "prefix" if json_mode else "text"
        json_prefix_index = json_suffix_index = 0
        json_has_value = False
        json_value_length = 0
        stopped = False
        finish_reason = "length"
        eos_suppressed_steps = 0
        started = time.perf_counter()
        for step in range(gen_args.max_new_tokens):
            if json_mode:
                if json_phase == "prefix":
                    next_id = json_prefix_ids[json_prefix_index]
                    json_prefix_index += 1
                    if json_prefix_index == len(json_prefix_ids):
                        json_phase = "value"
                    selected_rank = None
                    policy = "json_forced_structure"
                    eos_suppressed = False
                elif json_phase == "suffix":
                    next_id = json_suffix_ids[json_suffix_index]
                    json_suffix_index += 1
                    if json_suffix_index == len(json_suffix_ids):
                        json_phase = "done"
                    selected_rank = None
                    policy = "json_forced_structure"
                    eos_suppressed = False
                else:
                    next_id, _selected_token_text, closes_value, json_value_length, selected_rank = choose_json_value_token(
                        self.v0, current_logits, self.tokenizer, json_has_value, json_value_length, 256, gen_args.logit_bias
                    )
                    if closes_value:
                        json_phase = "suffix"
                    else:
                        json_has_value = True
                    policy = "json_qnn_value_constraint"
                    eos_suppressed = False
            else:
                next_id, candidates, policy, eos_suppressed = choose_profiled_next(
                    self.handoff, self.v0, current_logits, token_vocab_size, gen_args, rng, ids + generated_ids, stop_ids, step
                )
                selected_rank = None
            if eos_suppressed:
                eos_suppressed_steps += 1
            generated_ids.append(int(next_id))
            decoded_text = decode_visible_text(self.tokenizer, generated_ids) if 0 <= int(next_id) < token_vocab_size else generated_text
            generated_text = decoded_text
            token_text = ""
            if decoded_text.startswith(emitted_text):
                token_text = decoded_text[len(emitted_text):]
                emitted_text = decoded_text
            event = {"step": step, "token_id": int(next_id), "text": token_text, "elapsed_s": time.perf_counter() - started, "selection_policy": policy}
            event["eos_suppressed_by_min_new_tokens"] = bool(eos_suppressed)
            event["logit_bias_applied"] = bool(gen_args.logit_bias)
            if selected_rank is not None:
                event["selected_rank_in_candidate_pool"] = selected_rank
            events.append(event)
            if token_text and on_token is not None:
                on_token(token_text, not stream_started)
                stream_started = True
            stop_match = next((item for item in stop_strings if item and item in generated_text), None)
            if stop_match is not None:
                generated_text = generated_text.split(stop_match, 1)[0]
                stopped = True
                finish_reason = "stop"
                break
            if json_mode and json_phase == "done":
                stopped = True
                finish_reason = "stop"
                break
            if next_id in stop_ids:
                stopped = True
                finish_reason = "stop"
                break
            self.handoff.set_hidden(self.v0, feed, self.contexts["embedding_state"], next_id)
            self.v0.apply_position_feeds(feed, self.contexts["decode_input_specs"], logical, gen_args, self.contexts["decode_rope_context"])
            run_started = time.perf_counter()
            outputs = self.decode_sess.run(None, feed)
            decode_run_s.append(time.perf_counter() - run_started)
            logical += 1
            current_logits = np.asarray(outputs[0])
            self.handoff.update_decode_cache_from_outputs(feed, self.contexts["decode_output_names"], outputs)
        return {
            "generated_token_ids": generated_ids,
            "generated_text": generated_text,
            "stream_events": events,
            "decode_run_s": decode_run_s,
            "decode_speed": self.handoff.weighted_speed(len(decode_run_s), decode_run_s),
            "stopped_on_eos_or_stop": stopped,
            "finish_reason": finish_reason,
            "eos_suppressed_steps": eos_suppressed_steps,
            "logical_cache_length_after_decode": logical,
            "cache_stats_after_decode": self.v0.cache_stats(feed, logical),
            "json_state": {"phase": json_phase, "value_length": json_value_length} if json_mode else None,
            "json_check": self.handoff.json_check(generated_text) if json_mode else None,
        }

    def generate(self, request: dict, on_token=None) -> dict:
        messages = request.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be an array")
        tools = request.get("tools")
        tool_choice = request.get("tool_choice")
        if tools is not None and not isinstance(tools, list):
            raise ValueError("tools must be an array when supplied")
        if tool_choice is not None and tool_choice not in ("auto", "none"):
            raise ValueError("forced tool execution is not supported by this local runner")
        # OpenWebUI includes tools: [] / tool_choice: auto for ordinary chat.
        # Ignore passive metadata so the request remains a plain text completion.
        tools_ignored = bool(tools)
        model = request.get("model", MODEL_ID)
        if model not in {MODEL_ID, "lfm2.5-350m"}:
            raise ValueError(f"unknown model: {model}")
        response_format = request.get("response_format") or {"type": "text"}
        if not isinstance(response_format, dict) or response_format.get("type", "text") not in {"text", "json_object"}:
            raise ValueError("response_format.type must be text or json_object")
        json_mode = response_format.get("type") == "json_object"
        model_prompt = render_chatml(messages)
        ids = [int(item) for item in self.tokenizer.encode(model_prompt).ids]
        gen_args = make_generation_args(request, self.args.total_len, self.args.chunk, self.args.default_profile)
        gen_args.logit_bias = normalize_logit_bias(request.get("logit_bias"), self.tokenizer.get_vocab_size())
        if len(ids) + gen_args.max_new_tokens > self.args.total_len:
            raise OverflowError(f"ctx2048 limit exceeded: prompt={len(ids)}, max_tokens={gen_args.max_new_tokens}, limit={self.args.total_len}")
        stop_strings = normalize_stop(request.get("stop"))
        if json_mode and stop_strings:
            raise ValueError("stop is not supported with response_format json_object")
        stop_ids = set(self.v0.stop_token_ids(self.tokenizer, gen_args.stop_token_id))
        with self.lock:
            started = time.perf_counter()
            prefill = self.handoff.run_prefill_with_decode_tail(self.v0, self.chunk_sess, self.decode_sess, self.contexts, ids, gen_args)
            ttft = time.perf_counter() - started
            cache_prefill = self.v0.cache_stats(prefill["decode_feed_after_prefill"], prefill["logical_cache_length"])
            decoded = self._decode(prefill, ids, gen_args, stop_strings, json_mode, on_token)
            result = {
                "model_prompt": model_prompt,
                "prompt_token_ids": ids,
                "prompt_token_count": len(ids),
                "generated_token_ids": decoded["generated_token_ids"],
                "generated_text": decoded["generated_text"],
                "completion_tokens": len(decoded["generated_token_ids"]),
                "prefill_speed": prefill["prefill_speed"],
                "decode_speed": decoded["decode_speed"],
                "ttft_s_excluding_session_create": ttft,
                "logical_cache_length_after_prefill": prefill["logical_cache_length"],
                "logical_cache_length_after_decode": decoded["logical_cache_length_after_decode"],
                "cache_stats_after_prefill": cache_prefill,
                "cache_stats_after_decode": decoded["cache_stats_after_decode"],
                "prefill_strategy": prefill.get("prefill_strategy"),
                "stream_events": decoded["stream_events"],
                "stopped_on_eos_or_stop": decoded["stopped_on_eos_or_stop"],
                "finish_reason": decoded["finish_reason"],
                "requested_max_tokens": gen_args.max_new_tokens,
                "sampling_profile": gen_args.profile,
                "sampling_parameters": {
                    "temperature": gen_args.temperature,
                    "top_k": gen_args.top_k,
                    "top_p": gen_args.top_p,
                    "repetition_penalty": gen_args.repetition_penalty,
                    "repetition_last_n": gen_args.repetition_last_n,
                    "min_new_tokens": gen_args.min_new_tokens,
                    "eos_excluded_from_repetition": True,
                },
                "eos_suppressed_steps": decoded["eos_suppressed_steps"],
                "logit_bias": logit_bias_metadata(gen_args.logit_bias, stop_ids, json_mode),
                "json_mode": json_mode,
                "json_check": decoded["json_check"],
                "json_object_valid": bool(json_mode and (decoded["json_check"] or {}).get("syntax_valid") and isinstance((decoded["json_check"] or {}).get("parsed"), dict)),
                "json_constraint": "fixed JSON structure tokens plus QNN-logit-constrained answer value" if json_mode else None,
                "tools_ignored": tools_ignored,
                "embedding_backend": self.contexts["embedding_info"].get("backend"),
                "model_compute_backend": "QNNExecutionProvider / HTP",
                "fallback_disabled": True,
            }
            self.request_records.append({
                "request_id": request.get("_request_id"),
                "created": int(time.time()),
                "prompt_tokens": result["prompt_token_count"],
                "completion_tokens": result["completion_tokens"],
                "generated_text": result["generated_text"],
                "requested_max_tokens": result["requested_max_tokens"],
                "finish_reason": result["finish_reason"],
                "sampling_profile": result["sampling_profile"],
                "sampling_parameters": result["sampling_parameters"],
                "eos_suppressed_steps": result["eos_suppressed_steps"],
                "logit_bias": result["logit_bias"],
                "json_mode": json_mode,
                "tools_ignored": tools_ignored,
                "prefill_tok_s": (result["prefill_speed"] or {}).get("tok_per_s"),
                "decode_tok_s": (result["decode_speed"] or {}).get("tok_per_s"),
            })
            return result

    def close(self) -> dict:
        profiles = []
        if self.chunk_sess is not None:
            profiles.append(self.handoff.finish_profile(self.chunk_sess, "chunk"))
        if self.decode_sess is not None:
            profiles.append(self.handoff.finish_profile(self.decode_sess, "decode"))
        result = {
            "mode": "lfm2_5_q6a_openai_server",
            "fallback_disabled": True,
            "model_compute_backend": "QNNExecutionProvider / HTP",
            "chunk_session": self.chunk_session,
            "decode_session": self.decode_session,
            "after_chunk_load": self.after_chunk_load,
            "after_decode_load": self.after_decode_load,
            "embedding_info": self.contexts.get("embedding_info"),
            "profiles": profiles,
            "qnn_only_by_profile": self.handoff.profile_qnn_only(profiles),
            "requests": self.request_records,
            "proc_final": self.handoff.proc_status(),
            "thermal_final": self.handoff.thermal_snapshot(),
            "power_final": self.handoff.power_snapshot(),
        }
        (self.log_dir / "server_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.ready = False
        return result


class APIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, engine: QNNEngine):
        super().__init__(address, handler)
        self.engine = engine


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self) -> dict:
        length = self.headers.get("Content-Length")
        if length is None:
            raise ValueError("Content-Length is required")
        try:
            size = int(length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if size < 2 or size > 1_000_000:
            raise ValueError("request body must be between 2 and 1000000 bytes")
        body = json.loads(self.rfile.read(size).decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        return body

    def _chat_response(self, request: dict, stream: bool) -> None:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        request["_request_id"] = request_id
        created = int(time.time())
        print(json.dumps({
            "event": "chat_request",
            "request_id": request_id,
            "stream": stream,
            "profile_supplied": request.get("profile", request.get("qnn_profile")),
            "max_tokens_supplied": request.get("max_tokens"),
            "logit_bias_entry_count": len(request.get("logit_bias", {})) if isinstance(request.get("logit_bias"), dict) else None,
            "message_count": len(request.get("messages", [])) if isinstance(request.get("messages"), list) else None,
            "tools_count": len(request.get("tools", [])) if isinstance(request.get("tools"), list) else None,
        }, ensure_ascii=False), flush=True)

        def log_result(result: dict) -> None:
            print(json.dumps({
                "event": "chat_complete",
                "request_id": request_id,
                "prompt_tokens": result["prompt_token_count"],
                "completion_tokens": result["completion_tokens"],
                "requested_max_tokens": result["requested_max_tokens"],
                "finish_reason": result["finish_reason"],
                "sampling_profile": result["sampling_profile"],
                "eos_suppressed_steps": result["eos_suppressed_steps"],
                "logit_bias": result["logit_bias"],
            }, ensure_ascii=False), flush=True)

        if not stream:
            result = self.server.engine.generate(request)
            log_result(result)
            body = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["generated_text"]}, "finish_reason": result["finish_reason"]}],
                "usage": {"prompt_tokens": result["prompt_token_count"], "completion_tokens": result["completion_tokens"], "total_tokens": result["prompt_token_count"] + result["completion_tokens"]},
                "qnn_metrics": {key: result[key] for key in ("prefill_speed", "decode_speed", "ttft_s_excluding_session_create", "logical_cache_length_after_prefill", "logical_cache_length_after_decode", "requested_max_tokens", "finish_reason", "sampling_profile", "sampling_parameters", "eos_suppressed_steps", "logit_bias", "json_mode", "json_check", "json_object_valid", "json_constraint", "fallback_disabled")},
            }
            self._send_json(HTTPStatus.OK, body)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        sent_role = False

        def send_event(body: dict):
            self.wfile.write(f"data: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()

        def on_token(text: str, first: bool):
            nonlocal sent_role
            delta = {"content": text}
            if not sent_role:
                delta["role"] = "assistant"
                sent_role = True
            send_event({"id": request_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})

        result = self.server.engine.generate(request, on_token=on_token)
        log_result(result)
        if not sent_role:
            send_event({"id": request_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        send_event({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}],
            "usage": {"prompt_tokens": result["prompt_token_count"], "completion_tokens": result["completion_tokens"], "total_tokens": result["prompt_token_count"] + result["completion_tokens"]},
            "qnn_metrics": {key: result[key] for key in ("prefill_speed", "decode_speed", "ttft_s_excluding_session_create", "logical_cache_length_after_prefill", "logical_cache_length_after_decode", "requested_max_tokens", "finish_reason", "sampling_profile", "sampling_parameters", "eos_suppressed_steps", "logit_bias", "json_mode", "json_check", "json_object_valid", "json_constraint", "fallback_disabled")},
        })
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def do_GET(self):
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, self.server.engine.health())
        elif self.path == "/v1/models":
            self._send_json(HTTPStatus.OK, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local-qnn"}]})
        else:
            status, body = api_error("route not found", "not_found", HTTPStatus.NOT_FOUND)
            self._send_json(status, body)

    def do_POST(self):
        try:
            if self.path == "/v1/chat/completions":
                request = self._read_json()
                self._chat_response(request, bool(request.get("stream", False)))
                return
            if self.path == "/v1/admin/shutdown":
                status, body = api_error("admin shutdown is disabled for the persistent runner", "forbidden", HTTPStatus.FORBIDDEN)
                self._send_json(status, body)
                return
            status, body = api_error("route not found", "not_found", HTTPStatus.NOT_FOUND)
            self._send_json(status, body)
        except OverflowError as exc:
            status, body = api_error(str(exc), "context_length_exceeded", HTTPStatus.BAD_REQUEST, "messages")
            self._send_json(status, body)
        except (ValueError, json.JSONDecodeError) as exc:
            status, body = api_error(str(exc), "invalid_request", HTTPStatus.BAD_REQUEST)
            self._send_json(status, body)
        except BrokenPipeError:
            print(json.dumps({"event": "client_disconnect", "path": self.path}, ensure_ascii=False), flush=True)
            return
        except Exception as exc:
            print(json.dumps({"event": "request_error", "path": self.path, "error_type": type(exc).__name__}, ensure_ascii=False), flush=True)
            status, body = api_error(f"generation failed: {exc}", "generation_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            self._send_json(status, body)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Localhost OpenAI-style QNN server with transparent sampling controls.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--timestamp", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--log-root", type=Path, default=STATE_ROOT / "logs")
    parser.add_argument("--chunk-context", type=Path, default=DEFAULT_CHUNK_CONTEXT)
    parser.add_argument("--decode-context", type=Path, default=DEFAULT_DECODE_CONTEXT)
    parser.add_argument("--tokenizer", type=Path, default=STATE_ROOT / "models" / "tokenizer" / "tokenizer.json")
    parser.add_argument("--rope-cache", type=Path, default=STATE_ROOT / "models" / "host" / "rope_cache.npz")
    parser.add_argument("--v0-runner-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--embedding-int8-dir", type=Path, default=STATE_ROOT / "models" / "host" / "embedding_int8_rowwise")
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--total-len", type=int, default=2048)
    parser.add_argument("--default-profile", choices=sorted(SAMPLING_PROFILES), default=DEFAULT_PROFILE)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("V1.8b server only permits a loopback host")
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be in [1, 65535]")
    engine = None
    server = None
    try:
        engine = QNNEngine(args)
        server = APIServer((args.host, args.port), Handler, engine)

        def stop_handler(signum, _frame):
            print(json.dumps({"event": "signal_shutdown", "signal": signal.Signals(signum).name}, ensure_ascii=False), flush=True)
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        print(json.dumps({"ready": True, "url": f"http://{args.host}:{args.port}", "log_dir": str(engine.log_dir)}, ensure_ascii=False), flush=True)
        server.serve_forever(poll_interval=0.2)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if server is not None:
            server.server_close()
        if engine is not None:
            engine.close()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
