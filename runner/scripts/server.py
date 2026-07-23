#!/usr/bin/env python3
"""Loopback-by-default OpenAI-style API backed by the ctx2048 QNN runner.

OpenWebUI compatibility: passive tool metadata is accepted for normal chat,
but this runner never executes tools or emits tool calls.
"""

from __future__ import annotations

import argparse
from collections import deque
import gc
import ipaddress
import json
import math
import os
import queue
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from runtime_contract import collect_runtime_fingerprint, qnn_only_from_counts


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ROOT = Path(
    os.environ.get("LFM2_5_STATE_DIR")
    or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "lfm2.5-350m-q6a-qcs6490-qnn-npu"
)
MODEL_ID = "lfm2.5-350m-qnn-ctx2048"
DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS = 1024
MAX_SEED = (1 << 63) - 1
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


class AdmissionError(RuntimeError):
    def __init__(self, message: str, status: int, code: str):
        super().__init__(message)
        self.status = status
        self.code = code


class GenerationCancelled(RuntimeError):
    pass


def is_integer(value) -> bool:
    return type(value) is int


def is_finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_port(port) -> int:
    if not is_integer(port) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in [1, 65535]")
    return port


def reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


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
    if not is_finite_number(temperature) or temperature < 0 or temperature > 2:
        raise ValueError("temperature must be a number in [0, 2]")
    if not is_finite_number(top_p) or top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be a number in (0, 1]")
    if not is_integer(top_k) or top_k < 1 or top_k > 4096:
        raise ValueError("top_k must be an integer in [1, 4096]")
    if not is_finite_number(penalty) or penalty < 0.1 or penalty > 2:
        raise ValueError("repetition_penalty must be a number in [0.1, 2]")
    if not is_integer(repetition_last_n) or repetition_last_n < -1 or repetition_last_n > total_len:
        raise ValueError(f"repetition_last_n must be an integer in [-1, {total_len}]")
    if not is_integer(min_new_tokens) or min_new_tokens < 0 or min_new_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"min_new_tokens must be an integer in [0, {MAX_NEW_TOKENS}]")
    max_tokens = request.get("max_tokens", DEFAULT_MAX_NEW_TOKENS)
    if not is_integer(max_tokens) or max_tokens < 1 or max_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"max_tokens must be an integer in [1, {MAX_NEW_TOKENS}]")
    seed = request.get("seed", 2468)
    if not is_integer(seed) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {MAX_SEED}]")
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
    args.seed = seed
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
        values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = value
    else:
        raise ValueError("stop must be a string or an array of strings")
    if not 1 <= len(values) <= 4:
        raise ValueError("stop must contain between 1 and 4 strings")
    if any(not item for item in values):
        raise ValueError("stop strings must not be empty")
    if any(len(item) > 1024 for item in values):
        raise ValueError("stop strings must be at most 1024 characters")
    return values


class StopSequenceFilter:
    """Hold possible stop prefixes until they are known to be ordinary text."""

    def __init__(self, stop_strings: list[str]):
        self.stop_strings = [item for item in stop_strings if item]
        self.pending = ""
        self.output = ""
        self.match: str | None = None

    def push(self, text: str) -> tuple[str, str | None]:
        if self.match is not None or not text:
            return "", self.match
        if not self.stop_strings:
            self.output += text
            return text, None
        self.pending += text
        matches = [
            (self.pending.find(stop), index, stop)
            for index, stop in enumerate(self.stop_strings)
            if stop in self.pending
        ]
        if matches:
            position, _index, stop = min(matches)
            safe = self.pending[:position]
            self.output += safe
            self.pending = ""
            self.match = stop
            return safe, stop
        hold = 0
        for stop in self.stop_strings:
            maximum = min(len(stop) - 1, len(self.pending))
            for length in range(maximum, 0, -1):
                if self.pending.endswith(stop[:length]):
                    hold = max(hold, length)
                    break
        safe = self.pending[:-hold] if hold else self.pending
        self.pending = self.pending[-hold:] if hold else ""
        self.output += safe
        return safe, None

    def finish(self) -> str:
        if self.match is not None:
            self.pending = ""
            return ""
        safe = self.pending
        self.pending = ""
        self.output += safe
        return safe


def nested_arrays_finite(value, seen: set[int] | None = None) -> bool:
    seen = seen or set()
    identity = id(value)
    if identity in seen:
        return True
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, dict):
        return all(nested_arrays_finite(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(nested_arrays_finite(item, seen) for item in value)
    return True


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
        self.state_changed = threading.Condition()
        self.request_records = deque(maxlen=args.request_history_limit)
        self.requests_completed = 0
        self.requests_failed = 0
        self.active_generation = 0
        self.waiting_requests = 0
        self.cancel_events: set[threading.Event] = set()
        self.draining = False
        self.closed = False
        self.ready = False
        self.qnn_only_verified = False
        self.profile_provider_counts = None
        self.started_at = time.time()
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
        self.session_qnn_provider_created = all(
            "QNNExecutionProvider" in (item.get("session_providers") or [])
            for item in (self.chunk_session, self.decode_session)
        )
        import onnxruntime_qnn as oq

        self.runtime_fingerprint = collect_runtime_fingerprint(
            oq,
            provider_options=self.chunk_session.get("provider_options") or {},
            session_config={"session.disable_cpu_ep_fallback": "1"},
            chunk=args.chunk,
            total_length=args.total_len,
        )
        self.runtime_fingerprint["decode_execution_contract"] = {
            "provider_options": self.decode_session.get("provider_options") or {},
            "session_config": {"session.disable_cpu_ep_fallback": "1"},
            "chunk": 1,
            "total_length": args.total_len,
        }
        self.ready = True
        self.write_ready()

    def fallback_status(self) -> dict:
        return {
            "fallback_configured_disabled": True,
            "session_qnn_provider_created": bool(self.session_qnn_provider_created),
            "qnn_only_verified": bool(self.qnn_only_verified),
            "profile_provider_counts": self.profile_provider_counts,
        }

    def write_ready(self) -> None:
        data = {
            "ready": self.ready,
            "draining": self.draining,
            "model": MODEL_ID,
            "host": self.args.host,
            "port": self.args.port,
            "default_profile": self.args.default_profile,
            **self.fallback_status(),
            "embedding_backend": self.contexts["embedding_info"].get("backend"),
            "chunk_session": self.chunk_session,
            "decode_session": self.decode_session,
            "after_decode_load": self.after_decode_load,
            "runtime_fingerprint": self.runtime_fingerprint,
        }
        (self.log_dir / "ready.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def health(self) -> dict:
        with self.state_changed:
            return {
                "status": "ok" if self.ready else ("draining" if self.draining else "not_ready"),
                "ready": self.ready,
                "draining": self.draining,
                "model": MODEL_ID,
                "backend": "QNNExecutionProvider / HTP",
                **self.fallback_status(),
                "queue_serialized": True,
                "queue_limit": self.args.max_waiting_requests,
                "queue_waiting": self.waiting_requests,
                "generation_active": bool(self.active_generation),
                "default_profile": self.args.default_profile,
                "requests_completed": self.requests_completed,
                "requests_failed": self.requests_failed,
                "request_history_size": len(self.request_records),
                "request_history_limit": self.request_records.maxlen,
                "body_logging_enabled": self.args.log_bodies,
                "uptime_s": round(time.time() - self.started_at, 3),
            }

    def acquire_generation(self, cancel_event: threading.Event) -> None:
        deadline = time.monotonic() + self.args.queue_wait_timeout_s
        with self.state_changed:
            if self.draining or not self.ready:
                raise AdmissionError("server is draining", HTTPStatus.SERVICE_UNAVAILABLE, "server_draining")
            if self.active_generation and self.waiting_requests >= self.args.max_waiting_requests:
                raise AdmissionError("generation queue is full", HTTPStatus.TOO_MANY_REQUESTS, "queue_full")
            self.waiting_requests += 1
            self.cancel_events.add(cancel_event)
            try:
                while self.active_generation:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise AdmissionError(
                            "generation queue wait timed out",
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            "queue_timeout",
                        )
                    self.state_changed.wait(timeout=remaining)
                    if self.draining or cancel_event.is_set():
                        raise AdmissionError("server is draining", HTTPStatus.SERVICE_UNAVAILABLE, "server_draining")
                self.active_generation = 1
            except Exception:
                self.cancel_events.discard(cancel_event)
                raise
            finally:
                self.waiting_requests -= 1
                self.state_changed.notify_all()

    def release_generation(self, cancel_event: threading.Event, success: bool) -> None:
        with self.state_changed:
            self.active_generation = 0
            self.cancel_events.discard(cancel_event)
            if success:
                self.requests_completed += 1
            else:
                self.requests_failed += 1
            self.state_changed.notify_all()

    def record_failure(self, prepared: dict, exc: Exception) -> None:
        record = {
            "request_id": prepared.get("request_id"),
            "created": int(time.time()),
            "prompt_tokens": len(prepared.get("ids") or []),
            "requested_max_tokens": getattr(prepared.get("gen_args"), "max_new_tokens", None),
            "json_mode": bool(prepared.get("json_mode")),
            "tools_ignored": bool(prepared.get("tools_ignored")),
            "error_type": type(exc).__name__,
            "finish_reason": "error",
        }
        if self.args.log_bodies:
            record["model_prompt"] = prepared.get("model_prompt")
            record["prompt_token_ids"] = prepared.get("ids")
        with self.state_changed:
            self.request_records.append(record)

    def begin_shutdown(self) -> None:
        with self.state_changed:
            self.ready = False
            self.draining = True
            for cancel_event in self.cancel_events:
                cancel_event.set()
            self.state_changed.notify_all()
        self.write_ready()

    def wait_for_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self.state_changed:
            while self.active_generation or self.waiting_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.state_changed.wait(timeout=remaining)
            return True

    def prepare_request(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError("request body must be an object")
        if "stream" in request and type(request["stream"]) is not bool:
            raise ValueError("stream must be a boolean")
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
        if "n" in request and (not is_integer(request["n"]) or request["n"] != 1):
            raise ValueError("n must be the integer 1")
        response_format = (
            {"type": "text"}
            if request.get("response_format") is None
            else request["response_format"]
        )
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
        return {
            "request": request,
            "request_id": request.get("_request_id"),
            "model_prompt": model_prompt,
            "ids": ids,
            "gen_args": gen_args,
            "stop_strings": stop_strings,
            "stop_ids": stop_ids,
            "json_mode": json_mode,
            "tools_ignored": tools_ignored,
        }

    def _decode(
        self,
        prefill: dict,
        ids: list[int],
        gen_args: GenerationArgs,
        stop_strings: list[str],
        json_mode: bool,
        on_token,
        cancel_event: threading.Event,
    ) -> dict:
        feed = prefill["decode_feed_after_prefill"]
        current_logits = np.asarray(prefill["last_logits"])
        if not np.isfinite(current_logits).all():
            raise RuntimeError("prefill logits contain NaN or Infinity")
        logical = int(prefill["logical_cache_length"])
        generated_ids: list[int] = []
        raw_decoded_text = ""
        events: list[dict] = []
        decode_run_s: list[float] = []
        stop_filter = StopSequenceFilter(stop_strings)
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
        first_step_top_token_ids = None
        started = time.perf_counter()
        for step in range(gen_args.max_new_tokens):
            if cancel_event.is_set():
                raise GenerationCancelled("generation cancelled")
            candidates = None
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
                        self.v0,
                        current_logits,
                        self.tokenizer,
                        json_has_value,
                        json_value_length,
                        256,
                        gen_args.logit_bias,
                    )
                    if closes_value:
                        json_phase = "suffix"
                    else:
                        json_has_value = True
                    policy = "json_qnn_value_constraint"
                    eos_suppressed = False
            else:
                next_id, candidates, policy, eos_suppressed = choose_profiled_next(
                    self.handoff,
                    self.v0,
                    current_logits,
                    token_vocab_size,
                    gen_args,
                    rng,
                    ids + generated_ids,
                    stop_ids,
                    step,
                )
                selected_rank = None
                if step == 0:
                    first_step_top_token_ids = [
                        int(item["token_id"])
                        for item in (candidates or [])[:5]
                        if isinstance(item, dict) and "token_id" in item
                    ]
            if eos_suppressed:
                eos_suppressed_steps += 1
            generated_ids.append(int(next_id))
            decoded_text = (
                decode_visible_text(self.tokenizer, generated_ids)
                if 0 <= int(next_id) < token_vocab_size
                else raw_decoded_text
            )
            token_text = (
                decoded_text[len(raw_decoded_text):]
                if decoded_text.startswith(raw_decoded_text)
                else ""
            )
            raw_decoded_text = decoded_text
            visible_text, stop_match = stop_filter.push(token_text)
            event = {
                "step": step,
                "token_id": int(next_id),
                "text": visible_text,
                "elapsed_s": time.perf_counter() - started,
                "selection_policy": policy,
                "eos_suppressed_by_min_new_tokens": bool(eos_suppressed),
                "logit_bias_applied": bool(gen_args.logit_bias),
            }
            if step == 0:
                event["top_token_ids"] = first_step_top_token_ids
            if selected_rank is not None:
                event["selected_rank_in_candidate_pool"] = selected_rank
            events.append(event)
            if visible_text and on_token is not None:
                on_token(visible_text, not stream_started)
                stream_started = True
            if stop_match is not None:
                stopped = True
                finish_reason = "stop"
            elif json_mode and json_phase == "done":
                stopped = True
                finish_reason = "stop"
            elif next_id in stop_ids:
                stopped = True
                finish_reason = "stop"
            if stopped or step + 1 >= gen_args.max_new_tokens:
                tail = stop_filter.finish()
                if tail:
                    event["text"] += tail
                    if on_token is not None:
                        on_token(tail, not stream_started)
                        stream_started = True
                break
            if cancel_event.is_set():
                raise GenerationCancelled("generation cancelled")
            if json_mode and json_phase == "suffix":
                continue
            self.handoff.set_hidden(self.v0, feed, self.contexts["embedding_state"], next_id)
            self.v0.apply_position_feeds(
                feed,
                self.contexts["decode_input_specs"],
                logical,
                gen_args,
                self.contexts["decode_rope_context"],
            )
            run_started = time.perf_counter()
            outputs = self.decode_sess.run(None, feed)
            decode_run_s.append(time.perf_counter() - run_started)
            if not all(np.isfinite(np.asarray(value)).all() for value in outputs):
                raise RuntimeError("decode graph output contains NaN or Infinity")
            logical += 1
            current_logits = np.asarray(outputs[0])
            self.handoff.update_decode_cache_from_outputs(feed, self.contexts["decode_output_names"], outputs)
        generated_text = stop_filter.output
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
            "first_token_id": generated_ids[0] if generated_ids else None,
            "first_step_top_token_ids": first_step_top_token_ids,
            "all_outputs_finite": True,
        }

    def generate_prepared(
        self,
        prepared: dict,
        on_token=None,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        cancel_event = cancel_event or threading.Event()
        request = prepared["request"]
        ids = prepared["ids"]
        gen_args = prepared["gen_args"]
        json_mode = prepared["json_mode"]
        stop_ids = prepared["stop_ids"]
        with self.lock:
            if cancel_event.is_set():
                raise GenerationCancelled("generation cancelled")
            started = time.perf_counter()
            prefill = self.handoff.run_prefill_with_decode_tail(self.v0, self.chunk_sess, self.decode_sess, self.contexts, ids, gen_args)
            ttft = time.perf_counter() - started
            if not nested_arrays_finite(prefill):
                raise RuntimeError("prefill graph output contains NaN or Infinity")
            cache_prefill = self.v0.cache_stats(prefill["decode_feed_after_prefill"], prefill["logical_cache_length"])
            decoded = self._decode(
                prefill,
                ids,
                gen_args,
                prepared["stop_strings"],
                json_mode,
                on_token,
                cancel_event,
            )
            parsed_json = (decoded["json_check"] or {}).get("parsed")
            result = {
                "model_prompt": prepared["model_prompt"],
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
                "json_object_valid": bool(
                    json_mode
                    and (decoded["json_check"] or {}).get("syntax_valid")
                    and isinstance(parsed_json, dict)
                    and set(parsed_json) == {"answer"}
                    and isinstance(parsed_json["answer"], str)
                    and len(parsed_json["answer"]) <= 96
                ),
                "json_constraint": "fixed JSON structure tokens plus QNN-logit-constrained answer value" if json_mode else None,
                "tools_ignored": prepared["tools_ignored"],
                "embedding_backend": self.contexts["embedding_info"].get("backend"),
                "model_compute_backend": "QNNExecutionProvider / HTP",
                "first_token_id": decoded["first_token_id"],
                "first_step_top_token_ids": decoded["first_step_top_token_ids"],
                "all_outputs_finite": decoded["all_outputs_finite"],
                **self.fallback_status(),
            }
            record = {
                "request_id": prepared["request_id"],
                "created": int(time.time()),
                "prompt_tokens": result["prompt_token_count"],
                "completion_tokens": result["completion_tokens"],
                "requested_max_tokens": result["requested_max_tokens"],
                "finish_reason": result["finish_reason"],
                "sampling_profile": result["sampling_profile"],
                "sampling_parameters": result["sampling_parameters"],
                "eos_suppressed_steps": result["eos_suppressed_steps"],
                "logit_bias": result["logit_bias"],
                "json_mode": json_mode,
                "tools_ignored": prepared["tools_ignored"],
                "prefill_tok_s": (result["prefill_speed"] or {}).get("tok_per_s"),
                "decode_tok_s": (result["decode_speed"] or {}).get("tok_per_s"),
                "all_outputs_finite": result["all_outputs_finite"],
            }
            if self.args.log_bodies:
                record.update(
                    {
                        "model_prompt": result["model_prompt"],
                        "prompt_token_ids": result["prompt_token_ids"],
                        "generated_token_ids": result["generated_token_ids"],
                        "generated_text": result["generated_text"],
                    }
                )
            with self.state_changed:
                self.request_records.append(record)
            return result

    def generate(self, request: dict, on_token=None) -> dict:
        """Direct-call compatibility wrapper with the same bounded admission path."""
        prepared = self.prepare_request(request)
        cancel_event = threading.Event()
        self.acquire_generation(cancel_event)
        success = False
        try:
            result = self.generate_prepared(prepared, on_token=on_token, cancel_event=cancel_event)
            success = True
            return result
        finally:
            self.release_generation(cancel_event, success)

    def close(self) -> dict:
        if self.closed:
            return self.close_result
        self.begin_shutdown()
        idle_before_profile = self.wait_for_idle(self.args.shutdown_timeout_s)
        profiles = []
        acquired = self.lock.acquire(timeout=self.args.shutdown_timeout_s)
        if acquired:
            try:
                if self.chunk_sess is not None:
                    profiles.append(self.handoff.finish_profile(self.chunk_sess, "chunk"))
                if self.decode_sess is not None:
                    profiles.append(self.handoff.finish_profile(self.decode_sess, "decode"))
                self.chunk_sess = None
                self.decode_sess = None
            finally:
                self.lock.release()
        else:
            profiles.append(
                {
                    "label": "shutdown",
                    "profile_error": "timed out waiting for the QNN engine lock; profiling was not ended concurrently",
                }
            )
        qnn_only_by_profile = self.handoff.profile_qnn_only(profiles)
        self.profile_provider_counts = {
            item.get("label", "unknown"): item.get("provider_counts")
            for item in profiles
        }
        self.qnn_only_verified = bool(
            acquired
            and qnn_only_by_profile.get("chunk")
            and qnn_only_by_profile.get("decode")
        )
        result = {
            "mode": "lfm2_5_q6a_openai_server",
            **self.fallback_status(),
            "model_compute_backend": "QNNExecutionProvider / HTP",
            "chunk_session": self.chunk_session,
            "decode_session": self.decode_session,
            "after_chunk_load": self.after_chunk_load,
            "after_decode_load": self.after_decode_load,
            "embedding_info": self.contexts.get("embedding_info"),
            "profiles": profiles,
            "qnn_only_by_profile": qnn_only_by_profile,
            "runtime_fingerprint": self.runtime_fingerprint,
            "requests": list(self.request_records),
            "requests_completed": self.requests_completed,
            "requests_failed": self.requests_failed,
            "request_history_limit": self.request_records.maxlen,
            "body_logging_enabled": self.args.log_bodies,
            "shutdown": {
                "draining": self.draining,
                "idle_before_profile": idle_before_profile,
                "engine_lock_acquired": acquired,
                "profiling_race_prevented": bool(idle_before_profile and acquired),
            },
            "proc_final": self.handoff.proc_status(),
            "thermal_final": self.handoff.thermal_snapshot(),
            "power_final": self.handoff.power_snapshot(),
        }
        (self.log_dir / "server_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.ready = False
        self.closed = True
        self.close_result = result
        self.write_ready()
        return result


class APIServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

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
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

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
        body = json.loads(
            self.rfile.read(size).decode("utf-8"),
            parse_constant=reject_json_constant,
        )
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        return body

    @staticmethod
    def _qnn_metrics(result: dict) -> dict:
        keys = (
            "prefill_speed",
            "decode_speed",
            "ttft_s_excluding_session_create",
            "logical_cache_length_after_prefill",
            "logical_cache_length_after_decode",
            "requested_max_tokens",
            "finish_reason",
            "sampling_profile",
            "sampling_parameters",
            "eos_suppressed_steps",
            "logit_bias",
            "json_mode",
            "json_check",
            "json_object_valid",
            "json_constraint",
            "first_token_id",
            "first_step_top_token_ids",
            "all_outputs_finite",
            "fallback_configured_disabled",
            "session_qnn_provider_created",
            "qnn_only_verified",
            "profile_provider_counts",
        )
        return {key: result.get(key) for key in keys}

    def _chat_response(self, request: dict) -> None:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        request["_request_id"] = request_id
        created = int(time.time())
        prepared = self.server.engine.prepare_request(request)
        stream = request.get("stream", False)
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

        cancel_event = threading.Event()
        self.server.engine.acquire_generation(cancel_event)
        if not stream:
            success = False
            try:
                result = self.server.engine.generate_prepared(
                    prepared,
                    cancel_event=cancel_event,
                )
                success = True
                log_result(result)
            except Exception as exc:
                self.server.engine.record_failure(prepared, exc)
                raise
            finally:
                self.server.engine.release_generation(cancel_event, success)
            body = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["generated_text"]}, "finish_reason": result["finish_reason"]}],
                "usage": {"prompt_tokens": result["prompt_token_count"], "completion_tokens": result["completion_tokens"], "total_tokens": result["prompt_token_count"] + result["completion_tokens"]},
                "qnn_metrics": self._qnn_metrics(result),
            }
            self._send_json(HTTPStatus.OK, body)
            return

        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
        except Exception:
            cancel_event.set()
            self.server.engine.release_generation(cancel_event, False)
            raise

        self.connection.settimeout(self.server.engine.args.client_write_timeout_s)
        sent_role = False
        stream_queue = queue.Queue(maxsize=self.server.engine.args.stream_queue_size)
        done_marker = object()

        def publish(kind: str, value) -> None:
            while not cancel_event.is_set():
                try:
                    stream_queue.put((kind, value), timeout=0.1)
                    return
                except queue.Full:
                    continue
            raise GenerationCancelled("stream consumer disconnected")

        def send_event(body: dict):
            self.wfile.write(f"data: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8"))
            self.wfile.flush()

        def on_token(text: str, first: bool):
            publish("token", text)

        def generation_worker() -> None:
            success = False
            try:
                result = self.server.engine.generate_prepared(
                    prepared,
                    on_token=on_token,
                    cancel_event=cancel_event,
                )
                publish("result", result)
                success = True
            except Exception as exc:
                self.server.engine.record_failure(prepared, exc)
                if not cancel_event.is_set():
                    try:
                        publish(
                            "error",
                            {
                                "message": f"generation failed: {exc}",
                                "type": "server_error",
                                "param": None,
                                "code": "generation_error",
                            },
                        )
                    except GenerationCancelled:
                        pass
            finally:
                self.server.engine.release_generation(cancel_event, success)
                if not cancel_event.is_set():
                    while True:
                        try:
                            stream_queue.put(("done", done_marker), timeout=0.1)
                            break
                        except queue.Full:
                            if cancel_event.is_set():
                                break

        worker = threading.Thread(
            target=generation_worker,
            name=f"qnn-generation-{request_id}",
            daemon=False,
        )
        worker.start()
        try:
            while True:
                kind, value = stream_queue.get()
                if kind == "token":
                    delta = {"content": value}
                    if not sent_role:
                        delta["role"] = "assistant"
                        sent_role = True
                    send_event({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    })
                elif kind == "result":
                    result = value
                    log_result(result)
                    if not sent_role:
                        send_event({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                        })
                        sent_role = True
                    send_event({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}],
                        "usage": {
                            "prompt_tokens": result["prompt_token_count"],
                            "completion_tokens": result["completion_tokens"],
                            "total_tokens": result["prompt_token_count"] + result["completion_tokens"],
                        },
                        "qnn_metrics": self._qnn_metrics(result),
                    })
                elif kind == "error":
                    payload = json.dumps({"error": value}, ensure_ascii=False, separators=(",", ":"))
                    self.wfile.write(f"event: error\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                elif kind == "done":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    break
        except (BrokenPipeError, ConnectionResetError, TimeoutError, socket.timeout, OSError):
            cancel_event.set()
            print(json.dumps({"event": "client_disconnect", "request_id": request_id}, ensure_ascii=False), flush=True)
        finally:
            self.close_connection = True

    def do_GET(self):
        if self.path == "/health":
            health = self.server.engine.health()
            self._send_json(
                HTTPStatus.OK if health["ready"] else HTTPStatus.SERVICE_UNAVAILABLE,
                health,
            )
        elif self.path == "/v1/models":
            self._send_json(HTTPStatus.OK, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local-qnn"}]})
        else:
            status, body = api_error("route not found", "not_found", HTTPStatus.NOT_FOUND)
            self._send_json(status, body)

    def do_POST(self):
        try:
            if self.path == "/v1/chat/completions":
                request = self._read_json()
                self._chat_response(request)
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
        except AdmissionError as exc:
            status, body = api_error(str(exc), exc.code, exc.status)
            self._send_json(status, body)
        except GenerationCancelled as exc:
            status, body = api_error(str(exc), "generation_cancelled", HTTPStatus.SERVICE_UNAVAILABLE)
            self._send_json(status, body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, socket.timeout, OSError):
            print(json.dumps({"event": "client_disconnect", "path": self.path}, ensure_ascii=False), flush=True)
            return
        except Exception as exc:
            print(json.dumps({"event": "request_error", "path": self.path, "error_type": type(exc).__name__}, ensure_ascii=False), flush=True)
            status, body = api_error(f"generation failed: {exc}", "generation_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            self._send_json(status, body)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI-style QNN server; loopback-only unless --allow-lan is set."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--allow-lan",
        action="store_true",
        help="permit binding --host to a non-loopback address (no auth/TLS; do not expose on an untrusted network)",
    )
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
    parser.add_argument("--max-waiting-requests", type=int, default=4)
    parser.add_argument("--queue-wait-timeout-s", type=float, default=30.0)
    parser.add_argument("--stream-queue-size", type=int, default=32)
    parser.add_argument("--client-write-timeout-s", type=float, default=5.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=30.0)
    parser.add_argument("--request-history-limit", type=int, default=128)
    parser.add_argument(
        "--log-bodies",
        action="store_true",
        help="opt in to storing prompts, generated text, and token ids in bounded server history",
    )
    return parser


def validate_bind_host(host: str, allow_lan: bool) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None

    if address is not None and address.version == 6:
        raise SystemExit("IPv6 bind hosts are not supported")

    is_loopback = host == "localhost" or (address is not None and address.is_loopback)
    if is_loopback:
        return
    if not allow_lan:
        raise SystemExit(
            "server only permits a loopback host by default; "
            "pass --allow-lan to bind a non-loopback address"
        )
    print(
        json.dumps(
            {
                "warning": "non_loopback_bind",
                "host": host,
                "detail": "no authentication or TLS is provided; do not expose on an untrusted network",
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def validate_server_args(args: argparse.Namespace) -> None:
    try:
        validate_port(args.port)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for name in (
        "max_waiting_requests",
        "stream_queue_size",
        "request_history_limit",
    ):
        value = getattr(args, name)
        minimum = 0 if name == "max_waiting_requests" else 1
        if not is_integer(value) or value < minimum:
            raise SystemExit(f"{name.replace('_', '-')} must be an integer >= {minimum}")
    for name in (
        "queue_wait_timeout_s",
        "client_write_timeout_s",
        "shutdown_timeout_s",
    ):
        value = getattr(args, name)
        if not is_finite_number(value) or value <= 0:
            raise SystemExit(f"{name.replace('_', '-')} must be a finite number > 0")


def main() -> int:
    args = make_parser().parse_args()
    validate_bind_host(args.host, args.allow_lan)
    validate_server_args(args)
    engine = None
    server = None
    try:
        engine = QNNEngine(args)
        server = APIServer((args.host, args.port), Handler, engine)
        shutdown_started = threading.Event()

        def stop_handler(signum, _frame):
            if shutdown_started.is_set():
                return
            shutdown_started.set()
            print(json.dumps({"event": "signal_shutdown", "signal": signal.Signals(signum).name}, ensure_ascii=False), flush=True)
            engine.begin_shutdown()
            threading.Thread(
                target=server.shutdown,
                name="http-server-shutdown",
                daemon=False,
            ).start()

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
        if engine is not None:
            engine.begin_shutdown()
        if server is not None:
            server.server_close()
        if engine is not None:
            engine.wait_for_idle(args.shutdown_timeout_s)
            engine.close()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
