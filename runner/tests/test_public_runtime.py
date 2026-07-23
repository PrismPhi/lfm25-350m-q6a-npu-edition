from __future__ import annotations

import argparse
from collections import deque
import http.client
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_module(name: str, filename: str):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StreamTestEngine:
    def __init__(self, mode: str):
        self.mode = mode
        self.args = argparse.Namespace(
            client_write_timeout_s=1.0,
            stream_queue_size=2,
        )
        self.started = threading.Event()
        self.released = threading.Event()
        self.recorded = threading.Event()
        self.cancel_event = None

    def prepare_request(self, request):
        return {
            "request": request,
            "request_id": request.get("_request_id"),
        }

    def acquire_generation(self, cancel_event):
        self.cancel_event = cancel_event

    def generate_prepared(self, prepared, on_token, cancel_event):
        self.started.set()
        if self.mode == "wait_for_cancel":
            if not cancel_event.wait(2):
                raise RuntimeError("test cancellation was not requested")
            raise RuntimeError("synthetic cancellation")
        if self.mode == "cancel_without_terminal":
            cancel_event.set()
            raise RuntimeError("synthetic cancellation")
        if self.mode == "runtime_error":
            raise RuntimeError("synthetic QNN failure")
        on_token("hello", True)
        return {
            "prompt_token_count": 1,
            "completion_tokens": 1,
            "requested_max_tokens": 1,
            "finish_reason": "stop",
            "sampling_profile": "chat",
            "eos_suppressed_steps": 0,
            "logit_bias": {},
        }

    def record_failure(self, prepared, exc):
        self.recorded.set()

    def release_generation(self, cancel_event, success):
        self.released.set()

    def begin_shutdown(self):
        if self.cancel_event is not None:
            self.cancel_event.set()


class PublicRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = load_module("public_server", "server.py")
        cls.install = load_module("public_install", "install.py")
        cls.runtime = load_module("public_runtime_contract", "runtime_contract.py")

    def test_chat_template_is_stable(self):
        rendered = self.server.render_chatml([{"role": "user", "content": "Hello"}])
        self.assertTrue(rendered.startswith("<|startoftext|>"))
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))

    def test_json_mode_request_parameters(self):
        args = self.server.make_generation_args(
            {"max_tokens": 32, "temperature": 0},
            total_len=2048,
            chunk=16,
            default_profile="chat",
        )
        self.assertEqual(args.max_new_tokens, 32)
        self.assertEqual(args.chunk, 16)
        self.assertTrue(args.greedy)

    def test_server_bind_defaults_to_loopback_only(self):
        args = self.server.make_parser().parse_args([])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertFalse(args.allow_lan)
        self.server.validate_bind_host("127.0.0.2", allow_lan=False)
        with self.assertRaisesRegex(SystemExit, "loopback host by default"):
            self.server.validate_bind_host("0.0.0.0", allow_lan=False)

    def test_server_allow_lan_emits_structured_warning(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.server.validate_bind_host("0.0.0.0", allow_lan=True)
        warning = json.loads(stderr.getvalue())
        self.assertEqual(warning["warning"], "non_loopback_bind")
        self.assertEqual(warning["host"], "0.0.0.0")
        self.assertIn("no authentication or TLS", warning["detail"])

    def test_server_rejects_unsupported_ipv6_bind(self):
        with self.assertRaisesRegex(SystemExit, "IPv6 bind hosts are not supported"):
            self.server.validate_bind_host("::1", allow_lan=False)

    def test_asset_manifest_schema(self):
        manifest_path = SCRIPT_DIR.parent / "config" / "model-assets.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(len(manifest["files"]), 11)
        self.assertEqual(len({item["path"] for item in manifest["files"]}), 11)
        self.assertEqual(
            manifest["huggingface"]["pinned_revision"],
            self.install.PINNED_MODEL_REVISION,
        )

    def test_public_model_repository_is_installer_default(self):
        with patch.dict(os.environ, {}, clear=True):
            args = self.install.make_parser().parse_args([])
        self.assertIsNone(args.model_base_url)
        self.assertEqual(args.model_repository, self.install.DEFAULT_MODEL_REPOSITORY)
        self.assertEqual(args.model_revision, self.install.PINNED_MODEL_REVISION)
        self.assertEqual(
            self.install.DEFAULT_MODEL_BASE_URL,
            "https://huggingface.co/PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu/"
            "resolve/773ff42cc383cb61ecf32eb13d1f828634fbd0e1",
        )
        self.assertEqual(
            self.install.DEFAULT_STATE_DIR.name,
            "lfm2.5-350m-q6a-qcs6490-qnn-npu",
        )

    def test_new_model_base_url_environment_override(self):
        override = "https://mirror.example/models"
        with patch.dict(os.environ, {"LFM2_5_MODEL_BASE_URL": override}, clear=True):
            args = self.install.make_parser().parse_args([])
        self.assertEqual(args.model_base_url, override)

    def test_runtime_fingerprint_records_library_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ep = root / "libonnxruntime_providers_qnn.so"
            htp = root / "libQnnHtp.so"
            stub = root / "libQnnHtpV68Stub.so"
            skel = root / "libQnnHtpV68Skel.so"
            for path, payload in (
                (ep, b"ep"),
                (htp, b"htp"),
                (stub, b"stub"),
                (skel, b"skel"),
            ):
                path.write_bytes(payload)

            oq = types.SimpleNamespace(
                __version__="2.3.0",
                get_ep_name=lambda: "QNNExecutionProvider",
                get_library_path=lambda: str(ep),
                get_qnn_htp_path=lambda: str(htp),
            )
            modules = {
                "onnx": types.SimpleNamespace(__version__="1.22.0"),
                "onnxruntime": types.SimpleNamespace(__version__="1.27.0"),
                "tokenizers": types.SimpleNamespace(__version__="0.23.1"),
            }
            with patch.dict(sys.modules, modules), patch.dict(
                os.environ,
                {"ADSP_LIBRARY_PATH": "/custom;/dsp"},
                clear=False,
            ):
                fingerprint = self.runtime.collect_runtime_fingerprint(
                    oq,
                    provider_options={"backend_path": str(htp)},
                    session_config={"session.disable_cpu_ep_fallback": "1"},
                    chunk=16,
                    total_length=2048,
                )
            self.assertEqual(fingerprint["target"]["soc"], "QCS6490")
            self.assertEqual(fingerprint["target"]["htp_generation"], "v68")
            self.assertEqual(
                fingerprint["qnn"]["ep_library"]["sha256"],
                self.runtime.sha256_file(ep),
            )
            self.assertEqual(
                fingerprint["qnn"]["dsp"]["skel"]["sha256"],
                self.runtime.sha256_file(skel),
            )
            self.assertIsNone(fingerprint["qnn"]["qairt_version"])
            self.assertIsNone(fingerprint["qnn"]["qnn_runtime_version"])
            self.assertTrue(fingerprint["identity_sha256"])

    def test_adsp_library_path_preserves_existing_entries(self):
        merged = self.runtime.merge_adsp_library_path(
            "/custom;;/dsp;/custom",
            ("/ep", "/usr/lib/dsp/cdsp", "/dsp"),
        )
        self.assertEqual(merged, "/custom;/dsp;/ep;/usr/lib/dsp/cdsp")

    @staticmethod
    def _fingerprint(ep_hash="ep-a"):
        return {
            "packages": {
                "onnxruntime": "1.27.0",
                "onnxruntime_qnn": "2.3.0",
            },
            "qnn": {
                "provider_name": "QNNExecutionProvider",
                "ep_library": {"sha256": ep_hash},
                "htp_backend_library": {"sha256": "htp-a"},
                "dsp": {
                    "stub": {"sha256": "stub-a"},
                    "skel": {"sha256": "skel-a"},
                },
            },
            "target": {"soc": "QCS6490", "htp_generation": "v68"},
            "execution_contract": {
                "provider_options": {"backend_path": "/qnn/libQnnHtp.so"},
                "session_config": {"session.disable_cpu_ep_fallback": "1"},
                "chunk": 16,
                "total_length": 2048,
            },
        }

    def test_context_stamp_reuses_only_matching_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            context_dir = Path(directory)
            context = context_dir / "chunk_epcontext.onnx"
            context.write_bytes(b"context")
            fingerprint = self._fingerprint()
            stamp = {
                "schema_version": 2,
                "source_sha256": "source-a",
                "runtime_fingerprint": fingerprint,
                "context_files": [{
                    "name": context.name,
                    "size": context.stat().st_size,
                    "sha256": self.install.sha256_file(context),
                }],
                "qnn_only": True,
                "all_outputs_finite": True,
                "strict_summary": {"ok": True},
            }
            (context_dir / "source-stamp.json").write_text(
                json.dumps(stamp),
                encoding="utf-8",
            )
            self.assertEqual(
                self.install.context_stamp_status(context_dir, "source-a", fingerprint),
                (True, "identity_and_files_match"),
            )
            valid, reason = self.install.context_stamp_status(
                context_dir,
                "source-a",
                self._fingerprint(ep_hash="ep-b"),
            )
            self.assertFalse(valid)
            self.assertEqual(reason, "runtime_fingerprint_changed")

    def test_corrupt_context_stamp_forces_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            context_dir = Path(directory)
            (context_dir / "source-stamp.json").write_text("{", encoding="utf-8")
            self.assertEqual(
                self.install.context_stamp_status(
                    context_dir,
                    "source-a",
                    self._fingerprint(),
                ),
                (False, "missing_or_corrupt_stamp"),
            )

    def test_strict_execution_rejects_nonfinite_and_cpu_execution(self):
        nonfinite = self.runtime.strict_execution_status(
            session_created=True,
            graph_executed=True,
            finite_by_output={"logits": False},
            provider_counts={"QNNExecutionProvider": 1, "CPUExecutionProvider": 0},
        )
        self.assertFalse(nonfinite["ok"])
        fallback = self.runtime.strict_execution_status(
            session_created=True,
            graph_executed=True,
            finite_by_output={"logits": True},
            provider_counts={"QNNExecutionProvider": 1, "CPUExecutionProvider": 1},
        )
        self.assertFalse(fallback["ok"])
        self.assertFalse(fallback["qnn_only"])

    def test_semantic_canary_rejects_wrong_subject(self):
        canary = {
            "expected_subject": "Tokyo",
            "expected_first_token_id": 40550,
        }
        normal = {
            "choices": [{"message": {"content": "Japan"}}],
            "qnn_metrics": {"first_token_id": 40550, "all_outputs_finite": True},
        }
        structured = {
            "choices": [{"message": {"content": '{"answer":"Tokyo"}'}}],
            "qnn_metrics": {"all_outputs_finite": True},
        }
        with self.assertRaisesRegex(self.install.InstallError, "expected 'Tokyo'"):
            self.install.validate_semantic_canary(normal, structured, canary)

    def test_canary_definition_matches_recorded_golden(self):
        canary = self.install.load_canary(self.install.DEFAULT_CANARY)
        self.assertTrue(canary["evidence_verified"])
        self.assertEqual(canary["expected_subject"], "Tokyo")
        self.assertEqual(canary["expected_first_token_id"], 40550)
        self.assertEqual(len(canary["evidence_sha256"]), 64)

    def test_falsey_response_format_and_nonboolean_stream_are_rejected(self):
        engine = self.server.QNNEngine.__new__(self.server.QNNEngine)
        base = {"messages": [{"role": "user", "content": "hello"}]}
        with self.assertRaisesRegex(ValueError, "response_format"):
            engine.prepare_request({**base, "response_format": []})
        with self.assertRaisesRegex(ValueError, "stream must be a boolean"):
            engine.prepare_request({**base, "stream": "true"})

    def test_stop_filter_holds_multitoken_and_unicode_stops(self):
        def apply(chunks, stops):
            stop_filter = self.server.StopSequenceFilter(stops)
            visible = []
            for chunk in chunks:
                text, matched = stop_filter.push(chunk)
                visible.append(text)
                if matched:
                    break
            visible.append(stop_filter.finish())
            return "".join(visible), stop_filter.match

        streamed, match = apply(["abc<", "ST", "OP>leak"], ["<STOP>"])
        nonstreamed, _ = apply(["abc<STOP>leak"], ["<STOP>"])
        self.assertEqual(streamed, nonstreamed)
        self.assertEqual(streamed, "abc")
        self.assertEqual(match, "<STOP>")
        japanese, match = apply(["回答。終", "了で", "す残り"], ["終了です"])
        self.assertEqual(japanese, "回答。")
        self.assertEqual(match, "終了です")
        immediate = self.server.StopSequenceFilter([])
        self.assertEqual(immediate.push("即時")[0], "即時")

    def test_bool_and_nonfinite_numeric_inputs_are_rejected(self):
        fields = (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "repetition_last_n",
            "min_new_tokens",
            "max_tokens",
            "seed",
        )
        for field in fields:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.server.make_generation_args(
                    {field: True},
                    total_len=2048,
                    chunk=16,
                    default_profile="chat",
                )
        for field in ("temperature", "top_p", "repetition_penalty"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.server.make_generation_args(
                    {field: float("nan")},
                    total_len=2048,
                    chunk=16,
                    default_profile="chat",
                )
        with self.assertRaises(ValueError):
            self.server.validate_port(True)
        with self.assertRaises(ValueError):
            json.loads('{"temperature":NaN}', parse_constant=self.server.reject_json_constant)
        with self.assertRaises(ValueError):
            self.server.normalize_logit_bias({"1": float("inf")}, 10)

    def test_max_tokens_one_does_not_run_decode_graph(self):
        class Tokenizer:
            def get_vocab_size(self):
                return 10

            def decode(self, token_ids):
                return "x" if token_ids else ""

        class V0:
            def stop_token_ids(self, tokenizer, configured):
                return set()

            def choose_next(self, logits, vocab_size, args, rng):
                return 1, [{"token_id": 1}], "greedy"

            def cache_stats(self, feed, logical):
                return {"logical": logical}

        class Handoff:
            @staticmethod
            def apply_repetition_penalty(logits, history, penalty):
                return logits

            @staticmethod
            def weighted_speed(count, times):
                return {"tokens": count}

        class DecodeSession:
            def run(self, *_args):
                raise AssertionError("decode graph must not run")

        engine = self.server.QNNEngine.__new__(self.server.QNNEngine)
        engine.tokenizer = Tokenizer()
        engine.v0 = V0()
        engine.handoff = Handoff()
        engine.decode_sess = DecodeSession()
        engine.contexts = {}
        args = self.server.make_generation_args(
            {"max_tokens": 1, "temperature": 0},
            total_len=2048,
            chunk=16,
            default_profile="chat",
        )
        args.logit_bias = {}
        result = engine._decode(
            {
                "decode_feed_after_prefill": {},
                "last_logits": np.zeros(10, dtype=np.float32),
                "logical_cache_length": 10,
            },
            [],
            args,
            [],
            False,
            None,
            threading.Event(),
        )
        self.assertEqual(result["generated_text"], "x")
        self.assertEqual(result["decode_run_s"], [])
        self.assertEqual(result["logical_cache_length_after_decode"], 10)

    @staticmethod
    def _admission_engine(server_module, max_waiting=0):
        engine = server_module.QNNEngine.__new__(server_module.QNNEngine)
        engine.args = argparse.Namespace(
            queue_wait_timeout_s=0.1,
            max_waiting_requests=max_waiting,
        )
        engine.state_changed = threading.Condition()
        engine.active_generation = 0
        engine.waiting_requests = 0
        engine.cancel_events = set()
        engine.ready = True
        engine.draining = False
        engine.requests_completed = 0
        engine.requests_failed = 0
        engine.write_ready = lambda: None
        return engine

    def test_bounded_admission_and_disconnect_release(self):
        engine = self._admission_engine(self.server)
        first = threading.Event()
        engine.acquire_generation(first)
        with self.assertRaises(self.server.AdmissionError) as caught:
            engine.acquire_generation(threading.Event())
        self.assertEqual(caught.exception.status, 429)
        first.set()
        engine.release_generation(first, False)
        second = threading.Event()
        engine.acquire_generation(second)
        engine.release_generation(second, True)
        self.assertEqual(engine.requests_completed, 1)

    def test_shutdown_cancels_active_generation_before_idle(self):
        engine = self._admission_engine(self.server)
        active = threading.Event()
        engine.acquire_generation(active)
        engine.begin_shutdown()
        self.assertTrue(active.is_set())
        self.assertFalse(engine.ready)
        self.assertTrue(engine.draining)
        engine.release_generation(active, False)
        self.assertTrue(engine.wait_for_idle(0.1))
        self.assertFalse(self.server.APIServer.daemon_threads)

    def test_shutdown_waiter_exit_notifies_idle_wait(self):
        engine = self._admission_engine(self.server, max_waiting=1)
        engine.args.queue_wait_timeout_s = 2.0
        active = threading.Event()
        engine.acquire_generation(active)
        queued = threading.Event()
        waiter_done = threading.Event()

        def wait_for_generation():
            try:
                engine.acquire_generation(queued)
            except self.server.AdmissionError:
                pass
            finally:
                waiter_done.set()

        waiter = threading.Thread(target=wait_for_generation, daemon=False)
        waiter.start()
        deadline = time.monotonic() + 1.0
        while engine.waiting_requests != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(engine.waiting_requests, 1)

        engine.begin_shutdown()
        engine.release_generation(active, False)
        started = time.monotonic()
        self.assertTrue(engine.wait_for_idle(0.5))
        self.assertLess(time.monotonic() - started, 0.5)
        waiter.join(timeout=1)
        self.assertTrue(waiter_done.is_set())

    def test_request_history_is_bounded_and_body_logging_defaults_off(self):
        args = self.server.make_parser().parse_args([])
        self.assertFalse(args.log_bodies)
        engine = self.server.QNNEngine.__new__(self.server.QNNEngine)
        engine.args = argparse.Namespace(log_bodies=False)
        engine.state_changed = threading.Condition()
        engine.request_records = deque(maxlen=2)
        for index in range(3):
            engine.record_failure(
                {
                    "request_id": str(index),
                    "ids": [1, 2],
                    "gen_args": argparse.Namespace(max_new_tokens=8),
                    "json_mode": False,
                    "tools_ignored": False,
                    "model_prompt": "secret prompt",
                },
                RuntimeError("failure"),
            )
        self.assertEqual(len(engine.request_records), 2)
        self.assertEqual(engine.request_records[0]["request_id"], "1")
        self.assertNotIn("model_prompt", engine.request_records[0])

    def test_stream_validation_error_is_single_json_response(self):
        class RejectingEngine:
            def prepare_request(self, request):
                raise ValueError("validation failed before streaming")

        api = self.server.APIServer(("127.0.0.1", 0), self.server.Handler, RejectingEngine())
        thread = threading.Thread(target=api.serve_forever, daemon=False)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", api.server_port, timeout=2)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"stream": True}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            self.assertEqual(response.status, 400)
            self.assertNotIn("HTTP/1.1", body)
            self.assertEqual(json.loads(body)["error"]["code"], "invalid_request")
            connection.close()
        finally:
            api.shutdown()
            api.server_close()
            thread.join(timeout=2)

    def test_active_stream_shutdown_finishes_handler_and_server(self):
        engine = StreamTestEngine("wait_for_cancel")
        api = self.server.APIServer(("127.0.0.1", 0), self.server.Handler, engine)
        thread = threading.Thread(
            target=lambda: api.serve_forever(poll_interval=0.01),
            daemon=False,
        )
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", api.server_port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"stream": True}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(engine.started.wait(1))

            started = time.monotonic()
            engine.begin_shutdown()
            api.shutdown()
            api.server_close()
            thread.join(timeout=1)
            elapsed = time.monotonic() - started

            self.assertFalse(thread.is_alive())
            self.assertTrue(engine.released.wait(1))
            self.assertLess(elapsed, 1.5)
            self.assertNotIn("HTTP/1.1", response.read().decode("utf-8"))
        finally:
            connection.close()
            if thread.is_alive():
                api.shutdown()
                api.server_close()
                thread.join(timeout=2)

    def test_cancelled_stream_without_terminal_item_ends_consumer(self):
        engine = StreamTestEngine("cancel_without_terminal")
        api = self.server.APIServer(("127.0.0.1", 0), self.server.Handler, engine)
        thread = threading.Thread(target=api.serve_forever, daemon=False)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", api.server_port, timeout=2)
            started = time.monotonic()
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"stream": True}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertTrue(engine.released.wait(1))
            self.assertLess(time.monotonic() - started, 1.5)
            self.assertNotIn("HTTP/1.1", body)
            self.assertNotIn("data: [DONE]", body)
            connection.close()
        finally:
            api.shutdown()
            api.server_close()
            thread.join(timeout=2)

    def test_normal_stream_completes_when_terminal_queue_is_full(self):
        engine = StreamTestEngine("success")
        api = self.server.APIServer(("127.0.0.1", 0), self.server.Handler, engine)
        thread = threading.Thread(target=api.serve_forever, daemon=False)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", api.server_port, timeout=2)
            with patch.object(self.server.queue.Queue, "put_nowait", side_effect=self.server.queue.Full):
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps({"stream": True}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                body = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn('"content":"hello"', body)
            self.assertIn('"finish_reason":"stop"', body)
            self.assertIn("data: [DONE]", body)
            self.assertNotIn("event: error", body)
            self.assertTrue(engine.released.wait(1))
            connection.close()
        finally:
            api.shutdown()
            api.server_close()
            thread.join(timeout=2)

    def test_post_header_generation_error_uses_sse_error_event(self):
        engine = StreamTestEngine("runtime_error")
        api = self.server.APIServer(("127.0.0.1", 0), self.server.Handler, engine)
        thread = threading.Thread(target=api.serve_forever, daemon=False)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", api.server_port, timeout=2)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"stream": True}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn("event: error", body)
            self.assertIn('"code":"generation_error"', body)
            self.assertIn("data: [DONE]", body)
            self.assertNotIn("HTTP/1.1", body)
            self.assertTrue(engine.recorded.wait(1))
            self.assertTrue(engine.released.wait(1))
            connection.close()
        finally:
            api.shutdown()
            api.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
