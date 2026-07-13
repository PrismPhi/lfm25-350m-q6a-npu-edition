from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch


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


class PublicRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = load_module("public_server", "server.py")
        cls.install = load_module("public_install", "install.py")

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
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["files"]), 11)
        self.assertEqual(len({item["path"] for item in manifest["files"]}), 11)

    def test_public_model_repository_is_installer_default(self):
        with patch.dict(os.environ, {}, clear=True):
            args = self.install.make_parser().parse_args([])
        self.assertEqual(args.model_base_url, self.install.DEFAULT_MODEL_BASE_URL)
        self.assertEqual(
            args.model_base_url,
            "https://huggingface.co/PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu/resolve/main",
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


if __name__ == "__main__":
    unittest.main()
