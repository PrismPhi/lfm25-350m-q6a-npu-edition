#!/usr/bin/env python3
"""Install release assets, generate QNN EPContexts, and run a local smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_DIR = SCRIPT_DIR.parent
DEFAULT_STATE_DIR = (
    Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "lfm2.5-350m-q6a-qcs6490-qnn-npu"
)
DEFAULT_MANIFEST = RUNNER_DIR / "config" / "model-assets.json"
DEFAULT_MODEL_BASE_URL = (
    "https://huggingface.co/PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu/resolve/main"
)
CHUNK_REL = Path("qdq/chunk16_a16w8_qdq.onnx")
DECODE_REL = Path("qdq/decode_a16w8_qdq.onnx")


class InstallError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError("assets", f"cannot read manifest {path}: {exc}") from exc
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, list) or not files:
        raise InstallError("assets", "manifest must use schema_version 1 and contain files")
    for item in files:
        if not all(item.get(key) for key in ("path", "sha256", "size")):
            raise InstallError("assets", f"invalid manifest entry: {item}")
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise InstallError("assets", f"unsafe asset path: {rel}")
    return manifest


def verify_file(path: Path, item: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(item["size"])
        and sha256_file(path) == str(item["sha256"]).lower()
    )


def acquire_assets(manifest: dict, state_dir: Path, asset_dir: Path | None, base_url: str | None) -> list[dict]:
    model_dir = state_dir / "models"
    records = []
    for item in manifest["files"]:
        rel = Path(item["path"])
        target = model_dir / rel
        if verify_file(target, item):
            records.append({"path": str(rel), "status": "reused", "sha256": item["sha256"]})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        partial.unlink(missing_ok=True)
        if asset_dir is not None:
            source = asset_dir / rel
            if not source.is_file():
                raise InstallError("assets", f"local asset missing: {source}")
            shutil.copyfile(source, partial)
        elif base_url:
            url = base_url.rstrip("/") + "/" + rel.as_posix()
            try:
                with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            except (OSError, urllib.error.URLError) as exc:
                partial.unlink(missing_ok=True)
                raise InstallError("assets", f"download failed for {url}: {exc}") from exc
        else:
            raise InstallError(
                "assets",
                "set LFM2_5_MODEL_BASE_URL, pass --model-base-url, or use --asset-dir for an offline install",
            )
        if not verify_file(partial, item):
            partial.unlink(missing_ok=True)
            raise InstallError("assets", f"size or SHA-256 mismatch for {rel}")
        os.replace(partial, target)
        records.append({"path": str(rel), "status": "installed", "sha256": item["sha256"]})
    return records


def dependency_check() -> dict:
    code = """
import json
import numpy
import onnx
import onnxruntime as ort
import onnxruntime_qnn as oq
from tokenizers import Tokenizer
print(json.dumps({
    "python": __import__("sys").version,
    "onnx": onnx.__version__,
    "onnxruntime": ort.__version__,
    "qnn_ep_name": oq.get_ep_name(),
    "qnn_library": oq.get_library_path(),
    "qnn_htp_library": oq.get_qnn_htp_path(),
    "tokenizers": __import__("tokenizers").__version__,
}))
"""
    proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    if proc.returncode != 0:
        raise InstallError(
            "dependencies",
            "QNN Python environment is incomplete. Required: numpy, onnx, tokenizers, "
            f"onnxruntime and onnxruntime_qnn. Detail: {proc.stderr.strip()}",
        )
    return json.loads(proc.stdout)


def context_stamp_valid(context_dir: Path, source_sha: str) -> bool:
    stamp_path = context_dir / "source-stamp.json"
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if stamp.get("source_sha256") != source_sha:
        return False
    for item in stamp.get("context_files", []):
        path = context_dir / item["name"]
        if not path.is_file() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            return False
    return bool(stamp.get("qnn_only"))


def generate_context(label: str, source: Path, chunk: int, state_dir: Path, run_dir: Path, force: bool) -> dict:
    source_sha = sha256_file(source)
    context_dir = state_dir / "contexts" / label
    context_name = f"{label}_epcontext.onnx"
    if not force and context_stamp_valid(context_dir, source_sha):
        return {"label": label, "status": "reused", "source_sha256": source_sha, "qnn_only": True}

    log_dir = run_dir / f"epcontext_{label}"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "generate_epcontext.py"),
        "--timestamp",
        f"install_{label}",
        "--model",
        str(source),
        "--log-dir",
        str(log_dir),
        "--context-dir",
        str(context_dir),
        "--context-name",
        context_name,
        "--chunk",
        str(chunk),
        "--total-len",
        "2048",
        "--skip-cold-baseline",
    ]
    output_path = run_dir / f"epcontext_{label}.log"
    env = dict(
        os.environ,
        LFM2_5_STATE_DIR=str(state_dir),
    )
    with output_path.open("w", encoding="utf-8") as output:
        proc = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise InstallError("epcontext", f"{label} context generator exited with {proc.returncode}; see {output_path}")
    result_path = log_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = result.get("summary") or {}
    if not (summary.get("generate_ok") and summary.get("load_ok") and summary.get("load_qnn_only")):
        raise InstallError("epcontext", f"{label} failed QNN-only generation/load; see {result_path}")
    context_files = []
    for path in sorted(context_dir.iterdir()):
        if path.is_file() and path.name != "source-stamp.json":
            context_files.append(
                {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    stamp = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": source_sha,
        "qnn_only": True,
        "context_files": context_files,
        "probe_result": str(result_path),
    }
    (context_dir / "source-stamp.json").write_text(
        json.dumps(stamp, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "label": label,
        "status": "generated",
        "source_sha256": source_sha,
        "qnn_only": True,
        "load_run_s": summary.get("load_run_s"),
        "load_session_create_s": summary.get("load_session_create_s"),
    }


def http_json(url: str, body: dict | None = None, timeout: float = 15.0) -> dict:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def smoke_server(state_dir: Path, run_dir: Path, port: int) -> dict:
    timestamp = "install_smoke"
    server_log = run_dir / "server-smoke.log"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "server.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--timestamp",
        timestamp,
        "--log-root",
        str(run_dir),
        "--chunk-context",
        str(state_dir / "contexts/chunk/chunk_epcontext.onnx"),
        "--decode-context",
        str(state_dir / "contexts/decode/decode_epcontext.onnx"),
        "--tokenizer",
        str(state_dir / "models/tokenizer/tokenizer.json"),
        "--rope-cache",
        str(state_dir / "models/host/rope_cache.npz"),
        "--embedding-int8-dir",
        str(state_dir / "models/host/embedding_int8_rowwise"),
        "--v0-runner-dir",
        str(SCRIPT_DIR),
        "--chunk",
        "16",
        "--total-len",
        "2048",
    ]
    env = dict(
        os.environ,
        LFM2_5_STATE_DIR=str(state_dir),
    )
    with server_log.open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        try:
            health = None
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise InstallError("smoke", f"server exited early with {proc.returncode}; see {server_log}")
                try:
                    health = http_json(f"http://127.0.0.1:{port}/health", timeout=2)
                    if health.get("ready"):
                        break
                except (OSError, urllib.error.URLError, json.JSONDecodeError):
                    time.sleep(0.5)
            if not health or not health.get("ready") or not health.get("fallback_disabled"):
                raise InstallError("smoke", f"server did not become ready; see {server_log}")

            prompt = (
                "Please answer in one short sentence. Use the model normally and do not add "
                "extra commentary. The capital of Japan is"
            )
            normal = http_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                {
                    "model": "lfm2.5-350m-qnn-ctx2048",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 24,
                    "temperature": 0,
                },
                timeout=180,
            )
            text = normal["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise InstallError("smoke", "normal completion was empty")

            structured = http_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                {
                    "model": "lfm2.5-350m-qnn-ctx2048",
                    "messages": [{"role": "user", "content": "Return the capital of Japan as JSON."}],
                    "max_tokens": 32,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                timeout=180,
            )
            json_text = structured["choices"][0]["message"]["content"]
            parsed = json.loads(json_text)
            if not isinstance(parsed, dict):
                raise InstallError("smoke", "JSON mode did not return an object")
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)

    result_dir = run_dir / f"openai_server_{timestamp}"
    result_path = result_dir / "server_result.json"
    if not result_path.is_file():
        raise InstallError("smoke", f"server result missing: {result_path}")
    server_result = json.loads(result_path.read_text(encoding="utf-8"))
    qnn_only = server_result.get("qnn_only_by_profile") or {}
    if not (qnn_only.get("chunk") and qnn_only.get("decode")):
        raise InstallError("smoke", f"provider profile was not QNN-only: {qnn_only}")
    return {
        "health": health,
        "normal_text": text,
        "normal_finish_reason": normal["choices"][0]["finish_reason"],
        "json_text": json_text,
        "json_object": parsed,
        "qnn_only_by_profile": qnn_only,
        "server_result": str(result_path),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install and verify the LFM2.5-350M Q6A QNN runner.")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument(
        "--model-base-url",
        default=(
            os.environ.get("LFM2_5_MODEL_BASE_URL")
            or DEFAULT_MODEL_BASE_URL
        ),
        help="asset download base URL; defaults to the public Hugging Face model repository",
    )
    parser.add_argument("--smoke-port", type=int, default=18089)
    parser.add_argument("--force-context", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    started = time.monotonic()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    state_dir = args.state_dir.expanduser().resolve()
    run_dir = state_dir / "logs" / f"install_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fresh = not (state_dir / "install-result.json").exists()
    result = {
        "schema_version": 1,
        "timestamp": timestamp,
        "state_dir": str(state_dir),
        "fresh_state": fresh,
        "status": "running",
        "stages": {},
        "no_sudo_or_root": True,
    }
    try:
        print("[dependencies] checking QNN Python environment", flush=True)
        result["stages"]["dependencies"] = dependency_check()
        print("[assets] validating release assets", flush=True)
        manifest = load_manifest(args.manifest)
        result["manifest"] = {
            "path": str(args.manifest),
            "model_id": manifest.get("model_id"),
            "license": manifest.get("license"),
        }
        result["stages"]["assets"] = acquire_assets(
            manifest,
            state_dir,
            args.asset_dir.expanduser().resolve() if args.asset_dir else None,
            args.model_base_url,
        )
        print("[epcontext] generating or validating chunk context", flush=True)
        chunk = generate_context(
            "chunk",
            state_dir / "models" / CHUNK_REL,
            16,
            state_dir,
            run_dir,
            args.force_context,
        )
        print("[epcontext] generating or validating decode context", flush=True)
        decode = generate_context(
            "decode",
            state_dir / "models" / DECODE_REL,
            1,
            state_dir,
            run_dir,
            args.force_context,
        )
        result["stages"]["epcontext"] = {"chunk": chunk, "decode": decode}
        print("[smoke] starting temporary localhost server", flush=True)
        result["stages"]["smoke"] = smoke_server(state_dir, run_dir, args.smoke_port)
        result["status"] = "ok"
        result["elapsed_s"] = time.monotonic() - started
        result["start_command"] = (
            f"LFM2_5_STATE_DIR='{state_dir}' LFM2_5_PYTHON='{sys.executable}' "
            f"'{RUNNER_DIR / 'start_server.sh'}'"
        )
        print(f"[complete] chat-ready in {result['elapsed_s']:.1f} seconds", flush=True)
        print(f"[complete] start with: {result['start_command']}", flush=True)
        return_code = 0
    except Exception as exc:
        result["status"] = "error"
        result["elapsed_s"] = time.monotonic() - started
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        stage = exc.stage if isinstance(exc, InstallError) else "unexpected"
        result["failed_stage"] = stage
        print(str(exc), file=sys.stderr, flush=True)
        return_code = 1
    finally:
        output = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        (run_dir / "install-result.json").write_text(output, encoding="utf-8")
        (state_dir / "install-result.json").write_text(output, encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
