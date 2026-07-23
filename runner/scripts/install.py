#!/usr/bin/env python3
"""Install release assets, generate QNN EPContexts, and run a local smoke test."""

from __future__ import annotations

import argparse
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

from runtime_contract import (
    collect_runtime_fingerprint,
    runtime_identity,
    sha256_file,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_DIR = SCRIPT_DIR.parent
DEFAULT_STATE_DIR = (
    Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "lfm2.5-350m-q6a-qcs6490-qnn-npu"
)
DEFAULT_MANIFEST = RUNNER_DIR / "config" / "model-assets.json"
DEFAULT_CANARY = RUNNER_DIR / "config" / "install-canary.json"
DEFAULT_MODEL_REPOSITORY = "PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu"
PINNED_MODEL_REVISION = "773ff42cc383cb61ecf32eb13d1f828634fbd0e1"
DEFAULT_MODEL_BASE_URL = (
    f"https://huggingface.co/{DEFAULT_MODEL_REPOSITORY}/resolve/{PINNED_MODEL_REVISION}"
)
CHUNK_REL = Path("qdq/chunk16_a16w8_qdq.onnx")
DECODE_REL = Path("qdq/decode_a16w8_qdq.onnx")
LOAD_SESSION_CONFIG = {"session.disable_cpu_ep_fallback": "1"}


class InstallError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


def load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError("assets", f"cannot read manifest {path}: {exc}") from exc
    files = manifest.get("files")
    source = manifest.get("huggingface") or {}
    if manifest.get("schema_version") != 2 or not isinstance(files, list) or not files:
        raise InstallError("assets", "manifest must use schema_version 2 and contain files")
    if (
        source.get("repository") != DEFAULT_MODEL_REPOSITORY
        or source.get("pinned_revision") != PINNED_MODEL_REVISION
        or source.get("source_revision") != PINNED_MODEL_REVISION
        or not source.get("manifest_generated_at")
    ):
        raise InstallError("assets", "manifest Hugging Face source metadata is incomplete or unpinned")
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


def install_runtime_fingerprint(chunk: int, total_length: int) -> dict:
    try:
        import onnxruntime_qnn as oq

        return collect_runtime_fingerprint(
            oq,
            provider_options={"backend_path": str(oq.get_qnn_htp_path())},
            session_config=LOAD_SESSION_CONFIG,
            chunk=chunk,
            total_length=total_length,
        )
    except Exception as exc:
        raise InstallError(
            "dependencies",
            "QNN Python environment is incomplete. Required: numpy, onnx, tokenizers, "
            f"onnxruntime and onnxruntime_qnn. Detail: {exc}",
        ) from exc


def dependency_check() -> dict:
    fingerprint = install_runtime_fingerprint(16, 2048)
    packages = fingerprint["packages"]
    qnn = fingerprint["qnn"]
    return {
        "python": fingerprint["python"]["version"],
        "onnx": packages["onnx"],
        "onnxruntime": packages["onnxruntime"],
        "onnxruntime_qnn": packages["onnxruntime_qnn"],
        "qnn_ep_name": qnn["provider_name"],
        "qnn_library": qnn["ep_library"]["path"],
        "qnn_library_sha256": qnn["ep_library"]["sha256"],
        "qnn_htp_library": qnn["htp_backend_library"]["path"],
        "qnn_htp_library_sha256": qnn["htp_backend_library"]["sha256"],
        "tokenizers": packages["tokenizers"],
        "runtime_fingerprint": fingerprint,
    }


def context_stamp_status(context_dir: Path, source_sha: str, expected_fingerprint: dict) -> tuple[bool, str]:
    stamp_path = context_dir / "source-stamp.json"
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "missing_or_corrupt_stamp"
    if stamp.get("schema_version") != 2:
        return False, "unsupported_stamp_schema"
    if stamp.get("source_sha256") != source_sha:
        return False, "source_sha256_changed"
    stamped_fingerprint = stamp.get("runtime_fingerprint")
    if not isinstance(stamped_fingerprint, dict):
        return False, "runtime_fingerprint_missing"
    if runtime_identity(stamped_fingerprint) != runtime_identity(expected_fingerprint):
        return False, "runtime_fingerprint_changed"
    for item in stamp.get("context_files", []):
        try:
            name = str(item["name"])
            if Path(name).name != name:
                return False, "invalid_context_file_record"
            path = context_dir / name
            if not path.is_file() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
                return False, "context_file_changed"
        except (KeyError, OSError):
            return False, "invalid_context_file_record"
    if not stamp.get("context_files"):
        return False, "context_files_missing"
    if not (
        stamp.get("qnn_only")
        and stamp.get("all_outputs_finite")
        and (stamp.get("strict_summary") or {}).get("ok")
    ):
        return False, "strict_execution_not_verified"
    return True, "identity_and_files_match"


def context_stamp_valid(context_dir: Path, source_sha: str, expected_fingerprint: dict) -> bool:
    return context_stamp_status(context_dir, source_sha, expected_fingerprint)[0]


def generate_context(
    label: str,
    source: Path,
    chunk: int,
    state_dir: Path,
    run_dir: Path,
    force: bool,
    runtime_fingerprint: dict,
) -> dict:
    source_sha = sha256_file(source)
    context_dir = state_dir / "contexts" / label
    context_name = f"{label}_epcontext.onnx"
    stamp_valid, stamp_reason = context_stamp_status(context_dir, source_sha, runtime_fingerprint)
    if not force and stamp_valid:
        return {
            "label": label,
            "status": "reused",
            "reuse_reason": stamp_reason,
            "source_sha256": source_sha,
            "qnn_only": True,
            "all_outputs_finite": True,
            "runtime_identity_sha256": runtime_fingerprint["identity_sha256"],
        }
    regeneration_reason = "force_context_requested" if force else stamp_reason

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
    if not summary.get("ok"):
        raise InstallError("epcontext", f"{label} failed strict generation/load checks; see {result_path}")
    load_case = next(
        (case for case in result.get("cases", []) if case.get("label") == "load_external_epcontext"),
        {},
    )
    loaded_fingerprint = load_case.get("runtime_fingerprint")
    if not isinstance(loaded_fingerprint, dict):
        raise InstallError("epcontext", f"{label} load fingerprint is missing; see {result_path}")
    if runtime_identity(loaded_fingerprint) != runtime_identity(runtime_fingerprint):
        raise InstallError("epcontext", f"{label} load fingerprint differs from installer runtime; see {result_path}")
    context_files = []
    for path in sorted(context_dir.iterdir()):
        if path.is_file() and path.name != "source-stamp.json":
            context_files.append(
                {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    stamp = {
        "schema_version": 2,
        "source": str(source),
        "source_sha256": source_sha,
        "qnn_only": True,
        "all_outputs_finite": True,
        "context_files": context_files,
        "probe_result": str(result_path),
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_identity_sha256": runtime_fingerprint["identity_sha256"],
        "strict_summary": summary,
    }
    (context_dir / "source-stamp.json").write_text(
        json.dumps(stamp, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "label": label,
        "status": "generated",
        "regeneration_reason": regeneration_reason,
        "source_sha256": source_sha,
        "qnn_only": True,
        "all_outputs_finite": True,
        "runtime_identity_sha256": runtime_fingerprint["identity_sha256"],
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


def load_canary(path: Path) -> dict:
    try:
        canary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError("smoke", f"cannot read canary definition {path}: {exc}") from exc
    required = (
        "evidence",
        "prompt",
        "expected_subject",
        "expected_first_token_id",
        "seed",
        "temperature",
        "max_tokens",
        "json_max_tokens",
    )
    if canary.get("schema_version") != 1 or not all(key in canary for key in required):
        raise InstallError("smoke", "canary definition is incomplete")
    evidence_rel = Path(str(canary.get("evidence", "")))
    if (
        not str(evidence_rel)
        or evidence_rel.is_absolute()
        or ".." in evidence_rel.parts
    ):
        raise InstallError("smoke", "canary evidence path is missing or unsafe")
    evidence_path = RUNNER_DIR.parent / evidence_rel
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError("smoke", f"cannot read canary evidence {evidence_rel}: {exc}") from exc
    evidence_ids = evidence.get("generated_token_ids") or []
    if not (
        evidence.get("prompt") == canary["prompt"]
        and evidence.get("seed") == canary["seed"]
        and evidence.get("temperature") == canary["temperature"]
        and isinstance(evidence.get("generated_text"), str)
        and str(canary["expected_subject"]).casefold() in evidence["generated_text"].casefold()
        and evidence_ids
        and evidence_ids[0] == canary["expected_first_token_id"]
    ):
        raise InstallError("smoke", "canary definition does not match its verified evidence")
    canary["evidence_sha256"] = sha256_file(evidence_path)
    canary["evidence_verified"] = True
    return canary


def validate_semantic_canary(normal: dict, structured: dict, canary: dict) -> dict:
    expected = str(canary["expected_subject"])
    text = normal["choices"][0]["message"]["content"]
    if not isinstance(text, str) or expected.casefold() not in text.casefold():
        raise InstallError("smoke", f"normal semantic canary expected {expected!r}, got {text!r}")
    normal_metrics = normal.get("qnn_metrics") or {}
    if normal_metrics.get("first_token_id") != int(canary["expected_first_token_id"]):
        raise InstallError(
            "smoke",
            "normal first-token golden mismatch: "
            f"expected {canary['expected_first_token_id']}, got {normal_metrics.get('first_token_id')}",
        )
    if not normal_metrics.get("all_outputs_finite"):
        raise InstallError("smoke", "normal completion contained a non-finite QNN output")

    json_text = structured["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(json_text, parse_constant=lambda value: reject_nonfinite_json(value))
    except (ValueError, json.JSONDecodeError) as exc:
        raise InstallError("smoke", f"JSON mode returned invalid JSON: {exc}") from exc
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"answer"}
        or not isinstance(parsed["answer"], str)
        or parsed["answer"].strip().casefold() != expected.casefold()
    ):
        raise InstallError("smoke", f"JSON semantic canary expected answer={expected!r}, got {parsed!r}")
    structured_metrics = structured.get("qnn_metrics") or {}
    if not structured_metrics.get("all_outputs_finite"):
        raise InstallError("smoke", "JSON completion contained a non-finite QNN output")
    return {
        "normal_text": text,
        "normal_metrics": normal_metrics,
        "json_text": json_text,
        "json_object": parsed,
        "structured_metrics": structured_metrics,
    }


def reject_nonfinite_json(value: str):
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def smoke_server(state_dir: Path, run_dir: Path, port: int, canary_path: Path) -> dict:
    canary = load_canary(canary_path)
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
            if not (
                health
                and health.get("ready")
                and health.get("fallback_configured_disabled")
                and health.get("session_qnn_provider_created")
            ):
                raise InstallError("smoke", f"server did not become ready; see {server_log}")

            prompt = canary["prompt"]
            common = {
                "model": canary.get("model", "lfm2.5-350m-qnn-ctx2048"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": canary["temperature"],
                "seed": canary["seed"],
            }
            normal = http_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                {
                    **common,
                    "max_tokens": canary["max_tokens"],
                },
                timeout=180,
            )

            structured = http_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                {
                    **common,
                    "max_tokens": canary["json_max_tokens"],
                    "response_format": {"type": "json_object"},
                },
                timeout=180,
            )
            canary_result = validate_semantic_canary(normal, structured, canary)
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
    if not server_result.get("qnn_only_verified"):
        raise InstallError("smoke", "server did not mark post-profile QNN-only verification complete")
    return {
        "health": health,
        "canary": canary,
        "canary_definition": str(canary_path),
        "normal_text": canary_result["normal_text"],
        "normal_finish_reason": normal["choices"][0]["finish_reason"],
        "normal_first_token_id": canary_result["normal_metrics"].get("first_token_id"),
        "normal_first_step_top_token_ids": canary_result["normal_metrics"].get("first_step_top_token_ids"),
        "normal_all_outputs_finite": canary_result["normal_metrics"].get("all_outputs_finite"),
        "json_text": canary_result["json_text"],
        "json_object": canary_result["json_object"],
        "json_all_outputs_finite": canary_result["structured_metrics"].get("all_outputs_finite"),
        "semantic_canary_passed": True,
        "qnn_only_by_profile": qnn_only,
        "runtime_fingerprint": server_result.get("runtime_fingerprint"),
        "server_result": str(result_path),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install and verify the LFM2.5-350M Q6A QNN runner.")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--canary", type=Path, default=DEFAULT_CANARY)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument(
        "--model-base-url",
        default=os.environ.get("LFM2_5_MODEL_BASE_URL"),
        help="complete asset download base URL override for a mirror",
    )
    parser.add_argument(
        "--model-repository",
        default=os.environ.get("LFM2_5_MODEL_REPOSITORY", DEFAULT_MODEL_REPOSITORY),
        help="Hugging Face repository used when --model-base-url is not set",
    )
    parser.add_argument(
        "--model-revision",
        default=os.environ.get("LFM2_5_MODEL_REVISION", PINNED_MODEL_REVISION),
        help="immutable Hugging Face revision used when --model-base-url is not set",
    )
    parser.add_argument("--smoke-port", type=int, default=18089)
    parser.add_argument("--force-context", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if isinstance(args.smoke_port, bool) or not 1 <= args.smoke_port <= 65535:
        raise SystemExit("smoke-port must be an integer in [1, 65535]")
    started = time.monotonic()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    state_dir = args.state_dir.expanduser().resolve()
    run_dir = state_dir / "logs" / f"install_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fresh = not (state_dir / "install-result.json").exists()
    result = {
        "schema_version": 2,
        "timestamp": timestamp,
        "state_dir": str(state_dir),
        "fresh_state": fresh,
        "status": "running",
        "stages": {},
        "no_sudo_or_root": True,
    }
    try:
        print("[dependencies] checking QNN Python environment", flush=True)
        dependencies = dependency_check()
        result["stages"]["dependencies"] = dependencies
        print("[assets] validating release assets", flush=True)
        manifest = load_manifest(args.manifest)
        manifest_source = manifest["huggingface"]
        actual_revision = str(args.model_revision)
        actual_base_url = (
            str(args.model_base_url).rstrip("/")
            if args.model_base_url
            else f"https://huggingface.co/{args.model_repository}/resolve/{actual_revision}"
        )
        result["manifest"] = {
            "path": str(args.manifest),
            "model_id": manifest.get("model_id"),
            "license": manifest.get("license"),
            "repository": manifest_source["repository"],
            "pinned_revision": manifest_source["pinned_revision"],
            "source_revision": manifest_source["source_revision"],
            "manifest_generated_at": manifest_source["manifest_generated_at"],
        }
        custom_base_url = bool(args.model_base_url)
        result["asset_source"] = {
            "repository": args.model_repository,
            "revision": None if custom_base_url else actual_revision,
            "requested_revision": actual_revision,
            "base_url": actual_base_url,
            "base_url_override": custom_base_url,
            "revision_applied_to_base_url": not custom_base_url,
            "pinned_public_default": bool(
                not custom_base_url
                and args.model_repository == DEFAULT_MODEL_REPOSITORY
                and actual_revision == PINNED_MODEL_REVISION
            ),
        }
        result["stages"]["assets"] = acquire_assets(
            manifest,
            state_dir,
            args.asset_dir.expanduser().resolve() if args.asset_dir else None,
            actual_base_url,
        )
        chunk_fingerprint = dependencies["runtime_fingerprint"]
        decode_fingerprint = install_runtime_fingerprint(1, 2048)
        print("[epcontext] generating or validating chunk context", flush=True)
        chunk = generate_context(
            "chunk",
            state_dir / "models" / CHUNK_REL,
            16,
            state_dir,
            run_dir,
            args.force_context,
            chunk_fingerprint,
        )
        print("[epcontext] generating or validating decode context", flush=True)
        decode = generate_context(
            "decode",
            state_dir / "models" / DECODE_REL,
            1,
            state_dir,
            run_dir,
            args.force_context,
            decode_fingerprint,
        )
        result["stages"]["epcontext"] = {"chunk": chunk, "decode": decode}
        print("[smoke] starting temporary localhost server", flush=True)
        result["stages"]["smoke"] = smoke_server(
            state_dir,
            run_dir,
            args.smoke_port,
            args.canary.expanduser().resolve(),
        )
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
