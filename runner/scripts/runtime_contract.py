#!/usr/bin/env python3
"""Shared QNN runtime identity and strict execution checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


QNN_PROVIDER = "QNNExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
SOC_TARGET = "QCS6490"
HTP_TARGET = "v68"
REQUIRED_DSP_PATHS = ("/usr/lib/dsp/cdsp", "/usr/lib/dsp/adsp", "/dsp")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def merge_adsp_library_path(existing: str | None, required: list[str] | tuple[str, ...]) -> str:
    """Append required ADSP paths while preserving user order and removing duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*(re.split(r";", existing or "")), *required]:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return ";".join(merged)


def configure_qnn_environment(oq: Any) -> str:
    ep_dir = str(Path(oq.get_library_path()).resolve().parent)
    merged = merge_adsp_library_path(
        os.environ.get("ADSP_LIBRARY_PATH"),
        (ep_dir, *REQUIRED_DSP_PATHS),
    )
    os.environ["ADSP_LIBRARY_PATH"] = merged
    return merged


def _file_record(path: str | Path | None) -> dict | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    record = {
        "path": str(candidate.resolve(strict=False)),
        "name": candidate.name,
        "exists": candidate.is_file(),
        "size": candidate.stat().st_size if candidate.is_file() else None,
        "sha256": sha256_file(candidate) if candidate.is_file() else None,
    }
    return record


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip("\x00\r\n ")
    except OSError:
        return None


def _dsp_records(ep_dir: Path) -> dict:
    records = {}
    for kind, name in (
        ("stub", f"libQnnHtpV{HTP_TARGET.removeprefix('v')}Stub.so"),
        ("skel", f"libQnnHtpV{HTP_TARGET.removeprefix('v')}Skel.so"),
    ):
        records[kind] = _file_record(ep_dir / name)
    records["selection_basis"] = f"{SOC_TARGET} targets HTP {HTP_TARGET}"
    return records


def mapped_qnn_libraries() -> list[dict]:
    paths: set[str] = set()
    try:
        for line in Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace").splitlines():
            tail = line.rsplit(maxsplit=1)
            if len(tail) != 2 or not tail[1].startswith("/"):
                continue
            path = tail[1]
            lowered = Path(path).name.lower()
            if "qnn" in lowered or "onnxruntime_providers_qnn" in lowered:
                paths.add(path)
    except OSError:
        return []
    return [_file_record(path) for path in sorted(paths)]


def collect_runtime_fingerprint(
    oq: Any,
    *,
    provider_options: dict | None = None,
    session_config: dict | None = None,
    chunk: int,
    total_length: int,
) -> dict:
    import onnx
    import onnxruntime as ort
    import tokenizers

    adsp_path = configure_qnn_environment(oq)
    ep_library = Path(oq.get_library_path()).resolve()
    htp_library = Path(oq.get_qnn_htp_path()).resolve()
    module_version = getattr(oq, "__version__", None)
    distribution_version = _package_version("onnxruntime-qnn")
    qnn_version = module_version or distribution_version
    fingerprint = {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "executable_resolved": str(Path(sys.executable).resolve()),
        },
        "packages": {
            "onnx": getattr(onnx, "__version__", None),
            "onnxruntime": getattr(ort, "__version__", None),
            "onnxruntime_qnn": qnn_version,
            "onnxruntime_qnn_module_version": module_version,
            "onnxruntime_qnn_distribution_version": distribution_version,
            "tokenizers": getattr(tokenizers, "__version__", None),
            "numpy": getattr(np, "__version__", None),
        },
        "qnn": {
            "provider_name": oq.get_ep_name(),
            "ep_library": _file_record(ep_library),
            "htp_backend_library": _file_record(htp_library),
            "qairt_version": None,
            "qnn_runtime_version": None,
            "version_note": (
                "QAIRT and Qualcomm QNN runtime versions are not exposed by this Python package; "
                "the onnxruntime-qnn package version and absolute library hashes identify the runtime."
            ),
            "dsp": _dsp_records(ep_library.parent),
            "mapped_libraries": mapped_qnn_libraries(),
        },
        "target": {
            "soc": SOC_TARGET,
            "htp_generation": HTP_TARGET,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "device_tree_model": _read_text("/proc/device-tree/model"),
            "soc_id": _read_text("/sys/devices/soc0/soc_id"),
        },
        "execution_contract": {
            "provider_options": {str(key): str(value) for key, value in sorted((provider_options or {}).items())},
            "session_config": {str(key): str(value) for key, value in sorted((session_config or {}).items())},
            "chunk": int(chunk),
            "total_length": int(total_length),
            "fallback_configured_disabled": True,
        },
        "environment": {
            "ADSP_LIBRARY_PATH": adsp_path,
        },
    }
    fingerprint["identity"] = runtime_identity(fingerprint)
    fingerprint["identity_sha256"] = hashlib.sha256(
        json.dumps(fingerprint["identity"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return fingerprint


def runtime_identity(fingerprint: dict) -> dict:
    packages = fingerprint.get("packages") or {}
    qnn = fingerprint.get("qnn") or {}
    dsp = qnn.get("dsp") or {}
    contract = fingerprint.get("execution_contract") or {}
    return {
        "onnxruntime": packages.get("onnxruntime"),
        "onnxruntime_qnn": packages.get("onnxruntime_qnn"),
        "qnn_ep_library_sha256": ((qnn.get("ep_library") or {}).get("sha256")),
        "qnn_htp_library_sha256": ((qnn.get("htp_backend_library") or {}).get("sha256")),
        "htp_stub_sha256": ((dsp.get("stub") or {}).get("sha256")),
        "htp_skel_sha256": ((dsp.get("skel") or {}).get("sha256")),
        "provider_name": qnn.get("provider_name"),
        "provider_options": contract.get("provider_options") or {},
        "session_config": contract.get("session_config") or {},
        "soc": (fingerprint.get("target") or {}).get("soc"),
        "htp_generation": (fingerprint.get("target") or {}).get("htp_generation"),
        "chunk": contract.get("chunk"),
        "total_length": contract.get("total_length"),
    }


def all_outputs_finite(finite_by_output: dict | None) -> bool:
    return bool(finite_by_output) and all(value is True for value in finite_by_output.values())


def qnn_only_from_counts(provider_counts: dict | None) -> bool:
    counts = provider_counts or {}
    return bool(
        int(counts.get(QNN_PROVIDER, 0)) > 0
        and int(counts.get(CPU_PROVIDER, 0)) == 0
    )


def strict_execution_status(
    *,
    session_created: bool,
    graph_executed: bool,
    finite_by_output: dict | None,
    provider_counts: dict | None,
) -> dict:
    status = {
        "session_created": bool(session_created),
        "graph_executed": bool(graph_executed),
        "all_outputs_finite": all_outputs_finite(finite_by_output),
        "qnn_execution_count": int((provider_counts or {}).get(QNN_PROVIDER, 0)),
        "cpu_execution_count": int((provider_counts or {}).get(CPU_PROVIDER, 0)),
        "qnn_only": qnn_only_from_counts(provider_counts),
    }
    status["ok"] = all(
        (
            status["session_created"],
            status["graph_executed"],
            status["all_outputs_finite"],
            status["qnn_only"],
        )
    )
    return status


def strict_epcontext_summary(cases: list[dict]) -> dict:
    by_label = {case.get("label"): case for case in cases}
    generated = by_label.get("generate_external_epcontext") or {}
    loaded = by_label.get("load_external_epcontext") or {}
    summary = {
        "generate_ok": bool(generated.get("ok")),
        "generate_run_ok": bool(generated.get("strict_status", {}).get("ok")),
        "generate_finite": bool(generated.get("strict_status", {}).get("all_outputs_finite")),
        "generate_qnn_only": bool(generated.get("strict_status", {}).get("qnn_only")),
        "load_ok": bool(loaded.get("ok")),
        "load_run_ok": bool(loaded.get("strict_status", {}).get("ok")),
        "load_finite": bool(loaded.get("strict_status", {}).get("all_outputs_finite")),
        "load_qnn_only": bool(loaded.get("strict_status", {}).get("qnn_only")),
        "load_session_create_s": loaded.get("session_create_s"),
        "load_run_s": loaded.get("run_s"),
    }
    summary["ok"] = all(
        summary[key]
        for key in (
            "generate_ok",
            "generate_run_ok",
            "generate_finite",
            "generate_qnn_only",
            "load_ok",
            "load_run_ok",
            "load_finite",
            "load_qnn_only",
        )
    )
    return summary
