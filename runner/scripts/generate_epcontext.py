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

import numpy as np
import onnxruntime as ort

from probe_p4_patha2_full_chunk_graph import build_feeds
from probe_p4_patha2_chunk_mlp import summarize_model
from probe_p4_patha2_chunk_l2_attention import provider_counts_from_profile, qnn_register
from runtime_contract import (
    collect_runtime_fingerprint,
    strict_epcontext_summary,
    strict_execution_status,
)


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ROOT = Path(
    os.environ.get("LFM2_5_STATE_DIR")
    or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "lfm2.5-350m-q6a-qcs6490-qnn-npu"
)
DEFAULT_MODEL = STATE_ROOT / "models" / "qdq" / "chunk16_a16w8_qdq.onnx"
DEFAULT_RUN_ROOT = STATE_ROOT / "logs"
DEFAULT_MASK_VALUE = -64.0
DEFAULT_ROPE_THETA = 1_000_000.0


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


def generated_files(root: Path) -> list[dict]:
    files = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append({"path": str(path), "size": path.stat().st_size})
    return files


def read_number(path: str):
    try:
        return float(Path(path).read_text(encoding="utf-8").strip())
    except Exception:
        return None


def power_snapshot() -> dict:
    patterns = [
        "/sys/class/power_supply/*/power_now",
        "/sys/class/power_supply/*/current_now",
        "/sys/class/power_supply/*/voltage_now",
        "/sys/class/hwmon/hwmon*/power*_input",
    ]
    readings = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            value = read_number(path)
            if value is not None:
                readings[path] = value
    return {
        "readings": readings,
        "direct_power_available": any("power" in Path(path).name for path in readings),
        "note": "Raw world-readable sysfs only; no sudo/root and no external power meter.",
    }


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


def make_session(model_path: Path, log_dir: Path, label: str, session_configs=None, provider_options_extra=None):
    register_status, qdevs, provider_options = qnn_register()
    if not qdevs:
        raise RuntimeError("No QNN OrtEpDevice after provider registration")
    provider_options.update({str(k): str(v) for k, v in (provider_options_extra or {}).items()})

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_profiling = True
    so.profile_file_prefix = str(log_dir / f"{label}_profile")
    effective_session_configs = {"session.disable_cpu_ep_fallback": "1"}
    effective_session_configs.update(
        {str(key): str(value) for key, value in (session_configs or {}).items()}
    )
    for key, value in effective_session_configs.items():
        so.add_session_config_entry(str(key), str(value))
    so.add_provider_for_devices(qdevs, provider_options)

    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model_path), sess_options=so, enable_fallback=False)
    create_s = time.perf_counter() - t0
    return sess, {
        "register_status": register_status,
        "qnn_device_count": len(qdevs),
        "provider_options": provider_options,
        "session_configs": effective_session_configs,
        "session_create_s": create_s,
        "session_providers": list(sess.get_providers()),
        "fallback_configured_disabled": True,
        "session_qnn_provider_created": "QNNExecutionProvider" in sess.get_providers(),
    }


def run_once(sess, args) -> tuple[float, dict, dict]:
    feeds = build_feeds(sess, args.chunk, args.total_len - args.chunk, args.total_len, args.seed, args.mask_value, args.rope_theta)
    names = [item.name for item in sess.get_outputs()]
    t0 = time.perf_counter()
    outputs = sess.run(None, feeds)
    run_s = time.perf_counter() - t0
    shape_map = {name: list(np.asarray(value).shape) for name, value in zip(names, outputs)}
    finite = {name: bool(np.isfinite(np.asarray(value)).all()) for name, value in zip(names, outputs)}
    return run_s, shape_map, finite


def run_case(label: str, model_path: Path, log_dir: Path, args, session_configs=None, provider_options_extra=None, run_graph=True) -> dict:
    case = {
        "label": label,
        "model": str(model_path),
        "ok": False,
        "rss_before_mb": rss_mb(),
        "proc_status_before": proc_status(),
        "session_configs": dict(session_configs or {}),
        "provider_options_extra": dict(provider_options_extra or {}),
    }
    sess = None
    profile = None
    try:
        sess, info = make_session(model_path, log_dir, label, session_configs, provider_options_extra)
        case.update(info)
        graph_executed = False
        finite = None
        if run_graph:
            run_s, shape_map, finite = run_once(sess, args)
            graph_executed = True
            case["run_s"] = run_s
            case["prefill_tok_s"] = args.chunk / run_s if run_s else None
            case["output_shapes"] = shape_map
            case["all_outputs_finite"] = finite
        profile = sess.end_profiling()
        counts = provider_counts_from_profile(profile)
        case["profile"] = str(profile)
        case["provider_counts"] = counts
        case["qnn_only"] = bool(counts.get("QNNExecutionProvider", 0) > 0 and counts.get("CPUExecutionProvider", 0) == 0)
        case["strict_status"] = strict_execution_status(
            session_created=True,
            graph_executed=graph_executed,
            finite_by_output=finite,
            provider_counts=counts,
        )
        import onnxruntime_qnn as oq

        case["runtime_fingerprint"] = collect_runtime_fingerprint(
            oq,
            provider_options=case["provider_options"],
            session_config=case["session_configs"],
            chunk=args.chunk,
            total_length=args.total_len,
        )
        case["ok"] = bool(case["strict_status"]["ok"])
    except Exception as exc:
        case["error_type"] = type(exc).__name__
        case["error"] = str(exc)
        case["traceback"] = traceback.format_exc()
    finally:
        try:
            if sess is not None and profile is None:
                profile = sess.end_profiling()
                case["profile"] = str(profile)
        except Exception:
            pass
        del sess
        gc.collect()
        case["rss_after_mb"] = rss_mb()
        case["proc_status_after"] = proc_status()
    return case


def write_summary(log_dir: Path, result: dict) -> None:
    lines = [
        "# P4 Full Chunk EPContext Probe",
        "",
        f"- source_model: `{result.get('source_model')}`",
        f"- context_model: `{result.get('context_model')}`",
        f"- chunk: `{result.get('chunk')}`",
        f"- total_len: `{result.get('total_len')}`",
        f"- v1_complete: `{result.get('v1_complete')}`",
        "",
        "## Cases",
        "",
        "| case | ok | session create s | run s | prefill tok/s | qnn-only | provider counts | error |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for case in result.get("cases", []):
        error = (case.get("error") or "").replace("\n", " ")[:160]
        lines.append(
            f"| {case.get('label')} | {case.get('ok')} | {case.get('session_create_s')} | "
            f"{case.get('run_s')} | {case.get('prefill_tok_s')} | {case.get('qnn_only')} | "
            f"`{case.get('provider_counts')}` | `{error}` |"
        )
    lines.extend(["", "## Generated Files", ""])
    for item in result.get("generated_files_final", []):
        lines.append(f"- `{item['path']}` ({item['size']} bytes)")
    lines.extend([
        "",
        "This is a chunk graph EPContext and warm-load probe. It is not yet the prompt-to-text runner integration.",
        "",
    ])
    (log_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default=time.strftime("%Y%m%d_%H%M%S_full_chunk_epcontext"))
    parser.add_argument("--log-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--context-dir", type=Path)
    parser.add_argument("--context-name", default="patha2_full_chunk16_total128_logits_epcontext_external.onnx")
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--total-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--mask-value", type=float, default=DEFAULT_MASK_VALUE)
    parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
    parser.add_argument("--skip-cold-baseline", action="store_true")
    parser.add_argument("--skip-generate-run", action="store_true")
    args = parser.parse_args()

    log_dir = args.log_dir or (args.log_root / f"p4_full_chunk_epcontext_{args.timestamp}")
    context_dir = args.context_dir or (log_dir / "epcontext")
    context_model = context_dir / args.context_name
    log_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    for path in context_dir.glob("*"):
        if path.is_file():
            path.unlink()

    result = {
        "timestamp": args.timestamp,
        "mode": "p4_full_chunk_epcontext_probe",
        "source_model": str(args.model),
        "source_model_summary": summarize_model(args.model),
        "log_dir": str(log_dir),
        "context_dir": str(context_dir),
        "context_model": str(context_model),
        "chunk": args.chunk,
        "total_len": args.total_len,
        "v1_complete": False,
        "no_sudo_or_root": True,
        "power_before": power_snapshot(),
        "thermal_before": thermal_snapshot(),
        "cases": [],
    }
    try:
        if not args.skip_cold_baseline:
            result["cases"].append(run_case("baseline_cold_qnn", args.model, log_dir, args))
        gen_configs = {
            "ep.context_enable": "1",
            "ep.context_file_path": str(context_model),
            "ep.context_embed_mode": "0",
        }
        result["cases"].append(run_case("generate_external_epcontext", args.model, log_dir, args, gen_configs, run_graph=not args.skip_generate_run))
        result["generated_files_after_context_gen"] = generated_files(context_dir)
        if context_model.exists():
            result["cases"].append(run_case("load_external_epcontext", context_model, log_dir, args))
        else:
            result["cases"].append({
                "label": "load_external_epcontext",
                "ok": False,
                "error_type": "MissingContextModel",
                "error": f"{context_model} was not generated",
            })
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["generated_files_final"] = generated_files(context_dir)
        result["power_after"] = power_snapshot()
        result["thermal_after"] = thermal_snapshot()
        result["summary"] = strict_epcontext_summary(result["cases"])
        (log_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        write_summary(log_dir, result)
    success = "error" not in result and bool((result.get("summary") or {}).get("ok"))
    print(json.dumps({"ok": success, "log_dir": str(log_dir), "summary": str(log_dir / "summary.md")}, ensure_ascii=False), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
