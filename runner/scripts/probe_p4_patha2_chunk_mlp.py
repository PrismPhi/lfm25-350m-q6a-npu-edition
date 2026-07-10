#!/usr/bin/env python3
import argparse
import copy
import gc
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


try:
    import resource
except ImportError:  # Windows local build/CPU parity path.
    resource = None


RECORD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCEPTED_MODEL = (
    RECORD_ROOT
    / "artifacts"
    / "p4_source_model_extract"
    / "phase4_n4b_qdq_qnn_20260705_024907_allmul_nosigmoid_pc"
    / "models"
    / "patha2_n4b_allmul_nosigmoid_pc_qdq.onnx"
)
DEFAULT_LOG_ROOT = RECORD_ROOT / "logs"
HIDDEN = 1024
INTERMEDIATE = 4608


def rss_mb():
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def tensor_shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


class InitCopier:
    def __init__(self, accepted_model: Path):
        self.model = onnx.load(str(accepted_model), load_external_data=False)
        self.src = {init.name: init for init in self.model.graph.initializer}
        self.added = set()
        self.inits = []

    def copy(self, name: str, new_name: str | None = None):
        if name not in self.src:
            raise KeyError(f"initializer not found: {name}")
        out_name = new_name or name
        if out_name in self.added:
            return out_name
        item = copy.deepcopy(self.src[name])
        item.name = out_name
        self.inits.append(item)
        self.added.add(out_name)
        return out_name

    def copy_many(self, names):
        for name in names:
            self.copy(name)

    def copy_chunk_eps_pad(self, base: str, chunk: int):
        quant_name = f"{base}_quantized"
        scale_name = f"{base}_scale"
        zp_name = f"{base}_zero_point"
        arr = numpy_helper.to_array(self.src[quant_name])
        if list(arr.shape) != [1, 1, 1]:
            raise ValueError(f"expected eps pad [1,1,1], got {arr.shape} for {quant_name}")
        tiled = np.tile(arr, (1, chunk, 1))
        out_quant = f"{base}_chunk{chunk}_quantized"
        if out_quant not in self.added:
            self.inits.append(numpy_helper.from_array(tiled, name=out_quant))
            self.added.add(out_quant)
        self.copy(scale_name)
        self.copy(zp_name)
        return out_quant, scale_name, zp_name


def qdq(nodes, copier: InitCopier, value_name: str, scale_base: str, out_name: str):
    scale = copier.copy(f"{scale_base}_scale")
    zp = copier.copy(f"{scale_base}_zero_point")
    q = f"{out_name}_QuantizeLinear_Output"
    nodes.append(helper.make_node("QuantizeLinear", [value_name, scale, zp], [q], f"{out_name}_QuantizeLinear"))
    nodes.append(helper.make_node("DequantizeLinear", [q, scale, zp], [out_name], f"{out_name}_DequantizeLinear"))
    return out_name


def dq_const(nodes, copier: InitCopier, base: str, out_name: str | None = None):
    quant = copier.copy(f"{base}_quantized")
    scale = copier.copy(f"{base}_scale")
    zp = copier.copy(f"{base}_zero_point")
    y = out_name or f"{base}_DequantizeLinear_Output"
    nodes.append(helper.make_node("DequantizeLinear", [quant, scale, zp], [y], f"{base}_DequantizeLinear"))
    return y


def dq_chunk_eps(nodes, copier: InitCopier, base: str, chunk: int, out_name: str):
    quant, scale, zp = copier.copy_chunk_eps_pad(base, chunk)
    nodes.append(helper.make_node("DequantizeLinear", [quant, scale, zp], [out_name], f"{base}_chunk{chunk}_DequantizeLinear"))
    return out_name


def add_exact_lpnorm(nodes, copier: InitCopier, x_name: str, norm_prefix: str, chunk: int):
    eps = dq_chunk_eps(nodes, copier, f"{norm_prefix}_exact_lpnorm_eps_pad", chunk, f"{norm_prefix}_eps_pad_dq")
    scale_weight = dq_const(nodes, copier, f"{norm_prefix}_exact_lpnorm_scale_weight")
    nodes.append(helper.make_node("Concat", [x_name, eps], [f"{norm_prefix}_exact_lpnorm_concat"], f"{norm_prefix}/ExactLpNormConcatEps", axis=2))
    concat = qdq(nodes, copier, f"{norm_prefix}_exact_lpnorm_concat", f"{norm_prefix}_exact_lpnorm_concat", f"{norm_prefix}_exact_lpnorm_concat_dq")
    nodes.append(helper.make_node("LpNormalization", [concat], [f"{norm_prefix}_exact_lpnorm_unit_ext"], f"{norm_prefix}/ExactLpNormalization", axis=2, p=2))
    unit_ext = qdq(nodes, copier, f"{norm_prefix}_exact_lpnorm_unit_ext", f"{norm_prefix}_exact_lpnorm_unit_ext", f"{norm_prefix}_exact_lpnorm_unit_ext_dq")
    copier.copy_many(
        [
            f"{norm_prefix}_exact_lpnorm_slice_starts",
            f"{norm_prefix}_exact_lpnorm_slice_ends",
            f"{norm_prefix}_exact_lpnorm_slice_axes",
            f"{norm_prefix}_exact_lpnorm_slice_steps",
        ]
    )
    nodes.append(
        helper.make_node(
            "Slice",
            [
                unit_ext,
                f"{norm_prefix}_exact_lpnorm_slice_starts",
                f"{norm_prefix}_exact_lpnorm_slice_ends",
                f"{norm_prefix}_exact_lpnorm_slice_axes",
                f"{norm_prefix}_exact_lpnorm_slice_steps",
            ],
            [f"{norm_prefix}_exact_lpnorm_unit"],
            f"{norm_prefix}/ExactLpNormSliceOriginal",
        )
    )
    unit = qdq(nodes, copier, f"{norm_prefix}_exact_lpnorm_unit", f"{norm_prefix}_exact_lpnorm_unit_ext", f"{norm_prefix}_exact_lpnorm_unit_dq")
    nodes.append(helper.make_node("Mul", [unit, scale_weight], [f"{norm_prefix}_weighted"], f"{norm_prefix}/ExactLpNormScaleWeightMul"))
    return qdq(nodes, copier, f"{norm_prefix}_weighted", f"{norm_prefix}_weighted", f"{norm_prefix}_weighted_dq")


def add_linear(nodes, copier: InitCopier, x_name: str, weight_base: str, out_base: str):
    weight = dq_const(nodes, copier, weight_base)
    nodes.append(helper.make_node("MatMul", [x_name, weight], [f"{out_base}_mm"], f"{out_base}/MatMul"))
    return qdq(nodes, copier, f"{out_base}_mm", f"{out_base}_mm", f"{out_base}_mm_dq")


def build_l0_mlp_model(accepted_model: Path, output_model: Path, chunk: int):
    copier = InitCopier(accepted_model)
    nodes = []
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, chunk, HIDDEN])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, chunk, HIDDEN])]

    x = qdq(nodes, copier, "x", "l0_resid_y_raw", "x_resid_dq")
    norm = add_exact_lpnorm(nodes, copier, x, "l0_mlp_ffn_norm", chunk)
    gate = add_linear(nodes, copier, norm, "l0_mlp_gate_w_dq", "l0_mlp_gate")
    up = add_linear(nodes, copier, norm, "l0_mlp_up_w_dq", "l0_mlp_up")
    nodes.append(helper.make_node("Sigmoid", [gate], ["l0_mlp_gate_sig_raw"], "l0_mlp/SiluSigmoid"))
    sig = qdq(nodes, copier, "l0_mlp_gate_sig_raw", "l0_mlp_gate_sig_raw", "l0_mlp_gate_sig_raw_dq")
    nodes.append(helper.make_node("Mul", [gate, sig], ["l0_mlp_gate_act_raw"], "l0_mlp/SiluMul"))
    act = qdq(nodes, copier, "l0_mlp_gate_act_raw", "l0_mlp_gate_act_raw", "l0_mlp_gate_act_raw_dq")
    nodes.append(helper.make_node("Mul", [act, up], ["l0_mlp_ffn_mul_raw"], "l0_mlp/GateUpMul"))
    ffn = qdq(nodes, copier, "l0_mlp_ffn_mul_raw", "l0_mlp_ffn_mul_raw", "l0_mlp_ffn_mul_raw_dq")
    down = add_linear(nodes, copier, ffn, "l0_mlp_down_proj_w_dq", "l0_mlp_down_proj")
    nodes.append(helper.make_node("Add", [x, down], ["l0_mlp_y_raw"], "l0_mlp/ResidualAdd"))
    y = qdq(nodes, copier, "l0_mlp_y_raw", "l0_mlp_y_raw", "y")
    outputs[0].name = y

    graph = helper.make_graph(nodes, output_model.stem, inputs, outputs, copier.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))
    return summarize_model(output_model)


def summarize_model(path: Path):
    model = onnx.load(str(path), load_external_data=False)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "node_counts": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "initializer_count": len(model.graph.initializer),
        "inputs": [{"name": item.name, "shape": tensor_shape(item)} for item in model.graph.input],
        "outputs": [{"name": item.name, "shape": tensor_shape(item)} for item in model.graph.output],
    }


def run_cpu_parity(chunk_model: Path, one_model: Path, chunk: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.5, size=(1, chunk, HIDDEN)).astype(np.float32)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    t0 = time.perf_counter()
    chunk_sess = ort.InferenceSession(str(chunk_model), sess_options=so, providers=["CPUExecutionProvider"])
    chunk_create_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    one_sess = ort.InferenceSession(str(one_model), sess_options=so, providers=["CPUExecutionProvider"])
    one_create_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    chunk_y = chunk_sess.run(None, {"x": x})[0]
    chunk_run_s = time.perf_counter() - t2
    pieces = []
    t3 = time.perf_counter()
    for idx in range(chunk):
        pieces.append(one_sess.run(None, {"x": x[:, idx : idx + 1, :]})[0])
    one_loop_run_s = time.perf_counter() - t3
    ref = np.concatenate(pieces, axis=1)
    diff = chunk_y.astype(np.float64) - ref.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    denom = float(np.linalg.norm(chunk_y.reshape(-1)) * np.linalg.norm(ref.reshape(-1)))
    cosine = float(np.dot(chunk_y.reshape(-1), ref.reshape(-1)) / denom) if denom else None
    exact_equal = bool(np.array_equal(chunk_y, ref))
    # Batched MatMul/LpNormalization can differ slightly from a Python loop over
    # one-token sessions due to reduction/order choices. Keep this below QDQ
    # quantization noise and record the exact diff.
    return {
        "ok": bool(max_abs <= 1e-5 and cosine is not None and cosine >= 0.999999),
        "chunk_create_s": chunk_create_s,
        "one_create_s": one_create_s,
        "chunk_run_s": chunk_run_s,
        "one_loop_run_s": one_loop_run_s,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
        "exact_equal": exact_equal,
        "chunk_output_shape": list(chunk_y.shape),
        "reference_shape": list(ref.shape),
    }


def qnn_register():
    import onnxruntime_qnn as oq

    os.environ["ADSP_LIBRARY_PATH"] = f"{Path(oq.get_library_path()).parent};/usr/lib/dsp/cdsp;/usr/lib/dsp/adsp;/dsp"
    try:
        ort.register_execution_provider_library(oq.get_ep_name(), oq.get_library_path())
        register_status = "registered"
    except Exception as exc:
        register_status = f"register_exception:{exc!r}"
    qdevs = [d for d in ort.get_ep_devices() if getattr(d, "ep_name", None) == oq.get_ep_name()]
    provider_options = {"backend_path": str(oq.get_qnn_htp_path())}
    return oq, register_status, qdevs, provider_options


def provider_counts_from_profile(profile_path):
    counts = Counter()
    try:
        data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    for event in data:
        args = event.get("args") or {}
        provider = args.get("provider")
        if provider:
            counts[str(provider)] += 1
    return dict(sorted(counts.items()))


def run_qnn(model_path: Path, log_dir: Path, chunk: int, seed: int):
    case = {"ok": False, "model": str(model_path), "rss_before_mb": rss_mb()}
    sess = None
    try:
        oq, register_status, qdevs, provider_options = qnn_register()
        case["register_status"] = register_status
        case["qnn_devices"] = [getattr(d, "ep_name", repr(d)) for d in qdevs]
        if not qdevs:
            raise RuntimeError("No QNN OrtEpDevice after provider registration")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_profiling = True
        so.profile_file_prefix = str(log_dir / "qnn_l0_mlp_chunk_profile")
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_provider_for_devices(qdevs, provider_options)
        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(model_path), sess_options=so, enable_fallback=False)
        case["session_create_s"] = time.perf_counter() - t0
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 0.5, size=(1, chunk, HIDDEN)).astype(np.float32)
        t1 = time.perf_counter()
        outputs = sess.run(None, {"x": x})
        case["run_s"] = time.perf_counter() - t1
        case["output_shapes"] = [list(np.asarray(item).shape) for item in outputs]
        profile = sess.end_profiling()
        case["profile"] = str(profile)
        counts = provider_counts_from_profile(profile)
        case["provider_counts"] = counts
        case["qnn_only"] = bool(counts.get("QNNExecutionProvider", 0) > 0 and counts.get("CPUExecutionProvider", 0) == 0)
        case["ok"] = bool(case["qnn_only"])
    except Exception as exc:
        case["error_type"] = type(exc).__name__
        case["error"] = str(exc)
        case["traceback"] = traceback.format_exc()
        try:
            if sess is not None:
                profile = sess.end_profiling()
                case["profile"] = str(profile)
                case["provider_counts"] = provider_counts_from_profile(profile)
        except Exception:
            pass
    finally:
        del sess
        gc.collect()
        case["rss_after_mb"] = rss_mb()
    return case


def write_summary(log_dir: Path, result: dict):
    lines = [
        f"# P4 Path A2 Chunk MLP Probe {result['timestamp']}",
        "",
        "This probe builds a vectorized chunk MLP minigraph from the accepted V0 final QDQ constants.",
        "It is a P4 build component, not a V1 completion claim and not a runner architecture.",
        "",
        "## Contract",
        "",
        f"- chunk: `{result['chunk']}`",
        f"- layer: `0`",
        f"- accepted_model: `{result.get('accepted_model')}`",
        f"- chunk_model: `{(result.get('chunk_model') or {}).get('path')}`",
        f"- one_token_model: `{(result.get('one_token_model') or {}).get('path')}`",
        "- copied constants: accepted quantized weights, scales, zero-points, ExactLpNorm scale weights.",
        "- generated constants: eps pad tiled from accepted `[1,1,1]` to `[1,chunk,1]` using the same quantized value/scale/zero-point.",
        "",
        "## CPU Vectorization Parity",
        "",
    ]
    cpu = result.get("cpu_parity")
    if cpu:
        lines += [
            f"- ok: `{cpu.get('ok')}`",
            f"- max_abs: `{cpu.get('max_abs')}`",
            f"- mean_abs: `{cpu.get('mean_abs')}`",
            f"- cosine: `{cpu.get('cosine')}`",
            f"- exact_equal: `{cpu.get('exact_equal')}`",
            f"- chunk_run_s: `{cpu.get('chunk_run_s')}`",
            f"- one_loop_run_s: `{cpu.get('one_loop_run_s')}`",
        ]
    else:
        lines.append("- not run")
    qnn = result.get("qnn_run")
    if qnn:
        lines += [
            "",
            "## QNN Run",
            "",
            f"- ok: `{qnn.get('ok')}`",
            f"- qnn_only: `{qnn.get('qnn_only')}`",
            f"- session_create_s: `{qnn.get('session_create_s')}`",
            f"- run_s: `{qnn.get('run_s')}`",
            f"- provider_counts: `{qnn.get('provider_counts')}`",
            f"- output_shapes: `{qnn.get('output_shapes')}`",
        ]
        if qnn.get("error"):
            lines += [f"- error_type: `{qnn.get('error_type')}`", f"- error: `{qnn.get('error')}`"]
    lines += [
        "",
        "## Model Node Counts",
        "",
        f"- chunk_model: `{(result.get('chunk_model') or {}).get('node_counts')}`",
        f"- one_token_model: `{(result.get('one_token_model') or {}).get('node_counts')}`",
        "",
        "## Files",
        "",
        f"- result_json: `{log_dir / 'result.json'}`",
        f"- summary_md: `{log_dir / 'summary.md'}`",
    ]
    (log_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp")
    parser.add_argument("--accepted-model", type=Path, default=DEFAULT_ACCEPTED_MODEL)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--existing-chunk-model", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--qnn-run", action="store_true")
    args = parser.parse_args()

    timestamp = args.timestamp or time.strftime("%Y%m%d_%H%M%S_p4_patha2_chunk_mlp")
    log_dir = args.log_dir or (args.log_root / f"p4_patha2_chunk_mlp_{timestamp}")
    model_dir = log_dir / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "timestamp": timestamp,
        "mode": "p4_patha2_chunk_mlp_component_probe",
        "chunk": args.chunk,
        "accepted_model": None if args.skip_build else str(args.accepted_model),
        "log_dir": str(log_dir),
        "safety": {
            "destructive_ops": False,
            "sudo_root_used": False,
            "runner_architecture_changed": False,
        },
    }
    try:
        if args.skip_build:
            if not args.existing_chunk_model:
                raise ValueError("--existing-chunk-model is required with --skip-build")
            chunk_model = args.existing_chunk_model
            one_model = None
            result["chunk_model"] = summarize_model(chunk_model)
        else:
            chunk_model = model_dir / f"patha2_l0_mlp_chunk{args.chunk}_qdq.onnx"
            one_model = model_dir / "patha2_l0_mlp_chunk1_qdq.onnx"
            result["chunk_model"] = build_l0_mlp_model(args.accepted_model, chunk_model, args.chunk)
            result["one_token_model"] = build_l0_mlp_model(args.accepted_model, one_model, 1)
        if not args.skip_cpu and not args.skip_build:
            result["cpu_parity"] = run_cpu_parity(chunk_model, one_model, args.chunk, args.seed)
        if args.qnn_run:
            result["qnn_run"] = run_qnn(chunk_model, log_dir, args.chunk, args.seed)
    except Exception as exc:
        result["ok"] = False
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    else:
        cpu_ok = (result.get("cpu_parity") or {"ok": True}).get("ok")
        qnn_ok = (result.get("qnn_run") or {"ok": True}).get("ok")
        result["ok"] = bool(cpu_ok and qnn_ok)
    (log_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(log_dir, result)
    print(json.dumps({"ok": result.get("ok"), "log_dir": str(log_dir), "summary": str(log_dir / "summary.md")}, ensure_ascii=False))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
