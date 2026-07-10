#!/usr/bin/env python3
import argparse
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

from probe_p4_patha2_chunk_mlp import (
    DEFAULT_ACCEPTED_MODEL,
    DEFAULT_LOG_ROOT,
    HIDDEN,
    InitCopier,
    add_exact_lpnorm,
    add_linear,
    dq_const,
    qdq,
    rss_mb,
    summarize_model,
)


PAST_CONV = 3


def add_int64_init(copier: InitCopier, name: str, values):
    if name in copier.added:
        return name
    copier.inits.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
    copier.added.add(name)
    return name


def add_l0_mlp_after_resid(nodes, copier: InitCopier, resid_name: str, chunk: int):
    norm = add_exact_lpnorm(nodes, copier, resid_name, "l0_mlp_ffn_norm", chunk)
    gate = add_linear(nodes, copier, norm, "l0_mlp_gate_w_dq", "l0_mlp_gate")
    up = add_linear(nodes, copier, norm, "l0_mlp_up_w_dq", "l0_mlp_up")
    nodes.append(helper.make_node("Sigmoid", [gate], ["l0_mlp_gate_sig_raw"], "l0_mlp/SiluSigmoid"))
    sig = qdq(nodes, copier, "l0_mlp_gate_sig_raw", "l0_mlp_gate_sig_raw", "l0_mlp_gate_sig_raw_dq")
    nodes.append(helper.make_node("Mul", [gate, sig], ["l0_mlp_gate_act_raw"], "l0_mlp/SiluMul"))
    act = qdq(nodes, copier, "l0_mlp_gate_act_raw", "l0_mlp_gate_act_raw", "l0_mlp_gate_act_raw_dq")
    nodes.append(helper.make_node("Mul", [act, up], ["l0_mlp_ffn_mul_raw"], "l0_mlp/GateUpMul"))
    ffn = qdq(nodes, copier, "l0_mlp_ffn_mul_raw", "l0_mlp_ffn_mul_raw", "l0_mlp_ffn_mul_raw_dq")
    down = add_linear(nodes, copier, ffn, "l0_mlp_down_proj_w_dq", "l0_mlp_down_proj")
    nodes.append(helper.make_node("Add", [resid_name, down], ["l0_mlp_y_raw"], "l0_mlp/ResidualAdd"))
    return qdq(nodes, copier, "l0_mlp_y_raw", "l0_mlp_y_raw", "y")


def dq_conv_weight(nodes, copier: InitCopier):
    quant = copier.copy("l0_conv_w_dq_quantized")
    scale = copier.copy("l0_conv_w_dq_scale")
    zp = copier.copy("l0_conv_w_dq_zero_point")
    out = "l0_conv_w_dq_DequantizeLinear_Output"
    nodes.append(helper.make_node("DequantizeLinear", [quant, scale, zp], [out], "l0_conv_w_dq_DequantizeLinear", axis=0))
    return out


def build_l0_layer_model(accepted_model: Path, output_model: Path, chunk: int):
    copier = InitCopier(accepted_model)
    nodes = []
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, chunk, HIDDEN]),
        helper.make_tensor_value_info("past_conv0", TensorProto.FLOAT, [1, HIDDEN, PAST_CONV]),
    ]
    outputs = [
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, chunk, HIDDEN]),
        helper.make_tensor_value_info("present_conv0", TensorProto.FLOAT, [1, HIDDEN, PAST_CONV]),
    ]

    x = qdq(nodes, copier, "x", "x", "x_dq")
    past = qdq(nodes, copier, "past_conv0", "past_conv0", "past_conv0_dq")
    opnorm = add_exact_lpnorm(nodes, copier, x, "l0_opnorm", chunk)
    in_proj = add_linear(nodes, copier, opnorm, "l0_in_proj_w_dq", "l0_in_proj")
    nodes.append(helper.make_node("Transpose", [in_proj], ["l0_proj_t_raw"], "l0/InProjTranspose", perm=[0, 2, 1]))
    proj_t = qdq(nodes, copier, "l0_proj_t_raw", "l0_in_proj_mm", "l0_proj_t_raw_dq")
    copier.copy("l0_split_sizes")
    nodes.append(helper.make_node("Split", [proj_t, "l0_split_sizes"], ["l0_split_a_raw", "l0_split_b_raw", "l0_split_c_raw"], "l0/Split", axis=1))
    split_a = qdq(nodes, copier, "l0_split_a_raw", "l0_in_proj_mm", "l0_split_a_raw_dq")
    split_b = qdq(nodes, copier, "l0_split_b_raw", "l0_in_proj_mm", "l0_split_b_raw_dq")
    split_c = qdq(nodes, copier, "l0_split_c_raw", "l0_in_proj_mm", "l0_split_c_raw_dq")
    nodes.append(helper.make_node("Mul", [split_a, split_c], ["l0_gate_raw"], "l0/GateMul"))
    gate = qdq(nodes, copier, "l0_gate_raw", "l0_gate_raw", "l0_gate_raw_dq")
    nodes.append(helper.make_node("Concat", [past, gate], ["l0_conv_input_raw"], "l0/ConvInputConcat", axis=2))
    conv_input = qdq(nodes, copier, "l0_conv_input_raw", "l0_conv_input_raw", "l0_conv_input_raw_dq")

    present_starts = add_int64_init(copier, f"l0_present_slice_chunk{chunk}_starts", [chunk])
    present_ends = add_int64_init(copier, f"l0_present_slice_chunk{chunk}_ends", [chunk + PAST_CONV])
    present_axes = add_int64_init(copier, f"l0_present_slice_chunk{chunk}_axes", [2])
    present_steps = add_int64_init(copier, f"l0_present_slice_chunk{chunk}_steps", [1])
    nodes.append(helper.make_node("Slice", [conv_input, present_starts, present_ends, present_axes, present_steps], ["l0_present_slice_out"], "l0_present_slice/Slice"))
    present = qdq(nodes, copier, "l0_present_slice_out", "l0_conv_input_raw", "present_conv0")

    conv_w = dq_conv_weight(nodes, copier)
    nodes.append(helper.make_node("Conv", [conv_input, conv_w], ["l0_conv_raw"], "l0/DepthwiseConv", group=HIDDEN, pads=[0, 0], strides=[1]))
    conv = qdq(nodes, copier, "l0_conv_raw", "l0_conv_raw", "l0_conv_raw_dq")
    conv_starts = add_int64_init(copier, f"l0_conv_last_slice_chunk{chunk}_starts", [1])
    conv_ends = add_int64_init(copier, f"l0_conv_last_slice_chunk{chunk}_ends", [chunk + 1])
    conv_axes = add_int64_init(copier, f"l0_conv_last_slice_chunk{chunk}_axes", [2])
    conv_steps = add_int64_init(copier, f"l0_conv_last_slice_chunk{chunk}_steps", [1])
    nodes.append(helper.make_node("Slice", [conv, conv_starts, conv_ends, conv_axes, conv_steps], ["l0_conv_last_slice_out"], "l0_conv_last_slice/Slice"))
    conv_seq = qdq(nodes, copier, "l0_conv_last_slice_out", "l0_conv_raw", "l0_conv_last_slice_out_dq")
    nodes.append(helper.make_node("Mul", [split_b, conv_seq], ["l0_mix_raw"], "l0/MixMul"))
    mix = qdq(nodes, copier, "l0_mix_raw", "l0_mix_raw", "l0_mix_raw_dq")
    nodes.append(helper.make_node("Transpose", [mix], ["l0_mix_seq_raw"], "l0/MixTranspose", perm=[0, 2, 1]))
    mix_seq = qdq(nodes, copier, "l0_mix_seq_raw", "l0_mix_raw", "l0_mix_seq_raw_dq")
    out = add_linear(nodes, copier, mix_seq, "l0_out_proj_w_dq", "l0_out_proj")
    nodes.append(helper.make_node("Add", [x, out], ["l0_resid_y_raw"], "l0/AttentionResidualAdd"))
    resid = qdq(nodes, copier, "l0_resid_y_raw", "l0_resid_y_raw", "l0_resid_y_raw_dq")
    y = add_l0_mlp_after_resid(nodes, copier, resid, chunk)

    outputs[0].name = y
    outputs[1].name = present
    graph = helper.make_graph(nodes, output_model.stem, inputs, outputs, copier.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))
    return summarize_model(output_model)


def compare_arrays(a, b):
    diff = a.astype(np.float64) - b.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    denom = float(np.linalg.norm(a.reshape(-1)) * np.linalg.norm(b.reshape(-1)))
    cosine = float(np.dot(a.reshape(-1), b.reshape(-1)) / denom) if denom else None
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
        "exact_equal": bool(np.array_equal(a, b)),
    }


def run_cpu_parity(chunk_model: Path, one_model: Path, chunk: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.5, size=(1, chunk, HIDDEN)).astype(np.float32)
    past = rng.normal(0.0, 0.25, size=(1, HIDDEN, PAST_CONV)).astype(np.float32)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    t0 = time.perf_counter()
    chunk_sess = ort.InferenceSession(str(chunk_model), sess_options=so, providers=["CPUExecutionProvider"])
    chunk_create_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    one_sess = ort.InferenceSession(str(one_model), sess_options=so, providers=["CPUExecutionProvider"])
    one_create_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    chunk_y, chunk_present = chunk_sess.run(None, {"x": x, "past_conv0": past})
    chunk_run_s = time.perf_counter() - t2
    pieces = []
    state = past
    t3 = time.perf_counter()
    for idx in range(chunk):
        y_i, state = one_sess.run(None, {"x": x[:, idx : idx + 1, :], "past_conv0": state})
        pieces.append(y_i)
    one_loop_run_s = time.perf_counter() - t3
    ref_y = np.concatenate(pieces, axis=1)
    ref_present = state
    y_cmp = compare_arrays(chunk_y, ref_y)
    present_cmp = compare_arrays(chunk_present, ref_present)
    # Chunk execution changes reduction/order across MatMul, Conv, and two
    # ExactLpNorms. The cache must be exact; hidden output tolerance is kept to
    # a few QDQ LSBs and recorded explicitly.
    ok = (
        y_cmp["max_abs"] <= 5e-5
        and present_cmp["max_abs"] <= 1e-6
        and y_cmp["cosine"] is not None
        and y_cmp["cosine"] >= 0.999999
        and present_cmp["cosine"] is not None
        and present_cmp["cosine"] >= 0.999999
    )
    return {
        "ok": bool(ok),
        "chunk_create_s": chunk_create_s,
        "one_create_s": one_create_s,
        "chunk_run_s": chunk_run_s,
        "one_loop_run_s": one_loop_run_s,
        "y": y_cmp,
        "present_conv": present_cmp,
        "chunk_output_shapes": [list(chunk_y.shape), list(chunk_present.shape)],
        "reference_shapes": [list(ref_y.shape), list(ref_present.shape)],
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
    return register_status, qdevs, provider_options


def provider_counts_from_profile(profile_path):
    counts = Counter()
    try:
        data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    for event in data:
        provider = (event.get("args") or {}).get("provider")
        if provider:
            counts[str(provider)] += 1
    return dict(sorted(counts.items()))


def run_qnn(model_path: Path, log_dir: Path, chunk: int, seed: int):
    case = {"ok": False, "model": str(model_path), "rss_before_mb": rss_mb()}
    sess = None
    try:
        register_status, qdevs, provider_options = qnn_register()
        case["register_status"] = register_status
        case["qnn_devices"] = [getattr(d, "ep_name", repr(d)) for d in qdevs]
        if not qdevs:
            raise RuntimeError("No QNN OrtEpDevice after provider registration")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_profiling = True
        so.profile_file_prefix = str(log_dir / "qnn_l0_layer_chunk_profile")
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_provider_for_devices(qdevs, provider_options)
        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(model_path), sess_options=so, enable_fallback=False)
        case["session_create_s"] = time.perf_counter() - t0
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 0.5, size=(1, chunk, HIDDEN)).astype(np.float32)
        past = rng.normal(0.0, 0.25, size=(1, HIDDEN, PAST_CONV)).astype(np.float32)
        t1 = time.perf_counter()
        outputs = sess.run(None, {"x": x, "past_conv0": past})
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
        f"# P4 Path A2 Chunk L0 Layer Probe {result['timestamp']}",
        "",
        "This builds a vectorized layer0 conv+MLP minigraph from accepted V0 final QDQ constants.",
        "It is a P4 component probe, not a V1 completion claim or runner architecture.",
        "",
        "## Contract",
        "",
        f"- chunk: `{result['chunk']}`",
        f"- accepted_model: `{result.get('accepted_model')}`",
        f"- chunk_model: `{(result.get('chunk_model') or {}).get('path')}`",
        f"- one_token_model: `{(result.get('one_token_model') or {}).get('path')}`",
        "- inputs: `x=[1,chunk,1024]`, `past_conv0=[1,1024,3]`",
        "- outputs: `y=[1,chunk,1024]`, `present_conv0=[1,1024,3]`",
        "",
        "## CPU Chunk-vs-Sequential Parity",
        "",
    ]
    cpu = result.get("cpu_parity")
    if cpu:
        lines += [
            f"- ok: `{cpu.get('ok')}`",
            f"- y_max_abs: `{(cpu.get('y') or {}).get('max_abs')}`",
            f"- y_cosine: `{(cpu.get('y') or {}).get('cosine')}`",
            f"- present_max_abs: `{(cpu.get('present_conv') or {}).get('max_abs')}`",
            f"- present_cosine: `{(cpu.get('present_conv') or {}).get('cosine')}`",
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
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--existing-chunk-model", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--qnn-run", action="store_true")
    args = parser.parse_args()

    timestamp = args.timestamp or time.strftime("%Y%m%d_%H%M%S_p4_patha2_chunk_l0")
    log_dir = args.log_dir or (args.log_root / f"p4_patha2_chunk_l0_layer_{timestamp}")
    model_dir = log_dir / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": timestamp,
        "mode": "p4_patha2_chunk_l0_layer_component_probe",
        "chunk": args.chunk,
        "accepted_model": None if args.skip_build else str(args.accepted_model),
        "log_dir": str(log_dir),
        "safety": {"destructive_ops": False, "sudo_root_used": False, "runner_architecture_changed": False},
    }
    try:
        if args.skip_build:
            if not args.existing_chunk_model:
                raise ValueError("--existing-chunk-model is required with --skip-build")
            chunk_model = args.existing_chunk_model
            one_model = None
            result["chunk_model"] = summarize_model(chunk_model)
        else:
            chunk_model = model_dir / f"patha2_l0_layer_chunk{args.chunk}_qdq.onnx"
            one_model = model_dir / "patha2_l0_layer_chunk1_qdq.onnx"
            result["chunk_model"] = build_l0_layer_model(args.accepted_model, chunk_model, args.chunk)
            result["one_token_model"] = build_l0_layer_model(args.accepted_model, one_model, 1)
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
