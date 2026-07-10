#!/usr/bin/env python3
import argparse
import gc
import json
import math
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
    dq_const,
    qdq,
    rss_mb,
    summarize_model,
)
from probe_p4_patha2_chunk_l0_layer import add_int64_init


HEADS = 16
KV_HEADS = 8
HEAD_DIM = 64
TARGET_LAYER = 2
DEFAULT_TOTAL_LEN = 128
DEFAULT_CHUNK = 16


def copy_tiled_quant(copier: InitCopier, base: str, target_shape, tag: str):
    quant_name = f"{base}_quantized"
    scale_name = f"{base}_scale"
    zp_name = f"{base}_zero_point"
    arr = numpy_helper.to_array(copier.src[quant_name])
    target_shape = tuple(int(x) for x in target_shape)
    try:
        tiled = np.broadcast_to(arr, target_shape).copy()
    except ValueError:
        reps = [t // s if s != t else 1 for s, t in zip(arr.shape, target_shape)]
        tiled = np.tile(arr, reps)
        if tiled.shape != target_shape:
            raise ValueError(f"cannot tile {quant_name} from {arr.shape} to {target_shape}")
    out_quant = f"{base}_{tag}_quantized"
    if out_quant not in copier.added:
        copier.inits.append(numpy_helper.from_array(tiled, name=out_quant))
        copier.added.add(out_quant)
    copier.copy(scale_name)
    copier.copy(zp_name)
    return out_quant, scale_name, zp_name


def dq_tiled_quant(nodes, copier: InitCopier, base: str, target_shape, tag: str, out_name: str):
    quant, scale, zp = copy_tiled_quant(copier, base, target_shape, tag)
    nodes.append(helper.make_node("DequantizeLinear", [quant, scale, zp], [out_name], f"{base}_{tag}_DequantizeLinear"))
    return out_name


def add_exact_lpnorm_rank4(nodes, copier: InitCopier, x_name: str, norm_prefix: str, heads: int, chunk: int):
    eps = dq_tiled_quant(
        nodes,
        copier,
        f"{norm_prefix}_exact_lpnorm_eps_pad",
        (1, heads, chunk, 1),
        f"chunk{chunk}",
        f"{norm_prefix}_eps_pad_dq",
    )
    scale_weight = dq_const(nodes, copier, f"{norm_prefix}_exact_lpnorm_scale_weight")
    nodes.append(helper.make_node("Concat", [x_name, eps], [f"{norm_prefix}_exact_lpnorm_concat"], f"{norm_prefix}/ExactLpNormConcatEps", axis=3))
    concat = qdq(nodes, copier, f"{norm_prefix}_exact_lpnorm_concat", f"{norm_prefix}_exact_lpnorm_concat", f"{norm_prefix}_exact_lpnorm_concat_dq")
    nodes.append(helper.make_node("LpNormalization", [concat], [f"{norm_prefix}_exact_lpnorm_unit_ext"], f"{norm_prefix}/ExactLpNormalization", axis=3, p=2))
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


def add_exact_lpnorm_rank3(nodes, copier: InitCopier, x_name: str, norm_prefix: str, chunk: int):
    eps = dq_tiled_quant(
        nodes,
        copier,
        f"{norm_prefix}_exact_lpnorm_eps_pad",
        (1, chunk, 1),
        f"chunk{chunk}",
        f"{norm_prefix}_eps_pad_dq",
    )
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


def add_slice(nodes, copier: InitCopier, source: str, prefix: str, axis: int, start: int, end: int):
    starts = add_int64_init(copier, f"{prefix}_starts", [start])
    ends = add_int64_init(copier, f"{prefix}_ends", [end])
    axes = add_int64_init(copier, f"{prefix}_axes", [axis])
    steps = add_int64_init(copier, f"{prefix}_steps", [1])
    out = f"{prefix}_out"
    nodes.append(helper.make_node("Slice", [source, starts, ends, axes, steps], [out], f"{prefix}/Slice"))
    return out


def repeat_kv(nodes, copier: InitCopier, kv_name: str, prefix: str, slice_scale_base: str, repeat_scale_base: str):
    pieces = []
    for idx in range(KV_HEADS):
        sl = add_slice(nodes, copier, kv_name, f"{prefix}_slice_h{idx}", 1, idx, idx + 1)
        sl_dq = qdq(nodes, copier, sl, slice_scale_base, f"{prefix}_slice_h{idx}_dq")
        pieces.extend([sl_dq, sl_dq])
    nodes.append(helper.make_node("Concat", pieces, [f"{prefix}_rep_raw"], f"{prefix}/RepeatOrderedConcat", axis=1))
    return qdq(nodes, copier, f"{prefix}_rep_raw", repeat_scale_base, f"{prefix}_rep")


def add_rope(nodes, copier: InitCopier, x_name: str, prefix: str, heads: int, chunk: int, cos_input: str, sin_input: str):
    first = add_slice(nodes, copier, x_name, f"{prefix}_first", 3, 0, HEAD_DIM // 2)
    second = add_slice(nodes, copier, x_name, f"{prefix}_second", 3, HEAD_DIM // 2, HEAD_DIM)
    minus = dq_tiled_quant(
        nodes,
        copier,
        f"{prefix}_minus_one_dq",
        (1, heads, chunk, HEAD_DIM // 2),
        f"chunk{chunk}",
        f"{prefix}_minus_one_dq_out",
    )
    nodes.append(helper.make_node("Mul", [second, minus], [f"{prefix}_second_neg_raw"], f"{prefix}/NegAsMul"))
    second_neg = qdq(nodes, copier, f"{prefix}_second_neg_raw", f"{prefix}_second_neg_raw", f"{prefix}_second_neg")
    nodes.append(helper.make_node("Concat", [second_neg, first], [f"{prefix}_rot_raw"], f"{prefix}/RotateConcat", axis=3))
    rot = qdq(nodes, copier, f"{prefix}_rot_raw", f"{prefix}_rot_raw", f"{prefix}_rot")
    cos = qdq(nodes, copier, cos_input, f"{prefix}_cos_dq", f"{prefix}_cos")
    sin = qdq(nodes, copier, sin_input, f"{prefix}_sin_dq", f"{prefix}_sin")
    nodes.append(helper.make_node("Mul", [x_name, cos], [f"{prefix}_cos_part_raw"], f"{prefix}/CosMul"))
    cos_part = qdq(nodes, copier, f"{prefix}_cos_part_raw", f"{prefix}_cos_part_raw", f"{prefix}_cos_part")
    nodes.append(helper.make_node("Mul", [rot, sin], [f"{prefix}_sin_part_raw"], f"{prefix}/SinMul"))
    sin_part = qdq(nodes, copier, f"{prefix}_sin_part_raw", f"{prefix}_sin_part_raw", f"{prefix}_sin_part")
    nodes.append(helper.make_node("Add", [cos_part, sin_part], [f"{prefix}_add_raw"], f"{prefix}/Add"))
    return qdq(nodes, copier, f"{prefix}_add_raw", f"{prefix}_add_raw", f"{prefix}_y")


def build_attention_model(accepted_model: Path, output_model: Path, chunk: int, past_len: int, total_len: int):
    if past_len + chunk != total_len:
        raise ValueError(f"past_len + chunk must equal total_len, got {past_len}+{chunk}!={total_len}")
    copier = InitCopier(accepted_model)
    nodes = []
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, chunk, HIDDEN]),
        helper.make_tensor_value_info("past_k2", TensorProto.FLOAT, [1, KV_HEADS, past_len, HEAD_DIM]),
        helper.make_tensor_value_info("past_v2", TensorProto.FLOAT, [1, KV_HEADS, past_len, HEAD_DIM]),
        helper.make_tensor_value_info("l2attn_q_rope_cos_dq", TensorProto.FLOAT, [1, HEADS, chunk, HEAD_DIM]),
        helper.make_tensor_value_info("l2attn_q_rope_sin_dq", TensorProto.FLOAT, [1, HEADS, chunk, HEAD_DIM]),
        helper.make_tensor_value_info("l2attn_k_rope_cos_dq", TensorProto.FLOAT, [1, KV_HEADS, chunk, HEAD_DIM]),
        helper.make_tensor_value_info("l2attn_k_rope_sin_dq", TensorProto.FLOAT, [1, KV_HEADS, chunk, HEAD_DIM]),
        helper.make_tensor_value_info("l2attn_tail_mask", TensorProto.FLOAT, [1, HEADS, chunk, total_len]),
    ]
    outputs = [
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, chunk, HIDDEN]),
        helper.make_tensor_value_info("l2attn_present_k", TensorProto.FLOAT, [1, KV_HEADS, total_len, HEAD_DIM]),
        helper.make_tensor_value_info("l2attn_present_v", TensorProto.FLOAT, [1, KV_HEADS, total_len, HEAD_DIM]),
    ]

    x = qdq(nodes, copier, "x", "l1_mlp_y_raw", "x_dq")
    opnorm = add_exact_lpnorm_rank3(nodes, copier, x, "l2attn_opnorm", chunk)
    q = add_linear(nodes, copier, opnorm, "l2attn_q_proj_w_dq", "l2attn_q_proj")
    k = add_linear(nodes, copier, opnorm, "l2attn_k_proj_w_dq", "l2attn_k_proj")
    v = add_linear(nodes, copier, opnorm, "l2attn_v_proj_w_dq", "l2attn_v_proj")
    shape_q = add_int64_init(copier, f"l2attn_shape_q_chunk{chunk}", [1, chunk, HEADS, HEAD_DIM])
    shape_kv = add_int64_init(copier, f"l2attn_shape_kv_chunk{chunk}", [1, chunk, KV_HEADS, HEAD_DIM])
    nodes.append(helper.make_node("Reshape", [q, shape_q], ["l2attn_q_view"], "l2attn/q/View"))
    nodes.append(helper.make_node("Reshape", [k, shape_kv], ["l2attn_k_view"], "l2attn/k/View"))
    nodes.append(helper.make_node("Reshape", [v, shape_kv], ["l2attn_v_view"], "l2attn/v/View"))
    nodes.append(helper.make_node("Transpose", ["l2attn_q_view"], ["l2attn_q_heads_raw"], "l2attn/q/Transpose", perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Transpose", ["l2attn_k_view"], ["l2attn_k_heads_raw"], "l2attn/k/Transpose", perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Transpose", ["l2attn_v_view"], ["l2attn_v_heads_raw"], "l2attn/v/Transpose", perm=[0, 2, 1, 3]))
    q_heads = qdq(nodes, copier, "l2attn_q_heads_raw", "l2attn_q_proj_mm", "l2attn_q_heads")
    k_heads = qdq(nodes, copier, "l2attn_k_heads_raw", "l2attn_k_proj_mm", "l2attn_k_heads")
    v_heads = qdq(nodes, copier, "l2attn_v_heads_raw", "l2attn_v_proj_mm", "l2attn_v_heads")
    q_norm = add_exact_lpnorm_rank4(nodes, copier, q_heads, "l2attn_q_headnorm", HEADS, chunk)
    k_norm = add_exact_lpnorm_rank4(nodes, copier, k_heads, "l2attn_k_headnorm", KV_HEADS, chunk)
    q_rope = add_rope(nodes, copier, q_norm, "l2attn_q_rope", HEADS, chunk, "l2attn_q_rope_cos_dq", "l2attn_q_rope_sin_dq")
    k_rope = add_rope(nodes, copier, k_norm, "l2attn_k_rope", KV_HEADS, chunk, "l2attn_k_rope_cos_dq", "l2attn_k_rope_sin_dq")
    past_k = qdq(nodes, copier, "past_k2", "past_k2", "past_k2_dq")
    past_v = qdq(nodes, copier, "past_v2", "past_v2", "past_v2_dq")
    nodes.append(helper.make_node("Concat", [past_k, k_rope], ["l2attn_present_k_raw"], "l2attn/present_k/Concat", axis=2))
    nodes.append(helper.make_node("Concat", [past_v, v_heads], ["l2attn_present_v_raw"], "l2attn/present_v/Concat", axis=2))
    present_k = qdq(nodes, copier, "l2attn_present_k_raw", "l2attn_present_k_raw", "l2attn_present_k")
    present_v = qdq(nodes, copier, "l2attn_present_v_raw", "l2attn_present_v_raw", "l2attn_present_v")
    k_rep = repeat_kv(nodes, copier, present_k, "l2attn_k_repeat", "l2attn_present_k_raw", "l2attn_k_repeat_rep_raw")
    v_rep = repeat_kv(nodes, copier, present_v, "l2attn_v_repeat", "l2attn_present_v_raw", "l2attn_v_repeat_rep_raw")
    nodes.append(helper.make_node("Transpose", [k_rep], ["l2attn_k_t_raw"], "l2attn/KCacheTranspose", perm=[0, 1, 3, 2]))
    k_t = qdq(nodes, copier, "l2attn_k_t_raw", "l2attn_k_repeat_rep_raw", "l2attn_k_t")
    nodes.append(helper.make_node("MatMul", [q_rope, k_t], ["l2attn_scores_raw"], "l2attn/ScoresMatMul"))
    scores = qdq(nodes, copier, "l2attn_scores_raw", "l2attn_scores_raw", "l2attn_scores")
    scale = dq_tiled_quant(nodes, copier, "l2attn_scale_dq", (1, HEADS, chunk, total_len), f"chunk{chunk}_total{total_len}", "l2attn_scale_dq_tiled")
    nodes.append(helper.make_node("Mul", [scores, scale], ["l2attn_scaled_raw"], "l2attn/ScaleMul"))
    scaled = qdq(nodes, copier, "l2attn_scaled_raw", "l2attn_scaled_raw", "l2attn_scaled")
    tail_mask = qdq(nodes, copier, "l2attn_tail_mask", "l2attn_tail_mask", "l2attn_tail_mask_dq")
    nodes.append(helper.make_node("Add", [scaled, tail_mask], ["l2attn_masked_scaled_raw"], "l2attn/TailMaskAdd"))
    masked = qdq(nodes, copier, "l2attn_masked_scaled_raw", "l2attn_masked_scaled_raw", "l2attn_masked_scaled")
    nodes.append(helper.make_node("Softmax", [masked], ["l2attn_probs_raw"], "l2attn/Softmax", axis=-1))
    probs = qdq(nodes, copier, "l2attn_probs_raw", "l2attn_probs_raw", "l2attn_probs")
    nodes.append(helper.make_node("MatMul", [probs, v_rep], ["l2attn_context_raw"], "l2attn/ContextMatMul"))
    context = qdq(nodes, copier, "l2attn_context_raw", "l2attn_context_raw", "l2attn_context")
    nodes.append(helper.make_node("Transpose", [context], ["l2attn_context_seq_raw"], "l2attn/ContextTranspose", perm=[0, 2, 1, 3]))
    context_seq = qdq(nodes, copier, "l2attn_context_seq_raw", "l2attn_context_raw", "l2attn_context_seq")
    context_shape = add_int64_init(copier, f"l2attn_context_shape_chunk{chunk}", [1, chunk, HIDDEN])
    nodes.append(helper.make_node("Reshape", [context_seq, context_shape], ["l2attn_context_hidden"], "l2attn/ContextReshape"))
    out = add_linear(nodes, copier, "l2attn_context_hidden", "l2attn_o_proj_w_dq", "l2attn_o_proj")
    nodes.append(helper.make_node("Add", [x, out], ["l2_attn_resid_raw"], "l2/AttentionResidualAdd"))
    y = qdq(nodes, copier, "l2_attn_resid_raw", "l2_attn_resid_raw", "y")

    outputs[0].name = y
    outputs[1].name = present_k
    outputs[2].name = present_v
    graph = helper.make_graph(nodes, output_model.stem, inputs, outputs, copier.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))
    return summarize_model(output_model)


def rope_freq(theta: float):
    return np.asarray([theta ** (-i / (HEAD_DIM // 2)) for i in range(HEAD_DIM // 2)], dtype=np.float32)


def make_rope(position_start: int, chunk: int, heads: int, theta: float):
    positions = np.arange(position_start, position_start + chunk, dtype=np.float32)
    angles = positions.reshape(1, chunk, 1) * rope_freq(theta).reshape(1, 1, HEAD_DIM // 2)
    angles = np.concatenate([angles, angles], axis=-1).astype(np.float32)
    cos = np.tile(np.cos(angles).reshape(1, 1, chunk, HEAD_DIM), (1, heads, 1, 1)).astype(np.float32)
    sin = np.tile(np.sin(angles).reshape(1, 1, chunk, HEAD_DIM), (1, heads, 1, 1)).astype(np.float32)
    return cos, sin


def make_chunk_mask(chunk: int, past_len: int, total_len: int, mask_value: float):
    mask = np.zeros((1, HEADS, chunk, total_len), dtype=np.float32)
    for idx in range(chunk):
        future_start = past_len + idx + 1
        if future_start < total_len:
            mask[:, :, idx, future_start:] = mask_value
    return mask


def make_decode_tail_mask(invalid_prefix: int, total_len: int, mask_value: float):
    mask = np.zeros((1, HEADS, 1, total_len), dtype=np.float32)
    if invalid_prefix > 0:
        mask[:, :, :, :invalid_prefix] = mask_value
    return mask


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


def run_cpu_parity(chunk_model: Path, decode_model: Path, chunk: int, past_len: int, total_len: int, seed: int, mask_value: float, rope_theta: float):
    if total_len != DEFAULT_TOTAL_LEN:
        raise ValueError("rolling decode parity currently expects total_len=128")
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.5, size=(1, chunk, HIDDEN)).astype(np.float32)
    base_k = rng.normal(0.0, 0.25, size=(1, KV_HEADS, past_len, HEAD_DIM)).astype(np.float32)
    base_v = rng.normal(0.0, 0.25, size=(1, KV_HEADS, past_len, HEAD_DIM)).astype(np.float32)
    q_cos, q_sin = make_rope(past_len, chunk, HEADS, rope_theta)
    k_cos, k_sin = make_rope(past_len, chunk, KV_HEADS, rope_theta)
    chunk_mask = make_chunk_mask(chunk, past_len, total_len, mask_value)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    t0 = time.perf_counter()
    chunk_sess = ort.InferenceSession(str(chunk_model), sess_options=so, providers=["CPUExecutionProvider"])
    chunk_create_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    decode_sess = ort.InferenceSession(str(decode_model), sess_options=so, providers=["CPUExecutionProvider"])
    decode_create_s = time.perf_counter() - t1
    chunk_inputs = {
        "x": x,
        "past_k2": base_k,
        "past_v2": base_v,
        "l2attn_q_rope_cos_dq": q_cos,
        "l2attn_q_rope_sin_dq": q_sin,
        "l2attn_k_rope_cos_dq": k_cos,
        "l2attn_k_rope_sin_dq": k_sin,
        "l2attn_tail_mask": chunk_mask,
    }
    t2 = time.perf_counter()
    chunk_y, chunk_present_k, chunk_present_v = chunk_sess.run(None, chunk_inputs)
    chunk_run_s = time.perf_counter() - t2

    invalid_prefix = total_len - 1 - past_len
    state_k = np.concatenate([np.zeros((1, KV_HEADS, invalid_prefix, HEAD_DIM), dtype=np.float32), base_k], axis=2)
    state_v = np.concatenate([np.zeros((1, KV_HEADS, invalid_prefix, HEAD_DIM), dtype=np.float32), base_v], axis=2)
    pieces = []
    t3 = time.perf_counter()
    for idx in range(chunk):
        q_cos_i, q_sin_i = make_rope(past_len + idx, 1, HEADS, rope_theta)
        k_cos_i, k_sin_i = make_rope(past_len + idx, 1, KV_HEADS, rope_theta)
        step_invalid = max(0, invalid_prefix - idx)
        step_mask = make_decode_tail_mask(step_invalid, total_len, mask_value)
        y_i, present_k_i, present_v_i = decode_sess.run(
            None,
            {
                "x": x[:, idx : idx + 1, :],
                "past_k2": state_k,
                "past_v2": state_v,
                "l2attn_q_rope_cos_dq": q_cos_i,
                "l2attn_q_rope_sin_dq": q_sin_i,
                "l2attn_k_rope_cos_dq": k_cos_i,
                "l2attn_k_rope_sin_dq": k_sin_i,
                "l2attn_tail_mask": step_mask,
            },
        )
        pieces.append(y_i)
        state_k = present_k_i[:, :, 1:, :]
        state_v = present_v_i[:, :, 1:, :]
    decode_loop_run_s = time.perf_counter() - t3
    ref_y = np.concatenate(pieces, axis=1)
    y_cmp = compare_arrays(chunk_y, ref_y)
    k_cmp = compare_arrays(chunk_present_k, present_k_i)
    v_cmp = compare_arrays(chunk_present_v, present_v_i)
    # K RoPE/present uses the accepted l2attn_present_k_raw QDQ scale
    # 0.0003834549, so 2 LSB can appear from chunk-vs-loop ordering. Keep
    # parity under 3 LSB and record the exact max/mean/cosine.
    ok = (
        y_cmp["cosine"] is not None
        and y_cmp["cosine"] >= 0.99999
        and y_cmp["max_abs"] <= 2.0e-3
        and k_cmp["cosine"] is not None
        and k_cmp["cosine"] >= 0.99999
        and k_cmp["max_abs"] <= 1.2e-3
        and v_cmp["cosine"] is not None
        and v_cmp["cosine"] >= 0.99999
        and v_cmp["max_abs"] <= 1.2e-3
    )
    return {
        "ok": bool(ok),
        "chunk_create_s": chunk_create_s,
        "decode_create_s": decode_create_s,
        "chunk_run_s": chunk_run_s,
        "decode_loop_run_s": decode_loop_run_s,
        "y": y_cmp,
        "present_k_final": k_cmp,
        "present_v_final": v_cmp,
        "chunk_output_shapes": [list(chunk_y.shape), list(chunk_present_k.shape), list(chunk_present_v.shape)],
        "reference_shapes": [list(ref_y.shape), list(present_k_i.shape), list(present_v_i.shape)],
        "parity_note": "Sequential reference uses a fixed 128-wide rolling decode window with invalid-prefix tail masks; chunk uses equivalent future-token masks.",
        "cache_tolerance_note": "present_k/v threshold is 1.2e-3, under 3 LSB of accepted l2attn_present_k_raw_scale 0.0003834549.",
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


def run_qnn(model_path: Path, log_dir: Path, chunk: int, past_len: int, total_len: int, seed: int, mask_value: float, rope_theta: float):
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
        so.profile_file_prefix = str(log_dir / "qnn_l2_attention_chunk_profile")
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_provider_for_devices(qdevs, provider_options)
        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(model_path), sess_options=so, enable_fallback=False)
        case["session_create_s"] = time.perf_counter() - t0
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 0.5, size=(1, chunk, HIDDEN)).astype(np.float32)
        past_k = rng.normal(0.0, 0.25, size=(1, KV_HEADS, past_len, HEAD_DIM)).astype(np.float32)
        past_v = rng.normal(0.0, 0.25, size=(1, KV_HEADS, past_len, HEAD_DIM)).astype(np.float32)
        q_cos, q_sin = make_rope(past_len, chunk, HEADS, rope_theta)
        k_cos, k_sin = make_rope(past_len, chunk, KV_HEADS, rope_theta)
        mask = make_chunk_mask(chunk, past_len, total_len, mask_value)
        t1 = time.perf_counter()
        outputs = sess.run(
            None,
            {
                "x": x,
                "past_k2": past_k,
                "past_v2": past_v,
                "l2attn_q_rope_cos_dq": q_cos,
                "l2attn_q_rope_sin_dq": q_sin,
                "l2attn_k_rope_cos_dq": k_cos,
                "l2attn_k_rope_sin_dq": k_sin,
                "l2attn_tail_mask": mask,
            },
        )
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
        f"# P4 Path A2 Chunk L2 Attention Probe {result['timestamp']}",
        "",
        "This builds a vectorized L2 attention minigraph from accepted V0 final QDQ constants.",
        "It is a P4 component probe, not a V1 completion claim or runner architecture.",
        "",
        "## Contract",
        "",
        f"- chunk: `{result['chunk']}`",
        f"- chunk_past_len: `{result['chunk_past_len']}`",
        f"- decode_reference_past_len: `{result['decode_reference_past_len']}`",
        f"- total_len: `{result['total_len']}`",
        f"- mask_value: `{result['mask_value']}`",
        f"- rope_theta: `{result['rope_theta']}`",
        f"- accepted_model: `{result.get('accepted_model')}`",
        f"- chunk_model: `{(result.get('chunk_model') or {}).get('path')}`",
        f"- decode_reference_model: `{(result.get('decode_reference_model') or {}).get('path')}`",
        "- inputs include per-position RoPE cos/sin tensors and an additive tail/causal mask.",
        "- outputs: attention residual `y`, `l2attn_present_k`, and `l2attn_present_v`.",
        "",
        "## CPU Chunk-vs-Decode Parity",
        "",
    ]
    cpu = result.get("cpu_parity")
    if cpu:
        lines += [
            f"- ok: `{cpu.get('ok')}`",
            f"- y_max_abs: `{(cpu.get('y') or {}).get('max_abs')}`",
            f"- y_cosine: `{(cpu.get('y') or {}).get('cosine')}`",
            f"- present_k_max_abs: `{(cpu.get('present_k_final') or {}).get('max_abs')}`",
            f"- present_k_cosine: `{(cpu.get('present_k_final') or {}).get('cosine')}`",
            f"- present_v_max_abs: `{(cpu.get('present_v_final') or {}).get('max_abs')}`",
            f"- present_v_cosine: `{(cpu.get('present_v_final') or {}).get('cosine')}`",
            f"- chunk_run_s: `{cpu.get('chunk_run_s')}`",
            f"- decode_loop_run_s: `{cpu.get('decode_loop_run_s')}`",
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
        f"- decode_reference_model: `{(result.get('decode_reference_model') or {}).get('node_counts')}`",
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
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--total-len", type=int, default=DEFAULT_TOTAL_LEN)
    parser.add_argument("--seed", type=int, default=9012)
    parser.add_argument("--mask-value", type=float, default=-64.0)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--existing-chunk-model", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--qnn-run", action="store_true")
    args = parser.parse_args()

    timestamp = args.timestamp or time.strftime("%Y%m%d_%H%M%S_p4_patha2_chunk_l2_attention")
    log_dir = args.log_dir or (args.log_root / f"p4_patha2_chunk_l2_attention_{timestamp}")
    model_dir = log_dir / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    chunk_past_len = args.total_len - args.chunk
    decode_past_len = args.total_len - 1
    result = {
        "timestamp": timestamp,
        "mode": "p4_patha2_chunk_l2_attention_component_probe",
        "chunk": args.chunk,
        "chunk_past_len": chunk_past_len,
        "decode_reference_past_len": decode_past_len,
        "total_len": args.total_len,
        "mask_value": args.mask_value,
        "rope_theta": args.rope_theta,
        "accepted_model": None if args.skip_build else str(args.accepted_model),
        "log_dir": str(log_dir),
        "safety": {"destructive_ops": False, "sudo_root_used": False, "runner_architecture_changed": False},
    }
    try:
        if chunk_past_len < 0:
            raise ValueError("--total-len must be >= --chunk")
        if args.skip_build:
            if not args.existing_chunk_model:
                raise ValueError("--existing-chunk-model is required with --skip-build")
            chunk_model = args.existing_chunk_model
            decode_model = None
            result["chunk_model"] = summarize_model(chunk_model)
        else:
            chunk_model = model_dir / f"patha2_l2_attention_chunk{args.chunk}_past{chunk_past_len}_total{args.total_len}_qdq.onnx"
            decode_model = model_dir / f"patha2_l2_attention_decode1_past{decode_past_len}_total{args.total_len}_qdq.onnx"
            result["chunk_model"] = build_attention_model(args.accepted_model, chunk_model, args.chunk, chunk_past_len, args.total_len)
            result["decode_reference_model"] = build_attention_model(args.accepted_model, decode_model, 1, decode_past_len, args.total_len)
        if not args.skip_cpu and not args.skip_build:
            result["cpu_parity"] = run_cpu_parity(chunk_model, decode_model, args.chunk, chunk_past_len, args.total_len, args.seed, args.mask_value, args.rope_theta)
        if args.qnn_run:
            result["qnn_run"] = run_qnn(chunk_model, log_dir, args.chunk, chunk_past_len, args.total_len, args.seed, args.mask_value, args.rope_theta)
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
