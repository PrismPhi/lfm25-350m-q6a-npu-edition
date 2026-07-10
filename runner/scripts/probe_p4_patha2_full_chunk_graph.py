#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import time
import traceback
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
from probe_p4_patha2_chunk_l2_attention import (
    HEADS,
    HEAD_DIM,
    KV_HEADS,
    add_exact_lpnorm_rank3,
    add_exact_lpnorm_rank4,
    add_linear,
    add_rope,
    dq_tiled_quant,
    make_chunk_mask,
    make_rope,
    provider_counts_from_profile,
    qnn_register,
    repeat_kv,
)


CONV_LAYERS = [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]
ATTN_LAYERS = [2, 5, 8, 10, 12, 14]
PAST_CONV = 3
DEFAULT_CHUNK = 16
DEFAULT_TOTAL_LEN = 128
DEFAULT_MASK_VALUE = -64.0
DEFAULT_ROPE_THETA = 1_000_000.0
VOCAB = 65536


def dq_conv_weight(nodes, copier: InitCopier, layer: int):
    prefix = f"l{layer}"
    quant = copier.copy(f"{prefix}_conv_w_dq_quantized")
    scale = copier.copy(f"{prefix}_conv_w_dq_scale")
    zp = copier.copy(f"{prefix}_conv_w_dq_zero_point")
    out = f"{prefix}_conv_w_dq"
    nodes.append(helper.make_node("DequantizeLinear", [quant, scale, zp], [out], f"{prefix}_conv_w_dq/DequantizeLinear", axis=0))
    return out


def add_mlp_after_resid(nodes, copier: InitCopier, resid_name: str, mlp_prefix: str, output_raw: str, chunk: int):
    norm = add_exact_lpnorm_rank3(nodes, copier, resid_name, f"{mlp_prefix}_ffn_norm", chunk)
    gate = add_linear(nodes, copier, norm, f"{mlp_prefix}_gate_w_dq", f"{mlp_prefix}_gate")
    up = add_linear(nodes, copier, norm, f"{mlp_prefix}_up_w_dq", f"{mlp_prefix}_up")
    nodes.append(helper.make_node("Sigmoid", [gate], [f"{mlp_prefix}_gate_sig_raw"], f"{mlp_prefix}/SiluSigmoid"))
    sig = qdq(nodes, copier, f"{mlp_prefix}_gate_sig_raw", f"{mlp_prefix}_gate_sig_raw", f"{mlp_prefix}_gate_sig_raw_dq")
    nodes.append(helper.make_node("Mul", [gate, sig], [f"{mlp_prefix}_gate_act_raw"], f"{mlp_prefix}/SiluMul"))
    act = qdq(nodes, copier, f"{mlp_prefix}_gate_act_raw", f"{mlp_prefix}_gate_act_raw", f"{mlp_prefix}_gate_act_raw_dq")
    nodes.append(helper.make_node("Mul", [act, up], [f"{mlp_prefix}_ffn_mul_raw"], f"{mlp_prefix}/GateUpMul"))
    ffn = qdq(nodes, copier, f"{mlp_prefix}_ffn_mul_raw", f"{mlp_prefix}_ffn_mul_raw", f"{mlp_prefix}_ffn_mul_raw_dq")
    down = add_linear(nodes, copier, ffn, f"{mlp_prefix}_down_proj_w_dq", f"{mlp_prefix}_down_proj")
    nodes.append(helper.make_node("Add", [resid_name, down], [output_raw], f"{mlp_prefix}/ResidualAdd"))
    return qdq(nodes, copier, output_raw, output_raw, f"{output_raw}_dq")


def add_conv_layer(nodes, copier: InitCopier, x_name: str, layer: int, chunk: int):
    prefix = f"l{layer}"
    opnorm = add_exact_lpnorm_rank3(nodes, copier, x_name, f"{prefix}_opnorm", chunk)
    in_proj = add_linear(nodes, copier, opnorm, f"{prefix}_in_proj_w_dq", f"{prefix}_in_proj")
    nodes.append(helper.make_node("Transpose", [in_proj], [f"{prefix}_proj_t_raw"], f"{prefix}/InProjTranspose", perm=[0, 2, 1]))
    proj_t = qdq(nodes, copier, f"{prefix}_proj_t_raw", f"{prefix}_in_proj_mm", f"{prefix}_proj_t_raw_dq")
    copier.copy(f"{prefix}_split_sizes")
    nodes.append(
        helper.make_node(
            "Split",
            [proj_t, f"{prefix}_split_sizes"],
            [f"{prefix}_split_a_raw", f"{prefix}_split_b_raw", f"{prefix}_split_c_raw"],
            f"{prefix}/Split",
            axis=1,
        )
    )
    split_a = qdq(nodes, copier, f"{prefix}_split_a_raw", f"{prefix}_in_proj_mm", f"{prefix}_split_a_raw_dq")
    split_b = qdq(nodes, copier, f"{prefix}_split_b_raw", f"{prefix}_in_proj_mm", f"{prefix}_split_b_raw_dq")
    split_c = qdq(nodes, copier, f"{prefix}_split_c_raw", f"{prefix}_in_proj_mm", f"{prefix}_split_c_raw_dq")
    nodes.append(helper.make_node("Mul", [split_a, split_c], [f"{prefix}_gate_raw"], f"{prefix}/GateMul"))
    gate = qdq(nodes, copier, f"{prefix}_gate_raw", f"{prefix}_gate_raw", f"{prefix}_gate_raw_dq")
    past_name = f"past_conv{layer}"
    past = qdq(nodes, copier, past_name, past_name, f"{past_name}_dq")
    nodes.append(helper.make_node("Concat", [past, gate], [f"{prefix}_conv_input_raw"], f"{prefix}/ConvInputConcat", axis=2))
    conv_input = qdq(nodes, copier, f"{prefix}_conv_input_raw", f"{prefix}_conv_input_raw", f"{prefix}_conv_input_raw_dq")

    present_starts = add_int64_init(copier, f"{prefix}_present_slice_chunk{chunk}_starts", [chunk])
    present_ends = add_int64_init(copier, f"{prefix}_present_slice_chunk{chunk}_ends", [chunk + PAST_CONV])
    present_axes = add_int64_init(copier, f"{prefix}_present_slice_chunk{chunk}_axes", [2])
    present_steps = add_int64_init(copier, f"{prefix}_present_slice_chunk{chunk}_steps", [1])
    nodes.append(
        helper.make_node(
            "Slice",
            [conv_input, present_starts, present_ends, present_axes, present_steps],
            [f"{prefix}_present_slice_out"],
            f"{prefix}_present_slice/Slice",
        )
    )
    present = qdq(nodes, copier, f"{prefix}_present_slice_out", f"{prefix}_conv_input_raw", f"{prefix}_present_conv")

    conv_w = dq_conv_weight(nodes, copier, layer)
    nodes.append(helper.make_node("Conv", [conv_input, conv_w], [f"{prefix}_conv_raw"], f"{prefix}/DepthwiseConv", group=HIDDEN, pads=[0, 0], strides=[1]))
    conv = qdq(nodes, copier, f"{prefix}_conv_raw", f"{prefix}_conv_raw", f"{prefix}_conv_raw_dq")
    conv_starts = add_int64_init(copier, f"{prefix}_conv_last_slice_chunk{chunk}_starts", [1])
    conv_ends = add_int64_init(copier, f"{prefix}_conv_last_slice_chunk{chunk}_ends", [chunk + 1])
    conv_axes = add_int64_init(copier, f"{prefix}_conv_last_slice_chunk{chunk}_axes", [2])
    conv_steps = add_int64_init(copier, f"{prefix}_conv_last_slice_chunk{chunk}_steps", [1])
    nodes.append(
        helper.make_node(
            "Slice",
            [conv, conv_starts, conv_ends, conv_axes, conv_steps],
            [f"{prefix}_conv_last_slice_out"],
            f"{prefix}_conv_last_slice/Slice",
        )
    )
    conv_seq = qdq(nodes, copier, f"{prefix}_conv_last_slice_out", f"{prefix}_conv_raw", f"{prefix}_conv_last_slice_out_dq")
    nodes.append(helper.make_node("Mul", [split_b, conv_seq], [f"{prefix}_mix_raw"], f"{prefix}/MixMul"))
    mix = qdq(nodes, copier, f"{prefix}_mix_raw", f"{prefix}_mix_raw", f"{prefix}_mix_raw_dq")
    nodes.append(helper.make_node("Transpose", [mix], [f"{prefix}_mix_seq_raw"], f"{prefix}/MixTranspose", perm=[0, 2, 1]))
    mix_seq = qdq(nodes, copier, f"{prefix}_mix_seq_raw", f"{prefix}_mix_raw", f"{prefix}_mix_seq_raw_dq")
    out = add_linear(nodes, copier, mix_seq, f"{prefix}_out_proj_w_dq", f"{prefix}_out_proj")
    nodes.append(helper.make_node("Add", [x_name, out], [f"{prefix}_resid_y_raw"], f"{prefix}/AttentionResidualAdd"))
    resid = qdq(nodes, copier, f"{prefix}_resid_y_raw", f"{prefix}_resid_y_raw", f"{prefix}_resid_y_raw_dq")
    y = add_mlp_after_resid(nodes, copier, resid, f"{prefix}_mlp", f"{prefix}_mlp_y_raw", chunk)
    return y, present


def add_attention_layer(nodes, copier: InitCopier, x_name: str, layer: int, chunk: int, past_len: int, total_len: int):
    prefix = f"l{layer}attn"
    layer_prefix = f"l{layer}"
    opnorm = add_exact_lpnorm_rank3(nodes, copier, x_name, f"{prefix}_opnorm", chunk)
    q = add_linear(nodes, copier, opnorm, f"{prefix}_q_proj_w_dq", f"{prefix}_q_proj")
    k = add_linear(nodes, copier, opnorm, f"{prefix}_k_proj_w_dq", f"{prefix}_k_proj")
    v = add_linear(nodes, copier, opnorm, f"{prefix}_v_proj_w_dq", f"{prefix}_v_proj")
    shape_q = add_int64_init(copier, f"{prefix}_shape_q_chunk{chunk}", [1, chunk, HEADS, HEAD_DIM])
    shape_kv = add_int64_init(copier, f"{prefix}_shape_kv_chunk{chunk}", [1, chunk, KV_HEADS, HEAD_DIM])
    nodes.append(helper.make_node("Reshape", [q, shape_q], [f"{prefix}_q_view"], f"{prefix}/q/View"))
    nodes.append(helper.make_node("Reshape", [k, shape_kv], [f"{prefix}_k_view"], f"{prefix}/k/View"))
    nodes.append(helper.make_node("Reshape", [v, shape_kv], [f"{prefix}_v_view"], f"{prefix}/v/View"))
    nodes.append(helper.make_node("Transpose", [f"{prefix}_q_view"], [f"{prefix}_q_heads_raw"], f"{prefix}/q/Transpose", perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Transpose", [f"{prefix}_k_view"], [f"{prefix}_k_heads_raw"], f"{prefix}/k/Transpose", perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Transpose", [f"{prefix}_v_view"], [f"{prefix}_v_heads_raw"], f"{prefix}/v/Transpose", perm=[0, 2, 1, 3]))
    q_heads = qdq(nodes, copier, f"{prefix}_q_heads_raw", f"{prefix}_q_proj_mm", f"{prefix}_q_heads")
    k_heads = qdq(nodes, copier, f"{prefix}_k_heads_raw", f"{prefix}_k_proj_mm", f"{prefix}_k_heads")
    v_heads = qdq(nodes, copier, f"{prefix}_v_heads_raw", f"{prefix}_v_proj_mm", f"{prefix}_v_heads")
    q_norm = add_exact_lpnorm_rank4(nodes, copier, q_heads, f"{prefix}_q_headnorm", HEADS, chunk)
    k_norm = add_exact_lpnorm_rank4(nodes, copier, k_heads, f"{prefix}_k_headnorm", KV_HEADS, chunk)
    q_rope = add_rope(nodes, copier, q_norm, f"{prefix}_q_rope", HEADS, chunk, f"{prefix}_q_rope_cos_dq", f"{prefix}_q_rope_sin_dq")
    k_rope = add_rope(nodes, copier, k_norm, f"{prefix}_k_rope", KV_HEADS, chunk, f"{prefix}_k_rope_cos_dq", f"{prefix}_k_rope_sin_dq")
    past_k_name = f"past_k{layer}"
    past_v_name = f"past_v{layer}"
    past_k = qdq(nodes, copier, past_k_name, past_k_name, f"{past_k_name}_dq")
    past_v = qdq(nodes, copier, past_v_name, past_v_name, f"{past_v_name}_dq")
    nodes.append(helper.make_node("Concat", [past_k, k_rope], [f"{prefix}_present_k_raw"], f"{prefix}/present_k/Concat", axis=2))
    nodes.append(helper.make_node("Concat", [past_v, v_heads], [f"{prefix}_present_v_raw"], f"{prefix}/present_v/Concat", axis=2))
    present_k = qdq(nodes, copier, f"{prefix}_present_k_raw", f"{prefix}_present_k_raw", f"{prefix}_present_k")
    present_v = qdq(nodes, copier, f"{prefix}_present_v_raw", f"{prefix}_present_v_raw", f"{prefix}_present_v")
    k_rep = repeat_kv(nodes, copier, present_k, f"{prefix}_k_repeat", f"{prefix}_present_k_raw", f"{prefix}_k_repeat_rep_raw")
    v_rep = repeat_kv(nodes, copier, present_v, f"{prefix}_v_repeat", f"{prefix}_present_v_raw", f"{prefix}_v_repeat_rep_raw")
    nodes.append(helper.make_node("Transpose", [k_rep], [f"{prefix}_k_t_raw"], f"{prefix}/KCacheTranspose", perm=[0, 1, 3, 2]))
    k_t = qdq(nodes, copier, f"{prefix}_k_t_raw", f"{prefix}_k_repeat_rep_raw", f"{prefix}_k_t")
    nodes.append(helper.make_node("MatMul", [q_rope, k_t], [f"{prefix}_scores_raw"], f"{prefix}/ScoresMatMul"))
    scores = qdq(nodes, copier, f"{prefix}_scores_raw", f"{prefix}_scores_raw", f"{prefix}_scores")
    scale = dq_tiled_quant(nodes, copier, f"{prefix}_scale_dq", (1, HEADS, chunk, total_len), f"chunk{chunk}_total{total_len}", f"{prefix}_scale_dq_tiled")
    nodes.append(helper.make_node("Mul", [scores, scale], [f"{prefix}_scaled_raw"], f"{prefix}/ScaleMul"))
    scaled = qdq(nodes, copier, f"{prefix}_scaled_raw", f"{prefix}_scaled_raw", f"{prefix}_scaled")
    tail_mask = qdq(nodes, copier, f"{prefix}_tail_mask", f"{prefix}_tail_mask", f"{prefix}_tail_mask_dq")
    nodes.append(helper.make_node("Add", [scaled, tail_mask], [f"{prefix}_masked_scaled_raw"], f"{prefix}/TailMaskAdd"))
    masked = qdq(nodes, copier, f"{prefix}_masked_scaled_raw", f"{prefix}_masked_scaled_raw", f"{prefix}_masked_scaled")
    nodes.append(helper.make_node("Softmax", [masked], [f"{prefix}_probs_raw"], f"{prefix}/Softmax", axis=-1))
    probs = qdq(nodes, copier, f"{prefix}_probs_raw", f"{prefix}_probs_raw", f"{prefix}_probs")
    nodes.append(helper.make_node("MatMul", [probs, v_rep], [f"{prefix}_context_raw"], f"{prefix}/ContextMatMul"))
    context = qdq(nodes, copier, f"{prefix}_context_raw", f"{prefix}_context_raw", f"{prefix}_context")
    nodes.append(helper.make_node("Transpose", [context], [f"{prefix}_context_seq_raw"], f"{prefix}/ContextTranspose", perm=[0, 2, 1, 3]))
    context_seq = qdq(nodes, copier, f"{prefix}_context_seq_raw", f"{prefix}_context_raw", f"{prefix}_context_seq")
    context_shape = add_int64_init(copier, f"{prefix}_context_shape_chunk{chunk}", [1, chunk, HIDDEN])
    nodes.append(helper.make_node("Reshape", [context_seq, context_shape], [f"{prefix}_context_hidden"], f"{prefix}/ContextReshape"))
    out = add_linear(nodes, copier, f"{prefix}_context_hidden", f"{prefix}_o_proj_w_dq", f"{prefix}_o_proj")
    nodes.append(helper.make_node("Add", [x_name, out], [f"{layer_prefix}_attn_resid_raw"], f"{layer_prefix}/AttentionResidualAdd"))
    resid = qdq(nodes, copier, f"{layer_prefix}_attn_resid_raw", f"{layer_prefix}_attn_resid_raw", f"{layer_prefix}_attn_resid_raw_dq")
    y = add_mlp_after_resid(nodes, copier, resid, f"{layer_prefix}mlp", f"{layer_prefix}_y_raw", chunk)
    new_k = f"{prefix}_new_k"
    new_v = f"{prefix}_new_v"
    nodes.append(helper.make_node("Identity", [k_rope], [new_k], f"{prefix}/NewKOutput"))
    nodes.append(helper.make_node("Identity", [v_heads], [new_v], f"{prefix}/NewVOutput"))
    return y, present_k, present_v, new_k, new_v


def add_final_norm_and_logits(nodes, copier: InitCopier, x_name: str, chunk: int, include_logits: bool):
    final_norm = add_exact_lpnorm_rank3(nodes, copier, x_name, "final_norm", chunk)
    nodes.append(helper.make_node("Identity", [final_norm], ["final_norm_y"], "final_norm/KeepOutput"))
    if not include_logits:
        return "final_norm_y", None

    start = add_int64_init(copier, f"final_last_token_chunk{chunk}_starts", [chunk - 1])
    end = add_int64_init(copier, f"final_last_token_chunk{chunk}_ends", [chunk])
    axes = add_int64_init(copier, f"final_last_token_chunk{chunk}_axes", [1])
    steps = add_int64_init(copier, f"final_last_token_chunk{chunk}_steps", [1])
    nodes.append(helper.make_node("Slice", [final_norm, start, end, axes, steps], ["final_norm_last_raw"], "final_head/LastTokenSlice"))
    last = qdq(nodes, copier, "final_norm_last_raw", "final_norm_weighted", "final_norm_last_dq")
    flat_shape = add_int64_init(copier, f"final_head_shape_flat_chunk{chunk}", [1, HIDDEN])
    nodes.append(helper.make_node("Reshape", [last, flat_shape], ["final_head_flat"], "final_head/FlattenLast"))
    flat = qdq(nodes, copier, "final_head_flat", "final_norm_weighted", "final_head_flat_dq")
    lm_head = dq_const(nodes, copier, "lm_head_slice_w_dq")
    nodes.append(helper.make_node("MatMul", [flat, lm_head], ["lm_head_slice_mm"], "lm_head_slice/MatMul"))
    mm = qdq(nodes, copier, "lm_head_slice_mm", "lm_head_slice_mm", "lm_head_slice_mm_dq")
    logits_shape = add_int64_init(copier, f"final_logits_shape_last_chunk{chunk}", [1, 1, VOCAB])
    nodes.append(helper.make_node("Reshape", [mm, logits_shape], ["logits_QuantizeLinear_Input"], "final_head/LogitsReshapeLast"))
    logits = qdq(nodes, copier, "logits_QuantizeLinear_Input", "lm_head_slice_mm", "logits")
    return "final_norm_y", logits


def build_full_chunk_model(
    accepted_model: Path,
    output_model: Path,
    chunk: int,
    past_len: int,
    total_len: int,
    include_logits: bool,
    end_layer: int,
    cache_output_mode: str = "full",
):
    if past_len + chunk != total_len:
        raise ValueError(f"past_len + chunk must equal total_len, got {past_len}+{chunk}!={total_len}")
    if end_layer < 0 or end_layer > 15:
        raise ValueError(f"end_layer must be in [0,15], got {end_layer}")
    if cache_output_mode not in {"full", "new_only"}:
        raise ValueError(f"cache_output_mode must be 'full' or 'new_only', got {cache_output_mode!r}")

    copier = InitCopier(accepted_model)
    nodes = []
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, chunk, HIDDEN])]
    outputs = []

    for layer in CONV_LAYERS:
        if layer <= end_layer:
            inputs.append(helper.make_tensor_value_info(f"past_conv{layer}", TensorProto.FLOAT, [1, HIDDEN, PAST_CONV]))
    for layer in ATTN_LAYERS:
        if layer > end_layer:
            continue
        inputs.extend(
            [
                helper.make_tensor_value_info(f"past_k{layer}", TensorProto.FLOAT, [1, KV_HEADS, past_len, HEAD_DIM]),
                helper.make_tensor_value_info(f"past_v{layer}", TensorProto.FLOAT, [1, KV_HEADS, past_len, HEAD_DIM]),
                helper.make_tensor_value_info(f"l{layer}attn_q_rope_cos_dq", TensorProto.FLOAT, [1, HEADS, chunk, HEAD_DIM]),
                helper.make_tensor_value_info(f"l{layer}attn_q_rope_sin_dq", TensorProto.FLOAT, [1, HEADS, chunk, HEAD_DIM]),
                helper.make_tensor_value_info(f"l{layer}attn_k_rope_cos_dq", TensorProto.FLOAT, [1, KV_HEADS, chunk, HEAD_DIM]),
                helper.make_tensor_value_info(f"l{layer}attn_k_rope_sin_dq", TensorProto.FLOAT, [1, KV_HEADS, chunk, HEAD_DIM]),
                helper.make_tensor_value_info(f"l{layer}attn_tail_mask", TensorProto.FLOAT, [1, HEADS, chunk, total_len]),
            ]
        )

    hidden = qdq(nodes, copier, "x", "x", "x_dq")
    present_names = []
    for layer in range(end_layer + 1):
        if layer in CONV_LAYERS:
            hidden, present = add_conv_layer(nodes, copier, hidden, layer, chunk)
            present_names.append((present, [1, HIDDEN, PAST_CONV]))
        elif layer in ATTN_LAYERS:
            hidden, present_k, present_v, new_k, new_v = add_attention_layer(nodes, copier, hidden, layer, chunk, past_len, total_len)
            if cache_output_mode == "full":
                present_names.append((present_k, [1, KV_HEADS, total_len, HEAD_DIM]))
                present_names.append((present_v, [1, KV_HEADS, total_len, HEAD_DIM]))
            else:
                present_names.append((new_k, [1, KV_HEADS, chunk, HEAD_DIM]))
                present_names.append((new_v, [1, KV_HEADS, chunk, HEAD_DIM]))
        else:
            raise AssertionError(f"unclassified layer {layer}")

    nodes.append(helper.make_node("Identity", [hidden], ["hidden"], "full_chunk/KeepHidden"))
    if end_layer == 15:
        final_norm_y, logits = add_final_norm_and_logits(nodes, copier, hidden, chunk, include_logits)
        if include_logits:
            outputs.append(helper.make_tensor_value_info(logits, TensorProto.FLOAT, [1, 1, VOCAB]))
        outputs.append(helper.make_tensor_value_info(final_norm_y, TensorProto.FLOAT, [1, chunk, HIDDEN]))
    outputs.append(helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, chunk, HIDDEN]))
    for name, shape in present_names:
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))

    graph = helper.make_graph(nodes, output_model.stem, inputs, outputs, copier.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))
    return summarize_model(output_model)


def build_feeds(session, chunk: int, past_len: int, total_len: int, seed: int, mask_value: float, rope_theta: float):
    rng = np.random.default_rng(seed)
    feeds = {"x": rng.normal(0.0, 0.5, size=(1, chunk, HIDDEN)).astype(np.float32)}
    for layer in CONV_LAYERS:
        feeds[f"past_conv{layer}"] = rng.normal(0.0, 0.25, size=(1, HIDDEN, PAST_CONV)).astype(np.float32)
    mask = make_chunk_mask(chunk, past_len, total_len, mask_value).astype(np.float32)
    for layer in ATTN_LAYERS:
        feeds[f"past_k{layer}"] = rng.normal(0.0, 0.25, size=(1, KV_HEADS, past_len, HEAD_DIM)).astype(np.float32)
        feeds[f"past_v{layer}"] = rng.normal(0.0, 0.25, size=(1, KV_HEADS, past_len, HEAD_DIM)).astype(np.float32)
        q_cos, q_sin = make_rope(past_len, chunk, HEADS, rope_theta)
        k_cos, k_sin = make_rope(past_len, chunk, KV_HEADS, rope_theta)
        feeds[f"l{layer}attn_q_rope_cos_dq"] = q_cos
        feeds[f"l{layer}attn_q_rope_sin_dq"] = q_sin
        feeds[f"l{layer}attn_k_rope_cos_dq"] = k_cos
        feeds[f"l{layer}attn_k_rope_sin_dq"] = k_sin
        feeds[f"l{layer}attn_tail_mask"] = mask
    expected = {item.name for item in session.get_inputs()}
    missing = sorted(expected - set(feeds))
    if missing:
        raise KeyError(f"missing feeds: {missing}")
    return {name: feeds[name] for name in expected}


def run_cpu_shape_smoke(model_path: Path, chunk: int, past_len: int, total_len: int, seed: int, mask_value: float, rope_theta: float):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
    create_s = time.perf_counter() - t0
    feeds = build_feeds(sess, chunk, past_len, total_len, seed, mask_value, rope_theta)
    t1 = time.perf_counter()
    outputs = sess.run(None, feeds)
    run_s = time.perf_counter() - t1
    names = [item.name for item in sess.get_outputs()]
    shape_map = {name: list(value.shape) for name, value in zip(names, outputs)}
    finite = {name: bool(np.isfinite(value).all()) for name, value in zip(names, outputs)}
    return {
        "ok": bool(all(finite.values())),
        "session_create_s": create_s,
        "run_s": run_s,
        "output_shapes": shape_map,
        "all_outputs_finite": finite,
        "rss_after_mb": rss_mb(),
    }


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
        so.profile_file_prefix = str(log_dir / "qnn_full_chunk_profile")
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_provider_for_devices(qdevs, provider_options)
        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(model_path), sess_options=so, enable_fallback=False)
        case["session_create_s"] = time.perf_counter() - t0
        feeds = build_feeds(sess, chunk, past_len, total_len, seed, mask_value, rope_theta)
        t1 = time.perf_counter()
        outputs = sess.run(None, feeds)
        case["run_s"] = time.perf_counter() - t1
        names = [item.name for item in sess.get_outputs()]
        case["output_shapes"] = {name: list(np.asarray(value).shape) for name, value in zip(names, outputs)}
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
        "# P4 Path A2 Full Chunk Graph Probe",
        "",
        f"- mode: `{result.get('mode')}`",
        f"- chunk: `{result.get('chunk')}`",
        f"- past_len: `{result.get('past_len')}`",
        f"- total_len: `{result.get('total_len')}`",
        f"- include_logits: `{result.get('include_logits')}`",
        f"- end_layer: `{result.get('end_layer')}`",
        f"- cache_output_mode: `{result.get('cache_output_mode')}`",
        f"- v1_complete: `{result.get('v1_complete')}`",
        "",
    ]
    if "model" in result:
        model = result["model"]
        lines.extend(
            [
                "## Model",
                "",
                f"- path: `{model.get('path')}`",
                f"- size_bytes: `{model.get('size_bytes')}`",
                f"- initializer_count: `{model.get('initializer_count')}`",
                f"- node_counts: `{model.get('node_counts')}`",
                f"- inputs: `{len(model.get('inputs', []))}`",
                f"- outputs: `{len(model.get('outputs', []))}`",
                "",
            ]
        )
    if "cpu_shape_smoke" in result:
        smoke = result["cpu_shape_smoke"]
        lines.extend(
            [
                "## CPU Shape Smoke",
                "",
                f"- ok: `{smoke.get('ok')}`",
                f"- session_create_s: `{smoke.get('session_create_s')}`",
                f"- run_s: `{smoke.get('run_s')}`",
                f"- rss_after_mb: `{smoke.get('rss_after_mb')}`",
                f"- output_shapes: `{smoke.get('output_shapes')}`",
                "",
            ]
        )
    if "qnn_run" in result:
        qnn = result["qnn_run"]
        lines.extend(
            [
                "## QNN Run",
                "",
                f"- ok: `{qnn.get('ok')}`",
                f"- qnn_only: `{qnn.get('qnn_only')}`",
                f"- provider_counts: `{qnn.get('provider_counts')}`",
                f"- session_create_s: `{qnn.get('session_create_s')}`",
                f"- run_s: `{qnn.get('run_s')}`",
                f"- rss_after_mb: `{qnn.get('rss_after_mb')}`",
                f"- error_type: `{qnn.get('error_type')}`",
                f"- error: `{qnn.get('error')}`",
                "",
            ]
        )
    if "error" in result:
        lines.extend(["## Error", "", "```", result["error"], "```", ""])
    (log_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build a single full-model Path A2/N4b chunk graph from accepted QDQ constants.")
    parser.add_argument("--timestamp")
    parser.add_argument("--accepted-model", type=Path, default=DEFAULT_ACCEPTED_MODEL)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--total-len", type=int, default=DEFAULT_TOTAL_LEN)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--mask-value", type=float, default=DEFAULT_MASK_VALUE)
    parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
    parser.add_argument("--include-logits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--end-layer", type=int, default=15)
    parser.add_argument("--cache-output-mode", choices=["full", "new_only"], default="full")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--qnn-run", action="store_true")
    parser.add_argument("--existing-model", type=Path)
    args = parser.parse_args()

    if args.total_len < args.chunk:
        raise ValueError("--total-len must be >= --chunk")
    past_len = args.total_len - args.chunk
    timestamp = args.timestamp or time.strftime("%Y%m%d_%H%M%S_full_chunk")
    log_dir = args.log_dir or (args.log_root / f"p4_patha2_full_chunk_graph_{timestamp}")
    model_dir = log_dir / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.existing_model or (model_dir / f"patha2_full_chunk{args.chunk}_total{args.total_len}_qdq.onnx")
    result = {
        "mode": "p4_patha2_full_chunk_graph_probe",
        "timestamp": timestamp,
        "accepted_model": str(args.accepted_model),
        "log_dir": str(log_dir),
        "chunk": args.chunk,
        "past_len": past_len,
        "total_len": args.total_len,
        "include_logits": args.include_logits,
        "end_layer": args.end_layer,
        "cache_output_mode": args.cache_output_mode,
        "conv_layers": CONV_LAYERS,
        "attention_layers": ATTN_LAYERS,
        "v1_complete": False,
        "no_sudo_or_root": True,
    }
    try:
        if not args.skip_build:
            result["model"] = build_full_chunk_model(
                args.accepted_model,
                model_path,
                args.chunk,
                past_len,
                args.total_len,
                args.include_logits,
                args.end_layer,
                args.cache_output_mode,
            )
        else:
            result["model"] = summarize_model(model_path)
        if not args.skip_cpu:
            result["cpu_shape_smoke"] = run_cpu_shape_smoke(model_path, args.chunk, past_len, args.total_len, args.seed, args.mask_value, args.rope_theta)
        if args.qnn_run:
            result["qnn_run"] = run_qnn(model_path, log_dir, args.chunk, past_len, args.total_len, args.seed, args.mask_value, args.rope_theta)
    except Exception:
        result["error"] = traceback.format_exc()
    result_path = log_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(log_dir, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
