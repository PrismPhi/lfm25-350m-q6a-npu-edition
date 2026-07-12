#!/usr/bin/env python3
import argparse
import gc
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from tokenizers import Tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ROOT = Path(
    os.environ.get("LFM2_5_STATE_DIR")
    or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "lfm2.5-350m-q6a-qcs6490-qnn-npu"
)
DEFAULT_LOG_ROOT = STATE_ROOT / "logs"
DEFAULT_MODEL = STATE_ROOT / "contexts" / "decode" / "decode_epcontext.onnx"
DEFAULT_TOKENIZER = STATE_ROOT / "models" / "tokenizer" / "tokenizer.json"
DEFAULT_OFFICIAL_MODEL = STATE_ROOT / "models" / "host" / "rope_cache.npz"
HIDDEN = 1024
VOCAB = 65536
ROPE_RE = re.compile(r"^l(\d+)attn_([qk])_rope_(cos|sin)_dq$")
TAIL_MASK_SUFFIX = "attn_tail_mask"

SMOKE_PROMPTS = [
    ("jp_capital_qa", "質問: 日本の首都は？\n回答:"),
    ("jp_capital_prefix", "日本の首都は"),
    ("jp_momotaro_qa", "質問: 桃太郎の物語について短く教えてください。\n回答:"),
    ("jp_momotaro_prefix", "桃太郎は、"),
    ("en_capital_prefix", "The capital of Japan is"),
    ("en_momotaro", "Explain Momotaro in one short sentence."),
]

KEYWORDS = {
    "jp_capital_qa": ["東京", "首都", "日本", "Tokyo", "capital", "Japan"],
    "jp_capital_prefix": ["東京", "首都", "日本", "Tokyo", "capital", "Japan"],
    "jp_momotaro_qa": ["桃太郎", "鬼", "犬", "猿", "雉", "昔話", "Momotaro", "ogre", "dog", "monkey", "pheasant"],
    "jp_momotaro_prefix": ["桃太郎", "鬼", "犬", "猿", "雉", "昔話", "Momotaro", "ogre", "dog", "monkey", "pheasant"],
    "en_capital_prefix": ["Tokyo", "capital", "Japan", "東京", "首都", "日本"],
    "en_momotaro": ["Momotaro", "ogre", "dog", "monkey", "pheasant", "桃太郎", "鬼", "犬", "猿", "雉"],
}


def elem_dtype(elem_type: int):
    return {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.UINT8: np.uint8,
    }.get(elem_type, np.float32)


def dim_value(dim):
    return dim.dim_value if dim.dim_value > 0 else 1


def load_io_shapes(model_path: Path):
    model = onnx.load(model_path, load_external_data=False)
    inputs = {}
    outputs = []
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        inputs[inp.name] = {
            "shape": [dim_value(d) for d in tt.shape.dim],
            "dtype": elem_dtype(tt.elem_type),
            "elem_type": int(tt.elem_type),
        }
    for out in model.graph.output:
        tt = out.type.tensor_type
        outputs.append({"name": out.name, "shape": [dim_value(d) for d in tt.shape.dim], "elem_type": int(tt.elem_type)})
    return inputs, outputs


def value_info_shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(dim_value(dim))
    return dims


def discover_rope_inputs(model_path: Path):
    model = onnx.load(model_path, load_external_data=False)
    entries = {}
    for inp in model.graph.input:
        match = ROPE_RE.fullmatch(inp.name)
        if not match:
            continue
        entries[inp.name] = {
            "layer": int(match.group(1)),
            "q_or_k": match.group(2),
            "kind": match.group(3),
            "shape": value_info_shape(inp),
        }
    return entries


def load_rope_cache(source: Path):
    source = Path(source)
    if source.suffix == ".npz":
        with np.load(source) as arrays:
            missing = sorted({"cos_cache", "sin_cache"} - set(arrays.files))
            if missing:
                raise RuntimeError(f"RoPE cache arrays missing: {missing}")
            return (
                np.asarray(arrays["cos_cache"], dtype=np.float32),
                np.asarray(arrays["sin_cache"], dtype=np.float32),
            )
    model = onnx.load(str(source), load_external_data=True)
    arrays = {}
    for init in model.graph.initializer:
        if init.name in {"cos_cache", "sin_cache"}:
            arrays[init.name] = numpy_helper.to_array(init).astype(np.float32)
    missing = sorted({"cos_cache", "sin_cache"} - set(arrays))
    if missing:
        raise RuntimeError(f"official RoPE cache initializers missing: {missing}")
    return arrays["cos_cache"], arrays["sin_cache"]


def expand_rope(row32, shape):
    dim = int(shape[-1])
    if dim == row32.shape[0]:
        row = row32
    elif dim == row32.shape[0] * 2:
        row = np.concatenate([row32, row32]).astype(np.float32)
    else:
        raise RuntimeError(f"unsupported RoPE dim {dim} for cache row {row32.shape}")
    return np.broadcast_to(row.reshape(1, 1, 1, dim), tuple(shape)).copy()


def apply_rope_feed(feed, position: int, rope_context):
    if not rope_context:
        return []
    cos_cache = rope_context["cos_cache"]
    sin_cache = rope_context["sin_cache"]
    pos = min(int(position), int(cos_cache.shape[0]) - 1)
    cos32 = np.asarray(cos_cache[pos], dtype=np.float32).reshape(-1)
    sin32 = np.asarray(sin_cache[pos], dtype=np.float32).reshape(-1)
    applied = []
    for name, item in rope_context["rope_inputs"].items():
        row = cos32 if item["kind"] == "cos" else sin32
        feed[name] = expand_rope(row, item["shape"])
        applied.append(name)
    return applied


def make_tail_mask(shape, position: int, mask_value: float):
    dims = tuple(int(x) for x in shape)
    window = int(dims[-1])
    valid = min(int(position), window - 1) + 1
    mask = np.full(dims, float(mask_value), dtype=np.float32)
    mask[..., window - valid :] = 0.0
    return mask


def tail_mask_input_names(mapping):
    return sorted([name for name in mapping.keys() if name.endswith(TAIL_MASK_SUFFIX)])


def apply_tail_mask_feed(feed, input_specs, position: int, mask_value: float):
    applied = []
    for name in tail_mask_input_names(input_specs):
        feed[name] = make_tail_mask(input_specs[name]["shape"], position, mask_value)
        applied.append(name)
    return applied


def apply_position_feeds(feed, input_specs, position: int, args, rope_context):
    rope_applied = [] if args.disable_rope_feed else apply_rope_feed(feed, position, rope_context)
    mask_applied = [] if args.disable_tail_mask_feed else apply_tail_mask_feed(feed, input_specs, position, args.tail_mask_value)
    return rope_applied, mask_applied


def build_session(model_path: Path, log_dir: Path, log_severity: int, keep_profile: bool, provider_options_extra=None):
    import onnxruntime_qnn as oq

    os.environ["ADSP_LIBRARY_PATH"] = f"{Path(oq.get_library_path()).parent};/usr/lib/dsp/cdsp;/usr/lib/dsp/adsp;/dsp"
    try:
        ort.register_execution_provider_library(oq.get_ep_name(), oq.get_library_path())
        register_status = "registered"
    except Exception as exc:
        register_status = f"register_exception:{exc!r}"
    so = ort.SessionOptions()
    so.log_severity_level = int(log_severity)
    so.enable_profiling = True
    so.profile_file_prefix = str(log_dir / "practical_hybrid_v0_profile")
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {"backend_path": str(oq.get_qnn_htp_path())}
    if provider_options_extra:
        provider_options.update({str(k): str(v) for k, v in provider_options_extra.items()})
    qdevs = [d for d in ort.get_ep_devices() if getattr(d, "ep_name", None) == oq.get_ep_name()]
    if not qdevs:
        raise RuntimeError("No QNN OrtEpDevice after provider registration")
    so.add_provider_for_devices(qdevs, provider_options)
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model_path), sess_options=so, enable_fallback=False)
    create_s = time.perf_counter() - t0
    return sess, register_status, provider_options, [getattr(d, "ep_name", repr(d)) for d in qdevs], create_s


def parse_profile(path):
    counts = {}
    events = []
    if not path or not Path(path).exists():
        return counts, events
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return counts, events
    for ev in data:
        args = ev.get("args") or {}
        provider = args.get("provider")
        if provider:
            counts[provider] = counts.get(provider, 0) + 1
            events.append({
                "provider": provider,
                "op_name": args.get("op_name"),
                "event_name": ev.get("name"),
                "dur_us": int(ev.get("dur") or 0),
            })
    return counts, events


def load_embedding_state():
    import probe_split_chunk_embedding as emb

    values = emb.source_values()
    weight = np.asarray(values["lm_head.weight"][:VOCAB, :HIDDEN], dtype=np.float32)
    norm_weight = np.asarray(values["model.embedding_norm.weight"][:HIDDEN], dtype=np.float32)
    if weight.shape != (VOCAB, HIDDEN):
        raise RuntimeError(f"unexpected embedding weight shape: {weight.shape}")
    if norm_weight.shape != (HIDDEN,):
        raise RuntimeError(f"unexpected embedding norm shape: {norm_weight.shape}")
    return {"weight": weight, "norm_weight": norm_weight}


def load_official_raw_embedding_state(official_model: Path):
    model = onnx.load(str(official_model), load_external_data=True)
    for init in model.graph.initializer:
        if init.name == "model.embed_tokens.weight":
            weight = numpy_helper.to_array(init).astype(np.float32)
            if weight.shape[1] != HIDDEN:
                raise RuntimeError(f"unexpected official embedding shape: {weight.shape}")
            return {"official_raw": weight}
    raise RuntimeError("official model.embed_tokens.weight initializer not found")


def load_embedding_state_for_backend(backend: str, official_model: Path):
    if backend == "official_raw":
        return load_official_raw_embedding_state(official_model)
    if backend == "practical_rmsnorm":
        return load_embedding_state()
    raise ValueError(f"unknown embedding backend: {backend}")


def embedding_rmsnorm(row, norm_weight):
    x = np.asarray(row, dtype=np.float32)
    denom = np.sqrt(np.mean(x * x, dtype=np.float32) + np.float32(1.0e-5))
    return (x / denom * norm_weight).astype(np.float32, copy=False)


def make_initial_feed(input_specs):
    feed = {}
    for name, spec in input_specs.items():
        if name == "x":
            feed[name] = np.zeros((1, 1, HIDDEN), dtype=np.float32)
        else:
            feed[name] = np.zeros(spec["shape"], dtype=spec["dtype"])
    return feed


def set_hidden(feed, embedding_state, token_id: int):
    if "official_raw" in embedding_state:
        embedding_weight = embedding_state["official_raw"]
        if token_id < 0 or token_id >= embedding_weight.shape[0]:
            raise ValueError(f"token id {token_id} outside official embedding range")
        feed["x"] = np.asarray(embedding_weight[token_id], dtype=np.float32).reshape(1, 1, HIDDEN)
        return
    embedding_weight = embedding_state["weight"]
    if token_id < 0 or token_id >= embedding_weight.shape[0]:
        raise ValueError(f"token id {token_id} outside embedding range")
    hidden = embedding_rmsnorm(embedding_weight[token_id], embedding_state["norm_weight"])
    feed["x"] = hidden.reshape(1, 1, HIDDEN)


def update_cache_from_outputs(feed, output_names, outputs):
    by_name = {name: np.asarray(value) for name, value in zip(output_names, outputs)}
    for out_name, value in by_name.items():
        conv_match = re.fullmatch(r"l(\d+)_present_conv", out_name)
        k_match = re.fullmatch(r"l(\d+)attn_present_k", out_name)
        v_match = re.fullmatch(r"l(\d+)attn_present_v", out_name)
        if conv_match:
            key = "past_conv" + conv_match.group(1)
            if key in feed:
                feed[key] = value.astype(np.float32, copy=False)
        elif k_match:
            key = "past_k" + k_match.group(1)
            if key in feed:
                feed[key] = value[:, :, -feed[key].shape[2]:, :].astype(np.float32, copy=False)
        elif v_match:
            key = "past_v" + v_match.group(1)
            if key in feed:
                feed[key] = value[:, :, -feed[key].shape[2]:, :].astype(np.float32, copy=False)


def cache_stats(feed, logical_cache_length: int):
    groups = {
        "past_conv": [v for k, v in feed.items() if k.startswith("past_conv")],
        "past_k": [v for k, v in feed.items() if k.startswith("past_k")],
        "past_v": [v for k, v in feed.items() if k.startswith("past_v")],
    }
    out = {"logical_cache_length": int(logical_cache_length), "physical_attention_window": 127}
    for name, arrays in groups.items():
        total_abs = 0.0
        total_sq = 0.0
        nonzero = 0
        elems = 0
        shapes = []
        for arr in arrays:
            a = np.asarray(arr)
            shapes.append(list(a.shape))
            total_abs += float(np.sum(np.abs(a), dtype=np.float64))
            total_sq += float(np.sum(a.astype(np.float64) * a.astype(np.float64)))
            nonzero += int(np.count_nonzero(np.abs(a) > 1.0e-8))
            elems += int(a.size)
        out[name] = {
            "arrays": len(arrays),
            "elements": elems,
            "shapes": shapes[:3],
            "nonzero_gt_1e-8": nonzero,
            "sum_abs": total_abs,
            "l2": float(total_sq ** 0.5),
        }
    return out


def top_k(logits, k, tokenizer_vocab_size=None):
    vec = np.asarray(logits, dtype=np.float32).reshape(-1)
    limit = min(vec.shape[0], tokenizer_vocab_size or vec.shape[0])
    k = min(int(k), limit)
    idx = np.argpartition(-vec[:limit], k - 1)[:k]
    idx = idx[np.argsort(-vec[idx])]
    return [{"token_id": int(i), "logit": float(vec[i])} for i in idx]


def softmax_stable(values):
    x = np.asarray(values, dtype=np.float64)
    x = x - np.max(x)
    exp = np.exp(x)
    denom = exp.sum()
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(x) / len(x)
    return exp / denom


def choose_next(logits, tokenizer_vocab_size, args, rng):
    candidates = top_k(logits, max(args.top_k, 1), tokenizer_vocab_size)
    if args.greedy or args.temperature <= 0:
        return int(candidates[0]["token_id"]), candidates, "greedy"
    ids = np.asarray([x["token_id"] for x in candidates], dtype=np.int64)
    vals = np.asarray([x["logit"] for x in candidates], dtype=np.float64) / max(float(args.temperature), 1.0e-6)
    probs = softmax_stable(vals)
    if args.top_p < 1.0:
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        keep = order[cumulative <= args.top_p]
        if keep.size == 0:
            keep = order[:1]
        probs2 = probs[keep]
        probs2 = probs2 / probs2.sum()
        selected = int(rng.choice(ids[keep], p=probs2))
    else:
        selected = int(rng.choice(ids, p=probs))
    return selected, candidates, "sample"


def speed_stats(values):
    vals = [float(v) for v in values]
    total = sum(vals)
    return {
        "count": len(vals),
        "total_s": total,
        "min_s": min(vals) if vals else None,
        "avg_s": statistics.fmean(vals) if vals else None,
        "max_s": max(vals) if vals else None,
        "tok_per_s": (len(vals) / total) if total > 0 else None,
    }


def stop_token_ids(tokenizer: Tokenizer, explicit):
    ids = set(int(x) for x in explicit)
    for token in ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"]:
        value = tokenizer.token_to_id(token)
        if value is not None:
            ids.add(int(value))
    return sorted(ids)


def official_stop_token_ids(tokenizer, explicit):
    ids = set(int(x) for x in explicit)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        if isinstance(eos_token_id, (list, tuple, set)):
            ids.update(int(x) for x in eos_token_id if x is not None)
        else:
            ids.add(int(eos_token_id))
    token_to_id = getattr(tokenizer, "token_to_id", None)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(token_to_id):
        for token in ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"]:
            value = token_to_id(token)
            if value is not None:
                ids.add(int(value))
    elif callable(convert_tokens_to_ids):
        for token in ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"]:
            value = convert_tokens_to_ids(token)
            if value is not None and value != getattr(tokenizer, "unk_token_id", None):
                ids.add(int(value))
    return sorted(ids)


def has_mojibake(text: str):
    return any(ch in text for ch in ["�", "繧", "縺", "荳", "譁"])


def related_keywords(name, generated_text):
    keys = KEYWORDS.get(name, [])
    hits = [kw for kw in keys if kw and kw.lower() in generated_text.lower()]
    return hits


def collapse_quality(text: str):
    bad_patterns = ["::::", "....", "ははは", "is is", "the the"]
    bad_hits = [pat for pat in bad_patterns if pat.lower() in text.lower()]
    tokens = re.findall(r"\S+", text)
    repeated_tail = len(tokens) >= 6 and len(set(tokens[-6:])) <= 2
    return {
        "readable_nonempty": bool(isinstance(text, str) and text.strip()),
        "bad_pattern_hits": bad_hits,
        "repetition_suspected": bool(repeated_tail),
        "collapse_suspected": bool(bad_hits or repeated_tail),
    }


def load_official_common(official_model: Path):
    scripts_dir = official_model.parents[2] / "scripts"
    for path in [scripts_dir]:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from lfm25_onnx_common import build_feed, encode_prompt, extract_logits, load_config, load_tokenizer

    return build_feed, encode_prompt, extract_logits, load_config, load_tokenizer


def update_official_feed_for_next(sess, output_names, old_feed, outputs, next_id):
    by_name = {name: value for name, value in zip(output_names, outputs)}
    past_total = None
    for key, value in by_name.items():
        if key.startswith("present.") and key.endswith(".key"):
            past_total = int(value.shape[2])
            break
    if past_total is None:
        past_total = int(old_feed["attention_mask"].shape[1])

    feed = {}
    for inp in sess.get_inputs():
        name = inp.name
        if name == "input_ids":
            feed[name] = np.asarray([[next_id]], dtype=np.int64)
        elif name == "attention_mask":
            feed[name] = np.ones((1, past_total + 1), dtype=old_feed[name].dtype)
        elif name == "num_logits_to_keep":
            feed[name] = old_feed[name]
        elif name == "position_ids":
            feed[name] = np.asarray([[past_total]], dtype=old_feed[name].dtype)
        elif name == "cache_position":
            feed[name] = np.asarray([past_total], dtype=old_feed[name].dtype)
        elif name.startswith("past_conv."):
            feed[name] = by_name[name.replace("past_conv.", "present_conv.")].astype(np.float32)
        elif name.startswith("past_key_values."):
            parts = name.split(".")
            layer = parts[1]
            kind = parts[-1]
            feed[name] = by_name[f"present.{layer}.{kind}"].astype(np.float32)
        else:
            feed[name] = old_feed[name]
    return feed


def run_official_cpu_reference(prompt_items, official_model: Path, max_new_tokens: int, stop_ids_explicit):
    build_feed, encode_prompt, extract_logits, load_config, load_tokenizer = load_official_common(official_model)
    config = load_config(official_model)
    tokenizer = load_tokenizer(official_model)
    create_t0 = time.perf_counter()
    sess = ort.InferenceSession(str(official_model), providers=["CPUExecutionProvider"])
    session_create_s = time.perf_counter() - create_t0
    output_names = [out.name for out in sess.get_outputs()]
    stop_ids = set(official_stop_token_ids(tokenizer, stop_ids_explicit))
    refs = {}
    vocab_size_fn = getattr(tokenizer, "get_vocab_size", None)
    for name, prompt in prompt_items:
        encoded = encode_prompt(tokenizer, prompt)
        feed = build_feed(sess, encoded, config)
        generated = []
        steps = []
        run_times = []
        stopped = False
        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            outputs = sess.run(None, feed)
            run_times.append(time.perf_counter() - t0)
            _, logits = extract_logits(outputs, output_names)
            vec = np.asarray(logits[0, -1, :], dtype=np.float32)
            vocab_size = int(vocab_size_fn()) if callable(vocab_size_fn) else int(vec.shape[0])
            candidates = top_k(vec, 10, vocab_size)
            next_id = int(candidates[0]["token_id"])
            generated.append(next_id)
            steps.append({"step": step, "selected_token_id": next_id, "top_k_before": candidates})
            if next_id in stop_ids:
                stopped = True
                break
            feed = update_official_feed_for_next(sess, output_names, feed, outputs, next_id)
        generated_text = tokenizer.decode(generated)
        refs[name] = {
            "cpu_q8_reference_prompt_token_ids": [int(x) for x in encoded["input_ids"][0].tolist()],
            "cpu_q8_reference_generated_token_ids": [int(x) for x in generated],
            "cpu_q8_reference_generated_text": generated_text,
            "cpu_q8_reference_generated_text_repr": repr(generated_text),
            "cpu_q8_reference_stopped_on_eos_or_stop": stopped,
            "cpu_q8_reference_decode_run_s": run_times,
            "cpu_q8_reference_decode_speed": speed_stats(run_times),
            "cpu_q8_reference_decode_steps_head": steps[:4],
            "cpu_q8_reference_related_keyword_hits": related_keywords(name, generated_text),
        }
    del sess
    return {
        "cpu_q8_reference_model": str(official_model),
        "cpu_q8_reference_provider": "CPUExecutionProvider",
        "cpu_q8_reference_session_create_s": session_create_s,
        "items": refs,
    }


def attach_reference_and_score(prompt_results, reference_bundle):
    reference_items = (reference_bundle or {}).get("items") or {}
    reference_related = 0
    hybrid_related_when_reference_related = 0
    collapse_count = 0
    for item in prompt_results:
        ref = reference_items.get(item.get("name"), {})
        item.update(ref)
        ref_text = ref.get("cpu_q8_reference_generated_text", "")
        hybrid_text = item.get("generated_text", "")
        ref_hits = related_keywords(item.get("name"), ref_text)
        hybrid_hits = related_keywords(item.get("name"), hybrid_text)
        hybrid_quality = collapse_quality(hybrid_text)
        ref_quality = collapse_quality(ref_text)
        if ref_hits:
            reference_related += 1
            if hybrid_hits:
                hybrid_related_when_reference_related += 1
        if hybrid_quality["collapse_suspected"]:
            collapse_count += 1
        item["quality_oracle"] = {
            "cpu_q8_reference_related_keyword_hits": ref_hits,
            "hybrid_related_keyword_hits": hybrid_hits,
            "cpu_q8_reference_quality": ref_quality,
            "hybrid_quality": hybrid_quality,
        }
        item["parity_status"] = (
            "reference_na"
            if not ref_hits
            else ("same_subject_pass" if hybrid_hits and not hybrid_quality["collapse_suspected"] else "same_subject_fail")
        )
    ratio = (hybrid_related_when_reference_related / reference_related) if reference_related else None
    return {
        "cpu_q8_reference_related_count": reference_related,
        "hybrid_related_when_reference_related": hybrid_related_when_reference_related,
        "hybrid_reference_related_ratio": ratio,
        "hybrid_collapse_count": collapse_count,
        "semantic_parity_pass": bool(reference_related and ratio is not None and ratio >= 0.8 and collapse_count == 0),
    }


def run_one_prompt(name, prompt, sess, output_names, input_specs, embedding_state, tokenizer, args, rng, rope_context):
    feed = make_initial_feed(input_specs)
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise RuntimeError(f"tokenizer returned no prompt ids for {name}")
    if len(ids) > args.max_prompt_tokens:
        ids = ids[-args.max_prompt_tokens :]
    token_vocab_size = tokenizer.get_vocab_size()
    result = {
        "name": name,
        "prompt": prompt,
        "prompt_token_ids": [int(x) for x in ids],
        "prompt_token_count": len(ids),
        "prompt_detokenized": tokenizer.decode([int(x) for x in ids if int(x) < token_vocab_size]),
        "prefill_run_s": [],
        "decode_run_s": [],
        "generated_token_ids": [],
        "decode_steps": [],
        "rope_feed_input_count": len((rope_context or {}).get("rope_inputs", {})),
        "tail_mask_input_names": tail_mask_input_names(input_specs),
    }
    logical_cache_length = 0
    last_logits = None
    first_token_wall_t0 = time.perf_counter()
    for token_id in ids:
        set_hidden(feed, embedding_state, int(token_id))
        apply_position_feeds(feed, input_specs, logical_cache_length, args, rope_context)
        t0 = time.perf_counter()
        outputs = sess.run(None, feed)
        result["prefill_run_s"].append(time.perf_counter() - t0)
        logical_cache_length += 1
        last_logits = np.asarray(outputs[0])
        update_cache_from_outputs(feed, output_names, outputs)
    result["ttft_s_excluding_session_create"] = float(time.perf_counter() - first_token_wall_t0)
    result["prefill_top_k"] = top_k(last_logits, args.top_k, token_vocab_size)
    result["cache_stats_after_prefill"] = cache_stats(feed, logical_cache_length)

    stop_ids = set(stop_token_ids(tokenizer, args.stop_token_id))
    stopped = False
    for step in range(args.max_new_tokens):
        next_id, candidates, policy = choose_next(last_logits, token_vocab_size, args, rng)
        result["generated_token_ids"].append(int(next_id))
        result["decode_steps"].append({"step": step, "selected_token_id": int(next_id), "selection_policy": policy, "top_k_before": candidates})
        if next_id in stop_ids:
            stopped = True
            break
        set_hidden(feed, embedding_state, next_id)
        apply_position_feeds(feed, input_specs, logical_cache_length, args, rope_context)
        t0 = time.perf_counter()
        outputs = sess.run(None, feed)
        result["decode_run_s"].append(time.perf_counter() - t0)
        logical_cache_length += 1
        last_logits = np.asarray(outputs[0])
        update_cache_from_outputs(feed, output_names, outputs)

    generated_text = tokenizer.decode([int(x) for x in result["generated_token_ids"] if int(x) < token_vocab_size])
    result.update({
        "generated_text": generated_text,
        "generated_text_repr": repr(generated_text),
        "stopped_on_eos_or_stop": stopped,
        "prefill_speed": speed_stats(result["prefill_run_s"]),
        "decode_speed": speed_stats(result["decode_run_s"]),
        "final_top_k": top_k(last_logits, args.top_k, token_vocab_size),
        "cache_stats_after_decode": cache_stats(feed, logical_cache_length),
        "logical_cache_length_after_prefill": len(ids),
        "logical_cache_length_after_decode": logical_cache_length,
        "expected_logical_cache_length_after_decode": len(ids) + len(result["decode_run_s"]),
        "detokenize_ok": isinstance(generated_text, str) and len(generated_text) > 0 and not has_mojibake(generated_text),
        "mojibake_detected": has_mojibake(generated_text),
    })
    hits = related_keywords(name, generated_text)
    result["related_keyword_hits"] = hits
    result["prompt_related_output"] = bool(hits)
    return result


def write_summary(log_dir: Path, result: dict):
    lines = [
        f"# Practical Hybrid Prompt Runner V0 {result['timestamp']}",
        "",
        "## Status",
        "",
        f"- ok: `{result.get('ok')}`",
        f"- runtime_ok: `{result.get('runtime_ok')}`",
        f"- quality_minimum_pass: `{result.get('quality_minimum_pass')}`",
        f"- practical_npu_runtime: `{result.get('practical_npu_runtime')}`",
        f"- all_npu_proof: `{result.get('all_npu_proof')}`",
        f"- embedding_backend: `{result.get('embedding_backend')}`",
        f"- model_compute_backend: `{result.get('model_compute_backend')}`",
        f"- fallback_disabled: `{result.get('fallback_disabled')}`",
        f"- provider_counts: `{result.get('provider_counts')}`",
        f"- qnn_only_blocks_head: `{result.get('qnn_only_blocks_head')}`",
        f"- cpu_q8_reference_model: `{result.get('cpu_q8_reference_model')}`",
        f"- cpu_q8_reference_provider: `{result.get('cpu_q8_reference_provider')}`",
        f"- cpu_q8_reference_related_count: `{result.get('cpu_q8_reference_related_count')}`",
        f"- hybrid_related_when_reference_related: `{result.get('hybrid_related_when_reference_related')}`",
        f"- hybrid_reference_related_ratio: `{result.get('hybrid_reference_related_ratio')}`",
        f"- hybrid_collapse_count: `{result.get('hybrid_collapse_count')}`",
        f"- semantic_parity_pass: `{result.get('semantic_parity_pass')}`",
        f"- profile_deleted_after_parse: `{result.get('profile_deleted_after_parse')}`",
        f"- rope_feed: `{result.get('rope_feed')}`",
        f"- tail_mask_feed: `{result.get('tail_mask_feed')}`",
        f"- model: `{result.get('model')}`",
        f"- tokenizer: `{result.get('tokenizer')}`",
        f"- log_dir: `{result.get('log_dir')}`",
        "",
        "## Prompt Results",
        "",
    ]
    for item in result.get("prompt_results", []):
        lines += [
            f"### {item.get('name')}",
            "",
            f"- prompt: `{item.get('prompt')}`",
            f"- prompt_token_ids: `{item.get('prompt_token_ids')}`",
            f"- cpu_q8_reference_prompt_token_ids: `{item.get('cpu_q8_reference_prompt_token_ids')}`",
            f"- cpu_q8_reference_generated_token_ids: `{item.get('cpu_q8_reference_generated_token_ids')}`",
            f"- cpu_q8_reference_generated_text: `{item.get('cpu_q8_reference_generated_text')}`",
            f"- generated_token_ids: `{item.get('generated_token_ids')}`",
            f"- generated_text: `{item.get('generated_text')}`",
            f"- parity_status: `{item.get('parity_status')}`",
            f"- quality_oracle: `{item.get('quality_oracle')}`",
            f"- detokenize_ok: `{item.get('detokenize_ok')}`",
            f"- prompt_related_output: `{item.get('prompt_related_output')}`",
            f"- related_keyword_hits: `{item.get('related_keyword_hits')}`",
            f"- cpu_q8_reference_decode_speed: `{item.get('cpu_q8_reference_decode_speed')}`",
            f"- prefill_speed: `{item.get('prefill_speed')}`",
            f"- decode_speed: `{item.get('decode_speed')}`",
            f"- ttft_s_excluding_session_create: `{item.get('ttft_s_excluding_session_create')}`",
            f"- logical_cache_length_after_prefill: `{item.get('logical_cache_length_after_prefill')}`",
            f"- logical_cache_length_after_decode: `{item.get('logical_cache_length_after_decode')}`",
            f"- prefill_top_k: `{item.get('prefill_top_k')}`",
            f"- final_top_k: `{item.get('final_top_k')}`",
            "",
        ]
    lines += [
        "## Files",
        "",
        f"- result_json: `{log_dir / 'result.json'}`",
        f"- summary_md: `{log_dir / 'summary.md'}`",
    ]
    if result.get("error"):
        lines += ["", "## Error", "", f"- type: `{result.get('error_type')}`", f"- message: `{result.get('error')}`"]
    (log_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Practical NPU Prompt Runner V0: CPU embedding lookup + QNN/HTP LFM blocks/head.")
    parser.add_argument("--prompt")
    parser.add_argument("--smoke-tests", action="store_true")
    parser.add_argument("--timestamp")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--official-model", type=Path, default=DEFAULT_OFFICIAL_MODEL, help="official model used only for RoPE cache inputs when the practical model exposes RoPE inputs")
    parser.add_argument("--embedding-backend", choices=["practical_rmsnorm", "official_raw"], default="practical_rmsnorm")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-token-id", action="append", type=int, default=[])
    parser.add_argument("--tail-mask-value", type=float, default=-64.0)
    parser.add_argument("--disable-rope-feed", action="store_true")
    parser.add_argument("--disable-tail-mask-feed", action="store_true")
    parser.add_argument("--log-severity", type=int, default=3)
    parser.add_argument("--keep-profile", action="store_true")
    parser.add_argument("--enable-htp-fp16-precision", action="store_true")
    parser.add_argument("--disable-offload-graph-io-quantization", action="store_true")
    parser.add_argument("--qnn-provider-option", action="append", default=[], help="extra QNN provider option as key=value")
    args = parser.parse_args()
    if not args.smoke_tests and not args.prompt:
        raise SystemExit("pass --prompt TEXT or --smoke-tests")
    if args.greedy is False and args.temperature <= 0:
        args.greedy = True

    timestamp = args.timestamp or time.strftime("%Y%m%d_%H%M%S_practical_hybrid_v0")
    log_dir = args.log_root / f"practical_prompt_runner_v0_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": timestamp,
        "command": " ".join(__import__("sys").argv),
        "mode": "practical-hybrid",
        "all_npu_proof": False,
        "practical_npu_runtime": True,
        "embedding_backend": args.embedding_backend,
        "model_compute_backend": "qnn_htp",
        "cpu_allowed": ["tokenizer", "detokenizer", "embedding_lookup", "embedding_rmsnorm", "sampling", "cache_bookkeeping", "position_feed", "rope_constant_feed", "tail_mask_feed", "logging"],
        "npu_required": ["LFM_blocks", "attention", "MLP", "RMSNorm", "final_norm", "lm_head_logits", "cache_update_compute"],
        "fallback_disabled": True,
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "official_model_for_rope": str(args.official_model),
        "official_model_for_embedding": str(args.official_model) if args.embedding_backend == "official_raw" else None,
        "log_dir": str(log_dir),
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "greedy": args.greedy,
        "tail_mask_value": args.tail_mask_value,
        "disable_rope_feed": bool(args.disable_rope_feed),
        "disable_tail_mask_feed": bool(args.disable_tail_mask_feed),
        "enable_htp_fp16_precision": bool(args.enable_htp_fp16_precision),
        "disable_offload_graph_io_quantization": bool(args.disable_offload_graph_io_quantization),
        "qnn_provider_option_args": list(args.qnn_provider_option),
        "ok": False,
        "runtime_ok": False,
        "quality_minimum_pass": False,
    }
    profile = None
    try:
        input_specs, output_specs = load_io_shapes(args.model)
        if "x" not in input_specs:
            raise RuntimeError("hidden-input model must expose x input")
        rope_inputs = discover_rope_inputs(args.model)
        rope_context = None
        if rope_inputs and not args.disable_rope_feed:
            cos_cache, sin_cache = load_rope_cache(args.official_model)
            rope_context = {"rope_inputs": rope_inputs, "cos_cache": cos_cache, "sin_cache": sin_cache}
        result["input_shapes"] = {k: v["shape"] for k, v in input_specs.items()}
        result["output_shapes_declared"] = output_specs
        result["rope_feed"] = {
            "detected_input_count": len(rope_inputs),
            "enabled": bool(rope_inputs and not args.disable_rope_feed),
            "input_names_head": sorted(rope_inputs.keys())[:12],
            "cos_cache_shape": list(rope_context["cos_cache"].shape) if rope_context else None,
            "sin_cache_shape": list(rope_context["sin_cache"].shape) if rope_context else None,
        }
        result["tail_mask_feed"] = {
            "detected_input_names": tail_mask_input_names(input_specs),
            "enabled": bool(tail_mask_input_names(input_specs) and not args.disable_tail_mask_feed),
            "mask_value": float(args.tail_mask_value),
        }
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        embedding_state = load_embedding_state_for_backend(args.embedding_backend, args.official_model)
        provider_options_extra = {}
        if args.enable_htp_fp16_precision:
            provider_options_extra["enable_htp_fp16_precision"] = "1"
        if args.disable_offload_graph_io_quantization:
            provider_options_extra["offload_graph_io_quantization"] = "0"
        for item in args.qnn_provider_option:
            if "=" not in item:
                raise ValueError(f"invalid --qnn-provider-option {item!r}; expected key=value")
            key, value = item.split("=", 1)
            provider_options_extra[key] = value
        sess, register_status, provider_options, qnn_devices, create_s = build_session(
            args.model,
            log_dir,
            args.log_severity,
            args.keep_profile,
            provider_options_extra,
        )
        output_names = [o.name for o in sess.get_outputs()]
        result.update({
            "register_status": register_status,
            "provider_options": provider_options,
            "qnn_devices": qnn_devices,
            "session_create_s": create_s,
            "session_providers": list(sess.get_providers()),
            "available_providers": list(ort.get_available_providers()),
            "output_names": output_names,
        })
        prompt_items = SMOKE_PROMPTS if args.smoke_tests else [("prompt", args.prompt)]
        rng = np.random.default_rng(args.seed)
        prompt_results = []
        for name, prompt in prompt_items:
            prompt_results.append(run_one_prompt(name, prompt, sess, output_names, input_specs, embedding_state, tokenizer, args, rng, rope_context))
        profile = sess.end_profiling()
        provider_counts, provider_events = parse_profile(profile)
        profile_deleted = False
        if not args.keep_profile:
            try:
                Path(profile).unlink(missing_ok=True)
                profile_deleted = True
            except Exception:
                profile_deleted = False
        del sess
        gc.collect()
        reference_bundle = None
        parity = {
            "cpu_q8_reference_related_count": 0,
            "hybrid_related_when_reference_related": 0,
            "hybrid_reference_related_ratio": None,
            "hybrid_collapse_count": 0,
            "semantic_parity_pass": False,
        }
        try:
            reference_bundle = run_official_cpu_reference(prompt_items, args.official_model, args.max_new_tokens, args.stop_token_id)
            parity = attach_reference_and_score(prompt_results, reference_bundle)
            result.update({
                "cpu_q8_reference_model": reference_bundle.get("cpu_q8_reference_model"),
                "cpu_q8_reference_provider": reference_bundle.get("cpu_q8_reference_provider"),
                "cpu_q8_reference_session_create_s": reference_bundle.get("cpu_q8_reference_session_create_s"),
            })
        except Exception as exc:
            result.update({
                "cpu_q8_reference_error_type": type(exc).__name__,
                "cpu_q8_reference_error": str(exc),
                "cpu_q8_reference_traceback": traceback.format_exc(),
            })
        related_count = sum(1 for item in prompt_results if item.get("prompt_related_output"))
        detok_ok_count = sum(1 for item in prompt_results if item.get("detokenize_ok"))
        quality_minimum_pass = bool(parity.get("semantic_parity_pass")) if args.smoke_tests else related_count >= 1
        result.update({
            "prompt_results": prompt_results,
            "provider_counts": provider_counts,
            "provider_events_head": provider_events[:40],
            "profile": profile if args.keep_profile else None,
            "profile_deleted_after_parse": profile_deleted,
            "qnn_only_blocks_head": provider_counts.get("QNNExecutionProvider", 0) > 0 and provider_counts.get("CPUExecutionProvider", 0) == 0,
            "cpu_execution_provider_fallback_count": int(provider_counts.get("CPUExecutionProvider", 0)),
            "related_prompt_count": related_count,
            "detokenize_ok_count": detok_ok_count,
            "runtime_ok": bool(provider_counts.get("QNNExecutionProvider", 0) > 0 and provider_counts.get("CPUExecutionProvider", 0) == 0 and detok_ok_count == len(prompt_results)),
            "quality_minimum_pass": quality_minimum_pass,
            "quality_oracle": {
                "mode": "cpu_q8_reference_semantic_keyword_parity",
                "same_subject_threshold": 0.8,
                "collapse_must_be_zero": True,
            },
            "parity_status": "pass" if quality_minimum_pass else "fail",
        })
        result.update(parity)
        result["ok"] = bool(result["runtime_ok"] and (result["quality_minimum_pass"] if args.smoke_tests else True))
    except Exception as exc:
        result.update({"ok": False, "runtime_ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        (log_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        write_summary(log_dir, result)
        print("PRACTICAL_HYBRID_V0_RESULT " + json.dumps({
            "ok": result.get("ok"),
            "runtime_ok": result.get("runtime_ok"),
            "quality_minimum_pass": result.get("quality_minimum_pass"),
            "qnn_only_blocks_head": result.get("qnn_only_blocks_head"),
            "provider_counts": result.get("provider_counts"),
            "related_prompt_count": result.get("related_prompt_count"),
            "detokenize_ok_count": result.get("detokenize_ok_count"),
            "log_dir": str(log_dir),
            "summary": str(log_dir / "summary.md"),
            "error": result.get("error"),
        }, ensure_ascii=False, sort_keys=True))
        if not result.get("runtime_ok"):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
