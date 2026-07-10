#!/usr/bin/env python3
"""Prepare redistributable QDQ and host assets from the accepted source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


VOCAB = 65536
HIDDEN = 1024
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_or_link(source: Path, target: Path, hardlink: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    if hardlink:
        os.link(source, target)
    else:
        shutil.copyfile(source, target)


def find_initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    wanted = {"model.embed_tokens.weight", "cos_cache", "sin_cache"}
    arrays = {}
    for initializer in model.graph.initializer:
        if initializer.name in wanted:
            arrays[initializer.name] = numpy_helper.to_array(initializer)
    missing = sorted(wanted - set(arrays))
    if missing:
        raise RuntimeError(f"official model initializers missing: {missing}")
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-model", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--chunk-model", type=Path, required=True)
    parser.add_argument("--decode-model", type=Path, required=True)
    parser.add_argument("--model-license", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardlink-models", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copy_or_link(
        args.chunk_model.resolve(),
        output / "qdq" / "chunk16_a16w8_qdq.onnx",
        args.hardlink_models,
    )
    copy_or_link(
        args.decode_model.resolve(),
        output / "qdq" / "decode_a16w8_qdq.onnx",
        args.hardlink_models,
    )
    for name in TOKENIZER_FILES:
        source = args.tokenizer_dir / name
        if source.is_file():
            copy_or_link(source, output / "tokenizer" / name, False)
    copy_or_link(args.model_license, output / "MODEL_LICENSE", False)

    model = onnx.load(str(args.official_model), load_external_data=True)
    arrays = find_initializers(model)
    weight = np.asarray(arrays["model.embed_tokens.weight"], dtype=np.float32)
    if weight.shape != (VOCAB, HIDDEN):
        raise RuntimeError(f"unexpected embedding shape: {weight.shape}")
    max_abs = np.max(np.abs(weight), axis=1).astype(np.float32)
    scale = np.maximum(max_abs / np.float32(127.0), np.float32(1.0e-8)).astype(np.float32)
    quantized = np.rint(weight / scale[:, None]).clip(-127, 127).astype(np.int8)
    embedding_dir = output / "host" / "embedding_int8_rowwise"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    np.save(embedding_dir / "model_embed_tokens_weight_rowwise_int8.npy", quantized)
    np.save(embedding_dir / "model_embed_tokens_weight_rowwise_scale.npy", scale)
    embedding_metadata = {
        "format": "official_embed_tokens_rowwise_symmetric_int8",
        "shape": [VOCAB, HIDDEN],
        "q_dtype": "int8",
        "scale_dtype": "float32",
        "source_model_sha256": sha256_file(args.official_model),
        "modified_file_notice": "Derived and quantized from LiquidAI/LFM2.5-350M.",
    }
    (embedding_dir / "metadata.json").write_text(
        json.dumps(embedding_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    host_dir = output / "host"
    np.savez(
        host_dir / "rope_cache.npz",
        cos_cache=np.asarray(arrays["cos_cache"], dtype=np.float32),
        sin_cache=np.asarray(arrays["sin_cache"], dtype=np.float32),
    )

    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "asset-manifest.json":
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "model_id": "lfm2.5-350m-qnn-ctx2048",
        "source_model": "LiquidAI/LFM2.5-350M",
        "license": "LFM Open License v1.0",
        "modified_file_notice": (
            "QDQ ONNX, rowwise-int8 embedding and RoPE cache files are modified or "
            "mechanically transformed derivatives."
        ),
        "files": files,
    }
    (output / "asset-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "output_dir": str(output), "file_count": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
