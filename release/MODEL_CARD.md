---
license: other
license_name: lfm-open-license-v1.0
license_link: https://huggingface.co/PrismPhi/lfm25-350m-q6a-npu-edition/blob/main/MODEL_LICENSE
base_model: LiquidAI/LFM2.5-350M
language: [ja, en]
tags:
  - onnx
  - qnn
  - qualcomm
  - npu
  - qcs6490
  - lfm2.5
  - quantized
---

**日本語版 -> [MODEL_CARD.ja.md](MODEL_CARD.ja.md)**

GitHub repository: [lfm25-350m-q6a-npu-edition](https://github.com/PrismPhi/lfm25-350m-q6a-npu-edition)

# LFM2.5-350M Q6A NPU Edition Model Card

## Overview

A QCS6490/Q6A A16W8 QDQ ONNX derivative of Liquid AI's LFM2.5-350M. The model distribution contains raw QDQ and host-side constants only. Users generate EPContext on their own device.

## Files

- `qdq/chunk16_a16w8_qdq.onnx`
- `qdq/decode_a16w8_qdq.onnx`
- `host/embedding_int8_rowwise/*`
- `host/rope_cache.npz`
- `tokenizer/*`
- `MODEL_LICENSE`
- `asset-manifest.json`

## Change Notice

QDQ ONNX files reconstruct the source model into a QNN-oriented graph and apply A16W8 QDQ. Embedding is quantized to rowwise symmetric int8. RoPE cache is mechanically transformed from official ONNX initializers into NPZ.

## Intended Use

Local experiments, an OpenAI-compatible API, and OpenWebUI on QCS6490/Q6A. General long-form quality, other HTP generations, and ctx4096 are not guaranteed.

## Performance

The adopted runner measures 17.00-17.60 tok/s API decode and 0.31-1.03 s TTFT. See [README.md](../README.md) for conditions.

## License

LFM Open License v1.0. Free commercial use is limited to annual revenue below $10,000,000. Redistribution requires including the license, marking changes, and retaining attribution.
