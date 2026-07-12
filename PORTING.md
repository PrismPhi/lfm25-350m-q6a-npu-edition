**日本語版 -> [PORTING.ja.md](PORTING.ja.md)**

# Porting to Other HTP Generations

This project was verified only on the HTP v68 stack of QCS6490. It does not claim support for v69/v73/v75. It publishes a portability test procedure for revalidating the conclusions.

## Portability Test Kit

| Stage | Input | Passing condition |
|---|---|---|
| P0 environment | QNN-enabled Python | provider registration, HTP library load |
| P1 operator minigraph | MatMul, Conv, LpNormalization, Slice, Concat | create/load/run is QNN-only |
| P2 attention minigraph | q/k/v, RoPE, tail-mask | CPU cosine >=0.999, top-1 match |
| P3 layer canary | Conv layer, attention layer, MLP | finite, fallback 0 |
| P4 full chunk | chunk16/ctx2048 | warm load, cache output shape |
| P5 decode | chunk1 slim cache | >=15 tok/s as a guide, QNN-only |
| P6 handoff | chunk -> decode | cache/logits/top-1 gate |
| P7 prompt-to-text | 6 smoke + JSON | UTF-8, subject, JSON object |

## Items to Revalidate Per Generation

- Whether fp16 MatMul/Conv partitions natively.
- Whether MatMulNBits or blockwise quantization lowers through QNN EP.
- Whether LpNormalization or an RMSNorm-equivalent operator is native.
- Whether Slice+Concat GQA repeat can return to Tile.
- Whether a uint8 KV cache preserves the input/output contract.
- Context-binary compatibility. Regenerate EPContext on each target device and stack.
- VTCM, spill/fill, DDR bandwidth, and shared-memory mode.
- Whether QAIRT/ONNX Runtime QNN version changes alter the known failure modes.

## Procedure

1. Generate the smallest graph with `runner/scripts/generate_epcontext.py`.
2. Do not remove `session.disable_cpu_ep_fallback=1`.
3. Record create, load, and run as separate cases.
4. Inspect the profile JSON `QNNExecutionProvider` count.
5. After a failure, do not reuse the process; run a known-small health canary.
6. Preserve the P0-P3 matrix before moving to a full graph.

## Porting Decision

If a newer generation supports fp16 or native norm, it need not mechanically retain v68 workarounds. Conversely, provider registration alone does not prove full-graph support. Reselect export/QDQ topology from the minigraph results for that generation.

Remeasure performance with the same prompt, context, sampling, and wall definition. This repository does not guarantee its 17.00-17.60 tok/s on another generation.
