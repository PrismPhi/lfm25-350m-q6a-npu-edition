**日本語版 -> [PORTING.ja.md](PORTING.ja.md)**

# Porting to Other HTP Generations

This project was verified only on the HTP v68 stack of QCS6490. It does not claim support for v69/v73/v75. It publishes a portability test procedure for revalidating the conclusions.

## Portability Test Kit

| Stage | Input | Passing condition |
|---|---|---|
| P0 environment | QNN-enabled Python | runtime fingerprint, provider registration, HTP library load |
| P1 operator minigraph | MatMul, Conv, LpNormalization, Slice, Concat | create/load/run is finite and QNN-only |
| P2 attention minigraph | q/k/v, RoPE, tail-mask | CPU cosine >=0.999, top-1 match |
| P3 layer canary | Conv layer, attention layer, MLP | finite, fallback 0 |
| P4 full chunk | chunk16/ctx2048 | warm load, cache output shape |
| P5 decode | chunk1 slim cache | >=15 tok/s as a guide, QNN-only |
| P6 handoff | chunk -> decode | cache/logits/top-1 gate |
| P7 prompt-to-text | 6 smoke + JSON | UTF-8, subject, fixed JSON object, first-token golden |

## Items to Revalidate Per Generation

- Whether fp16 MatMul/Conv partitions natively.
- Whether MatMulNBits or blockwise quantization lowers through QNN EP.
- Whether LpNormalization or an RMSNorm-equivalent operator is native.
- Whether Slice+Concat GQA repeat can return to Tile.
- Whether a uint8 KV cache preserves the input/output contract.
- Context-binary compatibility. Regenerate EPContext on each target device and stack.
- Exact runtime identity: ONNX Runtime/onnxruntime-qnn versions, EP/HTP/Stub/Skel hashes, provider/session options, SoC/HTP generation, chunk, and total length. Record QAIRT/Qualcomm QNN runtime versions only when a supported API exposes them.
- `ADSP_LIBRARY_PATH` search order. Preserve existing entries and append required paths instead of replacing the environment.
- VTCM, spill/fill, DDR bandwidth, and shared-memory mode.
- Whether a known QAIRT/Qualcomm QNN runtime or ONNX Runtime QNN package change alters the known failure modes.

## Procedure

1. Capture the runtime fingerprint and generate the smallest graph with `runner/scripts/generate_epcontext.py`.
2. Do not remove `session.disable_cpu_ep_fallback=1`.
3. Record create, load, and run as separate cases; reject NaN/Inf output.
4. Require profile JSON counts `QNNExecutionProvider > 0` and `CPUExecutionProvider == 0`.
5. Reload the generated context and repeat the run/finite/profile gates.
6. Save the runtime identity in `source-stamp.json`; a mismatch must regenerate.
7. After a failure, do not reuse the process; run a known-small health canary.
8. Preserve the P0-P3 matrix before moving to a full graph.
9. Derive a semantic canary and first-token golden from a verified runtime rather than hardcoding an unsupported expectation.

## Porting Decision

If a newer generation supports fp16 or native norm, it need not mechanically retain v68 workarounds. Conversely, provider registration alone does not prove full-graph support. Reselect export/QDQ topology from the minigraph results for that generation.

Remeasure performance with the same prompt, context, sampling, and wall definition. This repository does not guarantee its 17.00-17.60 tok/s on another generation.

The released default asset revision `773ff42cc383cb61ecf32eb13d1f828634fbd0e1` identifies v68 validation assets, not proof that their EPContexts are portable. Raw QDQ may be reused only after its SHA is checked; EPContext must be regenerated.
