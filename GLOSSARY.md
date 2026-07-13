**日本語版 -> [GLOSSARY.ja.md](GLOSSARY.ja.md)**

# Glossary

| Term | Meaning in this repository |
|---|---|
| QNN-only | The profiled model-body graph executes only through `QNNExecutionProvider`; CPU EP fallback is disabled. Host preprocessing and bookkeeping are still allowed and disclosed. |
| EPContext | A device-generated QNN context used for faster subsequent loading. It is hardware/stack specific and is not distributed here. |
| chunk16 | Prefill graph that processes prompt tokens in blocks of sixteen at context length ctx2048. |
| slim decode | Single-token decode graph that returns only newly generated KV rather than the full cache. |
| handoff | Transfer of hidden/cache/position state from prefill into decode. It includes logical token length and mask semantics, not only tensor shapes. |
| minigraph | A small graph used to prove one operator or subgraph can create, load, run, and match a CPU reference before full-model integration. |
| canary | A small known-good execution used as an early health or partition check. |
| gate | A documented passing condition such as CPU parity, QNN-only execution, handoff parity, or readable generation. |
| oracle | The comparison reference, usually the official CPU Q8 model. |
| Path A2 | Historical research label for the official-weight graph-reconstruction line. It is not a public model format or product name. |
| N4b | Historical research label for the exact `LpNormalization`-based normalization rewrite used in the successful QNN graph line. |
| V0/V1/V1.8b/V1.9/V1.10 | Historical experiment groups. They are preserved for evidence traceability and are not semantic release versions. |
| V2a/V2b | Proposed follow-up research tracks: export/QDQ quality reconstruction and device-resident KV/runtime work, respectively. |
| Part 0 | Historical label for the chunk32 prefill experiment. It passed a speed check but failed functional handoff parity. |

## Domain concepts

| Concept | Background meaning (not specific to this repository) |
|---|---|
| QDQ | The Quantize-Dequantize representation places explicit QuantizeLinear/DequantizeLinear nodes in the graph to convey each tensor's quantization scale and zero point to the backend. A supporting backend can lower these into integer or quantized ops, but the presence of QDQ nodes alone does not guarantee integer execution. |
| A16W8 | A quantization policy that represents activations in 16-bit and weights in 8-bit. It lets activations keep wider dynamic range and precision and can reduce quantization error compared with A8W8, while memory bandwidth or compute cost can increase. |
| GQA | Grouped-Query Attention uses more query heads than key/value heads, so several query heads share the same key/value head. Implementations need repeat, broadcast, or head-mapping to align with the query-head count, but this does not necessarily mean physical duplication. |
| RoPE | Rotary Position Embedding injects relative position into attention by rotating paired query and key components by a position-dependent angle, instead of adding a positional vector. |
| activation range collapse | An informal diagnostic term for a quantization range becoming too narrow when calibration fails to capture an activation's true amplitude or outliers. Runtime values then saturate or clip, and the error accumulates through later stages and degrades output quality. |
| LpNormalization | An ONNX operator that computes the Lp norm along a given axis and returns the input divided by that norm, i.e. the normalized tensor rather than the norm value itself. This project uses it as a building block when decomposing a normalization the target does not directly support into equivalent ops. It is not a drop-in replacement for RMSNorm; an equivalent RMSNorm decomposition must also account for the dimension-dependent factor, epsilon, and learned weight. |
