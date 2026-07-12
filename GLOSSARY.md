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
