**日本語版 -> [REPRODUCIBILITY.ja.md](REPRODUCIBILITY.ja.md)**

# Reproducibility

## 2 Reproduction Levels

The user path generates EPContexts on Q6A from distributed A16W8 QDQ models. The research path rebuilds the released graph from official LFM2.5-350M. Historical labels Path A2 and N4b refer to the official-weight graph reconstruction and exact LpNormalization rewrite; see the [glossary](GLOSSARY.md). The research path is sensitive to QAIRT and operator differences, so each intermediate passing condition must be checked before evaluating the full graph.

## User Path

1. Prepare a QNN-enabled Python.
2. Acquire 11 release assets and verify SHA-256 with `runner/config/model-assets.json`.
3. Generate external EPContexts on the device from chunk16/decode raw QDQ.
4. Run 1 time immediately after generation and once after reload with QNNExecutionProvider only.
5. Test a normal prompt and JSON mode through a temporary localhost server.
6. Start the persistent server with `runner/start_server.sh`.

```bash
bash runner/install.sh --python /path/to/qnn-venv/bin/python
```

Passing requires matching SHA for 11 assets; `generate_ok=true`, `load_ok=true`, and `load_qnn_only=true` for chunk/decode; non-empty normal output; JSON parseable as an object; and QNN-only chunk/decode profiles at shutdown.

## Full Research Pipeline

| Phase | Operation | Required gate |
|---|---|---|
| R0 | acquire official `LiquidAI/LFM2.5-350M` and official ONNX | pin tokenizer/config/model SHA |
| R1 | build operator/initializer inventory | layer count, hidden 1024, vocab 65536, cache contract match |
| R2 | transplant official weights into the reconstructed graph | CPU Q8 logits cosine and top-1 |
| R3 | replace RMSNorm with the exact LpNormalization construction | minigraph QNN-only + CPU parity |
| R4 | add Slice+Concat GQA repeat, RoPE, causal tail-mask | q/k/v official-input minigraph parity |
| R5 | apply A16W8 QDQ to Conv/MLP/attention/final norm/lm_head | layer canary with fallback disabled |
| R6 | build chunk16/ctx2048 and chunk1 slim-decode graphs | PC static check, QNN create/load |
| R7 | perform chunk-to-decode cache handoff | cache/logits/top-1 gate |
| R8 | tokenizer -> prefill -> decode -> detokenize | 6 smoke, JSON, long generation, profile |
| R9 | externalize EPContext | warm load <=5 s, QNN-only profile |

The public chunk/decode graph builder is `runner/scripts/probe_p4_patha2_full_chunk_graph.py`; the filename retains its historical experiment label for traceability. Create model-distribution staging from the released QDQ and host assets with:

```bash
python3 scripts/prepare_model_release.py \
  --official-model /path/to/model_q8.onnx \
  --tokenizer-dir /path/to/official-model-dir \
  --chunk-model /path/to/chunk16_a16w8_qdq.onnx \
  --decode-model /path/to/decode_a16w8_qdq.onnx \
  --model-license MODEL_LICENSE \
  --output-dir /path/to/release-assets
```

## Invariants

- QNN sessions use `session.disable_cpu_ep_fallback=1` and `enable_fallback=False`.
- A provider name list containing CPU is not proof of failure or success. Use profile provider counts to prove QNN-only execution.
- Declare tokenizer/detokenizer, sampling, embedding, and cache bookkeeping as host work.
- Padded cache rows remain masked and the logical cache length after handoff matches actual tokens.
- Do not distribute EPContext/QNN/QAIRT binaries; generate them on the target device.

## Evidence

Public numbers use the [evidence index](records/EVIDENCE_INDEX.md) and `records/evidence/*.json` as their single source. Raw private research records are not published directly because they contain personal paths and device identifiers.
