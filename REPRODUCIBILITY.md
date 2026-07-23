**日本語版 -> [REPRODUCIBILITY.ja.md](REPRODUCIBILITY.ja.md)**

# Reproducibility

## 2 Reproduction Levels

The user path generates EPContexts on Q6A from distributed A16W8 QDQ models. The research path rebuilds the released graph from official LFM2.5-350M. Historical labels Path A2 and N4b refer to the official-weight graph reconstruction and exact LpNormalization rewrite; see the [glossary](GLOSSARY.md). The research path is sensitive to QAIRT and operator differences, so each intermediate passing condition must be checked before evaluating the full graph.

## User Path

1. Prepare a QNN-enabled Python.
2. Acquire 11 release assets from the manifest-pinned revision and verify SHA-256 with `runner/config/model-assets.json`.
3. Capture the runtime fingerprint and final merged `ADSP_LIBRARY_PATH`.
4. Generate external EPContexts on the device from chunk16/decode raw QDQ.
5. Execute immediately after generation and after reload; require finite outputs, QNN count above 0, and CPU count equal to 0.
6. Run the deterministic normal/JSON semantic canary through a temporary localhost server.
7. Start the persistent server with `runner/start_server.sh`.

```bash
bash runner/install.sh --python /path/to/qnn-venv/bin/python
```

Passing requires matching SHA for 11 assets; strict generation and reload execution with every output finite; chunk/decode profile counts with `QNNExecutionProvider > 0` and `CPUExecutionProvider == 0`; normal output containing `Tokyo`; JSON output `{"answer":"Tokyo"}`; first token ID `40550`; and post-profile `qnn_only_verified=true`.

## Runtime Contract

| Component | Verified value |
|---|---|
| Python | `3.12.3` |
| ONNX | `1.22.0` |
| ONNX Runtime | `1.27.0` |
| onnxruntime-qnn | `2.3.0` |
| tokenizers | `0.23.1` |
| SoC / HTP | `QCS6490 / v68` |
| QNN EP SHA-256 | `ebcec5c0b52cc4bb96542accd17f4a410174bfc7a44599586f17852aa9ae78ef` |
| QNN HTP SHA-256 | `9e1a73ed4f3e7cf3ef3199a23dfb3885b12a64edea30e607b0687b25b28e94f7` |
| HTP Stub SHA-256 | `68e4c5f932ea006efa311c6763a4b25081f11663ae70390e6a58e41142f82c9f` |
| HTP Skel SHA-256 | `1c4ef2cec89209ffb32d670cde10c3ff66733e44d34dd23c0b76f319afc5a6dd` |
| QAIRT version | `null` because no supported runtime API exposed it |
| Qualcomm QNN runtime version | `null` because no supported runtime API exposed it |

The onnxruntime-qnn package version is not treated as the QAIRT or Qualcomm QNN runtime version. `source-stamp.json` stores the available fingerprint together with provider options, session config, chunk, total length, source SHA, context-file hashes, and strict execution results. Matching identity and files permit reuse. A library hash, ONNX Runtime/onnxruntime-qnn version, target, provider/session option, graph shape, source SHA, or stamp-integrity change forces regeneration.

The default download resolves revision `773ff42cc383cb61ecf32eb13d1f828634fbd0e1`, not a moving branch. Override a mirror or revision only explicitly.

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
- Keep fallback configuration, QNN session creation, and post-profile QNN-only verification as separate states. A provider name list containing CPU is not proof of execution fallback; use profile provider counts.
- Preserve user entries when merging `ADSP_LIBRARY_PATH`; record the final value in the runtime fingerprint.
- Declare tokenizer/detokenizer, sampling, embedding, and cache bookkeeping as host work.
- Padded cache rows remain masked and the logical cache length after handoff matches actual tokens.
- Do not distribute EPContext/QNN/QAIRT binaries; generate them on the target device.

## Evidence

Public numbers use the [evidence index](records/EVIDENCE_INDEX.md) and `records/evidence/*.json` as their single source. Earlier syntax-only JSON install checks remain historical; the current semantic-canary result is a separate dated entry. Raw private research records are not published directly because they contain personal paths and device identifiers.
