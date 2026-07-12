**日本語版 -> [PUBLIC_SCOPE.ja.md](PUBLIC_SCOPE.ja.md)**

# Publication Scope

## Published

- `runner/`: installer, OpenAI-compatible server, CLI client, QNN runtime helpers, configuration, tests
- `records/`: sanitized phase summary, decision ledger, numeric JSON, evidence index
- Equal Japanese/English README, engineering findings, glossary, reproducibility, failure modes, porting, limits, API, license, and NOTICE
- `scripts/prepare_model_release.py`: build QDQ/host asset staging
- `scripts/audit_release.py`: pre-publication audit
- Separate model distribution: 2 released QDQ files, tokenizer, rowwise-int8 embedding, RoPE cache, MODEL_LICENSE

## Not Published

- Personal paths, device hostname/IP, SSH information, secrets
- Raw private audit tree and unprocessed logs
- QNN/QAIRT/ORT-QNN shared libraries
- EPContext and `*_qnn.bin`
- GGUF files
- Virtual environments, core dumps, profiles containing personal environment information
- Rejected large candidate ONNX files

## Kept as Records

- Summaries of the V1.8b/V1.9/V1.10 requantization failures
- Part 0 chunk32 speed pass and functional failure
- Machine-readable fresh-install and idempotent-rerun results
- [V2a/V2b ledger](records/V2_LEDGER.md)

## Publication State

1. GitHub: <https://github.com/PrismPhi/radxa-dragon-q6a-qcs6490-lfm2.5-350m-qnn-npu>
2. Hugging Face: <https://huggingface.co/PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu>
3. Public updates require bilingual review, automated release audit, and Q6A validation when runtime behavior changes.

Later releases must not add credentials, personal environment data, or QNN/QAIRT/EPContext binaries.
