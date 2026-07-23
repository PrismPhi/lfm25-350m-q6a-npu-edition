**日本語版 -> [EVIDENCE_INDEX.ja.md](EVIDENCE_INDEX.ja.md)**

# Public Evidence Index

| Evidence | Content |
|---|---|
| `evidence/headline-metrics.json` | released API performance, CPU comparison, long-form ratio |
| `evidence/experiment-ledger.json` | 3 V1.8b/V1.9/V1.10 requantization experiments |
| `evidence/chunk32-part0.json` | chunk32 speed, QNN-only, handoff rejection |
| `evidence/install-validation.json` | historical installs plus current runtime-contract upgrade/rerun, semantic canary, finite and profile gates |
| `evidence/install-canary-golden.json` | verified `Tokyo` output, token IDs, and QNN-only source run for the deterministic canary |
| `evidence/url-validation-20260723.json` | HTTP 200 validation for 10 public links and 11 pinned assets |
| `../runner/config/model-assets.json` | size/SHA-256 of 11 release assets and pinned HF revision |
| `../runner/config/install-canary.json` | public canary prompt, expected subject, seed, and first-token golden |
| `decisions/CHUNK32.md` | chunk32 decision record (historical label Part 0) |

Public JSON contains no personal path, hostname/IP, credential, or raw profile. Original audit archives remain frozen on the source PC and are not copied into this public tree.

The pre-2026-07-23 entries in `evidence/install-validation.json` are historical syntax-only JSON checks and remain unchanged. The `runtime_contract_validation_2026_07_23` entry is the current strict validation: 11 assets verified, both contexts regenerated from the old stamp schema and then reused, normal/JSON `Tokyo`, first token `40550`, finite outputs, and post-profile QNN-only counts.

## External Sources

- LFM2.5-350M: <https://huggingface.co/LiquidAI/LFM2.5-350M>
- Released assets: <https://huggingface.co/PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu>
- Public source: <https://github.com/PrismPhi/radxa-dragon-q6a-qcs6490-lfm2.5-350m-qnn-npu>
- LFM Open License v1.0: <https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE>
- Liquid AI license guide: <https://docs.liquid.ai/lfm/help/model-license>

## Numeric Discipline

- Distinguish wall, QNN run, and whole-process denominators for tok/s.
- State when TTFT excludes session creation.
- State power as unmeasured and thermal as a proxy.
- Prove QNN-only using profile provider counts.
- Keep fallback configuration, QNN session creation, and profile-verified QNN-only as separate fields.
- Require finite output and a semantic canary; syntax-only JSON is insufficient.
- Do not treat compiler generation of a context binary as success.
