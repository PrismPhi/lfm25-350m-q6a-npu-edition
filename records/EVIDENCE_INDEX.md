**日本語版 -> [EVIDENCE_INDEX.ja.md](EVIDENCE_INDEX.ja.md)**

# Public Evidence Index

| Evidence | Content |
|---|---|
| `evidence/headline-metrics.json` | released API performance, CPU comparison, long-form ratio |
| `evidence/experiment-ledger.json` | 3 V1.8b/V1.9/V1.10 requantization experiments |
| `evidence/chunk32-part0.json` | chunk32 speed, QNN-only, handoff rejection |
| `evidence/install-validation.json` | local install and 2 public-URL fresh/rerun validations, QNN-only smoke |
| `../runner/config/model-assets.json` | size/SHA-256 of 11 release assets |
| `decisions/CHUNK32.md` | chunk32 decision record (historical label Part 0) |

Public JSON contains no personal path, hostname/IP, credential, or raw profile. Original audit archives remain frozen on the source PC and are not copied into this public tree.

## External Sources

- LFM2.5-350M: <https://huggingface.co/LiquidAI/LFM2.5-350M>
- LFM Open License v1.0: <https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE>
- Liquid AI license guide: <https://docs.liquid.ai/lfm/help/model-license>

## Numeric Discipline

- Distinguish wall, QNN run, and whole-process denominators for tok/s.
- State when TTFT excludes session creation.
- State power as unmeasured and thermal as a proxy.
- Prove QNN-only using profile provider counts.
- Do not treat compiler generation of a context binary as success.
