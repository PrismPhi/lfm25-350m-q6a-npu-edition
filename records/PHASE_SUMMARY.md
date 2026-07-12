**日本語版 -> [PHASE_SUMMARY.ja.md](PHASE_SUMMARY.ja.md)**

# Phase Summary

The phase names below are historical experiment identifiers retained for evidence traceability, not semantic release versions. See the [glossary](../GLOSSARY.md).

| Phase | Result | Remaining decision |
|---|---|---|
| V0 | established prompt -> tokenizer -> QNN -> detokenize | upstream hidden parity was the main issue |
| V1 graph reconstruction (Path A2/N4b) | official weight transplant, exact LpNormalization rewrite, tail-mask, A16W8 | separated weight/layout from runtime blockers |
| V1 chunk | chunk16/ctx2048 QNN-only prefill | required host cache handoff |
| V1 slim decode | about 17 tok/s with new-only KV output | device-resident KV is V2b |
| V1.7 | OpenAI-compatible API and WebUI | tracked early stop and UTF-8 in long generation |
| V1.8b | chat profile, min_new_tokens, logit_bias, JSON mode | long-form ratio 0.621 |
| V1.9 | rejected post-quantization range unification on PC | scale equality alone cannot preserve quality |
| V1.10 | proved calibration-time group-max on PC | QNN partition topology remained broken |
| Public Part 0 | proved chunk32 at 296.33 tok/s | rejected on handoff |
| Public install | local fresh 62.2 s, public URL fresh 126.9-233.5 s, rerun 5.5-5.8 s | users must provide a QNN environment |

## Released Configuration

The released configuration is chunk16 + slim decode, ctx2048, with QNN fallback disabled. API decode is 17.00-17.60 tok/s, strict JSON works, and a 442-token Japanese completion reached a normal stop.

## What Was Tried and Why It Was Not Released

The calibration-only V1 levers were closed by the 3 V1.8b, V1.9, and V1.10 experiments. Chunk32 passed its speed target but closed on the handoff gate. The next technical work is V2a export/QDQ-topology reconstruction or V2b device-resident KV runtime.
