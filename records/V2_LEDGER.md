**日本語版 -> [V2_LEDGER.ja.md](V2_LEDGER.ja.md)**

# V2 Next-Project Ledger

This ledger records proposed follow-up research. V2a and V2b are track labels defined in the [glossary](../GLOSSARY.md), not claims of support in the current release.

| Track | Objective | Current evidence | Completion gate |
|---|---|---|---|
| V2a | rebuild export/QDQ topology with channelwise/outlier-aware handling | method demonstrated and 3 failure samples from V1.8b/V1.9/V1.10 | CPU Q8 quality parity, QNN-only partition, 6-prompt smoke |
| V2b | keep KV device-resident and reduce host I/O | V1 uses host cache and ctx2048 | 25-30 tok/s, ctx4096, actual-token cache, QNN-only profile |

## Priority

V2a is the quality track for repairing quantization boundaries and partition topology. V2b is the performance track for improving decode runtime and context. Apparent speed alone does not replace the current V1; a candidate must satisfy quality, disabled fallback, the cache contract, and long-form completion together.

## Publication Handling

Track these as issues or a project when work begins. Each experiment records its reproduction command, input SHA-256, QNN profile, and release decision. Do not claim untested HTP generations as supported or claim that 25-30 tok/s has already been achieved.
