**日本語版 -> [PITFALLS.ja.md](PITFALLS.ja.md)**

# Failure Routes and Diagnostics

| Route | Symptom | Cause | Avoidance | Public evidence |
|---|---|---|---|---|
| fp16 MatMul | QNN finalize/partition failure | target pattern does not form on the v68 stack | start from a minigraph; do not assume fp16 support | `records/evidence/experiment-ledger.json` |
| MatMulNBits | unsupported/fallback | mismatch with QNN EP weight-only lowering | use per-tensor A16W8 QDQ MatMul | `records/PHASE_SUMMARY.md` |
| float island | graph split and CPU assignment | float operator remains between QDQ boundaries | keep layer QDQ boundaries contiguous | `records/PHASE_SUMMARY.md` |
| decomposed norm | close on CPU but collapses on QNN | primitive sequence is hostile to QNN quantization | N4b exact LpNormalization pattern | `records/PHASE_SUMMARY.md` |
| mask x scale | fixed attention or leakage | large negative mask interacts with activation scale | mask -64.0, safe scale, invalid-past test | `records/evidence/chunk32-part0.json` |
| epsilon domination | RMSNorm amplitude collapse | epsilon dominates small inputs | pin official epsilon and require minigraph parity | `records/PHASE_SUMMARY.md` |
| cosine blind spot | high cosine but different token | a small logit difference flips top-1 | gate cosine, top-1, and generated subject together | `records/evidence/experiment-ledger.json` |
| padded-row leak | fast chunk graph but broken handoff | padded/cache rows enter decode | actual-token cache length, tail mask, handoff gate | `records/evidence/chunk32-part0.json` |
| Tile GQA repeat | QNN rejection | dynamic Tile shape is unsupported on v68 | use Slice+Concat repeat | `records/PHASE_SUMMARY.md` |
| u8 KV | invalid tensor/config | cache dtype/quant contract mismatch | float host cache + QDQ graph boundary | `records/PHASE_SUMMARY.md` |
| rpcmem lifecycle | following known-good graph also fails | HTP/CDSP process state after a failure | exit process, run health canary, reboot only if necessary | `records/PHASE_SUMMARY.md` |
| shared memory off | exit 139 | memory mode and graph are inconsistent | keep the default shared-memory setting | `records/PHASE_SUMMARY.md` |
| V1.8b requantization | QNN session failure after PC pass | repeated Slice remains CPU-assigned | keep the accepted context | `records/evidence/experiment-ledger.json` |
| V1.9 post-unification | PC logits collapse | range unification amplifies quantization error | stop at the PC gate | `records/evidence/experiment-ledger.json` |
| V1.10 group-max | QNN session failure after PC pass | range repair does not fix partition topology | rebuild export/QDQ boundaries in V2a | `records/evidence/experiment-ledger.json` |
| chunk32 | 296.33 tok/s but handoff failure | cache padding/position contract mismatch | reject and retain chunk16 | `records/evidence/chunk32-part0.json` |

## Shortest Diagnostic Order

1. Verify QNN-only create/load with a 1-operator minigraph.
2. Verify QDQ boundaries and dtype with a layer canary.
3. Inject official inputs to separate weight/layout from upstream state.
4. Compare cache, position, RoPE, and tail-mask using separate taps.
5. Compare top-1 and generated tokens in addition to logits cosine.
6. Pass handoff parity before prompt-to-text testing.

A generated QNN `.bin` is not success if session initialization fails. QNN success requires 3 checks: a fallback-disabled session, 1 execution, and a QNN-only profile provider count.
