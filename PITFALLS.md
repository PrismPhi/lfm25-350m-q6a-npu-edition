**日本語版 -> [PITFALLS.ja.md](PITFALLS.ja.md)**

# Known Failure Modes and Diagnostics

Historical labels in this table are defined in the [glossary](GLOSSARY.md).

| Route | Symptom | Cause | Avoidance | Public evidence |
|---|---|---|---|---|
| fp16 MatMul | QNN finalize/partition failure | target pattern does not form on the v68 stack | start from a minigraph; do not assume fp16 support | `records/evidence/experiment-ledger.json` |
| MatMulNBits | unsupported/fallback | mismatch with QNN EP weight-only lowering | use per-tensor A16W8 QDQ MatMul | `records/PHASE_SUMMARY.md` |
| float island | graph split and CPU assignment | float operator remains between QDQ boundaries | keep layer QDQ boundaries contiguous | `records/PHASE_SUMMARY.md` |
| decomposed norm | close on CPU but collapses on QNN | primitive sequence is hostile to QNN quantization | exact LpNormalization rewrite (historical label N4b) | `records/PHASE_SUMMARY.md` |
| SimplifiedLayerNormalization | QNN fallback/session failure | unsupported on the tested provider stack | use the validated exact LpNormalization rewrite | `records/PHASE_SUMMARY.md` |
| mask x scale | fixed attention or leakage | large negative mask interacts with activation scale | mask -64.0, safe scale, invalid-past test | `records/evidence/chunk32-part0.json` |
| epsilon domination | RMSNorm amplitude collapse | epsilon dominates small inputs | pin official epsilon and require minigraph parity | `records/PHASE_SUMMARY.md` |
| cosine blind spot | high cosine but different token | a small logit difference flips top-1 | gate cosine, top-1, and generated subject together | `records/evidence/experiment-ledger.json` |
| padded-row leak | fast chunk graph but broken handoff | padded/cache rows enter decode | actual-token cache length, tail mask, handoff gate | `records/evidence/chunk32-part0.json` |
| Tile GQA repeat | QNN rejection | dynamic Tile shape is unsupported on v68 | use Slice+Concat repeat | `records/PHASE_SUMMARY.md` |
| incomplete quantizer target list | Slice/Concat graph escapes to CPU | shape/data-movement operators were omitted from the target list | include Slice, Concat, Transpose, and Reshape | `records/PHASE_SUMMARY.md` |
| quantized Sigmoid | fixed repetition or semantic collapse | activation quantization breaks the gated MLP path | leave Sigmoid outside the quantized operator set | `records/PHASE_SUMMARY.md` |
| u8 KV | invalid tensor/config | cache dtype/quant contract mismatch | float host cache + QDQ graph boundary | `records/PHASE_SUMMARY.md` |
| rpcmem lifecycle | following known-good graph also fails | HTP/CDSP process state after a failure | exit process, run health canary, reboot only if necessary | `records/PHASE_SUMMARY.md` |
| shared memory off | exit 139 | memory mode and graph are inconsistent | keep the default shared-memory setting | `records/PHASE_SUMMARY.md` |
| V1.8b requantization | QNN session failure after PC pass | repeated Slice remains CPU-assigned | keep the released known-good context | `records/evidence/experiment-ledger.json` |
| V1.9 post-unification | PC logits collapse | range unification amplifies quantization error | stop at the PC gate | `records/evidence/experiment-ledger.json` |
| V1.10 group-max | QNN session failure after PC pass | range repair does not fix partition topology | rebuild export/QDQ boundaries in V2a | `records/evidence/experiment-ledger.json` |
| chunk32 | 296.33 tok/s but handoff failure | cache padding/position contract mismatch | reject and retain chunk16 | `records/evidence/chunk32-part0.json` |
| full-graph GQA mismatch | poor q/k/v despite a correct attention minigraph | upstream hidden-state or normalization error | inject official q/k/v before changing attention weights/layout | `records/PHASE_SUMMARY.md` |
| stale EPContext reuse | old context loads under a changed runtime | source-only stamp omits the QNN stack identity | compare `source-stamp.json` runtime fingerprint and regenerate on any mismatch | `records/evidence/install-validation.json` |
| ADSP path overwrite | a user DSP path disappears or HTP load becomes environment-dependent | QNN registration replaces `ADSP_LIBRARY_PATH` | preserve order, remove empty/duplicate entries, and append required paths | `records/evidence/install-validation.json` |
| compiler-only success | context exists but output is NaN/Inf or executes on CPU | generation/load/run were not one strict gate | require finite outputs, QNN count above 0, and CPU count equal to 0 after generation and reload | `records/evidence/install-validation.json` |
| syntax-only install smoke | wrong subject passes because JSON parses | smoke checks only non-empty text/object syntax | fixed `Tokyo` semantic canary plus first token ID `40550` | `records/evidence/install-canary-golden.json` |
| streaming stop leak | part of a stop string reaches the client | stop detection occurs after sending token deltas | hold possible prefixes until a full multi-token/Unicode stop is ruled out | `runner/tests/test_public_runtime.py` |
| slow stream blocks inference | disconnected/slow socket retains the QNN lock | inference writes directly to the network | bounded delta queue, write timeout, and cancellation | `runner/tests/test_public_runtime.py` |
| shutdown/profile race | `end_profiling()` runs during generation | daemon request thread outlives the engine | drain, cancel/wait, acquire the engine lock, then profile and close | `runner/tests/test_public_runtime.py` |
| moving HF revision | an old Git commit installs different or unavailable assets | default URL uses `resolve/main` | pin revision `773ff42cc383cb61ecf32eb13d1f828634fbd0e1` and retain explicit overrides | `runner/config/model-assets.json` |

## Shortest Diagnostic Order

1. Capture the runtime fingerprint and verify the pinned asset SHA.
2. Verify QNN-only create/load/run with a 1-operator minigraph and finite outputs.
3. Verify QDQ boundaries and dtype with a layer canary.
4. Inject official inputs to separate weight/layout from upstream state.
5. Compare cache, position, RoPE, and tail-mask using separate taps.
6. Compare top-1 and generated tokens in addition to logits cosine.
7. Pass handoff parity before prompt-to-text testing.
8. Run the semantic canary and post-shutdown profile gate.

A generated QNN `.bin` is not success by itself. QNN success requires session creation, graph execution, every output finite, `QNNExecutionProvider > 0`, `CPUExecutionProvider == 0`, and the same checks after context reload.
