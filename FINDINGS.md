**日本語版 -> [FINDINGS.ja.md](FINDINGS.ja.md)**

# Engineering Findings

This document summarizes the reusable engineering conclusions behind the released runner. Historical experiment labels are defined in the [glossary](GLOSSARY.md); they are not product versions.

## Runtime Boundary

- QNNExecutionProvider/HTP executes the model-body chunk prefill and token decode graphs with CPU EP fallback disabled.
- Tokenization, detokenization, rowwise-int8 embedding lookup, sampling, stop handling, JSON framing, cache bookkeeping, masks, positions, and RoPE constants remain host work.
- Provider registration is not proof of acceleration. A valid QNN result requires fallback-disabled session creation, one successful execution, and QNN-only provider counts in the profile.

## Graph Construction

- Decomposed grouped-query attention is viable on HTP v68 when KV-head repetition uses `Slice` + `Concat`. Dynamic `Tile` was rejected by QNN.
- Quantizer target operators must include shape and data-movement operators such as `Slice`, `Concat`, `Transpose`, and `Reshape`; otherwise those nodes can escape to CPU even when arithmetic operators are quantized.
- Layer-style QDQ boundaries and safe activation scales are required around repeated KV, transposed keys, scores, masked scores, probabilities, context, and context output.
- A right-aligned causal tail mask prevents padded or invalid cache rows from leaking into attention. Mask values and activation scales must be tested together.
- An exact `LpNormalization`-based normalization rewrite ran QNN-only with close CPU parity. `SimplifiedLayerNormalization` was unsupported on the tested stack, while decomposed primitive normalization was numerically fragile.
- Quantizing the Sigmoid path caused fixed repetition and semantic collapse. The released graph leaves Sigmoid outside the quantized operator set.

## Numerical Parity

- Official-input injection is the fastest way to separate a bad attention implementation from bad upstream hidden state. The GQA minigraph matched closely when fed official q/k/v, proving that the attention pattern and projection weight layout were not the primary blocker.
- The weak value projection observed in early full-graph tests came from upstream hidden-state and normalization error, not from the V matrix layout.
- Scalar QDQ scale searches reached diminishing returns. Quality repair must consider channelwise/outlier-aware boundaries or rebuild the export/QDQ topology.
- Cosine similarity alone is insufficient near the top logits. Parity gates must include top-1 agreement, generated tokens, readable subject agreement, and repetition/collapse checks.

## Cache and Performance

- Prefill-to-decode handoff is a functional contract, not only a tensor-shape match. Logical cache length must use actual prompt tokens, while padded rows remain masked.
- Returning only newly generated KV during decode improves throughput, but the released runtime still keeps KV on the host. Host I/O is therefore the main route to higher decode speed and longer context.
- A wider prefill chunk can benchmark faster yet still be unusable. The chunk32 experiment was rejected because its padded cache/position handoff diverged; the released configuration remains chunk16.
- Short prompts can report a low API prefill rate because the first partial chunk follows the decode path. TTFT is the more representative interactive metric for those prompts.

## Runtime Reliability

- Creation of a QNN context `.bin` does not prove that the graph can be loaded or executed. Generation, reload, execution, and provider profiling are separate gates.
- After a failed HTP graph, rpcmem/CDSP process state can affect the next otherwise-valid graph. Exit the failed process and run a small known-good health check before considering a reboot.
- EPContext is generated on the target device and is intentionally excluded from distribution. Raw QDQ and host assets are the portable release boundary.
- The public installer validates all distributed assets, creates both EPContexts, runs a QNN-only canary, and checks normal and constrained-JSON API responses.

## API and Output Handling

- Cumulative detokenization avoids replacement characters caused by decoding incomplete UTF-8 byte-level token fragments one at a time.
- Long output requires explicit separation of model EOS, user stop strings, context exhaustion, and server transport termination.
- JSON mode constrains object syntax; it does not guarantee factual values or an application-specific schema.
- QNN generation is serialized within one process. HTTP connections can be concurrent, but model execution is protected by a single runtime lock.

Detailed measurements and negative experiments are indexed in [records/EVIDENCE_INDEX.md](records/EVIDENCE_INDEX.md).
