**日本語版 -> [KNOWN_LIMITS.ja.md](KNOWN_LIMITS.ja.md)**

# Known Limits

## Long-Form Quality

The mean Hybrid QNN / CPU Q8 completion-length ratio is 0.621, with 3/6 prompts at or above 0.70. 3 historical experiments, V1.8b, V1.9, and V1.10, tested simple recalibration, post-quantization range unification, and calibration-time group-max constraints. None produced a release-quality improvement. The labels are defined in the [glossary](GLOSSARY.md).

The active configuration can complete a 442-token Japanese response with a normal stop, but length is not factual quality. Verify important explanations and summaries against another oracle such as CPU Q8.

## User Controls

| knob | default | purpose |
|---|---:|---|
| profile | chat | temperature 0.8, top-k 40, top-p 0.95 |
| extraction profile | explicit | temperature 0.1, top-k 50, top-p 1.0 |
| repetition_penalty | 1.1 | chat repetition control |
| repetition_last_n | 64 | limit to the last 64 tokens |
| min_new_tokens | 0 | explicitly suppress an early EOS |
| logit_bias | none | per-token [-100,100] bias |

`min_new_tokens` and EOS `logit_bias` do not guarantee quality. No EOS bias is applied by default.

## Context

The released context is 2048. ctx4096 was shown to be constructible within memory, but decode I/O is too large without device-resident KV and is deferred to the proposed V2b research track.

## Chat Template

The server uses a limited ChatML renderer for string content with system/user/assistant roles. It works without jinja2. Images, audio, arbitrary tool execution, complex content parts, and prompts ending with assistant are unsupported.

## JSON

`response_format: {"type":"json_object"}` is a constrained mode that guarantees object syntax. It does not guarantee factual values or a schema. A free-form prompt alone does not guarantee valid JSON.

## Concurrency

The inference engine is serialized within 1 process. HTTP connections may arrive concurrently, but a lock processes 1 QNN generation at a time.

## Power and Memory

Power is unmeasured because no world-readable `power_now` or hwmon power source is available. Only a thermal proxy is recorded. RSS is a host-process value and may not include all DSP-side memory.

## Distribution

QNN/QAIRT libraries, EPContext binaries, and GGUF are not bundled. EPContexts are generated on the target Q6A. GGUF is referenced only through its official distribution location.
