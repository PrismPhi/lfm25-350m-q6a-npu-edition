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

`response_format: {"type":"json_object"}` is a constrained mode that always emits a fixed single-key object `{"answer": "<value>"}`, where the value comes from QNN logits and is capped at 96 characters. It does not support arbitrary or user-specified schemas and does not guarantee factual values. A free-form prompt alone does not guarantee valid JSON.

## Concurrency

The inference engine remains serialized within 1 process and runs 1 QNN generation at a time. The default admission limit is 4 waiting requests with a 30 s wait cap; overflow returns 429 and wait timeout/draining returns 503. Streaming uses a bounded 32-delta queue so network writes do not occur under the QNN lock.

## Streaming and Shutdown

Configured stop strings use a holdback buffer, so a possible multi-token/Unicode stop prefix is delayed until it is ordinary text. With no stop string, no holdback delay is added. A disconnected or write-timed-out client requests cancellation; cancellation is observed between QNN graph runs and cannot interrupt a graph run already inside the provider.

SIGINT and SIGTERM enter draining state, reject new work, cancel/wait for active generation, acquire the engine lock, end profiling, release sessions, and write the final result. The default write timeout is 5 s and shutdown wait cap is 30 s.

## Runtime Compatibility

EPContext reuse is restricted to an exact `source-stamp.json` identity that includes ONNX Runtime/onnxruntime-qnn versions and library hashes, HTP Stub/Skel hashes, provider/session options, `QCS6490 / v68`, chunk, and total length. A mismatch or corrupt stamp causes regeneration. The tested package does not expose supported QAIRT or Qualcomm QNN runtime version APIs, so those fields are `null`; the absolute library paths and SHA-256 identify the tested runtime.

The default Hugging Face revision is pinned to `773ff42cc383cb61ecf32eb13d1f828634fbd0e1`. Explicit mirror/revision overrides are supported but are outside the public default validation.

## Logging

Persistent request history is a bounded deque with default limit 128. By default it stores metadata only, not prompt text, generated text, or token ID lists. `--log-bodies` explicitly opts in to those bodies. Standard output request logs also remain metadata-only.

## Power and Memory

Power is unmeasured because no world-readable `power_now` or hwmon power source is available. Only a thermal proxy is recorded. RSS is a host-process value and may not include all DSP-side memory.

## Distribution

QNN/QAIRT libraries, EPContext binaries, and GGUF are not bundled. EPContexts are generated on the target Q6A. GGUF is referenced only through its official distribution location.
