**日本語版 -> [API.ja.md](API.ja.md)**

# OpenAI-Compatible API

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | readiness, backend, fallback, queue |
| GET | `/v1/models` | list of 1 model |
| POST | `/v1/chat/completions` | non-streaming/streaming chat completion |

The default base URL is `http://127.0.0.1:18080/v1`.

## Supported Request Fields

- `model`: `lfm2.5-350m-qnn-ctx2048` or `lfm2.5-350m`
- `messages`: string content with role `system`, `user`, or `assistant`
- `stream`: boolean
- `n`: only integer 1
- `max_tokens`: 1-1024
- `temperature`: 0-2
- `top_p`: (0,1]
- `top_k`: 1-4096
- `repetition_penalty`: 0.1-2
- `repetition_last_n`: -1-2048
- `min_new_tokens`: 0-1024
- `seed`: integer 0-9223372036854775807
- `stop`: 1 non-empty string or an array of 1-4 non-empty strings; each at most 1024 characters
- `logit_bias`: object from token ID to [-100,100]
- `profile`: `chat` or `extraction`
- `response_format`: `{"type":"text"}` or `{"type":"json_object"}`

`tools: []` and `tool_choice: "auto"` are accepted and ignored for OpenWebUI compatibility. The server does not execute tools or emit tool calls.

Boolean values are rejected for integer/float fields. NaN, Infinity, and -Infinity are rejected by JSON parsing and numeric validation, including `logit_bias` values. The port must be an integer in 1-65535.

## Normal Request

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-350m-qnn-ctx2048","messages":[{"role":"user","content":"日本の首都は？"}],"max_tokens":64}'
```

## JSON Mode

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-350m-qnn-ctx2048","messages":[{"role":"user","content":"日本の首都をJSONで返して"}],"response_format":{"type":"json_object"},"max_tokens":64}'
```

JSON mode always emits a fixed single-key object `{"answer": "<value>"}`: the structural tokens are fixed and the value is selected from QNN logits, capped at 96 characters. Arbitrary or user-specified schemas are not supported. Inspect `qnn_metrics.json_object_valid` and `qnn_metrics.json_check` in the response.

## Streaming

`stream=true` returns SSE deltas and ends with `data: [DONE]`. JSON parsing, all field/model/message validation, tokenization, context-length checking, and queue admission finish before the 200 SSE header is sent. A runtime failure after that header is an SSE `event: error`, followed by stream termination; the server does not attempt a second HTTP response.

The server decodes the cumulative ID sequence instead of each byte-level token separately, avoiding replacement characters from incomplete UTF-8 fragments. A holdback buffer prevents all or part of a configured multi-token/Unicode stop string from being emitted. Stream and non-stream visible text therefore use the same stop semantics. With no configured stop, deltas are emitted immediately.

## Queue and Cancellation

Inference and socket writes are separated by a bounded queue. The default stream queue holds 32 deltas, the request queue admits 4 waiters, and queue wait is capped at 30 s. Client disconnect or a 5 s write timeout requests generation cancellation and releases the serialized QNN slot after the current provider run returns.

## Shutdown and Logging

SIGINT and SIGTERM set readiness false and `draining=true`, reject new requests, cancel/wait for active work, acquire the QNN engine lock, end profiles, release sessions, and then write `server_result.json`. Request threads are non-daemon.

Request history is bounded at 128 entries. Prompt/generated bodies and token ID lists are off by default and require `--log-bodies`; metadata counters and error types remain available in `/health` and the final result.

## Metrics

`qnn_metrics` contains prefill/decode tok/s, TTFT, actual-token cache lengths, finish reason, sampling, JSON validation, first token ID, first-step top token IDs, and finite-output state. TTFT excludes session creation. When `max_tokens=1`, the final token is not fed through an unnecessary decode graph run, so decode run count is 0.

Fallback reporting distinguishes `fallback_configured_disabled`, `session_qnn_provider_created`, and `qnn_only_verified`. The last value remains false while the server is running because provider counts are finalized only by ending the profile; `server_result.json` sets it true only when chunk/decode profiles both have `QNNExecutionProvider > 0` and `CPUExecutionProvider == 0`.

## Errors

| code | HTTP | Meaning |
|---|---:|---|
| `invalid_request` | 400 | invalid field, type, or range |
| `context_length_exceeded` | 400 | prompt + max_tokens exceeds 2048 |
| `queue_full` | 429 | bounded wait queue is full |
| `queue_timeout` | 503 | queue wait exceeded the configured cap |
| `server_draining` | 503 | shutdown has started |
| `generation_cancelled` | 503 | generation was cancelled before a normal response |
| `not_found` | 404 | unsupported route |
| `generation_error` | 500 | QNN/runtime generation failure |

Authentication, TLS, and rate limiting are not built in. Use loopback with an SSH tunnel.
