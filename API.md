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
- `max_tokens`: 1-1024
- `temperature`: 0-2
- `top_p`: (0,1]
- `top_k`: 1-4096
- `repetition_penalty`: 0.1-2
- `repetition_last_n`: -1-2048
- `min_new_tokens`: 0-1024
- `seed`: integer
- `stop`: a string or up to 4 strings
- `logit_bias`: object from token ID to [-100,100]
- `profile`: `chat` or `extraction`
- `response_format`: `{"type":"text"}` or `{"type":"json_object"}`

`tools: []` and `tool_choice: "auto"` are accepted and ignored for OpenWebUI compatibility. The server does not execute tools or emit tool calls.

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

`stream=true` returns SSE deltas and ends with `data: [DONE]`. The server decodes the cumulative ID sequence instead of each byte-level token separately, avoiding replacement characters from incomplete UTF-8 fragments.

## Metrics

`qnn_metrics` contains prefill/decode tok/s, TTFT, actual-token cache lengths, finish reason, sampling, JSON validation, and fallback state. TTFT excludes session creation.

## Errors

| code | HTTP | Meaning |
|---|---:|---|
| `invalid_request` | 400 | invalid field, type, or range |
| `context_length_exceeded` | 400 | prompt + max_tokens exceeds 2048 |
| `not_found` | 404 | unsupported route |
| `generation_error` | 500 | QNN/runtime generation failure |

Authentication, TLS, and rate limiting are not built in. Use loopback with an SSH tunnel.
