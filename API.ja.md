**English version -> [API.md](API.md)**

# OpenAI互換API

## Endpoint

| Method | Path | 内容 |
|---|---|---|
| GET | `/health` | readiness、backend、fallback、queue |
| GET | `/v1/models` | 1 modelの一覧 |
| POST | `/v1/chat/completions` | 非stream/stream chat completion |

既定base URLは`http://127.0.0.1:18080/v1`です。

## 対応request

- `model`: `lfm2.5-350m-qnn-ctx2048`または`lfm2.5-350m`
- `messages`: roleが`system`、`user`、`assistant`の文字列content
- `stream`: boolean
- `max_tokens`: 1-1024
- `temperature`: 0-2
- `top_p`: (0,1]
- `top_k`: 1-4096
- `repetition_penalty`: 0.1-2
- `repetition_last_n`: -1-2048
- `min_new_tokens`: 0-1024
- `seed`: integer
- `stop`: stringまたは最大4 string
- `logit_bias`: token IDから[-100,100]へのobject
- `profile`: `chat`または`extraction`
- `response_format`: `{"type":"text"}`または`{"type":"json_object"}`

`tools: []`と`tool_choice: "auto"`はOpenWebUI互換のため受理して無視します。toolを実行せず、tool callも生成しません。

## 通常request

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-350m-qnn-ctx2048","messages":[{"role":"user","content":"日本の首都は？"}],"max_tokens":64}'
```

## JSON mode

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-350m-qnn-ctx2048","messages":[{"role":"user","content":"日本の首都をJSONで返して"}],"response_format":{"type":"json_object"},"max_tokens":64}'
```

JSON modeは常に固定の単一キーobject `{"answer": "<value>"}`を返します。構造トークンは固定で、値はQNN logitsから選ばれ96文字でcapされます。任意またはユーザー指定のschemaは非対応です。responseの`qnn_metrics.json_object_valid`と`qnn_metrics.json_check`を確認してください。

## Streaming

`stream=true`はSSE形式でdeltaを返し、最後に`data: [DONE]`を送ります。UTF-8 byte-level tokenをtoken単位でdecodeせず、累積ID列からdecodeするため、途中の置換文字を抑制します。

## Metrics

`qnn_metrics`にはprefill/decode tok/s、TTFT、実token cache長、finish reason、sampling、JSON検証、fallback状態が入ります。session生成時間はTTFTに含みません。

## Error

| code | HTTP | 意味 |
|---|---:|---|
| `invalid_request` | 400 | field/type/rangeが不正 |
| `context_length_exceeded` | 400 | prompt + max_tokensが2048を超過 |
| `not_found` | 404 | routeが未対応 |
| `generation_error` | 500 | QNN/runtime generation失敗 |

認証、TLS、rate limitは内蔵しません。loopback + SSH tunnelで使用してください。
