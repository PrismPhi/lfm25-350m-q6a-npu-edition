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
- `n`: integer 1のみ
- `max_tokens`: 1-1024
- `temperature`: 0-2
- `top_p`: (0,1]
- `top_k`: 1-4096
- `repetition_penalty`: 0.1-2
- `repetition_last_n`: -1-2048
- `min_new_tokens`: 0-1024
- `seed`: integer 0-9223372036854775807
- `stop`: 1件の非空string、または1-4件の非空string配列。各1024文字以下
- `logit_bias`: token IDから[-100,100]へのobject
- `profile`: `chat`または`extraction`
- `response_format`: `{"type":"text"}`または`{"type":"json_object"}`

`tools: []`と`tool_choice: "auto"`はOpenWebUI互換のため受理して無視します。toolを実行せず、tool callも生成しません。

integer/float fieldではboolean値を拒否します。JSON parseと数値検証でNaN、Infinity、-Infinityを拒否し、`logit_bias` valueにも適用します。portは1-65535のintegerでなければなりません。

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

`stream=true`はSSE形式でdeltaを返し、最後に`data: [DONE]`を送ります。JSON parse、全field/model/message検証、tokenization、context長確認、queue admissionを200 SSE headerより前に完了します。header送信後のruntime failureはSSE `event: error`として送り、streamを終了します。別のHTTP responseは送ろうとしません。

UTF-8 byte-level tokenをtoken単位でdecodeせず、累積ID列からdecodeするため、途中の置換文字を抑制します。holdback bufferにより、設定したmulti-token/Unicode stop stringの全部または一部を送信しません。このためstreamとnon-streamのvisible textは同じstop semanticsになります。stop未指定時はdeltaを即時送信します。

## queueとcancellation

inferenceとsocket writeはbounded queueで分離します。既定stream queueは32 delta、request queueは待機4件、queue待機上限は30 sです。client切断または5 sのwrite timeoutはgeneration cancellationを要求し、現在のprovider runが戻った後に直列QNN slotを解放します。

## shutdownとlogging

SIGINTとSIGTERMはreadinessをfalse、`draining=true`とし、新規requestを拒否し、active workをcancel/waitし、QNN engine lockを取得後、profile終了、session解放、`server_result.json`書き込みの順に進みます。request threadはnon-daemonです。

request履歴は128 entryに制限します。prompt/generated本文とtoken ID listは既定offで、`--log-bodies`を指定した場合だけ保存します。metadata counterとerror typeは`/health`と最終resultへ残します。

## Metrics

`qnn_metrics`にはprefill/decode tok/s、TTFT、実token cache長、finish reason、sampling、JSON検証、first token ID、first-step top token ID、finite出力状態が入ります。session生成時間はTTFTに含みません。`max_tokens=1`では最終tokenを不要なdecode graph runへ投入しないため、decode run countは0です。

fallback表示は`fallback_configured_disabled`、`session_qnn_provider_created`、`qnn_only_verified`を区別します。最後の値はprofile終了時にprovider countが確定するまでserver実行中はfalseです。`server_result.json`ではchunk/decode profileが両方`QNNExecutionProvider > 0`かつ`CPUExecutionProvider == 0`の場合だけtrueになります。

## Error

| code | HTTP | 意味 |
|---|---:|---|
| `invalid_request` | 400 | field/type/rangeが不正 |
| `context_length_exceeded` | 400 | prompt + max_tokensが2048を超過 |
| `queue_full` | 429 | bounded待機queueが満杯 |
| `queue_timeout` | 503 | queue待機が設定上限を超過 |
| `server_draining` | 503 | shutdown開始済み |
| `generation_cancelled` | 503 | 通常response前にgenerationがcancelされた |
| `not_found` | 404 | routeが未対応 |
| `generation_error` | 500 | QNN/runtime generation失敗 |

認証、TLS、rate limitは内蔵しません。loopback + SSH tunnelで使用してください。
