**English version -> [KNOWN_LIMITS.md](KNOWN_LIMITS.md)**

# 既知制限

## 長文品質

Hybrid QNN / CPU Q8のcompletion長比は平均0.621で、0.70以上は3/6 promptです。V1.8b、V1.9、V1.10の3実験で、単純な再校正、事後range統一、校正時group-max制約を試しましたが採用可能な改善にはなりませんでした。

現行構成は442 tokenの日本語長文を通常stopで完走できますが、長さと事実品質は同じではありません。重要な説明・要約はCPU Q8など別oracleで確認してください。

## 利用者側ノブ

| knob | 既定 | 用途 |
|---|---:|---|
| profile | chat | temperature 0.8、top-k 40、top-p 0.95 |
| extraction profile | 明示指定 | temperature 0.1、top-k 50、top-p 1.0 |
| repetition_penalty | 1.1 | chatの反復抑制 |
| repetition_last_n | 64 | 直近64 tokenだけを対象 |
| min_new_tokens | 0 | 早いEOSを明示的に抑制 |
| logit_bias | なし | tokenごとの[-100,100] bias |

`min_new_tokens`やEOSへの`logit_bias`は品質保証ではありません。既定ではEOS biasを加えません。

## context

採用contextは2048です。ctx4096はmemory上の生成可能性を確認しましたが、device-resident KVでない現構造ではdecode I/Oが大きく、V2bへ延期しました。

## chat template

serverはsystem/user/assistantの文字列contentを扱う限定ChatML rendererです。jinja2がない環境でも動作します。画像、audio、任意tool execution、複雑なcontent part、assistantで終わるpromptは未対応です。

## JSON

`response_format: {"type":"json_object"}`はobject構文を保証する制約モードです。値の事実性やschemaは保証しません。free-form promptだけでは有効JSONを保証しません。

## 並列性

推論engineは1 process内で直列化されます。HTTP connectionは並行に受けられても、QNN generationはlockで1件ずつ処理します。

## 電力とmemory

world-readableな`power_now`/hwmon power sourceがないため、消費電力は未測定です。thermal proxyのみ記録しました。RSSはhost process値で、DSP側memoryを完全には含まない可能性があります。

## 配布

QNN/QAIRT library、EPContext binary、GGUFは同梱しません。EPContextは対象Q6Aで生成します。GGUFは公式配布先へのリンクだけを案内します。
