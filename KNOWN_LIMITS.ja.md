**English version -> [KNOWN_LIMITS.md](KNOWN_LIMITS.md)**

# 既知制限

## 長文品質

Hybrid QNN / CPU Q8のcompletion長比は平均0.621で、0.70以上は3/6 promptです。V1.8b、V1.9、V1.10という3件の履歴実験で、単純な再校正、事後range統一、校正時group-max制約を試しましたが、公開品質を満たす改善にはなりませんでした。ラベルの意味は[用語集](GLOSSARY.ja.md)で説明しています。

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

公開contextは2048です。ctx4096はmemory上の生成可能性を確認しましたが、device-resident KVでない現構造ではdecode I/Oが大きく、研究候補V2bへ延期しました。

## chat template

serverはsystem/user/assistantの文字列contentを扱う限定ChatML rendererです。jinja2がない環境でも動作します。画像、audio、任意tool execution、複雑なcontent part、assistantで終わるpromptは未対応です。

## JSON

`response_format: {"type":"json_object"}`は常に固定の単一キーobject `{"answer": "<value>"}`を返す制約モードで、値はQNN logitsから選ばれ96文字でcapされます。任意またはユーザー指定のschemaは非対応で、値の事実性は保証しません。free-form promptだけでは有効JSONを保証しません。

## 並列性

推論engineは引き続き1 process内で直列化され、QNN generationを1件ずつ実行します。既定のadmission上限は待機4件、待機上限30 sです。超過は429、待機timeout/drainingは503を返します。Streamingはbounded 32-delta queueを使うため、QNN lock保持中にnetwork writeを行いません。

## streamingとshutdown

stop string指定時はholdback bufferを使うため、multi-token/Unicode stopの可能性があるprefixは通常textと確定するまで遅延します。stop未指定時はholdback遅延を加えません。client切断またはwrite timeout時はcancellationを要求します。cancellationはQNN graph runの間で検出し、provider内ですでに実行中のgraph runは中断できません。

SIGINTとSIGTERMはdrainingへ移行し、新規workを拒否し、active generationをcancel/waitし、engine lock取得後にprofile終了、session解放、最終result書き込みを行います。既定write timeoutは5 s、shutdown待機上限は30 sです。

## runtime互換性

EPContextを再利用できるのは、ONNX Runtime/onnxruntime-qnn versionとlibrary hash、HTP Stub/Skel hash、provider/session option、`QCS6490 / v68`、chunk、total lengthを含む`source-stamp.json` identityが完全一致する場合だけです。不一致または壊れたstampでは再生成します。検証packageは対応QAIRTまたはQualcomm QNN runtime version APIを公開しないため、それらのfieldは`null`です。絶対library pathとSHA-256で検証runtimeを識別します。

既定Hugging Face revisionは`773ff42cc383cb61ecf32eb13d1f828634fbd0e1`へ固定しています。明示的なmirror/revision overrideは使えますが、公開既定検証の対象外です。

## logging

常駐request履歴は既定上限128のbounded dequeです。既定ではmetadataだけを保存し、prompt text、generated text、token ID listは保存しません。`--log-bodies`で明示的に本文保存を有効化できます。標準出力のrequest logもmetadata-onlyです。

## 電力とmemory

world-readableな`power_now`/hwmon power sourceがないため、消費電力は未測定です。thermal proxyのみ記録しました。RSSはhost process値で、DSP側memoryを完全には含まない可能性があります。

## 配布

QNN/QAIRT library、EPContext binary、GGUFは同梱しません。EPContextは対象Q6Aで生成します。GGUFは公式配布先へのリンクだけを案内します。
