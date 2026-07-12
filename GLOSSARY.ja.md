**English version -> [GLOSSARY.md](GLOSSARY.md)**

# 用語集

| 用語 | 本リポジトリでの意味 |
|---|---|
| QNN-only | profile対象のモデル本体graphが`QNNExecutionProvider`だけで実行され、CPU EP fallbackが無効な状態です。host側の前処理とbookkeepingは許可範囲を明記して使用します。 |
| EPContext | 次回以降のloadを高速化するdevice生成QNN contextです。hardware/stack固有なので本配布には含めません。 |
| chunk16 | ctx2048でprompt tokenを十六個単位に処理するprefill graphです。 |
| slim decode | full cacheではなく新規生成KVだけを返すsingle-token decode graphです。 |
| handoff | prefillからdecodeへhidden/cache/position stateを渡す処理です。tensor shapeだけでなく、論理token長とmask semanticsも含みます。 |
| minigraph | full modelへ統合する前に、operatorまたはsubgraphの生成、load、実行、CPU reference一致を確認する小graphです。 |
| canary | runtime healthやpartitionを早期確認するための小さな既知正常実行です。 |
| gate | CPU parity、QNN-only実行、handoff parity、可読生成など、文書化した合格条件です。 |
| oracle | 比較基準です。通常は公式CPU Q8 modelを指します。 |
| Path A2 | 公式weightを使ったgraph再構築系列の履歴ラベルです。公開model形式や製品名ではありません。 |
| N4b | 成功したQNN graph系列で使ったexact `LpNormalization`ベースnorm書き換えの履歴ラベルです。 |
| V0/V1/V1.8b/V1.9/V1.10 | 証跡追跡のために残した過去の実験group名です。semantic release versionではありません。 |
| V2a/V2b | 今後の研究候補です。それぞれexport/QDQ品質再構築とdevice-resident KV/runtime改善を指します。 |
| Part 0 | chunk32 prefill実験の履歴ラベルです。速度確認は通過しましたが、機能上のhandoff parityに失敗しました。 |
