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

## ドメイン概念

| 概念 | 背景的な意味（このrepo固有ではない一般概念） |
|---|---|
| QDQ | Quantize-Dequantize表現。グラフ中に明示的なQuantizeLinear／DequantizeLinearノードを置き、tensorの量子化scaleやzero pointをbackendへ伝える表現方式。対応backendでは整数・量子化演算へ変換できるが、QDQがあるだけで必ず整数実行になるとは限らない。 |
| A16W8 | activationを16-bit、weightを8-bitで表現する量子化方針。activationに広いdynamic rangeや精度を持たせやすく、A8W8より量子化誤差を抑えられる場合がある一方、メモリ帯域や計算コストが増える場合がある。 |
| GQA | Grouped-Query Attention。query head数をKV head数より多くし、複数のquery headで同じkey/value headを共有する方式。実装上はquery head数へ対応させるrepeat、broadcast、head mappingなどが必要になるが、必ず物理的に複製するとは限らない。 |
| RoPE | Rotary Position Embedding。位置埋め込みを加算する代わりに、queryとkeyの成分対を位置依存の角度で回転させ、attentionへ相対的な位置情報を注入する方式。 |
| activation range collapse | calibrationがactivationの実際の振幅や外れ値を十分に捉えられず、量子化rangeが過度に狭くなる現象を指す説明的な診断語。実行時の値が飽和・clipされ、誤差が後段へ蓄積して出力品質を劣化させる。 |
| LpNormalization | 指定axisに沿ってLp normを求め、入力をそのnormで割って正規化したtensorを返すONNX演算子。norm値そのものを返す演算子ではない。本プロジェクトでは、targetが直接対応しない正規化処理を等価な演算へ分解する際の構成要素として利用する。RMSNormのdrop-in代替ではなく、等価なRMSNorm分解では次元由来係数、epsilon、学習済みweightも別途反映する必要がある。 |
