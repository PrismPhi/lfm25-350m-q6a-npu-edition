**English version -> [FINDINGS.md](FINDINGS.md)**

# 技術知見

この文書は、公開ランナーに至るまでに再利用可能と判断した技術的結論をまとめたものです。履歴上の実験ラベルは[用語集](GLOSSARY.ja.md)で定義しており、製品バージョンではありません。

## 実行範囲

- QNNExecutionProvider/HTPは、CPU EP fallbackを無効にした状態でモデル本体のchunk prefillとtoken decode graphを実行します。
- tokenizer、detokenizer、rowwise-int8 embedding lookup、sampling、stop処理、JSON framing、cache bookkeeping、mask、position、RoPE定数生成はhost処理です。
- providerが登録されただけではNPU実行の証明になりません。fallback無効でのsession生成、実行成功、profile上のQNN-only provider countがそろって初めてQNN成功と判定します。

## graph構築

- decomposed GQAは、KV headの反復に`Slice` + `Concat`を使うとHTP v68で実行できます。dynamic `Tile`はQNNに拒否されました。
- quantizerの対象operatorには、`Slice`、`Concat`、`Transpose`、`Reshape`などのshape/data-movement operatorも含める必要があります。算術operatorだけを量子化すると、それらのnodeがCPUへ逃げる場合があります。
- repeated KV、転置済みkey、score、mask適用後score、probability、context、context outputには、layer形式のQDQ境界と安全なactivation scaleが必要です。
- 右寄せcausal tail maskにより、padding済みまたは無効なcache行がattentionへ漏れるのを防げます。mask値とactivation scaleは組み合わせて検証する必要があります。
- exact `LpNormalization`ベースのnorm書き換えは、CPU parityを保ちながらQNN-onlyで動作しました。`SimplifiedLayerNormalization`は検証stackで非対応で、primitiveへ分解したnormは数値的に不安定でした。
- Sigmoid経路を量子化すると固定反復と意味崩壊が発生しました。公開graphではSigmoidを量子化対象から外しています。

## 数値parity

- official入力注入は、attention実装の問題とupstream hidden stateの問題を最短で分離する方法です。official q/k/vを与えたGQA minigraphは高い一致を示し、attention patternとprojection weight layoutが主要原因ではないことを確認しました。
- 初期full graphで弱く見えたV projectionは、V行列layoutではなく、上流hidden stateとnorm誤差が原因でした。
- scalar QDQ scale探索は改善が頭打ちになりました。品質修復にはchannelwise/outlier-awareな境界処理、またはexport/QDQ topologyの再構築が必要です。
- top logit付近ではcosine similarityだけでは不十分です。top-1一致、生成token、subject一致、反復・崩壊検査を同時にgateへ含める必要があります。

## cacheと性能

- prefill-to-decode handoffはtensor shapeだけでなく機能上の契約です。論理cache長には実prompt token数を使い、padding行はmaskされたままにします。
- decode時に新規KVだけを返すとthroughputは改善しますが、公開runtimeのKVはhost保持です。そのためhost I/O削減がdecode高速化と長context化の主要経路です。
- prefill chunkを広げると単体benchmarkが速くても、実用可能とは限りません。chunk32実験はpadding済みcache/position handoffが一致しなかったため不採用とし、公開構成はchunk16を維持しています。
- 短いpromptでは、最初のpartial chunkがdecode経路を通るためAPI prefill rateが低く出る場合があります。この場合の対話待ち時間はTTFTの方が実態を表します。

## runtimeの信頼性

- QNN context `.bin`が生成されても、graphのloadや実行成功は保証されません。生成、再load、実行、provider profileを別々のgateとして確認します。
- HTP graph失敗後はrpcmem/CDSPのprocess stateが次の正常graphへ影響する場合があります。再起動を検討する前に、失敗processを終了して既知の小さなhealth checkを実行します。
- EPContextは対象device上で生成し、配布には含めません。raw QDQとhost資産をportableな配布境界とします。
- 公開installerは全配布資産を検証し、両EPContextを生成して、QNN-only canary、通常API応答、制約付きJSON応答を確認します。

## APIと出力処理

- byte-level tokenを個別decodeせず、累積token列をdetokenizeすることで不完全UTF-8断片によるreplacement characterを防ぎます。
- 長文出力では、model EOS、利用者stop文字列、context上限、server transport終了を別々に扱う必要があります。
- JSON modeが保証するのはobject構文です。値の事実性やapplication固有schemaは保証しません。
- QNN generationは単一process内で直列化されます。HTTP connectionは並行に受けられますが、model実行は単一runtime lockで保護されます。

詳細な測定値と失敗実験は[証跡索引](records/EVIDENCE_INDEX.ja.md)から参照できます。
