**English version -> [PORTING.md](PORTING.md)**

# 他HTP世代への移植

本プロジェクトで実機確認したのはQCS6490のHTP v68 stackです。v69/v73/v75を「対応済み」とは表明しません。公開するのは、同じ結論を再検証するための移植性テスト手順です。

## 移植性テストキット

| Stage | 入力 | 合格条件 |
|---|---|---|
| P0 environment | QNN対応Python | runtime fingerprint、provider登録、HTP library load |
| P1 operator minigraph | MatMul、Conv、LpNormalization、Slice、Concat | create/load/runがfiniteかつQNN-only |
| P2 attention minigraph | q/k/v、RoPE、tail-mask | CPU cosine >=0.999、top-1一致 |
| P3 layer canary | Conv layer、attention layer、MLP | finite、fallback 0 |
| P4 full chunk | chunk16/ctx2048 | warm-load、cache output shape |
| P5 decode | chunk1 slim cache | >=15 tok/sを目安、QNN-only |
| P6 handoff | chunk -> decode | cache/logits/top-1 gate |
| P7 prompt-to-text | 6 smoke + JSON | UTF-8、subject、固定JSON object、first-token golden |

## 世代ごとに再検証する項目

- fp16 MatMul/Convがnative partitionされるか。
- MatMulNBitsやblockwise quantizationがQNN EPでloweringされるか。
- LpNormalization/RMSNorm相当operatorがnative対応するか。
- Slice+Concat GQA repeatをTileへ戻せるか。
- uint8 KV cacheが入力/出力contractを保てるか。
- context binaryの互換性。EPContextは世代/stackごとに対象デバイス上で再生成する。
- 厳密なruntime identity。ONNX Runtime/onnxruntime-qnn version、EP/HTP/Stub/Skel hash、provider/session option、SoC/HTP generation、chunk、total length。QAIRT/Qualcomm QNN runtime versionは対応APIで取得できる場合だけ記録する。
- `ADSP_LIBRARY_PATH`の検索順序。環境を置換せず既存entryを保持して必須pathを追加する。
- VTCM、spill/fill、DDR帯域、shared-memory mode。
- 取得できたQAIRT/Qualcomm QNN runtimeまたはONNX Runtime QNN packageのversion差で既知の失敗パターンが変わるか。

## 実行手順

1. runtime fingerprintを取得し、`runner/scripts/generate_epcontext.py`で最小graphを生成する。
2. `session.disable_cpu_ep_fallback=1`を外さない。
3. create、load、runを別caseとして記録し、NaN/Inf出力をrejectする。
4. profile JSON countが`QNNExecutionProvider > 0`かつ`CPUExecutionProvider == 0`であることを必須にする。
5. 生成contextを再ロードし、run/finite/profile gateを繰り返す。
6. runtime identityを`source-stamp.json`へ保存し、不一致なら必ず再生成する。
7. 失敗後は同processを再利用せず、既知の小canaryでHTP healthを確認する。
8. full graphへ進む前にP0-P3のmatrixを保存する。
9. 未検証の期待値を固定せず、検証済みruntimeからsemantic canaryとfirst-token goldenを導出する。

## 移植時の判断

新世代でfp16やnative normが通っても、v68の回避策を機械的に残す必要はありません。反対に、provider登録だけ成功してもfull graph対応を意味しません。各世代の最小graph結果からexport/QDQ topologyを再選択してください。

性能値は同じprompt、context、sampling、wall定義で再測定します。本リポジトリの17.00-17.60 tok/sを他世代の期待値として保証しません。

公開既定asset revision `773ff42cc383cb61ecf32eb13d1f828634fbd0e1`はv68検証資産を識別しますが、EPContextの移植可能性を証明しません。raw QDQはSHA確認後のみ再利用でき、EPContextは再生成が必要です。
