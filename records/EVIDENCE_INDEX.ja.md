**English version -> [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md)**

# 公開証跡索引

| 証跡 | 内容 |
|---|---|
| `evidence/headline-metrics.json` | 公開API性能、CPU比較、長文比 |
| `evidence/experiment-ledger.json` | V1.8b/V1.9/V1.10再量子化3実験 |
| `evidence/chunk32-part0.json` | chunk32速度、QNN-only、handoff不採用 |
| `evidence/install-validation.json` | historical installと現在のruntime-contract upgrade/rerun、semantic canary、finite/profile gate |
| `evidence/install-canary-golden.json` | 決定論的canaryの検証済み`Tokyo`出力、token ID、QNN-only source run |
| `evidence/url-validation-20260723.json` | 公開link 10件と固定asset 11件のHTTP 200検証 |
| `../runner/config/model-assets.json` | 配布11資産のsize/SHA-256と固定HF revision |
| `../runner/config/install-canary.json` | 公開canary prompt、期待subject、seed、first-token golden |
| `decisions/CHUNK32.ja.md` | chunk32判断記録（履歴ラベルPart 0） |

公開JSONは個人パス、hostname/IP、credential、未加工profileを含みません。元の監査archiveはPC側で凍結保存し、公開ツリーへはコピーしません。

`evidence/install-validation.json`の2026-07-23より前のentryはhistoricalな構文only JSON checkで、変更せず残しています。現在の厳格検証は`runtime_contract_validation_2026_07_23` entryです。11資産を検証し、旧stamp schemaから両contextを再生成後に再利用し、通常/JSONの`Tokyo`、first token `40550`、finite出力、profile後QNN-only countを確認しました。

## 外部検証元

- LFM2.5-350M: <https://huggingface.co/LiquidAI/LFM2.5-350M>
- 公開asset: <https://huggingface.co/PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu>
- 公開source: <https://github.com/PrismPhi/radxa-dragon-q6a-qcs6490-lfm2.5-350m-qnn-npu>
- LFM Open License v1.0: <https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE>
- Liquid AI license guide: <https://docs.liquid.ai/lfm/help/model-license>

## 数値規律

- tok/sの分母をwall、QNN run、whole-processで区別する。
- TTFTはsession生成を除外すると明記する。
- powerは未測定、thermalはproxyと明記する。
- QNN-onlyはprofile provider countで確認する。
- fallback設定、QNN session作成、profile検証済みQNN-onlyを別fieldとして扱う。
- finite出力とsemantic canaryを必須にし、構文only JSONでは不十分とする。
- compilerがcontext binaryを生成しただけでは成功としない。
