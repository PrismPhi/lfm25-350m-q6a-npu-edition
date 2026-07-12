**English version -> [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md)**

# 公開証跡索引

| 証跡 | 内容 |
|---|---|
| `evidence/headline-metrics.json` | 公開API性能、CPU比較、長文比 |
| `evidence/experiment-ledger.json` | V1.8b/V1.9/V1.10再量子化3実験 |
| `evidence/chunk32-part0.json` | chunk32速度、QNN-only、handoff不採用 |
| `evidence/install-validation.json` | local installと2回のpublic URL fresh/rerun検証、QNN-only smoke |
| `../runner/config/model-assets.json` | 配布11資産のsize/SHA-256 |
| `decisions/CHUNK32.ja.md` | chunk32判断記録（履歴ラベルPart 0） |

公開JSONは個人パス、hostname/IP、credential、未加工profileを含みません。元の監査archiveはPC側で凍結保存し、公開ツリーへはコピーしません。

## 外部検証元

- LFM2.5-350M: <https://huggingface.co/LiquidAI/LFM2.5-350M>
- LFM Open License v1.0: <https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE>
- Liquid AI license guide: <https://docs.liquid.ai/lfm/help/model-license>

## 数値規律

- tok/sの分母をwall、QNN run、whole-processで区別する。
- TTFTはsession生成を除外すると明記する。
- powerは未測定、thermalはproxyと明記する。
- QNN-onlyはprofile provider countで確認する。
- compilerがcontext binaryを生成しただけでは成功としない。
