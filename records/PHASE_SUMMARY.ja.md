**English version -> [PHASE_SUMMARY.md](PHASE_SUMMARY.md)**

# フェーズ別サマリー

以下のphase名は証跡追跡のために残した履歴上の実験IDで、semantic release versionではありません。[用語集](../GLOSSARY.ja.md)も参照してください。

| Phase | 成果 | 残った判断 |
|---|---|---|
| V0 | prompt -> tokenizer -> QNN -> detokenizeを成立 | upstream hidden parityが主要課題 |
| V1 graph再構築（Path A2/N4b） | 公式weight移植、exact LpNormalization書き換え、tail-mask、A16W8 | weight/layoutとruntime blockerを分離 |
| V1 chunk | chunk16/ctx2048 QNN-only prefill | host cache handoffが必須 |
| V1 slim decode | new-only KV outputで約17 tok/s | device-resident KVはV2b |
| V1.7 | OpenAI互換APIとWebUI | 長文の早期stop/UTF-8を追跡 |
| V1.8b | chat profile、min_new_tokens、logit_bias、JSON mode | 長文比0.621 |
| V1.9 | 事後range統一をPCで否定 | scale統一だけでは品質維持不可 |
| V1.10 | 校正時group-maxをPCで実証 | QNN partition topologyは未修復 |
| Public Part 0 | chunk32 296.33 tok/sを実証 | handoff失敗で不採用 |
| Public install | local fresh 62.2 s、public URL fresh 126.9-288.5 s、rerun 5.5-5.8 s | QNN環境は利用者の前提条件 |

## 公開構成

公開構成はchunk16 + slim decode、ctx2048、QNN fallback無効です。API decodeは17.00-17.60 tok/s、strict JSONは有効、442 completion tokenの日本語長文を通常stopで完走しました。

## 試した改善と公開しなかった理由

V1のcalibration-only leverはV1.8b、V1.9、V1.10の3実験で閉じました。chunk32は速度目標を通過しましたがhandoff gateで閉じました。次の技術作業はV2aのexport/QDQ topology再構築か、V2bのdevice-resident KV runtimeです。
