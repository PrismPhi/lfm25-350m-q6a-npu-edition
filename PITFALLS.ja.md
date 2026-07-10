**English version -> [PITFALLS.md](PITFALLS.md)**

# 死亡ルートと診断

| ルート | 症状 | 原因 | 回避策 | 公開証跡 |
|---|---|---|---|---|
| fp16 MatMul | QNN finalize/partition失敗 | v68 stackで対象patternが成立しない | fp16対応を仮定せずminigraphから始める | `records/evidence/experiment-ledger.json` |
| MatMulNBits | unsupported/fallback | QNN EPのweight-only loweringと不一致 | per-tensor A16W8 QDQ MatMulを使う | `records/PHASE_SUMMARY.ja.md` |
| float island | graph分割、CPU assignment | QDQ境界間にfloat operatorが残る | layer単位でQDQ境界を連続化 | `records/PHASE_SUMMARY.ja.md` |
| norm分解 | CPUでは近いがQNNで崩壊 | primitive列がQNN量子化と敵対 | N4b exact LpNormalization pattern | `records/PHASE_SUMMARY.ja.md` |
| mask x scale | attentionが固定化/漏洩 | 大きい負maskとactivation scaleが相互作用 | mask値-64.0、safe scale、invalid-past test | `records/evidence/chunk32-part0.json` |
| eps支配 | RMSNorm amplitude崩壊 | 小入力でepsが支配 | official epsilonを固定しminigraph parity | `records/PHASE_SUMMARY.ja.md` |
| cosine盲点 | cosine高値でもtokenが違う | 小さいlogit差がtop-1を反転 | cosineとtop-1、生成subjectを同時gate | `records/evidence/experiment-ledger.json` |
| pad行リーク | chunk単体は速いがhandoff崩壊 | pad/cache行がdecodeへ混入 | actual-token cache長、tail mask、handoff gate | `records/evidence/chunk32-part0.json` |
| Tile GQA repeat | QNN reject | dynamic Tile shapeがv68で非対応 | Slice+Concat repeatを使う | `records/PHASE_SUMMARY.ja.md` |
| u8 KV | invalid tensor/config | cache dtype/quant contractが不一致 | float host cache + QDQ graph boundary | `records/PHASE_SUMMARY.ja.md` |
| rpcmem lifecycle | 次の正常graphまで失敗 | 失敗後のHTP/CDSP process state | process終了、health canary、必要時のみ再起動 | `records/PHASE_SUMMARY.ja.md` |
| shared memory off | exit 139 | stackのmemory modeとgraphが不整合 | 既定shared-memory設定を維持 | `records/PHASE_SUMMARY.ja.md` |
| V1.8b再量子化 | PC pass後QNN session失敗 | repeated SliceがCPU assignment | accepted contextを維持 | `records/evidence/experiment-ledger.json` |
| V1.9事後統一 | PC logits崩壊 | range統一が量子化誤差を増幅 | PC gateで実機転送を止める | `records/evidence/experiment-ledger.json` |
| V1.10 group-max | PC pass後QNN session失敗 | range修正はpartition topologyを直さない | V2aでexport/QDQ境界から再構築 | `records/evidence/experiment-ledger.json` |
| chunk32 | 296.33 tok/sだがhandoff失敗 | cache pad/position contractの不一致 | 不採用、chunk16を維持 | `records/evidence/chunk32-part0.json` |

## 最短診断順

1. 1 operatorのminigraphでQNN-only create/loadを確認する。
2. layer canaryでQDQ境界とdtypeを確認する。
3. official入力を注入しweight/layoutを分離する。
4. cache、position、RoPE、tail-maskを個別tapで比較する。
5. logits cosineだけでなくtop-1と生成tokenを比較する。
6. prompt-to-textの前にhandoff parityを通す。

QNNが`.bin`を生成してもsession初期化が失敗した場合は成功ではありません。fallback無効のsession作成、1回実行、profile provider countの3点が揃って初めてQNN成功です。
