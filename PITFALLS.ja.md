**English version -> [PITFALLS.md](PITFALLS.md)**

# 既知の失敗パターンと診断

表中の履歴ラベルは[用語集](GLOSSARY.ja.md)で定義しています。

| ルート | 症状 | 原因 | 回避策 | 公開証跡 |
|---|---|---|---|---|
| fp16 MatMul | QNN finalize/partition失敗 | v68 stackで対象patternが成立しない | fp16対応を仮定せずminigraphから始める | `records/evidence/experiment-ledger.json` |
| MatMulNBits | unsupported/fallback | QNN EPのweight-only loweringと不一致 | per-tensor A16W8 QDQ MatMulを使う | `records/PHASE_SUMMARY.ja.md` |
| float island | graph分割、CPU assignment | QDQ境界間にfloat operatorが残る | layer単位でQDQ境界を連続化 | `records/PHASE_SUMMARY.ja.md` |
| norm分解 | CPUでは近いがQNNで崩壊 | primitive列がQNN量子化と敵対 | exact LpNormalization書き換え（履歴ラベルN4b） | `records/PHASE_SUMMARY.ja.md` |
| SimplifiedLayerNormalization | QNN fallback/session失敗 | 検証したprovider stackでは非対応 | 検証済みexact LpNormalization書き換えを使う | `records/PHASE_SUMMARY.ja.md` |
| mask x scale | attentionが固定化/漏洩 | 大きい負maskとactivation scaleが相互作用 | mask値-64.0、safe scale、invalid-past test | `records/evidence/chunk32-part0.json` |
| eps支配 | RMSNorm amplitude崩壊 | 小入力でepsが支配 | official epsilonを固定しminigraph parity | `records/PHASE_SUMMARY.ja.md` |
| cosine盲点 | cosine高値でもtokenが違う | 小さいlogit差がtop-1を反転 | cosineとtop-1、生成subjectを同時gate | `records/evidence/experiment-ledger.json` |
| pad行リーク | chunk単体は速いがhandoff崩壊 | pad/cache行がdecodeへ混入 | actual-token cache長、tail mask、handoff gate | `records/evidence/chunk32-part0.json` |
| Tile GQA repeat | QNN reject | dynamic Tile shapeがv68で非対応 | Slice+Concat repeatを使う | `records/PHASE_SUMMARY.ja.md` |
| quantizer対象operator不足 | Slice/Concat graphがCPUへ逃げる | shape/data-movement operatorが対象listにない | Slice、Concat、Transpose、Reshapeも含める | `records/PHASE_SUMMARY.ja.md` |
| Sigmoid量子化 | 固定反復または意味崩壊 | activation量子化がgated MLP経路を壊す | Sigmoidを量子化対象から外す | `records/PHASE_SUMMARY.ja.md` |
| u8 KV | invalid tensor/config | cache dtype/quant contractが不一致 | float host cache + QDQ graph boundary | `records/PHASE_SUMMARY.ja.md` |
| rpcmem lifecycle | 次の正常graphまで失敗 | 失敗後のHTP/CDSP process state | process終了、health canary、必要時のみ再起動 | `records/PHASE_SUMMARY.ja.md` |
| shared memory off | exit 139 | stackのmemory modeとgraphが不整合 | 既定shared-memory設定を維持 | `records/PHASE_SUMMARY.ja.md` |
| V1.8b再量子化 | PC pass後QNN session失敗 | repeated SliceがCPU assignment | 公開済みの既知正常contextを維持 | `records/evidence/experiment-ledger.json` |
| V1.9事後統一 | PC logits崩壊 | range統一が量子化誤差を増幅 | PC gateで実機転送を止める | `records/evidence/experiment-ledger.json` |
| V1.10 group-max | PC pass後QNN session失敗 | range修正はpartition topologyを直さない | V2aでexport/QDQ境界から再構築 | `records/evidence/experiment-ledger.json` |
| chunk32 | 296.33 tok/sだがhandoff失敗 | cache pad/position contractの不一致 | 不採用、chunk16を維持 | `records/evidence/chunk32-part0.json` |
| full-graph GQA不一致 | attention minigraphが正しくてもq/k/vが悪い | upstream hidden stateまたはnorm誤差 | attention weight/layout変更前にofficial q/k/vを注入する | `records/PHASE_SUMMARY.ja.md` |
| 古いEPContext再利用 | runtime変更後も旧contextをloadする | sourceだけのstampにQNN stack identityがない | `source-stamp.json`のruntime fingerprintを比較し、不一致なら再生成 | `records/evidence/install-validation.json` |
| ADSP path上書き | ユーザーDSP pathが消える、またはHTP loadが環境依存になる | QNN登録が`ADSP_LIBRARY_PATH`を置換する | 順序を保持し、空/重複entryを除き、必須pathを追加 | `records/evidence/install-validation.json` |
| compilerだけ成功 | contextはあるが出力がNaN/Inf、またはCPU実行 | generate/load/runが単一の厳格gateになっていない | 生成・再ロード後にfinite出力、QNN countが0より大、CPU countが0を必須化 | `records/evidence/install-validation.json` |
| 構文だけのinstall smoke | JSONがparseできるため誤subjectが通る | 非空text/object構文だけを確認 | 固定`Tokyo` semantic canaryとfirst token ID `40550` | `records/evidence/install-canary-golden.json` |
| streaming stop漏洩 | stop stringの一部をclientへ送る | token delta送信後にstop検出する | multi-token/Unicode stopでないと確定するまでprefixをhold | `runner/tests/test_public_runtime.py` |
| 遅いstreamが推論を占有 | 切断済み/低速socketがQNN lockを保持 | inferenceがnetworkへ直接writeする | bounded delta queue、write timeout、cancellation | `runner/tests/test_public_runtime.py` |
| shutdown/profile競合 | generation中に`end_profiling()`が走る | daemon request threadがengineより長く生存 | drain、cancel/wait、engine lock取得、profile、closeの順にする | `runner/tests/test_public_runtime.py` |
| moving HF revision | 古いGit commitが別資産または取得不能資産をinstallする | 既定URLが`resolve/main` | revision `773ff42cc383cb61ecf32eb13d1f828634fbd0e1`へ固定し、明示overrideを残す | `runner/config/model-assets.json` |

## 最短診断順

1. runtime fingerprintを取得し、固定asset SHAを確認する。
2. 1 operatorのminigraphでfinite出力とQNN-only create/load/runを確認する。
3. layer canaryでQDQ境界とdtypeを確認する。
4. official入力を注入しweight/layoutを分離する。
5. cache、position、RoPE、tail-maskを個別tapで比較する。
6. logits cosineだけでなくtop-1と生成tokenを比較する。
7. prompt-to-textの前にhandoff parityを通す。
8. semantic canaryとshutdown後profile gateを通す。

QNNが`.bin`を生成しただけでは成功ではありません。session作成、graph実行、全出力finite、`QNNExecutionProvider > 0`、`CPUExecutionProvider == 0`が揃い、context再ロード後にも同じ確認を通して初めてQNN成功です。
