**English version -> [CHUNK32.md](CHUNK32.md)**

# Part 0 chunk32決定

## 結論

単一chunk32/ctx2048候補は不採用です。速度gateとQNN-only gateは通過しましたが、chunk-to-decode handoffの機能gateを通りませんでした。追加候補は生成しません。

| 項目 | 結果 | Gate |
|---|---:|---|
| QNN-only create/load/run | pass | 必須 |
| 初回context生成側 | 262.19 tok/s | >=250 tok/s |
| warm-load | 296.33 tok/s | >=250 tok/s |
| cache最小cosine | 0.3098 | fail |
| step0 logits最小cosine | 0.7537 | >=0.999 |
| 生成top-1一致 | 2/48 | >=0.90 |
| same subject | 6/6 | 補助指標 |

速い単体graphでもhandoff後のcache/logitsが一致しないため、実用runnerへ採用できません。headlineはchunk16のままです。

raw候補SHA-256は`776bce99eafa9d0a536a1817c9a1a2f54681840476fd7a788cc79437db261286`、Q6A証跡archive SHA-256は`54baef6f1195c5a10e4754e6087eb8d692057e3e99197e30a6e59076fc4db47b`です。公開ツリーには巨大raw/context binaryを含めず、数値JSONだけを保存します。
