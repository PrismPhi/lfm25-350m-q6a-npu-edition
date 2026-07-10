**日本語版 -> [CHUNK32.ja.md](CHUNK32.ja.md)**

# Part 0 chunk32 Decision

## Decision

The single chunk32/ctx2048 candidate is rejected. It passed speed and QNN-only gates but failed the chunk-to-decode handoff functional gate. No additional candidate will be generated.

| Item | Result | Gate |
|---|---:|---|
| QNN-only create/load/run | pass | required |
| initial context-generation side | 262.19 tok/s | >=250 tok/s |
| warm load | 296.33 tok/s | >=250 tok/s |
| minimum cache cosine | 0.3098 | fail |
| minimum step0 logits cosine | 0.7537 | >=0.999 |
| generated top-1 match | 2/48 | >=0.90 |
| same subject | 6/6 | supporting metric |

A fast standalone graph is not usable when cache/logits after handoff do not match. The headline remains chunk16.

The raw candidate SHA-256 is `776bce99eafa9d0a536a1817c9a1a2f54681840476fd7a788cc79437db261286`. The Q6A evidence archive SHA-256 is `54baef6f1195c5a10e4754e6087eb8d692057e3e99197e30a6e59076fc4db47b`. The public tree excludes large raw/context binaries and preserves only numeric JSON.
