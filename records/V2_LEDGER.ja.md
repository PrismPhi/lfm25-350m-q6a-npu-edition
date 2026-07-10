**English version -> [V2_LEDGER.md](V2_LEDGER.md)**

# V2 次期プロジェクト台帳

この台帳は公開後に開始する次期projectを定義します。現行V1での対応済み機能を示すものではありません。

| Track | 目的 | 現在の証跡 | 完了gate |
|---|---|---|---|
| V2a | export/QDQ topologyをchannelwise/outlier-awareに再構築 | 手法実証済み、V1.8b/V1.9/V1.10の失敗標本3点 | CPU Q8 quality parity、QNN-only partition、6 prompt smoke |
| V2b | KVをdevice-residentにしhost I/Oを削減 | V1はhost cache、ctx2048 | 25-30 tok/s、ctx4096、actual-token cache、QNN-only profile |

## 優先順位

V2aは量子化境界とpartition topologyを修復する品質trackです。V2bはdecode runtimeとcontextを改善する性能trackです。片方の見かけの速度だけで現行V1を置き換えず、quality、fallback無効、cache契約、長文完走を同時に満たした時だけ採用候補とします。

## 公開時の扱い

公開時にissueまたはprojectとして登録し、各実験は再現command、入力SHA-256、QNN profile、採否理由を残します。未検証の他HTP世代への対応や25-30 tok/sを達成済みとは表記しません。
