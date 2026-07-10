**English version -> [README.md](README.md)**

# LFM2.5-350M Q6A NPU Edition

LFM2.5-350MをQCS6490/Q6AのQNN HTPで動かす、実験的なprompt-to-textランナーです。CPUはtokenizer、rowwise-int8 embedding lookup、sampling、stop処理、cache bookkeepingを担当し、モデル本体のchunk prefill/decodeはQNNExecutionProviderで実行します。CPU EP fallbackは無効です。

このツリーは[GitHub](https://github.com/PrismPhi/lfm25-350m-q6a-npu-edition)で公開し、モデル資産とHFモデルカードは[Hugging Face](https://huggingface.co/PrismPhi/lfm25-350m-q6a-npu-edition)で公開します。

## 確定構成

| 項目 | 実測 | 条件 |
|---|---:|---|
| context | 2048 token | chunk16 prefill + slim decode |
| API prefill | 33.87-143.53 tok/s | 3つの実用API task、prompt依存 |
| API decode | 17.00-17.60 tok/s | 同じ3 task |
| TTFT | 0.31-1.03 s | session生成時間を除外 |
| 長文 | 442 completion token | 日本語説明、通常stop |
| strict JSON | valid | `{"answer":"東京"}` |
| resident server RSS | 758 -> 813 MiB | final API sample前後 |
| 消費電力 | 未測定 | world-readable telemetryなし、thermal proxyのみ |
| fresh install | 62.2 s | Q6A、ローカル資産、EPContext生成+smokeまで |
| idempotent rerun | 5.5 s | 11資産と両contextを再利用 |

API prefillの下限は、短promptの初回partial chunkがdecode経路を通るため低く出ます。対応taskのTTFTは0.31 sで、対話上の実待ち時間は0.31 sです。

## CPU比較

| Phase | Backend | Throughput | 測定法 |
|---|---|---:|---|
| prefill | Hybrid QNN chunk | 160-191 tok/s | chunk16、ctx2048 |
| prefill | CPU Q4 `llama-bench` | 112.6 tok/s | prompt processing、ctx2048 |
| decode | Hybrid QNN | 16.28 tok/s | 制御API比較の平均 |
| decode | ORT CPU Q8 | 14.73 tok/s | 制御API比較の平均 |
| decode | CPU Q4 `llama-bench` | 24.9 tok/s | 別`llama-bench`測定、参考値 |

prefillは両方ctx2048の実測ですが、Hybrid QNNはchunk graph、CPU Q4は`llama-bench`です。decodeのHybrid QNNとORT CPU Q8は制御API比較、CPU Q4は別`llama-bench`の参考値です。

## 要件

- QCS6490/Q6A、Linux aarch64
- Qualcomm QNN/QAIRTを利用できるユーザー環境
- QNN対応ONNX Runtime Python環境
- 約2.5 GiBの空き領域
- テスト済み: Python 3.12.3、ONNX 1.22.0、ONNX Runtime 1.27.0、tokenizers 0.23.1

QNN/QAIRT共有ライブラリやEPContextバイナリは本配布に含みません。

## クイックスタート

モデル資産はHugging Faceの公開repositoryから取得します。

```bash
export LFM25_MODEL_BASE_URL="https://huggingface.co/PrismPhi/lfm25-350m-q6a-npu-edition/resolve/main"
bash runner/install.sh --python /path/to/qnn-venv/bin/python
bash runner/start_server.sh
```

別ターミナルから:

```bash
python3 runner/scripts/client.py --prompt "日本の首都は？" --max-tokens 64
python3 runner/scripts/client.py --prompt "日本の首都をJSONで返して" --json-object --max-tokens 64
```

`install.sh`は依存確認、11資産のSHA-256検証、デバイス上EPContext生成、QNN-only canary、通常応答、JSON応答まで実行します。失敗時は`dependencies`、`assets`、`epcontext`、`smoke`の段階名を表示します。

## OpenWebUI

サーバーは既定でloopbackの`127.0.0.1:18080`だけをlistenします。別PCやコンテナからはSSH tunnelを推奨します。

```bash
ssh -N -L 18081:127.0.0.1:18080 q6a-user@q6a-host
```

OpenWebUIのOpenAI互換base URLを`http://host.docker.internal:18081/v1`、API keyを任意の非空文字列に設定します。LAN bindは認証/TLSを提供しないため、信頼できないネットワークでは使用しないでください。

## ライセンス

コードはApache License 2.0です。派生QDQ、embedding、RoPE資産はLFM Open License v1.0です。同ライセンスは派生物再配布を認めますが、ライセンス同梱、変更表示、帰属保持が必要です。商用利用の無償範囲は年間売上$10,000,000未満です。詳細は[LICENSES.ja.md](LICENSES.ja.md)と[MODEL_LICENSE](MODEL_LICENSE)を確認してください。

## 読む順番

- [再現手順](REPRODUCIBILITY.ja.md)
- [死亡ルートと診断](PITFALLS.ja.md)
- [他HTP世代への移植](PORTING.ja.md)
- [既知制限](KNOWN_LIMITS.ja.md)
- [API仕様](API.ja.md)
- [公開範囲](PUBLIC_SCOPE.ja.md)
- [研究記録](records/PHASE_SUMMARY.ja.md)
