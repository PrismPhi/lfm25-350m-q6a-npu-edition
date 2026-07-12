**English version -> [PUBLIC_SCOPE.md](PUBLIC_SCOPE.md)**

# 公開範囲

## 公開するもの

- `runner/`: installer、OpenAI互換server、CLI client、QNN runtime helper、設定、test
- `records/`: サニタイズ済みphase summary、決定台帳、数値JSON、証跡索引
- 日英同格のREADME、技術知見、用語集、再現手順、失敗パターン、移植、制限、API、ライセンス、NOTICE
- `scripts/prepare_model_release.py`: QDQ/host資産staging作成
- `scripts/audit_release.py`: 公開前監査
- 別モデル配布: 公開済みQDQ 2本、tokenizer、rowwise-int8 embedding、RoPE cache、MODEL_LICENSE

## 公開しないもの

- 個人パス、実機hostname/IP、SSH情報、秘密情報
- 元の非公開監査ツリーと未加工log
- QNN/QAIRT/ORT-QNN共有library
- EPContext/`*_qnn.bin`
- GGUF本体
- venv、core dump、profileの個人環境情報
- 不採用の巨大候補ONNX

## recordsに留めるもの

- V1.8b/V1.9/V1.10再量子化失敗の要約
- chunk32 Part 0の速度通過/機能不通過
- fresh installと冪等再実行の機械可読結果
- [V2a/V2b台帳](records/V2_LEDGER.ja.md)

## 公開状態

1. GitHub: <https://github.com/PrismPhi/lfm25-350m-q6a-npu-edition>
2. Hugging Face: <https://huggingface.co/PrismPhi/lfm25-350m-q6a-npu-edition>
3. 公開更新では日英レビューと自動release監査を行い、runtime動作を変更した場合はQ6A実機検証も行う。

公開後の更新でもcredential、個人環境情報、QNN/QAIRT/EPContext binaryを追加しません。
