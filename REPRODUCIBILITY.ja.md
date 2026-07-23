**English version -> [REPRODUCIBILITY.md](REPRODUCIBILITY.md)**

# 再現手順

## 再現の2段階

利用者向け再現は、配布済みA16W8 QDQからQ6A上でEPContextを生成する経路です。研究者向け再現は、公式LFM2.5-350Mから公開graphを再構築する経路です。履歴ラベルPath A2とN4bは、公式weightによるgraph再構築とexact LpNormalization書き換えを指します。詳細は[用語集](GLOSSARY.ja.md)を参照してください。研究者向け経路はQAIRT/operator差に敏感なため、full graphを評価する前に各中間合格条件を確認します。

## 利用者向け経路

1. QNN対応Pythonを準備する。
2. manifestで固定したrevisionから配布資産11件を取得し、`runner/config/model-assets.json`でSHA-256を照合する。
3. runtime fingerprintとmerge後の最終`ADSP_LIBRARY_PATH`を記録する。
4. chunk16/decode raw QDQから外部EPContextをデバイス上生成する。
5. 生成直後と再ロード後に実行し、全出力finite、QNN countが0より大きいこと、CPU countが0であることを必須にする。
6. 決定論的な通常/JSON semantic canaryを一時localhost serverで実行する。
7. `runner/start_server.sh`で常駐serverを開始する。

```bash
bash runner/install.sh --python /path/to/qnn-venv/bin/python
```

合格条件は、11資産のSHA一致、全出力finiteである厳格な生成・再ロード実行、chunk/decode profile countが`QNNExecutionProvider > 0`かつ`CPUExecutionProvider == 0`、通常応答が`Tokyo`を含むこと、JSON応答が`{"answer":"Tokyo"}`であること、first token IDが`40550`であること、profile後に`qnn_only_verified=true`となることです。

## runtime contract

| component | 検証済み値 |
|---|---|
| Python | `3.12.3` |
| ONNX | `1.22.0` |
| ONNX Runtime | `1.27.0` |
| onnxruntime-qnn | `2.3.0` |
| tokenizers | `0.23.1` |
| SoC / HTP | `QCS6490 / v68` |
| QNN EP SHA-256 | `ebcec5c0b52cc4bb96542accd17f4a410174bfc7a44599586f17852aa9ae78ef` |
| QNN HTP SHA-256 | `9e1a73ed4f3e7cf3ef3199a23dfb3885b12a64edea30e607b0687b25b28e94f7` |
| HTP Stub SHA-256 | `68e4c5f932ea006efa311c6763a4b25081f11663ae70390e6a58e41142f82c9f` |
| HTP Skel SHA-256 | `1c4ef2cec89209ffb32d670cde10c3ff66733e44d34dd23c0b76f319afc5a6dd` |
| QAIRT version | 対応runtime APIで取得できないため`null` |
| Qualcomm QNN runtime version | 対応runtime APIで取得できないため`null` |

onnxruntime-qnn package versionをQAIRTまたはQualcomm QNN runtime versionとして扱いません。`source-stamp.json`には取得できたfingerprintに加えてprovider option、session config、chunk、total length、source SHA、context file hash、厳格実行結果を保存します。identityとfileが一致する場合だけ再利用します。library hash、ONNX Runtime/onnxruntime-qnn version、target、provider/session option、graph shape、source SHA、stamp integrityのいずれかが変われば再生成します。

既定downloadはmoving branchではなくrevision `773ff42cc383cb61ecf32eb13d1f828634fbd0e1`を解決します。mirrorや別revisionは明示的に指定した場合だけ使います。

## 研究者向け全パイプライン

| Phase | 操作 | 必須gate |
|---|---|---|
| R0 | 公式`LiquidAI/LFM2.5-350M`と公式ONNXを取得 | tokenizer/config/model SHAを固定 |
| R1 | operator/initializer inventoryを作る | layer数、hidden 1024、vocab 65536、cache contract一致 |
| R2 | 再構築graphへ公式weightを移植 | CPU Q8 logits cosineとtop-1 |
| R3 | RMSNormをexact LpNormalization構成へ置換 | minigraph QNN-only + CPU parity |
| R4 | attentionへSlice+Concat GQA repeat、RoPE、causal tail-maskを導入 | q/k/v official-input minigraph parity |
| R5 | Conv/MLP/attention/final norm/lm_headをA16W8 QDQ化 | fallback無効のlayer canary |
| R6 | chunk16/ctx2048 graphとchunk1 slim decode graphを構築 | PC static check、QNN create/load |
| R7 | chunk-to-decode cache handoff | cache/logits/top-1 parity |
| R8 | tokenizer -> prefill -> decode -> detokenize | 6 smoke、JSON、長文、profile |
| R9 | 外部EPContext化 | warm load <=5 s、QNN-only profile |

chunk/decode graph構築の公開コードは`runner/scripts/probe_p4_patha2_full_chunk_graph.py`です。filenameには証跡追跡のため履歴上の実験ラベルを残しています。公開QDQとhost資産のモデル配布stagingは次で作ります。

```bash
python3 scripts/prepare_model_release.py \
  --official-model /path/to/model_q8.onnx \
  --tokenizer-dir /path/to/official-model-dir \
  --chunk-model /path/to/chunk16_a16w8_qdq.onnx \
  --decode-model /path/to/decode_a16w8_qdq.onnx \
  --model-license MODEL_LICENSE \
  --output-dir /path/to/release-assets
```

## 不変条件

- QNN sessionは`session.disable_cpu_ep_fallback=1`かつ`enable_fallback=False`。
- fallback設定、QNN session作成、profile後のQNN-only検証を別状態として扱う。Provider名一覧にCPUが見えても実行fallbackの証明ではなく、実行profileのprovider countを使う。
- `ADSP_LIBRARY_PATH`をmergeするときはユーザーentryを保持し、最終値をruntime fingerprintへ記録する。
- tokenizer/detokenizer、sampling、embedding、cache bookkeepingはhost処理として明記する。
- padされたcache行はmaskされ、handoff後の論理cache長は実token数と一致する。
- EPContext/QNN/QAIRT binaryは配布せず、対象デバイスで生成する。

## 証跡

公開用の数値は[証跡索引](records/EVIDENCE_INDEX.ja.md)と`records/evidence/*.json`を単一ソースとします。以前の構文only JSON install checkはhistoricalとして残し、現在のsemantic canary結果は別の日付付きentryにしています。非公開の生研究記録は個人パスや実機識別子を含むため、そのまま公開しません。
