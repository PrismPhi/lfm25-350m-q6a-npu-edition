**English version -> [REPRODUCIBILITY.md](REPRODUCIBILITY.md)**

# 再現手順

## 再現の2段階

利用者向け再現は、配布済みA16W8 QDQからQ6A上でEPContextを生成する経路です。研究者向け再現は、公式LFM2.5-350MからPath A2/N4b graphを再構築する経路です。後者はQAIRT/operator差に敏感で、各gateを飛ばして最終graphだけを評価してはいけません。

## 利用者向け経路

1. QNN対応Pythonを準備する。
2. 配布資産11件を取得し、`runner/config/model-assets.json`でSHA-256を照合する。
3. chunk16/decode raw QDQから外部EPContextをデバイス上生成する。
4. 生成直後と再ロード後にQNNExecutionProviderのみで1回実行する。
5. 通常promptとJSON modeを一時localhost serverで実行する。
6. `runner/start_server.sh`で常駐serverを開始する。

```bash
export LFM25_MODEL_BASE_URL="https://huggingface.co/PrismPhi/lfm25-350m-q6a-npu-edition/resolve/main"
bash runner/install.sh --python /path/to/qnn-venv/bin/python
```

合格条件は、11資産のSHA一致、chunk/decodeの`generate_ok=true`、`load_ok=true`、`load_qnn_only=true`、通常応答が非空、JSONがobjectとしてparse可能、終了profileのchunk/decodeが両方QNN-onlyです。

## 研究者向け全パイプライン

| Phase | 操作 | 必須gate |
|---|---|---|
| R0 | 公式`LiquidAI/LFM2.5-350M`と公式ONNXを取得 | tokenizer/config/model SHAを固定 |
| R1 | operator/initializer inventoryを作る | layer数、hidden 1024、vocab 65536、cache contract一致 |
| R2 | Path A2へ公式weightを移植 | CPU Q8 logits cosineとtop-1 |
| R3 | RMSNormをN4b exact LpNormalization構成へ置換 | minigraph QNN-only + CPU parity |
| R4 | attentionへSlice+Concat GQA repeat、RoPE、causal tail-maskを導入 | q/k/v official-input minigraph parity |
| R5 | Conv/MLP/attention/final norm/lm_headをA16W8 QDQ化 | fallback無効のlayer canary |
| R6 | chunk16/ctx2048 graphとchunk1 slim decode graphを構築 | PC static check、QNN create/load |
| R7 | chunk-to-decode cache handoff | cache/logits/top-1 parity |
| R8 | tokenizer -> prefill -> decode -> detokenize | 6 smoke、JSON、長文、profile |
| R9 | 外部EPContext化 | warm load <=5 s、QNN-only profile |

chunk/decode graph構築の公開コードは`runner/scripts/probe_p4_patha2_full_chunk_graph.py`です。採用QDQとhost資産のモデル配布stagingは次で作ります。

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
- Provider名一覧にCPUが見えても成功としない。実行profileのprovider countでQNN-onlyを確認する。
- tokenizer/detokenizer、sampling、embedding、cache bookkeepingはhost処理として明記する。
- padされたcache行はmaskされ、handoff後の論理cache長は実token数と一致する。
- EPContext/QNN/QAIRT binaryは配布せず、対象デバイスで生成する。

## 証跡

公開用の数値は[証跡索引](records/EVIDENCE_INDEX.ja.md)と`records/evidence/*.json`を単一ソースとします。元の2週間分の非公開監査ツリーは個人パスや実機識別子を含むため、そのまま公開しません。
