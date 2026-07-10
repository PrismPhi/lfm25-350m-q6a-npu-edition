**English version -> [LICENSES.md](LICENSES.md)**

# ライセンスと再配布

## コード

本リポジトリのコードと独自ドキュメントは[Apache License 2.0](LICENSE)です。

## モデル派生物

LFM2.5-350M由来のA16W8 QDQ ONNX、rowwise-int8 embedding、RoPE cache、tokenizerは[LFM Open License v1.0](MODEL_LICENSE)です。

公式条文のSection 2はDerivative Worksの作成・配布を認め、Section 4はSource/Object formでの再配布を認めています。再配布時は次が必要です。

1. LFM Open License v1.0のコピーを渡す。
2. 変更したファイルに目立つ変更表示を付ける。
3. 著作権、特許、商標、帰属表示を保持する。
4. 元配布にNOTICEがある場合は該当NOTICEを保持する。

Section 5により、商用利用の無償範囲は年間売上$10,000,000未満です。それ以上のLegal EntityのCommercial Useは同ライセンスでは許諾されません。これは法的助言ではありません。

公式参照:

- <https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE>
- <https://docs.liquid.ai/lfm/help/model-license>

## 配布しないもの

Qualcomm QNN/QAIRT binary、ONNX Runtime QNN binary、EPContext/QNN context binary、GGUFは本配布に含みません。各利用者が適切な権利の下でQNN環境を用意し、EPContextを対象デバイス上で生成します。
