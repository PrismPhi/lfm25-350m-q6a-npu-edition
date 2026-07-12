---
license: other
license_name: lfm-open-license-v1.0
license_link: https://huggingface.co/PrismPhi/lfm25-350m-q6a-npu-edition/blob/main/MODEL_LICENSE
base_model: LiquidAI/LFM2.5-350M
language: [ja, en]
tags:
  - onnx
  - qnn
  - qualcomm
  - npu
  - qcs6490
  - lfm2.5
  - quantized
---

**English version -> [MODEL_CARD.md](MODEL_CARD.md)**

GitHubリポジトリ: [lfm25-350m-q6a-npu-edition](https://github.com/PrismPhi/lfm25-350m-q6a-npu-edition)

# LFM2.5-350M Q6A NPU Edition Model Card

> **非公式モデル派生物:** 本モデル配布はLiquid AI、Qualcomm、Radxa、Microsoft、OpenAI、Anthropicの公式、承認、提携、スポンサー付き配布ではありません。
>
> **AI支援開発の開示:** 調査、コード生成・編集、デバッグ、実験整理、文書作成にOpenAI CodexおよびAnthropic Claude Codeを使用しました。生成物は人間がレビューし、Q6A実機検証、採否判断、公開判断を行いました。

## 概要

Liquid AIのLFM2.5-350Mから派生したQCS6490/Q6A向けA16W8 QDQ ONNXです。モデル配布はraw QDQとhost-side定数だけを含み、EPContextは利用者のデバイス上で生成します。

## ファイル

- `qdq/chunk16_a16w8_qdq.onnx`
- `qdq/decode_a16w8_qdq.onnx`
- `host/embedding_int8_rowwise/*`
- `host/rope_cache.npz`
- `tokenizer/*`
- `MODEL_LICENSE`
- `asset-manifest.json`

## 変更表示

QDQ ONNXは元モデルをQNN向けgraphへ再構成し、A16W8 QDQを適用した派生物です。embeddingはrowwise symmetric int8へ量子化し、RoPE cacheは公式ONNX initializerからNPZへ機械変換しました。

## 用途

QCS6490/Q6A上のローカル実験、OpenAI互換API、OpenWebUI接続を想定します。一般用途の長文品質、他HTP世代、ctx4096は保証しません。

## インストール

[GitHubのクイックスタート](https://github.com/PrismPhi/lfm25-350m-q6a-npu-edition/blob/main/README.ja.md#クイックスタート)に、検証済みのワンコマンド導入手順があります。11個の配布資産をダウンロード・SHA-256検証し、Q6A上でQNN EPContextを生成して、QNN-onlyおよびAPIスモークテストを実行します。

## 性能

採用runnerのAPI decodeは17.00-17.60 tok/s、TTFTは0.31-1.03秒です。測定条件は[GitHubの確定構成と比較表](https://github.com/PrismPhi/lfm25-350m-q6a-npu-edition/blob/main/README.ja.md#確定構成)を参照してください。

## ライセンス

LFM Open License v1.0です。商用利用の無償範囲は年間売上$10,000,000未満で、再配布にはライセンス同梱、変更表示、帰属保持が必要です。
