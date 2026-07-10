**日本語版 -> [LICENSES.ja.md](LICENSES.ja.md)**

# Licenses and Redistribution

## Code

Code and original documentation in this repository are under [Apache License 2.0](LICENSE).

## Model Derivatives

The A16W8 QDQ ONNX, rowwise-int8 embedding, RoPE cache, and tokenizer derived from LFM2.5-350M are under [LFM Open License v1.0](MODEL_LICENSE).

Section 2 of the official text permits creating and distributing Derivative Works. Section 4 permits redistribution in Source or Object form. Redistribution requires:

1. Giving recipients a copy of LFM Open License v1.0.
2. Placing prominent change notices on modified files.
3. Retaining copyright, patent, trademark, and attribution notices.
4. Retaining an applicable NOTICE if one exists in the upstream distribution.

Under Section 5, free commercial use is limited to Legal Entities with annual revenue below $10,000,000. Commercial Use above the threshold is not licensed by this license. This is not legal advice.

Official references:

- <https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE>
- <https://docs.liquid.ai/lfm/help/model-license>

## Not Distributed

Qualcomm QNN/QAIRT binaries, ONNX Runtime QNN binaries, EPContext/QNN context binaries, and GGUF are not included. Each user prepares a QNN environment under appropriate rights and generates EPContext on the target device.
