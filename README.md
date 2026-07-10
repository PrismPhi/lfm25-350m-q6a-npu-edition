**日本語版 -> [README.ja.md](README.ja.md)**

# LFM2.5-350M Q6A NPU Edition

An experimental prompt-to-text runner for LFM2.5-350M on the QNN HTP of QCS6490/Q6A. The CPU handles tokenization, rowwise-int8 embedding lookup, sampling, stop processing, and cache bookkeeping. QNNExecutionProvider executes chunk prefill and decode for the model body. CPU EP fallback is disabled.

This tree is published on [GitHub](https://github.com/PrismPhi/lfm25-350m-q6a-npu-edition). Model assets and the HF model card are published on [Hugging Face](https://huggingface.co/PrismPhi/lfm25-350m-q6a-npu-edition).

> **Unofficial project:** This project is not an official, endorsed, affiliated, or sponsored project of Liquid AI, Qualcomm, Radxa, Microsoft, OpenAI, or Anthropic.
>
> **AI-assisted development disclosure:** OpenAI Codex and Anthropic Claude Code were used for research, code generation and editing, debugging, experiment organization, and documentation. A human reviewed the outputs and made the Q6A hardware-validation, adoption, and publication decisions.

## Validated Configuration

| Item | Measurement | Condition |
|---|---:|---|
| context | 2048 token | chunk16 prefill + slim decode |
| API prefill | 33.87-143.53 tok/s | 3 practical API tasks; prompt-dependent |
| API decode | 17.00-17.60 tok/s | same 3 tasks |
| TTFT | 0.31-1.03 s | excludes session creation |
| long generation | 442 completion token | Japanese explanation; normal stop |
| strict JSON | valid | `{"answer":"東京"}` |
| resident server RSS | 758 -> 813 MiB | before/after final API sample |
| power | unmeasured | no world-readable telemetry; thermal proxy only |
| fresh install | 62.2 s | Q6A; local assets; through EPContext generation and smoke |
| public URL fresh install | 126.9 s | GitHub clone, 11 HF assets, through EPContext generation and smoke |
| idempotent rerun | 5.5 s | reused 11 assets and both contexts |

The low end of API prefill occurs because the first partial chunk for a short prompt uses the decode path. TTFT for the corresponding task is 0.31 s, so the actual interactive wait is 0.31 s.

## CPU Comparison

| Phase | Backend | Throughput | method |
|---|---|---:|---|
| prefill | Hybrid QNN chunk | 160-191 tok/s | chunk16, ctx2048 |
| prefill | CPU Q4 `llama-bench` | 112.6 tok/s | prompt processing, ctx2048 |
| decode | Hybrid QNN | 16.28 tok/s | controlled API-comparison mean |
| decode | ORT CPU Q8 | 14.73 tok/s | controlled API-comparison mean |
| decode | CPU Q4 `llama-bench` | 24.9 tok/s | separate `llama-bench` measurement; reference only |

Both prefill values are measured at ctx2048, but Hybrid QNN uses the chunk graph and CPU Q4 uses `llama-bench`. For decode, Hybrid QNN and ORT CPU Q8 use the controlled API comparison; CPU Q4 is a separate `llama-bench` reference.

## Requirements

- QCS6490/Q6A with Linux aarch64
- A user-provided Qualcomm QNN/QAIRT environment
- A QNN-enabled ONNX Runtime Python environment
- Approximately 2.5 GiB of free space
- Tested: Python 3.12.3, ONNX 1.22.0, ONNX Runtime 1.27.0, tokenizers 0.23.1

QNN/QAIRT shared libraries and EPContext binaries are not included.

## Quick Start

Model assets are acquired from the public Hugging Face repository.

```bash
export LFM25_MODEL_BASE_URL="https://huggingface.co/PrismPhi/lfm25-350m-q6a-npu-edition/resolve/main"
bash runner/install.sh --python /path/to/qnn-venv/bin/python
bash runner/start_server.sh
```

From another terminal:

```bash
python3 runner/scripts/client.py --prompt "日本の首都は？" --max-tokens 64
python3 runner/scripts/client.py --prompt "日本の首都をJSONで返して" --json-object --max-tokens 64
```

`install.sh` checks dependencies, verifies the SHA-256 of 11 assets, generates device EPContexts, runs a QNN-only canary, and tests normal and JSON responses. Errors name the failed `dependencies`, `assets`, `epcontext`, or `smoke` stage.

## OpenWebUI

By default, the server listens only on loopback at `127.0.0.1:18080`. An SSH tunnel is recommended for another PC or a container.

```bash
ssh -N -L 18081:127.0.0.1:18080 q6a-user@q6a-host
```

Set the OpenWebUI OpenAI-compatible base URL to `http://host.docker.internal:18081/v1` and use any non-empty API key. LAN binding provides neither authentication nor TLS and must not be used on an untrusted network.

## License

Code is under Apache License 2.0. Derived QDQ, embedding, and RoPE assets are under LFM Open License v1.0. The license permits distribution of derivatives, subject to including the license, marking changes, and retaining attribution. Free commercial use is limited to entities with annual revenue below $10,000,000. See [LICENSES.md](LICENSES.md) and [MODEL_LICENSE](MODEL_LICENSE).

## Reading Order

- [Reproducibility](REPRODUCIBILITY.md)
- [Failure routes and diagnostics](PITFALLS.md)
- [Porting to other HTP generations](PORTING.md)
- [Known limits](KNOWN_LIMITS.md)
- [API specification](API.md)
- [Publication scope](PUBLIC_SCOPE.md)
- [Research records](records/PHASE_SUMMARY.md)
