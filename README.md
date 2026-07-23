**日本語版 -> [README.ja.md](README.ja.md)**

# LFM2.5-350M on Radxa Dragon Q6A (QCS6490 QNN NPU)

An experimental prompt-to-text runner for LFM2.5-350M on the QNN HTP of QCS6490/Q6A. The CPU handles tokenization, rowwise-int8 embedding lookup, sampling, stop processing, and cache bookkeeping. QNNExecutionProvider executes chunk prefill and decode for the model body. CPU EP fallback is disabled.

This tree is published on [GitHub](https://github.com/PrismPhi/radxa-dragon-q6a-qcs6490-lfm2.5-350m-qnn-npu). Model assets and the HF model card are published on [Hugging Face](https://huggingface.co/PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu).

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
| public URL fresh install | 126.9-288.5 s | 3 recorded GitHub/HF fresh-install validations; network-dependent |
| idempotent rerun | 5.5-5.8 s | reused 11 assets and both contexts |
| runtime-contract upgrade | 68.1 s | rebuilt both contexts from the previous stamp schema |
| runtime-contract rerun | 6.1 s | reused 11 assets and both fingerprint-matched contexts |

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
- Tested: Python 3.12.3, ONNX 1.22.0, ONNX Runtime 1.27.0, onnxruntime-qnn 2.3.0, tokenizers 0.23.1

QNN/QAIRT shared libraries and EPContext binaries are not included.

The canonical project name uses `LFM2.5`; runtime overrides use the `LFM2_5_*` prefix.

## Runtime Identity

| Field | Validated value |
|---|---|
| target | `QCS6490 / HTP v68` |
| QNN EP | `QNNExecutionProvider` |
| QNN EP library SHA-256 | `ebcec5c0b52cc4bb96542accd17f4a410174bfc7a44599586f17852aa9ae78ef` |
| QNN HTP library SHA-256 | `9e1a73ed4f3e7cf3ef3199a23dfb3885b12a64edea30e607b0687b25b28e94f7` |
| HTP v68 Stub SHA-256 | `68e4c5f932ea006efa311c6763a4b25081f11663ae70390e6a58e41142f82c9f` |
| HTP v68 Skel SHA-256 | `1c4ef2cec89209ffb32d670cde10c3ff66733e44d34dd23c0b76f319afc5a6dd` |
| QAIRT version | `null` because the tested package does not expose it |
| Qualcomm QNN runtime version | `null` because the tested package does not expose it |

The installer does not infer QAIRT or Qualcomm QNN runtime versions from the onnxruntime-qnn package version. It records the Python/package versions, absolute QNN library paths and hashes, target, provider options, session config, final `ADSP_LIBRARY_PATH`, chunk, and total length in `install-result.json`, each `source-stamp.json`, and server evidence. Existing `ADSP_LIBRARY_PATH` entries are preserved and deduplicated. A changed runtime identity or corrupt stamp forces EPContext regeneration instead of reuse.

The default model URL is pinned to Hugging Face revision `773ff42cc383cb61ecf32eb13d1f828634fbd0e1`; `--model-base-url`, `--model-repository`, and `--model-revision` remain explicit override mechanisms.

## Quick Start

Model assets are acquired from the public Hugging Face repository.

```bash
bash runner/install.sh --python /path/to/qnn-venv/bin/python
bash runner/start_server.sh
```

From another terminal:

```bash
python3 runner/scripts/client.py --prompt "日本の首都は？" --max-tokens 64
python3 runner/scripts/client.py --prompt "日本の首都をJSONで返して" --json-object --max-tokens 64
```

`install.sh` downloads the pinned assets from the public Hugging Face repository by default, checks dependencies, verifies the SHA-256 of 11 assets, generates device EPContexts, and requires finite QNN-only generation, reload, and execution. Its deterministic semantic canary requires normal output containing `Tokyo`, JSON output `{"answer":"Tokyo"}`, and first token ID `40550`. Use `--model-base-url` for a mirror or `--asset-dir` for an offline install. Errors name the failed `dependencies`, `assets`, `epcontext`, or `smoke` stage.

## OpenWebUI

By default, the server listens only on loopback at `127.0.0.1:18080`. An SSH tunnel is recommended for another PC or a container.

```bash
ssh -N -L 18081:127.0.0.1:18080 q6a-user@q6a-host
```

Set the OpenWebUI OpenAI-compatible base URL to `http://host.docker.internal:18081/v1` and use any non-empty API key. A non-loopback `--host` is refused unless you pass `--allow-lan`; even then it provides neither authentication nor TLS and must not be used on an untrusted network. To opt in on a trusted LAN:

```bash
LFM2_5_HOST=0.0.0.0 bash runner/start_server.sh --allow-lan
```

## License

Code is under Apache License 2.0. Derived QDQ, embedding, and RoPE assets are under LFM Open License v1.0. The license permits distribution of derivatives, subject to including the license, marking changes, and retaining attribution. Free commercial use is limited to entities with annual revenue below $10,000,000. See [LICENSES.md](LICENSES.md) and [MODEL_LICENSE](MODEL_LICENSE).

## Reading Order

- [Engineering findings](FINDINGS.md)
- [Glossary](GLOSSARY.md)
- [Reproducibility](REPRODUCIBILITY.md)
- [Known failure modes and diagnostics](PITFALLS.md)
- [Porting to other HTP generations](PORTING.md)
- [Known limits](KNOWN_LIMITS.md)
- [API specification](API.md)
- [Publication scope](PUBLIC_SCOPE.md)
- [Research records](records/PHASE_SUMMARY.md)
