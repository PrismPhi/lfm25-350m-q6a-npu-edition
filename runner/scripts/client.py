#!/usr/bin/env python3
"""Small CLI client for the localhost OpenAI-compatible QNN server."""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request


MODEL_ID = "lfm2.5-350m-qnn-ctx2048"


def parse_logit_bias(raw: str | None) -> dict | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"logit_bias must be JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("logit_bias must be an object")
    normalized = {}
    for raw_token_id, raw_bias in value.items():
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError("logit_bias keys must be integer token ids") from exc
        if isinstance(raw_bias, bool) or not isinstance(raw_bias, (int, float)) or not math.isfinite(raw_bias):
            raise argparse.ArgumentTypeError("logit_bias values must be finite numbers")
        if not -100.0 <= float(raw_bias) <= 100.0:
            raise argparse.ArgumentTypeError("logit_bias values must be in [-100, 100]")
        normalized[str(token_id)] = float(raw_bias)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI-compatible QNN client")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--profile", choices=["chat", "extraction"])
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--repetition-last-n", type=int)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--logit-bias", type=parse_logit_bias, help="JSON object such as '{\"7\": -2.0}'; omitted means no bias")
    parser.add_argument("--json-object", action="store_true")
    args = parser.parse_args()

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": False,
        "max_tokens": args.max_tokens,
    }
    for source, target in (
        ("profile", "profile"),
        ("temperature", "temperature"),
        ("top_k", "top_k"),
        ("top_p", "top_p"),
        ("repetition_penalty", "repetition_penalty"),
        ("repetition_last_n", "repetition_last_n"),
    ):
        value = getattr(args, source)
        if value is not None:
            payload[target] = value
    if args.min_new_tokens:
        payload["min_new_tokens"] = args.min_new_tokens
    if args.logit_bias is not None:
        payload["logit_bias"] = args.logit_bias
    if args.json_object:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(body, file=sys.stderr)
        return 2
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
