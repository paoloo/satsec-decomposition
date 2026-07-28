#!/usr/bin/env python3
"""Run the released crackme prompts against an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_json(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"endpoint returned HTTP {error.code}: {detail}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="model name accepted by the endpoint")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=root / "artifacts/transfer_diagnostic/crackmes/prompts.jsonl",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key in {args.api_key_env}")
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    prompts = [
        json.loads(line)
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seeds = [0, 1, 2, 3, 4]
    runs = []
    for prompt in prompts:
        case_dir = args.out / prompt["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        for repetition, seed in enumerate(seeds):
            payload = {
                "model": args.model,
                "messages": prompt["messages"],
                "temperature": 0.7,
                "top_p": 0.95,
                "seed": seed,
                "max_tokens": 4096,
                "response_format": {"type": "json_object"},
            }
            request_path = case_dir / f"repeat-{repetition}.request.json"
            response_path = case_dir / f"repeat-{repetition}.response.json"
            request_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            started = time.monotonic()
            response = request_json(endpoint, api_key, payload, args.timeout)
            elapsed = time.monotonic() - started
            response_path.write_text(
                json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            choice = response.get("choices", [{}])[0]
            runs.append(
                {
                    "case_id": prompt["case_id"],
                    "repetition": repetition,
                    "seed": seed,
                    "elapsed_seconds": elapsed,
                    "finish_reason": choice.get("finish_reason"),
                    "resolved_model": response.get("model"),
                    "response_sha256": sha256(response_path),
                }
            )
    manifest = {
        "schema_version": "1.0.0",
        "model": args.model,
        "provider": "user-supplied OpenAI-compatible endpoint",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 4096,
        "seeds": seeds,
        "response_format": "json_object",
        "prompts": str(args.prompts),
        "prompts_sha256": sha256(args.prompts),
        "repetitions": len(seeds),
        "runs": runs,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
