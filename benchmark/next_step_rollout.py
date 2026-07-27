"""Autoregressive next-step evaluation without reference-prefix leakage.

For every held-out decompose case, this runner asks for one step at a time. After the
first call, the next prompt contains only the model's own previously generated step
summaries; reference titles and identifiers are never inserted into the context. The
reference number of steps is used as an explicit evaluation horizon and recorded as
``horizon_source=reference_length``. Raw prompts and outputs are retained in JSONL.
"""
from __future__ import annotations

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from satsec.training.decomp_score import parse_plan


def read_cases(path: str, split: str) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return [r for r in rows if r["meta"].get("split") == split
            and r["meta"].get("type") == "decompose"]


def next_prompt(decompose_user: str, history: list[str]) -> str:
    stem = re.sub(
        r"\n\nDecompose this objective into an ordered plan of verifiable steps\.\s*$",
        "",
        decompose_user,
    )
    prior = "\n".join(history) if history else "(none yet)"
    return f"{stem}\n\nSteps so far:\n{prior}\n\nGive the single next step."


def generate(model, tok, messages: list[dict], args, seed: int) -> str:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=args.max_input_tokens).to(model.device)
    torch.manual_seed(seed)
    kwargs = {"max_new_tokens": args.max_new_tokens,
              "pad_token_id": tok.eos_token_id, "do_sample": False}
    if args.temperature > 0:
        kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    return tok.decode(output[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/tuning_set.v2.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.adapter or args.base,
        revision=None if args.adapter else args.model_revision,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.model_revision, dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    cases = read_cases(args.data, args.split)
    total = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for ex in cases:
            system = next(m["content"] for m in ex["messages"] if m["role"] == "system")
            user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            gold = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            horizon = len(parse_plan(gold))
            history: list[str] = []
            for step in range(1, horizon + 1):
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": next_prompt(user, history)},
                ]
                output = generate(model, tok, messages, args, args.seed + step - 1)
                parsed = parse_plan(output)
                if parsed and parsed[0].technique_id:
                    p = parsed[0]
                    history.append(f"{step}. {p.title or 'Generated step'} [{p.technique_id}]")
                else:
                    history.append(f"{step}. [unparseable generated step]")
                fh.write(json.dumps({
                    "config": args.config,
                    "base": args.base,
                    "model_revision": args.model_revision,
                    "adapter": args.adapter or "",
                    "seed": args.seed,
                    "case": ex["meta"]["case"],
                    "type": "next_step",
                    "step": step,
                    "rollout": "autoregressive",
                    "horizon": horizon,
                    "horizon_source": "reference_length",
                    "prompt_messages": messages,
                    "prior_generated_summaries": history[:-1],
                    "generation": {
                        "temperature": args.temperature,
                        "top_p": args.top_p if args.temperature > 0 else None,
                        "max_new_tokens": args.max_new_tokens,
                        "max_input_tokens": args.max_input_tokens,
                    },
                    "output": output,
                }, ensure_ascii=False) + "\n")
                total += 1
    print(f"[next_step_rollout] {len(cases)} cases / {total} generated steps -> {args.out}")


if __name__ == "__main__":
    main()
