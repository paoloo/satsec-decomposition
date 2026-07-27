"""Generate decomposition plans for the fixed cases, base vs. adapter.

Companion to attacker_generate.py, but for the planning front end: it loads a base
instruct model, optionally applies the decomposition LoRA adapter, and generates a
plan for each fixed-split objective. Output is a predictions JSONL scored OFFLINE by
satsec.training.decomp_score (no scoring happens here). Run it in the reference CUDA
container with a suitable GPU.

This drives the paper's evaluation protocol:
  Q1 (marginal value): run once with --adapter (config "+adapter") and once without
      (legacy config "base+retrieval", now called candidate-only); the candidate set
      G(o) is identical (it is in the prompt),
      so the only difference is the tuned weights.
  Q2 (scale): repeat at --base Qwen/Qwen2.5-1.5B-Instruct and 7B.
  Q3 (mode transfer): add --type next_step to score single-step continuations.

Five seeds with --temperature > 0 gives the mean+/-std the protocol reports; greedy
(--temperature 0) is deterministic and ignores --seed.

  python3 decomp_generate.py --base Qwen/Qwen2.5-1.5B-Instruct \
      --adapter models/satsec-decomp-1.5b --config "+adapter (1.5B)" \
      --data tuning_set.jsonl --seed 0 --out preds_1p5b_adapter_s0.jsonl
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_split(path: str, split: str, typ: str) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        meta = ex.get("meta", {})
        if meta.get("split") == split and meta.get("type") == typ:
            rows.append(ex)
    return rows


def generate(model, tok, msgs, max_new_tokens, max_input_tokens, temperature, seed) -> str:
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt", truncation=True,
              max_length=max_input_tokens).to(model.device)
    torch.manual_seed(seed)
    gen = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if temperature and temperature > 0:
        gen.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inp, **gen)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--model-revision", required=True,
                    help="immutable Hugging Face commit for the base model")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for candidate-only")
    ap.add_argument("--config", required=True, help="row label, e.g. '+adapter (1.5B)'")
    ap.add_argument("--data", default="data/tuning_set.v2.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--type", choices=["decompose", "next_step"], default="decompose")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter or args.base,
                                        revision=None if args.adapter else args.model_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.model_revision, dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = read_split(args.data, args.split, args.type)
    with open(args.out, "w", encoding="utf-8") as fh:
        for ex in rows:
            msgs = [m for m in ex["messages"] if m["role"] != "assistant"]
            text = generate(model, tok, msgs, args.max_new_tokens, args.max_input_tokens,
                            args.temperature, args.seed)
            fh.write(json.dumps({
                "config": args.config,
                "base": args.base,
                "model_revision": args.model_revision,
                "adapter": args.adapter or "",
                "seed": args.seed,
                "case": ex["meta"].get("case", ex["meta"].get("root", "")),
                "type": args.type,
                "step": ex["meta"].get("step"),
                "prompt_messages": msgs,
                "generation": {
                    "temperature": args.temperature,
                    "top_p": 0.95 if args.temperature > 0 else None,
                    "max_new_tokens": args.max_new_tokens,
                    "max_input_tokens": args.max_input_tokens,
                },
                "output": text,
            }, ensure_ascii=False) + "\n")
    print(f"[decomp_generate] config={args.config!r} type={args.type} seed={args.seed} "
          f"-> {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
