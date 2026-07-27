"""Stronger prompting baselines for the decomposition front end (no adapter).

decomp_generate.py compares the adapter against a bare candidate-only prompt. A reviewer
will ask: "isn't this just prompting? did you prompt the base fairly?" This runner answers
that by giving the SAME base model (no adapter) two stronger prompts and scoring them with
the same protocol, so the adapter's margin is measured against a well-prompted base, not a
crippled one:

  --strategy schema   : the exact output schema (Plan: / numbered steps / Technique,
                        Action, Check fields) is spelled out in the system prompt, so any
                        gap is not merely a formatting failure.
  --strategy fewshot  : K in-context decompose exemplars drawn from the TRAIN split, added
                        as prior chat turns before the real objective. Exemplars never come
                        from the case under test (no leakage), matching the disjoint-by-case
                        rule the adapter is held to.

Output is the same predictions JSONL that satsec.training.decomp_score consumes, with a
config label like "base+fewshot (1.5B)" so it slots straight into Table (protocol). Run
it in the reference CUDA container with a suitable GPU.

  python3 decomp_baseline_generate.py --base Qwen/Qwen2.5-1.5B-Instruct \
      --strategy fewshot --shots 2 --config "base+fewshot (1.5B)" \
      --data tuning_set.jsonl --seed 0 --out preds_base_fewshot_s0.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCHEMA_HINT = (
    "\n\nProduce the plan in EXACTLY this format and nothing else:\n"
    "Plan:\n"
    "1. <short step title>\n"
    "   - Technique: <SPARTA-ID> <name> [<tactic>]\n"
    "   - Action: <one sentence>\n"
    "   - Check: <one deterministic check>\n"
    "2. <next step>\n"
    "   ...\n"
    "Use only SPARTA technique ids that appear in the Grounding block, keep the steps in "
    "execution order, and give every step a Technique and a Check."
)


def read_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def split_rows(rows: list[dict], split: str, typ: str) -> list[dict]:
    return [r for r in rows if r["meta"].get("split") == split and r["meta"].get("type") == typ]


def fewshot_prefix(train_decompose: list[dict], exclude_case: str, k: int,
                   rng: random.Random) -> tuple[list[dict], list[str]]:
    """K (user, assistant) exemplar turns from train cases other than the test case."""
    pool = [r for r in train_decompose if r["meta"]["case"] != exclude_case]
    rng.shuffle(pool)
    msgs: list[dict] = []
    cases: list[str] = []
    for ex in pool[:k]:
        user = next(m for m in ex["messages"] if m["role"] == "user")
        asst = next(m for m in ex["messages"] if m["role"] == "assistant")
        msgs.append({"role": "user", "content": user["content"]})
        msgs.append({"role": "assistant", "content": asst["content"]})
        cases.append(ex["meta"]["case"])
    return msgs, cases


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
    ap.add_argument("--strategy", choices=["schema", "fewshot"], required=True)
    ap.add_argument("--shots", type=int, default=2, help="fewshot: exemplar count")
    ap.add_argument(
        "--exemplar-seed", type=int, default=0,
        help="fewshot exemplar-selection seed; keep fixed across decoding seeds",
    )
    ap.add_argument("--config", required=True, help="row label, e.g. 'base+fewshot (1.5B)'")
    ap.add_argument("--data", default="data/tuning_set.v2.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_rows(args.data)
    test_rows = split_rows(rows, args.split, "decompose")
    train_decompose = split_rows(rows, "train", "decompose")
    rng = random.Random(args.exemplar_seed)

    tok = AutoTokenizer.from_pretrained(args.base, revision=args.model_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.model_revision, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    with open(args.out, "w", encoding="utf-8") as fh:
        for ex in test_rows:
            case = ex["meta"].get("case", ex["meta"].get("root", ""))
            sys_msg = next(m for m in ex["messages"] if m["role"] == "system")
            user_msg = next(m for m in ex["messages"] if m["role"] == "user")

            if args.strategy == "schema":
                msgs = [{"role": "system", "content": sys_msg["content"] + SCHEMA_HINT},
                        {"role": "user", "content": user_msg["content"]}]
                fewshot_cases: list[str] = []
            else:  # fewshot
                msgs = [sys_msg]
                prefix, fewshot_cases = fewshot_prefix(
                    train_decompose, case, args.shots, rng)
                msgs += prefix
                msgs.append({"role": "user", "content": user_msg["content"]})

            text = generate(model, tok, msgs, args.max_new_tokens, args.max_input_tokens,
                            args.temperature, args.seed)
            fh.write(json.dumps({
                "config": args.config,
                "base": args.base,
                "model_revision": args.model_revision,
                "adapter": "",
                "strategy": args.strategy,
                "shots": args.shots if args.strategy == "fewshot" else 0,
                "exemplar_seed": args.exemplar_seed if args.strategy == "fewshot" else None,
                "fewshot_cases": fewshot_cases,
                "seed": args.seed,
                "case": case,
                "type": "decompose",
                "step": None,
                "output": text,
                "prompt_messages": msgs,
                "generation": {
                    "temperature": args.temperature,
                    "top_p": 0.95 if args.temperature > 0 else None,
                    "max_new_tokens": args.max_new_tokens,
                    "max_input_tokens": args.max_input_tokens,
                },
            }, ensure_ascii=False) + "\n")

    print(f"[decomp_baseline] strategy={args.strategy} config={args.config!r} "
          f"seed={args.seed} -> {len(test_rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
