"""Positional-copy (oracle-candidate-order) baseline for the decomposition eval.

This is the non-learned reference the hostile review demanded (F1/F2): for each
held-out case it emits *every* technique in the retrieved grounding G(o), in the
order the ids appear in the grounding block, each with a boilerplate check. It uses
no model and no learning.

Purpose: expose how much of completeness / grounding / check is trivially reachable
when G(o) contains exactly the reference technique set (no distractors). On the
current (pre-distractor) corpus it scores completeness ~1.0, grounding 1.0, check 1.0
and only ordering (~0.67-0.71) discriminates. After the distractor-augmentation fix,
its precision should collapse (it emits the distractors), which is the point.

Usage (conda env mcp, from decomp-adapt/):
    TB=../LLM-satsec/satsec-testbed
    PYTHONPATH=$TB/src python3 $TB/benchmark/decomp_poscopy_baseline.py \
        --data data/tuning_set.jsonl --out /tmp/poscopy.jsonl
    PYTHONPATH=$TB/src python3 -m satsec.training.decomp_score \
        --pred /tmp/poscopy.jsonl --data data/tuning_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

_GROUND = re.compile(r"SPARTA\s+([A-Z]{2,4}-\d{4}(?:\.\d{2})?)")


def build(src: str, out: str, config: str, seeds: int = 5) -> int:
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    dec = [r for r in rows
           if r["meta"].get("type") == "decompose" and r["meta"].get("split") == "test"]
    preds = []
    for r in dec:
        case = r["meta"].get("case")
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        seen: list[str] = []
        for g in _GROUND.findall(user):          # candidate order of appearance
            if g not in seen:
                seen.append(g)
        lines = ["Plan:"]
        for i, gid in enumerate(seen, 1):
            lines.append(f"{i}. Apply technique {gid}")
            lines.append(f"   - Technique: {gid}")
            lines.append(f"   - Action: Perform the step associated with {gid}.")
            lines.append(f"   - Check: Confirm the effect of {gid} is observed.")
        plan = "\n".join(lines)
        for seed in range(seeds):                # identical seeds -> std 0
            preds.append({"config": config, "seed": seed, "case": case,
                          "type": "decompose", "output": plan})
    with open(out, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p) + "\n")
    return len(dec)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="references JSONL (decompose, split=test)")
    ap.add_argument("--out", required=True, help="predictions JSONL to write")
    ap.add_argument("--config", default="poscopy (oracle-candidate-order)")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args(argv)
    n = build(args.data, args.out, args.config, args.seeds)
    print(f"wrote positional-copy predictions for {n} decompose cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
