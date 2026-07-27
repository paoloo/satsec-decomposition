#!/usr/bin/env python3
"""Summarize greedy fixed-split scores across independently trained adapters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev

from satsec.training.decomp_score import load_references, load_valid_sparta_ids, score_plan


METRICS = ("completeness", "precision", "ordering", "grounding", "check")
TRAIN_SEED = re.compile(r"train-seed=(\d+)")


def analyze(path: Path, refs, valid_ids) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_seed: dict[int, list[dict]] = {}
    for row in rows:
        if row.get("type") != "decompose" or row.get("case") not in refs:
            continue
        match = TRAIN_SEED.search(row["config"])
        if not match:
            raise ValueError(f"missing train-seed label in {row['config']!r}")
        by_seed.setdefault(int(match.group(1)), []).append(row)
    seed_means = {}
    for seed, selected in sorted(by_seed.items()):
        if {row["case"] for row in selected} != set(refs):
            raise ValueError(f"training seed {seed} does not cover every fixed case")
        scores = [score_plan(row["output"], refs[row["case"]], valid_ids) for row in selected]
        seed_means[str(seed)] = {metric: mean(score[metric] for score in scores) for metric in METRICS}
    if len(seed_means) < 3:
        raise ValueError(f"expected at least three training seeds, found {len(seed_means)}")
    aggregate = {}
    for metric in METRICS:
        values = [scores[metric] for scores in seed_means.values()]
        aggregate[metric] = {
            "mean": mean(values),
            "sample_std": stdev(values),
            "min": min(values),
            "max": max(values),
        }
    return {"prediction_file": str(path), "training_seeds": seed_means, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["0.5b", "1.5b", "7b"])
    parser.add_argument("--out", default="artifacts/results/training_seed_analysis.json")
    parser.add_argument("--markdown-out", default="artifacts/results/training_seed_analysis.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    refs = load_references(str(root / "data/tuning_set.v2.jsonl"))
    valid_ids = load_valid_sparta_ids()
    report = {
        "design": "Three independently trained adapters per size; greedy decoding fixes generation variance.",
        "case_count": len(refs),
        "sizes": {},
    }
    for size in args.sizes:
        path = root / f"artifacts/training_seed/fixed_{size}/training_seed_all.jsonl"
        report["sizes"][size] = analyze(path, refs, valid_ids)
        report["sizes"][size]["prediction_file"] = str(path.relative_to(root))
    out = root / args.out
    md = root / args.markdown_out
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Training-seed stability", "",
        "Each cell is mean +/- sample standard deviation across three independently trained",
        "adapters. Every adapter is evaluated greedily on the same six fixed cases.", "",
        "| Size | Recall | Precision | Ordering | Candidate validity | Check presence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for size, data in report["sizes"].items():
        cells = []
        for metric in METRICS:
            value = data["aggregate"][metric]
            cells.append(f"{value['mean']:.3f} +/- {value['sample_std']:.3f}")
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(root)}")
    print(f"wrote {md.relative_to(root)}")


if __name__ == "__main__":
    main()
