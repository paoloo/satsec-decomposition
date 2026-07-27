#!/usr/bin/env python3
"""Compare seed-varying and fixed-exemplar two-shot prompting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from satsec.training.decomp_score import aggregate, load_references, load_valid_sparta_ids


METRICS = ("completeness", "precision", "ordering", "grounding", "check")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["0.5b", "1.5b", "7b"])
    parser.add_argument("--out", default="artifacts/results/fixed_fewshot_analysis.json")
    parser.add_argument("--markdown-out", default="artifacts/results/fixed_fewshot_analysis.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    refs = load_references(str(root / "data/tuning_set.v2.jsonl"))
    valid_ids = load_valid_sparta_ids()
    report = {
        "warning": (
            "The original two-shot condition changed both decoding randomness and exemplar identity. "
            "The fixed condition holds exemplar selection at exemplar_seed=0 across decoding seeds."
        ),
        "sizes": {},
    }
    for size in args.sizes:
        original_path = root / f"artifacts/raw_predictions/fixed_{size}_all.jsonl"
        fixed_dir = root / f"artifacts/raw_predictions/fixed_fewshot/{size}"
        fixed_rows: list[str] = []
        for path in sorted(fixed_dir.glob("fewshot_fixed_*.jsonl")):
            fixed_rows.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if not fixed_rows:
            raise ValueError(f"no fixed-exemplar outputs under {fixed_dir}")
        combined = fixed_dir / "fewshot_fixed_all.jsonl"
        combined.write_text("\n".join(fixed_rows) + "\n", encoding="utf-8")
        original_rows = [json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines()
                         if line.strip()]
        originals = [row for row in original_rows if row["config"].startswith("base+2shot")]
        fixed_json_rows = [json.loads(line) for line in fixed_rows]
        controlled_rows = [
            row for row in original_rows if not row["config"].startswith("base+2shot")
        ] + fixed_json_rows
        controlled_path = root / f"artifacts/raw_predictions/fixed_{size}_controlled_all.jsonl"
        controlled_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in controlled_rows) + "\n",
            encoding="utf-8",
        )
        original_exemplars: dict[str, dict[str, list[str]]] = {}
        for row in originals:
            original_exemplars.setdefault(str(row["seed"]), {})[row["case"]] = row.get(
                "fewshot_cases", []
            )
        fixed_exemplars: dict[str, list[str]] = {}
        for case in sorted(refs):
            choices = {
                tuple(row.get("fewshot_cases", [])) for row in fixed_json_rows if row["case"] == case
            }
            if len(choices) != 1:
                raise ValueError(f"fixed exemplar selection varies for case {case}: {choices}")
            fixed_exemplars[case] = list(next(iter(choices)))
        original_table = aggregate(str(original_path), refs, valid_ids)
        original_key = next(key for key in original_table if key.startswith("base+2shot"))
        fixed_table = aggregate(str(combined), refs, valid_ids)
        fixed_key = next(iter(fixed_table))
        report["sizes"][size] = {
            "original_seed_varying_exemplars": {
                metric: {"mean": original_table[original_key][metric][0],
                         "sample_std": original_table[original_key][metric][1]}
                for metric in METRICS
            },
            "fixed_exemplars": {
                metric: {"mean": fixed_table[fixed_key][metric][0],
                         "sample_std": fixed_table[fixed_key][metric][1]}
                for metric in METRICS
            },
            "original_exemplar_cases_by_decoding_seed_and_test_case": original_exemplars,
            "fixed_exemplar_cases_by_test_case": fixed_exemplars,
            "controlled_prediction_file": str(controlled_path.relative_to(root)),
        }
    out = root / args.out
    md = root / args.markdown_out
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Fixed-exemplar two-shot control", "",
        "The controlled row fixes the two training exemplars while varying only decoding seed.", "",
        "| Size | Condition | Recall | Precision | Ordering | Candidate validity | Check presence |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for size, data in report["sizes"].items():
        for label, key in (("varying exemplars", "original_seed_varying_exemplars"),
                           ("fixed exemplars", "fixed_exemplars")):
            cells = [
                f"{data[key][metric]['mean']:.3f} +/- {data[key][metric]['sample_std']:.3f}"
                for metric in METRICS
            ]
            lines.append(f"| {size} | {label} | " + " | ".join(cells) + " |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(root)}")
    print(f"wrote {md.relative_to(root)}")


if __name__ == "__main__":
    main()
