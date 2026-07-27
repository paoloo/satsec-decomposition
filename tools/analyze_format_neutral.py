#!/usr/bin/env python3
"""Score emitted SPARTA identifier sequences without requiring plan formatting.

This diagnostic removes the parser/serialization advantage: it extracts the first
occurrence of every SPARTA identifier anywhere in an output, preserves that order, and
scores candidate selection and ordering against the same references. Actions and checks
are intentionally excluded and still require expert assessment.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

from satsec.training.decomp_score import (
    Step,
    grounding_validity,
    load_references,
    load_valid_sparta_ids,
    ordering_fidelity,
    precision,
    completeness,
)
from tools.analyze_fixed_split import case_resampling_interval, config_roles, sign_flip_p


SIZES = ("0.5b", "1.5b", "7b")
TECHNIQUE = re.compile(r"\b([A-Z]{2,4}-\d{4}(?:\.\d{2})?)\b")
METRICS = ("recall", "precision", "ordering", "candidate_validity")


def identifier_steps(text: str) -> list[Step]:
    seen: set[str] = set()
    steps: list[Step] = []
    for identifier in TECHNIQUE.findall(text):
        if identifier in seen:
            continue
        seen.add(identifier)
        steps.append(Step(index=len(steps) + 1, technique_id=identifier))
    return steps


def score(text: str, ref, valid_ids: set[str]) -> dict[str, float]:
    pred = identifier_steps(text)
    return {
        "recall": completeness(pred, ref.steps),
        "precision": precision(pred, ref.steps),
        "ordering": ordering_fidelity(pred, ref.steps),
        "candidate_validity": grounding_validity(pred, ref.grounding_ids, valid_ids),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    refs = load_references(str(root / "data/tuning_set.v2.jsonl"))
    valid_ids = load_valid_sparta_ids()
    report = {
        "definition": "First occurrence of every SPARTA identifier anywhere in output; formatting ignored.",
        "case_count": len(refs),
        "sizes": {},
    }
    for size in SIZES:
        controlled = root / f"artifacts/raw_predictions/fixed_{size}_controlled_all.jsonl"
        prediction_path = controlled if controlled.exists() else (
            root / f"artifacts/raw_predictions/fixed_{size}_all.jsonl"
        )
        rows = [json.loads(line) for line in
                prediction_path.read_text().splitlines()
                if line.strip()]
        rows = [row for row in rows if row.get("type") == "decompose" and row["case"] in refs]
        roles = config_roles({row["config"] for row in rows})
        case_means: dict[str, dict[str, dict[str, float]]] = {}
        overall: dict[str, dict[str, float]] = {}
        for role, config in roles.items():
            case_means[role] = {}
            for case in sorted(refs):
                values = [score(row["output"], refs[case], valid_ids) for row in rows
                          if row["config"] == config and row["case"] == case]
                case_means[role][case] = {
                    metric: mean(value[metric] for value in values) for metric in METRICS
                }
            overall[role] = {
                metric: mean(case_means[role][case][metric] for case in refs)
                for metric in METRICS
            }
        paired = {}
        for baseline in ("schema", "two-shot"):
            paired[baseline] = {}
            for metric in METRICS:
                deltas = [case_means["adapter"][case][metric]
                          - case_means[baseline][case][metric] for case in sorted(refs)]
                paired[baseline][metric] = {
                    "mean": mean(deltas),
                    "case_resampling_95pct_interval": case_resampling_interval(deltas),
                    "exact_sign_flip_p": sign_flip_p(deltas),
                }
        report["sizes"][size] = {
            "overall": overall,
            "case_means": case_means,
            "adapter_paired_comparisons": paired,
        }

    out_json = root / "artifacts/results/format_neutral_analysis.json"
    out_md = root / "artifacts/results/format_neutral_analysis.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Format-neutral identifier analysis", "",
        "The first occurrence of each SPARTA identifier is extracted anywhere in an output.",
        "No Plan/Technique/Action/Check layout is required. Actions and checks are not judged.", "",
    ]
    for size, data in report["sizes"].items():
        lines += [f"## {size}", "", "| Configuration | Recall | Precision | Ordering | Candidate validity |",
                  "|---|---:|---:|---:|---:|"]
        for role, values in data["overall"].items():
            lines.append(f"| {role} | {values['recall']:.3f} | {values['precision']:.3f} | "
                         f"{values['ordering']:.3f} | {values['candidate_validity']:.3f} |")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json.relative_to(root)}")
    print(f"wrote {out_md.relative_to(root)}")


if __name__ == "__main__":
    main()
