#!/usr/bin/env python3
"""Exploratory case-level analysis for the corrected v2 fixed split.

The paper's five generation seeds are repeated samples on the same six cases, not
independent experimental units. This script therefore first averages generations within
each case, then treats the six cases as the units for paired descriptive comparisons.
Intervals resample cases and sign-flip p-values are exact over the six paired deltas.
Neither should be interpreted as population-level inference from a representative sample.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import mean

from satsec.training.decomp_score import (
    _match,
    load_references,
    load_valid_sparta_ids,
    parse_plan,
    score_plan,
)


SIZES = ("0.5b", "1.5b", "7b")
METRICS = ("completeness", "precision", "ordering", "grounding", "check")
BASELINES = ("schema", "two-shot")
ROLE_ORDER = ("adapter", "schema", "two-shot", "candidate-only", "positional-copy")


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def case_resampling_interval(deltas: list[float]) -> list[float]:
    """Exact percentile interval over all n**n case-resamples (n=6 here)."""
    n = len(deltas)
    draws = [mean(deltas[i] for i in sample)
             for sample in itertools.product(range(n), repeat=n)]
    return [quantile(draws, 0.025), quantile(draws, 0.975)]


def sign_flip_p(deltas: list[float]) -> float:
    """Exact two-sided sign-flip test over paired case means."""
    observed = abs(mean(deltas))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        null.append(abs(mean(s * d for s, d in zip(signs, deltas))))
    return sum(v >= observed - 1e-12 for v in null) / len(null)


def leave_one_case_out_sensitivity(deltas: list[float], cases: list[str]) -> dict:
    """Describe how a paired mean changes when each authored reference case is omitted."""
    omitted = {
        case: mean(value for index, value in enumerate(deltas) if index != omitted_index)
        for omitted_index, case in enumerate(cases)
    }
    values = list(omitted.values())
    return {
        "leave_one_case_out_min": min(values),
        "leave_one_case_out_max": max(values),
        "all_positive": all(value > 0 for value in values),
        "all_negative": all(value < 0 for value in values),
        "omitted_case_means": omitted,
    }


def config_roles(configs: set[str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for config in sorted(configs):
        if config.startswith("+adapter ("):
            roles["adapter"] = config
        elif config.startswith("base+schema"):
            roles["schema"] = config
        elif config.startswith("base+2shot"):
            roles["two-shot"] = config
        elif config.startswith(("base+retrieval", "candidate-only")):
            roles["candidate-only"] = config
        elif config.startswith("poscopy"):
            roles["positional-copy"] = config
    required = {"adapter", "schema", "two-shot", "candidate-only", "positional-copy"}
    missing = required - roles.keys()
    if missing:
        raise ValueError(f"missing configurations: {sorted(missing)}")
    return {role: roles[role] for role in ROLE_ORDER}


def ordering_counts(pred_text: str, ref) -> dict[str, float | int]:
    pred = parse_plan(pred_text)
    pairs = sorted(_match(pred, ref.steps))
    pred_order = [pj for _, pj in pairs]
    comparable = 0
    correct = 0
    for i in range(len(pred_order)):
        for j in range(i + 1, len(pred_order)):
            comparable += 1
            if pred_order[i] < pred_order[j]:
                correct += 1
    return {
        "matched_steps": len(pairs),
        "comparable_pairs": comparable,
        "correct_pairs": correct,
    }


def analyze_size(root: Path, size: str, refs, valid_ids, prediction_template: str) -> dict:
    pred_path = root / prediction_template.format(size=size)
    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [row for row in rows if row.get("type") == "decompose" and row["case"] in refs]
    roles = config_roles({row["config"] for row in rows})

    # role -> case -> per-generation metric dictionaries
    by_case: dict[str, dict[str, list[dict[str, float]]]] = {
        role: {case: [] for case in sorted(refs)} for role in roles
    }
    support: dict[str, dict[str, float | int]] = {}
    formatting: dict[str, dict[str, float | int]] = {}

    for role, config in roles.items():
        selected = [row for row in rows if row["config"] == config]
        total_steps = technique_steps = complete_steps = 0
        parsed_predictions = complete_predictions = 0
        matched = comparable = correct = ge2 = 0
        for row in selected:
            ref = refs[row["case"]]
            by_case[role][row["case"]].append(score_plan(row["output"], ref, valid_ids))
            steps = parse_plan(row["output"])
            if steps:
                parsed_predictions += 1
            is_complete = bool(steps) and all(
                step.technique_id and step.action.strip() and step.check.strip() for step in steps
            )
            complete_predictions += int(is_complete)
            total_steps += len(steps)
            technique_steps += sum(bool(step.technique_id) for step in steps)
            complete_steps += sum(bool(step.technique_id and step.action.strip() and step.check.strip())
                                  for step in steps)
            counts = ordering_counts(row["output"], ref)
            matched += int(counts["matched_steps"])
            comparable += int(counts["comparable_pairs"])
            correct += int(counts["correct_pairs"])
            ge2 += int(counts["matched_steps"] >= 2)

        n = len(selected)
        support[role] = {
            "predictions": n,
            "mean_matched_steps": matched / n if n else 0.0,
            "predictions_with_at_least_two_matches": ge2,
            "comparable_pairs": comparable,
            "pair_weighted_ordering": correct / comparable if comparable else None,
        }
        formatting[role] = {
            "predictions": n,
            "parsed_prediction_rate": parsed_predictions / n if n else 0.0,
            "fully_structured_prediction_rate": complete_predictions / n if n else 0.0,
            "technique_field_step_rate": technique_steps / total_steps if total_steps else 0.0,
            "complete_step_rate": complete_steps / total_steps if total_steps else 0.0,
        }

    case_means: dict[str, dict[str, dict[str, float]]] = {}
    for role, cases in by_case.items():
        case_means[role] = {}
        for case, scores in cases.items():
            if not scores:
                raise ValueError(f"no rows for {size}/{role}/{case}")
            case_means[role][case] = {
                metric: mean(score[metric] for score in scores) for metric in METRICS
            }

    paired: dict[str, dict[str, dict]] = {}
    cases = sorted(refs)
    for baseline in BASELINES:
        paired[baseline] = {}
        for metric in METRICS:
            deltas = [case_means["adapter"][case][metric]
                      - case_means[baseline][case][metric] for case in cases]
            paired[baseline][metric] = {
                "adapter_minus_baseline_mean": mean(deltas),
                "case_resampling_95pct_interval": case_resampling_interval(deltas),
                "exact_sign_flip_p": sign_flip_p(deltas),
                "per_case_deltas": dict(zip(cases, deltas)),
                "reference_case_sensitivity": leave_one_case_out_sensitivity(deltas, cases),
            }

    return {
        "prediction_file": str(pred_path.relative_to(root)),
        "cases": sorted(refs),
        "case_means": case_means,
        "paired_comparisons": paired,
        "ordering_support": support,
        "formatting": formatting,
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Corrected v2 fixed-split exploratory analysis",
        "",
        "The six held-out cases are the paired units. Generation seeds are averaged within",
        "each case. Intervals resample these six cases; they do not establish population-level",
        "generalization. Exact sign-flip tests have very low resolution at n=6 and are reported",
        "as diagnostics, not as a pass/fail significance filter.",
        "",
    ]
    for size, data in report["sizes"].items():
        lines += [f"## {size}", "", "### Adapter minus prompted baseline", "",
                  "| Baseline | Metric | Mean delta | 95% case-resampling interval | Exact p |",
                  "|---|---:|---:|---:|---:|"]
        for baseline in BASELINES:
            for metric in METRICS:
                item = data["paired_comparisons"][baseline][metric]
                lo, hi = item["case_resampling_95pct_interval"]
                lines.append(
                    f"| {baseline} | {metric} | {item['adapter_minus_baseline_mean']:.3f} "
                    f"| [{lo:.3f}, {hi:.3f}] | {item['exact_sign_flip_p']:.3f} |"
                )
        lines += ["", "### Ordering support", "",
                  "| Configuration | Predictions | Mean matched steps | >=2 matches | Comparable pairs | Pair-weighted ordering |",
                  "|---|---:|---:|---:|---:|---:|"]
        for role, item in data["ordering_support"].items():
            pair_order = item["pair_weighted_ordering"]
            pair_text = "N/A" if pair_order is None else f"{pair_order:.3f}"
            lines.append(
                f"| {role} | {item['predictions']} | {item['mean_matched_steps']:.3f} "
                f"| {item['predictions_with_at_least_two_matches']} | {item['comparable_pairs']} "
                f"| {pair_text} |"
            )
        lines += ["", "### Authored-reference case sensitivity", "",
                  "Each range is the adapter-minus-baseline mean after omitting each of the six",
                  "authored fixed-case references in turn.", "",
                  "| Baseline | Metric | Leave-one-case-out range | Stable sign |",
                  "|---|---:|---:|---:|"]
        for baseline in BASELINES:
            for metric in METRICS:
                sensitivity = data["paired_comparisons"][baseline][metric][
                    "reference_case_sensitivity"
                ]
                stable = sensitivity["all_positive"] or sensitivity["all_negative"]
                lines.append(
                    f"| {baseline} | {metric} | "
                    f"[{sensitivity['leave_one_case_out_min']:.3f}, "
                    f"{sensitivity['leave_one_case_out_max']:.3f}] | "
                    f"{'yes' if stable else 'no'} |"
                )
        lines += ["", "### Formatting diagnostic", "",
                  "| Configuration | Parsed predictions | Fully structured predictions | Technique fields | Complete steps |",
                  "|---|---:|---:|---:|---:|"]
        for role, item in data["formatting"].items():
            lines.append(
                f"| {role} | {item['parsed_prediction_rate']:.3f} "
                f"| {item['fully_structured_prediction_rate']:.3f} "
                f"| {item['technique_field_step_rate']:.3f} "
                f"| {item['complete_step_rate']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tuning_set.v2.jsonl")
    parser.add_argument("--json-out", default="artifacts/results/fixed_case_analysis.json")
    parser.add_argument("--markdown-out", default="artifacts/results/fixed_case_analysis.md")
    parser.add_argument(
        "--prediction-template",
        default="artifacts/raw_predictions/fixed_{size}_controlled_all.jsonl",
        help="path relative to code root; {size} is replaced by 0.5b, 1.5b, and 7b",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    refs = load_references(str(root / args.data))
    valid_ids = load_valid_sparta_ids()
    report = {
        "warning": (
            "Exploratory six-case paired analysis. Generation seeds are repeated outputs, "
            "not independent training runs or test cases."
        ),
        "case_count": len(refs),
        "training_seed_count": 1,
        "sizes": {
            size: analyze_size(root, size, refs, valid_ids, args.prediction_template)
            for size in SIZES
        },
    }

    json_path = root / args.json_out
    md_path = root / args.markdown_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report) + "\n", encoding="utf-8")
    print(f"wrote {json_path.relative_to(root)}")
    print(f"wrote {md_path.relative_to(root)}")


if __name__ == "__main__":
    main()
