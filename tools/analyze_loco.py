#!/usr/bin/env python3
"""Paired descriptive analysis for greedy leave-one-case-out predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from satsec.training.decomp_score import load_references, load_valid_sparta_ids, score_plan
from tools.analyze_format_neutral import score as score_format_neutral


METRICS = ("completeness", "precision", "ordering", "grounding", "check")


def role(config: str) -> str:
    if config.startswith("+adapter LOCO"):
        return "adapter"
    if config.startswith("base+schema LOCO"):
        return "schema"
    if config.startswith("base+2shot LOCO"):
        return "two-shot"
    raise ValueError(f"unknown LOCO config {config!r}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_design(rows: list[dict], reference_cases: set[str], manifest: dict) -> None:
    """Fail closed if the collected files do not represent the declared LOCO design."""
    folds = manifest.get("folds", [])
    held_out = [case for fold in folds for case in fold.get("holdout", [])]
    if manifest.get("n_folds") != 24 or len(folds) != 24:
        raise ValueError("LOCO analysis requires exactly 24 manifest folds")
    if len(held_out) != len(set(held_out)) or set(held_out) != reference_cases:
        raise ValueError("manifest must hold out every reference case exactly once")
    for row in rows:
        if row.get("type") != "decompose" or row.get("case") not in reference_cases:
            continue
        if row.get("generation", {}).get("temperature") != 0:
            raise ValueError(f"non-greedy prediction for {row.get('case')}")
        if role(row["config"]) == "two-shot":
            exemplars = row.get("fewshot_cases")
            if row.get("exemplar_seed") != 0 or not isinstance(exemplars, list) or len(exemplars) != 2:
                raise ValueError(f"invalid controlled exemplars for {row.get('case')}")
            if row["case"] in exemplars:
                raise ValueError(f"held-out case leaked into exemplars for {row['case']}")


def validate_training_manifests(
    adapters_dir: Path, folds_manifest_path: Path, manifest: dict
) -> str:
    """Bind every trained adapter manifest to its exact fold file and seed."""
    manifest_hashes: list[str] = []
    for fold in manifest["folds"]:
        fold_name = Path(fold["dir"]).name
        run_path = adapters_dir / fold_name / "run_manifest.json"
        if not run_path.is_file():
            raise ValueError(f"missing training manifest for {fold_name}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        fold_data = folds_manifest_path.parent / fold["dir"] / "tuning_set.jsonl"
        if run.get("dataset_sha256") != sha256(fold_data):
            raise ValueError(f"training dataset hash mismatch for {fold_name}")
        if run.get("train_examples") != fold.get("train"):
            raise ValueError(f"training example count mismatch for {fold_name}")
        config = run.get("config", {})
        if config.get("seed") != 42 or config.get("data_seed") not in (None, 42):
            raise ValueError(f"unexpected training seed for {fold_name}")
        manifest_hashes.append(sha256(run_path))
    return hashlib.sha256("\n".join(sorted(manifest_hashes)).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--folds-manifest", required=True)
    parser.add_argument("--adapters-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_path = root / args.data
    pred_path = root / args.pred
    manifest_path = root / args.folds_manifest
    refs = load_references(str(data_path))
    valid_ids = load_valid_sparta_ids()
    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_design(rows, set(refs), manifest)
    training_manifest_digest = validate_training_manifests(
        root / args.adapters_dir, manifest_path, manifest
    )
    scores: dict[str, dict[str, dict[str, float]]] = {
        name: {} for name in ("adapter", "schema", "two-shot")
    }
    neutral_scores: dict[str, dict[str, dict[str, float]]] = {
        name: {} for name in ("adapter", "schema", "two-shot")
    }
    for row in rows:
        if row.get("type") != "decompose" or row.get("case") not in refs:
            continue
        name = role(row["config"])
        case = row["case"]
        if case in scores[name]:
            raise ValueError(f"duplicate {name}/{case}; greedy LOCO expects one prediction")
        scores[name][case] = score_plan(row["output"], refs[case], valid_ids)
        neutral_scores[name][case] = score_format_neutral(row["output"], refs[case], valid_ids)
    for name, cases in scores.items():
        if set(cases) != set(refs):
            raise ValueError(f"{name} covers {len(cases)} of {len(refs)} cases")

    overall = {
        name: {metric: mean(value[metric] for value in cases.values()) for metric in METRICS}
        for name, cases in scores.items()
    }
    neutral_overall = {
        name: {
            metric: mean(value[metric] for value in cases.values())
            for metric in ("recall", "precision", "ordering", "candidate_validity")
        }
        for name, cases in neutral_scores.items()
    }

    def paired_report(values_by_role, metrics):
        result = {}
        for baseline in ("schema", "two-shot"):
            result[baseline] = {}
            for metric in metrics:
                deltas = {
                    case: values_by_role["adapter"][case][metric]
                    - values_by_role[baseline][case][metric]
                    for case in sorted(refs)
                }
                result[baseline][metric] = {
                    "adapter_minus_baseline_mean": mean(deltas.values()),
                    "wins": sum(value > 1e-12 for value in deltas.values()),
                    "ties": sum(abs(value) <= 1e-12 for value in deltas.values()),
                    "losses": sum(value < -1e-12 for value in deltas.values()),
                    "per_case_deltas": deltas,
                }
        return result

    paired = paired_report(scores, METRICS)
    neutral_paired = paired_report(
        neutral_scores, ("recall", "precision", "ordering", "candidate_validity")
    )
    report = {
        "design": (
            "Greedy 24-fold leave-one-case-out. Each case is generated by an adapter trained "
            "on the other 23 cases; prompted baselines draw exemplars only from that fold's training cases."
        ),
        "warning": "The 24 authored cases are the complete benchmark corpus, not a population sample.",
        "case_count": len(refs),
        "provenance": {
            "predictions": str(Path(args.pred)),
            "predictions_sha256": sha256(pred_path),
            "references": str(Path(args.data)),
            "references_sha256": sha256(data_path),
            "folds_manifest": str(Path(args.folds_manifest)),
            "folds_manifest_sha256": sha256(manifest_path),
            "training_manifests": str(Path(args.adapters_dir)),
            "training_manifests_digest_sha256": training_manifest_digest,
            "model_revisions": sorted({row.get("model_revision") for row in rows}),
        },
        "overall": overall,
        "paired": paired,
        "format_neutral": {
            "definition": "First occurrence of every SPARTA identifier anywhere in output; formatting ignored.",
            "overall": neutral_overall,
            "paired": neutral_paired,
        },
    }
    out = root / args.out
    md = root / args.markdown_out
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Greedy leave-one-case-out analysis", "", report["warning"], "",
        "| Configuration | Recall | Precision | Ordering | Candidate validity | Check presence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in overall.items():
        lines.append("| " + name + " | " + " | ".join(f"{values[m]:.3f}" for m in METRICS) + " |")
    lines += ["", "## Format-neutral identifier extraction", "",
              "| Configuration | Recall | Precision | Ordering | Candidate validity |",
              "|---|---:|---:|---:|---:|"]
    for name, values in neutral_overall.items():
        lines.append(
            f"| {name} | {values['recall']:.3f} | {values['precision']:.3f} | "
            f"{values['ordering']:.3f} | {values['candidate_validity']:.3f} |"
        )
    lines += ["", "| Baseline | Metric | Mean delta | Win/tie/loss cases |",
              "|---|---|---:|---:|"]
    for baseline, metrics in paired.items():
        for metric, values in metrics.items():
            lines.append(
                f"| {baseline} | {metric} | {values['adapter_minus_baseline_mean']:.3f} | "
                f"{values['wins']}/{values['ties']}/{values['losses']} |"
            )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(root)}")
    print(f"wrote {md.relative_to(root)}")


if __name__ == "__main__":
    main()
