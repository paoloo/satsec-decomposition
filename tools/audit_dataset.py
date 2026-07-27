"""Fail-fast audit for the leakage-controlled dataset snapshot."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from satsec.training.decomp_score import load_references, load_valid_sparta_ids, score_plan


LEGACY_MARKERS = (
    "decomposition of ", "steps:", "decomposes into:", "teaching point:",
)
STEP_HEADER = re.compile(r"^\s*\d+\.\s+(.+?)\s*$", re.MULTILINE)
GROUNDING_ID = re.compile(r"SPARTA\s+([A-Z]{2,4}-\d{4}(?:\.\d{2})?)")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def content(ex: dict, role: str) -> str:
    return next(m["content"] for m in ex["messages"] if m["role"] == role)


def objective_segment(user: str) -> str:
    return user.split("\n\nGrounding:\n", 1)[0]


def step_titles(assistant: str) -> list[str]:
    return STEP_HEADER.findall(assistant)


def audit(rows: list[dict], data_path: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    counts = Counter()
    by_case: dict[str, list[dict]] = defaultdict(list)

    for ex in rows:
        meta = ex.get("meta", {})
        case = meta.get("case", "")
        typ = meta.get("type", "")
        counts[("type", typ)] += 1
        counts[("split", meta.get("split", ""))] += 1
        by_case[case].append(ex)
        if meta.get("dataset_version") != "2.0.0":
            errors.append(f"{case}/{typ}: dataset_version is not 2.0.0")
        if meta.get("prompt_policy") != "objective-only-v2":
            errors.append(f"{case}/{typ}: prompt_policy is not objective-only-v2")

        user = content(ex, "user")
        obj = objective_segment(user)
        if "\n" in obj:
            errors.append(f"{case}/{typ}: objective segment has more than one line")
        lowered = obj.lower()
        for marker in LEGACY_MARKERS:
            if marker in lowered:
                errors.append(f"{case}/{typ}: legacy leakage marker {marker!r}")

    decompose_by_case = {
        case: next((ex for ex in examples if ex["meta"]["type"] == "decompose"), None)
        for case, examples in by_case.items()
    }
    for case, dec in decompose_by_case.items():
        if dec is None:
            errors.append(f"{case}: no decompose reference")
            continue
        user = content(dec, "user")
        obj = objective_segment(user).lower()
        titles = step_titles(content(dec, "assistant"))
        for title in titles:
            if len(title) >= 18 and title.lower() in obj:
                errors.append(f"{case}: gold title leaked in objective: {title!r}")

        ground_ids = GROUNDING_ID.findall(user)
        if len(set(ground_ids)) != 8:
            errors.append(f"{case}: expected 8 distinct grounding ids, got {len(set(ground_ids))}")

        # Every next-step prompt may expose only the gold prefix, never a future title.
        for ex in by_case[case]:
            if ex["meta"]["type"] != "next_step":
                continue
            k = int(ex["meta"]["step"])
            ns_user = content(ex, "user").lower()
            for future in titles[k:]:
                if len(future) >= 18 and future.lower() in ns_user:
                    errors.append(
                        f"{case}/next_step/{k}: future gold title leaked: {future!r}")

    # Regression for the reviewer-reported GNSS mapping error.
    gnss = decompose_by_case.get("GNSS-spoofing")
    if gnss:
        gold = content(gnss, "assistant")
        if "EX-0002" in gold:
            errors.append("GNSS gold still maps spoofing carry-off to EX-0002")
        if "EX-0014.04" not in gold:
            errors.append("GNSS gold does not contain EX-0014.04")

    # Gold plans must score perfectly under the shipped deterministic scorer.
    valid = load_valid_sparta_ids()
    for split in ("train", "test"):
        refs = load_references(str(data_path), split)
        for case, ref in refs.items():
            dec = decompose_by_case[case]
            scores = score_plan(content(dec, "assistant"), ref, valid)
            if any(abs(value - 1.0) > 1e-12 for value in scores.values()):
                errors.append(f"{case}: gold self-score is not perfect: {scores}")

    expected_next = sum(len(step_titles(content(dec, "assistant")))
                        for dec in decompose_by_case.values() if dec)
    actual_next = counts[("type", "next_step")]
    if actual_next != expected_next:
        errors.append(f"next_step count {actual_next} != gold step count {expected_next}")

    return {
        "dataset": str(data_path),
        "rows": len(rows),
        "cases": len(by_case),
        "decompose": counts[("type", "decompose")],
        "next_step": actual_next,
        "train": counts[("split", "train")],
        "test": counts[("split", "test")],
        "errors": errors,
        "warnings": warnings,
        "status": "pass" if not errors else "fail",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()
    path = Path(args.data)
    report = audit(read_rows(path), path)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        Path(args.report).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
