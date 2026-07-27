#!/usr/bin/env python3
"""Validate an optional future author/non-author comparison without imputing judgments.

This packet is a community-validation extension, not a gate for the paper's authored
benchmark operationalizations. If the optional study is undertaken, every reference needs
one author-team and one genuinely independent first-pass review. The 54 generated outputs
remain a separate optional semantic-quality study.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import median


DIMENSIONS = (
    "technique_mapping_correct_1_to_5",
    "action_coherent_1_to_5",
    "order_plausible_1_to_5",
    "checks_deterministic_1_to_5",
)
BINARY = (
    "critical_error_yes_no",
    "acceptable_yes_no",
    "correction_required_yes_no",
)
ROLES = {"internal-author", "independent-non-author"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_items(path: Path) -> dict[str, dict]:
    items = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            items[item["item_id"]] = item
    return items


def _valid_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _valid_utc(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.tzinfo is not None
    except ValueError:
        return False


def _is_placeholder(row: dict) -> bool:
    """Templates may pre-populate item and role; blank judgments are not evidence."""
    ignored = {"item_id", "reviewer_role"}
    return not any(str(value).strip() for key, value in row.items() if key not in ignored)


def validate(items: dict[str, dict], rows: list[dict]) -> tuple[list[str], dict[str, list[dict]]]:
    errors: list[str] = []
    by_item: dict[str, list[dict]] = defaultdict(list)
    expected = {item_id for item_id, item in items.items() if item["phase"] == "reference"}

    for line_no, row in enumerate(rows, 2):
        item_id = row.get("item_id", "").strip()
        role = row.get("reviewer_role", "").strip()
        if _is_placeholder(row):
            continue
        reviewer = row.get("reviewer_id", "").strip()
        if not item_id or not reviewer:
            errors.append(f"line {line_no}: item_id and reviewer_id are required")
            continue
        if item_id not in expected:
            errors.append(f"line {line_no}: {item_id!r} is not a reference item")
            continue
        if role not in ROLES:
            errors.append(f"line {line_no}: reviewer_role must be one of {sorted(ROLES)}")
        independence = row.get("independent_of_authors_yes_no", "").strip().lower()
        expected_independence = "yes" if role == "independent-non-author" else "no"
        if independence != expected_independence:
            errors.append(
                f"line {line_no}: {role or 'unknown role'} requires "
                f"independent_of_authors_yes_no={expected_independence}"
            )
        required_text = (
            "expertise_summary", "conflict_of_interest", "material_reviewed",
            "review_record_id",
        )
        for field in required_text:
            if not row.get(field, "").strip():
                errors.append(f"line {line_no}: {field} is required")
        years = row.get("years_relevant_experience", "").strip()
        if not years.isdigit() or int(years) < 1:
            errors.append(f"line {line_no}: years_relevant_experience must be a positive integer")
        reviewed = row.get("review_date_yyyy_mm_dd", "").strip()
        if not _valid_date(reviewed):
            errors.append(f"line {line_no}: review_date_yyyy_mm_dd must be an ISO date")
        frozen = row.get("first_pass_frozen_at_utc", "").strip()
        if not _valid_utc(frozen):
            errors.append(f"line {line_no}: first_pass_frozen_at_utc must be timezone-aware ISO-8601")
        digest = row.get("signed_review_record_sha256", "").strip().lower()
        if not SHA256_RE.fullmatch(digest):
            errors.append(f"line {line_no}: signed_review_record_sha256 must be 64 lowercase hex digits")
        if row.get("signed_attestation_yes_no", "").strip().lower() != "yes":
            errors.append(f"line {line_no}: signed_attestation_yes_no must be yes")
        for field in DIMENSIONS:
            if row.get(field, "").strip() not in {"1", "2", "3", "4", "5"}:
                errors.append(f"line {line_no}: {field} must be an integer 1--5")
        for field in BINARY:
            if row.get(field, "").strip().lower() not in {"yes", "no"}:
                errors.append(f"line {line_no}: {field} must be yes or no")
        critical = row.get("critical_error_yes_no", "").strip().lower() == "yes"
        correction = row.get("correction_required_yes_no", "").strip().lower() == "yes"
        if critical and not row.get("notes", "").strip():
            errors.append(f"line {line_no}: notes are required for a critical error")
        if correction and not row.get("correction_summary", "").strip():
            errors.append(f"line {line_no}: correction_summary is required when correction is yes")
        by_item[item_id].append(row)

    for item_id in sorted(expected):
        item_rows = by_item[item_id]
        roles = [row.get("reviewer_role", "").strip() for row in item_rows]
        if set(roles) != ROLES or len(roles) != 2:
            errors.append(
                f"{item_id}: requires exactly one completed review for each role; "
                f"found {sorted(role for role in roles if role)}"
            )
        reviewers = [row.get("reviewer_id", "").strip() for row in item_rows]
        if len(reviewers) != len(set(reviewers)):
            errors.append(f"{item_id}: the two roles must use distinct reviewer IDs")
    return errors, by_item


def agreement_by_item(by_item: dict[str, list[dict]], field: str, numeric: bool) -> dict:
    pairs = exact = within_one = 0
    for rows in by_item.values():
        if len(rows) != 2:
            continue
        a = rows[0][field].strip().lower()
        b = rows[1][field].strip().lower()
        pairs += 1
        exact += int(a == b)
        if numeric:
            within_one += int(abs(int(a) - int(b)) <= 1)
    out = {"pair_count": pairs, "exact_agreement": exact / pairs if pairs else 0.0}
    if numeric:
        out["within_one_agreement"] = within_one / pairs if pairs else 0.0
    return out


def read_adjudications(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["item_id"].strip(): row
            for row in csv.DictReader(handle)
            if row.get("item_id", "").strip()
        }


def summarize(items: dict[str, dict], by_item: dict[str, list[dict]], adjudications: dict) -> dict:
    item_reports = {}
    disagreements = []
    for item_id in sorted(by_item):
        rows = by_item[item_id]
        medians = {field: median(int(row[field]) for row in rows) for field in DIMENSIONS}
        numeric_disagreement = [
            field for field in DIMENSIONS if abs(int(rows[0][field]) - int(rows[1][field])) >= 2
        ]
        binary_disagreement = [
            field for field in BINARY if rows[0][field].lower() != rows[1][field].lower()
        ]
        if numeric_disagreement or binary_disagreement:
            disagreements.append({
                "item_id": item_id,
                "numeric_dimensions_differing_by_two_or_more": numeric_disagreement,
                "binary_disagreements": binary_disagreement,
                "adjudication": adjudications.get(item_id),
            })
        item_reports[item_id] = {
            "case_id": items[item_id]["case_id"],
            "reviewer_roles": sorted(row["reviewer_role"] for row in rows),
            "dimension_medians": medians,
            "critical_error_count": sum(row["critical_error_yes_no"].lower() == "yes" for row in rows),
            "acceptable_rate": sum(row["acceptable_yes_no"].lower() == "yes" for row in rows) / 2,
            "correction_required_count": sum(
                row["correction_required_yes_no"].lower() == "yes" for row in rows
            ),
        }
    gate = all(
        value["critical_error_count"] == 0
        and value["correction_required_count"] == 0
        and all(score >= 4 for score in value["dimension_medians"].values())
        for value in item_reports.values()
    )
    return {
        "status": "complete",
        "optional_reference_threshold_passed": gate,
        "reference_acceptance_threshold": (
            "both required roles present; every reference has median >=4 on every dimension, "
            "no critical-error rating, and no unresolved correction"
        ),
        "agreement": {
            field: agreement_by_item(by_item, field, field in DIMENSIONS)
            for field in DIMENSIONS + BINARY
        },
        "disagreements": disagreements,
        "items": item_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="artifacts/expert_audit/items.jsonl")
    parser.add_argument("--responses", default="artifacts/expert_audit/reference_responses.csv")
    parser.add_argument("--adjudications", default="artifacts/expert_audit/adjudications.csv")
    parser.add_argument("--out", default="artifacts/results/optional_expert_audit_analysis.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    items = read_items(root / args.items)
    with (root / args.responses).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors, by_item = validate(items, rows)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        report = {
            "status": "incomplete", "optional_reference_threshold_passed": False,
            "errors": errors,
            "warning": "The optional community-review study is incomplete; no result is claimed.",
        }
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"audit incomplete: {len(errors)} validation errors; wrote {out.relative_to(root)}")
        return 2
    adjudications = read_adjudications(root / args.adjudications)
    report = summarize(items, by_item, adjudications)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    passed = report["optional_reference_threshold_passed"]
    print(f"optional audit complete; threshold passed={passed}; wrote {out.relative_to(root)}")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
