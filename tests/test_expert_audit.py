import csv
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "expert_audit", ROOT / "tools" / "analyze_expert_audit.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def row(item_id: str, role: str, reviewer: str, **updates):
    record = {
        "item_id": item_id,
        "reviewer_role": role,
        "reviewer_id": reviewer,
        "independent_of_authors_yes_no": (
            "yes" if role == "independent-non-author" else "no"
        ),
        "review_date_yyyy_mm_dd": "2026-07-27",
        "expertise_summary": "security review",
        "years_relevant_experience": "5",
        "conflict_of_interest": "none",
        "material_reviewed": "frozen reference packet",
        "first_pass_frozen_at_utc": "2026-07-27T12:00:00Z",
        "review_record_id": f"signed-{reviewer}",
        "signed_attestation_yes_no": "yes",
        "signed_review_record_sha256": hashlib.sha256(reviewer.encode()).hexdigest(),
        "technique_mapping_correct_1_to_5": "4",
        "action_coherent_1_to_5": "4",
        "order_plausible_1_to_5": "4",
        "checks_deterministic_1_to_5": "4",
        "critical_error_yes_no": "no",
        "acceptable_yes_no": "yes",
        "correction_required_yes_no": "no",
        "correction_summary": "",
        "notes": "",
    }
    record.update(updates)
    return record


def test_reference_gate_requires_both_named_roles():
    items = {"REF-01": {"phase": "reference", "case_id": "case"}}
    errors, _ = AUDIT.validate(items, [row("REF-01", "internal-author", "A")])
    assert any("exactly one completed review for each role" in error for error in errors)


def test_independent_role_must_attest_independence():
    items = {"REF-01": {"phase": "reference", "case_id": "case"}}
    rows = [
        row("REF-01", "internal-author", "A"),
        row(
            "REF-01", "independent-non-author", "B",
            independent_of_authors_yes_no="no",
        ),
    ]
    errors, _ = AUDIT.validate(items, rows)
    assert any("independent_of_authors_yes_no=yes" in error for error in errors)


def test_blank_template_rows_are_not_reviews():
    items = {"REF-01": {"phase": "reference", "case_id": "case"}}
    rows = [
        {"item_id": "REF-01", "reviewer_role": "internal-author"},
        {"item_id": "REF-01", "reviewer_role": "independent-non-author"},
    ]
    errors, by_item = AUDIT.validate(items, rows)
    assert not by_item["REF-01"]
    assert any("exactly one completed review for each role" in error for error in errors)


def test_complete_pair_passes_and_unresolved_correction_blocks_gate():
    items = {"REF-01": {"phase": "reference", "case_id": "case"}}
    rows = [
        row("REF-01", "internal-author", "A"),
        row("REF-01", "independent-non-author", "B"),
    ]
    errors, by_item = AUDIT.validate(items, rows)
    assert errors == []
    assert AUDIT.summarize(items, by_item, {})["optional_reference_threshold_passed"] is True

    rows[1]["correction_required_yes_no"] = "yes"
    rows[1]["correction_summary"] = "change mapping"
    errors, by_item = AUDIT.validate(items, rows)
    assert errors == []
    assert AUDIT.summarize(items, by_item, {})["optional_reference_threshold_passed"] is False


def test_template_has_two_reference_rows_per_item():
    path = ROOT / "artifacts" / "expert_audit" / "reference_responses.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 48
    assert {row["reviewer_role"] for row in rows} == AUDIT.ROLES
