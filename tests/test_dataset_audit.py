from pathlib import Path

from tools.audit_dataset import audit, read_rows
from tools.build_case_provenance import markdown_report


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tuning_set.v2.jsonl"


def test_released_dataset_passes_leakage_audit():
    report = audit(read_rows(DATA), DATA)
    assert report["status"] == "pass", report["errors"]
    assert report["cases"] == 24
    assert report["decompose"] == 24


def test_reference_evidence_coverage_is_exhaustive():
    import json

    report = json.loads((ROOT / "artifacts/results/case_provenance.json").read_text())
    assert report["case_count"] == 24
    assert report["coverage"] == {
        "cases_with_recorded_source": 24,
        "references_with_all_techniques_grounded": 24,
        "references_with_complete_step_fields": 24,
        "unique_reference_techniques": 34,
        "unique_reference_techniques_live_verified": 34,
    }
    assert "not independent semantic certification" in markdown_report(report)
