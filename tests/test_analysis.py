import pytest

from tools.analyze_fixed_split import (
    case_resampling_interval,
    leave_one_case_out_sensitivity,
    quantile,
    sign_flip_p,
)
from tools.analyze_format_neutral import identifier_steps
from tools.analyze_loco import validate_design


def test_quantile_interpolates() -> None:
    assert quantile([0.0, 10.0], 0.25) == 2.5


def test_constant_case_difference_has_degenerate_interval() -> None:
    assert case_resampling_interval([0.25, 0.25]) == [0.25, 0.25]


def test_exact_sign_flip_is_two_sided() -> None:
    assert sign_flip_p([1.0, 1.0]) == 0.5
    assert sign_flip_p([0.0, 0.0]) == 1.0


def test_leave_one_case_out_sensitivity_reports_range_and_sign() -> None:
    result = leave_one_case_out_sensitivity([1.0, 2.0, 3.0], ["a", "b", "c"])
    assert result["leave_one_case_out_min"] == 1.5
    assert result["leave_one_case_out_max"] == 2.5
    assert result["all_positive"] is True
    assert result["all_negative"] is False


def test_identifier_steps_ignore_layout_and_deduplicate() -> None:
    steps = identifier_steps("REC-0005.01, then EXF-0003; repeat REC-0005.01")
    assert [step.technique_id for step in steps] == ["REC-0005.01", "EXF-0003"]


def test_loco_design_rejects_held_out_fewshot_exemplar() -> None:
    cases = {f"case-{i:02d}" for i in range(24)}
    manifest = {
        "n_folds": 24,
        "folds": [{"holdout": [case]} for case in sorted(cases)],
    }
    rows = [{
        "type": "decompose",
        "case": "case-00",
        "config": "base+2shot LOCO greedy (0.5b)",
        "generation": {"temperature": 0},
        "exemplar_seed": 0,
        "fewshot_cases": ["case-00", "case-01"],
    }]
    with pytest.raises(ValueError, match="leaked"):
        validate_design(rows, cases, manifest)
