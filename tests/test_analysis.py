import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.analyze_fixed_split import (
    case_resampling_interval,
    config_roles,
    leave_one_case_out_sensitivity,
    quantile,
    sign_flip_p,
)
from tools.analyze_format_neutral import identifier_steps
from tools.analyze_loco import validate_design
from tools.build_manifest_index import build as build_manifest_index
from tools.analyze_transfer_diagnostic import analyze as analyze_transfer


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


def test_config_roles_have_canonical_order() -> None:
    configs = {
        "poscopy control",
        "candidate-only baseline",
        "base+2shot fixed",
        "base+schema strict",
        "+adapter (seed 42)",
    }
    assert list(config_roles(configs)) == [
        "adapter",
        "schema",
        "two-shot",
        "candidate-only",
        "positional-copy",
    ]


def test_config_role_order_is_hash_seed_independent() -> None:
    program = """
from tools.analyze_fixed_split import config_roles
values = {'poscopy x', 'candidate-only x', 'base+2shot x', 'base+schema x', '+adapter (x)'}
print(','.join(config_roles(values)))
"""
    outputs = []
    for hash_seed in ("1", "19", "271"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(completed.stdout)
    assert len(set(outputs)) == 1


def test_transfer_capsule_recomputes_reported_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    report = analyze_transfer(root / "artifacts/transfer_diagnostic/crackmes")
    assert report["attempted_runs"] == 15
    assert report["strict_valid_runs"] == 2
    assert report["format_neutral_json_runs"] == 15
    assert report["summary"]["completeness"]["mean"] == pytest.approx(0.9333333333)
    for metric in (
        "precision",
        "ordering_fidelity",
        "grounding_validity",
        "semantic_check_presence",
    ):
        assert report["summary"][metric]["mean"] == 1.0
    assert report["summary"]["exact_contract_rate"]["mean"] == pytest.approx(0.1333333333)
    assert report["summary"]["exact_contract_rate"]["std"] == pytest.approx(0.1632993162)


def test_cve_capsule_has_closed_target_and_network_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    capsule = root / "artifacts/transfer_diagnostic/cve-2025-32433"
    probe = (capsule / "lab/oracle/bounded_probe.py").read_text(encoding="utf-8")
    compose = (capsule / "lab/compose.yaml").read_text(encoding="utf-8")
    assert 'ALLOWED_TARGETS = {"vulnerable", "patched"}' in probe
    assert 'MARKER_VALUE = "RH-LAB-PREAUTH-EXEC"' in probe
    assert "internal: true" in compose
    assert "ports:" not in compose
    assert "cap_drop:" in compose


def test_manifest_index_covers_every_reported_adapter() -> None:
    root = Path(__file__).resolve().parents[1]
    index = build_manifest_index(root)
    assert index["adapter_count"] == 57
    assert index["counts"] == {"fixed": 3, "training-seed": 6, "loco": 48}


def test_adapter_file_inventory_covers_every_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads(
        (root / "artifacts/manifests/adapter_files.json").read_text(encoding="utf-8")
    )
    assert metadata["adapter_count"] == 57
    assert metadata["total_adapter_bytes"] == 3_431_601_408
    keys = {
        (entry["size"], entry["adapter_subfolder"])
        for entry in metadata["entries"]
    }
    assert len(keys) == 57
    assert all(len(entry["adapter_sha256"]) == 64 for entry in metadata["entries"])


def test_published_index_uses_immutable_adapter_revisions() -> None:
    root = Path(__file__).resolve().parents[1]
    index = json.loads(
        (root / "artifacts/manifests/index.json").read_text(encoding="utf-8")
    )
    expected = json.loads(
        (root / "artifacts/manifests/revisions.json").read_text(encoding="utf-8")
    )
    assert index["schema_version"] == "1.1.0"
    assert all(
        entry["adapter_revision"] == expected[entry["size"]]
        for entry in index["entries"]
    )
    assert all(entry["adapter_sha256"] for entry in index["entries"])
    assert all(entry["adapter_file_url"] for entry in index["entries"])
    assert all(
        entry["configuration_sha256"] == entry["adapter_config_sha256"]
        for entry in index["entries"]
    )


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
