"""Tests for the decomposition scorer (dependency-free)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from satsec.training import decomp_score as ds  # noqa: E402

_PLAN = """Plan:
1. Find an unencrypted downlink
   - Technique: REC-0005.01 Eavesdropping [ST0001]
   - Action: Survey downlinks for unencrypted broadcast traffic.
   - Check: live subscriber IPs are observed on an unencrypted downlink.
2. Send victim traffic to a subscriber IP
   - Technique: EXF-0003.02 Downlink Exfiltration [ST0008]
   - Action: Direct hosts to a subscriber IP so replies broadcast.
   - Check: data sent to the subscriber IP is observable on the downlink.
"""

_REF = ds.Reference(
    case="unit",
    grounding_ids={"REC-0005.01", "EXF-0003.02", "EXF-0003"},
    steps=ds.parse_plan(_PLAN),
)
_VALID = {"REC-0005.01", "EXF-0003.02", "EXF-0003", "REC-0005"}


def test_parse_plan_extracts_fields():
    steps = ds.parse_plan(_PLAN)
    assert len(steps) == 2
    assert steps[0].technique_id == "REC-0005.01"
    assert steps[1].technique_id == "EXF-0003.02"
    assert steps[0].check.startswith("live subscriber IPs")


def test_gold_self_scores_perfect():
    sc = ds.score_plan(_PLAN, _REF, _VALID)
    assert sc == {"completeness": 1.0, "precision": 1.0, "ordering": 1.0,
                  "grounding": 1.0, "check": 1.0}


def test_precision_penalizes_distractors():
    # Emit the two reference steps plus a third off-reference (distractor) technique.
    extra = _PLAN + (
        "3. Extra unrelated step\n"
        "   - Technique: DE-0009 Some Distractor [ST0002]\n"
        "   - Action: Do something not in the reference plan.\n"
        "   - Check: irrelevant.\n"
    )
    sc = ds.score_plan(extra, _REF, _VALID | {"DE-0009"})
    assert sc["completeness"] == 1.0            # both references still recovered
    assert abs(sc["precision"] - 2 / 3) < 1e-9  # 2 of 3 emitted ids are references


def test_reversed_plan_flags_ordering():
    pred = ds.parse_plan(_PLAN)
    pred.reverse()
    assert ds.ordering_fidelity(pred, _REF.steps) == 0.0
    assert ds.completeness(pred, _REF.steps) == 1.0  # both steps still recovered


def test_missing_step_lowers_completeness():
    one = "\n".join(_PLAN.splitlines()[:5])  # header + only step 1
    assert ds.score_plan(one, _REF, _VALID)["completeness"] == 0.5


def test_offgrounding_technique_fails_validity():
    bad = _PLAN.replace("REC-0005.01", "ZZ-9999")
    sc = ds.score_plan(bad, _REF, _VALID)
    assert sc["grounding"] == 0.5  # one of two techniques now invalid/off-grounding
    assert sc["completeness"] == 0.5  # the relabeled step no longer matches the ref


def test_sibling_subtechnique_is_not_in_exact_candidate_set():
    sibling = _PLAN.replace("REC-0005.01", "REC-0005.02")
    sc = ds.score_plan(sibling, _REF, _VALID | {"REC-0005.02"})
    assert sc["grounding"] == 0.5


def test_missing_check_lowers_field_presence():
    no_check = "\n".join(l for l in _PLAN.splitlines() if "- Check:" not in l)
    assert ds.check_field_presence(ds.parse_plan(no_check)) == 0.0
