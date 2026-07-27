"""Score grounded decomposition plans against reference plans.

The decomposition adapter turns an objective + retrieved grounding into an ordered
plan of verifiable steps; this module measures how well a generated plan matches the
reference decomposition sigma(o). It is pure Python (no torch/transformers) so the
scoring runs offline, separately from generation, exactly like the injected-benchmark
findings are scored apart from attacker_generate.py.

Plan text format (produced by build_tuning_set.py and expected from the model):

    Plan:
    1. <title>
       - Technique: <SPARTA-ID> <name> [<tactic>]
       - Action: <one sentence>
       - Check: <deterministic check>
    2. ...

A next_step completion is a single such step block (no "Plan:" header).

Five metrics, matching the paper's evaluation protocol:
  * completeness       -- fraction of reference steps recovered, where a reference step
                          is recovered when the plan emits a step naming the same SPARTA
                          technique id. The deterministic check is paraphrastic free
                          text, so it is NOT required to match here; check quality is
                          measured separately by check-field presence. (Check-text similarity
                          is still used only to break ties when several emitted steps
                          carry the same technique id.)
  * ordering_fidelity  -- 1 - (precedence violations / comparable pairs) on matched steps
  * grounding_validity -- fraction of emitted techniques that are real SPARTA ids AND
                          were present in the grounding G(o) supplied at inference
  * check              -- fraction of emitted steps carrying a non-empty parsed check
                          field; it does not assess semantics or executability
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CORPUS_DIR = os.path.join(_ROOT, "data", "corpus")

# SPARTA technique id, e.g. REC-0005, EXF-0003.02, DE-0002.03 (2-4 letters).
_TECH_RE = re.compile(r"\b([A-Z]{2,4}-\d{4}(?:\.\d{2})?)\b")
# Step header, tolerant of the surface a base model uses without the adapter:
# "1.", "1)", "Step 1:", "Step 1." all open a step.
_STEP_HEAD_RE = re.compile(r"^\s*(?:step\s+)?(\d+)\s*[.):]\s*(.*\S)?\s*$", re.IGNORECASE)
# A labeled field line: strip markdown/bullet noise, then read the label + value.
_FIELD_RE = re.compile(r"^\s*[-*+]?\s*\**\s*(technique|action|check)\b\**\s*:?\s*(.*\S)?\s*$",
                       re.IGNORECASE)
_GROUNDING_RE = re.compile(r"SPARTA\s+([A-Z]{2,4}-\d{4}(?:\.\d{2})?)")

# Check-text token Jaccard is used only to rank tie-broken pairings (several emitted
# steps sharing one technique id), not to gate a match. Checks are paraphrastic.


@dataclass
class Step:
    index: int
    title: str = ""
    technique_id: str = ""
    action: str = ""
    check: str = ""


@dataclass
class Reference:
    case: str
    grounding_ids: set[str] = field(default_factory=set)  # exact identifiers present in G(o)
    steps: list[Step] = field(default_factory=list)


def parse_plan(text: str) -> list[Step]:
    """Parse plan (or single next_step) text into ordered steps, tolerant of the surface
    a base model produces without the adapter. A step opens on a header line ("1.",
    "Step 1:", ...) and collects Technique/Action/Check fields until the next header.
    The technique id is read from a Technique/SPARTA line; if a step names no explicit
    technique line, the first SPARTA id appearing in the block is used."""
    steps: list[Step] = []
    cur: Step | None = None
    for raw in text.splitlines():
        head = _STEP_HEAD_RE.match(raw)
        if head:
            cur = Step(index=int(head.group(1)), title=(head.group(2) or "").strip())
            steps.append(cur)
            continue
        if cur is None:
            continue
        fld = _FIELD_RE.match(raw)
        if fld:
            key, val = fld.group(1).lower(), (fld.group(2) or "").strip()
            if key == "technique":
                if not cur.technique_id:
                    m = _TECH_RE.search(val)
                    if m:
                        cur.technique_id = m.group(1)
            elif key == "action":
                cur.action = cur.action or val
            elif key == "check":
                cur.check = cur.check or val
            continue
        # Unlabeled technique line, e.g. "**SPARTA EXF-0003**: Signal Interception".
        if not cur.technique_id and "sparta" in raw.lower():
            m = _TECH_RE.search(raw)
            if m:
                cur.technique_id = m.group(1)
    return steps


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _match(pred: list[Step], ref: list[Step]) -> list[tuple[int, int]]:
    """Greedy one-to-one match ref->pred on exact SPARTA technique id. When several
    emitted steps carry the same technique id, check-text similarity breaks the tie so
    the best-aligned pairing wins. Returns (ref_i, pred_j) pairs."""
    cands: list[tuple[float, int, int]] = []
    for ri, r in enumerate(ref):
        for pj, p in enumerate(pred):
            if not r.technique_id or p.technique_id != r.technique_id:
                continue
            cands.append((_jaccard(r.check, p.check), ri, pj))
    cands.sort(reverse=True)
    used_r: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, ri, pj in cands:
        if ri in used_r or pj in used_p:
            continue
        used_r.add(ri)
        used_p.add(pj)
        pairs.append((ri, pj))
    return pairs


def completeness(pred: list[Step], ref: list[Step]) -> float:
    if not ref:
        return 0.0
    return len(_match(pred, ref)) / len(ref)


def precision(pred: list[Step], ref: list[Step]) -> float:
    """Selectivity: fraction of distinct emitted technique ids that are in the reference
    set. Completeness is recall (did you recover the reference techniques); precision is
    the complement (did you avoid emitting the distractor techniques now present in G(o)).
    A dump-everything baseline maxes completeness but not precision."""
    emitted = {s.technique_id for s in pred if s.technique_id}
    if not emitted:
        return 0.0
    ref_ids = {s.technique_id for s in ref if s.technique_id}
    return len(emitted & ref_ids) / len(emitted)


def ordering_fidelity(pred: list[Step], ref: list[Step]) -> float:
    pairs = sorted(_match(pred, ref))  # by ref index
    if len(pairs) < 2:
        return 1.0 if pairs else 0.0
    pred_order = [pj for _, pj in pairs]
    inversions = comparable = 0
    for i in range(len(pred_order)):
        for j in range(i + 1, len(pred_order)):
            comparable += 1
            if pred_order[i] > pred_order[j]:
                inversions += 1
    return 1.0 - inversions / comparable


def grounding_validity(pred: list[Step], grounding_ids: set[str], valid_ids: set[str]) -> float:
    """Fraction of emitted identifiers that are valid and occur exactly in ``G(o)``.

    Parent and sub-technique identifiers are distinct candidates. In particular, an
    emitted sibling must not receive credit merely because its parent matches the parent
    of a supplied candidate.
    """
    emitted = [s for s in pred if s.technique_id]
    if not emitted:
        return 0.0
    ok = 0
    for s in emitted:
        tid = s.technique_id
        is_sparta = tid in valid_ids
        in_ground = tid in grounding_ids
        if is_sparta and in_ground:
            ok += 1
    return ok / len(emitted)


def check_field_presence(pred: list[Step]) -> float:
    """Fraction of parsed steps with a non-empty Check field; no semantic claim."""
    if not pred:
        return 0.0
    return sum(1 for s in pred if s.check.strip()) / len(pred)


# Compatibility alias for code that consumed the historical, overbroad name.
check_wellformed = check_field_presence


def score_plan(pred_text: str, ref: Reference, valid_ids: set[str]) -> dict[str, float]:
    pred = parse_plan(pred_text)
    return {
        "completeness": completeness(pred, ref.steps),
        "precision": precision(pred, ref.steps),
        "ordering": ordering_fidelity(pred, ref.steps),
        "grounding": grounding_validity(pred, ref.grounding_ids, valid_ids),
        "check": check_field_presence(pred),
    }


def load_valid_sparta_ids() -> set[str]:
    """SPARTA technique ids known to the corpus (records id 'sparta-<ID>')."""
    ids: set[str] = set()
    for path in glob.glob(os.path.join(CORPUS_DIR, "*.json")):
        for rec in json.load(open(path, encoding="utf-8")):
            rid = rec.get("id", "")
            if rid.startswith("sparta-") and rec.get("kind") == "technique":
                ids.add(rid[len("sparta-"):])
    return ids


def load_references(jsonl_path: str, split: str = "test") -> dict[str, Reference]:
    """Build one Reference per case from the decompose examples of the given split.
    Grounding ids are read from the user turn's ``SPARTA <id>`` grounding lines."""
    refs: dict[str, Reference] = {}
    for line in open(jsonl_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        meta = ex.get("meta", {})
        if meta.get("split") != split or meta.get("type") != "decompose":
            continue
        case = meta.get("case", meta.get("root", ""))
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        assistant = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        refs[case] = Reference(
            case=case,
            grounding_ids=set(_GROUNDING_RE.findall(user)),
            steps=parse_plan(assistant),
        )
    return refs


# --------------------------------------------------------------------------
# Offline aggregation: predictions JSONL -> per-config table rows (mean+/-std)
#
# A predictions file is one JSON object per line, produced by the GPU runner
# (benchmark/decomp_generate.py):
#   {"config": str, "seed": int, "case": str, "type": "decompose",
#    "output": "<plan text>"}
# We score each decompose prediction against its case reference, average the five
# metrics over the held-out cases within a seed, then report mean+/-std across seeds.
# --------------------------------------------------------------------------

_METRICS = ("completeness", "precision", "ordering", "grounding", "check")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def aggregate(pred_path: str, refs: dict[str, Reference],
              valid_ids: set[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {config: {metric: (mean, std)}} over seeds, averaging cases per seed."""
    # config -> seed -> metric -> list of per-case scores
    acc: dict[str, dict[int, dict[str, list[float]]]] = {}
    for line in open(pred_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("type") != "decompose":
            continue
        ref = refs.get(row["case"])
        if ref is None:
            continue
        sc = score_plan(row["output"], ref, valid_ids)
        seed_acc = acc.setdefault(row["config"], {}).setdefault(int(row.get("seed", 0)), {})
        for k in _METRICS:
            seed_acc.setdefault(k, []).append(sc[k])

    out: dict[str, dict[str, tuple[float, float]]] = {}
    for config, by_seed in acc.items():
        per_metric: dict[str, list[float]] = {k: [] for k in _METRICS}
        for _seed, metrics in by_seed.items():
            for k in _METRICS:
                per_metric[k].append(_mean(metrics[k]))  # mean over cases in this seed
        out[config] = {k: (_mean(per_metric[k]), _std(per_metric[k])) for k in _METRICS}
    return out


def aggregate_next_step(pred_path: str, refs: dict[str, Reference],
                        valid_ids: set[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """Score next_step predictions (Q3). Each row is a single emitted step tied to a
    reference step index (meta.step, 1-based); we report per-step technique accuracy,
    grounding validity, and check-field presence, meaned over steps per seed then
    reported mean+/-std across seeds. Metric keys reuse the table columns (completeness
    here means per-step technique accuracy; ordering is not defined step-wise -> 1.0)."""
    acc: dict[str, dict[int, dict[str, list[float]]]] = {}
    for line in open(pred_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("type") != "next_step":
            continue
        ref = refs.get(row["case"])
        k = row.get("step")
        if ref is None or not k or k > len(ref.steps):
            continue
        ref_step = ref.steps[k - 1]
        pred_steps = parse_plan(row["output"])
        pred = pred_steps[0] if pred_steps else Step(index=k)
        tech_ok = 1.0 if pred.technique_id and pred.technique_id == ref_step.technique_id else 0.0
        ground = grounding_validity([pred], ref.grounding_ids, valid_ids)
        check = check_field_presence([pred])
        seed_acc = acc.setdefault(row["config"], {}).setdefault(int(row.get("seed", 0)), {})
        seed_acc.setdefault("completeness", []).append(tech_ok)
        seed_acc.setdefault("precision", []).append(tech_ok)  # single emitted step
        seed_acc.setdefault("ordering", []).append(1.0)
        seed_acc.setdefault("grounding", []).append(ground)
        seed_acc.setdefault("check", []).append(check)

    out: dict[str, dict[str, tuple[float, float]]] = {}
    for config, by_seed in acc.items():
        per_metric: dict[str, list[float]] = {k: [] for k in _METRICS}
        for _seed, metrics in by_seed.items():
            for k in _METRICS:
                per_metric[k].append(_mean(metrics[k]))
        out[config] = {k: (_mean(per_metric[k]), _std(per_metric[k])) for k in _METRICS}
    return out


def _format_table(table: dict[str, dict[str, tuple[float, float]]]) -> str:
    hdr = f"{'Config':28s} " + " ".join(f"{m.capitalize():>16s}" for m in _METRICS)
    lines = [hdr, "-" * len(hdr)]
    for config, metrics in table.items():
        cells = " ".join(f"{metrics[m][0]:6.2f}+/-{metrics[m][1]:4.2f}" for m in _METRICS)
        lines.append(f"{config:28s} {cells}")
    return "\n".join(lines)


def _jsonable(table: dict[str, dict[str, tuple[float, float]]]) -> dict:
    return {
        config: {metric: {"mean": values[0], "sample_std": values[1]}
                 for metric, values in metrics.items()}
        for config, metrics in table.items()
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Score decomposition predictions -> table rows.")
    ap.add_argument("--pred", required=True, help="predictions JSONL from decomp_generate.py")
    ap.add_argument("--data", required=True, help="tuning_set.jsonl (for references)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--json-out", help="optional machine-readable score report")
    args = ap.parse_args(argv)

    refs = load_references(args.data, args.split)
    valid = load_valid_sparta_ids()
    print(f"references: {len(refs)} held-out cases | valid SPARTA ids: {len(valid)}\n")
    table = aggregate(args.pred, refs, valid)
    if table:
        print("decompose mode (Q1/Q2):")
        print(_format_table(table))
    ns = aggregate_next_step(args.pred, refs, valid)
    if ns:
        print("\nnext_step mode (Q3): completeness = per-step technique accuracy")
        print(_format_table(ns))
    if args.json_out:
        report = {
            "predictions": args.pred,
            "dataset": args.data,
            "split": args.split,
            "reference_cases": len(refs),
            "decompose": _jsonable(table),
            "next_step": _jsonable(ns),
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
