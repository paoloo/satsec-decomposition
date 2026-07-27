"""Generate the leakage-controlled decomposition dataset from the attack-tree graph.

Given a high-level objective plus an unordered, distractor-augmented grounding block,
the target is an ordered sequence of verifiable, standards-traceable steps. Version 2
excludes the attack-tree root narrative from the prompt because the legacy snapshot's
narrative described the reference sequence and leaked the gold plan.

Two example types per attack tree:
  * decompose  -- objective + context -> the full ordered plan
  * next_step  -- objective + context + steps-so-far -> the single next step

Train/test split is by case. This prevents case overlap but is not itself a defense
against prompt leakage; ``tools/audit_dataset.py`` checks that property separately.

Data is assembled here; training runs elsewhere on GPU. Output is chat-format JSONL.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CORPUS_DIR = os.path.join(_ROOT, "data", "corpus")
OUT_DEFAULT = os.path.join(_ROOT, "data", "tuning_set.v2.jsonl")

# Cases reserved for evaluation of the decomposition behavior (kept out of train).
# Six cases spanning distinct attack families so the fixed split is not concentrated in
# one behavior. It remains development-facing and is not a population-generalization test.
DEFAULT_HOLDOUT = [
    "Pavur-SATCOM-eavesdrop", "Turla-satellite-C2", "Space-Odyssey-unauth-TC",
    "JTAG-debug-firmware", "GNSS-spoofing", "TC-replay-no-SDLS",
]

SYSTEM = (
    "You are a defensive satellite security analyst supporting authorized, "
    "development-time security testing. Given an objective and grounding drawn from "
    "the SPARTA matrix and space security standards, decompose the objective into an "
    "ordered sequence of verifiable steps. Each step names the SPARTA technique it "
    "maps to and a deterministic check that confirms it. Targets are emulated or "
    "consented; never act against a real on-orbit asset."
)

_CHECK_RE = re.compile(r"Deterministic check:\s*(.+)$", re.IGNORECASE | re.DOTALL)
_FIRST_SENT = re.compile(r"^(.*?\.)(?:\s|$)")
# Canonical SPARTA technique id, e.g. REC-0005, EX-0014.04. Distractors are drawn only
# from ids of this form so they are format-indistinguishable from reference techniques
# (the SV-XX-N space-vehicle threat records are a different taxonomy and are excluded).
_CANON_TECH_RE = re.compile(r"^[A-Z]{2,4}-\d{4}(?:\.\d{2})?$")


def _load_corpus_records() -> list[dict]:
    recs: list[dict] = []
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.json"))):
        recs.extend(json.load(open(path, encoding="utf-8")))
    return recs


def _extract_check(text: str) -> str:
    m = _CHECK_RE.search(text)
    return m.group(1).strip() if m else ""


def _action(text: str) -> str:
    """First sentence of the body, minus the trailing 'Deterministic check' part."""
    body = _CHECK_RE.split(text)[0]
    body = re.sub(r"Sub-step[^.]*\.\s*", "", body).strip()
    m = _FIRST_SENT.match(body)
    return (m.group(1) if m else body[:160]).strip()


def _truncate(text: str, n: int = 280) -> str:
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def _sparta_line(node: dict) -> str:
    s = node.get("sparta")
    if not s:
        note = (node.get("metadata") or {}).get("mapping_note", "no direct SPARTA technique")
        return f"(no SPARTA technique; {note})"
    return f"{s['technique_id']} {s['name']} [{s['tactic_id']}]"


def _technique_id(node: dict) -> str:
    s = node.get("sparta")
    return s["technique_id"] if s else ""


def _parent(tech_id: str) -> str:
    """Drop the sub-technique suffix: REC-0005.01 -> REC-0005."""
    return tech_id.split(".", 1)[0]


def _tech_line(tid: str, tech_by_id: dict[str, dict]) -> str | None:
    """Render one grounding line for a SPARTA technique id, or None if unknown."""
    rec = tech_by_id.get(f"sparta-{tid}")
    if not rec:
        return None
    return f"- SPARTA {tid} {rec['title'].split(': ', 1)[-1]}: {_truncate(rec.get('text', ''))}"


def _ref_tech_ids(root: dict, children: list[dict]) -> list[str]:
    """Reference technique ids cited by the case (root then children, deduped, in order)."""
    out: list[str] = []
    seen: set[str] = set()
    for node in [root, *children]:
        tid = _technique_id(node)
        if tid and tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def _sofar(children: list[dict], k: int, enrich: bool) -> str:
    """The 'steps so far' block for a next_step prompt. When enrich is set, each prior step
    also shows the SPARTA technique it used, giving the model its planning state (which
    techniques are already covered) rather than titles alone. These ids are planning
    state. Completion masking does not imply that contextual facts cannot affect the
    learned parameters."""
    if k == 0:
        return "(none yet)"
    lines = []
    for i in range(1, k + 1):
        child = children[i - 1]
        if enrich:
            tid = _technique_id(child)
            tag = f" [{tid}]" if tid else ""
            lines.append(f"{i}. {child['title']}{tag}")
        else:
            lines.append(f"{i}. {child['title']}")
    return "\n".join(lines)


def _render_step(idx: int, child: dict) -> str:
    check = _extract_check(child.get("text", ""))
    lines = [
        f"{idx}. {child['title']}",
        f"   - Technique: {_sparta_line(child)}",
        f"   - Action: {_action(child.get('text', ''))}",
    ]
    if check:
        lines.append(f"   - Check: {check}")
    return "\n".join(lines)


def _context(root: dict, children: list[dict], tech_by_id: dict[str, dict],
             tms_by_case: dict[str, list[dict]], pool: list[str] | None = None,
             target_total: int = 0) -> str:
    """Grounding block: the SPARTA techniques cited and the case threat model.

    When ``target_total`` > 0, the technique block is padded with plausible in-domain
    distractor techniques (drawn from ``pool``) up to ``target_total`` distinct entries,
    and the combined reference+distractor lines are shuffled so the retrieval order does
    not leak the reference order. Distractors never enter the gold plan: the model must
    SELECT the reference subset out of a noisy G(o) and ORDER it, which is what makes
    completeness/precision/grounding non-trivial. The distractor draw is deterministic
    per case (md5(case) seed) so the dataset is reproducible."""
    blocks: list[str] = []
    case = (root.get("metadata") or {}).get("case", "")
    for tm in tms_by_case.get(case, []):
        blocks.append(f"- Threat model: {tm['title']}. {_truncate(tm.get('text', ''))}")

    ref_ids = _ref_tech_ids(root, children)
    seen: set[str] = set()
    tech_lines: list[str] = []
    for tid in ref_ids:
        line = _tech_line(tid, tech_by_id)
        if line and tid not in seen:
            seen.add(tid)
            tech_lines.append(line)

    if target_total > 0 and pool is not None:
        rng = random.Random(int(hashlib.md5(case.encode()).hexdigest(), 16) % 2**32)
        ref_parents = {_parent(t) for t in seen}
        # candidate distractors: in-domain techniques whose PARENT is not a reference
        # parent (avoid near-miss ids that would blur select/ground), deduped vs ref.
        cands = [tid for tid in pool if tid not in seen and _parent(tid) not in ref_parents]
        rng.shuffle(cands)
        n = max(0, target_total - len(seen))
        for tid in cands[:n]:
            line = _tech_line(tid, tech_by_id)
            if line:
                tech_lines.append(line)
        rng.shuffle(tech_lines)  # hide reference order in the retrieved grounding

    blocks.extend(tech_lines)
    return "\n".join(blocks) if blocks else "(no additional grounding)"


def _plan(children: list[dict]) -> str:
    return "\n".join(_render_step(i, c) for i, c in enumerate(children, 1))


def build_examples(holdout: list[str], enrich_nextstep: bool = False,
                   target_total: int = 0) -> list[dict]:
    recs = _load_corpus_records()
    by_id = {r["id"]: r for r in recs}
    tech_by_id = {r["id"]: r for r in recs if r["id"].startswith("sparta-") and r["kind"] == "technique"}
    tms_by_case: dict[str, list[dict]] = {}
    for r in recs:
        if r["kind"] == "threat_model":
            tms_by_case.setdefault((r.get("metadata") or {}).get("case", ""), []).append(r)

    # Distractor pool: in-domain SPARTA technique ids of canonical form (excludes the
    # SV-XX-N threat records, whose different id shape would make a distractor trivially
    # identifiable). Sorted for a deterministic candidate order before the per-case shuffle.
    pool = sorted(t for tid in tech_by_id
                  if _CANON_TECH_RE.match(t := tid[len("sparta-"):]))

    roots = [r for r in recs if r["kind"] == "attack_tree" and r.get("children")
             and r["id"] != "at-telecommand-authority"]  # exclude illustrative seed root

    examples: list[dict] = []
    for root in roots:
        case = (root.get("metadata") or {}).get("case", "")
        split = "test" if case in holdout else "train"
        children = [by_id[c] for c in root["children"] if c in by_id]
        ctx = _context(root, children, tech_by_id, tms_by_case,
                       pool=pool, target_total=target_total)
        # Gold-leakage control: the legacy root narrative often enumerated the exact
        # reference chain. The model sees only the high-level root title here. Factual
        # candidates remain available in the shuffled grounding block below.
        objective = f"Objective: {root['title'].replace('ROOT: ', '')}"

        # type 1: full decomposition
        user = (f"{objective}\n\nGrounding:\n{ctx}\n\n"
                "Decompose this objective into an ordered plan of verifiable steps.")
        assistant = "Plan:\n" + _plan(children)
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "meta": {"type": "decompose", "case": case, "root": root["id"],
                     "split": split, "dataset_version": "2.0.0",
                     "prompt_policy": "objective-only-v2"},
        })

        # type 2: next-step planning (incremental)
        for k in range(len(children)):
            sofar = _sofar(children, k, enrich_nextstep)
            user = (f"{objective}\n\nGrounding:\n{ctx}\n\nSteps so far:\n{sofar}\n\n"
                    "Give the single next step.")
            assistant = _render_step(k + 1, children[k])
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "meta": {"type": "next_step", "case": case, "root": root["id"],
                         "step": k + 1, "split": split, "dataset_version": "2.0.0",
                         "prompt_policy": "objective-only-v2"},
            })
    return examples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", nargs="*", default=DEFAULT_HOLDOUT,
                    help="cases reserved for the test split")
    ap.add_argument("--enrich-nextstep", action="store_true",
                    help="show prior steps' SPARTA ids in the next_step 'steps so far' block")
    ap.add_argument("--distractors", type=int, default=0, metavar="N",
                    help="pad each grounding block to N total techniques with plausible "
                         "in-domain distractors and shuffle the block (0 = off, legacy)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args(argv)

    examples = build_examples(args.holdout, enrich_nextstep=args.enrich_nextstep,
                              target_total=args.distractors)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    by_type = Counter(e["meta"]["type"] for e in examples)
    by_split = Counter(e["meta"]["split"] for e in examples)
    by_case = Counter(e["meta"]["case"] for e in examples)
    print(f"wrote {len(examples)} examples -> {args.out}")
    print(f"  by type:  {dict(by_type)}")
    print(f"  by split: {dict(by_split)}  (holdout cases: {args.holdout})")
    print(f"  cases:    {len(by_case)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
