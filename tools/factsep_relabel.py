"""Grounding-relabeling probe: relabel identifiers while keeping plan shape fixed.

This controlled intervention tests whether predictions track a consistent relabeling of
the identifiers supplied in the grounding. It does not inspect or constrain model
parameters and cannot establish that facts are absent from weights.

For each held-out case we apply a deterministic bijection over the SPARTA technique ids:
every technique id used in the case (in the grounding block AND in the reference plan) is
consistently rewritten to a DISJOINT valid SPARTA id that does not appear in the original
case, and its inline human name is swapped to the replacement technique's real name. The
step count, the step order, the grounding-set membership, and the check text are all
preserved. Only the factual identity of each technique changes.

What this probes:
  * A structure-learner copies the (relabeled) grounding ids into the same ordered slots,
    so completeness/ordering against the relabeled reference stay HIGH and the invariance
    gap |original - relabeled| is small.
  * A model insensitive to the supplied mapping may emit the ORIGINAL ids; against the
    relabeled reference completeness and grounding validity then drop. A large gap is
    evidence of poor grounding reliance, but a small gap is not a no-memorization proof.

Output is a drop-in data file (all cases split=test, ``meta.relabel_map`` recorded) that
``decomp_generate.py`` and ``decomp_score.py`` consume unchanged. Run the adapter and base
on both the original held-out file and this relabeled file, score each, and report the
per-config invariance gap. Pure Python, no GPU. Deterministic given --seed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTBED = os.path.dirname(_HERE)
CORPUS_DIR = os.path.join(_TESTBED, "data", "corpus")

# Full technique id including an optional sub-technique.  Relabel full identifiers,
# never a parent plus a preserved suffix: preserving '.04' while changing its parent
# can manufacture a nonexistent SPARTA id.
_ID_RE = re.compile(r"\b([A-Z]{2,4}-\d{4}(?:\.\d{2})?)\b")
_CANON_ID_RE = re.compile(r"^[A-Z]{2,4}-\d{4}(?:\.\d{2})?$")


def load_tech_names() -> dict[str, str]:
    """Exact technique-id -> human name, read from the corpus records."""
    names: dict[str, str] = {}
    for path in glob.glob(os.path.join(CORPUS_DIR, "*.json")):
        for rec in json.load(open(path, encoding="utf-8")):
            rid = rec.get("id", "")
            if rid.startswith("sparta-") and rec.get("kind") == "technique":
                tid = rid[len("sparta-"):]
                # title looks like "REC-0005: Eavesdrop"; keep the name half.
                name = rec.get("title", "").split(": ", 1)[-1]
                if _CANON_ID_RE.fullmatch(tid):
                    names.setdefault(tid, name)
    return names


def ids_in(text: str) -> set[str]:
    return {m.group(1) for m in _ID_RE.finditer(text)}


def build_case_map(used: set[str], pool: list[str], rng: random.Random) -> dict[str, str]:
    """Bijection over exact ids used in a case -> disjoint, valid exact ids."""
    targets = [p for p in pool if p not in used]
    rng.shuffle(targets)
    if len(targets) < len(used):
        raise SystemExit("not enough disjoint SPARTA ids to relabel; corpus too small")
    return {src: targets[i] for i, src in enumerate(sorted(used))}


def relabel_text(text: str, cmap: dict[str, str], names: dict[str, str]) -> str:
    """Rewrite every technique id (and its trailing inline name) under the case map."""
    def repl(m: re.Match) -> str:
        tid = m.group(1)
        return cmap.get(tid, tid)

    out = _ID_RE.sub(repl, text)

    # swap the inline human name that follows a relabeled id so grounding stays coherent:
    # "NEWID Old Name" / "NEWID Old Name [TACTIC]" / "SPARTA NEWID Old Name: desc"
    for src, dst in cmap.items():
        old_name = re.escape(names.get(src, ""))
        new_name = names.get(dst, "")
        if not old_name or not new_name:
            continue
        out = re.sub(rf"({re.escape(dst)})\s+{old_name}",
                     rf"\1 {new_name}", out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="snapshot tuning_set.jsonl")
    ap.add_argument("--split", default="test", help="which split to relabel (default test)")
    ap.add_argument("--out", required=True, help="relabeled data file (all split=test)")
    ap.add_argument("--seed", type=int, default=0, help="bijection seed (reproducible)")
    args = ap.parse_args(argv)

    names = load_tech_names()
    # Relabel only to exact canonical ids known in the corpus/scorer.
    pool = sorted(p for p in names if _CANON_ID_RE.match(p))
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]

    # A case shares one bijection across its decompose + next_step examples: derive the
    # map from the case's decompose example (has grounding + full plan = every id) and
    # reuse it, so the relabel is internally consistent across example types.
    case_used: dict[str, set[str]] = {}
    for ex in rows:
        if ex["meta"].get("split") != args.split:
            continue
        u = set()
        for m in ex["messages"]:
            u |= ids_in(m["content"])
        case_used.setdefault(ex["meta"]["case"], set()).update(u)

    case_map: dict[str, dict[str, str]] = {}
    for case, used in case_used.items():
        rng = random.Random(f"{args.seed}:{case}")
        case_map[case] = build_case_map(used, pool, rng)

    out_rows = []
    for ex in rows:
        if ex["meta"].get("split") != args.split:
            continue
        cmap = case_map[ex["meta"]["case"]]
        ex2 = json.loads(json.dumps(ex))
        for m in ex2["messages"]:
            m["content"] = relabel_text(m["content"], cmap, names)
        ex2["meta"]["split"] = "test"
        ex2["meta"]["relabel_map"] = cmap
        out_rows.append(ex2)

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_cases = len(case_map)
    print(f"[factsep_relabel] relabeled {len(out_rows)} examples over {n_cases} cases "
          f"-> {args.out}")
    for case, cmap in case_map.items():
        pairs = ", ".join(f"{s}->{d}" for s, d in list(cmap.items())[:3])
        print(f"  {case:28s} {len(cmap)} ids  e.g. {pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
