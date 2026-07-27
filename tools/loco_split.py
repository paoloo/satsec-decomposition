"""Leave-one-case-out (LOCO) cross-validation splits for the decomposition adapter.

The single 84/6 split in the decomp-adapt paper measures behaviour on six held-out
cases. That is a small sample. LOCO reuses the SAME 24-case corpus to evaluate EVERY
case out-of-training: for each fold one case (or a small group) is held out, the adapter
is trained on the rest, and the held-out case is scored. Averaged over folds, this turns
"six held-out cases" into "every case in the corpus, measured when it was unseen",
without any new data.

This tool only reshuffles the ``meta.split`` field of the published snapshot
(``decomp-adapt/data/tuning_set.jsonl``); the examples themselves are byte-identical to a
rebuild via ``build_tuning_set.py --holdout <case>`` (splitting is purely by case). It
does no GPU work. It emits, under ``--out-dir``:

  fold_XX_<case>/tuning_set.jsonl   one training file per fold (that case = test, rest = train)
  refs_all.jsonl                    every case marked split=test, for offline scoring
  folds.json                        manifest: fold id -> held-out case(s), train/test counts

Then, per fold, train on ``fold_XX_*/tuning_set.jsonl`` (train split) and generate on its
test split; concatenate all folds' predictions into one JSONL and score once against
``refs_all.jsonl`` with a single shared config label so the aggregator pools the folds:

  python3 -m satsec.training.decomp_score --pred loco_preds.jsonl \
      --data <out-dir>/refs_all.jsonl --split test

See benchmark/run_loco.sh for the container driver that loops the folds.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter


def read_rows(path: str) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cases_in_order(rows: list[dict]) -> list[str]:
    """Distinct case names in first-seen order (stable, so folds are reproducible)."""
    seen: list[str] = []
    for r in rows:
        c = r["meta"]["case"]
        if c not in seen:
            seen.append(c)
    return seen


def make_folds(cases: list[str], k: int | None) -> list[list[str]]:
    """Leave-one-out (k=None) or k grouped folds. Grouped folds keep families that are
    adjacent in corpus order together, which is fine because scoring pools all cases."""
    if k is None or k >= len(cases):
        return [[c] for c in cases]
    if k <= 0:
        raise SystemExit("--k must be positive")
    folds: list[list[str]] = [[] for _ in range(k)]
    for i, c in enumerate(cases):
        folds[i % k].append(c)
    return [f for f in folds if f]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def reassigned(rows: list[dict], holdout: set[str]) -> list[dict]:
    """Copy rows with meta.split set from the holdout set (does not mutate the input)."""
    out = []
    for r in rows:
        r2 = json.loads(json.dumps(r))
        r2["meta"]["split"] = "test" if r2["meta"]["case"] in holdout else "train"
        out.append(r2)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="published snapshot tuning_set.jsonl")
    ap.add_argument("--out-dir", required=True, help="directory for the fold files")
    ap.add_argument("--k", type=int, default=None,
                    help="grouped k-fold instead of leave-one-out (default: leave-one-out)")
    args = ap.parse_args(argv)

    rows = read_rows(args.data)
    cases = cases_in_order(rows)
    folds = make_folds(cases, args.k)
    os.makedirs(args.out_dir, exist_ok=True)

    # references file: every case is a test case somewhere, so mark all test for scoring.
    write_jsonl(os.path.join(args.out_dir, "refs_all.jsonl"),
                reassigned(rows, set(cases)))

    manifest = {"n_cases": len(cases), "n_folds": len(folds), "folds": []}
    for i, holdout in enumerate(folds):
        tag = holdout[0] if len(holdout) == 1 else f"grp{i:02d}"
        # keep the tag filesystem-safe
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in tag)
        fold_dir = os.path.join(args.out_dir, f"fold_{i:02d}_{safe}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_rows = reassigned(rows, set(holdout))
        write_jsonl(os.path.join(fold_dir, "tuning_set.jsonl"), fold_rows)
        counts = Counter(r["meta"]["split"] for r in fold_rows)
        manifest["folds"].append({
            "fold": i, "dir": os.path.relpath(fold_dir, args.out_dir),
            "holdout": holdout, "train": counts["train"], "test": counts["test"],
        })

    with open(os.path.join(args.out_dir, "folds.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[loco_split] {len(cases)} cases -> {len(folds)} folds in {args.out_dir}")
    for f in manifest["folds"]:
        print(f"  fold {f['fold']:02d}: holdout={f['holdout']}  "
              f"train={f['train']} test={f['test']}  ({f['dir']})")
    print(f"[loco_split] refs_all.jsonl + folds.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
