"""Grounding-relabeling consistency: score original vs relabeled predictions.

Companion to tools/factsep_relabel.py. Given a config's predictions on the ORIGINAL
held-out cases and on the RELABELED cases (same structure, disjoint SPARTA ids), this
reports each metric on both and the gap |original - relabeled|. A small gap is evidence
that the model follows the supplied identifier mapping in this controlled intervention.
It is not evidence that vulnerability facts are absent from model weights. Pure Python,
no GPU.

  python3 factsep_gap.py \
      --orig-pred preds_adapter_orig.jsonl --orig-data tuning_set.jsonl \
      --relabel-pred preds_adapter_relabel.jsonl --relabel-data factsep_test.jsonl
"""
from __future__ import annotations

import argparse

from satsec.training.decomp_score import (
    aggregate,
    load_references,
    load_valid_sparta_ids,
)

_METRICS = ("completeness", "precision", "ordering", "grounding", "check")


def _table(pred: str, data: str, split: str, valid) -> dict:
    refs = load_references(data, split)
    return aggregate(pred, refs, valid)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-pred", required=True)
    ap.add_argument("--orig-data", required=True)
    ap.add_argument("--relabel-pred", required=True)
    ap.add_argument("--relabel-data", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args(argv)

    valid = load_valid_sparta_ids()
    orig = _table(args.orig_pred, args.orig_data, args.split, valid)
    rel = _table(args.relabel_pred, args.relabel_data, args.split, valid)

    configs = [c for c in orig if c in rel]
    if not configs:
        print("no shared config between the two prediction files")
        return 1

    print("Grounding-relabeling consistency (original vs relabeled; smaller |delta| "
          "means greater consistency under this intervention)\n")
    hdr = f"{'Config':28s} {'metric':13s} {'original':>10s} {'relabeled':>10s} {'|gap|':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for c in configs:
        for m in _METRICS:
            o = orig[c][m][0]
            r = rel[c][m][0]
            print(f"{c:28s} {m:13s} {o:10.2f} {r:10.2f} {abs(o - r):7.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
