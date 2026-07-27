"""Export the decomposition tuning set to a Hugging Face dataset repo.

Reads ``data/tuning_set.v2.jsonl`` (chat format + a ``meta`` block) and
pushes a ``DatasetDict`` with train/test splits. The splits are disjoint by
*case*: the cases in ``build_tuning_set.DEFAULT_HOLDOUT`` are reserved for
evaluation and never appear in train, so publishing the split as-is preserves
that guarantee.

The dataset card (README.md) and license (LICENSE) that live next to this script
are uploaded to the same repo, so the defensive-scope statement and provenance
travel with the data.

Usage (from the testbed root, conda env ``mcp``):

    export PYTHONPATH=~/Workspace/Papers/msc/LLM-satsec/satsec-testbed/src

    # 1. Inspect locally, no upload:
    python hf/export_hf.py --repo you/satsec-decomposition --dry-run

    # 2. Push PRIVATE while the paper is in review:
    python hf/export_hf.py --repo you/satsec-decomposition --private

Requires: ``pip install "datasets>=2.19" "huggingface_hub>=0.23"`` and a WRITE
token (``huggingface-cli login``).
"""
from __future__ import annotations

import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DEFAULT_SRC = os.path.join(_ROOT, "data", "tuning_set.v2.jsonl")
CARD = os.path.join(_HERE, "README.md")
LICENSE = os.path.join(_HERE, "LICENSE")


def load_rows(path: str) -> dict[str, list[dict]]:
    """Flatten each JSONL record into HF columns, bucketed by meta.split.

    ``step`` only appears on next_step rows in the source; fill it with None on
    the rest so the Parquet schema keeps the column instead of dropping it (HF
    infers the schema from the first row, which would otherwise lose ``step``).
    """
    rows: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            meta = ex["meta"]
            row = {"messages": ex["messages"], **meta}
            row.setdefault("step", None)
            rows.setdefault(meta["split"], []).append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="target dataset repo, e.g. you/satsec-decomposition")
    ap.add_argument("--data", default=DEFAULT_SRC,
                    help="audited v2 JSONL source")
    ap.add_argument("--private", action="store_true",
                    help="create/push the repo as private (recommended during review)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the DatasetDict locally and save_to_disk, do not upload")
    args = ap.parse_args(argv)

    from datasets import Dataset, DatasetDict  # imported late so --help works without deps

    rows = load_rows(args.data)
    dd = DatasetDict({s: Dataset.from_list(r) for s, r in sorted(rows.items())})
    print(dd)

    if args.dry_run:
        out = os.path.join(_ROOT, "data", "hf_export")
        dd.save_to_disk(out)
        print(f"saved locally -> {out}  (no upload)")
        return 0

    dd.push_to_hub(args.repo, private=args.private)

    # Ship the card, license, and the raw JSONL (human-readable source of truth,
    # kept under raw/ so load_dataset only picks up the Parquet in data/).
    from huggingface_hub import upload_file
    repo_kwargs = dict(repo_id=args.repo, repo_type="dataset")
    if os.path.exists(CARD):
        upload_file(path_or_fileobj=CARD, path_in_repo="README.md", **repo_kwargs)
    if os.path.exists(LICENSE):
        upload_file(path_or_fileobj=LICENSE, path_in_repo="LICENSE", **repo_kwargs)
    if os.path.exists(args.data):
        upload_file(path_or_fileobj=args.data, path_in_repo="raw/tuning_set.v2.jsonl",
                    **repo_kwargs)

    visibility = "private" if args.private else "public"
    print(f"pushed ({visibility}) -> https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
