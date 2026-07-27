"""Merge raw prediction JSONL files deterministically and reject duplicate rows."""
from __future__ import annotations

import argparse
import glob
import json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", action="append", dest="patterns", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = sorted({path for pattern in args.patterns for path in glob.glob(pattern)})
    seen: set[tuple] = set()
    count = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in paths:
            for line in open(path, encoding="utf-8"):
                row = json.loads(line)
                key = (row.get("config"), row.get("seed"), row.get("case"),
                       row.get("type"), row.get("step"))
                if key in seen:
                    raise SystemExit(f"duplicate prediction key {key} from {path}")
                seen.add(key)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(f"merged {len(paths)} files / {count} unique rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
