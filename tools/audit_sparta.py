"""Audit gold technique identifiers against the current official SPARTA pages.

The online check verifies that every gold identifier resolves and that the page exposes
the expected identifier/name. It cannot decide whether a case-to-technique semantic
mapping is correct; that remains an expert-review item and is stated in the report.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

from satsec.training.decomp_score import parse_plan


def technique_url(tid: str) -> str:
    if "." in tid:
        parent, child = tid.split(".", 1)
        return f"https://sparta.aerospace.org/technique/{parent}/{child}/"
    return f"https://sparta.aerospace.org/technique/{tid}/"


def expected_name(assistant: str, tid: str) -> str:
    pattern = re.compile(rf"Technique:\s*{re.escape(tid)}\s+(.+?)(?:\s+\[[^]]+\])?\s*$",
                         re.MULTILINE)
    match = pattern.search(assistant)
    return match.group(1).strip() if match else ""


def strip_html(text: str) -> str:
    text = re.sub(r"<script\b.*?</script>|<style\b.*?</style>", " ", text,
                  flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/tuning_set.v2.jsonl")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--report", default="artifacts/results/sparta_audit.json")
    args = ap.parse_args()

    used: dict[str, dict] = {}
    cases: dict[str, set[str]] = defaultdict(set)
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        if ex["meta"]["type"] != "decompose":
            continue
        assistant = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        for step in parse_plan(assistant):
            tid = step.technique_id
            cases[tid].add(ex["meta"]["case"])
            used.setdefault(tid, {"id": tid, "expected_name": expected_name(assistant, tid),
                                  "url": technique_url(tid)})

    errors: list[str] = []
    records: list[dict] = []
    for tid in sorted(used):
        rec = {**used[tid], "cases": sorted(cases[tid]), "online_status": "not-run"}
        if args.online:
            request = urllib.request.Request(rec["url"], headers={"User-Agent": "satsec-audit/2.0"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = strip_html(response.read().decode("utf-8", errors="replace"))
                id_ok = re.search(rf"\bID:\s*{re.escape(tid)}\b", body) is not None
                name_ok = rec["expected_name"].casefold() in body.casefold()
                rec.update(online_status="pass" if id_ok and name_ok else "fail",
                           id_match=id_ok, name_match=name_ok)
                if rec["online_status"] == "fail":
                    errors.append(f"{tid}: official page id/name mismatch")
            except Exception as exc:  # network failures belong in the machine report
                rec.update(online_status="error", error=f"{type(exc).__name__}: {exc}")
                errors.append(f"{tid}: could not retrieve official page")
        records.append(rec)

    report = {
        "audit_date": date.today().isoformat(),
        "dataset": args.data,
        "unique_gold_techniques": len(records),
        "online": args.online,
        "scope": "identifier existence and expected-name match only",
        "semantic_review_required": True,
        "known_corrected_mapping": {
            "case": "GNSS-spoofing",
            "removed": "EX-0002 (PNT Geofencing)",
            "gold": "EX-0014.04 (PNT Spoofing)",
        },
        "records": records,
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("unique_gold_techniques", "online", "errors", "status")},
                     indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

