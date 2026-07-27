#!/usr/bin/env python3
"""Build a complete 24-case source manifest from the published corpus records."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path


TECHNIQUE_RE = re.compile(r"- Technique:\s+(?:SPARTA\s+)?([A-Z]{2,4}-\d{4}(?:\.\d{2})?)")
GROUNDING_RE = re.compile(r"^- SPARTA\s+([A-Z]{2,4}-\d{4}(?:\.\d{2})?)\b", re.MULTILINE)
STEP_RE = re.compile(r"^\d+\.\s+", re.MULTILINE)
ACTION_RE = re.compile(r"^\s+- Action:\s+\S", re.MULTILINE)
CHECK_RE = re.compile(r"^\s+- Check:\s+\S", re.MULTILINE)


def markdown_report(report: dict) -> str:
    coverage = report["coverage"]
    lines = [
        "# Authored-reference evidence coverage",
        "",
        "This is a construct-transparency audit, not independent semantic certification.",
        "References are author-defined benchmark operationalizations reviewed against the",
        "recorded public sources, SPARTA identifiers, and explicit construction rules.",
        "",
        "## Exhaustive checks",
        "",
        f"- Cases with a resolved, non-empty recorded source: {coverage['cases_with_recorded_source']}/24",
        f"- References with an Action and Check field on every step: {coverage['references_with_complete_step_fields']}/24",
        f"- References whose technique identifiers all occur in the supplied candidate set: {coverage['references_with_all_techniques_grounded']}/24",
        f"- Unique reference technique identifiers checked against live SPARTA pages: {coverage['unique_reference_techniques_live_verified']}/{coverage['unique_reference_techniques']}",
        "",
        "## Per-case coverage",
        "",
        "| Case | Split | Steps | Unique techniques | Source | Complete fields | All techniques supplied |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["cases"]:
        lines.append(
            f"| {item['case']} | {item['split']} | {item['reference_step_count']} "
            f"| {item['unique_reference_technique_count']} | yes "
            f"| {'yes' if item['all_steps_have_action_and_check'] else 'no'} "
            f"| {'yes' if item['all_reference_techniques_in_candidates'] else 'no'} |"
        )
    lines += [
        "",
        "Source records support case provenance and mechanisms. Exact step boundaries,",
        "mappings, order, and check wording remain authored task definitions.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tuning_set.v2.jsonl")
    parser.add_argument("--out", default="artifacts/results/case_provenance.json")
    parser.add_argument(
        "--markdown-out", default="artifacts/results/reference_evidence_coverage.md"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_path = root / args.data
    records: dict[str, list[tuple[str, dict]]] = {}
    for filename in sorted(glob.glob(str(root / "data/corpus/*.json"))):
        rel = str(Path(filename).relative_to(root))
        for record in json.loads(Path(filename).read_text(encoding="utf-8")):
            records.setdefault(record.get("id", ""), []).append((rel, record))

    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    decompositions = [row for row in rows if row["meta"]["type"] == "decompose"]
    manifest = []
    for row in decompositions:
        meta = row["meta"]
        hits = records.get(meta["root"], [])
        if len(hits) != 1:
            raise ValueError(f"root {meta['root']!r} resolves to {len(hits)} corpus records")
        filename, record = hits[0]
        source = record.get("source", "").strip()
        if not source:
            raise ValueError(f"root {meta['root']!r} has no source field")
        record_meta = record.get("metadata", {})
        if record_meta.get("case") != meta["case"]:
            raise ValueError(f"case mismatch for {meta['root']!r}")
        prompt = row["messages"][-2]["content"]
        reference = row["messages"][-1]["content"]
        step_count = len(STEP_RE.findall(reference))
        action_count = len(ACTION_RE.findall(reference))
        check_count = len(CHECK_RE.findall(reference))
        technique_ids = TECHNIQUE_RE.findall(reference)
        grounding_ids = set(GROUNDING_RE.findall(prompt))
        if step_count == 0:
            raise ValueError(f"reference {meta['case']!r} has no parsed steps")
        manifest.append({
            "case": meta["case"],
            "split": meta["split"],
            "root": meta["root"],
            "root_record_file": filename,
            "source_as_recorded": source,
            "attribution_or_scope": record_meta.get("attribution"),
            "sparta_audit_status": record_meta.get("sparta_status"),
            "reference_step_count": step_count,
            "unique_reference_technique_count": len(set(technique_ids)),
            "all_steps_have_action_and_check": (
                action_count == step_count and check_count == step_count
            ),
            "all_reference_techniques_in_candidates": set(technique_ids) <= grounding_ids,
            "reference_status": (
                "author-defined benchmark operationalization; reviewed within the author "
                "team against cited sources and SPARTA; not independently certified"
            ),
        })
    cases = {item["case"] for item in manifest}
    if len(manifest) != 24 or len(cases) != 24:
        raise ValueError(f"expected 24 unique decompose cases, found {len(manifest)}/{len(cases)}")
    all_reference_ids = set()
    for row in decompositions:
        all_reference_ids.update(TECHNIQUE_RE.findall(row["messages"][-1]["content"]))
    sparta_path = root / "artifacts/results/sparta_audit.json"
    verified_ids = set()
    if sparta_path.exists():
        sparta = json.loads(sparta_path.read_text(encoding="utf-8"))
        verified_ids = {
            item["id"] for item in sparta.get("records", [])
            if item.get("online_status") == "pass"
            and item.get("id_match") is True
            and item.get("name_match") is True
        }
    coverage = {
        "cases_with_recorded_source": sum(bool(item["source_as_recorded"]) for item in manifest),
        "references_with_complete_step_fields": sum(
            item["all_steps_have_action_and_check"] for item in manifest
        ),
        "references_with_all_techniques_grounded": sum(
            item["all_reference_techniques_in_candidates"] for item in manifest
        ),
        "unique_reference_techniques": len(all_reference_ids),
        "unique_reference_techniques_live_verified": len(all_reference_ids & verified_ids),
    }
    report = {
        "dataset": args.data,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "case_count": len(manifest),
        "reference_role": (
            "Authored benchmark operationalization used to measure reference agreement; "
            "not independently certified ground truth."
        ),
        "coverage": coverage,
        "cases": manifest,
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_out = root / args.markdown_out
    md_out.write_text(markdown_report(report), encoding="utf-8")
    print(f"wrote {out.relative_to(root)} ({len(manifest)} cases)")
    print(f"wrote {md_out.relative_to(root)}")


if __name__ == "__main__":
    main()
