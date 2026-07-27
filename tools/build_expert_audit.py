#!/usr/bin/env python3
"""Build a deterministic packet for an optional future community audit.

The packet contains all 24 author references and one generation seed from the six fixed
adapter, schema, and two-shot conditions at each model size. It prepares an audit; it
does not claim that independent validation has occurred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SIZES = ("0.5b", "1.5b", "7b")
CONDITIONS = ("adapter", "schema", "fewshot")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def split_prompt(text: str) -> tuple[str, str]:
    marker = "\n\nGrounding:\n"
    if marker not in text:
        raise ValueError("unexpected user-prompt format")
    objective, rest = text.split(marker, 1)
    grounding, suffix = rest.rsplit("\n\nDecompose this objective", 1)
    if not suffix:
        raise ValueError("empty prompt suffix")
    return objective.removeprefix("Objective: ").strip(), grounding.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/expert_audit")
    parser.add_argument("--key-out", default="artifacts/results/expert_audit_key.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(root / "data" / "tuning_set.v2.jsonl")
    gold_rows = [row for row in rows if row["meta"]["type"] == "decompose"]
    gold_by_case = {row["meta"]["case"]: row for row in gold_rows}
    cases = sorted(gold_by_case)
    if len(cases) != 24:
        raise ValueError(f"expected 24 corpus cases, found {len(cases)}")
    fixed_cases = {
        row["meta"]["case"] for row in gold_rows if row["meta"]["split"] == "test"
    }
    if len(fixed_cases) != 6:
        raise ValueError(f"expected six fixed cases, found {len(fixed_cases)}")

    items: list[dict] = []
    for number, case in enumerate(cases, 1):
        row = gold_by_case[case]
        objective, grounding = split_prompt(row["messages"][-2]["content"])
        items.append({
            "item_id": f"REF-{number:02d}",
            "phase": "reference",
            "case_id": case,
            "objective": objective,
            "candidate_set": grounding,
            "plan": row["messages"][-1]["content"],
        })

    generated: list[tuple[str, str, str, dict]] = []
    for size in SIZES:
        for condition in CONDITIONS:
            path = root / "artifacts" / "raw_predictions" / f"fixed_{size}" / f"{condition}_0.jsonl"
            selected = [row for row in read_jsonl(path)
                        if row.get("type") == "decompose" and row.get("case") in fixed_cases]
            if len(selected) != len(fixed_cases):
                raise ValueError(
                    f"expected {len(fixed_cases)} rows in {path}, found {len(selected)}"
                )
            generated.extend((row["case"], size, condition, row) for row in selected)

    generated.sort(key=lambda item: hashlib.sha256(
        f"expert-audit-v2|{item[0]}|{item[1]}|{item[2]}".encode()).hexdigest())
    key: dict[str, dict[str, str | int]] = {}
    for number, (case, size, condition, row) in enumerate(generated, 1):
        objective, grounding = split_prompt(row["prompt_messages"][-1]["content"])
        item_id = f"OUT-{number:03d}"
        items.append({
            "item_id": item_id,
            "phase": "generated_output",
            "case_id": case,
            "objective": objective,
            "candidate_set": grounding,
            "plan": row["output"],
        })
        key[item_id] = {
            "case_id": case,
            "model_size": size,
            "condition": condition,
            "decoding_seed": int(row["seed"]),
            "source_config": row["config"],
        }

    with (out / "items.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    fields = [
        "item_id", "reviewer_role", "reviewer_id", "independent_of_authors_yes_no",
        "review_date_yyyy_mm_dd", "expertise_summary", "years_relevant_experience",
        "conflict_of_interest", "material_reviewed", "first_pass_frozen_at_utc",
        "review_record_id", "signed_attestation_yes_no", "signed_review_record_sha256",
        "technique_mapping_correct_1_to_5", "action_coherent_1_to_5",
        "order_plausible_1_to_5", "checks_deterministic_1_to_5",
        "critical_error_yes_no", "acceptable_yes_no", "correction_required_yes_no",
        "correction_summary", "notes",
    ]
    with (out / "reference_responses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            if item["phase"] == "reference":
                for role in ("internal-author", "independent-non-author"):
                    writer.writerow({"item_id": item["item_id"], "reviewer_role": role})

    with (out / "generated_output_responses.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            if item["phase"] == "generated_output":
                writer.writerow({"item_id": item["item_id"]})

    adjudication_fields = [
        "item_id", "adjudicator_id", "adjudication_date_yyyy_mm_dd",
        "disagreement_summary", "disposition", "correction_applied_yes_no",
        "correction_ledger_entry", "notes",
    ]
    with (out / "adjudications.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=adjudication_fields).writeheader()

    readme = """# Optional community-review packet

Status: **optional extension, not undertaken**. The paper treats its references as
author-defined benchmark operationalizations and does not require this packet to pass.
It is supplied so later work can compare author-team judgments with a qualified non-author.

`items.jsonl` contains 78 items: all 24 author references (`REF-*`) and 54 blinded
generated outputs (`OUT-*`) for the six fixed cases, covering one preregistered
representative decoding seed (seed 0), three model sizes, and adapter/schema/two-shot conditions. The output labels
do not reveal model size or condition. The adjudication key is intentionally written
outside this directory to `../results/expert_audit_key.json`; do not give it to auditors.

If undertaken, the frozen comparison threshold covers the 24 `REF-*` items. The 54
`OUT-*` items are a separately reported semantic-output study. Blank templates are the
expected released state and do not indicate missing evidence for the present paper.

Audit protocol:

1. Complete `reference_responses.csv`: exactly one `internal-author` and one
   `independent-non-author` row per reference. Use stable blinded IDs. Record expertise,
   years of relevant practice, conflicts, review date, exact material reviewed, and whether
   the reviewer is independent of the authors. A non-author reviewer qualifies through
   documented professional or research experience in security plus direct familiarity with
   at least one of satellite, embedded, or communications security and with SPARTA-style
   technique mapping; record that basis rather than relying on a title alone.
2. Rate each dimension from 1 (incorrect/unusable) to 5 (correct/strong). Judge mappings
   against SPARTA, actions against the stated authorized development-time objective,
   order against causal prerequisites, and checks for deterministic executability.
3. Mark any safety, factual, mapping, or causal defect that invalidates the plan as a
   critical error. Mark overall acceptability independently of formatting quality.
4. Freeze each first-pass record before discussion or condition disclosure. Keep the signed
   form privately, put its stable record ID and SHA-256 in every applicable row, and set the
   attestation field only after signing. This anonymous packet publishes hashes, not names.
5. Record requested corrections without changing the first-pass rows. Put later disagreement
   resolutions in `adjudications.csv` and source changes in `correction_ledger.csv`.
6. The optional study's frozen threshold is: both roles on every reference, no critical-error rating, no
   unresolved correction, and median >=4 on every dimension. The analyzer reports agreement
   only for ratings of the same item and fails closed on missing role or provenance fields.

Only after real reviewers volunteer to perform this follow-up, run
`python tools/analyze_expert_audit.py`. The untouched template intentionally produces an
`incomplete` optional-study report and a nonzero exit status.

The packet deliberately does not convert blank ratings into a result. Analysis must occur
only after real, completed responses are received.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    key_path = root / args.key_out
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({
        "warning": "Keep this condition key from auditors until ratings are frozen.",
        "sampling": "All 24 references; fixed-case outputs at decoding seed 0; all three sizes; adapter, schema, two-shot.",
        "items": key,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(root)} ({len(items)} items)")
    print(f"wrote blinded-key file {key_path.relative_to(root)}")


if __name__ == "__main__":
    main()
