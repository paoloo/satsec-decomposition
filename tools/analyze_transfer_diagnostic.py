#!/usr/bin/env python3
"""Recompute the public crackme transfer diagnostic from released API responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import deque
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SEMANTIC_METRICS = (
    "completeness",
    "precision",
    "ordering_fidelity",
    "grounding_validity",
)
REPORT_METRICS = SEMANTIC_METRICS + (
    "semantic_check_presence",
    "exact_check_presence",
    "exact_contract_rate",
)
CHECK_ALIASES = {"check", "deterministic_check"}
FENCED_JSON = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
SENSITIVE_TEXT = re.compile(
    r"(?:/Users/|/home/|\batadev\b|\bdev-coyote\b|authorization\s*:|bearer\s+)",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_json(text: str) -> dict[str, Any]:
    candidates = [match.group(1) for match in FENCED_JSON.finditer(text)] + [text]
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model output is not a complete JSON object")


def normalized_key(value: str) -> str:
    return re.sub(r"[\s-]+", "_", value.lower().strip())


def present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(present(item) for item in value)
    if isinstance(value, dict):
        return bool(value)
    return False


def find_check(step: dict[str, Any]) -> tuple[str, Any] | None:
    for key, value in step.items():
        if normalized_key(key) in CHECK_ALIASES and present(value):
            return key, value
    return None


def zero_score(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "completeness": 0.0,
        "precision": 0.0,
        "ordering_fidelity": 1.0,
        "grounding_validity": 0.0,
        "counts": {
            "reference_steps": len(reference["steps"]),
            "predicted_steps": 0,
            "recovered_steps": 0,
            "distinct_reference_techniques": len(
                {step["technique"] for step in reference["steps"]}
            ),
            "distinct_emitted_techniques": 0,
            "selected_reference_techniques": 0,
            "comparable_precedence_pairs": 0,
            "precedence_violations": 0,
            "grounded_steps": 0,
        },
        "matches": [],
    }


def score_decomposition(
    reference: dict[str, Any], predicted_steps: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_steps = reference["steps"]
    remaining: dict[str, deque[int]] = {}
    for index, step in enumerate(reference_steps):
        remaining.setdefault(step["technique"], deque()).append(index)

    matches = []
    for predicted_index, step in enumerate(predicted_steps):
        queue = remaining.get(step["technique"])
        if queue:
            matches.append(
                {
                    "reference_index": queue.popleft(),
                    "predicted_index": predicted_index,
                    "technique": step["technique"],
                }
            )

    comparable_pairs = 0
    precedence_violations = 0
    for left in range(len(matches)):
        for right in range(left + 1, len(matches)):
            first = matches[left]
            second = matches[right]
            if first["reference_index"] == second["reference_index"]:
                continue
            comparable_pairs += 1
            reference_direction = first["reference_index"] < second["reference_index"]
            predicted_direction = first["predicted_index"] < second["predicted_index"]
            precedence_violations += int(reference_direction != predicted_direction)

    grounding = set(reference["grounding"])
    reference_techniques = {step["technique"] for step in reference_steps}
    emitted_techniques = {step["technique"] for step in predicted_steps}
    selected_reference = len(emitted_techniques & reference_techniques)
    grounded_steps = sum(step["technique"] in grounding for step in predicted_steps)
    return {
        "completeness": len(matches) / len(reference_steps),
        "precision": selected_reference / len(emitted_techniques) if emitted_techniques else 0.0,
        "ordering_fidelity": (
            1.0
            if comparable_pairs == 0
            else 1.0 - precedence_violations / comparable_pairs
        ),
        "grounding_validity": grounded_steps / len(predicted_steps) if predicted_steps else 0.0,
        "counts": {
            "reference_steps": len(reference_steps),
            "predicted_steps": len(predicted_steps),
            "recovered_steps": len(matches),
            "distinct_reference_techniques": len(reference_techniques),
            "distinct_emitted_techniques": len(emitted_techniques),
            "selected_reference_techniques": selected_reference,
            "comparable_precedence_pairs": comparable_pairs,
            "precedence_violations": precedence_violations,
            "grounded_steps": grounded_steps,
        },
        "matches": matches,
    }


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    return {"mean": mean(values), "std": pstdev(values)}


def load_prompts(path: Path) -> dict[str, dict[str, Any]]:
    prompts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            prompts[row["case_id"]] = row
    return prompts


def analyze(capsule: Path) -> dict[str, Any]:
    prompts_path = capsule / "prompts.jsonl"
    references_dir = capsule / "references"
    runs_dir = capsule / "runs"
    manifest_path = runs_dir / "manifest.json"
    manifest = read_json(manifest_path)
    prompts = load_prompts(prompts_path)
    references = {
        reference["case_id"]: reference
        for path in sorted(references_dir.glob("*.json"))
        for reference in [read_json(path)]
    }

    if sha256(prompts_path) != manifest["prompts_sha256"]:
        raise ValueError("prompt file does not match the manifest SHA-256")
    if set(prompts) != set(references):
        raise ValueError("prompt and reference case sets differ")
    expected_seeds = manifest["seeds"]
    if expected_seeds != [0, 1, 2, 3, 4]:
        raise ValueError(f"unexpected seed sequence: {expected_seeds}")
    if len(manifest["runs"]) != len(references) * len(expected_seeds):
        raise ValueError("manifest does not contain exactly one run per case and seed")

    cases: dict[str, Any] = {}
    for case_id in sorted(references):
        reference = references[case_id]
        attempts = sorted(
            (run for run in manifest["runs"] if run["case_id"] == case_id),
            key=lambda run: run["repetition"],
        )
        if [run["seed"] for run in attempts] != expected_seeds:
            raise ValueError(f"seed mismatch for {case_id}")
        runs = []
        for attempt in attempts:
            repetition = attempt["repetition"]
            request_path = runs_dir / case_id / f"repeat-{repetition}.request.json"
            response_path = runs_dir / case_id / f"repeat-{repetition}.response.json"
            request = read_json(request_path)
            response = read_json(response_path)
            if request["seed"] != attempt["seed"]:
                raise ValueError(f"request seed mismatch for {case_id}/{repetition}")
            if request["messages"] != prompts[case_id]["messages"]:
                raise ValueError(f"request prompt mismatch for {case_id}/{repetition}")
            serialized_response = response_path.read_text(encoding="utf-8")
            if SENSITIVE_TEXT.search(serialized_response):
                raise ValueError(f"unsanitized response metadata in {response_path}")
            message = response["choices"][0]["message"]
            if set(message) != {"role", "content"}:
                raise ValueError(f"response contains unreleased message fields in {response_path}")

            score = zero_score(reference)
            parse_error = None
            exact_contract = False
            semantic_check_presence = 0.0
            exact_check_presence = 0.0
            observed_check_keys: list[str] = []
            try:
                parsed = extract_json(message["content"])
                prediction = parsed.get("plan", parsed)
                steps = prediction.get("steps")
                if not isinstance(steps, list) or not steps:
                    raise ValueError("prediction has no steps array")
                if any(
                    not isinstance(step, dict)
                    or not isinstance(step.get("action"), str)
                    or not isinstance(step.get("technique"), str)
                    for step in steps
                ):
                    raise ValueError("one or more steps lack action or technique text")
                check_matches = [find_check(step) for step in steps]
                semantic_check_presence = sum(
                    match is not None for match in check_matches
                ) / len(steps)
                exact_check_presence = sum(
                    present(step.get("check")) for step in steps
                ) / len(steps)
                observed_check_keys = sorted({match[0] for match in check_matches if match})
                exact_contract = all(
                    step["action"].strip()
                    and step["technique"].strip()
                    and present(step.get("check"))
                    for step in steps
                )
                score = score_decomposition(reference, steps)
            except (KeyError, TypeError, ValueError) as error:
                parse_error = str(error)

            runs.append(
                {
                    "repetition": repetition,
                    "seed": attempt["seed"],
                    "request_file": str(request_path.relative_to(capsule)),
                    "response_file": str(response_path.relative_to(capsule)),
                    "response_sha256": sha256(response_path),
                    "parse_error": parse_error,
                    "exact_contract": exact_contract,
                    "semantic_check_presence": semantic_check_presence,
                    "exact_check_presence": exact_check_presence,
                    "observed_check_keys": observed_check_keys,
                    "metrics": score,
                }
            )
        cases[case_id] = {"runs": runs}

    per_repetition = []
    for repetition in range(len(expected_seeds)):
        selected = [cases[case]["runs"][repetition] for case in sorted(cases)]
        per_repetition.append(
            {
                "repetition": repetition,
                "seed": expected_seeds[repetition],
                **{
                    metric: mean(run["metrics"][metric] for run in selected)
                    for metric in SEMANTIC_METRICS
                },
                "semantic_check_presence": mean(
                    run["semantic_check_presence"] for run in selected
                ),
                "exact_check_presence": mean(run["exact_check_presence"] for run in selected),
                "exact_contract_rate": mean(float(run["exact_contract"]) for run in selected),
            }
        )

    return {
        "schema_version": "1.0.0",
        "diagnostic": (
            "Post-hoc format-neutral analysis; check and deterministic_check are accepted "
            "as semantic check aliases, while exact-contract rates require literal check."
        ),
        "model": manifest["model_build"],
        "model_reproducibility": manifest["model_reproducibility"],
        "prompts_sha256": sha256(prompts_path),
        "manifest_sha256": sha256(manifest_path),
        "attempted_runs": len(manifest["runs"]),
        "strict_valid_runs": sum(bool(run.get("parsed")) for run in manifest["runs"]),
        "format_neutral_json_runs": sum(
            run["parse_error"] is None
            for case in cases.values()
            for run in case["runs"]
        ),
        "cases": cases,
        "per_repetition_macro": per_repetition,
        "summary": {
            metric: summarize([row[metric] for row in per_repetition])
            for metric in REPORT_METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--capsule",
        type=Path,
        default=root / "artifacts/transfer_diagnostic/crackmes",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "artifacts/transfer_diagnostic/crackmes/analysis.json",
    )
    args = parser.parse_args()
    report = analyze(args.capsule.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "attempted_runs", "strict_valid_runs", "format_neutral_json_runs", "summary"
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
