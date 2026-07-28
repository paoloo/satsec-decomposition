#!/usr/bin/env python3
"""Build the public manifest-to-adapter index with stable paths and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL_REPOSITORIES = {
    "0.5b": "paolocmo/satsec-decomposition-qwen2.5-0.5b-adapters",
    "1.5b": "paolocmo/satsec-decomposition-qwen2.5-1.5b-adapters",
    "7b": "paolocmo/satsec-decomposition-qwen2.5-7b-adapters",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def size_from_path(path: Path) -> str:
    for size in MODEL_REPOSITORIES:
        if size in str(path):
            return size
    raise ValueError(f"cannot infer model size from {path}")


def adapter_subfolder(path: Path) -> tuple[str, str]:
    parts = path.parts
    if "fixed" in parts and "training_seed" not in parts and "loco" not in parts:
        return "fixed", "fixed/seed-42"
    if "training_seed" in parts:
        seed = path.parent.name.replace("seed_", "seed-")
        return "training-seed", f"training-seeds/{seed}"
    if "loco" in parts:
        fold = path.parent.name
        return "loco", f"loco/{fold}"
    raise ValueError(f"cannot infer adapter kind from {path}")


def build(
    root: Path,
    revisions: dict[str, str] | None = None,
    adapter_files: dict[str, dict] | None = None,
) -> dict:
    manifests_root = root / "artifacts/manifests"
    entries = []
    for path in sorted(manifests_root.glob("**/run_manifest.json")):
        config_path = path.with_name("adapter_config.json")
        if not config_path.is_file():
            raise FileNotFoundError(f"missing public adapter configuration: {config_path}")
        size = size_from_path(path)
        kind, subfolder = adapter_subfolder(path)
        run = json.loads(path.read_text(encoding="utf-8"))
        revision = revisions.get(size) if revisions else None
        repo = MODEL_REPOSITORIES[size]
        file_record = adapter_files.get(f"{size}:{subfolder}") if adapter_files else None
        config_digest = sha256(config_path)
        if file_record and config_digest != file_record["adapter_config_sha256"]:
            raise ValueError(f"remote/public adapter configuration mismatch: {size}:{subfolder}")
        entries.append(
            {
                "kind": kind,
                "size": size,
                "training_seed": run["config"]["seed"],
                "dataset_sha256": run["dataset_sha256"],
                "base_model": run["base_model"],
                "base_model_revision": run["model_revision"],
                "manifest": str(path.relative_to(root)),
                "manifest_sha256": sha256(path),
                "configuration": str(config_path.relative_to(root)),
                "configuration_sha256": config_digest,
                "adapter_repository": repo,
                "adapter_revision": revision,
                "adapter_subfolder": subfolder,
                "adapter_file": (
                    f"{subfolder}/adapter_model.safetensors" if file_record else None
                ),
                "adapter_bytes": file_record["adapter_bytes"] if file_record else None,
                "adapter_sha256": file_record["adapter_sha256"] if file_record else None,
                "adapter_config_file": (
                    f"{subfolder}/adapter_config.json" if file_record else None
                ),
                "adapter_config_sha256": (
                    file_record["adapter_config_sha256"] if file_record else None
                ),
                "adapter_url": (
                    f"https://huggingface.co/{repo}/tree/{revision}/{subfolder}"
                    if revision
                    else None
                ),
                "adapter_file_url": (
                    f"https://huggingface.co/{repo}/resolve/{revision}/"
                    f"{subfolder}/adapter_model.safetensors"
                    if revision and file_record
                    else None
                ),
                "adapter_config_url": (
                    f"https://huggingface.co/{repo}/resolve/{revision}/"
                    f"{subfolder}/adapter_config.json"
                    if revision and file_record
                    else None
                ),
            }
        )
    counts = {
        kind: sum(entry["kind"] == kind for entry in entries)
        for kind in ("fixed", "training-seed", "loco")
    }
    if counts != {"fixed": 3, "training-seed": 6, "loco": 48}:
        raise ValueError(f"unexpected manifest counts: {counts}")
    if adapter_files and len(adapter_files) != len(entries):
        raise ValueError(
            f"unexpected adapter-file count: {len(adapter_files)} (expected {len(entries)})"
        )
    if adapter_files and any(entry["adapter_sha256"] is None for entry in entries):
        raise ValueError("adapter-file metadata does not cover every manifest")
    return {
        "schema_version": "1.1.0",
        "adapter_count": len(entries),
        "manifest_count": len(entries),
        "counts": counts,
        "repositories": MODEL_REPOSITORIES,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--revisions", type=Path)
    parser.add_argument(
        "--adapter-files",
        type=Path,
        default=root / "artifacts/manifests/adapter_files.json",
    )
    parser.add_argument("--out", type=Path, default=root / "artifacts/manifests/index.json")
    args = parser.parse_args()
    revisions = (
        json.loads(args.revisions.read_text(encoding="utf-8")) if args.revisions else None
    )
    adapter_files = None
    if args.adapter_files.exists():
        metadata = json.loads(args.adapter_files.read_text(encoding="utf-8"))
        adapter_files = {
            f"{entry['size']}:{entry['adapter_subfolder']}": entry
            for entry in metadata["entries"]
        }
    report = build(root, revisions, adapter_files)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
