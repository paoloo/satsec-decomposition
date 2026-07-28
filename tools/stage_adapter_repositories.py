#!/usr/bin/env python3
"""Stage the 57 public adapter folders and record immutable file metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from tools.build_manifest_index import adapter_subfolder, size_from_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_folder(root: Path, manifest: Path) -> Path:
    relative = manifest.relative_to(root / "artifacts/manifests")
    if relative.parts[0] == "fixed":
        return root / "models" / manifest.parent.name
    return root / "artifacts" / Path(*relative.parts[:-1])


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage(root: Path, stage_root: Path, metadata_path: Path) -> dict:
    if stage_root.exists() and any(stage_root.iterdir()):
        raise ValueError(f"staging directory is not empty: {stage_root}")
    stage_root.mkdir(parents=True, exist_ok=True)

    for size in ("0.5b", "1.5b", "7b"):
        repo_root = stage_root / size
        repo_root.mkdir()
        shutil.copy2(root / f"hf/models/{size}/README.md", repo_root / "README.md")

    entries = []
    manifests_root = root / "artifacts/manifests"
    manifests = sorted(manifests_root.glob("**/run_manifest.json"))
    for manifest in manifests:
        size = size_from_path(manifest)
        _, subfolder = adapter_subfolder(manifest)
        source = source_folder(root, manifest)
        weight = source / "adapter_model.safetensors"
        config = source / "adapter_config.json"
        if not weight.is_file() or not config.is_file():
            raise FileNotFoundError(f"incomplete adapter folder: {source}")
        if weight.stat().st_size < 1024:
            raise ValueError(f"implausibly small adapter: {weight}")

        destination = stage_root / size / subfolder
        destination.mkdir(parents=True)
        link_or_copy(weight, destination / weight.name)
        shutil.copy2(config, destination / config.name)
        shutil.copy2(manifest, destination / manifest.name)
        entries.append(
            {
                "size": size,
                "adapter_subfolder": subfolder,
                "adapter_bytes": weight.stat().st_size,
                "adapter_sha256": sha256(weight),
                "adapter_config_sha256": sha256(config),
                "manifest_sha256": sha256(manifest),
            }
        )

    if len(entries) != 57:
        raise ValueError(f"staged {len(entries)} adapters, expected 57")
    report = {
        "schema_version": "1.0.0",
        "adapter_count": len(entries),
        "total_adapter_bytes": sum(entry["adapter_bytes"] for entry in entries),
        "entries": sorted(
            entries, key=lambda entry: (entry["size"], entry["adapter_subfolder"])
        ),
    }
    metadata_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def metadata_from_inventory(root: Path, inventory_path: Path, metadata_path: Path) -> dict:
    """Validate a remote staging inventory against public manifests."""
    files: dict[tuple[str, str], dict[str, tuple[str, int]]] = {}
    for number, line in enumerate(inventory_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, raw_bytes, raw_path = line.split("\t", 2)
        except ValueError as exc:
            raise ValueError(f"invalid inventory line {number}") from exc
        path = Path(raw_path)
        if path.parts[:2] != ("/", "stage") or len(path.parts) < 5:
            raise ValueError(f"unexpected staged path: {path}")
        size = path.parts[2]
        subfolder = str(Path(*path.parts[3:-1]))
        files.setdefault((size, subfolder), {})[path.name] = (digest, int(raw_bytes))

    expected_names = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "run_manifest.json",
    }
    manifest_by_key = {}
    for manifest in sorted((root / "artifacts/manifests").glob("**/run_manifest.json")):
        size = size_from_path(manifest)
        _, subfolder = adapter_subfolder(manifest)
        manifest_by_key[(size, subfolder)] = manifest
    if set(files) != set(manifest_by_key):
        missing = sorted(set(manifest_by_key) - set(files))
        extra = sorted(set(files) - set(manifest_by_key))
        raise ValueError(f"inventory mismatch; missing={missing}, extra={extra}")

    entries = []
    for (size, subfolder), staged_files in sorted(files.items()):
        if set(staged_files) != expected_names:
            raise ValueError(f"unexpected files for {size}:{subfolder}: {sorted(staged_files)}")
        manifest = manifest_by_key[(size, subfolder)]
        manifest_digest, _ = staged_files["run_manifest.json"]
        if manifest_digest != sha256(manifest):
            raise ValueError(f"remote/public manifest mismatch: {size}:{subfolder}")
        weight_digest, weight_bytes = staged_files["adapter_model.safetensors"]
        config_digest, _ = staged_files["adapter_config.json"]
        entries.append(
            {
                "size": size,
                "adapter_subfolder": subfolder,
                "adapter_bytes": weight_bytes,
                "adapter_sha256": weight_digest,
                "adapter_config_sha256": config_digest,
                "manifest_sha256": manifest_digest,
            }
        )

    report = {
        "schema_version": "1.0.0",
        "adapter_count": len(entries),
        "total_adapter_bytes": sum(entry["adapter_bytes"] for entry in entries),
        "entries": entries,
    }
    if report["adapter_count"] != 57:
        raise ValueError(f"inventoried {report['adapter_count']} adapters, expected 57")
    metadata_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stage", type=Path)
    source.add_argument("--inventory", type=Path)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "artifacts/manifests/adapter_files.json",
    )
    args = parser.parse_args()
    if args.stage:
        report = stage(root, args.stage.resolve(), args.metadata.resolve())
    else:
        report = metadata_from_inventory(
            root, args.inventory.resolve(), args.metadata.resolve()
        )
    print(json.dumps({key: report[key] for key in ("adapter_count", "total_adapter_bytes")}))


if __name__ == "__main__":
    main()
