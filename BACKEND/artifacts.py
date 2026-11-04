"""Validate the audited artifact bundle before deserializing trusted files."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path


def verify_bundle(project_root):
    root = Path(project_root)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for package, expected in manifest["versions"].items():
        if version(package) != expected:
            raise RuntimeError(
                f"{package} must be version {expected} for the saved artifacts; "
                "install requirements.txt."
            )
    for name, expected in manifest["sha256"].items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required Elaria artifact not found: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact bundle integrity mismatch: {name}")
    return manifest
