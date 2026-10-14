#!/usr/bin/env python3
"""Check tracked/candidate files and exporter receipts before staging/push."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".feather", ".h5", ".hdf5", ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".xml", ".rds", ".bed", ".bim", ".fam", ".zip", ".gz", ".tar", ".log"}
REQUIRED_MODEL_MODULES = [
    "pipelines/models/__init__.py", "pipelines/models/audit_public.py",
    "pipelines/models/export.py", "pipelines/models/final_pair.py",
    "pipelines/models/public_artifact.py", "pipelines/models/publish_results.py",
    "pipelines/models/sanitize.py",
]


def release_files(root):
    if not root.exists():
        return set()
    if root.is_symlink():
        raise ValueError("Release directory may not be a symlink")
    manifest_path = root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") not in {"cipher-public-release-v1", "cipher-final-pair-v1"} or manifest.get("sample_rows") is not False:
        raise ValueError("Invalid public-release receipt")
    approved = {"release_manifest.json"}
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe receipt path")
        name = relative.as_posix()
        if not (name == "model_manifest.json" or (name in {"public_models/risk.joblib", "public_models/event.joblib"}) or (name.startswith("public_results/metrics_") and name.endswith(".csv"))):
            raise ValueError("Unreviewed release file type")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing/symlinked release file")
        if path.stat().st_size >= 95 * 1024 * 1024:
            raise ValueError("Model exceeds ordinary Git size budget; publish reviewed assets separately")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Release checksum mismatch")
        approved.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != approved:
        raise ValueError("Unlisted files in release directory")
    return {"release/" + p for p in approved}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--staged", action="store_true")
    p.add_argument("--stage", action="store_true", help="Stage audited code and receipted release files only")
    args = p.parse_args()
    missing = [name for name in REQUIRED_MODEL_MODULES if not (ROOT / name).is_file()]
    ignored = subprocess.run(["git", "check-ignore", "--no-index", "--"] + REQUIRED_MODEL_MODULES,
                             cwd=ROOT, capture_output=True, text=True)
    if missing or ignored.stdout.strip():
        raise SystemExit("Required model source is missing or ignored by Git")
    if ignored.returncode not in {0, 1}:
        raise SystemExit("Unable to verify model source Git rules")
    if args.staged:
        tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
        if set(REQUIRED_MODEL_MODULES) - tracked:
            raise SystemExit("Required model source is absent from the Git index")
    approved = release_files(ROOT / "release")
    command = ["git", "diff", "--cached", "--name-only", "-z"] if args.staged else ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    names = subprocess.check_output(command, cwd=ROOT).decode().split("\0")
    problems = []
    for name in filter(None, names):
        path = ROOT / name
        if not path.exists():
            continue  # staged deletions are permitted
        parts = PurePosixPath(name).parts
        if path.is_symlink() or any(part in {"private_runs", "data", "models", "outputs", "logs", "checkpoints"} for part in parts[:-1] if not name.startswith("pipelines/")):
            problems.append(name + ": private path/symlink")
        if path.suffix.lower() in FORBIDDEN and name not in approved:
            problems.append(name + ": data/binary outside verified release")
        if path.suffix.lower() == ".json" and name not in approved and name not in {"config/pipeline.example.json", "docs/PIPELINE_SOURCE_MANIFEST.json"}:
            problems.append(name + ": unreviewed JSON payload")
        if name.startswith("release/") and name not in approved:
            problems.append(name + ": unreceipted release file")
        if path.suffix.lower() in {".py", ".md", ".json", ".yaml", ".yml", ".sh", ".ipynb"}:
            source = path.read_text()
            if re.search(r"[\u3400-\u9fff]", source):
                problems.append(name + ": non-English repository text")
            if re.search(r"/(?:Users|home|share|data200T)/", source):
                problems.append(name + ": machine-specific absolute path")
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text())
            if any(c.get("outputs") or c.get("execution_count") is not None for c in notebook.get("cells", []) if c.get("cell_type") == "code"):
                problems.append(name + ": notebook contains execution output")
        if path.stat().st_size >= 95 * 1024 * 1024:
            problems.append(name + ": too large for ordinary Git")
    if problems:
        raise SystemExit("Blocked:\n" + "\n".join(problems))
    if approved:
        subprocess.run([sys.executable, "-m", "pipelines.models.audit_public", str(ROOT / "release")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-c", "from pipelines.models.final_pair import load_pair; load_pair('release')"], cwd=ROOT, check=True)
    if args.stage:
        code = ["README.md", ".gitignore", "docs", "config", "pipelines", "scripts", "tests", "notebooks", "environment.yml", "requirements-clinical.txt", "requirements-training.txt"]
        subprocess.run(["git", "add", "--"] + code, cwd=ROOT, check=True)
        if approved:
            subprocess.run(["git", "add", "-f", "--"] + sorted(approved), cwd=ROOT, check=True)
        subprocess.run([sys.executable, __file__, "--staged"], cwd=ROOT, check=True)
    print("Public release check passed (%d receipted files)." % len(approved))


if __name__ == "__main__":
    main()
