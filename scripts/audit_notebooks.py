#!/usr/bin/env python3
"""Flag portability and reproducibility hazards in CIPHER notebooks."""

import json
import re
from pathlib import Path


PATTERNS = {
    "absolute path": re.compile(r"/(?:home|Users|data\w*)/"),
    "shell command": re.compile(r"^\s*!", re.MULTILINE),
    "manual selection": re.compile(r"manual|allowed_journals|highlight_genes", re.I),
    "disabled p/FDR filter": re.compile(r"(?:P_CUTOFF|FDR_CUTOFF|alpha)\s*=\s*1(?:\.0)?\b", re.I),
    "risk quantile": re.compile(r"(?:HIGH_RISK_Q|LOW_RISK_Q|HIGH_RISK_FRAC|LOW_RISK_FRAC)\s*="),
    "top-N/fraction selection": re.compile(r"(?:TOP_N|BOTTOM_N|TOP_FRAC|BOTTOM_FRAC)\s*="),
    "bootstrap setting": re.compile(r"N_BOOT\s*="),
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks").rglob("*.ipynb"))
    total = 0
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        for index, cell in enumerate(payload.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            for label, pattern in PATTERNS.items():
                if pattern.search(source):
                    total += 1
                    print(f"{notebook.relative_to(root)}: cell {index}: {label}")
    print(f"\n{total} flagged cell/category combinations in {len(notebooks)} notebooks.")


if __name__ == "__main__":
    main()
