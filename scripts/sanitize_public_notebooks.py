#!/usr/bin/env python3
"""Remove raw-data analysis blocks from the public CIPHER notebooks.

Cell numbers refer to the archived notebooks supplied for this release. Removed
code cells are replaced by a short markdown disclosure so figure provenance is
not misleading. The public boundary starts at derived artifacts such as model
predictions, IG/Grad-CAM, ranks, embeddings and aggregate result tables.
"""

import json
from pathlib import Path


REMOVE = {
    "main_figures/AAA_final_fig3.ipynb": {
        13: "original fundus images and participant-level image selection",
        14: "original fundus image export and participant-level image selection",
        15: "original fundus image vessel extraction",
        21: "raw ECG waveform loading and rendering",
        27: "sample-level protein/SNP input values",
        34: "superseded raw-input correlation block",
        40: "superseded raw-input correlation block",
        56: "protein abundance/expression-compatible panel scoring",
        57: "protein abundance/expression-compatible panel scoring",
    },
    "main_figures/AAA_final_fig4.ipynb": {
        17: "sample-level protein expression/input values",
        20: "sample-level protein expression/input values",
        26: "sample-level protein expression/input values",
        34: "sample-level protein and SNP input values",
        35: "sample-level protein and SNP input values",
        36: "sample-level protein and SNP input values",
        37: "sample-level protein input values",
        38: "original fundus images",
        39: "raw ECG waveforms",
    },
    "main_figures/AAA_final_fig5.ipynb": {
        1: "participant-level clinical and baseline data",
        2: "participant-level clinical and baseline data",
        3: "participant-level baseline clinical data",
        8: "participant-level clinical data",
        12: "participant-level clinical subgroup data",
        14: "participant-level clinical data",
        20: "participant-level clinical data",
        21: "participant-level clinical modeling and baseline cohort data",
        22: "participant-level clinical modeling and baseline cohort data",
        25: "local notebook shell search",
        26: "participant-level baseline and modality-membership data",
    },
    "supplementary_figures/AAA_final_figs1.ipynb": {
        1: "participant-level baseline clinical data",
        2: "participant-level baseline event and clinical data",
        5: "local shell listing",
    },
    "supplementary_figures/AAA_final_figs2.ipynb": {
        6: "sample-level protein expression/input values",
    },
    "supplementary_figures/AAA_final_figs5.ipynb": {
        1: "raw ECG waveform loading and rendering",
    },
    "supplementary_figures/AAA_final_figs6.ipynb": {
        1: "original fundus images and participant-level image selection",
        2: "original fundus images and participant-level image selection",
        3: "original fundus images and participant-level image selection",
    },
}


def disclosure(reason: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"public_release": "raw_data_code_removed"},
        "source": [
            "### Restricted-data analysis not included\n",
            "\n",
            f"This block used {reason}. The public workflow starts from its "
            "de-identified derived output; no raw-data access or processing code "
            "is included in this release.\n",
        ],
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    notebooks = repo / "notebooks"
    removed = 0

    for relative, cells in REMOVE.items():
        path = notebooks / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        for one_based_index, reason in cells.items():
            index = one_based_index - 1
            current = payload["cells"][index]
            if current.get("cell_type") == "markdown" and current.get("metadata", {}).get(
                "public_release"
            ) == "raw_data_code_removed":
                continue
            if current.get("cell_type") != "code":
                raise RuntimeError(f"Expected code cell: {relative} cell {one_based_index}")
            payload["cells"][index] = disclosure(reason)
            removed += 1
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"Replaced {removed} restricted-data code cells with disclosure notes.")


if __name__ == "__main__":
    main()
