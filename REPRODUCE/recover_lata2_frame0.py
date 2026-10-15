"""Recover the omitted baseline frame for LATA2 from its source calcium TIFF.

The stored LATA2/BEFOREDRUG fluorescence CSV has no values in frame 0, even
though the source image exists. The registration export starts at frame 1.
This script checks the saved frame 1 trace against that registered image before
extracting frame 0 with the same mask and background regions. It never changes
the source CSV.

Run with the local source tree that contains ``Total data`` and ``LATA2TRIAL``::

    python REPRODUCE/recover_lata2_frame0.py --source-root PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import cafin_core as core  # noqa: E402


SOURCE_REPAIRS = HERE / "source_repairs"
FRAME_ZERO_NAME = "lata2_before_frame0.csv"
PROVENANCE_NAME = "lata2_before_frame0_provenance.json"
FRAME_ZERO_RELATIVE = Path(
    "Total data/lat_trial2_beforedrug/ca2/"
    "AVG_C1-latA_trial2_beforedrug0000.tif"
)
FRAME_ONE_RELATIVE = Path(
    "Total data/lat_trial2_beforedrug_rigid_output/rigid_green_registered/"
    "registered_AVG_C1-latA_trial2_beforedrug0001.tif"
)
MASK_RELATIVE = Path("LATA2TRIAL/BEFOREDRUG/mask_0.tiff")
RAW_RELATIVE = Path("LATA2TRIAL/BEFOREDRUG/all_cells_raw.csv")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_means(image: np.ndarray, mask: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    counts = np.bincount(mask.ravel(), minlength=int(mask.max()) + 1)
    sums = np.bincount(
        mask.ravel(), weights=image.ravel(), minlength=len(counts)
    )
    if np.any(counts[cell_ids] == 0):
        raise ValueError("A source cell ID is missing from mask_0.tiff")
    return sums[cell_ids] / counts[cell_ids]


def recover(
    source_root: Path,
    *,
    output_dir: Path = SOURCE_REPAIRS,
    expected_cells: int = 479,
) -> tuple[Path, Path]:
    """Validate stored frame 1, then save one frame-0 row and its provenance."""
    source_root = Path(source_root).resolve()
    paths = {
        "frame0_tiff": source_root / FRAME_ZERO_RELATIVE,
        "frame1_registered_tiff": source_root / FRAME_ONE_RELATIVE,
        "mask_tiff": source_root / MASK_RELATIVE,
        "source_raw_csv": source_root / RAW_RELATIVE,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = pd.read_csv(paths["source_raw_csv"])
    mask = tifffile.imread(paths["mask_tiff"])
    frame0 = tifffile.imread(paths["frame0_tiff"])
    frame1 = tifffile.imread(paths["frame1_registered_tiff"])
    if mask.ndim != 2 or frame0.shape != mask.shape or frame1.shape != mask.shape:
        raise ValueError("Mask and calcium frames must be matching 2-D images")
    if mask.dtype.kind not in "iu" or np.any(mask < 0):
        raise ValueError("The source mask must contain nonnegative integer labels")
    ids = np.unique(mask)
    ids = ids[ids > 0]
    columns = [f"Cell_{int(i)}" for i in ids]
    if len(ids) != expected_cells or list(raw.columns) != ["Frame", *columns]:
        raise ValueError(
            f"Expected Frame and {expected_cells} mask-matched cell columns"
        )
    if len(raw) < 2 or raw.loc[0, "Frame"] != 0 or raw.loc[1, "Frame"] != 1:
        raise ValueError("Stored fluorescence CSV must begin with frames 0 and 1")
    if not raw.loc[0, columns].isna().all():
        raise ValueError("Stored frame 0 is populated; no repair is needed")
    if raw.loc[1, columns].isna().any():
        raise ValueError("Stored frame 1 has missing cell values")

    boxes = core.auto_bg_boxes(frame1)
    backgrounds = core.background_values({0: frame0, 1: frame1}, boxes)
    corrected = core.bg_subtract({0: frame0, 1: frame1}, boxes)
    id_numbers = ids.astype(int)
    frame1_means = _cell_means(corrected[1], mask, id_numbers)
    saved_frame1 = raw.loc[1, columns].to_numpy(dtype=float)
    max_error = float(np.max(np.abs(frame1_means - saved_frame1)))
    if not np.isfinite(max_error) or max_error >= 1e-6:
        raise ValueError(
            "Frame 1 does not reproduce the saved source trace: "
            f"maximum absolute error {max_error:.9g}"
        )

    frame0_means = _cell_means(corrected[0], mask, id_numbers)
    repaired = pd.DataFrame([[0, *frame0_means]], columns=["Frame", *columns])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / FRAME_ZERO_NAME
    provenance_path = output_dir / PROVENANCE_NAME
    repaired.to_csv(csv_path, index=False)
    provenance = {
        "trial": "LATA2TRIAL",
        "condition": "BEFOREDRUG",
        "recovered_frame": 0,
        "n_cells": len(columns),
        "method": "saved static mask and source calcium frame, with source background subtraction",
        "source_files": {
            key: {
                "path": str(path.relative_to(source_root)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for key, path in paths.items()
        },
        "background_boxes_xyxy": [list(map(int, box)) for box in boxes],
        "background_by_frame": {
            str(i): {
                "mean": float(backgrounds[i][0]),
                "box_medians": list(map(float, backgrounds[i][1])),
            }
            for i in (0, 1)
        },
        "frame1_validation": {
            "n_cells_compared": len(columns),
            "max_absolute_error": max_error,
            "tolerance": 1e-6,
        },
        "repaired_csv_sha256": _sha256(csv_path),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return csv_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True,
        help="Local directory containing Total data/ and LATA2TRIAL/",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SOURCE_REPAIRS,
        help="Directory for the checked frame-0 row and provenance",
    )
    args = parser.parse_args()
    row, provenance = recover(args.source_root, output_dir=args.output_dir)
    print(f"Validated and saved {row} and {provenance}")


if __name__ == "__main__":
    main()
