"""Recalculate the four paper fields using each cell's own fluorescence baseline.

Input is the stored per-cell *raw fluorescence* CSV for each field. This step does
not repeat image registration, segmentation, or fluorescence extraction. It keeps
those source files intact and writes separate, auditable normalized datasets.

Run from any directory: python REPRODUCE/normalize_f0_per_cell.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT_ROOT = HERE / "normalized_per_cell"
SOURCE_REPAIRS = HERE / "source_repairs"
FIELDS = (
    ("LATA1TRAIL", "BEFOREDRUG"),
    ("LATA1TRAIL", "AFTERDRUG"),
    ("LATA2TRIAL", "BEFOREDRUG"),
    ("LATA2TRIAL", "AFTERDRUG"),
)
N_BASE = 10
F0_FLOOR = 1.0


def source_dir(trial: str, condition: str) -> Path:
    """LATA1 uses traces regenerated from TIFF; LATA2 uses stored pipeline traces."""
    if (trial, condition) not in FIELDS:
        raise ValueError(f"Unknown field: {trial}/{condition}")
    if trial == "LATA1TRAIL":
        return HERE / "regenerated" / trial / condition
    return ROOT / trial / condition


def apply_frame_zero_repair(
    raw_df: pd.DataFrame,
    trial: str,
    condition: str,
    raw_path: Path,
    *,
    repair_root: Path = SOURCE_REPAIRS,
) -> tuple[pd.DataFrame, dict | None]:
    """Fill the known missing LATA2 baseline frame only after source checks."""
    if (trial, condition) != ("LATA2TRIAL", "BEFOREDRUG"):
        return raw_df, None
    if raw_df.empty or "Frame" not in raw_df.columns or raw_df.loc[0, "Frame"] != 0:
        raise ValueError("LATA2 baseline traces must begin with frame 0")
    cells = list(raw_df.columns[1:])
    if not cells or not all(c.startswith("Cell_") for c in cells):
        raise ValueError("LATA2 baseline traces need Cell_ columns")
    if not raw_df.loc[0, cells].isna().all():
        return raw_df, None

    repair_root = Path(repair_root)
    row_path = repair_root / "lata2_before_frame0.csv"
    provenance_path = repair_root / "lata2_before_frame0_provenance.json"
    for path in (row_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing source-verified frame-0 repair: {path}")
    repair_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
    recorded_hash = repair_provenance["source_files"]["source_raw_csv"]["sha256"]
    repair_hash = hashlib.sha256(row_path.read_bytes()).hexdigest()
    validation = repair_provenance["frame1_validation"]
    if source_hash != recorded_hash or repair_hash != repair_provenance["repaired_csv_sha256"]:
        raise ValueError("Frame-0 repair does not match its verified source files")
    if (
        repair_provenance["trial"] != trial
        or repair_provenance["condition"] != condition
        or repair_provenance["recovered_frame"] != 0
        or validation["n_cells_compared"] != len(cells)
        or validation["max_absolute_error"] >= validation["tolerance"]
    ):
        raise ValueError("Frame-0 repair provenance is incomplete or invalid")
    repaired_row = pd.read_csv(row_path)
    if (
        len(repaired_row) != 1
        or list(repaired_row.columns) != list(raw_df.columns)
        or repaired_row.loc[0, "Frame"] != 0
        or not np.isfinite(repaired_row.loc[0, cells].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Frame-0 repair row does not match the source cell columns")
    updated = raw_df.copy()
    updated.loc[0, cells] = repaired_row.loc[0, cells].to_numpy(dtype=float)
    repair_info = {
        "recovered_frame": 0,
        "source_repair_csv": str(row_path.relative_to(ROOT)).replace("\\", "/"),
        "source_repair_sha256": repair_hash,
        "source_repair_provenance": str(provenance_path.relative_to(ROOT)).replace("\\", "/"),
        "source_repair_provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "frame1_validation": validation,
        "source_files": repair_provenance["source_files"],
    }
    return updated, repair_info


def normalize_traces(
    raw_df: pd.DataFrame, *, n_base: int = N_BASE, floor: float = F0_FLOOR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match cafin_core.compute_dff0(method='percell') and retain each F0i.

    Each cell selects its own lowest-intensity rows. The same cell-specific F0i
    then normalizes all of that cell's frames; no population F0 is calculated.
    """
    cells = [c for c in raw_df.columns if c.startswith("Cell_")]
    if not cells or raw_df.empty or "Frame" not in raw_df.columns:
        raise ValueError("Raw CSV must contain Frame and at least one Cell_ column")
    if n_base < 1 or floor <= 0:
        raise ValueError("n_base and floor must be positive")

    n = len(raw_df)
    effective_n_base = int(min(n_base, max(3, n // 3)))
    normalized = raw_df.copy()
    f0_rows = []
    for cell in cells:
        raw = raw_df[cell].to_numpy(float)
        # Identical row selection and F0 floor to cafin_core.f0_per_cell.
        rows = sorted(int(i) for i in np.argsort(np.nan_to_num(raw, nan=np.inf))[:effective_n_base])
        f0 = max(float(np.nanmean(raw[rows])), floor)
        normalized[cell] = (raw - f0) / f0
        f0_rows.append(
            {
                "cell": cell,
                "f0i": f0,
                "baseline_row_indices": json.dumps(rows),
                "baseline_frames": json.dumps(raw_df["Frame"].iloc[rows].tolist()),
            }
        )
    return normalized, pd.DataFrame(f0_rows)


def normalize_field(
    trial: str, condition: str, *, output_root: Path = OUTPUT_ROOT
) -> Path:
    """Write a field's normalized traces, per-cell baselines, ROI, and provenance."""
    source = source_dir(trial, condition)
    raw_path = source / "all_cells_raw.csv"
    roi_path = source / "roi_cells_normalized.csv"
    centroid_path = source / "centroids_0.csv"
    for path in (raw_path, roi_path, centroid_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")

    raw_df = pd.read_csv(raw_path)
    raw_df, repair_info = apply_frame_zero_repair(raw_df, trial, condition, raw_path)
    normalized, f0_table = normalize_traces(raw_df)
    roi_source = pd.read_csv(roi_path, nrows=0)
    roi_columns = list(roi_source.columns)
    missing = [c for c in roi_columns if c not in normalized.columns]
    if "Frame" not in roi_columns or missing:
        raise ValueError(f"ROI columns do not match raw cell traces: {missing}")

    output = Path(output_root) / trial / condition
    output.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output / "all_cells_normalized.csv", index=False)
    f0_table.to_csv(output / "f0_per_cell.csv", index=False)
    normalized[roi_columns].to_csv(output / "roi_cells_normalized.csv", index=False)
    shutil.copyfile(centroid_path, output / "centroids_0.csv")

    provenance = {
        "trial": trial,
        "condition": condition,
        "source_raw_csv": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "source_raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "source_roi_membership_csv": str(roi_path.relative_to(ROOT)).replace("\\", "/"),
        "source_centroids_csv": str(centroid_path.relative_to(ROOT)).replace("\\", "/"),
        "source_type": "raw-TIFF-regenerated traces" if trial == "LATA1TRAIL" else "stored pipeline traces",
        "normalization": "percell",
        "formula": "dF/F0i = (F_t_i - F0i) / F0i",
        "baseline_selection": "each cell's own lowest-intensity frames",
        "n_base_requested": N_BASE,
        "n_base_effective": int(min(N_BASE, max(3, len(raw_df) // 3))),
        "f0_floor": F0_FLOOR,
        "n_frames": len(raw_df),
        "n_cells": len(f0_table),
        "n_roi_cells": len(roi_columns) - 1,
    }
    if repair_info is not None:
        provenance["source_repair"] = repair_info
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    for trial, condition in FIELDS:
        output = normalize_field(trial, condition)
        print(f"{trial}/{condition}: {output}")


if __name__ == "__main__":
    main()
