"""
recompute_from_raw.py -- regenerate ΔF/F₀ from the RAW microscopy TIFFs, running
the full documented chain end-to-end (no reliance on the pre-existing CSVs):

  1. rigid ECC registration: membrane -> frame 0, warp applied to calcium
  2. Cellpose cyto3 segmentation (channels=[2,0], diameter 15) on registered frame-0 membrane
  3. background subtraction of the calcium channel (3 auto regions, 1.5xIQR, avg of medians)
  4. per-cell mean-intensity traces -> ΔF/F₀ = (Ft-F0)/F0  (F0 = lowest-activity frames)
  5. ROI export (cells intersecting a rectangular ROI) -> separate CSV
  6. centroids from the mask

Only LATA1 has raw TIFF stacks, so this regenerates LATA1 BEFOREDRUG + AFTERDRUG.
LATA2 (no raw TIFFs) continues to use its existing pipeline CSVs; reproduce.py
labels the provenance of each field.

Outputs -> REPRODUCE/regenerated/LATA1TRAIL/<COND>/
Run:  python recompute_from_raw.py
"""
import os, sys, glob, json
try:
    sys.stdout.reconfigure(encoding="utf-8")   # allow Δ/dF-F0 glyphs in console prints
except Exception:
    pass
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cafin_core as cc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "regenerated", "LATA1TRAIL")
TRIAL = "LATA1TRAIL"
# ROI rectangles (x1,y1,x2,y2) reconstructed from the group's original roi_cells selection
ROI_BOX = {"BEFOREDRUG": (146, 205, 355, 325), "AFTERDRUG": (144, 195, 372, 310)}


def base_of(folder):
    f = sorted(glob.glob(os.path.join(folder, "*.tif")))[0]
    return os.path.basename(f)[:-8]


def recompute(cond):
    memf = os.path.join(ROOT, TRIAL, cond, "membrane")
    caf = os.path.join(ROOT, TRIAL, cond, "ca2")
    mem_base = base_of(memf); ca_base = base_of(caf)
    n = len([f for f in os.listdir(memf) if f.startswith(mem_base) and f.endswith(".tif")])
    print(f"\n=== {cond}: {n} frames | mem_base={mem_base} ===")

    # 1. register (rigid ECC), membrane -> frame0, warp applied to calcium
    print("  registering (rigid ECC)...")
    reg = cc.register_series(memf, caf, mem_base, ca_base, n, method="rigid",
                             do_tracking=False, mask0=None,
                             progress=lambda f, m: None)

    # 2. segment registered frame-0 membrane with cyto3 + channels=[2,0]
    print("  segmenting (Cellpose cyto3, channels=[2,0], diam 15)...")
    mask0, used_gpu = cc.segment(reg["reg_mem"][0], diameter=15, gpu=False, model_type="cyto3")
    n_cells = int(mask0.max())
    print(f"  -> {n_cells} cells (gpu={used_gpu})")

    # 3+4. background-subtracted per-cell traces, then ΔF/F₀
    print("  background subtraction + traces + ΔF/F₀...")
    raw_df = cc.extract_traces(reg, mask0, "static", bg=True)      # bg=True: 3 auto regions, 1.5xIQR
    dff_df, base = cc.compute_dff0(raw_df, method="lowest")

    # 5. ROI export
    box = ROI_BOX[cond]
    roi_ids = cc.roi_cell_ids(mask0, box)
    roi_cols = ["Frame"] + [f"Cell_{i}" for i in roi_ids if f"Cell_{i}" in dff_df.columns]
    roi_df = dff_df[roi_cols]

    # 6. centroids from mask
    cens = [{"cell_number": int(p.label), "x_position": float(p.centroid[1]),
             "y_position": float(p.centroid[0])} for p in regionprops(mask0)]
    cen_df = pd.DataFrame(cens)

    # write
    d = os.path.join(OUT, cond); os.makedirs(d, exist_ok=True)
    raw_df.to_csv(os.path.join(d, "all_cells_raw.csv"), index=False)
    dff_df.to_csv(os.path.join(d, "all_cells_normalized.csv"), index=False)
    roi_df.to_csv(os.path.join(d, "roi_cells_normalized.csv"), index=False)
    cen_df.to_csv(os.path.join(d, "centroids_0.csv"), index=False)
    tifffile.imwrite(os.path.join(d, "mask_0.tiff"), mask0.astype(np.uint16))
    meta = {"cond": cond, "n_frames": n, "n_cells": n_cells, "n_roi_cells": len(roi_ids),
            "roi_box": box, "baseline_frames": base, "segmentation": "cyto3 channels=[2,0] diam=15",
            "registration": "rigid ECC", "background": "3 auto regions, 1.5xIQR, avg-of-medians"}
    json.dump(meta, open(os.path.join(d, "provenance.json"), "w"), indent=2)
    print(f"  wrote regenerated CSVs -> {d}  ({n_cells} cells, {len(roi_ids)} in ROI)")
    return meta


def main():
    os.makedirs(OUT, exist_ok=True)
    metas = [recompute(c) for c in ["BEFOREDRUG", "AFTERDRUG"]]
    json.dump(metas, open(os.path.join(HERE, "regenerated", "recompute_summary.json"), "w"), indent=2)
    print("\nDONE. Regenerated LATA1 from raw images end-to-end.")


if __name__ == "__main__":
    main()
