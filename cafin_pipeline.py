"""
cafin_pipeline.py  --  parameterized, corrected CAFIN pipeline.

Given a trial folder (with membrane/ and ca2/ tiff stacks) it:
  1. rigid ECC registers membrane -> frame0, applies warp to calcium
  2. Cellpose cyto3 (with 3.x/4.x compatibility) segments the registered frame-0 membrane
  3. extracts per-cell calcium traces and computes dF/F0

dF/F0 (per user's choice): F and F0 are taken from the RAW (registered,
NOT background-subtracted) calcium, with a floor on F0, so the dim calcium
channel does not blow the ratio up.  F0 = mean of the lowest-activity frames.

Also returns tissue-level metrics used in the Results:
  amplitude, active fraction, peak frequency, spatial heterogeneity (CV),
  temporal synchronization (mean pairwise correlation).
"""
import os, cv2, warnings
import numpy as np
import pandas as pd
import tifffile
import skimage.io as skio
from skimage.measure import regionprops
from scipy.signal import find_peaks
import cafin_core as cc
warnings.filterwarnings("ignore")

PEAK_THRESHOLD = 0.5     # dF/F0 to count as a transient
F0_FLOOR = 1.0           # guard against divide-by-tiny


def _stretch8(img, lo=1, hi=99):
    img = img.astype(np.float32)
    p_lo, p_hi = np.percentile(img, [lo, hi])
    p_hi = p_hi if p_hi > p_lo else p_lo + 1
    return (np.clip((img - p_lo) / (p_hi - p_lo), 0, 1) * 255).astype(np.uint8)


def _register(data_root, mem_base, ca_base, num_frames, out):
    raw_dir = os.path.join(out, "rigid_raw_registered")
    green_dir = os.path.join(out, "rigid_green_registered")
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(green_dir, exist_ok=True)
    memf = os.path.join(data_root, "membrane"); caf = os.path.join(data_root, "ca2")

    first_mem = skio.imread(os.path.join(memf, f"{mem_base}0000.tif"))
    tifffile.imwrite(os.path.join(raw_dir, f"m0000.tif"), first_mem)
    first_ca = skio.imread(os.path.join(caf, f"{ca_base}0000.tif"))
    tifffile.imwrite(os.path.join(green_dir, "c0000.tif"), first_ca)

    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-6)
    for i in range(1, num_frames):
        mp = os.path.join(memf, f"{mem_base}{i:04d}.tif")
        if not os.path.exists(mp): continue
        mov = skio.imread(mp)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(first_mem.astype(np.float32), mov.astype(np.float32),
                                           warp, cv2.MOTION_EUCLIDEAN, crit)
        except cv2.error:
            pass
        reg = cv2.warpAffine(mov, warp, (first_mem.shape[1], first_mem.shape[0]),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        tifffile.imwrite(os.path.join(raw_dir, f"m{i:04d}.tif"), reg.astype(np.uint16))
        gp = os.path.join(caf, f"{ca_base}{i:04d}.tif")
        if os.path.exists(gp):
            g = skio.imread(gp)
            rg = cv2.warpAffine(g, warp, (g.shape[1], g.shape[0]),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            tifffile.imwrite(os.path.join(green_dir, f"c{i:04d}.tif"), rg.astype(np.uint16))
    return raw_dir, green_dir, first_mem


def _gpu_available():
    """Use the shared CUDA, DirectML, and Apple-MPS detector."""
    try:
        return cc.gpu_status()[0]
    except Exception:
        return False


def _segment(raw_dir, out):
    img = skio.imread(os.path.join(raw_dir, "m0000.tif"))
    masks, _used_gpu = cc.segment(img, diameter=15, gpu=_gpu_available(), model_type="cyto3")
    masks = np.asarray(masks).astype(np.uint16)
    seg = os.path.join(out, "cell_segmentation_work"); os.makedirs(seg, exist_ok=True)
    tifffile.imwrite(os.path.join(seg, "mask_0.tiff"), masks)
    return masks


def _traces_dff0(green_dir, masks, num_frames):
    """dF/F0 from RAW registered calcium + F0 floor (user's choice)."""
    ids = np.unique(masks); ids = ids[ids > 0]
    raw = {c: [] for c in ids}
    for f in range(num_frames):
        p = os.path.join(green_dir, f"c{f:04d}.tif")
        if not os.path.exists(p):
            for c in ids: raw[c].append(np.nan); continue
        ca = tifffile.imread(p)
        for c in ids:
            px = ca[masks == c]
            raw[c].append(float(px.mean()) if px.size else np.nan)
    favg = [np.nanmean([raw[c][f] for c in ids]) for f in range(num_frames)]
    nb = min(10, max(3, num_frames // 3))
    base = sorted(sorted(range(num_frames), key=lambda f: favg[f])[:nb])
    dff = {}
    for c in ids:
        f0 = np.nanmean([raw[c][f] for f in base])
        f0 = max(f0, F0_FLOOR)
        dff[c] = [(v - f0) / f0 if not np.isnan(v) else np.nan for v in raw[c]]
    raw_df = pd.DataFrame({f"Cell_{c}": raw[c] for c in ids}); raw_df.insert(0, "Frame", range(num_frames))
    dff_df = pd.DataFrame({f"Cell_{c}": dff[c] for c in ids}); dff_df.insert(0, "Frame", range(num_frames))
    return raw_df, dff_df, base


def _metrics(dff_df, num_frames):
    cells = [c for c in dff_df.columns if c.startswith("Cell_")]
    A = dff_df[cells].to_numpy(float)
    A = np.nan_to_num(A, nan=0.0)                       # frames x cells
    peak_per_cell, amps = [], []
    for j in range(A.shape[1]):
        pk, _ = find_peaks(A[:, j], height=PEAK_THRESHOLD, distance=2)
        peak_per_cell.append(len(pk))
        if len(pk): amps.extend(A[pk, j])
    peak_per_cell = np.array(peak_per_cell)
    percell_max = A.max(axis=0)
    active = percell_max > PEAK_THRESHOLD
    # spatial heterogeneity: CV of per-cell peak amplitude across the field
    pm = percell_max[percell_max > 0]
    heterogeneity_cv = float(np.std(pm) / np.mean(pm)) if pm.size else np.nan
    # temporal synchronization: mean pairwise Pearson r among active cells
    act = A[:, active]
    if act.shape[1] >= 2:
        cc = np.corrcoef(act.T)
        iu = np.triu_indices_from(cc, k=1)
        synchronization = float(np.nanmean(cc[iu]))
    else:
        synchronization = np.nan
    return {
        "n_cells": int(A.shape[1]),
        "active_fraction": float(active.mean()),
        "mean_peak_amp_dff0": float(np.mean(amps)) if amps else np.nan,
        "median_peak_amp_dff0": float(np.median(amps)) if amps else np.nan,
        "peak_freq_per_cell_per_frame": float(peak_per_cell.sum() / (A.shape[1] * num_frames)),
        "mean_peaks_per_cell": float(peak_per_cell.mean()),
        "spatial_heterogeneity_cv": heterogeneity_cv,
        "temporal_synchronization_r": synchronization,
    }


def run(label, data_root, mem_base, ca_base):
    memf = os.path.join(data_root, "membrane")
    num_frames = len([f for f in os.listdir(memf) if f.startswith(mem_base) and f.endswith(".tif")])
    out = data_root + "_output"; os.makedirs(out, exist_ok=True)
    print(f"\n########## {label}  ({num_frames} frames) ##########")
    raw_dir, green_dir, first_mem = _register(data_root, mem_base, ca_base, num_frames, out)
    print("  registered.")
    masks = _segment(raw_dir, out)
    print(f"  segmented {int(masks.max())} cells.")
    raw_df, dff_df, base = _traces_dff0(green_dir, masks, num_frames)
    data_dir = os.path.join(out, "calcium_analysis", "data"); os.makedirs(data_dir, exist_ok=True)
    raw_df.to_csv(os.path.join(data_dir, "all_cells_raw.csv"), index=False)
    dff_df.to_csv(os.path.join(data_dir, "all_cells_normalized.csv"), index=False)
    m = _metrics(dff_df, num_frames); m["label"] = label; m["num_frames"] = num_frames
    m["baseline_frames"] = base
    print(f"  metrics: {m}")
    return m
