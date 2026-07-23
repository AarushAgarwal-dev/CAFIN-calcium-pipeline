"""
CAFIN calcium-analysis pipeline  --  CORRECTED / RUNNABLE version
=================================================================
Fixes applied vs. the original notebook (CafinPaper1.ipynb):

1.  cellpose 4.x (cpsam) API.  The notebook used CellposeDenoiseModel /
    model_type="cyto3" / channels=[2,0], all removed in cellpose>=4.  This
    crashed segmentation, so no mask was produced and every downstream
    "result" was empty -> the numbers in the manuscript were not real.
2.  Rigid registration now also writes the calcium reference frame 0000
    (identity warp).  The original never saved it, so frame 0 (the delta F/F0
    baseline) was silently dropped everywhere downstream.
3.  Fully non-interactive: background regions and ROI are chosen
    automatically so the whole thing runs headless / start-to-finish.
4.  HEAVY-CONTRAST registration overlay: frame-0 (reference) is drawn in
    GREEN, the moving/registered frame in MAGENTA, each percentile-stretched.
    Perfectly aligned pixels -> white/grey; misalignment -> green/magenta
    ghosting.  Produced BEFORE vs AFTER registration so the correction is
    visually obvious.

Run:
    python run_pipeline_fixed.py
"""

import os, csv, cv2, warnings
import numpy as np
import pandas as pd
import tifffile
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import skimage.io as io
from skimage import exposure
from skimage.measure import regionprops, label
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ CONFIG
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT          = os.path.join(HERE, "lat_trial1_afterdrug")
membrane_folder    = os.path.join(DATA_ROOT, "membrane")
calcium_folder     = os.path.join(DATA_ROOT, "ca2")
membrane_base_name = "AVG_C2-latA_trial1_afterdrug"
calcium_base_name  = "AVG_C1-latA_trial1_afterdrug"
output_directory   = DATA_ROOT + "_output"

# auto-detect number of frames
_frames = [f for f in os.listdir(membrane_folder)
           if f.startswith(membrane_base_name) and f.endswith(".tif")]
num_frames = len(_frames)
print(f"Detected {num_frames} membrane frames in {membrane_folder}")

os.makedirs(output_directory, exist_ok=True)


# ------------------------------------------------------------------ helpers
def stretch(img, lo=1, hi=99):
    """Percentile contrast stretch -> uint8 (heavy contrast for display)."""
    img = img.astype(np.float32)
    p_lo, p_hi = np.percentile(img, [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1
    out = np.clip((img - p_lo) / (p_hi - p_lo), 0, 1)
    return (out * 255).astype(np.uint8)


def two_color_overlay(reference, moving):
    """frame0 (reference) -> GREEN, moving -> MAGENTA. High-contrast composite."""
    ref = stretch(reference)
    mov = stretch(moving)
    h, w = ref.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = mov          # R  (magenta = R+B)
    rgb[..., 1] = ref          # G  (reference)
    rgb[..., 2] = mov          # B
    return rgb


# ============================================================ PART 1  REGISTER
def register_ecc_rigid(base_gray, moving_gray):
    base_gray = base_gray.astype(np.float32)
    moving_gray = moving_gray.astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-6)
    try:
        _, warp = cv2.findTransformECC(base_gray, moving_gray, warp,
                                       cv2.MOTION_EUCLIDEAN, criteria)
    except cv2.error:
        print("   ! ECC failed on a frame, using identity warp")
    return warp


def part1_registration():
    print("\n=== PART 1: rigid ECC registration ===")
    raw_dir   = os.path.join(output_directory, "rigid_raw_registered")
    green_dir = os.path.join(output_directory, "rigid_green_registered")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(green_dir, exist_ok=True)

    first_mem = io.imread(os.path.join(membrane_folder, f"{membrane_base_name}0000.tif"))
    tifffile.imwrite(os.path.join(raw_dir, f"raw_registered_{membrane_base_name}0000.tif"), first_mem)

    # FIX #2: also persist the calcium reference frame 0 (identity)
    first_ca = io.imread(os.path.join(calcium_folder, f"{calcium_base_name}0000.tif"))
    tifffile.imwrite(os.path.join(green_dir, f"registered_{calcium_base_name}0000.tif"), first_ca)

    # keep frame-0 and the raw (un-registered) moving frames for the overlay
    overlay_pairs = []   # (frame_idx, moving_raw, moving_registered)

    for i in range(1, num_frames):
        mpath = os.path.join(membrane_folder, f"{membrane_base_name}{i:04d}.tif")
        if not os.path.exists(mpath):
            continue
        moving = io.imread(mpath)
        warp = register_ecc_rigid(first_mem, moving)
        registered = cv2.warpAffine(moving, warp, (first_mem.shape[1], first_mem.shape[0]),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        tifffile.imwrite(os.path.join(raw_dir, f"raw_registered_{membrane_base_name}{i:04d}.tif"),
                         registered.astype(np.uint16))

        gpath = os.path.join(calcium_folder, f"{calcium_base_name}{i:04d}.tif")
        if os.path.exists(gpath):
            green = io.imread(gpath)
            reg_green = cv2.warpAffine(green, warp, (green.shape[1], green.shape[0]),
                                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            tifffile.imwrite(os.path.join(green_dir, f"registered_{calcium_base_name}{i:04d}.tif"),
                             reg_green.astype(np.uint16))

        overlay_pairs.append((i, moving, registered))
        if i % 10 == 0:
            print(f"   registered frame {i}/{num_frames-1}")

    print(f"   done. membrane -> {raw_dir}")
    return first_mem, overlay_pairs


# ============================================================ HEAVY-CONTRAST OVERLAY
def part_overlay(first_mem, overlay_pairs):
    print("\n=== HEAVY-CONTRAST overlay (frame0 = GREEN, moving = MAGENTA) ===")
    frames = []
    font = ImageFont.load_default()
    for idx, moving, registered in overlay_pairs:
        before = two_color_overlay(first_mem, moving)       # frame0 vs unregistered
        after  = two_color_overlay(first_mem, registered)   # frame0 vs registered
        gap = np.full((before.shape[0], 6, 3), 40, np.uint8)
        combo = np.hstack([before, gap, after])
        pim = Image.fromarray(combo)
        d = ImageDraw.Draw(pim)
        d.text((10, 8),  f"BEFORE reg  frame {idx}", fill="white", font=font)
        d.text((before.shape[1] + 16, 8), f"AFTER reg  frame {idx}", fill="white", font=font)
        d.text((10, before.shape[0] - 16), "green=frame0  magenta=moving  white=aligned",
               fill="yellow", font=font)
        frames.append(pim)

    gif_path = os.path.join(output_directory, "registration_overlay_contrast.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f"   GIF  -> {gif_path}")

    # representative still: the frame with the largest raw motion
    worst = max(overlay_pairs,
                key=lambda t: np.mean(np.abs(stretch(first_mem).astype(int) - stretch(t[1]).astype(int))))
    idx, moving, registered = worst
    png = np.hstack([two_color_overlay(first_mem, moving),
                     np.full((first_mem.shape[0], 6, 3), 40, np.uint8),
                     two_color_overlay(first_mem, registered)])
    png_path = os.path.join(output_directory, f"registration_overlay_frame{idx}.png")
    Image.fromarray(png).save(png_path)
    print(f"   still (largest-motion frame {idx}) -> {png_path}")
    return gif_path, png_path


# ============================================================ PART 2  SEGMENT
def part2_segment():
    print("\n=== PART 2: Cellpose segmentation (cellpose 4.x / cpsam) ===")
    from cellpose import models
    seg_dir = os.path.join(output_directory, "cell_segmentation_work")
    os.makedirs(seg_dir, exist_ok=True)

    raw_dir = os.path.join(output_directory, "rigid_raw_registered")
    img = io.imread(os.path.join(raw_dir, f"raw_registered_{membrane_base_name}0000.tif"))
    img8 = stretch(img)   # cpsam likes normalized single-channel input

    try:
        import torch; _gpu = torch.cuda.is_available()
    except Exception:
        _gpu = False
    model = models.CellposeModel(gpu=_gpu)          # FIX #1: cpsam; CUDA if available, else CPU
    res = model.eval(img8, diameter=15, flow_threshold=0.4, cellprob_threshold=0.0)
    masks = res[0]                                   # cellpose4 returns (masks, flows, styles)

    n = int(masks.max())
    print(f"   segmented {n} cells")
    mask_path = os.path.join(seg_dir, "mask_0.tiff")
    tifffile.imwrite(mask_path, masks.astype(np.uint16))

    # numbered mask overlay (high contrast)
    from skimage.segmentation import find_boundaries
    disp = np.dstack([img8] * 3)
    disp[find_boundaries(masks, mode="outer")] = [255, 0, 0]
    pim = Image.fromarray(disp)
    d = ImageDraw.Draw(pim)
    font = ImageFont.load_default()
    for p in regionprops(masks):
        y, x = p.centroid
        d.text((x - 3, y - 4), str(p.label), fill="yellow", font=font)
    numbered = os.path.join(seg_dir, "mask_numbered.png")
    pim.save(numbered)
    print(f"   mask -> {mask_path}\n   numbered overlay -> {numbered}")
    return mask_path, numbered, n


# ============================================================ PART 3  BACKGROUND
def auto_background_boxes(img, k=3, box=40):
    """Pick k darkest non-overlapping boxes as background reference regions."""
    h, w = img.shape
    step = box
    cands = []
    for y in range(0, h - box, step):
        for x in range(0, w - box, step):
            cands.append((np.median(img[y:y+box, x:x+box]), x, y))
    cands.sort()
    return [(x, y, x + box, y + box) for _, x, y in cands[:k]]


def remove_outliers(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    return data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)]


def part3_background():
    print("\n=== PART 3: background subtraction (auto background regions) ===")
    red_in  = os.path.join(output_directory, "rigid_raw_registered")
    green_in = os.path.join(output_directory, "rigid_green_registered")
    red_out  = os.path.join(output_directory, "Bkrm_membrane")
    green_out = os.path.join(output_directory, "Bkrm_ca2")
    os.makedirs(red_out, exist_ok=True)
    os.makedirs(green_out, exist_ok=True)

    first_red = tifffile.imread(os.path.join(red_in, f"raw_registered_{membrane_base_name}0000.tif"))
    boxes = auto_background_boxes(first_red)
    print(f"   background boxes: {boxes}")

    for frame in range(num_frames):
        fs = f"{frame:04d}"
        rp = os.path.join(red_in,  f"raw_registered_{membrane_base_name}{fs}.tif")
        gp = os.path.join(green_in, f"registered_{calcium_base_name}{fs}.tif")
        if not (os.path.exists(rp) and os.path.exists(gp)):
            continue
        red = tifffile.imread(rp).astype(np.uint16)
        green = tifffile.imread(gp).astype(np.uint16)
        br, bg = [], []
        for (x1, y1, x2, y2) in boxes:
            br.append(np.median(remove_outliers(red[y1:y2, x1:x2].flatten())))
            bg.append(np.median(remove_outliers(green[y1:y2, x1:x2].flatten())))
        red_c   = np.clip(red   - np.median(br), 0, 65535).astype(np.uint16)
        green_c = np.clip(green - np.median(bg), 0, 65535).astype(np.uint16)
        tifffile.imwrite(os.path.join(red_out,   f"processed_red_{fs}.tif"),   red_c)
        tifffile.imwrite(os.path.join(green_out, f"processed_green_{fs}.tif"), green_c)
    print(f"   calcium -> {green_out}")


# ============================================================ PART 4  Ca ANALYSIS
def part4_calcium(mask_path):
    print("\n=== PART 4: per-cell calcium extraction + dF/F0 ===")
    ca_dir = os.path.join(output_directory, "Bkrm_ca2")
    data_dir = os.path.join(output_directory, "calcium_analysis", "data")
    os.makedirs(data_dir, exist_ok=True)

    mask = tifffile.imread(mask_path)
    cell_ids = np.unique(mask); cell_ids = cell_ids[cell_ids > 0]

    traces = {c: [] for c in cell_ids}
    for frame in range(num_frames):
        p = os.path.join(ca_dir, f"processed_green_{frame:04d}.tif")
        if not os.path.exists(p):
            for c in cell_ids: traces[c].append(np.nan)
            continue
        ca = tifffile.imread(p)
        for c in cell_ids:
            px = ca[mask == c]
            traces[c].append(float(px.mean()) if px.size else np.nan)

    # baseline = 10 lowest-activity frames (population)
    frame_avg = [np.nanmean([traces[c][f] for c in cell_ids]) for f in range(num_frames)]
    baseline_frames = sorted(sorted(range(num_frames), key=lambda f: frame_avg[f])[:10])
    print(f"   baseline frames: {baseline_frames}")

    norm = {}
    for c in cell_ids:
        f0 = np.nanmean([traces[c][f] for f in baseline_frames])
        f0 = f0 if f0 > 0 else 1
        norm[c] = [(v - f0) / f0 if not np.isnan(v) else np.nan for v in traces[c]]

    raw_df = pd.DataFrame({f"Cell_{c}": traces[c] for c in cell_ids}); raw_df.insert(0, "Frame", range(num_frames))
    norm_df = pd.DataFrame({f"Cell_{c}": norm[c] for c in cell_ids}); norm_df.insert(0, "Frame", range(num_frames))
    raw_df.to_csv(os.path.join(data_dir, "all_cells_raw.csv"), index=False)
    norm_df.to_csv(os.path.join(data_dir, "all_cells_normalized.csv"), index=False)
    # ROI export = all cells (headless: whole field)
    norm_df.to_csv(os.path.join(data_dir, "roi_cells_normalized.csv"), index=False)
    print(f"   wrote all_cells_raw.csv / all_cells_normalized.csv -> {data_dir}")
    return norm_df, baseline_frames


# ============================================================ PART 5  PEAK STATS
def part5_stats(norm_df):
    print("\n=== PART 5: peak statistics ===")
    cells = [c for c in norm_df.columns if c.startswith("Cell_")]
    N = norm_df.shape[0]
    time = np.linspace(0, 90.0, N)
    threshold = 0.5   # dF/F0 threshold for a genuine transient
    all_amp, all_intervals, all_areas, n_peaks_per_cell = [], [], [], []
    active_cells = 0
    for c in cells:
        sig = norm_df[c].values.astype(float)
        sig = np.nan_to_num(sig, nan=0.0)
        pk, _ = find_peaks(sig, height=threshold, distance=2)
        n_peaks_per_cell.append(len(pk))
        if len(pk):
            active_cells += 1
            all_amp.extend(sig[pk])
            if len(pk) > 1:
                all_intervals.extend(np.diff(time[pk]))
            for idx in pk:
                l = idx
                while l > 0 and sig[l] > threshold: l -= 1
                r = idx
                while r < len(sig) - 1 and sig[r] > threshold: r += 1
                all_areas.append(np.trapezoid(np.clip(sig[l:r+1] - threshold, 0, None), time[l:r+1]))

    def s(a):
        a = np.asarray(a, float)
        return (np.mean(a), np.std(a), np.median(a)) if a.size else (np.nan, np.nan, np.nan)

    stats = {
        "n_cells": len(cells),
        "active_cells": active_cells,
        "pct_active": 100.0 * active_cells / max(1, len(cells)),
        "total_peaks": int(np.sum(n_peaks_per_cell)),
        "peaks_per_cell_mean": float(np.mean(n_peaks_per_cell)),
        "amp_mean_std_med": s(all_amp),
        "interval_mean_std_med_s": s(all_intervals),
        "area_mean_std_med": s(all_areas),
    }
    return stats


# ============================================================ MAIN
if __name__ == "__main__":
    first_mem, overlay_pairs = part1_registration()
    gif_path, png_path = part_overlay(first_mem, overlay_pairs)
    mask_path, numbered, n_cells = part2_segment()
    part3_background()
    norm_df, baseline_frames = part4_calcium(mask_path)
    stats = part5_stats(norm_df)

    print("\n" + "=" * 60)
    print("REAL RESULTS  (lat_trial1_afterdrug)")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:24s}: {v}")
    print("\nOutputs in:", output_directory)
