"""
cafin_core.py  --  compute engine for the CAFIN calcium GUI.

Everything the Streamlit app needs, with NO GUI code, so it can be unit-run.

Registration methods:  "rigid" (OpenCV ECC), "opticalflow" (skimage TV-L1),
                       "elastic" (SimpleITK B-spline), "none".
Cell handling:         "static"  (one frame-0 Cellpose mask, reused)
                       "tracking"(the frame-0 mask is warped into every frame
                                  so each cell is followed as the tissue deforms)

The 2x2 grid (rigid/elastic x static/tracking) is the manager's "4 options";
opticalflow / none are extra modes.
"""
import os, cv2, warnings
import numpy as np
import pandas as pd
import tifffile
import skimage.io as skio
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from scipy.signal import find_peaks
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- basics
def count_frames(folder, base):
    return len([f for f in os.listdir(folder) if f.startswith(base) and f.endswith(".tif")])


def stretch8(img, lo=1, hi=99, clahe=False):
    """Percentile-stretch to 8-bit. If clahe=True, also apply CLAHE (local adaptive
    contrast) so faint membrane boundaries stay visible in the overlay."""
    img = img.astype(np.float32)
    p_lo, p_hi = np.percentile(img, [lo, hi])
    p_hi = p_hi if p_hi > p_lo else p_lo + 1
    out = (np.clip((img - p_lo) / (p_hi - p_lo), 0, 1) * 255).astype(np.uint8)
    if clahe:
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        out = cl.apply(out)
    return out


def two_color_overlay(reference, moving, clahe=True, gate_bg=True):
    """reference -> GREEN, moving -> MAGENTA; white/grey = aligned (heavy contrast).
    CLAHE is on by default so misalignment (green/magenta fringing) is easy to see.
    gate_bg keeps the background black (computed from the RAW images, before CLAHE)
    so amplified background noise does not appear as a magenta wash."""
    ref = stretch8(reference, clahe=clahe); mov = stretch8(moving, clahe=clahe)
    rgb = np.zeros((*ref.shape, 3), dtype=np.uint8)
    rgb[..., 0] = mov; rgb[..., 1] = ref; rgb[..., 2] = mov
    if gate_bg:
        # These membrane images sit on a background *pedestal* (~250 counts, not 0),
        # and warping adds true-zero border pixels — so a fixed threshold fails.
        # Otsu on the valid (non-zero) pixels separates tissue from pedestal robustly.
        from skimage.filters import threshold_otsu

        def _fg(img):
            s = stretch8(img)                      # 8-bit, no CLAHE
            valid = img > 0                        # ignore warp-fill border
            vals = s[valid]
            try:
                t = threshold_otsu(vals) if vals.size else 255
            except Exception:
                t = 40
            return (s > t) & valid

        fg = _fg(reference) | _fg(moving)
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
        rgb[~fg] = 0
    return rgb


# ============================================================ REGISTRATION
def _ecc(fixed, moving, iters=1500):
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
    try:
        _, warp = cv2.findTransformECC(fixed.astype(np.float32), moving.astype(np.float32),
                                       warp, cv2.MOTION_EUCLIDEAN, crit)
    except cv2.error:
        pass
    return warp


def _warp_affine(img, M, shape, nearest=False):
    flags = (cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR) + cv2.WARP_INVERSE_MAP
    return cv2.warpAffine(img, M, (shape[1], shape[0]), flags=flags)


def _flow(fixed, moving):
    """skimage TV-L1 optical flow; returns (v,u) that warps `moving` -> `fixed`."""
    from skimage.registration import optical_flow_tvl1
    return optical_flow_tvl1(stretch8(fixed).astype(np.float32),
                             stretch8(moving).astype(np.float32),
                             attachment=8, num_warp=3)


def _flow_warp(img, flow, nearest=False):
    from skimage.transform import warp as skwarp
    v, u = flow
    h, w = img.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.array([rr + v, cc + u])
    order = 0 if nearest else 1
    out = skwarp(img.astype(np.float32), coords, order=order, mode="edge", preserve_range=True)
    return out.astype(img.dtype)


def _bspline(fixed, moving, mesh=8, iters=50):
    """Native ITK B-spline registration via SimpleITK (kept for the paper's
    rigid-vs-elastic comparison figure). The GUI's 'elastic' method uses itk-elastix
    (see _elastix). Transform maps fixed->moving, i.e. resamples `moving` into `fixed`."""
    import SimpleITK as sitk
    fx = sitk.GetImageFromArray(stretch8(fixed).astype(np.float32))
    mv = sitk.GetImageFromArray(stretch8(moving).astype(np.float32))
    tx = sitk.BSplineTransformInitializer(fx, [mesh, mesh])
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.15)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iters)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(tx, inPlace=True)
    try:
        R.Execute(fx, mv)
    except Exception:
        pass
    return tx


def _bspline_apply(moving, tx, nearest=False):
    import SimpleITK as sitk
    mv = sitk.GetImageFromArray(moving.astype(np.float32))
    interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    res = sitk.Resample(mv, mv, tx, interp, 0.0)
    return sitk.GetArrayFromImage(res)


# ---- itk-elastix (the elastix toolbox) non-rigid registration --------------
# Quality presets: (MaximumNumberOfIterations, NumberOfResolutions, FinalGridSpacing px)
_ELASTIX_Q = {"fast": (200, 3, 40), "balanced": (500, 4, 32), "accurate": (1000, 4, 20)}


def _elastix(fixed, moving, quality="balanced"):
    """Non-rigid B-spline registration via itk-elastix. Returns the transform
    parameter object (apply to other channels/masks with _elastix_apply), or None
    on failure. `fixed`/`moving` are 2-D arrays; the transform maps moving->fixed."""
    import itk
    iters, nres, grid = _ELASTIX_Q.get(quality, _ELASTIX_Q["balanced"])
    fx = itk.image_from_array(stretch8(fixed).astype(np.float32))
    mv = itk.image_from_array(stretch8(moving).astype(np.float32))
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap("bspline", nres)
    pm["MaximumNumberOfIterations"] = [str(iters)]
    pm["FinalGridSpacingInPhysicalUnits"] = [str(grid)]
    po.AddParameterMap(pm)
    try:
        _, tp = itk.elastix_registration_method(fx, mv, parameter_object=po,
                                                log_to_console=False)
        return tp
    except Exception:
        return None


def _elastix_apply(moving, tp, nearest=False):
    """Apply an elastix transform (from _elastix) to another image via transformix."""
    import itk
    if tp is None:
        return moving
    if nearest:                               # order-0 interpolation for label masks
        for j in range(tp.GetNumberOfParameterMaps()):
            tp.SetParameter(j, "FinalBSplineInterpolationOrder", "0")
    mv = itk.image_from_array(moving.astype(np.float32))
    res = itk.transformix_filter(mv, tp)
    return np.asarray(res)


def register_series(memf, caf, mem_base, ca_base, num_frames, method,
                    do_tracking=False, mask0=None, frame_step=1, progress=None,
                    elastic_quality="balanced"):
    """
    Returns dict:
      raw_mem[i], reg_mem[i], reg_ca[i]  (aligned to frame0)
      raw_ca[i]                           (original calcium)
      mask_per_frame[i]                   (only if do_tracking; frame-i-space mask)
      frames                              (list of frame indices actually processed)
    """
    frames = list(range(0, num_frames, frame_step))
    if 0 not in frames:
        frames = [0] + frames

    def rd(folder, base, i):
        p = os.path.join(folder, f"{base}{i:04d}.tif")
        return skio.imread(p) if os.path.exists(p) else None

    fixed_mem = rd(memf, mem_base, 0)
    fixed_ca = rd(caf, ca_base, 0)
    shape = fixed_mem.shape

    out = {"raw_mem": {}, "reg_mem": {}, "reg_ca": {}, "raw_ca": {},
           "mask_per_frame": {}, "frames": frames, "shape": shape}
    out["raw_mem"][0] = fixed_mem; out["reg_mem"][0] = fixed_mem
    out["raw_ca"][0] = fixed_ca;  out["reg_ca"][0] = fixed_ca
    if do_tracking and mask0 is not None:
        out["mask_per_frame"][0] = mask0

    for k, i in enumerate([f for f in frames if f != 0]):
        mv_mem = rd(memf, mem_base, i)
        mv_ca = rd(caf, ca_base, i)
        if mv_mem is None or mv_ca is None:
            continue
        out["raw_mem"][i] = mv_mem; out["raw_ca"][i] = mv_ca

        if method == "none":
            out["reg_mem"][i] = mv_mem; out["reg_ca"][i] = mv_ca
            if do_tracking and mask0 is not None:
                out["mask_per_frame"][i] = mask0

        elif method == "rigid":
            M = _ecc(fixed_mem, mv_mem)
            out["reg_mem"][i] = _warp_affine(mv_mem, M, shape)
            out["reg_ca"][i] = _warp_affine(mv_ca, M, shape)
            if do_tracking and mask0 is not None:
                M2 = _ecc(mv_mem, fixed_mem)                 # reverse pass
                out["mask_per_frame"][i] = _warp_affine(mask0, M2, shape, nearest=True)

        elif method == "opticalflow":
            fl = _flow(fixed_mem, mv_mem)
            out["reg_mem"][i] = _flow_warp(mv_mem, fl)
            out["reg_ca"][i] = _flow_warp(mv_ca, fl)
            if do_tracking and mask0 is not None:
                fl2 = _flow(mv_mem, fixed_mem)               # reverse pass
                out["mask_per_frame"][i] = _flow_warp(mask0.astype(np.int32), fl2, nearest=True)

        elif method == "elastic":
            tp = _elastix(fixed_mem, mv_mem, quality=elastic_quality)   # itk-elastix B-spline
            if tp is None:                                              # elastix failed -> passthrough
                out["reg_mem"][i] = mv_mem; out["reg_ca"][i] = mv_ca
            else:
                out["reg_mem"][i] = _elastix_apply(mv_mem, tp).astype(mv_mem.dtype)
                out["reg_ca"][i] = _elastix_apply(mv_ca, tp).astype(mv_ca.dtype)
            if do_tracking and mask0 is not None:
                tp2 = _elastix(mv_mem, fixed_mem, quality=elastic_quality)   # reverse pass
                if tp2 is None:
                    out["mask_per_frame"][i] = mask0
                else:
                    m = _elastix_apply(mask0.astype(np.float32), tp2, nearest=True)
                    out["mask_per_frame"][i] = np.rint(m).astype(np.int32)
        else:
            raise ValueError(f"unknown method {method}")

        if progress:
            progress((k + 1) / max(1, len(frames) - 1), f"Registered frame {i}")
    return out


# ============================================================ SEGMENTATION
def cuda_status():
    """Returns (available: bool, message: str). GPU acceleration applies to the
    Cellpose (torch) segmentation step; the registration backends are CPU."""
    try:
        import torch
    except Exception as e:
        return False, f"torch not importable ({e})"
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA device"
        return True, f"CUDA available — {name} (torch {torch.__version__})"
    build = "CPU-only build" if "+cpu" in torch.__version__ else "no CUDA device"
    return False, (f"CUDA not available ({build}, torch {torch.__version__}). "
                   "Install a CUDA build, e.g. "
                   "`pip install torch --index-url https://download.pytorch.org/whl/cu124`")


def segment(reg_mem0, diameter=15, gpu=False, model_type="cyto3"):
    """Segment cells on the reference membrane frame.

    The manuscript specifies the Cellpose `cyto3` model with channels=[2,0]. That
    API exists in Cellpose 3.x. Cellpose 4.x replaced it with the single `cpsam`
    model and dropped the `channels`/`model_type` arguments. This function honors
    the manuscript settings when the installed version supports them and otherwise
    falls back to the 4.x default, so segmentation runs on either version.
    """
    from cellpose import models
    use_gpu = bool(gpu) and cuda_status()[0]     # only use GPU if actually available
    img8 = stretch8(reg_mem0, clahe=True)        # local contrast helps boundary detection
    # Try the manuscript's Cellpose 3.x API (cyto3 + channels=[2,0]); fall back to 4.x.
    try:
        model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
        masks = model.eval(img8, diameter=diameter, channels=[2, 0],
                           flow_threshold=0.4, cellprob_threshold=0.0)[0]
    except TypeError:
        model = models.CellposeModel(gpu=use_gpu)   # Cellpose 4.x (cpsam)
        masks = model.eval(img8, diameter=diameter,
                           flow_threshold=0.4, cellprob_threshold=0.0)[0]
    return np.asarray(masks).astype(np.int32), use_gpu


def _clean_mask(mask, min_area=60, fill_holes=True):
    """Remove tiny objects, optionally fill holes, and re-label 1..N (light version
    of the wound pipeline's clean step, used before per-frame tracking)."""
    from skimage.morphology import remove_small_objects
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import label as sklabel
    m = np.asarray(mask).astype(np.int32)
    if min_area > 0:
        m = remove_small_objects(m, min_size=int(min_area))
    if fill_holes:
        filled = binary_fill_holes(m > 0)
        m[(filled) & (m == 0)] = 0   # keep labels; holes inside a cell stay that cell
    return sklabel(m > 0).astype(np.int32) if m.max() == 0 else m


def segment_stack(images, diameter=15, gpu=False, model_type="cyto3",
                  min_area=60, progress=None):
    """Segment EVERY frame independently (needed for true cell tracking).
    `images` is a list/iterable of 2-D membrane frames. Builds the Cellpose model
    once and evaluates each frame. Returns (T,H,W) int array of cleaned masks."""
    from cellpose import models
    use_gpu = bool(gpu) and cuda_status()[0]
    imgs = [stretch8(im, clahe=True) for im in images]
    try:
        model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
        use_ch = True
    except TypeError:
        model = models.CellposeModel(gpu=use_gpu)
        use_ch = False
    out = []
    n = len(imgs)
    for k, im in enumerate(imgs):
        kw = dict(diameter=diameter, flow_threshold=0.4, cellprob_threshold=0.0)
        if use_ch:
            kw["channels"] = [2, 0]
        m = model.eval(im, **kw)[0]
        out.append(_clean_mask(np.asarray(m).astype(np.int32), min_area=min_area))
        if progress:
            progress((k + 1) / max(1, n), f"Segmenting frame {k + 1}/{n}")
    h, w = out[0].shape
    return np.stack(out).astype(np.int32)


def numbered_mask_overlay(reg_mem0, mask):
    from PIL import Image, ImageDraw, ImageFont
    disp = np.dstack([stretch8(reg_mem0)] * 3)
    disp[find_boundaries(mask, mode="outer")] = [255, 0, 0]
    im = Image.fromarray(disp); d = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("arial.ttf", 9)
    except Exception: font = ImageFont.load_default()
    for p in regionprops(mask):
        y, x = p.centroid
        d.text((x - 3, y - 4), str(p.label), fill="yellow", font=font)
    return np.asarray(im)


# ============================================================ BACKGROUND
def auto_bg_boxes(img, k=3, box=40):
    h, w = img.shape
    c = [(np.median(img[y:y+box, x:x+box]), x, y)
         for y in range(0, h - box, box) for x in range(0, w - box, box)]
    c.sort()
    return [(x, y, x + box, y + box) for _, x, y in c[:k]]


def _rm_outliers(a):
    q1, q3 = np.percentile(a, [25, 75]); iqr = q3 - q1
    return a[(a >= q1 - 1.5 * iqr) & (a <= q3 + 1.5 * iqr)]


def bg_subtract(stack, boxes):
    out = {}
    for i, img in stack.items():
        vals = [np.median(_rm_outliers(img[y1:y2, x1:x2].flatten())) for (x1, y1, x2, y2) in boxes]
        out[i] = np.clip(img.astype(np.float32) - np.mean(vals), 0, None)   # avg of the 3 medians (per manuscript)
    return out


# ============================================================ REGION OF INTEREST
def roi_cell_ids(mask, box):
    """Return the integer IDs of every cell whose mask intersects rectangular ROI
    box=(x1,y1,x2,y2). A cell is included if at least one of its pixels overlaps
    the ROI (matches the manuscript's ROI-selection rule)."""
    x1, y1, x2, y2 = box
    sub = mask[y1:y2, x1:x2]
    ids = np.unique(sub); ids = ids[ids > 0]
    return sorted(int(i) for i in ids)


# ============================================================ TRACES / dF/F0
def extract_traces(reg, mask0, cell_handling, bg=False):
    """Returns raw per-cell trace DataFrame (rows=frame index, cols=Cell_i)."""
    frames = reg["frames"]
    ids = np.unique(mask0); ids = ids[ids > 0]

    if cell_handling == "tracking":
        ca_source = {i: reg["raw_ca"][i] for i in frames if i in reg["raw_ca"]}
        masks = reg["mask_per_frame"]
    else:
        ca_source = {i: reg["reg_ca"][i] for i in frames if i in reg["reg_ca"]}
        masks = {i: mask0 for i in frames}

    if bg:
        boxes = auto_bg_boxes(next(iter(ca_source.values())))
        ca_source = bg_subtract(ca_source, boxes)

    traces = {c: [] for c in ids}
    for i in frames:
        ca = ca_source.get(i); m = masks.get(i)
        if ca is None or m is None:
            for c in ids: traces[c].append(np.nan)
            continue
        for c in ids:
            px = ca[m == c]
            traces[c].append(float(px.mean()) if px.size else np.nan)
    df = pd.DataFrame({f"Cell_{c}": traces[c] for c in ids})
    df.insert(0, "Frame", frames)
    return df


def extract_tracked_traces(mask_per_frame, ca_by_frame, frames, bg=False):
    """Per-cell calcium traces from TRUE-tracked masks (global ids across frames).
    `mask_per_frame`/`ca_by_frame` are {frame_index: 2-D array} in the SAME space.
    A cell contributes wherever its global id is present; NaN when absent."""
    ids = set()
    for f in frames:
        m = mask_per_frame.get(f)
        if m is not None:
            ids.update(int(i) for i in np.unique(m) if i > 0)
    ids = sorted(ids)
    ca_src = dict(ca_by_frame)
    if bg:
        boxes = auto_bg_boxes(next(iter(ca_src.values())))
        ca_src = bg_subtract(ca_src, boxes)
    traces = {c: [] for c in ids}
    for f in frames:
        ca = ca_src.get(f); m = mask_per_frame.get(f)
        for c in ids:
            if ca is None or m is None:
                traces[c].append(np.nan); continue
            px = ca[m == c]
            traces[c].append(float(px.mean()) if px.size else np.nan)
    df = pd.DataFrame({f"Cell_{c}": traces[c] for c in ids})
    df.insert(0, "Frame", frames)
    return df


def compute_dff0(raw_df, method="lowest", floor=1.0, n_base=10):
    cells = [c for c in raw_df.columns if c.startswith("Cell_")]
    n = raw_df.shape[0]
    favg = raw_df[cells].mean(axis=1).values
    nb = min(n_base, max(3, n // 3))
    if method == "lowest":
        base = sorted(sorted(range(n), key=lambda f: favg[f])[:nb])
    elif method == "first":
        base = list(range(min(nb, n)))
    else:
        base = list(range(max(0, n - nb), n))
    dff = raw_df.copy()
    for c in cells:
        v = raw_df[c].values.astype(float)
        f0 = max(np.nanmean(v[base]), floor)
        dff[c] = (v - f0) / f0
    return dff, base


# ============================================================ CLUSTERING
def cluster_traces(dff_df, n_pca=25, n_clusters=4, seed=0):
    """Cluster cells by the shape of their ΔF/F0 traces.
    Standardize each cell's trace -> PCA to `n_pca` features -> KMeans(`n_clusters`).
    Returns dict: ids, labels (per cell), coords (PCA), n_pca_used, explained_var.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    cells = [c for c in dff_df.columns if c.startswith("Cell_")]
    ids = np.array([int(c.split("_")[1]) for c in cells])
    X = np.nan_to_num(dff_df[cells].to_numpy(float)).T          # cells x frames
    # standardize each cell's trace (row) so clustering is by shape, not amplitude offset
    Xs = StandardScaler().fit_transform(X)
    n_comp = int(max(2, min(n_pca, Xs.shape[0] - 1, Xs.shape[1])))
    pca = PCA(n_components=n_comp, random_state=seed)
    coords = pca.fit_transform(Xs)
    k = int(max(2, min(n_clusters, Xs.shape[0])))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(coords)
    return {"ids": ids, "labels": labels, "coords": coords, "n_pca_used": n_comp,
            "explained_var": float(pca.explained_variance_ratio_.sum()), "k": k}


# ============================================================ METRICS
def metrics(dff_df, threshold=0.5):
    cells = [c for c in dff_df.columns if c.startswith("Cell_")]
    A = np.nan_to_num(dff_df[cells].to_numpy(float), nan=0.0)
    n = A.shape[0]
    ppc, amps, intervals, areas = [], [], [], []
    t = np.linspace(0, 1, n)
    for j in range(A.shape[1]):
        pk, _ = find_peaks(A[:, j], height=threshold, distance=2)
        ppc.append(len(pk))
        if len(pk):
            amps.extend(A[pk, j])
            if len(pk) > 1: intervals.extend(np.diff(pk))
            for idx in pk:
                l = idx
                while l > 0 and A[l, j] > threshold: l -= 1
                r = idx
                while r < n - 1 and A[r, j] > threshold: r += 1
                areas.append(np.trapezoid(np.clip(A[l:r+1, j] - threshold, 0, None)))
    ppc = np.array(ppc); pm = A.max(axis=0); active = pm > threshold
    het = float(np.std(pm[pm > 0]) / np.mean(pm[pm > 0])) if np.any(pm > 0) else np.nan
    act = A[:, active]
    if act.shape[1] >= 2:
        cc = np.corrcoef(act.T); iu = np.triu_indices_from(cc, k=1)
        sync = float(np.nanmean(cc[iu]))
    else:
        sync = np.nan
    return {
        "cells": int(A.shape[1]),
        "active_pct": round(100 * float(active.mean()), 1),
        "mean_peak_dff0": round(float(np.mean(amps)), 3) if amps else np.nan,
        "median_peak_dff0": round(float(np.median(amps)), 3) if amps else np.nan,
        "peaks_per_cell": round(float(ppc.mean()), 2),
        "transient_rate_per_frame": round(float(ppc.sum() / (A.shape[1] * n)), 4),
        "spatial_heterogeneity_cv": round(het, 3),
        "temporal_sync_r": round(sync, 3),
    }, {"amps": amps, "intervals": intervals, "areas": areas}
