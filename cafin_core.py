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

# NumPy 1.26, used by the stable Cellpose 3 environment, names this integration
# function ``trapz``. NumPy 2 added ``trapezoid``. Keep one spelling throughout
# CAFIN so the same calculations run in either supported environment.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

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
def gpu_status():
    """Detect a usable torch GPU backend for the Cellpose segmentation step.
    Returns (available: bool, message: str, backend: str).

    Supported backends:
      cuda      NVIDIA
      rocm      AMD on Linux (ROCm torch builds report through the CUDA API)
      directml  AMD / Intel GPUs on Windows, via `pip install torch-directml`
      mps       Apple Silicon
    Registration (OpenCV ECC, itk-elastix) is CPU-only regardless of this.
    """
    try:
        import torch
    except Exception as e:
        return False, f"torch not importable ({e})", "none"

    # CUDA covers both NVIDIA and AMD ROCm builds of torch
    try:
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "GPU"
            if getattr(torch.version, "hip", None):
                return True, f"AMD ROCm — {name} (torch {torch.__version__})", "rocm"
            return True, f"NVIDIA CUDA — {name} (torch {torch.__version__})", "cuda"
    except Exception:
        pass

    # DirectML: AMD and Intel GPUs on Windows
    try:
        import torch_directml
        if torch_directml.is_available() and torch_directml.device_count() > 0:
            try:
                name = torch_directml.device_name(0)
            except Exception:
                name = "DirectML device"
            return True, f"AMD/Intel DirectML — {name}", "directml"
    except Exception:
        pass

    # Apple Silicon
    try:
        if torch.backends.mps.is_available():
            return True, "Apple MPS GPU", "mps"
    except Exception:
        pass

    # Nothing usable: give a platform-appropriate hint
    import sys as _sys
    vendor = _gpu_vendor_hint()
    if _sys.platform.startswith("win") and vendor in ("amd", "intel", "unknown"):
        hint = ("For an AMD or Intel GPU on Windows install DirectML:\n"
                "    pip install torch-directml\n"
                "(NVIDIA instead: pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121)")
    elif vendor == "amd":
        hint = ("For an AMD GPU on Linux install a ROCm build:\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/rocm5.7")
    else:
        hint = ("Install a CUDA build:\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cu121")
    build = "CPU-only build" if "+cpu" in torch.__version__ else "no GPU backend"
    return False, f"No GPU backend ({build}, torch {torch.__version__}).\n{hint}", "none"


def _gpu_vendor_hint():
    """Best-effort GPU vendor detection ('amd' | 'nvidia' | 'intel' | 'unknown'),
    used only to print a helpful install hint."""
    try:
        import cv2
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.haveOpenCL():
            v = (cv2.ocl.Device_getDefault().vendorName() or "").lower()
            if "advanced micro" in v or "amd" in v:
                return "amd"
            if "nvidia" in v:
                return "nvidia"
            if "intel" in v:
                return "intel"
    except Exception:
        pass
    return "unknown"


def torch_device(backend=None):
    """Return a torch.device for the given backend (auto-detected if None), else None."""
    try:
        import torch
    except Exception:
        return None
    if backend is None:
        ok, _, backend = gpu_status()
        if not ok:
            return None
    if backend in ("cuda", "rocm"):
        return torch.device("cuda")
    if backend == "mps":
        return torch.device("mps")
    if backend == "directml":
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            return None
    return None


def cuda_status():
    """Backward-compatible wrapper: (available, message) for any GPU backend."""
    ok, msg, _ = gpu_status()
    return ok, msg


def build_cellpose(gpu=False, model_type="cyto3"):
    """Create a CellposeModel on the best available device.
    Returns (model, use_channels, backend). `backend` is 'cpu' when no GPU is used."""
    from cellpose import models
    dev, backend = None, "cpu"
    if gpu:
        ok, _, be = gpu_status()
        if ok:
            dev = torch_device(be)
            if dev is not None:
                backend = be
    kw = {"gpu": dev is not None}
    if dev is not None:
        kw["device"] = dev                       # CUDA / ROCm / DirectML / MPS device

    def create(args):
        try:                                     # Cellpose 3.x API (cyto3 + channels)
            return models.CellposeModel(model_type=model_type, **args), True
        except TypeError:                        # Cellpose 4.x (cpsam, no model_type/channels)
            return models.CellposeModel(**args), False

    try:
        model, use_channels = create(kw)
        return model, use_channels, backend
    except Exception:
        # A driver can report an available GPU but fail while Cellpose creates
        # its model (notably on mismatched MPS/DirectML installations). Keep
        # the GUI usable by creating the same model on CPU instead.
        if dev is None:
            raise
        model, use_channels = create({"gpu": False})
        return model, use_channels, "cpu"


def _cp_eval(model, img8, diameter, use_ch):
    kw = dict(diameter=diameter, flow_threshold=0.4, cellprob_threshold=0.0)
    if use_ch:
        kw["channels"] = [2, 0]
    return model.eval(img8, **kw)[0]


def segment(reg_mem0, diameter=15, gpu=False, model_type="cyto3"):
    """Segment cells on the reference membrane frame.

    The manuscript specifies the Cellpose `cyto3` model with channels=[2,0]. That
    API exists in Cellpose 3.x. Cellpose 4.x replaced it with the single `cpsam`
    model and dropped the `channels`/`model_type` arguments. This function honors
    the manuscript settings when the installed version supports them and otherwise
    falls back to the 4.x default, so segmentation runs on either version.
    """
    img8 = stretch8(reg_mem0, clahe=True)        # local contrast helps boundary detection
    model, use_ch, backend = build_cellpose(gpu=gpu, model_type=model_type)
    try:
        masks = _cp_eval(model, img8, diameter, use_ch)
    except Exception:                            # GPU backend failed -> redo on CPU
        model, use_ch, backend = build_cellpose(gpu=False, model_type=model_type)
        masks = _cp_eval(model, img8, diameter, use_ch)
    return np.asarray(masks).astype(np.int32), (backend != "cpu")


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
    imgs = [stretch8(im, clahe=True) for im in images]
    model, use_ch, backend = build_cellpose(gpu=gpu, model_type=model_type)
    out = []
    n = len(imgs)
    for k, im in enumerate(imgs):
        try:
            m = _cp_eval(model, im, diameter, use_ch)
        except Exception:                        # GPU backend failed -> continue on CPU
            model, use_ch, backend = build_cellpose(gpu=False, model_type=model_type)
            m = _cp_eval(model, im, diameter, use_ch)
        out.append(_clean_mask(np.asarray(m).astype(np.int32), min_area=min_area))
        if progress:
            progress((k + 1) / max(1, n), f"Segmenting frame {k + 1}/{n} ({backend})")
    return np.stack(out).astype(np.int32)


def numbered_mask_overlay(reg_mem0, mask, clahe=False, font_size=9):
    from PIL import Image, ImageDraw, ImageFont
    disp = np.dstack([stretch8(reg_mem0, clahe=clahe)] * 3)
    disp[find_boundaries(mask, mode="outer")] = [255, 0, 0]
    im = Image.fromarray(disp); d = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("arial.ttf", font_size)
    except Exception: font = ImageFont.load_default()
    for p in regionprops(mask):
        y, x = p.centroid
        d.text((x - 3, y - 4), str(p.label), fill="yellow", font=font)
    return np.asarray(im)


def cellpose_colored_mask(mask):
    """Return Cellpose's native RGB label rendering on a white background."""
    from cellpose import plot
    return plot.mask_rgb(np.asarray(mask).astype(np.int32))


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


def background_values(stack, boxes):
    """Per-frame background estimate, exactly as bg_subtract computes it:
    1.5xIQR outlier removal in each box, median per box, then the mean of the boxes.
    Returns {frame: (mean_bg, [per_box_medians])} so the regions can be checked."""
    out = {}
    for i, img in stack.items():
        vals = [float(np.median(_rm_outliers(img[y1:y2, x1:x2].flatten())))
                for (x1, y1, x2, y2) in boxes]
        out[i] = (float(np.mean(vals)), vals)
    return out


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


def f0_per_cell(raw_df, method="percell", floor=1.0, n_base=10):
    """Baseline fluorescence for every cell: {cell: (F0i, [row indices used])}.

    method
      "percell"  F0i = mean of that cell's OWN n_base lowest frames  (default)
      "min"      F0i = that cell's single lowest value
      "lowest"   one shared set of frames, picked from the population mean (legacy)
      "first"    the first n_base frames (a true pre-stimulus F0, if the recording
                 starts before the stimulus)
      "last"     the last n_base frames

    The per-cell methods give each cell its own baseline frames, so a cell that is
    quiet early and a cell that is quiet late are each normalised to their own resting
    level rather than to a window chosen from the population average.
    """
    cells = [c for c in raw_df.columns if c.startswith("Cell_")]
    n = raw_df.shape[0]
    nb = int(min(n_base, max(3, n // 3)))
    shared = None
    if method == "lowest":
        favg = raw_df[cells].mean(axis=1).values
        shared = sorted(sorted(range(n), key=lambda f: favg[f])[:nb])
    elif method == "first":
        shared = list(range(min(nb, n)))
    elif method == "last":
        shared = list(range(max(0, n - nb), n))

    out = {}
    for c in cells:
        v = raw_df[c].to_numpy(float)
        if shared is not None:
            rows = shared
        elif method == "min":
            rows = [int(np.nanargmin(v))]
        else:                                   # "percell": this cell's own lowest frames
            rows = sorted(int(i) for i in np.argsort(np.nan_to_num(v, nan=np.inf))[:nb])
        out[c] = (max(float(np.nanmean(v[rows])), floor), rows)
    return out


def compute_dff0(raw_df, method="percell", floor=1.0, n_base=10, return_f0=False):
    """Normalise each cell by its OWN baseline: ΔF/F0i = (F − F0i) / F0i.

    See f0_per_cell for the baseline methods. Returns (dff, base), where `base` is the
    shared baseline rows or None when every cell uses its own. With return_f0=True the
    per-cell {cell: (F0i, rows)} mapping is returned as a third element.
    """
    f0i = f0_per_cell(raw_df, method=method, floor=floor, n_base=n_base)
    dff = raw_df.copy()
    for c, (f0, _rows) in f0i.items():
        dff[c] = (raw_df[c].to_numpy(float) - f0) / f0
    shared = None
    if method in ("lowest", "first", "last") and f0i:
        shared = next(iter(f0i.values()))[1]
    return (dff, shared, f0i) if return_f0 else (dff, shared)


# ============================================================ PEAK FEATURES
def peak_features(dff_df, threshold=0.5, frame_interval=1.0, min_distance=2):
    """Per-cell peak-shape metrics, one row per cell.

    Columns
      n_peaks       number of detected transients
      t_first_peak  time of the first peak
      auc           area under the ΔF/F0 curve (clipped at 0)
      amplitude     mean ΔF/F0 at the detected peaks
      fwhm          mean full width at half maximum of the peaks
      dt_peak       mean interval between consecutive peaks

    `frame_interval` converts frames to real time (e.g. minutes per frame); leave at
    1.0 to report everything in frames. Cells with no detected peak get NaN for the
    peak-dependent columns, so they drop out of those panels instead of biasing them.
    """
    from scipy.signal import peak_widths
    cells = [c for c in dff_df.columns if c.startswith("Cell_")]
    rows = []
    for c in cells:
        v = np.nan_to_num(dff_df[c].to_numpy(float))
        pk, _ = find_peaks(v, height=threshold, distance=min_distance)
        if len(pk):
            amp = float(np.mean(v[pk]))
            t1 = float(pk[0] * frame_interval)
            fw = float(np.mean(peak_widths(v, pk, rel_height=0.5)[0]) * frame_interval)
            dt = float(np.mean(np.diff(pk)) * frame_interval) if len(pk) > 1 else np.nan
        else:
            amp = t1 = fw = dt = np.nan
        rows.append(dict(cell=int(c.split("_")[1]), n_peaks=int(len(pk)), t_first_peak=t1,
                         auc=float(np.trapezoid(np.clip(v, 0, None)) * frame_interval),
                         amplitude=amp, fwhm=fw, dt_peak=dt))
    return pd.DataFrame(rows)


# ============================================================ CLUSTERING
# The labels are deliberately public so the GUI, saved tables, and any scripts
# use the same biologically readable names for cluster inputs.
CELL_CLUSTER_FEATURES = {
    "n_peaks": "Number of detected peaks",
    "t_first_peak": "Time to first peak",
    "auc": "Area under the trace",
    "amplitude": "Mean peak amplitude",
    "fwhm": "Mean peak width (FWHM)",
    "dt_peak": "Mean time between peaks",
    "active_frame_fraction": "Active-frame fraction",
    "mean_dff0": "Mean ΔF/F0i",
    "max_dff0": "Maximum ΔF/F0i",
    "t_max_dff0": "Time to maximum ΔF/F0i",
    "tissue_mean_correlation": "Correlation with tissue-mean trace",
}

TISSUE_CLUSTER_FEATURES = {
    "tissue_mean_dff0": "Tissue mean ΔF/F0i",
    "active_cell_fraction": "Active-cell fraction",
    "tissue_median_dff0": "Tissue median ΔF/F0i",
    "cell_to_cell_sd": "Cell-to-cell signal spread (SD)",
}


def cell_clustering_features(dff_df, threshold=0.5, frame_interval=1.0,
                             min_distance=2):
    """Return one clustering-ready feature row per cell.

    Peak-dependent values remain NaN when a cell has no detected peak. The
    clustering function handles those values explicitly: no first peak is
    placed at the end of the recording and the other missing peak-shape values
    are median-imputed after a separate ``n_peaks`` feature records silence.
    """
    cells = [c for c in dff_df.columns if c.startswith("Cell_")]
    ids = np.array([int(c.split("_")[1]) for c in cells])
    A = np.nan_to_num(dff_df[cells].to_numpy(float), nan=0.0)
    n_frames = len(A)
    feats = peak_features(dff_df, threshold=threshold, frame_interval=frame_interval,
                          min_distance=min_distance).set_index("cell")
    table = feats.reindex(ids).reset_index().rename(columns={"cell": "cell_id"})
    # A non-detected first peak means that the cell has not initiated within the
    # recording. Treat it as after the last sampled frame rather than falsely
    # treating it as frame zero.
    end_time = float(max(0, n_frames) * frame_interval)
    table["t_first_peak"] = table["t_first_peak"].fillna(end_time)
    table["has_detected_peak"] = table["n_peaks"].gt(0)
    table["active_frame_fraction"] = (A > threshold).mean(axis=0) if n_frames else 0.0
    table["mean_dff0"] = A.mean(axis=0) if n_frames else 0.0
    table["max_dff0"] = A.max(axis=0) if n_frames else 0.0
    table["t_max_dff0"] = (A.argmax(axis=0) * frame_interval) if n_frames else 0.0

    tissue_mean = A.mean(axis=1) if n_frames else np.array([])
    correlations = []
    for j in range(len(ids)):
        if n_frames < 2 or np.std(A[:, j]) == 0 or np.std(tissue_mean) == 0:
            correlations.append(np.nan)
        else:
            correlations.append(float(np.corrcoef(A[:, j], tissue_mean)[0, 1]))
    table["tissue_mean_correlation"] = correlations
    return table


def tissue_clustering_features(dff_df, threshold=0.5):
    """Return one tissue-state feature row per frame for frame clustering."""
    cells = [c for c in dff_df.columns if c.startswith("Cell_")]
    frames = (dff_df["Frame"].to_numpy() if "Frame" in dff_df.columns
              else np.arange(len(dff_df)))
    A = np.nan_to_num(dff_df[cells].to_numpy(float), nan=0.0)
    if A.shape[1] == 0:
        raise ValueError("No cell traces are available for tissue-state clustering.")
    return pd.DataFrame({
        "Frame": frames,
        "tissue_mean_dff0": A.mean(axis=1),
        "active_cell_fraction": (A > threshold).mean(axis=1),
        "tissue_median_dff0": np.median(A, axis=1),
        "cell_to_cell_sd": A.std(axis=1),
    })


def _impute_scale_group(values, names):
    """Impute a feature family, z-score columns, then balance family weight.

    Whole traces may contain hundreds of time points while first-peak time is
    one value. Dividing a family's z-scored matrix by sqrt(number of columns)
    prevents the trace family from automatically outweighing every selected
    biological feature merely because it has more columns.
    """
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(values, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[1] == 0:
        return None, []
    finite = np.isfinite(X)
    medians = np.nanmedian(np.where(finite, X, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    X = np.where(finite, X, medians[None, :])
    varying = np.nanstd(X, axis=0) > 1e-12
    if not np.any(varying):
        return None, []
    X = X[:, varying]
    kept_names = [name for name, keep in zip(names, varying) if keep]
    X = StandardScaler().fit_transform(X)
    return X / np.sqrt(X.shape[1]), kept_names


def _cluster_feature_groups(ids, groups, n_pca=25, n_clusters=4, seed=0):
    """PCA + K-means for named, equally weighted feature families."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    ids = np.asarray(ids)
    if len(ids) < 2:
        raise ValueError("Need at least two observations to cluster.")
    matrices, feature_names, group_names, dropped_groups = [], [], [], []
    for group_name, values, names in groups:
        scaled, kept = _impute_scale_group(values, names)
        if scaled is not None:
            matrices.append(scaled)
            feature_names.extend(kept)
            group_names.append(group_name)
        else:
            dropped_groups.append(group_name)
    if not matrices:
        raise ValueError("The selected clustering features do not vary across observations.")
    X = np.column_stack(matrices)
    n_comp = int(min(max(1, n_pca), X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_comp, random_state=seed)
    scores = pca.fit_transform(X)
    # Avoid empty K-means clusters when only a few feature rows are distinct.
    distinct = np.unique(np.round(scores, decimals=12), axis=0).shape[0]
    k = int(min(max(2, n_clusters), len(ids), distinct))
    if k < 2:
        raise ValueError("The selected clustering features contain only one unique state.")
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(scores)
    coords = np.zeros((len(ids), 2), dtype=float)
    coords[:, :min(2, scores.shape[1])] = scores[:, :min(2, scores.shape[1])]
    return {
        "ids": ids,
        "labels": labels,
        "coords": coords,
        "scores": scores,
        "n_pca_used": n_comp,
        "explained_var": float(pca.explained_variance_ratio_.sum()),
        "k": k,
        "feature_names": feature_names,
        "feature_groups": group_names,
        "dropped_feature_groups": dropped_groups,
    }


def cluster_cells(dff_df, include_trace=True, selected_features=None, threshold=0.5,
                  frame_interval=1.0, n_pca=25, n_clusters=4, seed=0):
    """Cluster cells using checked trace, peak, activity, and tissue-coupling inputs.

    ``include_trace`` keeps the original whole-trace PCA workflow. Entries in
    ``selected_features`` must be keys in :data:`CELL_CLUSTER_FEATURES`.
    """
    selected_features = list(selected_features or [])
    unknown = set(selected_features) - set(CELL_CLUSTER_FEATURES)
    if unknown:
        raise ValueError(f"Unknown cell clustering features: {sorted(unknown)}")
    cells = [c for c in dff_df.columns if c.startswith("Cell_")]
    ids = np.array([int(c.split("_")[1]) for c in cells])
    if not include_trace and not selected_features:
        raise ValueError("Select the whole trace or at least one biological feature.")
    table = cell_clustering_features(dff_df, threshold=threshold,
                                     frame_interval=frame_interval)
    groups = []
    if include_trace:
        trace = np.nan_to_num(dff_df[cells].to_numpy(float), nan=0.0).T
        frame_ids = (dff_df["Frame"].to_numpy() if "Frame" in dff_df.columns
                     else np.arange(trace.shape[1]))
        groups.append(("Whole ΔF/F0i trace", trace,
                       [f"trace_frame_{frame}" for frame in frame_ids]))
    for key in selected_features:
        groups.append((CELL_CLUSTER_FEATURES[key], table[[key]].to_numpy(float), [key]))
    out = _cluster_feature_groups(ids, groups, n_pca=n_pca, n_clusters=n_clusters, seed=seed)
    out.update({"feature_table": table, "include_trace": bool(include_trace),
                "selected_features": selected_features, "mode": "cells"})
    return out


def cluster_tissue_states(dff_df, selected_features=None, threshold=0.5,
                          n_pca=25, n_clusters=4, seed=0):
    """Cluster frames into tissue-level activity states using checked features."""
    selected_features = list(selected_features or [])
    unknown = set(selected_features) - set(TISSUE_CLUSTER_FEATURES)
    if unknown:
        raise ValueError(f"Unknown tissue clustering features: {sorted(unknown)}")
    if not selected_features:
        raise ValueError("Select at least one tissue-level feature.")
    table = tissue_clustering_features(dff_df, threshold=threshold)
    groups = [(TISSUE_CLUSTER_FEATURES[key], table[[key]].to_numpy(float), [key])
              for key in selected_features]
    out = _cluster_feature_groups(table["Frame"].to_numpy(), groups, n_pca=n_pca,
                                  n_clusters=n_clusters, seed=seed)
    out.update({"feature_table": table, "selected_features": selected_features,
                "include_trace": False, "mode": "tissue_states"})
    return out


def cluster_traces(dff_df, n_pca=25, n_clusters=4, seed=0):
    """Backward-compatible whole-trace cell clustering wrapper."""
    return cluster_cells(dff_df, include_trace=True, selected_features=[], n_pca=n_pca,
                         n_clusters=n_clusters, seed=seed)


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


# ============================================================ NETWORK ANALYSIS
def analyze_pixel_network_legacy(
    reg_or_ca,
    mask0=None,
    frames=None,
    bg_boxes=None,
    do_bg=False,
    roi_box=None,
    n_samples=250,
    seed=0,
    tissue_r_thresh=0.30,
    r2_thresh=0.70,
    positive_edges_only=False,
    k_clique=6,
    restrict_to_mask=True,
    dataset_name="",
):
    """Pixel-level correlation network analysis with k-clique community detection.

    Based on FocalPlane NetworkX calcium workflow:
    1. Extracts pixel traces from background-corrected registered calcium frames.
    2. Restricts sampled pixels to valid tissue mask (and ROI box if provided).
    3. Samples pixels randomly with reproducible seed (default 250, max 1000).
    4. Computes correlation of each pixel with tissue-average trace; filters by threshold.
    5. Computes pairwise Pearson correlation and R² edge matrix between retained pixels.
    6. Builds NetworkX graph with multi-factor safety preflight checks.
    7. Detects k-clique percolation communities and tracks overlapping node memberships.

    Returns dict with nodes_df, edges_df, summary_df, graph, and network metrics.
    """
    import networkx as nx
    from networkx.algorithms.community import k_clique_communities

    # 1. Resolve calcium frames source
    if isinstance(reg_or_ca, dict) and "reg_ca" in reg_or_ca:
        frames_list = frames or reg_or_ca.get("frames", sorted(reg_or_ca["reg_ca"].keys()))
        ca_source = {i: reg_or_ca["reg_ca"][i] for i in frames_list if i in reg_or_ca["reg_ca"]}
    elif isinstance(reg_or_ca, dict):
        ca_source = dict(reg_or_ca)
        frames_list = frames or sorted(ca_source.keys())
    elif isinstance(reg_or_ca, np.ndarray):
        if reg_or_ca.ndim == 3:
            frames_list = frames or list(range(len(reg_or_ca)))
            ca_source = {i: reg_or_ca[i] for i in range(len(reg_or_ca))}
        else:
            return {"error": "Calcium array must be 3D (T, H, W).", "safety": False}
    else:
        return {"error": "Invalid calcium data provided.", "safety": False}

    if not ca_source or not frames_list:
        return {"error": "No calcium frames available for network analysis.", "safety": False}

    # 2. Background correction if enabled
    if do_bg or bg_boxes is not None:
        first_img = next((img for img in ca_source.values() if img is not None), None)
        if first_img is not None:
            if bg_boxes is None:
                bg_boxes = auto_bg_boxes(first_img)
            ca_source = bg_subtract(ca_source, bg_boxes)

    # 3. Stack frames
    valid_frames = [f for f in frames_list if f in ca_source and ca_source[f] is not None]
    if len(valid_frames) < 2:
        return {"error": "Need at least 2 frames for correlation network analysis.", "safety": False}

    stack = np.stack([ca_source[f].astype(np.float32) for f in valid_frames], axis=0)
    T, H, W = stack.shape

    # 4. Determine valid pixel mask
    if mask0 is not None:
        m0 = np.asarray(mask0)
        if m0.shape != (H, W):
            m0 = cv2.resize(m0, (W, H), interpolation=cv2.INTER_NEAREST)
        valid = (m0 > 0) if restrict_to_mask else np.ones((H, W), dtype=bool)
    else:
        valid = np.ones((H, W), dtype=bool)

    if roi_box is not None:
        x1, y1, x2, y2 = [int(v) for v in roi_box]
        x1, x2 = sorted((max(0, x1), min(W, x2)))
        y1, y2 = sorted((max(0, y1), min(H, y2)))
        roi_mask = np.zeros((H, W), dtype=bool)
        if x2 > x1 and y2 > y1:
            roi_mask[y1:y2, x1:x2] = True
        valid = valid & roi_mask

    valid_yx = np.argwhere(valid)
    n_valid = len(valid_yx)

    if n_valid < k_clique:
        return {
            "error": f"Fewer than {k_clique} valid pixels found in selected region ({n_valid} available).",
            "safety": False,
            "n_valid_pixels": n_valid,
        }

    # 5. Compute tissue-average trace across all valid tissue/ROI pixels
    tissue_mean = np.nanmean(stack[:, valid], axis=1)  # shape (T,)
    tissue_std = float(np.nanstd(tissue_mean))
    if np.isnan(tissue_std) or tissue_std <= 1e-12:
        return {"error": "Tissue-average trace has zero variance; cannot compute correlations.", "safety": False}

    # 6. Sample pixels with reproducible seed (default 250, capped at 1000)
    n_samples_effective = int(min(max(1, n_samples), 1000, n_valid))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(n_valid, size=n_samples_effective, replace=False)
    sampled_yx = valid_yx[sampled_idx]  # shape (N, 2)

    # 7. Extract pixel traces and filter by tissue correlation
    # traces: shape (N, T)
    traces = stack[:, sampled_yx[:, 0], sampled_yx[:, 1]].T
    tissue_r = np.zeros(n_samples_effective, dtype=np.float32)

    for i in range(n_samples_effective):
        tr = traces[i]
        tr_std = float(np.std(tr))
        if tr_std <= 1e-12 or np.isnan(tr_std):
            tissue_r[i] = np.nan
        else:
            tissue_r[i] = float(np.corrcoef(tr, tissue_mean)[0, 1])

    # Tissue correlation filter (positive threshold by default)
    if float(tissue_r_thresh) > 0:
        keep_mask = (tissue_r >= float(tissue_r_thresh)) & (tissue_r > 0) & np.isfinite(tissue_r)
    else:
        keep_mask = (tissue_r >= float(tissue_r_thresh)) & np.isfinite(tissue_r)
    retained_indices = np.where(keep_mask)[0]
    n_retained = len(retained_indices)

    if n_retained < k_clique:
        return {
            "error": (
                f"Only {n_retained} pixel(s) passed the positive tissue-correlation threshold "
                f"(r ≥ {tissue_r_thresh:.2f}), but k-clique requires at least {k_clique} nodes."
            ),
            "safety": False,
            "n_valid_pixels": n_valid,
            "n_sampled": n_samples_effective,
            "n_retained": n_retained,
            "sampled_yx": sampled_yx,
            "tissue_r": tissue_r,
        }

    # 8. Pairwise Pearson correlation and R² calculation
    retained_yx = sampled_yx[retained_indices]
    retained_traces = traces[retained_indices]  # (n_retained, T)
    retained_tissue_r = tissue_r[retained_indices]

    means = np.mean(retained_traces, axis=1, keepdims=True)
    stds = np.std(retained_traces, axis=1, keepdims=True)
    stds = np.maximum(stds, 1e-12)
    Z = (retained_traces - means) / stds
    C = (Z @ Z.T) / float(T)
    np.clip(C, -1.0, 1.0, out=C)
    R2 = C ** 2
    np.fill_diagonal(C, 1.0)
    np.fill_diagonal(R2, 1.0)

    # Edge filtering
    r2_thresh_val = float(r2_thresh)
    if positive_edges_only:
        edge_mask = (R2 >= r2_thresh_val) & (C > 0)
    else:
        edge_mask = (R2 >= r2_thresh_val)
    np.fill_diagonal(edge_mask, False)

    iu = np.triu_indices(n_retained, k=1)
    edge_pairs = [(int(i), int(j)) for i, j in zip(iu[0], iu[1]) if edge_mask[i, j]]
    n_edges = len(edge_pairs)

    density = float((2.0 * n_edges) / (n_retained * (n_retained - 1))) if n_retained > 1 else 0.0
    mean_degree = float((2.0 * n_edges) / float(n_retained)) if n_retained > 0 else 0.0

    # 9. Multi-factor safety preflight check
    if k_clique > n_retained:
        return {
            "error": f"k-clique size ({k_clique}) is larger than the number of retained nodes ({n_retained}).",
            "safety": True,
            "n_retained": n_retained,
            "n_edges": n_edges,
            "density": density,
        }
    if n_edges > 50000:
        return {
            "error": (
                f"Graph is too dense ({n_edges:,} edges > 50,000 limit). "
                f"Increase the Pearson R² threshold or reduce sample size."
            ),
            "safety": True,
            "n_retained": n_retained,
            "n_edges": n_edges,
            "density": density,
        }
    if density > 0.85 and n_retained >= 20:
        return {
            "error": (
                f"Graph density ({density:.2f}) is too high for k-clique community detection. "
                f"Increase the Pearson R² threshold or reduce sample size."
            ),
            "safety": True,
            "n_retained": n_retained,
            "n_edges": n_edges,
            "density": density,
        }
    if mean_degree > (0.70 * n_retained) and n_retained >= 25:
        return {
            "error": (
                f"Mean node degree ({mean_degree:.1f}) is too high for k-clique community detection. "
                f"Increase the Pearson R² threshold or reduce sample size."
            ),
            "safety": True,
            "n_retained": n_retained,
            "n_edges": n_edges,
            "density": density,
        }

    # 10. Build NetworkX graph
    G = nx.Graph()
    for idx in range(n_retained):
        G.add_node(
            idx,
            y=int(retained_yx[idx, 0]),
            x=int(retained_yx[idx, 1]),
            tissue_r=float(retained_tissue_r[idx]),
        )
    for i, j in edge_pairs:
        G.add_edge(i, j, weight=float(R2[i, j]), r=float(C[i, j]), r2=float(R2[i, j]))

    degrees = [int(G.degree[i]) for i in range(n_retained)]
    max_deg = max(degrees) if degrees else 0
    n_components = int(nx.number_connected_components(G)) if n_retained > 0 else 0

    # Clique count safety preflight check
    import itertools
    try:
        clique_gen = nx.find_cliques(G)
        sample_cliques = list(itertools.islice(clique_gen, 25001))
        if len(sample_cliques) > 25000:
            return {
                "error": (
                    f"Graph contains too many candidate cliques (>25,000). "
                    f"Increase the Pearson R² threshold or reduce sample size."
                ),
                "safety": True,
                "n_retained": n_retained,
                "n_edges": n_edges,
                "density": density,
            }
    except Exception:
        pass

    # 11. k-clique community detection
    try:
        raw_communities = list(k_clique_communities(G, k=int(k_clique)))
    except Exception:
        raw_communities = []

    # Sort communities by size descending
    raw_communities.sort(key=lambda s: len(s), reverse=True)
    n_communities = len(raw_communities)

    # 12. Map nodes to communities (tracking overlapping membership)
    node_community_lists = []
    primary_communities = []
    overlap_counts = []

    for i in range(n_retained):
        c_ids = [c for c, comm in enumerate(raw_communities) if i in comm]
        node_community_lists.append(c_ids)
        primary_communities.append(c_ids[0] if c_ids else -1)
        overlap_counts.append(len(c_ids))

    n_overlapping = sum(1 for cnt in overlap_counts if cnt > 1)
    n_assigned = sum(1 for cnt in overlap_counts if cnt >= 1)
    n_unassigned = sum(1 for cnt in overlap_counts if cnt == 0)

    # 13. Construct output DataFrames
    nodes_df = pd.DataFrame({
        "node_id": list(range(n_retained)),
        "y": retained_yx[:, 0].astype(int),
        "x": retained_yx[:, 1].astype(int),
        "tissue_r": np.round(retained_tissue_r, 4),
        "degree": degrees,
        "community_ids": [";".join(str(c) for c in c_ids) if c_ids else "None" for c_ids in node_community_lists],
        "primary_community": primary_communities,
        "overlap_count": overlap_counts,
    })

    if edge_pairs:
        edges_df = pd.DataFrame({
            "node_i": [p[0] for p in edge_pairs],
            "node_j": [p[1] for p in edge_pairs],
            "source_y": [int(retained_yx[p[0], 0]) for p in edge_pairs],
            "source_x": [int(retained_yx[p[0], 1]) for p in edge_pairs],
            "target_y": [int(retained_yx[p[1], 0]) for p in edge_pairs],
            "target_x": [int(retained_yx[p[1], 1]) for p in edge_pairs],
            "pearson_r": [round(float(C[p[0], p[1]]), 4) for p in edge_pairs],
            "r_squared": [round(float(R2[p[0], p[1]]), 4) for p in edge_pairs],
        })
    else:
        edges_df = pd.DataFrame(
            columns=["node_i", "node_j", "source_y", "source_x", "target_y", "target_x", "pearson_r", "r_squared"]
        )

    summary_rows = [
        ("Dataset", str(dataset_name)),
        ("ROI active", bool(roi_box is not None)),
        ("ROI box", str(roi_box) if roi_box else "None"),
        ("Random seed", int(seed)),
        ("Sampled pixels", int(n_samples_effective)),
        ("Valid tissue pixels", int(n_valid)),
        ("Retained nodes after tissue filter", int(n_retained)),
        ("Network edges", int(n_edges)),
        ("Graph density", round(float(density), 4)),
        ("Mean degree", round(float(mean_degree), 2)),
        ("Connected components", int(n_components)),
        ("k-clique size", int(k_clique)),
        ("Number of communities", int(n_communities)),
        ("Assigned nodes", int(n_assigned)),
        ("Unassigned nodes", int(n_unassigned)),
        ("Overlapping nodes", int(n_overlapping)),
        ("Tissue correlation threshold", float(tissue_r_thresh)),
        ("Pearson R² threshold", float(r2_thresh)),
        ("Positive edges only", bool(positive_edges_only)),
        ("Restrict to segmented tissue", bool(restrict_to_mask)),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["parameter", "value"])

    return {
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "summary_df": summary_df,
        "graph": G,
        "n_valid_pixels": n_valid,
        "n_sampled": n_samples_effective,
        "n_retained": n_retained,
        "n_nodes": n_retained,
        "n_edges": n_edges,
        "density": density,
        "mean_degree": mean_degree,
        "n_components": n_components,
        "n_communities": n_communities,
        "n_assigned": n_assigned,
        "n_unassigned": n_unassigned,
        "n_overlapping": n_overlapping,
        "k_clique": k_clique,
        "communities": raw_communities,
        "sampled_yx": sampled_yx,
        "retained_yx": retained_yx,
        "tissue_r": retained_tissue_r,
        "all_sampled_tissue_r": tissue_r,
        "safety_triggered": False,
        "error": None,
    }


# The communication analysis is deliberately defined after the original
# exploratory pixel implementation above.  The public function below is the
# one used by the GUI and tests: one node is one segmented cell, not one pixel.
def analyze_cell_network(
    dff_df,
    mask0=None,
    roi_box=None,
    n_samples=250,
    seed=0,
    tissue_r_thresh=0.30,
    r2_thresh=0.70,
    positive_edges_only=False,
    tissue_positive_only=True,
    k_clique=6,
    restrict_to_mask=True,
    max_edges_for_clique=25000,
    dataset_name="",
):
    """Build a cell-level calcium communication network.

    This follows the referenced NetworkX workflow while using CAFIN's
    extracted single-cell ΔF/F0i traces as the node signals. Each cell is a
    node, its trace is correlated with the field-average cell trace for the
    node filter, and pairwise Pearson R² determines graph edges. k-clique
    percolation then identifies possibly overlapping communities.

    ``positive_edges_only=False`` reproduces the R² convention: a strong
    negative Pearson correlation also produces an edge. The tissue-reference
    filter is positive by default because it removes cells that do not follow
    the majority activity. Spatial coordinates come from ``mask0``; if no
    mask is available, coordinates are reported as NaN but the graph remains
    valid.
    """
    import networkx as nx
    from networkx.algorithms.community import k_clique_communities

    if not isinstance(dff_df, pd.DataFrame):
        return {
            "error": "Cell communication analysis requires a per-cell trace DataFrame.",
            "safety": False,
        }
    cell_cols = [c for c in dff_df.columns if str(c).startswith("Cell_")]
    if len(cell_cols) < 2:
        return {"error": "At least two cell traces are required for communication analysis.",
                "safety": False}
    if len(dff_df) < 3:
        return {"error": "At least three time points are required for correlation analysis.",
                "safety": False}
    if not 0.0 <= float(r2_thresh) <= 1.0:
        return {"error": "The Pearson R² threshold must be between 0 and 1.", "safety": False}
    if int(k_clique) < 2:
        return {"error": "The k-clique size must be at least 2.", "safety": False}

    try:
        cell_ids = np.asarray([int(str(c).split("_", 1)[1]) for c in cell_cols], dtype=int)
    except (IndexError, ValueError):
        return {"error": "Cell trace columns must use the Cell_<integer> naming convention.",
                "safety": False}

    A = dff_df[cell_cols].to_numpy(dtype=float)
    A = np.where(np.isfinite(A), A, np.nan)
    n_frames, n_cells = A.shape

    # Keep only cells represented by the selected segmented mask. This makes
    # the all-cell and post-analysis ROI runs obey the same spatial rule.
    centroids = {}
    mask_ids = set(cell_ids.tolist())
    if mask0 is not None:
        m0 = np.asarray(mask0)
        for p in regionprops(m0.astype(np.int32)):
            cy, cx = p.centroid
            centroids[int(p.label)] = (float(cx), float(cy))
        if restrict_to_mask:
            mask_ids &= set(centroids)

    eligible = np.ones(n_cells, dtype=bool)
    if restrict_to_mask and mask0 is not None:
        eligible &= np.asarray([cid in mask_ids for cid in cell_ids], dtype=bool)
    if roi_box is not None:
        if mask0 is not None:
            roi_ids = set(roi_cell_ids(np.asarray(mask0), roi_box))
            eligible &= np.asarray([cid in roi_ids for cid in cell_ids], dtype=bool)
        else:
            x1, y1, x2, y2 = [int(v) for v in roi_box]
            x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
            eligible &= np.asarray([
                cid in centroids and x1 <= centroids[cid][0] < x2 and y1 <= centroids[cid][1] < y2
                for cid in cell_ids
            ], dtype=bool)

    eligible_idx = np.flatnonzero(eligible)
    n_available = int(len(eligible_idx))
    if n_available < int(k_clique):
        return {
            "error": f"Fewer than {k_clique} eligible cells are available ({n_available}).",
            "safety": False,
            "n_valid_cells": n_available,
        }

    # A field trace is the mean of the eligible cell activities, not a mean of
    # arbitrary image pixels. The node sample is reproducible and capped to
    # keep pairwise correlation and clique detection responsive.
    n_samples_effective = int(min(max(1, int(n_samples)), 250, n_available))
    rng = np.random.default_rng(int(seed))
    sampled_idx = rng.choice(eligible_idx, size=n_samples_effective, replace=False)
    sampled_cell_ids = cell_ids[sampled_idx]
    tissue_mean = np.nanmean(A[:, eligible_idx], axis=1)

    def _corr(a, b):
        good = np.isfinite(a) & np.isfinite(b)
        if int(good.sum()) < 3:
            return np.nan
        aa, bb = a[good], b[good]
        if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
            return np.nan
        return float(np.corrcoef(aa, bb)[0, 1])

    sampled_tissue_r = np.asarray([_corr(A[:, i], tissue_mean) for i in sampled_idx], dtype=float)
    threshold = float(tissue_r_thresh)
    if tissue_positive_only:
        keep = (sampled_tissue_r >= threshold) & (sampled_tissue_r > 0)
    else:
        keep = sampled_tissue_r >= threshold
    keep &= np.isfinite(sampled_tissue_r)
    retained_local = np.flatnonzero(keep)
    n_retained = int(len(retained_local))
    if n_retained < int(k_clique):
        return {
            "error": (f"Only {n_retained} cells passed the tissue-correlation filter "
                      f"(r ≥ {threshold:.2f}); k-clique requires at least {k_clique} nodes."),
            "safety": False,
            "n_valid_cells": n_available,
            "n_sampled": n_samples_effective,
            "n_retained": n_retained,
            "sampled_cell_ids": sampled_cell_ids,
            "tissue_r": sampled_tissue_r,
        }

    retained_idx = sampled_idx[retained_local]
    retained_cell_ids = cell_ids[retained_idx]
    traces = A[:, retained_idx].T
    centered = traces - np.nanmean(traces, axis=1, keepdims=True)
    scales = np.nanstd(centered, axis=1, keepdims=True)
    valid_trace = np.isfinite(scales[:, 0]) & (scales[:, 0] > 1e-12)
    if not np.all(valid_trace):
        retained_idx = retained_idx[valid_trace]
        retained_cell_ids = retained_cell_ids[valid_trace]
        retained_local = retained_local[valid_trace]
        traces = traces[valid_trace]
        n_retained = int(len(retained_idx))
        centered = traces - np.nanmean(traces, axis=1, keepdims=True)
        scales = np.nanstd(centered, axis=1, keepdims=True)
    if n_retained < int(k_clique):
        return {"error": "Too few non-constant cell traces remain after filtering.", "safety": False,
                "n_retained": n_retained}

    # There are no NaNs after the valid-trace check in normal CAFIN output;
    # pairwise finite masking keeps this robust for partially missing traces.
    C = np.eye(n_retained, dtype=float)
    for i in range(n_retained):
        for j in range(i + 1, n_retained):
            r = _corr(traces[i], traces[j])
            C[i, j] = C[j, i] = 0.0 if not np.isfinite(r) else r
    np.clip(C, -1.0, 1.0, out=C)
    R2 = C ** 2
    edge_mask = R2 >= float(r2_thresh)
    if positive_edges_only:
        edge_mask &= C > 0
    np.fill_diagonal(edge_mask, False)
    upper = np.triu_indices(n_retained, k=1)
    edge_pairs = [(int(i), int(j)) for i, j in zip(upper[0], upper[1]) if edge_mask[i, j]]
    n_edges = int(len(edge_pairs))
    density = float(2 * n_edges / (n_retained * (n_retained - 1))) if n_retained > 1 else 0.0
    mean_degree = float(2 * n_edges / n_retained) if n_retained else 0.0

    # These guards run before NetworkX clique enumeration. No user override is
    # offered because an accidental complete graph can otherwise freeze GUI.
    if n_edges > int(max_edges_for_clique):
        return {"error": (f"Graph has {n_edges:,} edges, above the safe limit of "
                           f"{int(max_edges_for_clique):,}. Increase R² or reduce sampled cells."),
                "safety": True, "n_retained": n_retained, "n_edges": n_edges, "density": density}
    if n_retained >= 20 and density > 0.85:
        return {"error": (f"Graph density is {density:.2f}, too high for safe k-clique detection. "
                           "Increase R² or reduce sampled cells."), "safety": True,
                "n_retained": n_retained, "n_edges": n_edges, "density": density}
    if n_retained >= 25 and mean_degree > 0.70 * (n_retained - 1):
        return {"error": (f"Mean cell degree is {mean_degree:.1f}, too high for safe k-clique detection. "
                           "Increase R² or reduce sampled cells."), "safety": True,
                "n_retained": n_retained, "n_edges": n_edges, "density": density}

    G = nx.Graph()
    for i, cid in enumerate(retained_cell_ids):
        x, y = centroids.get(int(cid), (np.nan, np.nan))
        G.add_node(i, cell_id=int(cid), x=x, y=y,
                   tissue_r=float(sampled_tissue_r[retained_local[i]]))
    for i, j in edge_pairs:
        G.add_edge(i, j, weight=float(R2[i, j]), r=float(C[i, j]), r2=float(R2[i, j]))

    communities = list(k_clique_communities(G, int(k_clique)))
    communities.sort(key=lambda s: (-len(s), min(s) if s else -1))
    membership = [[c for c, comm in enumerate(communities) if i in comm] for i in range(n_retained)]
    primary = [ids[0] if ids else -1 for ids in membership]
    overlap = [len(ids) for ids in membership]
    degrees = [int(G.degree(i)) for i in range(n_retained)]
    n_components = int(nx.number_connected_components(G))
    coords = [centroids.get(int(cid), (np.nan, np.nan)) for cid in retained_cell_ids]
    nodes_df = pd.DataFrame({
        "node_id": np.arange(n_retained, dtype=int),
        "cell_id": retained_cell_ids.astype(int),
        "x": [p[0] for p in coords], "y": [p[1] for p in coords],
        "tissue_r": np.round([sampled_tissue_r[retained_local[i]] for i in range(n_retained)], 4),
        "degree": degrees,
        "community_ids": [";".join(map(str, ids)) if ids else "None" for ids in membership],
        "primary_community": primary,
        "overlap_count": overlap,
    })
    edge_columns = ["node_i", "node_j", "cell_i", "cell_j", "source_x", "source_y",
                    "target_x", "target_y", "pearson_r", "r_squared"]
    edges_df = pd.DataFrame([
        (i, j, int(retained_cell_ids[i]), int(retained_cell_ids[j]), coords[i][0], coords[i][1],
         coords[j][0], coords[j][1], round(float(C[i, j]), 4), round(float(R2[i, j]), 4))
        for i, j in edge_pairs
    ], columns=edge_columns)
    summary_rows = [
        ("Dataset", str(dataset_name)), ("Node definition", "segmented cell"),
        ("ROI active", bool(roi_box is not None)), ("ROI box", str(roi_box) if roi_box else "None"),
        ("Random seed", int(seed)), ("Sampled cells", n_samples_effective),
        ("Eligible cells", n_available), ("Retained nodes after tissue filter", n_retained),
        ("Network edges", n_edges), ("Graph density", round(density, 4)),
        ("Mean degree", round(mean_degree, 2)), ("Connected components", n_components),
        ("k-clique size", int(k_clique)), ("Number of communities", len(communities)),
        ("Assigned nodes", int(sum(v > 0 for v in overlap))),
        ("Unassigned nodes", int(sum(v == 0 for v in overlap))),
        ("Overlapping nodes", int(sum(v > 1 for v in overlap))),
        ("Tissue correlation threshold", threshold), ("Tissue positive-only filter", bool(tissue_positive_only)),
        ("Pearson R² threshold", float(r2_thresh)), ("Positive edges only", bool(positive_edges_only)),
        ("Restrict to segmented mask", bool(restrict_to_mask)),
    ]
    return {
        "nodes_df": nodes_df, "edges_df": edges_df,
        "summary_df": pd.DataFrame(summary_rows, columns=["parameter", "value"]),
        "graph": G, "n_valid_cells": n_available, "n_sampled": n_samples_effective,
        "n_retained": n_retained, "n_nodes": n_retained, "n_edges": n_edges,
        "density": density, "mean_degree": mean_degree, "n_components": n_components,
        "n_communities": len(communities), "n_assigned": int(sum(v > 0 for v in overlap)),
        "n_unassigned": int(sum(v == 0 for v in overlap)),
        "n_overlapping": int(sum(v > 1 for v in overlap)), "k_clique": int(k_clique),
        "communities": communities, "sampled_cell_ids": sampled_cell_ids,
        "retained_cell_ids": retained_cell_ids, "tissue_r": sampled_tissue_r,
        "error": None, "safety": False,
    }


def analyze_calcium_network(dff_df, **kwargs):
    """Public communication-analysis API: one node per segmented cell."""
    return analyze_cell_network(dff_df=dff_df, **kwargs)
