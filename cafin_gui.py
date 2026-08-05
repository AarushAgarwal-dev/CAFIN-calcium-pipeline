"""
CAFIN Calcium Analysis GUI  (Streamlit)
=======================================
Motion correction -> segmentation -> ΔF/F0i -> statistics, all inline.

Run:
    streamlit run cafin_gui.py

Registration / analysis methods (choose one):
    * Rigid           OpenCV ECC (global rotation + translation)
    * Elastic         itk-elastix B-spline (non-rigid)
    * Cell tracking   every frame is segmented and cells are linked into stable IDs
Extras: data preview, interactive ROI selection (before or after a run), frame navigation
with auto-loop, PCA + K-means trace clustering, and optional AI interpretation via Bedrock.
"""
import os, glob, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import imageio.v2 as imageio
import skimage.io as skio
from skimage.segmentation import find_boundaries

import cafin_core as cc

# Compat shim: streamlit-drawable-canvas calls the OLD
#   streamlit.elements.image.image_to_url(image, width:int, clamp, channels, fmt, id)
# but newer Streamlit (>=1.5x) moved it to streamlit.elements.lib.image_utils with a new
# signature (image, layout_config:LayoutConfig, ...). Adapt the old call to the new one.
try:
    import streamlit.elements.image as _stimg
    if not hasattr(_stimg, "image_to_url"):
        from streamlit.elements.lib.image_utils import image_to_url as _new_i2u
        from streamlit.elements.lib.layout_utils import LayoutConfig as _LC

        def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
            try:
                return _new_i2u(image, _LC(width=width), clamp, channels, output_format, image_id)
            except TypeError:                       # very old signature still present
                return _new_i2u(image, width, clamp, channels, output_format, image_id)

        _stimg.image_to_url = _image_to_url_compat
except Exception:
    pass

st.set_page_config(page_title="CAFIN Calcium Analysis", layout="wide")
ss = st.session_state


def _draw_box(gray_u8, box, color=(255, 235, 0), t=2):
    """Return an RGB image with the ROI rectangle drawn on a grayscale frame."""
    x1, y1, x2, y2 = [int(v) for v in box]
    rgb = np.dstack([gray_u8] * 3).copy()
    x1, x2 = sorted((max(0, x1), min(rgb.shape[1], x2)))
    y1, y2 = sorted((max(0, y1), min(rgb.shape[0], y2)))
    rgb[y1:y2, x1:x1 + t] = color; rgb[y1:y2, max(0, x2 - t):x2] = color
    rgb[y1:y1 + t, x1:x2] = color; rgb[max(0, y2 - t):y2, x1:x2] = color
    return rgb


def color_name(rgb):
    """Name a colour from its actual RGB, so the label always matches what is drawn."""
    import colorsys
    r, g, b = [float(v) / 255.0 for v in rgb[:3]]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hd = h * 360
    if s < 0.12:                                    # unsaturated
        return "black" if v < 0.25 else ("white" if v > 0.93 else
                                         ("light gray" if v > 0.6 else "gray"))
    if hd < 15 or hd >= 345: base = "red"
    elif hd < 40:  base = "orange"
    elif hd < 65:  base = "yellow"
    elif hd < 95:  base = "olive"
    elif hd < 160: base = "green"
    elif hd < 200: base = "teal"
    elif hd < 255: base = "blue"
    elif hd < 290: base = "purple"
    elif hd < 330: base = "magenta"
    else:          base = "pink"
    if base in ("red", "orange"):                   # dark/desaturated warm tones read as brown
        if v < 0.65 and s > 0.30:
            return "brown"
        if s < 0.35 and v < 0.88:
            return "light brown"
    if v > 0.80 and s < 0.55:
        return "light " + base
    if v < 0.45:
        return "dark " + base
    return base


def cluster_palette(k):
    """Return (colors, names) with one entry per cluster. tab20 keeps 20 clearly
    distinct, nameable colours; beyond that we spread hues so clusters stay
    visually separable. Names are derived from the colours and de-duplicated."""
    if k <= 20:
        cols = (plt.get_cmap("tab20")(np.linspace(0, 1, 20)) * 255)[:, :3][:max(k, 1)]
    else:
        cols = (plt.get_cmap("hsv")(np.linspace(0, 1, k, endpoint=False)) * 255)[:, :3]
    names, seen = [], {}
    for c in cols:
        n = color_name(c)
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n} {seen[n]}")
    return cols, names


def roi_box_selector(gray, key, current_box=None):
    """Drag-a-rectangle selector over a grayscale frame (Plotly box-select).
    Returns a new (x1, y1, x2, y2) when the user has just dragged one, else None.
    Falls back to numeric inputs if Plotly is unavailable."""
    H, W = gray.shape
    try:
        import plotly.graph_objects as go
        step = max(3, W // 120)                          # invisible selectable grid
        yy, xx = np.mgrid[0:H:step, 0:W:step]
        fig = go.Figure(go.Image(z=np.dstack([gray] * 3)))
        fig.add_trace(go.Scattergl(x=xx.ravel(), y=yy.ravel(), mode="markers",
                                   marker=dict(size=step, opacity=0), hoverinfo="skip",
                                   showlegend=False))
        if current_box:
            x1, y1, x2, y2 = current_box
            fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2,
                          line=dict(color="#ffeb00", width=2), fillcolor="rgba(255,235,0,0.15)")
        fig.update_xaxes(visible=False, range=[0, W]); fig.update_yaxes(visible=False, range=[H, 0])
        fig.update_layout(dragmode="select", margin=dict(l=0, r=0, t=0, b=0),
                          height=min(560, int(560 * H / W)) if W else 512)
        ev = st.plotly_chart(fig, on_select="rerun", selection_mode="box",
                             width="stretch", key=key)
        pts = []
        try:
            pts = ev["selection"]["points"]
        except Exception:
            try:
                pts = ev.selection.points
            except Exception:
                pts = []
        if pts:
            xs = [p["x"] for p in pts]; ys = [p["y"] for p in pts]
            return (int(max(0, min(xs))), int(max(0, min(ys))),
                    int(min(W, max(xs))), int(min(H, max(ys))))
        return None
    except Exception as e:
        st.error(f"Interactive selector unavailable ({e}). Enter ROI bounds manually:")
        cur = current_box or (W // 4, H // 4, 3 * W // 4, 3 * H // 4)
        cA = st.columns(4)
        x1 = cA[0].number_input("x1", 0, W, int(cur[0]), key=key + "_x1")
        y1 = cA[1].number_input("y1", 0, H, int(cur[1]), key=key + "_y1")
        x2 = cA[2].number_input("x2", 0, W, int(cur[2]), key=key + "_x2")
        y2 = cA[3].number_input("y2", 0, H, int(cur[3]), key=key + "_y2")
        st.image(_draw_box(gray, (x1, y1, x2, y2)), width="stretch")
        box = (x1, y1, x2, y2)
        return box if box != tuple(current_box or ()) else None


def pick_folder(initial=None, title="Select folder"):
    """Open a native folder-selection dialog. Works because the app runs locally
    (the dialog opens on the machine running Streamlit). Returns a path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.askdirectory(master=root, title=title,
                                       initialdir=initial if initial and os.path.isdir(initial)
                                       else os.getcwd())
        root.destroy()
        return path or None
    except Exception:
        return None


def find_trial_folders(root_dir, max_depth=3):
    """Return folders under root_dir that contain both membrane/ and ca2/ subfolders."""
    out = []
    root_dir = os.path.abspath(root_dir)
    base = root_dir.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root_dir):
        if dirpath.count(os.sep) - base >= max_depth:
            dirnames[:] = []
            continue
        if "membrane" in dirnames and "ca2" in dirnames:
            out.append(dirpath)
    return sorted(out)


def frame_controls(key, n, label="Frame", start=None):
    """Prev/Next buttons + slider + auto-loop toggle for stepping through frames.
    Returns (index, playing, speed). Each tab passes a distinct `key`. The slider is
    keyed (unique id); the auto-loop advance is applied BEFORE the slider is created,
    since a keyed widget's state cannot be modified after instantiation."""
    if n <= 1:
        return 0, False, 0.05
    sk = key + "_slider"                                # slider's own key = source of truth
    if ss.get(sk) is None:
        ss[sk] = (n - 1) if start is None else int(min(max(0, start), n - 1))
    if ss.pop(key + "_advance", False):                 # apply pending auto-loop step
        ss[sk] = (int(ss[sk]) + 1) % n
    ss[sk] = int(min(max(0, int(ss[sk])), n - 1))
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    if c1.button("◀ Prev", key=key + "_prev"):
        ss[sk] = (ss[sk] - 1) % n
    if c2.button("Next ▶", key=key + "_next"):
        ss[sk] = (ss[sk] + 1) % n
    play = c3.toggle("▶ Auto-loop", key=key + "_play")
    speed = c4.slider("loop speed (s/frame)", 0.05, 1.0, 0.05, 0.05, key=key + "_speed")
    idx = int(st.slider(label, 0, n - 1, key=sk))
    return idx, play, speed


def frame_loop(key, idx, play, speed, n):
    """If playing, flag the next frame and rerun; frame_controls applies the step."""
    if play and n > 1:
        ss[key + "_advance"] = True
        time.sleep(speed)
        st.rerun()


def detect_base(folder):
    for t in sorted(os.path.basename(f) for f in glob.glob(os.path.join(folder, "*.tif"))):
        m = re.match(r"(.*?)(\d{4})\.tif$", t)
        if m:
            return m.group(1)
    return ""


# =========================================================== SIDEBAR
st.sidebar.title("CAFIN pipeline")
st.sidebar.caption("Motion correction → segmentation → ΔF/F0i → statistics")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if "trial_path" not in ss:
    ss["trial_path"] = os.path.join(APP_DIR, "lat_trial1_afterdrug")

st.sidebar.markdown("### Data folder")
_b1, _b2 = st.sidebar.columns(2)
if _b1.button("📂 Browse…", width="stretch", help="Open a folder picker on this machine"):
    _p = pick_folder(ss["trial_path"], "Select the trial folder (with membrane/ and ca2/)")
    if _p:
        ss["trial_path"] = os.path.normpath(_p)
        st.rerun()
if _b2.button("🔍 Find trials", width="stretch", help="Scan for folders containing membrane/ and ca2/"):
    found = find_trial_folders(APP_DIR) or find_trial_folders(os.path.dirname(APP_DIR))
    ss["found_trials"] = found
    if not found:
        st.sidebar.warning("No folders with membrane/ and ca2/ found nearby.")

if ss.get("found_trials"):
    _opts = ["(choose a detected folder)"] + ss["found_trials"]
    _sel = st.sidebar.selectbox("Detected trial folders", _opts,
                                format_func=lambda p: p if p.startswith("(") else os.path.relpath(p, APP_DIR))
    if not _sel.startswith("(") and os.path.abspath(_sel) != os.path.abspath(ss["trial_path"]):
        ss["trial_path"] = os.path.normpath(_sel)
        st.rerun()

trial = st.sidebar.text_input("Trial folder (contains membrane/ and ca2/)", value=ss["trial_path"])
if trial and os.path.normpath(trial) != os.path.normpath(ss["trial_path"]):
    ss["trial_path"] = os.path.normpath(trial)
memf, caf = os.path.join(trial, "membrane"), os.path.join(trial, "ca2")

mem_base = ca_base = ""
n_frames = 0
ok_data = os.path.isdir(memf) and os.path.isdir(caf)
if ok_data:
    mem_base, ca_base = detect_base(memf), detect_base(caf)
    n_frames = cc.count_frames(memf, mem_base)
    st.sidebar.success(f"membrane base: {mem_base}\n\ncalcium base: {ca_base}\n\nframes: {n_frames}")
else:
    st.sidebar.error("membrane/ and ca2/ subfolders not found in that path.")

mem_base = st.sidebar.text_input("Membrane base name", mem_base)
ca_base = st.sidebar.text_input("Calcium base name", ca_base)

# --------------------------------------------------- methods
st.sidebar.markdown("### Analysis method")
choice = st.sidebar.radio(
    "Registration / analysis method",
    ["Rigid (ECC)", "Elastic (itk-elastix)", "Cell tracking"],
    help="Rigid / Elastic = motion-correct, then read calcium with a fixed frame-0 mask. "
         "Cell tracking = segment every frame independently and link cells into stable IDs "
         "(Linlin Li's tracker: Hungarian matching + gap closing). No registration — exactly the "
         "wound-closure pipeline: segment → clean → track.")
elastic_quality = "balanced"
link_tracking = False
TP = None
if choice == "Rigid (ECC)":
    method, handling = "rigid", "static"
elif choice == "Elastic (itk-elastix)":
    method, handling = "elastic", "static"
    elastic_quality = st.sidebar.select_slider("Elastic quality (itk-elastix)",
                                               ["fast", "balanced", "accurate"], value="balanced")
    st.sidebar.caption("itk-elastix non-rigid B-spline ≈ 0.5 / 3 / 5 s per frame.")
else:   # Cell tracking — exactly the wound-closure pipeline (segment -> clean -> track)
    link_tracking = True
    import cafin_track as ctk
    method, handling = "none", "linktrack"        # no registration, matching the shared pipeline
    with st.sidebar.expander("Tracking parameters", expanded=False):
        _tm = st.selectbox("Assignment", ["hungarian", "greedy"], 0)
        _gap = st.number_input("Gap-closing frames", 0, 10, 2)
        _maxd = st.number_input("Max centroid distance (px)", 5, 400, 50)
        _minov = st.number_input("Min pixel overlap", 1, 5000, 50)
        _miniou = st.number_input("Min IoU", 0.0, 1.0, 0.1, 0.05)
        _startf = st.number_input("Reference frame index", 0, 5000, 0)
        _wov = st.number_input("weight — overlap", 0.0, 1.0, 0.5, 0.05)
        _wsz = st.number_input("weight — size", 0.0, 1.0, 0.3, 0.05)
        _wce = st.number_input("weight — centroid", 0.0, 1.0, 0.2, 0.05)
    TP = ctk.TrackParams(track_method=_tm, track_gap_frames=int(_gap),
                         track_max_distance_px=int(_maxd), track_pixel_threshold=int(_minov),
                         track_min_iou=float(_miniou), track_start_frame=int(_startf),
                         track_w_overlap=float(_wov), track_w_size=float(_wsz),
                         track_w_centroid=float(_wce))
    st.sidebar.warning("Segments EVERY frame with Cellpose → slow on CPU. "
                       "Use **frame-step** below to subsample long series.")

# --- GPU / CUDA ---
cuda_ok, cuda_msg, gpu_backend = cc.gpu_status()
use_gpu = st.sidebar.checkbox("Use GPU for segmentation", value=cuda_ok,
                              help="Works with NVIDIA (CUDA), AMD (DirectML on Windows, ROCm on "
                                   "Linux), and Apple Silicon. Registration stays on CPU.")
(st.sidebar.success if cuda_ok else st.sidebar.info)(cuda_msg)

with st.sidebar.expander("Advanced parameters", expanded=False):
    diameter = st.number_input("Cellpose cell diameter (px)", 5, 60, 15)
    frame_step = st.number_input("Process every Nth frame (speed)", 1, 20, 1)
    max_frames = st.number_input("Max frames (0 = all)", 0, 2000, 0)
    do_bg = st.checkbox("Background subtraction (auto regions)", value=False)
    dff_method = st.selectbox("Baseline for F0i", ["percell", "min", "lowest", "first", "last"],
                              index=0,
                              help="percell: each cell uses its own lowest frames. "
                                   "min: that cell's single lowest value. "
                                   "lowest/first/last: one shared window for every cell.")
    f0_floor = st.number_input("F0i floor (avoid divide-by-tiny)", 0.1, 100.0, 1.0)
    peak_thr = st.number_input("Peak threshold (ΔF/F0i)", 0.0, 5.0, 0.5, 0.1)

roi_on = st.sidebar.checkbox("Restrict analysis to a region of interest (ROI)", value=False,
                             help="Draw a rectangle below; cells intersecting it are analyzed.")

run = st.sidebar.button("▶  Run analysis", type="primary", width="stretch", disabled=not ok_data)

st.title("Ca²⁺ transient analysis")
st.caption(f"**Method:** {choice}  •  registration = `{method}`  •  cells = `{handling}`"
           + ("  •  ROI restricted" if roi_on else ""))


# =========================================================== ROI SELECTION (pre-run)
def load_first_ca():
    p = os.path.join(caf, f"{ca_base}0000.tif")
    return skio.imread(p) if os.path.exists(p) else None


if roi_on and ok_data:
    with st.expander("① Draw the region of interest — drag a box on the frame",
                     expanded=True):
        first_ca = load_first_ca()
        if first_ca is None:
            st.warning("Could not read the first calcium frame.")
        else:
            H, W = first_ca.shape
            gray = cc.stretch8(first_ca, clahe=True)
            st.caption("Drag a rectangle on the image to set the ROI. Drag again to replace it. "
                       "You can also set or change it after the analysis, in the ROI tab.")
            _newb = roi_box_selector(gray, "roi_plot", ss.get("roi_box"))
            if _newb:
                ss["roi_box"] = _newb
            if ss.get("roi_box"):
                b = ss["roi_box"]
                st.success(f"ROI set — x[{b[0]},{b[2]}] y[{b[1]},{b[3]}]. Click **Run analysis**.")
            else:
                st.info("No box drawn yet — drag a rectangle on the image above.")


# =========================================================== PREVIEW (before analysis)
if ok_data and n_frames:
    with st.expander("👁  Preview data (before running the analysis)", expanded=("res" not in ss)):
        pfi, pplay, pspd = frame_controls("prev", n_frames, start=0)
        try:
            _pm_path = os.path.join(memf, f"{mem_base}{pfi:04d}.tif")
            _pc_path = os.path.join(caf, f"{ca_base}{pfi:04d}.tif")
            pm = skio.imread(_pm_path) if os.path.exists(_pm_path) else None
            pc = skio.imread(_pc_path) if os.path.exists(_pc_path) else None
            pv1, pv2 = st.columns(2)
            if pm is not None:
                pv1.image(cc.stretch8(pm, clahe=True),
                          caption=f"Membrane — {os.path.basename(_pm_path)}", width="stretch")
            else:
                pv1.warning("Membrane frame not found.")
            if pc is not None:
                pv2.image(cc.stretch8(pc, clahe=True),
                          caption=f"Calcium — {os.path.basename(_pc_path)}", width="stretch")
            else:
                pv2.warning("Calcium frame not found.")
            ref = pm if pm is not None else pc
            if ref is not None:
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Frames", n_frames)
                q2.metric("Image size", f"{ref.shape[1]} × {ref.shape[0]}")
                q3.metric("Bit depth", str(ref.dtype))
                if pc is not None:
                    q4.metric("Calcium range", f"{int(pc.min())}–{int(pc.max())}")
            st.caption("Step through the raw frames to check the data and judge how much motion "
                       "there is before choosing a registration method.")
            frame_loop("prev", pfi, pplay, pspd, n_frames)
        except Exception as e:
            st.warning(f"Could not preview frames: {e}")

# =========================================================== RUN
if run:
    n_use = n_frames if max_frames == 0 else min(max_frames, n_frames)
    prog = st.progress(0.0, text="Starting…")
    tstats = None
    with st.status("Running pipeline…", expanded=True) as status:
        dev = f"GPU ({gpu_backend})" if (use_gpu and cuda_ok) else "CPU"

        if link_tracking:                                     # ---- segment every frame -> clean -> track ----
            st.write("Loading frames…")
            reg = cc.register_series(memf, caf, mem_base, ca_base, n_use, "none",
                                     do_tracking=False, mask0=None, frame_step=frame_step,
                                     progress=lambda f, m: prog.progress(min(0.10 * f, 0.10), text=m))
            frames = reg["frames"]
            mem_frames = [reg["reg_mem"][f] for f in frames]
            st.write(f"Segmenting every frame ({len(frames)}) with Cellpose ({dev}) + cleaning…")
            masks = cc.segment_stack(mem_frames, diameter=diameter, gpu=use_gpu, min_area=60,
                                     progress=lambda f, m: prog.progress(0.10 + min(0.68 * f, 0.68), text=m))
            st.write(f"Linking cells across frames ({TP.track_method}, gap≤{TP.track_gap_frames})…")
            tracked, tsummary, tstats = ctk.run(masks, TP, frame_idx=frames,
                                                progress=lambda f, m: prog.progress(0.8 + min(0.18 * f, 0.18), text=m))
            ref = int(np.clip(TP.track_start_frame, 0, len(frames) - 1))
            mask0 = tracked[ref]
            reg["mask_per_frame"] = {f: tracked[k] for k, f in enumerate(frames)}
            ca_by_frame = {f: reg["reg_ca"][f] for f in frames if f in reg["reg_ca"]}
            st.write(f"→ {tstats['n_tracks']} tracks ({tstats['full_coverage']} full-coverage). "
                     f"Extracting ΔF/F0i …")
            raw_df = cc.extract_tracked_traces(reg["mask_per_frame"], ca_by_frame, frames, bg=do_bg)
            eff_handling = "tracking"
        else:                                                 # ---- static / warp-tracking ----
            st.write(f"Segmenting reference frame (Cellpose cyto3, {dev})…")
            first_mem = skio.imread(os.path.join(memf, f"{mem_base}0000.tif"))
            mask0, used_gpu = cc.segment(first_mem, diameter=diameter, gpu=use_gpu)
            st.write(f"→ {int(mask0.max())} cells segmented on {'GPU' if used_gpu else 'CPU'}.")
            st.write(f"Registering ({method}) …")
            reg = cc.register_series(memf, caf, mem_base, ca_base, n_use, method,
                                     do_tracking=(handling == "tracking"), mask0=mask0,
                                     frame_step=frame_step, elastic_quality=elastic_quality,
                                     progress=lambda f, m: prog.progress(min(f, 1.0), text=m))
            st.write("Extracting traces + ΔF/F0i …")
            raw_df = cc.extract_traces(reg, mask0, handling, bg=do_bg)
            eff_handling = handling

        roi_ids = None
        if roi_on and ss.get("roi_box"):
            roi_ids = cc.roi_cell_ids(mask0, ss["roi_box"])
            st.write(f"→ ROI: {len(roi_ids)} of {int(mask0.max())} cells intersect the rectangle.")

        dff_df, base, f0i = cc.compute_dff0(raw_df, method=dff_method, floor=f0_floor,
                                            return_f0=True)
        status.update(label="Done", state="complete", expanded=False)
    prog.empty()
    ss["res"] = dict(reg=reg, mask0=mask0, raw=raw_df, dff=dff_df, base=base, mode=choice,
                     method=method, handling=eff_handling, link=link_tracking, tstats=tstats,
                     mem_base=mem_base, ca_base=ca_base, roi_ids=roi_ids,
                     f0_floor=f0_floor, dff_method=dff_method, do_bg=do_bg, f0i=f0i,
                     roi_box=ss.get("roi_box") if roi_on else None, peak_thr=peak_thr)
    ss["frame_idx"] = None  # reset navigation

if "res" not in ss:
    st.info("Set the trial folder + method in the sidebar, then click **Run analysis**.")
    st.stop()

R = ss["res"]
reg, mask0, raw_df, dff_all = R["reg"], R["mask0"], R["raw"], R["dff"]
frames = reg["frames"]
peak_thr = R["peak_thr"]

# ------------------------ general vs ROI cell sets
roi_ids = R.get("roi_ids")
dff_roi = None
if roi_ids:
    _keep = ["Frame"] + [f"Cell_{i}" for i in roi_ids if f"Cell_{i}" in dff_all.columns]
    if len(_keep) > 1:
        dff_roi = dff_all[_keep]
dff_df = dff_all                                   # general set, used for headline and exports
stats, dist = cc.metrics(dff_all, threshold=peak_thr)


def roi_split(render, key):
    """Show the general (all-cell) view, and when an ROI is set add a second
    sub-tab with the same analysis restricted to the ROI cells."""
    if dff_roi is not None:
        _a, _b = st.tabs(["All cells", f"ROI cells ({len(roi_ids)})"])
        with _a:
            render(dff_all, key + "_all")
        with _b:
            render(dff_roi, key + "_roi")
    else:
        render(dff_all, key + "_all")


# headline metrics (whole field)
c = st.columns(4)
c[0].metric("Cells", stats["cells"])
c[1].metric("Active %", stats["active_pct"])
c[2].metric("Mean peak ΔF/F0i", stats["mean_peak_dff0"])
c[3].metric("Sync (r)", stats["temporal_sync_r"])
if dff_roi is not None:
    st.caption(f"An ROI is set ({len(roi_ids)} cells). The data tabs below each have an "
               "**ROI cells** sub-tab next to the general **All cells** view.")

TAB_REG, TAB_SEG, TAB_ROI = "🎞 Registration", "🧫 Segmentation", "🎯 ROI"
TAB_TRC, TAB_CLU, TAB_STA = "📈 Traces / ΔF/F0i", "🧩 Clustering", "📊 Statistics"
TAB_TRK, TAB_DL = "🎯 Tracking", "⬇ Downloads"
_names = [TAB_REG, TAB_SEG, TAB_ROI, TAB_TRC, TAB_STA, TAB_CLU]
if R["handling"] == "tracking":                    # only present when tracking was used
    _names.append(TAB_TRK)
_names.append(TAB_DL)
T = dict(zip(_names, st.tabs(_names)))

# ---------------------------------------------------------- Registration + movie
with T[TAB_REG]:
    st.subheader("Overlay  (green = frame 0, magenta = moving, white = aligned)")
    fr = [f for f in frames if f != 0] or frames
    n = len(fr)
    if ss.get("frame_idx") is None:
        ss["frame_idx"] = n - 1
    b1, b2, b3, b4 = st.columns([1, 1, 2, 2])
    if b1.button("◀ Prev"):
        ss["frame_idx"] = (ss["frame_idx"] - 1) % n
    if b2.button("Next ▶"):
        ss["frame_idx"] = (ss["frame_idx"] + 1) % n
    playing = b3.toggle("▶ Auto-loop", key="playing")
    speed = b4.slider("loop speed (s/frame)", 0.05, 1.0, 0.05, 0.05)
    idx = st.slider("Frame", 0, n - 1, int(ss["frame_idx"]))
    ss["frame_idx"] = idx
    fsel = fr[idx]

    first_mem = reg["raw_mem"][0]
    before = cc.two_color_overlay(first_mem, reg["raw_mem"].get(fsel, first_mem))
    after = cc.two_color_overlay(first_mem, reg["reg_mem"].get(fsel, first_mem))
    a, b = st.columns(2)
    a.image(before, caption=f"BEFORE — frame {fsel}", width="stretch")
    b.image(after, caption=f"AFTER — frame {fsel}", width="stretch")

    if st.button("Build overlay GIF (all frames)"):
        gif = []
        for f in fr:
            bb = cc.two_color_overlay(first_mem, reg["raw_mem"][f])
            aa = cc.two_color_overlay(first_mem, reg["reg_mem"][f])
            gap = np.full((bb.shape[0], 6, 3), 40, np.uint8)
            gif.append(np.hstack([bb, gap, aa]))
        out = os.path.join(trial + "_output", "gui_overlay.gif")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        imageio.mimsave(out, gif, duration=0.12)
        ss["gif_path"] = out
        st.image(out, caption="Left: before | Right: after", width="stretch")

    if playing:
        time.sleep(speed)
        ss["frame_idx"] = (idx + 1) % n
        st.rerun()

# ---------------------------------------------------------- Segmentation
with T[TAB_SEG]:
    st.subheader(f"Cellpose segmentation — {int(mask0.max())} cells")
    st.image(cc.numbered_mask_overlay(reg["reg_mem"][0], mask0),
             caption="Frame-0 membrane with numbered cell ROIs", width="stretch")

    st.subheader("Segmentation overlays (boundaries in red, cell IDs in yellow)")
    sfi, splay, sspd = frame_controls("seg", len(frames))
    sf = frames[sfi]
    seg_mask = reg["mask_per_frame"].get(sf, mask0) if R.get("link") else mask0
    _mem_base = reg["reg_mem"].get(sf, reg["reg_mem"][frames[0]])
    _ca_base = reg["reg_ca"].get(sf, _mem_base)
    _seg_mem = cc.numbered_mask_overlay(_mem_base, seg_mask, clahe=True)
    _seg_ca = cc.numbered_mask_overlay(_ca_base, seg_mask, clahe=True)
    sc1, sc2 = st.columns(2)
    sc1.image(_seg_mem, caption=f"Segmentation + membrane ({mem_base}), frame {sf}", width="stretch")
    sc2.image(_seg_ca, caption=f"Segmentation + calcium ({ca_base}), frame {sf}", width="stretch")
    frame_loop("seg", sfi, splay, sspd, len(frames))

# ---------------------------------------------------------- ROI tab
with T[TAB_ROI]:
    # ---- set or change the ROI now, without re-running the pipeline ----
    with st.expander("✏️  Select or change the ROI", expanded=not R.get("roi_box")):
        st.caption("Drag a rectangle to set the ROI. This re-filters the traces that were already "
                   "extracted, so there is no need to run the analysis again.")
        _ca0 = reg["reg_ca"].get(frames[0], reg["reg_mem"][frames[0]])
        _newroi = roi_box_selector(cc.stretch8(_ca0, clahe=True), "roi_plot_post", R.get("roi_box"))
        if _newroi:
            R["roi_box"] = _newroi
            R["roi_ids"] = cc.roi_cell_ids(mask0, _newroi)
            ss["roi_box"] = _newroi
            st.rerun()
        if R.get("roi_box"):
            if st.button("Clear ROI (use all cells)"):
                R["roi_box"] = None
                R["roi_ids"] = None
                st.rerun()

    if R.get("roi_box"):
        st.subheader(f"Region of interest — {len(roi_ids)} cells intersect the rectangle")
        x1, y1, x2, y2 = R["roi_box"]
        rfi, rplay, rspd = frame_controls("roiv", len(frames))
        rf = frames[rfi]
        rmask = reg["mask_per_frame"].get(rf, mask0) if R.get("link") else mask0
        base_img = cc.stretch8(reg["reg_ca"].get(rf, reg["reg_mem"].get(rf, reg["reg_mem"][frames[0]])),
                               clahe=True)
        disp = np.dstack([base_img] * 3).astype(float)
        inside = np.isin(rmask, roi_ids)
        disp[find_boundaries(rmask, mode="outer")] = [190, 60, 60]
        disp[inside] = 0.55 * disp[inside] + 0.45 * np.array([0, 200, 220])
        disp = np.clip(disp, 0, 255).astype(np.uint8)
        disp[y1:y2, x1:x1 + 2] = [255, 235, 0]; disp[y1:y2, x2 - 2:x2] = [255, 235, 0]
        disp[y1:y1 + 2, x1:x2] = [255, 235, 0]; disp[y2 - 2:y2, x1:x2] = [255, 235, 0]
        st.image(disp, caption=f"ROI rectangle (yellow); intersecting cells (cyan), calcium frame {rf}",
                 width="stretch")
        frame_loop("roiv", rfi, rplay, rspd, len(frames))
    else:
        st.info("No ROI set. Drag a rectangle in the panel above to restrict the analysis to a "
                "region; the traces are already extracted, so it applies straight away.")

# ---------------------------------------------------------- Traces
def render_traces(df, sfx):
    cells = [c for c in df.columns if c.startswith("Cell_")]
    st.subheader(f"Traces before and after normalization ({len(cells)} cells)")
    tfi, tplay, tspd = frame_controls("trc" + sfx, len(frames))
    tf = frames[tfi]

    # matching RAW (pre-normalization) columns for the same cells
    raw_cols = [c for c in cells if c in raw_df.columns]
    raw_sub = raw_df[["Frame"] + raw_cols]
    base_rows = R.get("base") or []               # shared window, empty for per-cell methods
    f0_floor = float(R.get("f0_floor", 1.0))
    _meth = R.get("dff_method", "percell")
    # the exact per-cell F0i that compute_dff0 used for this run
    f0i = R.get("f0i") or cc.f0_per_cell(raw_df, method=_meth, floor=f0_floor)
    f0_tbl = pd.DataFrame([
        {"cell": int(c.split("_")[1]),
         "F0i_used": float(f0i[c][0]),
         "baseline_frames": ",".join(str(frames[r]) for r in f0i[c][1] if r < len(frames))}
        for c in raw_cols if c in f0i])

    p1, p2 = st.columns(2)
    with p1:
        fig0, ax0 = plt.subplots(figsize=(6, 3.6))
        for cn in raw_cols:
            ax0.plot(raw_sub["Frame"], raw_sub[cn], lw=0.4, alpha=0.22, color="dimgray")
        ax0.plot(raw_sub["Frame"], raw_sub[raw_cols].mean(axis=1), lw=2, color="black",
                 label="population mean")
        for r in base_rows:                       # baseline rows used for F_ref
            if r < len(frames):
                ax0.axvspan(frames[r] - 0.5, frames[r] + 0.5, color="gold", alpha=0.18)
        ax0.axvline(tf, color="crimson", lw=1.4, alpha=0.85)
        ax0.set_xlabel("Frame"); ax0.set_ylabel("raw intensity (a.u.)")
        ax0.set_title("BEFORE normalization (raw)", fontsize=10)
        ax0.legend(fontsize=7); ax0.grid(alpha=0.3)
        st.pyplot(fig0)
    with p2:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for cn in cells:
            ax.plot(df["Frame"], df[cn], lw=0.4, alpha=0.22, color="steelblue")
        ax.plot(df["Frame"], df[cells].mean(axis=1), lw=2, color="crimson",
                label="population mean")
        for r in base_rows:
            if r < len(frames):
                ax.axvspan(frames[r] - 0.5, frames[r] + 0.5, color="gold", alpha=0.18)
        ax.axvline(tf, color="black", lw=1.4, alpha=0.85)
        ax.set_xlabel("Frame"); ax.set_ylabel("ΔF/F0i")
        ax.set_title("AFTER normalization (ΔF/F0i)", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        st.pyplot(fig)
    if base_rows:
        st.caption(f"Gold band = the {len(base_rows)} shared baseline frames (method: {_meth}). "
                   f"Red/black line = current frame {tf}.")
    else:
        st.caption(f"Each cell uses its own baseline frames (method: {_meth}), so there is no single "
                   f"band to shade. See the F0i panel below. Red/black line = current frame {tf}.")

    # ---------------- F0i actually used, per cell ----------------
    with st.expander(f"F0i per cell  (ΔF/F0i = (F − F0i) / F0i,  floor {f0_floor:g})", expanded=False):
        if _meth in ("percell", "min"):
            st.markdown(
                "**Every cell is normalised by its own F0i.** F0i is that cell's baseline "
                + ("(the mean of its own lowest frames)." if _meth == "percell"
                   else "(its single lowest value).")
                + " Cells that are quiet early and cells that are quiet late each get their own "
                  "resting level, rather than sharing one window picked from the population average."
            )
        else:
            st.warning(
                f"The **{_meth}** method gives every cell the *same* baseline frames "
                f"({min(base_rows)}–{max(base_rows)} of {len(frames)}), so a cell that is active "
                f"during that window is normalised by an inflated baseline. Switch to **percell** "
                f"for a true per-cell F0i."
            )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline frames / cell",
                  len(base_rows) if base_rows else len(next(iter(f0i.values()))[1]))
        m2.metric("Median F0i", f"{f0_tbl['F0i_used'].median():.1f}")
        m3.metric("F0i range", f"{f0_tbl['F0i_used'].min():.0f}–{f0_tbl['F0i_used'].max():.0f}")
        m4.metric("Cells at floor", int((f0_tbl["F0i_used"] <= f0_floor + 1e-9).sum()))
        fg, ag = plt.subplots(1, 2, figsize=(11, 2.8))
        ag[0].hist(f0_tbl["F0i_used"].dropna(), bins=40, color="darkseagreen",
                   edgecolor="black", lw=0.4)
        ag[0].set_xlabel("F0i used (raw intensity)"); ag[0].set_ylabel("cells"); ag[0].grid(alpha=0.3)
        ag[0].set_title("F0i distribution across cells", fontsize=9)
        # which frames end up serving as baseline, across all cells
        _use = np.zeros(len(frames))
        for c in f0_tbl["cell"]:
            for r in f0i[f"Cell_{c}"][1]:
                if r < len(_use):
                    _use[r] += 1
        ag[1].bar(range(len(frames)), _use, color="steelblue")
        ag[1].set_xlabel("Frame"); ag[1].set_ylabel("cells using it")
        ag[1].set_title("how often each frame serves as baseline", fontsize=9); ag[1].grid(alpha=0.3)
        fg.tight_layout()
        st.pyplot(fg)
        st.dataframe(f0_tbl.round(2), width="stretch", height=200)
        st.download_button("⬇ per-cell F0i (CSV)", f0_tbl.to_csv(index=False).encode(),
                           "f0i_values.csv", "text/csv", key="dlf0" + sfx)

    # ---------------- dynamic background check ----------------
    with st.expander("Background subtraction check (step through the frames)", expanded=True):
        if not R.get("do_bg"):
            st.info("Background subtraction was **off** for this run. The regions below show what "
                    "would be sampled if you enable it in the sidebar and re-run.")
        try:
            ca_by_frame = {f: reg["reg_ca"][f] for f in frames if f in reg["reg_ca"]}
            boxes = cc.auto_bg_boxes(ca_by_frame[frames[0]])
            bgv = cc.background_values(ca_by_frame, boxes)
            bfi, bplay, bspd = frame_controls("bgchk" + sfx, len(frames))
            bf = frames[bfi]
            # Fixed display scale across all frames (no CLAHE): a background check has to show
            # ABSOLUTE intensity, and per-frame stretching/CLAHE would normalise away exactly
            # the frame-to-frame changes we are trying to see.
            _samp = np.concatenate([ca_by_frame[f][::4, ::4].ravel()
                                    for f in frames if f in ca_by_frame])
            _lo, _hi = np.percentile(_samp, [1, 99.5])
            _span = max(float(_hi - _lo), 1.0)

            def _fixed8(im):
                return (np.clip((im.astype(np.float32) - _lo) / _span, 0, 1) * 255).astype(np.uint8)

            g1, g2 = st.columns([1, 1])
            with g1:
                img = _fixed8(ca_by_frame[bf])
                disp = np.dstack([img] * 3)
                for (x1, y1, x2, y2) in boxes:
                    disp[y1:y2, x1:x1 + 2] = [0, 229, 255]; disp[y1:y2, x2 - 2:x2] = [0, 229, 255]
                    disp[y1:y1 + 2, x1:x2] = [0, 229, 255]; disp[y2 - 2:y2, x1:x2] = [0, 229, 255]
                st.image(disp, caption=f"3 background regions on calcium frame {bf} "
                                       f"(fixed intensity scale {_lo:.0f}–{_hi:.0f})",
                         width="stretch")
                _mean_bg, _per_box = bgv[bf]
                q = st.columns(4)
                q[0].metric(f"frame {bf} background", f"{_mean_bg:.1f}")
                for _j, _v in enumerate(_per_box):
                    q[_j + 1].metric(f"box {_j + 1}", f"{_v:.1f}")
            with g2:
                fb, ab = plt.subplots(figsize=(6, 3.4))
                fr = [f for f in frames if f in bgv]
                ab.plot(fr, [bgv[f][0] for f in fr], "-o", ms=3, color="teal", label="mean of 3")
                for j in range(len(boxes)):
                    ab.plot(fr, [bgv[f][1][j] for f in fr], lw=0.8, alpha=0.5,
                            label=f"box {j + 1}")
                ab.axvline(bf, color="crimson", lw=1.4)
                ab.set_xlabel("Frame"); ab.set_ylabel("background (a.u.)")
                ab.legend(fontsize=7); ab.grid(alpha=0.3)
                ab.set_title("background per frame", fontsize=10)
                st.pyplot(fb)
            vals = np.array([bgv[f][0] for f in fr])
            st.caption(f"Background drift over the recording: {vals.min():.1f} to {vals.max():.1f} "
                       f"(spread {vals.max() - vals.min():.1f}). The regions should stay dark and "
                       f"cell-free in every frame; step through with the controls above to check.")
            frame_loop("bgchk" + sfx, bfi, bplay, bspd, len(frames))
        except Exception as e:
            st.warning(f"Could not build the background check: {e}")

    st.subheader("Activity heatmap (cells × frames)")
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    arr = np.nan_to_num(df[cells].to_numpy(float)).T
    im = ax2.imshow(arr, aspect="auto", cmap="magma", vmin=0,
                    vmax=np.percentile(arr, 99) if arr.size else 1, interpolation="nearest")
    ax2.axvline(tfi, color="cyan", lw=1.5, alpha=0.9)                          # looping cursor
    ax2.set_xlabel("Frame"); ax2.set_ylabel("Cell")
    fig2.colorbar(im, ax=ax2, label="ΔF/F0i")
    st.pyplot(fig2)
    frame_loop("trc" + sfx, tfi, tplay, tspd, len(frames))


with T[TAB_TRC]:
    roi_split(render_traces, "trc")

# ---------------------------------------------------------- Clustering
def render_clustering(df, sfx):
    st.subheader("Trace clustering (PCA → K-means)")
    cells = [c for c in df.columns if c.startswith("Cell_")]
    if len(cells) < 4:
        st.info("Need at least 4 cells to cluster.")
    else:
        cc1, cc2 = st.columns(2)
        _pca_max = int(max(5, min(200, len(cells) - 1, len(frames))))
        n_pca = cc1.slider("PCA features", 2, _pca_max, min(25, _pca_max), key="npca" + sfx,
                           help=f"Capped at {_pca_max} by the number of cells and frames.")
        k = cc2.slider("Number of clusters (k)", 2, int(max(2, min(30, len(cells)))), 4,
                       key="nclu" + sfx)
        cl = cc.cluster_traces(df, n_pca=n_pca, n_clusters=k)
        ids, labels, coords = cl["ids"], cl["labels"], cl["coords"]
        st.caption(f"PCA kept {cl['n_pca_used']} components "
                   f"({cl['explained_var']*100:.0f}% variance) · {cl['k']} clusters.")
        colors, cnames = cluster_palette(cl["k"])

        colL, colR = st.columns(2)
        # cluster map on tissue
        base_img = cc.stretch8(reg["reg_mem"][0], clahe=True)
        cmap_img = np.dstack([base_img] * 3).astype(float) * 0.35
        for cid, lab in zip(ids, labels):
            cmap_img[mask0 == cid] = colors[lab % len(colors)]
        with colL:
            st.image(np.clip(cmap_img, 0, 255).astype(np.uint8),
                     caption="Cells colored by cluster", width="stretch")
        # PCA scatter
        with colR:
            figp, axp = plt.subplots(figsize=(5.5, 5))
            for lab in range(cl["k"]):
                m = labels == lab
                axp.scatter(coords[m, 0], coords[m, 1], s=14, alpha=0.7,
                            color=colors[lab % len(colors)] / 255, label=f"C{lab} (n={m.sum()})")
            axp.set_xlabel("PC1"); axp.set_ylabel("PC2"); axp.legend(fontsize=8); axp.grid(alpha=0.3)
            st.pyplot(figp)
        # cluster-average traces
        figc, axc = plt.subplots(figsize=(11, 4))
        A = np.nan_to_num(df[cells].to_numpy(float)).T
        idmap = {c: i for i, c in enumerate(ids)}
        for lab in range(cl["k"]):
            rows = [idmap[c] for c, l in zip(ids, labels) if l == lab]
            if rows:
                axc.plot(df["Frame"], A[rows].mean(0), lw=2, color=colors[lab % len(colors)] / 255,
                         label=f"Cluster {lab} (n={len(rows)})")
        axc.set_xlabel("Frame"); axc.set_ylabel("mean ΔF/F0i"); axc.legend(); axc.grid(alpha=0.3)
        axc.set_title("Cluster-average traces")
        st.pyplot(figc)

        cluster_df = pd.DataFrame({"cell_id": ids, "cluster": labels})
        if sfx.endswith("_all"):
            ss["cluster_df"] = cluster_df      # the all-cell run feeds the Downloads tab
        st.download_button("⬇ cluster assignments (CSV)", cluster_df.to_csv(index=False).encode(),
                           "cluster_assignments.csv", "text/csv", key="dlclu" + sfx)

        # ---------- AI story from the clusters (Amazon Bedrock, open-source model) ----------
        st.divider()
        st.markdown("#### 🧠 AI findings from the clusters — Amazon Bedrock (open-source model)")
        import cafin_ai
        from scipy.signal import find_peaks as _fp
        from skimage.measure import regionprops as _rp
        A2 = np.nan_to_num(df[cells].to_numpy(float))              # frames x cells
        nfr = A2.shape[0]
        thirds = np.array_split(np.arange(nfr), 3)
        peakc = A2.max(0); meanc = A2.mean(0)
        npk = [int(len(_fp(A2[:, j], height=peak_thr, distance=2)[0])) for j in range(A2.shape[1])]
        cent = {int(p.label): (round(float(p.centroid[1]), 1), round(float(p.centroid[0]), 1))
                for p in _rp(mask0)}
        clist = []
        for lab in range(cl["k"]):
            sel = np.where(labels == lab)[0]
            if not len(sel):
                continue
            cids = [int(ids[j]) for j in sel]
            sub = A2[:, sel]
            if len(sel) >= 2:
                cm = np.corrcoef(sub.T); iu = np.triu_indices_from(cm, 1)
                sync = float(np.nanmean(cm[iu]))
            else:
                sync = float("nan")
            cxy = [cent[c] for c in cids if c in cent]
            clist.append(dict(
                cluster=lab, color=cnames[lab], n_cells=int(len(sel)),
                mean_peak_dff0=round(float(peakc[sel].mean()), 2),
                mean_dff0=round(float(meanc[sel].mean()), 3),
                active_fraction=round(float((peakc[sel] > peak_thr).mean()), 2),
                mean_transients_per_cell=round(float(np.mean([npk[j] for j in sel])), 2),
                activity_early=round(float(sub[thirds[0]].mean()), 3),
                activity_mid=round(float(sub[thirds[1]].mean()), 3),
                activity_late=round(float(sub[thirds[2]].mean()), 3),
                within_cluster_sync_r=(round(sync, 3) if sync == sync else None),
                mean_centroid_xy=([cxy and round(float(np.mean([p[0] for p in cxy])), 1),
                                   cxy and round(float(np.mean([p[1] for p in cxy])), 1)] if cxy else None)))
        payload = dict(dataset=os.path.basename(trial.rstrip("/\\")), method=R.get("mode"),
                       n_frames=int(nfr), n_cells=int(len(ids)), n_clusters=int(cl["k"]),
                       pca_components=int(cl["n_pca_used"]),
                       pca_variance_explained=round(float(cl["explained_var"]), 2),
                       peak_threshold_dff0=float(peak_thr), image_size=list(mask0.shape),
                       clusters=clist)
        # ---- full per-cluster/population time-series across ALL frames (entire context) ----
        def _ds(arr, cap=150):
            arr = np.asarray(arr, float)
            if len(arr) <= cap:
                return [round(float(x), 3) for x in arr]
            idx = np.linspace(0, len(arr) - 1, cap).astype(int)
            return [round(float(arr[i]), 3) for i in idx]

        cluster_series = []
        for lab in range(cl["k"]):
            sel = np.where(labels == lab)[0]
            if len(sel):
                cluster_series.append(dict(cluster=lab, color=cnames[lab], n_cells=int(len(sel)),
                                           mean_trace_dff0=_ds(A2[:, sel].mean(1))))
        full_payload = dict(dataset=os.path.basename(trial.rstrip("/\\")), method=R.get("mode"),
                            n_frames=int(nfr), n_cells=int(len(ids)), n_clusters=int(cl["k"]),
                            peak_threshold_dff0=float(peak_thr),
                            frames_downsampled_to=min(nfr, 150),
                            population_mean_dff0=_ds(A2.mean(1)),
                            active_fraction_per_frame=_ds((A2 > peak_thr).mean(1)),
                            cluster_summary=clist, cluster_mean_traces=cluster_series)

        st.markdown("**Background / context** (sent to the model to ground its interpretation):")
        background = st.text_area(
            "Describe the experiment", key="ai_background" + sfx,
            placeholder="e.g. 3 dpf zebrafish larval fin; Latrunculin A 200 µM in 1% DMSO applied "
                        "immediately before imaging; GCaMP calcium indicator; confocal, ~1 frame/… s, "
                        "90 frames; membrane channel for segmentation…",
            height=90, label_visibility="collapsed")

        aci1, aci2 = st.columns([2, 1])
        ai_model = aci1.selectbox("Open-source model (Bedrock)", cafin_ai.MODEL_CHOICES, index=0,
                                  key="aimodel" + sfx,
                                  help="Llama / Mixtral / DeepSeek via Bedrock Converse API. "
                                       "Needs AWS credentials + model access enabled.")
        ai_region = aci2.text_input("AWS region", cafin_ai.DEFAULT_REGION, key="airegion" + sfx)
        bcol0, bcol1, bcol2 = st.columns([1, 2, 2])
        if bcol0.button("🔌 Test connection", key="aiconn" + sfx):
            ss["ai_conn" + sfx] = cafin_ai.check_credentials(ai_region)
        if bcol1.button("✍  Story from clusters", type="primary", key="aistory" + sfx):
            with st.spinner(f"Asking {ai_model}…"):
                ok, text = cafin_ai.interpret_clusters(payload, model_id=ai_model, region=ai_region,
                                                       background=background)
            ss["ai_story" + sfx] = text if ok else None
            ss["ai_err" + sfx] = None if ok else text
        if bcol2.button("🎞  Full temporal analysis (all frames)", key="aifull" + sfx):
            with st.spinner(f"Asking {ai_model} over the full time-series…"):
                ok, text = cafin_ai.interpret_clusters(full_payload, model_id=ai_model, region=ai_region,
                                                       background=background, full=True, max_tokens=1600)
            ss["ai_story_full" + sfx] = text if ok else None
            ss["ai_err" + sfx] = None if ok else text
        if ss.get("ai_conn" + sfx):
            okc, msgc = ss["ai_conn" + sfx]
            (st.success if okc else st.error)(msgc)
        if ss.get("ai_err" + sfx):
            st.error(ss["ai_err" + sfx])
        if ss.get("ai_story" + sfx):
            st.markdown("##### Findings — cluster snapshot")
            st.markdown(ss["ai_story" + sfx])
        if ss.get("ai_story_full" + sfx):
            st.markdown("##### Findings — full time-course (all frames)")
            st.markdown(ss["ai_story_full" + sfx])
        if ss.get("ai_story" + sfx) or ss.get("ai_story_full" + sfx):
            st.caption(f"Generated by {ai_model} (open-source) via Amazon Bedrock. Treat as "
                       "hypotheses, not conclusions.")
        # ---------- follow-up questions, with conversation memory ----------
        st.divider()
        st.markdown("##### 💬 Ask about these clusters")
        chat_key = "ai_chat" + sfx
        if chat_key not in ss:
            ss[chat_key] = []
        SUGGEST = [
            ("Initiators / followers",
             "Identify initiators and followers from the calcium traces after the PCA "
             "clustering. Which cluster is the initiator, and which follow it? Cite the "
             "timing numbers you used."),
            ("Most synchronized",
             "Which clusters are the most and least synchronized, and what could explain "
             "that difference?"),
            ("Wave propagation",
             "Is there evidence of a wave propagating between clusters? Use the cluster "
             "positions and their activity timing."),
            ("What next?",
             "What experiment or analysis would best test the leading hypothesis from "
             "these clusters?"),
        ]
        _cols = st.columns(len(SUGGEST))
        asked = None
        for _i, (_lbl, _q) in enumerate(SUGGEST):
            if _cols[_i].button(_lbl, key=f"sug{_i}{sfx}", help=_q, width="stretch"):
                asked = _q

        for _m in ss[chat_key]:
            with st.chat_message(_m["role"]):
                st.markdown(_m["text"])

        with st.form(key="chatform" + sfx, clear_on_submit=True):
            _typed = st.text_input("Your question", placeholder=
                                   "e.g. Which cluster is the initiator, and why?",
                                   label_visibility="collapsed")
            _sent = st.form_submit_button("Ask")
        _question = asked or (_typed if _sent and _typed.strip() else None)
        if _question:
            _hist = list(ss[chat_key])
            ss[chat_key].append({"role": "user", "text": _question})
            with st.spinner(f"Asking {ai_model}…"):
                _ok, _ans = cafin_ai.chat(full_payload, _hist, _question, model_id=ai_model,
                                          region=ai_region, background=background)
            ss[chat_key].append({"role": "assistant",
                                 "text": _ans if _ok else f"⚠️ {_ans}"})
            st.rerun()
        if ss[chat_key] and st.button("Clear conversation", key="clrchat" + sfx):
            ss[chat_key] = []
            st.rerun()

        with st.expander("data sent to the model"):
            st.write("Cluster snapshot payload:"); st.json(payload)
            st.write("Full time-series payload:"); st.json(full_payload)


with T[TAB_CLU]:
    roi_split(render_clustering, "clu")

# ---------------------------------------------------------- Tracking
if TAB_TRK in T:
  with T[TAB_TRK]:
    if R["handling"] == "tracking":
        link = R.get("link")
        if link and R.get("tstats"):
            ts = R["tstats"]
            m0, m1, m2 = st.columns(3)
            m0.metric("Tracks", ts["n_tracks"])
            m1.metric("Full-coverage tracks", ts["full_coverage"])
            m2.metric("New cells appeared", ts["n_new"])
            st.caption("Segment-every-frame + link (Hungarian matching + gap closing). Each color = "
                       "one stable global cell id; consistent color across frames = a followed cell.")
        st.subheader("Cell tracking — stable IDs across frames" if link
                     else "Cell tracking — mask follows tissue deformation")
        memsrc = reg["reg_mem"] if link else reg["raw_mem"]
        nT = len(frames)
        if ss.get("trk_idx") is None:
            ss["trk_idx"] = nT - 1
        t1, t2, t3, t4 = st.columns([1, 1, 2, 2])
        if t1.button("◀ Prev", key="trk_prev"):
            ss["trk_idx"] = (ss["trk_idx"] - 1) % nT
        if t2.button("Next ▶", key="trk_next"):
            ss["trk_idx"] = (ss["trk_idx"] + 1) % nT
        trk_play = t3.toggle("▶ Auto-loop", key="trk_play")
        trk_speed = t4.slider("loop speed (s/frame)", 0.05, 1.0, 0.05, 0.05, key="trk_speed")
        tidx = st.slider("Frame", 0, nT - 1, int(ss["trk_idx"]), key="trk_slider")
        ss["trk_idx"] = tidx
        ftrk = frames[tidx]
        base_img = cc.stretch8(memsrc.get(ftrk, memsrc[frames[0]]), clahe=True)
        tracked = reg["mask_per_frame"].get(ftrk, mask0)
        if link:                                    # color each cell by its global id (stable hue)
            disp = np.dstack([base_img] * 3).astype(float) * 0.4
            cols = (plt.get_cmap("tab20")(np.linspace(0, 1, 20)) * 255)[:, :3]
            for cid in np.unique(tracked):
                if cid > 0:
                    disp[tracked == cid] = cols[int(cid) % 20]
            a, b = st.columns(2)
            a.image(np.clip(disp, 0, 255).astype(np.uint8),
                    caption=f"Tracked cells on frame {ftrk} (color = global id)", width="stretch")
            dsn = np.dstack([base_img] * 3); dsn[find_boundaries(tracked, mode="outer")] = [0, 255, 0]
            b.image(dsn, caption=f"Tracked boundaries on frame {ftrk}", width="stretch")
        else:
            dt = np.dstack([base_img] * 3); dt[find_boundaries(tracked, mode="outer")] = [0, 255, 0]
            dsn = np.dstack([base_img] * 3); dsn[find_boundaries(mask0, mode="outer")] = [255, 80, 0]
            a, b = st.columns(2)
            a.image(dt, caption=f"TRACKED mask on frame {ftrk} (green)", width="stretch")
            b.image(dsn, caption=f"STATIC frame-0 mask on frame {ftrk} (orange)", width="stretch")
        if trk_play:                                    # auto-loop through frames
            time.sleep(trk_speed)
            ss["trk_idx"] = (tidx + 1) % nT
            st.rerun()
    else:
        st.info("Pick **Cell tracking** in the sidebar to follow cells across frames.")

# ---------------------------------------------------------- Statistics
def render_stats(df, sfx):
    st_, dist_ = cc.metrics(df, threshold=peak_thr)

    # ---------- per-cell peak dynamics, one point per cell ----------
    st.subheader("Peak dynamics")
    fi = st.number_input("Frame interval (minutes per frame; leave at 1 to report in frames)",
                         0.001, 60.0, 1.0, step=0.1, key="fint" + sfx)
    feats = cc.peak_features(df, threshold=peak_thr, frame_interval=fi)
    unit = "min" if abs(fi - 1.0) > 1e-9 else "frames"
    panels = [("n_peaks", "# peaks", ""),
              ("t_first_peak", "1$^{st}$ peak", unit),
              ("auc", "A.U.C.", ""),
              ("amplitude", "Amplitude ($\\Delta$F/F$_0$)", ""),
              ("fwhm", "F.W.H.M.", unit),
              ("dt_peak", "$\\Delta t_{peak}$", unit)]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 6, figsize=(15, 3.8))
    for ax, (col, title, un) in zip(axes, panels):
        vals = feats[col].to_numpy(float)
        vals = vals[~np.isnan(vals)]
        if vals.size:
            ax.boxplot(vals, widths=0.55, showfliers=False,
                       boxprops=dict(color="0.35"), medianprops=dict(color="0.35"),
                       whiskerprops=dict(color="0.35"), capprops=dict(color="0.35"))
            ax.scatter(rng.normal(1, 0.055, vals.size), vals, s=16, alpha=0.4,
                       color="0.45", edgecolors="none", zorder=3)
        ax.set_ylabel(f"{title} ({un})" if un else title, fontsize=10)
        ax.set_xticks([])
        ax.yaxis.grid(True, ls="--", alpha=0.55)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    n_ok = int(feats["n_peaks"].gt(0).sum())
    fig.suptitle(f"{len(feats)} cells  ({n_ok} with at least one detected peak, "
                 f"threshold {peak_thr} ΔF/F0i)", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    st.download_button("⬇ per-cell peak features (CSV)", feats.to_csv(index=False).encode(),
                       "peak_features.csv", "text/csv", key="dlfeat" + sfx)
    st.caption("Box = median and quartiles, whiskers = 1.5 IQR, one grey point per cell. "
               "Cells with no detected peak are excluded from the peak-shape panels.")

    st.subheader("Tissue-level metrics")
    st.dataframe(pd.DataFrame([st_]).T.rename(columns={0: "value"}), width="stretch")

    st.subheader("Fraction of cells above threshold, per frame")
    cells = [c for c in df.columns if c.startswith("Cell_")]
    frac = (df[cells].to_numpy(float) > peak_thr).mean(axis=1)
    fig3, ax3 = plt.subplots(figsize=(11, 3.5))
    ax3.plot(df["Frame"], frac, "-o", ms=3)
    ax3.set_ylim(0, 1); ax3.set_xlabel("Frame"); ax3.set_ylabel("fraction active"); ax3.grid(alpha=0.3)
    st.pyplot(fig3)


with T[TAB_STA]:
    roi_split(render_stats, "sta")

# ---------------------------------------------------------- Downloads
with T[TAB_DL]:
    # ------------------------------------------------ save straight to a folder
    st.subheader("Save results to a folder")
    if "save_dir" not in ss:
        ss["save_dir"] = trial + "_output"
    sd1, sd2 = st.columns([1, 3])
    if sd1.button("📂 Choose folder…", width="stretch"):
        _sp = pick_folder(ss["save_dir"], "Select a folder to save the results into")
        if _sp:
            ss["save_dir"] = os.path.normpath(_sp)
            st.rerun()
    _sd = sd2.text_input("Output folder", value=ss["save_dir"])
    if _sd and os.path.normpath(_sd) != os.path.normpath(ss["save_dir"]):
        ss["save_dir"] = os.path.normpath(_sd)

    # what can be saved from the current results
    _avail = {
        "Raw traces (all_cells_raw.csv)": ("all_cells_raw.csv", "csv", raw_df),
        "ΔF/F0i traces (cells_normalized.csv)": ("cells_normalized.csv", "csv", dff_df),
        "Metrics (metrics.csv)": ("metrics.csv", "csv", pd.DataFrame([stats])),
        "Segmentation mask (mask_0.tiff)": ("mask_0.tiff", "tiff", mask0),
    }
    if roi_ids:
        _keep = ["Frame"] + [f"Cell_{i}" for i in roi_ids if f"Cell_{i}" in dff_all.columns]
        _avail["ROI-only ΔF/F0i (roi_cells_normalized.csv)"] = ("roi_cells_normalized.csv", "csv",
                                                               dff_all[_keep])
    if ss.get("cluster_df") is not None:
        _avail["Cluster assignments (cluster_assignments.csv)"] = ("cluster_assignments.csv", "csv",
                                                                   ss["cluster_df"])
    if R.get("link") and reg.get("mask_per_frame"):
        _avail["Tracked masks (tracked_masks.tiff)"] = ("tracked_masks.tiff", "stack", None)
    if ss.get("gif_path") and os.path.exists(ss["gif_path"]):
        _avail["Registration overlay GIF"] = ("registration_overlay.gif", "copy", ss["gif_path"])
    for _k, _lbl in (("ai_story", "AI findings — clusters (ai_findings_clusters.md)"),
                     ("ai_story_full", "AI findings — full time-course (ai_findings_timecourse.md)")):
        if ss.get(_k):
            _avail[_lbl] = (_lbl.split("(")[-1].rstrip(")"), "text", ss[_k])

    picks = st.multiselect("Pick what to save", list(_avail), default=list(_avail))
    if st.button("💾 Save selected", type="primary", disabled=not picks):
        try:
            os.makedirs(ss["save_dir"], exist_ok=True)
            written = []
            for name in picks:
                fname, kind, obj = _avail[name]
                dest = os.path.join(ss["save_dir"], fname)
                if kind == "csv":
                    obj.to_csv(dest, index=False)
                elif kind == "tiff":
                    import tifffile
                    tifffile.imwrite(dest, np.asarray(obj).astype(np.uint16))
                elif kind == "stack":
                    import tifffile
                    tifffile.imwrite(dest, np.stack([reg["mask_per_frame"][f] for f in frames
                                                     if f in reg["mask_per_frame"]]).astype(np.uint16))
                elif kind == "text":
                    open(dest, "w", encoding="utf-8").write(obj)
                elif kind == "copy":
                    import shutil
                    shutil.copyfile(obj, dest)
                written.append(fname)
            st.success(f"Saved {len(written)} file(s) to {ss['save_dir']}")
            st.caption(", ".join(written))
        except Exception as e:
            st.error(f"Could not save: {e}")

    # ------------------------------------------------ browser downloads
    st.divider()
    st.subheader("Or download individually")
    st.download_button("⬇ raw traces (CSV)", raw_df.to_csv(index=False).encode(),
                       "all_cells_raw.csv", "text/csv")
    st.download_button("⬇ ΔF/F0i traces — current cell set (CSV)", dff_df.to_csv(index=False).encode(),
                       "cells_normalized.csv", "text/csv")
    if roi_ids:
        keep = ["Frame"] + [f"Cell_{i}" for i in roi_ids if f"Cell_{i}" in dff_all.columns]
        st.download_button("⬇ ROI-only ΔF/F0i (CSV)", dff_all[keep].to_csv(index=False).encode(),
                           "roi_cells_normalized.csv", "text/csv")
    st.download_button("⬇ metrics (CSV)", pd.DataFrame([stats]).to_csv(index=False).encode(),
                       "metrics.csv", "text/csv")
    if "gif_path" in ss and os.path.exists(ss["gif_path"]):
        with open(ss["gif_path"], "rb") as fh:
            st.download_button("⬇ overlay GIF", fh.read(), "registration_overlay.gif", "image/gif")
