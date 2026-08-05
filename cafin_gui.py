"""
CAFIN Calcium Analysis GUI  (Streamlit)
=======================================
Motion correction -> segmentation -> ΔF/F0 -> statistics, all inline.

Run:
    streamlit run cafin_gui.py

Registration / analysis methods (choose one):
    * Rigid           OpenCV ECC (global rotation+translation)
    * Elastic         SimpleITK B-spline (non-rigid; the elastix-equivalent that runs here)
    * Cell tracking   the frame-0 mask is warped into every frame so each cell is followed
Extras: interactive ROI selection, frame navigation + auto-loop, PCA trace clustering.
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
st.sidebar.caption("Motion correction → segmentation → ΔF/F0 → statistics")

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
    dff_method = st.selectbox("ΔF/F0 baseline (F0)", ["lowest", "first", "last"], index=0)
    f0_floor = st.number_input("F0 floor (avoid divide-by-tiny)", 0.1, 100.0, 1.0)
    peak_thr = st.number_input("Peak threshold (ΔF/F0)", 0.0, 5.0, 0.5, 0.1)

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
            rgb = np.dstack([gray] * 3)
            try:
                import plotly.graph_objects as go
                st.caption("Use the **box-select** cursor (default) and **drag a rectangle** on the "
                           "image. Drag again to replace it. Then click **Run analysis**.")
                step = max(3, W // 120)                       # invisible selectable grid
                yy, xx = np.mgrid[0:H:step, 0:W:step]
                fig = go.Figure(go.Image(z=rgb))
                fig.add_trace(go.Scattergl(x=xx.ravel(), y=yy.ravel(), mode="markers",
                                           marker=dict(size=step, opacity=0), hoverinfo="skip",
                                           showlegend=False))
                if ss.get("roi_box"):
                    x1, y1, x2, y2 = ss["roi_box"]
                    fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2,
                                  line=dict(color="#ffeb00", width=2), fillcolor="rgba(255,235,0,0.15)")
                fig.update_xaxes(visible=False, range=[0, W]); fig.update_yaxes(visible=False, range=[H, 0])
                fig.update_layout(dragmode="select", margin=dict(l=0, r=0, t=0, b=0),
                                  height=min(560, int(560 * H / W)) if W else 512)
                ev = st.plotly_chart(fig, on_select="rerun", selection_mode="box",
                                     width="stretch", key="roi_plot")
                pts = []
                try:
                    pts = ev["selection"]["points"]
                except Exception:
                    try:
                        pts = ev.selection.points
                    except Exception:
                        pts = []
                if pts:
                    xs = [p["x"] for p in pts]; yr = [p["y"] for p in pts]
                    ss["roi_box"] = (int(max(0, min(xs))), int(max(0, min(yr))),
                                     int(min(W, max(xs))), int(min(H, max(yr))))
                if ss.get("roi_box"):
                    b = ss["roi_box"]
                    st.success(f"ROI set — x[{b[0]},{b[2]}] y[{b[1]},{b[3]}]. Click **Run analysis**.")
                else:
                    st.info("No box drawn yet — drag a rectangle on the image above.")
            except Exception as e:                           # plotly unavailable -> numeric fallback
                st.error(f"Interactive selector unavailable ({e}). Enter ROI bounds manually:")
                cA = st.columns(4)
                x1 = cA[0].number_input("x1", 0, W, W // 4); y1 = cA[1].number_input("y1", 0, H, H // 4)
                x2 = cA[2].number_input("x2", 0, W, 3 * W // 4); y2 = cA[3].number_input("y2", 0, H, 3 * H // 4)
                ss["roi_box"] = (x1, y1, x2, y2)
                st.image(_draw_box(gray, ss["roi_box"]), width="stretch")


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
                     f"Extracting ΔF/F0 …")
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
            st.write("Extracting traces + ΔF/F0 …")
            raw_df = cc.extract_traces(reg, mask0, handling, bg=do_bg)
            eff_handling = handling

        roi_ids = None
        if roi_on and ss.get("roi_box"):
            roi_ids = cc.roi_cell_ids(mask0, ss["roi_box"])
            st.write(f"→ ROI: {len(roi_ids)} of {int(mask0.max())} cells intersect the rectangle.")

        dff_df, base = cc.compute_dff0(raw_df, method=dff_method, floor=f0_floor)
        status.update(label="Done", state="complete", expanded=False)
    prog.empty()
    ss["res"] = dict(reg=reg, mask0=mask0, raw=raw_df, dff=dff_df, base=base, mode=choice,
                     method=method, handling=eff_handling, link=link_tracking, tstats=tstats,
                     mem_base=mem_base, roi_ids=roi_ids,
                     roi_box=ss.get("roi_box") if roi_on else None, peak_thr=peak_thr)
    ss["frame_idx"] = None  # reset navigation

if "res" not in ss:
    st.info("Set the trial folder + method in the sidebar, then click **Run analysis**.")
    st.stop()

R = ss["res"]
reg, mask0, raw_df, dff_all = R["reg"], R["mask0"], R["raw"], R["dff"]
frames = reg["frames"]
peak_thr = R["peak_thr"]

# ------------------------ ROI-aware cell set
roi_ids = R.get("roi_ids")
if roi_ids:
    pick = st.radio("Cell set", [f"ROI cells ({len(roi_ids)})", "All cells"], horizontal=True, index=0)
    if pick.startswith("ROI"):
        keep = ["Frame"] + [f"Cell_{i}" for i in roi_ids if f"Cell_{i}" in dff_all.columns]
        dff_df = dff_all[keep]
    else:
        dff_df = dff_all
else:
    dff_df = dff_all
stats, dist = cc.metrics(dff_df, threshold=peak_thr)

# headline metrics
c = st.columns(4)
c[0].metric("Cells", stats["cells"])
c[1].metric("Active %", stats["active_pct"])
c[2].metric("Mean peak ΔF/F0", stats["mean_peak_dff0"])
c[3].metric("Sync (r)", stats["temporal_sync_r"])

tabs = st.tabs(["🎞 Registration + movie", "🧫 Segmentation", "🎯 ROI", "📈 Traces / ΔF/F0",
                "🧩 Clustering", "🎯 Tracking", "📊 Statistics", "⬇ Downloads"])

# ---------------------------------------------------------- Registration + movie
with tabs[0]:
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
with tabs[1]:
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
with tabs[2]:
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
        st.info("Enable **Restrict analysis to a region of interest** in the sidebar, draw a "
                "rectangle above the Run button, then re-run.")

# ---------------------------------------------------------- Traces
with tabs[3]:
    st.subheader("Per-cell ΔF/F0 traces")
    cells = [c for c in dff_df.columns if c.startswith("Cell_")]
    tfi, tplay, tspd = frame_controls("trc", len(frames))
    tf = frames[tfi]
    fig, ax = plt.subplots(figsize=(11, 4))
    for cn in cells:
        ax.plot(dff_df["Frame"], dff_df[cn], lw=0.4, alpha=0.25, color="steelblue")
    ax.plot(dff_df["Frame"], dff_df[cells].mean(axis=1), lw=2.2, color="crimson", label="population mean")
    for f in R["base"]:
        ax.axvspan(f - 0.5, f + 0.5, color="gold", alpha=0.12)
    ax.axvline(tf, color="black", lw=1.5, alpha=0.8, label=f"frame {tf}")     # looping cursor
    ax.set_xlabel("Frame"); ax.set_ylabel("ΔF/F0"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("gold = baseline (F0) frames")
    st.pyplot(fig)

    st.subheader("Activity heatmap (cells × frames)")
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    arr = np.nan_to_num(dff_df[cells].to_numpy(float)).T
    im = ax2.imshow(arr, aspect="auto", cmap="magma", vmin=0,
                    vmax=np.percentile(arr, 99) if arr.size else 1, interpolation="nearest")
    ax2.axvline(tfi, color="cyan", lw=1.5, alpha=0.9)                          # looping cursor
    ax2.set_xlabel("Frame"); ax2.set_ylabel("Cell")
    fig2.colorbar(im, ax=ax2, label="ΔF/F0")
    st.pyplot(fig2)
    frame_loop("trc", tfi, tplay, tspd, len(frames))

# ---------------------------------------------------------- Clustering
with tabs[4]:
    st.subheader("Trace clustering (PCA → K-means)")
    cells = [c for c in dff_df.columns if c.startswith("Cell_")]
    if len(cells) < 4:
        st.info("Need at least 4 cells to cluster.")
    else:
        cc1, cc2 = st.columns(2)
        n_pca = cc1.slider("PCA features", 5, 30, 25)
        k = cc2.slider("Number of clusters (k)", 2, 8, 4)
        cl = cc.cluster_traces(dff_df, n_pca=n_pca, n_clusters=k)
        ids, labels, coords = cl["ids"], cl["labels"], cl["coords"]
        st.caption(f"PCA kept {cl['n_pca_used']} components "
                   f"({cl['explained_var']*100:.0f}% variance) · {cl['k']} clusters.")
        colors = (plt.get_cmap("tab10")(np.linspace(0, 1, 10)) * 255)[:, :3]

        colL, colR = st.columns(2)
        # cluster map on tissue
        base_img = cc.stretch8(reg["reg_mem"][0], clahe=True)
        cmap_img = np.dstack([base_img] * 3).astype(float) * 0.35
        for cid, lab in zip(ids, labels):
            cmap_img[mask0 == cid] = colors[lab % 10]
        with colL:
            st.image(np.clip(cmap_img, 0, 255).astype(np.uint8),
                     caption="Cells colored by cluster", width="stretch")
        # PCA scatter
        with colR:
            figp, axp = plt.subplots(figsize=(5.5, 5))
            for lab in range(cl["k"]):
                m = labels == lab
                axp.scatter(coords[m, 0], coords[m, 1], s=14, alpha=0.7,
                            color=colors[lab % 10] / 255, label=f"C{lab} (n={m.sum()})")
            axp.set_xlabel("PC1"); axp.set_ylabel("PC2"); axp.legend(fontsize=8); axp.grid(alpha=0.3)
            st.pyplot(figp)
        # cluster-average traces
        figc, axc = plt.subplots(figsize=(11, 4))
        A = np.nan_to_num(dff_df[cells].to_numpy(float)).T
        idmap = {c: i for i, c in enumerate(ids)}
        for lab in range(cl["k"]):
            rows = [idmap[c] for c, l in zip(ids, labels) if l == lab]
            if rows:
                axc.plot(dff_df["Frame"], A[rows].mean(0), lw=2, color=colors[lab % 10] / 255,
                         label=f"Cluster {lab} (n={len(rows)})")
        axc.set_xlabel("Frame"); axc.set_ylabel("mean ΔF/F0"); axc.legend(); axc.grid(alpha=0.3)
        axc.set_title("Cluster-average traces")
        st.pyplot(figc)

        cluster_df = pd.DataFrame({"cell_id": ids, "cluster": labels})
        ss["cluster_df"] = cluster_df          # made available to the Downloads tab
        st.download_button("⬇ cluster assignments (CSV)", cluster_df.to_csv(index=False).encode(),
                           "cluster_assignments.csv", "text/csv")

        # ---------- AI story from the clusters (Amazon Bedrock, open-source model) ----------
        st.divider()
        st.markdown("#### 🧠 AI findings from the clusters — Amazon Bedrock (open-source model)")
        import cafin_ai
        from scipy.signal import find_peaks as _fp
        from skimage.measure import regionprops as _rp
        CNAMES = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
        A2 = np.nan_to_num(dff_df[cells].to_numpy(float))          # frames x cells
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
                cluster=lab, color=CNAMES[lab % 10], n_cells=int(len(sel)),
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
                cluster_series.append(dict(cluster=lab, color=CNAMES[lab % 10], n_cells=int(len(sel)),
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
            "Describe the experiment", key="ai_background",
            placeholder="e.g. 3 dpf zebrafish larval fin; Latrunculin A 200 µM in 1% DMSO applied "
                        "immediately before imaging; GCaMP calcium indicator; confocal, ~1 frame/… s, "
                        "90 frames; membrane channel for segmentation…",
            height=90, label_visibility="collapsed")

        aci1, aci2 = st.columns([2, 1])
        ai_model = aci1.selectbox("Open-source model (Bedrock)", cafin_ai.MODEL_CHOICES, index=0,
                                  help="Llama / Mixtral / DeepSeek via Bedrock Converse API. "
                                       "Needs AWS credentials + model access enabled.")
        ai_region = aci2.text_input("AWS region", cafin_ai.DEFAULT_REGION)
        bcol0, bcol1, bcol2 = st.columns([1, 2, 2])
        if bcol0.button("🔌 Test connection"):
            ss["ai_conn"] = cafin_ai.check_credentials(ai_region)
        if bcol1.button("✍  Story from clusters", type="primary"):
            with st.spinner(f"Asking {ai_model}…"):
                ok, text = cafin_ai.interpret_clusters(payload, model_id=ai_model, region=ai_region,
                                                       background=background)
            ss["ai_story"] = text if ok else None
            ss["ai_err"] = None if ok else text
        if bcol2.button("🎞  Full temporal analysis (all frames)"):
            with st.spinner(f"Asking {ai_model} over the full time-series…"):
                ok, text = cafin_ai.interpret_clusters(full_payload, model_id=ai_model, region=ai_region,
                                                       background=background, full=True, max_tokens=1600)
            ss["ai_story_full"] = text if ok else None
            ss["ai_err"] = None if ok else text
        if ss.get("ai_conn"):
            okc, msgc = ss["ai_conn"]
            (st.success if okc else st.error)(msgc)
        if ss.get("ai_err"):
            st.error(ss["ai_err"])
        if ss.get("ai_story"):
            st.markdown("##### Findings — cluster snapshot")
            st.markdown(ss["ai_story"])
        if ss.get("ai_story_full"):
            st.markdown("##### Findings — full time-course (all frames)")
            st.markdown(ss["ai_story_full"])
        if ss.get("ai_story") or ss.get("ai_story_full"):
            st.caption(f"Generated by {ai_model} (open-source) via Amazon Bedrock. Treat as "
                       "hypotheses, not conclusions.")
        with st.expander("data sent to the model"):
            st.write("Cluster snapshot payload:"); st.json(payload)
            st.write("Full time-series payload:"); st.json(full_payload)

# ---------------------------------------------------------- Tracking
with tabs[5]:
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
with tabs[6]:
    st.subheader("Tissue-level metrics")
    st.dataframe(pd.DataFrame([stats]).T.rename(columns={0: "value"}), width="stretch")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ["amps", "intervals", "areas"],
                              ["Peak amplitude (ΔF/F0)", "Inter-peak interval (frames)", "Peak area"]):
        data = np.asarray(dist[key], float)
        if data.size:
            ax.violinplot(data, showmeans=True)
            ax.scatter(np.random.normal(1, 0.04, data.size), data, s=8, alpha=0.4, color="steelblue")
        ax.set_title(title); ax.set_xticks([])
    st.pyplot(fig)

    st.subheader("Fraction of cells above threshold, per frame")
    cells = [c for c in dff_df.columns if c.startswith("Cell_")]
    frac = (dff_df[cells].to_numpy(float) > peak_thr).mean(axis=1)
    fig3, ax3 = plt.subplots(figsize=(11, 3.5))
    ax3.plot(dff_df["Frame"], frac, "-o", ms=3)
    ax3.set_ylim(0, 1); ax3.set_xlabel("Frame"); ax3.set_ylabel("fraction active"); ax3.grid(alpha=0.3)
    st.pyplot(fig3)

# ---------------------------------------------------------- Downloads
with tabs[7]:
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
        "ΔF/F0 traces (cells_normalized.csv)": ("cells_normalized.csv", "csv", dff_df),
        "Metrics (metrics.csv)": ("metrics.csv", "csv", pd.DataFrame([stats])),
        "Segmentation mask (mask_0.tiff)": ("mask_0.tiff", "tiff", mask0),
    }
    if roi_ids:
        _keep = ["Frame"] + [f"Cell_{i}" for i in roi_ids if f"Cell_{i}" in dff_all.columns]
        _avail["ROI-only ΔF/F0 (roi_cells_normalized.csv)"] = ("roi_cells_normalized.csv", "csv",
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
    st.download_button("⬇ ΔF/F0 traces — current cell set (CSV)", dff_df.to_csv(index=False).encode(),
                       "cells_normalized.csv", "text/csv")
    if roi_ids:
        keep = ["Frame"] + [f"Cell_{i}" for i in roi_ids if f"Cell_{i}" in dff_all.columns]
        st.download_button("⬇ ROI-only ΔF/F0 (CSV)", dff_all[keep].to_csv(index=False).encode(),
                           "roi_cells_normalized.csv", "text/csv")
    st.download_button("⬇ metrics (CSV)", pd.DataFrame([stats]).to_csv(index=False).encode(),
                       "metrics.csv", "text/csv")
    if "gif_path" in ss and os.path.exists(ss["gif_path"]):
        with open(ss["gif_path"], "rb") as fh:
            st.download_button("⬇ overlay GIF", fh.read(), "registration_overlay.gif", "image/gif")
