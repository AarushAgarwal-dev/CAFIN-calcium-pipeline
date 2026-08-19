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
import io, os, glob, re, time, sys, platform, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw
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
    """Open a native folder picker and return ``(path, error_message)``.

    macOS uses Finder's AppleScript dialog first because Tk folder dialogs can
    fail when Streamlit is running from a background server process. Tk is kept
    as a fallback for Windows, Linux, and unusual macOS setups.
    """
    errors = []
    if platform.system() == "Darwin":
        try:
            safe_title = str(title).replace("\\", "\\\\").replace('"', '\\"')
            script = (f'try\nPOSIX path of (choose folder with prompt "{safe_title}")\n'
                      'on error number -128\nreturn ""\nend try')
            result = subprocess.run(["osascript", "-e", script], text=True,
                                    capture_output=True, timeout=90, check=False)
            if result.returncode == 0:
                selected = result.stdout.strip()
                return (os.path.normpath(selected), None) if selected else (None, None)
            errors.append(result.stderr.strip() or "macOS folder picker did not open")
        except Exception as exc:
            errors.append(f"macOS folder picker: {exc}")

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
        return (os.path.normpath(path), None) if path else (None, None)
    except Exception as exc:
        errors.append(f"Tk folder picker: {exc}")
    return None, "; ".join(errors) or "No folder picker is available"


def _as_rgb8(image):
    """Convert a grayscale/RGB image into a GIF-safe uint8 RGB frame."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.dstack([arr] * 3)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"GIF frames must be 2-D or RGB images, not shape {arr.shape}.")
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
    return np.nan_to_num(np.clip(arr, 0, 255), nan=0).astype(np.uint8)


def _stack_gif_frames(*images, gap=6):
    """Place image panels side by side, padding shorter panels with black."""
    panels = [_as_rgb8(im) for im in images]
    if not panels:
        raise ValueError("No image panels were supplied for the GIF.")
    height = max(im.shape[0] for im in panels)
    width = sum(im.shape[1] for im in panels) + gap * max(0, len(panels) - 1)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for panel in panels:
        out[:panel.shape[0], x:x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    return out


def _label_gif_frame(image, label):
    """Add a compact frame label without modifying the scientific image data."""
    pil = Image.fromarray(_as_rgb8(image))
    draw = ImageDraw.Draw(pil)
    draw.rectangle((4, 4, 10 + 7 * len(str(label)), 24), fill=(0, 0, 0))
    draw.text((8, 7), str(label), fill=(255, 255, 255))
    return np.asarray(pil)


def _gif_frame_ids(frame_ids, max_frames=240):
    """Keep full playback for short recordings and evenly sample long recordings."""
    ids = list(frame_ids)
    if len(ids) <= max_frames:
        return ids
    picks = np.linspace(0, len(ids) - 1, max_frames, dtype=int)
    return [ids[i] for i in np.unique(picks)]


def _gif_bytes(images, duration_s):
    """Encode GIF bytes in memory so downloads work on Windows and macOS."""
    frames = [_as_rgb8(im) for im in images]
    if not frames:
        raise ValueError("No usable frames were available for the GIF.")
    height = max(im.shape[0] for im in frames)
    width = max(im.shape[1] for im in frames)
    padded = []
    for frame in frames:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:frame.shape[0], :frame.shape[1]] = frame
        padded.append(Image.fromarray(canvas))
    buffer = io.BytesIO()
    padded[0].save(buffer, format="GIF", save_all=True, append_images=padded[1:],
                   duration=max(20, int(float(duration_s) * 1000)), loop=0, disposal=2)
    return buffer.getvalue(), len(padded)


def gif_download_control(key, filename, build_frames, duration_s, max_frames=240):
    """Prepare an in-memory GIF only on demand, then expose a download button."""
    cache = f"_gif_export_{key}"
    if st.button("🎞 Prepare GIF", key=cache + "_prepare",
                 help="Creates a GIF from the same frames used by this playback."):
        try:
            with st.spinner("Preparing GIF…"):
                images = build_frames()
                data, count = _gif_bytes(images, duration_s)
            ss[cache] = {"data": data, "count": count}
        except Exception as exc:
            ss.pop(cache, None)
            st.error(f"Could not prepare this GIF: {exc}")
    saved = ss.get(cache)
    if saved:
        st.download_button("⬇ Download GIF", saved["data"], file_name=filename,
                           mime="image/gif", key=cache + "_download")
        st.caption(f"GIF ready: {saved['count']} frame{'s' if saved['count'] != 1 else ''}. "
                   "Long recordings are sampled evenly to keep downloads manageable.")


def _figure_rgb(figure):
    """Render a Matplotlib figure into one RGB GIF frame and release it."""
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    image = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(height, width, 4)[..., :3].copy()
    plt.close(figure)
    return image


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
if sys.version_info >= (3, 13):
    st.sidebar.warning("Python 3.13 is not tested with every Cellpose/GPU build. "
                       "Python 3.11 is recommended; CPU analysis remains available.")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(APP_DIR, "demo_data")
if not os.path.exists(os.path.join(DEMO_DIR, "membrane", "demo_mem_0000.tif")):
    try:
        from demo_data import make_demo
        make_demo()
    except Exception:
        pass
if "trial_path" not in ss:
    ss["trial_path"] = DEMO_DIR

st.sidebar.markdown("### Data folder")
_b1, _b2, _b3 = st.sidebar.columns(3)
if _b1.button("📂 Browse…", width="stretch", help="Open a folder picker on this machine"):
    _p, _picker_error = pick_folder(ss["trial_path"], "Select the trial folder (with membrane/ and ca2/)")
    if _p:
        ss["trial_path"] = os.path.normpath(_p)
        st.rerun()
    elif _picker_error:
        st.sidebar.warning("The folder picker could not open. Paste the folder path below instead. "
                           f"Details: {_picker_error}")
if _b2.button("🔍 Find trials", width="stretch", help="Scan for folders containing membrane/ and ca2/"):
    found = find_trial_folders(APP_DIR) or find_trial_folders(os.path.dirname(APP_DIR))
    ss["found_trials"] = found
    if not found:
        st.sidebar.warning("No folders with membrane/ and ca2/ found nearby.")
if _b3.button("🧪 Demo", width="stretch", help="Load the included synthetic demonstration recording"):
    ss["trial_path"] = DEMO_DIR
    st.rerun()

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
    n_ca = cc.count_frames(caf, ca_base)
    if not mem_base or not ca_base:
        ok_data = False
        st.sidebar.error("No valid TIFF sequence found. Filenames must end in four digits, "
                         "for example sample_0000.tif.")
    elif n_frames != n_ca:
        ok_data = False
        st.sidebar.error(f"Channel frame counts do not match: membrane={n_frames}, calcium={n_ca}.")
    elif n_frames == 0:
        ok_data = False
        st.sidebar.error("The channel folders contain no readable .tif frames.")
    else:
        st.sidebar.success(f"membrane base: {mem_base}\n\ncalcium base: {ca_base}"
                           f"\n\nmatched frames: {n_frames}")
        if os.path.abspath(trial) == os.path.abspath(DEMO_DIR):
            st.sidebar.info("Synthetic demo data. Use it to learn the GUI only, not as evidence.")
else:
    st.sidebar.error("Choose a trial folder containing membrane/ and ca2/ subfolders, or click Demo.")

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

            def _preview_gif_frames():
                movie = []
                for _frame_no in _gif_frame_ids(range(n_frames)):
                    _mem_path = os.path.join(memf, f"{mem_base}{_frame_no:04d}.tif")
                    _ca_path = os.path.join(caf, f"{ca_base}{_frame_no:04d}.tif")
                    _mem = skio.imread(_mem_path) if os.path.exists(_mem_path) else None
                    _ca = skio.imread(_ca_path) if os.path.exists(_ca_path) else None
                    if _mem is None or _ca is None:
                        continue
                    _pair = _stack_gif_frames(cc.stretch8(_mem, clahe=True),
                                              cc.stretch8(_ca, clahe=True))
                    movie.append(_label_gif_frame(_pair, f"Frame {_frame_no}: membrane | calcium"))
                return movie

            gif_download_control("preview", "cafin_raw_preview.gif", _preview_gif_frames, pspd)
            frame_loop("prev", pfi, pplay, pspd, n_frames)
        except Exception as e:
            st.warning(f"Could not preview frames: {e}")

# =========================================================== RUN
if run:
    # GIFs are derived from the previous result set. Clear their in-memory download
    # caches before a new analysis so no stale GIF can be mistaken for new results.
    for _gif_cache_key in [k for k in ss if k.startswith("_gif_export_")]:
        del ss[_gif_cache_key]
    # Network results belong to the exact registered movie and cell traces.
    # Never expose a previous recording's graph after a new analysis runs.
    for _network_key in [k for k in ss if k.startswith(("_cellnet_", "_net_"))]:
        del ss[_network_key]
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
TAB_NET = "🕸 Network Analysis"
TAB_TRK, TAB_DL = "🎯 Tracking", "⬇ Downloads"
_names = [TAB_REG, TAB_SEG, TAB_ROI, TAB_TRC, TAB_STA, TAB_CLU, TAB_NET]
if R["handling"] == "tracking":                    # only present when tracking was used
    _names.append(TAB_TRK)
_names.append(TAB_DL)
T = dict(zip(_names, st.tabs(_names)))

# ---------------------------------------------------------- Registration + movie
with T[TAB_REG]:
    st.subheader("Overlay  (green = frame 0, magenta = moving, white = aligned)")
    fr = [f for f in frames if f != 0] or frames
    n = len(fr)
    ridx, rplay, rspeed = frame_controls("reg", n, start=n - 1)
    fsel = fr[ridx]

    first_mem = reg["raw_mem"][0]
    mem_before = cc.two_color_overlay(first_mem, reg["raw_mem"].get(fsel, first_mem))
    mem_after = cc.two_color_overlay(first_mem, reg["reg_mem"].get(fsel, first_mem))
    a, b = st.columns(2)
    a.image(mem_before, caption=f"Membrane BEFORE — frame {fsel}", width="stretch")
    b.image(mem_after, caption=f"Membrane AFTER — frame {fsel}", width="stretch")
    first_ca = reg["raw_ca"].get(0, reg["reg_ca"].get(0))
    ca_before = cc.two_color_overlay(first_ca, reg["raw_ca"].get(fsel, first_ca))
    ca_after = cc.two_color_overlay(first_ca, reg["reg_ca"].get(fsel, first_ca))
    a, b = st.columns(2)
    a.image(ca_before, caption=f"Ca²⁺ BEFORE — frame {fsel}", width="stretch")
    b.image(ca_after, caption=f"Ca²⁺ AFTER — frame {fsel}", width="stretch")

    def _registration_gif_frames():
        movie = []
        for _frame_no in _gif_frame_ids(fr):
            _mem_before = cc.two_color_overlay(first_mem, reg["raw_mem"].get(_frame_no, first_mem))
            _mem_after = cc.two_color_overlay(first_mem, reg["reg_mem"].get(_frame_no, first_mem))
            _ca_before = cc.two_color_overlay(first_ca, reg["raw_ca"].get(_frame_no, first_ca))
            _ca_after = cc.two_color_overlay(first_ca, reg["reg_ca"].get(_frame_no, first_ca))
            movie.append(_label_gif_frame(
                _stack_gif_frames(_stack_gif_frames(_mem_before, _mem_after),
                                  _stack_gif_frames(_ca_before, _ca_after)),
                f"Frame {_frame_no}: membrane before | after · Ca²⁺ before | after"))
        return movie

    gif_download_control("registration", "cafin_registration_overlay.gif",
                         _registration_gif_frames, rspeed)
    frame_loop("reg", ridx, rplay, rspeed, n)

# ---------------------------------------------------------- Segmentation
with T[TAB_SEG]:
    st.subheader(f"Cellpose segmentation — {int(mask0.max())} cells")
    mc1, mc2 = st.columns(2)
    mc1.image(cc.numbered_mask_overlay(reg["reg_mem"][0], mask0),
              caption="Frame-0 membrane with numbered cell ROIs", width="stretch")
    mc2.image(cc.cellpose_colored_mask(mask0),
              caption="Cellpose colored mask (direct label output)", width="stretch")

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

    def _segmentation_gif_frames():
        movie = []
        for _frame_no in _gif_frame_ids(frames):
            _mask = reg["mask_per_frame"].get(_frame_no, mask0) if R.get("link") else mask0
            _mem = reg["reg_mem"].get(_frame_no, reg["reg_mem"][frames[0]])
            _ca = reg["reg_ca"].get(_frame_no, _mem)
            _pair = _stack_gif_frames(cc.numbered_mask_overlay(_mem, _mask, clahe=True),
                                      cc.numbered_mask_overlay(_ca, _mask, clahe=True))
            movie.append(_label_gif_frame(_pair, f"Frame {_frame_no}: membrane | calcium"))
        return movie

    gif_download_control("segmentation", "cafin_segmentation_overlays.gif",
                         _segmentation_gif_frames, sspd)
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
        _roi_ids = list(R.get("roi_ids") or [])
        st.subheader(f"Region of interest — {len(_roi_ids)} cells intersect the rectangle")
        x1, y1, x2, y2 = R["roi_box"]
        rfi, rplay, rspd = frame_controls("roiv", len(frames))
        rf = frames[rfi]

        def _roi_display(_frame_no):
            rmask = reg["mask_per_frame"].get(_frame_no, mask0) if R.get("link") else mask0
            base_img = cc.stretch8(reg["reg_ca"].get(
                _frame_no, reg["reg_mem"].get(_frame_no, reg["reg_mem"][frames[0]])), clahe=True)
            display = np.dstack([base_img] * 3).astype(float)
            inside = np.isin(rmask, _roi_ids)
            display[find_boundaries(rmask, mode="outer")] = [190, 60, 60]
            display[inside] = 0.55 * display[inside] + 0.45 * np.array([0, 200, 220])
            display = np.clip(display, 0, 255).astype(np.uint8)
            h, w = display.shape[:2]
            xa, xb = sorted((max(0, min(w, int(x1))), max(0, min(w, int(x2)))))
            ya, yb = sorted((max(0, min(h, int(y1))), max(0, min(h, int(y2)))))
            if xb > xa and yb > ya:
                display[ya:yb, xa:min(xa + 2, xb)] = [255, 235, 0]
                display[ya:yb, max(xa, xb - 2):xb] = [255, 235, 0]
                display[ya:min(ya + 2, yb), xa:xb] = [255, 235, 0]
                display[max(ya, yb - 2):yb, xa:xb] = [255, 235, 0]
            return display

        disp = _roi_display(rf)
        st.image(disp, caption=f"ROI rectangle (yellow); intersecting cells (cyan), calcium frame {rf}",
                 width="stretch")

        def _roi_gif_frames():
            return [_label_gif_frame(_roi_display(_frame_no),
                                     f"Frame {_frame_no}: ROI cells in cyan")
                    for _frame_no in _gif_frame_ids(frames)]

        gif_download_control("roi", "cafin_roi_playback.gif", _roi_gif_frames, rspd)
        frame_loop("roiv", rfi, rplay, rspd, len(frames))
    else:
        st.info("No ROI set. Drag a rectangle in the panel above to restrict the analysis to a "
                "region; the traces are already extracted, so it applies straight away.")

# ---------------------------------------------------------- Traces
def render_traces(df, sfx):
    cells = [c for c in df.columns if c.startswith("Cell_")]
    st.subheader(f"Traces before and after normalization ({len(cells)} cells)")
    trace_layers = st.multiselect(
        "Choose what to show in trace plots",
        ["Individual cells", "Population mean", "Current-frame marker", "Baseline window"],
        default=["Individual cells", "Population mean", "Current-frame marker", "Baseline window"],
        key="trace_layers" + sfx,
        help="These controls change only the display, not the analysis or saved data.")
    shown_cells = st.multiselect(
        "Individual cells to display",
        cells, default=cells[:min(50, len(cells))], key="trace_cells" + sfx,
        help="Limit overplotting while retaining every cell in population summaries and metrics.")
    show_heatmap = st.checkbox("Show cell-by-frame activity heatmap", True,
                               key="trace_heatmap" + sfx)
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
        if "Individual cells" in trace_layers:
            for cn in [c for c in shown_cells if c in raw_cols]:
                ax0.plot(raw_sub["Frame"], raw_sub[cn], lw=0.65, alpha=0.35)
        if "Population mean" in trace_layers:
            ax0.plot(raw_sub["Frame"], raw_sub[raw_cols].mean(axis=1), lw=2, color="black",
                     label="population mean")
        if "Baseline window" in trace_layers:
            for r in base_rows:
                if r < len(frames):
                    ax0.axvspan(frames[r] - 0.5, frames[r] + 0.5, color="gold", alpha=0.18)
        if "Current-frame marker" in trace_layers:
            ax0.axvline(tf, color="crimson", lw=1.4, alpha=0.85)
        ax0.set_xlabel("Frame"); ax0.set_ylabel("raw intensity (a.u.)")
        ax0.set_title("BEFORE normalization (raw)", fontsize=10)
        ax0.legend(fontsize=7); ax0.grid(alpha=0.3)
        st.pyplot(fig0)
    with p2:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        if "Individual cells" in trace_layers:
            for cn in shown_cells:
                ax.plot(df["Frame"], df[cn], lw=0.65, alpha=0.35)
        if "Population mean" in trace_layers:
            ax.plot(df["Frame"], df[cells].mean(axis=1), lw=2, color="crimson",
                    label="population mean")
        if "Baseline window" in trace_layers:
            for r in base_rows:
                if r < len(frames):
                    ax.axvspan(frames[r] - 0.5, frames[r] + 0.5, color="gold", alpha=0.18)
        if "Current-frame marker" in trace_layers:
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

    def _trace_gif_frames():
        movie = []
        movie_indices = _gif_frame_ids(range(len(frames)), max_frames=120)
        display_cells = [c for c in shown_cells if c in cells][:50]
        heat = np.nan_to_num(df[cells].to_numpy(float)).T
        heat_max = np.percentile(heat, 99) if heat.size else 1
        for _frame_index in movie_indices:
            _current_frame = frames[_frame_index]
            _n_axes = 3 if show_heatmap else 2
            figure, axes = plt.subplots(1, _n_axes, figsize=(4.8 * _n_axes, 3.6), dpi=100)
            axes = np.atleast_1d(axes)
            raw_ax, dff_ax = axes[0], axes[1]
            if "Individual cells" in trace_layers:
                for _cell in [c for c in display_cells if c in raw_cols]:
                    raw_ax.plot(raw_sub["Frame"], raw_sub[_cell], lw=0.55, alpha=0.28,
                                color="dimgray")
                for _cell in display_cells:
                    dff_ax.plot(df["Frame"], df[_cell], lw=0.55, alpha=0.28, color="steelblue")
            if "Population mean" in trace_layers and raw_cols:
                raw_ax.plot(raw_sub["Frame"], raw_sub[raw_cols].mean(axis=1), lw=1.8,
                            color="black", label="population mean")
                dff_ax.plot(df["Frame"], df[cells].mean(axis=1), lw=1.8,
                            color="crimson", label="population mean")
            if "Baseline window" in trace_layers:
                for _row in base_rows:
                    if _row < len(frames):
                        _x = frames[_row]
                        raw_ax.axvspan(_x - 0.5, _x + 0.5, color="gold", alpha=0.16)
                        dff_ax.axvspan(_x - 0.5, _x + 0.5, color="gold", alpha=0.16)
            if "Current-frame marker" in trace_layers:
                raw_ax.axvline(_current_frame, color="crimson", lw=1.4)
                dff_ax.axvline(_current_frame, color="black", lw=1.4)
            raw_ax.set(title="Raw intensity", xlabel="Frame", ylabel="a.u.")
            dff_ax.set(title="ΔF/F0i", xlabel="Frame", ylabel="ΔF/F0i")
            for axis in (raw_ax, dff_ax):
                axis.grid(alpha=0.25)
                if "Population mean" in trace_layers:
                    axis.legend(fontsize=7)
            if show_heatmap:
                heat_ax = axes[2]
                heat_ax.imshow(heat, aspect="auto", cmap="magma", vmin=0,
                               vmax=heat_max if heat_max > 0 else 1, interpolation="nearest")
                if "Current-frame marker" in trace_layers:
                    heat_ax.axvline(_frame_index, color="cyan", lw=1.4)
                heat_ax.set(title="Cell activity", xlabel="Frame", ylabel="Cell")
            figure.suptitle(f"Frame {_current_frame}", fontsize=10)
            figure.tight_layout()
            movie.append(_figure_rgb(figure))
        return movie

    _trace_name = "cafin_traces_roi.gif" if "roi" in sfx else "cafin_traces_all_cells.gif"
    gif_download_control("traces" + sfx, _trace_name, _trace_gif_frames, tspd, max_frames=120)

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

            def _background_gif_frames():
                movie = []
                for _frame_no in _gif_frame_ids(frames):
                    _image = _fixed8(ca_by_frame[_frame_no])
                    _display = np.dstack([_image] * 3)
                    for (_x1, _y1, _x2, _y2) in boxes:
                        _display[_y1:_y2, _x1:_x1 + 2] = [0, 229, 255]
                        _display[_y1:_y2, _x2 - 2:_x2] = [0, 229, 255]
                        _display[_y1:_y1 + 2, _x1:_x2] = [0, 229, 255]
                        _display[_y2 - 2:_y2, _x1:_x2] = [0, 229, 255]
                    _mean, _ = bgv[_frame_no]
                    movie.append(_label_gif_frame(
                        _display, f"Frame {_frame_no}: background {_mean:.1f} a.u."))
                return movie

            _bg_gif_name = ("cafin_background_check_roi.gif" if "roi" in sfx
                            else "cafin_background_check_all_cells.gif")
            gif_download_control("background" + sfx, _bg_gif_name,
                                 _background_gif_frames, bspd)
            frame_loop("bgchk" + sfx, bfi, bplay, bspd, len(frames))
        except Exception as e:
            st.warning(f"Could not build the background check: {e}")

    if show_heatmap:
        st.subheader("Activity heatmap (cells × frames)")
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        arr = np.nan_to_num(df[cells].to_numpy(float)).T
        im = ax2.imshow(arr, aspect="auto", cmap="magma", vmin=0,
                        vmax=np.percentile(arr, 99) if arr.size else 1, interpolation="nearest")
        if "Current-frame marker" in trace_layers:
            ax2.axvline(tfi, color="cyan", lw=1.5, alpha=0.9)
        ax2.set_xlabel("Frame"); ax2.set_ylabel("Cell")
        fig2.colorbar(im, ax=ax2, label="ΔF/F0i")
        st.pyplot(fig2)
    frame_loop("trc" + sfx, tfi, tplay, tspd, len(frames))


with T[TAB_TRC]:
    roi_split(render_traces, "trc")

# ---------------------------------------------------------- Clustering
def render_clustering(df, sfx):
    st.subheader("Clustering (PCA → K-means)")
    cells = [c for c in df.columns if c.startswith("Cell_")]
    if not cells:
        st.info("No cell traces are available to cluster.")
    else:
        target = st.radio("Cluster what?", ["Cells", "Tissue states (frames)"], horizontal=True,
                          key="clu_target" + sfx,
                          help="Cell clustering groups cells. Tissue-state clustering groups frames by "
                               "the selected whole-field metrics.")
        selected_cell_features, selected_tissue_features = [], []
        include_trace, frame_interval, show_tissue_summary = False, 1.0, False
        if target == "Cells":
            if len(cells) < 4:
                st.info("Need at least 4 cells for cell clustering. Tissue-state clustering is still available.")
                return
            st.markdown("**Choose the checked inputs used to cluster cells**")
            st.caption("Whole traces preserve dynamic shape. Checked scalar features add biological "
                       "measurements such as first-peak time or activity. Each checked family is "
                       "balanced so a long trace does not drown out a one-value feature.")
            include_trace = st.checkbox("Whole ΔF/F0i traces", value=True, key="clu_trace" + sfx,
                                        help="Original PCA + K-means trace-shape clustering.")
            peak_on = st.checkbox("Peak dynamics", value=False, key="clu_peak_on" + sfx,
                                  help="Use checked per-cell peak measurements as clustering inputs.")
            if peak_on:
                st.caption("Cells without a first detected peak are placed at the end of the recording "
                           "for that feature. Other missing peak-shape values are median-imputed; "
                           "number of peaks retains the information that a cell was silent.")
                frame_interval = st.number_input(
                    "Frame interval for peak timing (minutes; use 1 for frames)", 0.001, 60.0, 1.0,
                    step=0.1, key="clu_frame_interval" + sfx)
                pcols = st.columns(3)
                peak_keys = ["n_peaks", "t_first_peak", "auc", "amplitude", "fwhm", "dt_peak"]
                for i, key in enumerate(peak_keys):
                    if pcols[i % len(pcols)].checkbox(cc.CELL_CLUSTER_FEATURES[key],
                                                       value=(key == "t_first_peak"),
                                                       key="clu_feat_" + key + sfx):
                        selected_cell_features.append(key)
            activity_on = st.checkbox("Cell activity summary", value=False,
                                      key="clu_activity_on" + sfx,
                                      help="Use activity magnitude and recruitment per cell.")
            if activity_on:
                acols = st.columns(2)
                for i, key in enumerate(["active_frame_fraction", "mean_dff0", "max_dff0", "t_max_dff0"]):
                    if acols[i % len(acols)].checkbox(cc.CELL_CLUSTER_FEATURES[key], value=False,
                                                       key="clu_feat_" + key + sfx):
                        selected_cell_features.append(key)
            coupling_on = st.checkbox("Cell-to-tissue coupling", value=False,
                                      key="clu_coupling_on" + sfx,
                                      help="Use how closely each cell follows the tissue-mean trace.")
            if coupling_on:
                key = "tissue_mean_correlation"
                if st.checkbox(cc.CELL_CLUSTER_FEATURES[key], value=True,
                               key="clu_feat_" + key + sfx):
                    selected_cell_features.append(key)
            show_tissue_summary = st.checkbox("Show tissue-level summary alongside cell clusters", value=False,
                                               key="clu_show_tissue" + sfx,
                                               help="Whole-tissue metrics have one value per field, so they are "
                                                    "shown beside cell clusters rather than used to split cells.")
            n_observations = len(cells)
        else:
            st.markdown("**Choose the checked tissue-level measurements used to cluster frames**")
            st.caption("This is a time-state analysis: each point is an imaging frame, not a cell. "
                       "It can identify quiet, recruitment, and high-activity tissue states.")
            tissue_cols = st.columns(2)
            for i, key in enumerate(cc.TISSUE_CLUSTER_FEATURES):
                default = key in ("tissue_mean_dff0", "active_cell_fraction")
                if tissue_cols[i % len(tissue_cols)].checkbox(cc.TISSUE_CLUSTER_FEATURES[key], value=default,
                                                               key="clu_tissue_" + key + sfx):
                    selected_tissue_features.append(key)
            n_observations = len(df)

        cc1, cc2 = st.columns(2)
        _pca_max = int(max(1, min(200, max(1, n_observations - 1), len(frames))))
        n_pca = cc1.slider("PCA components", 1, _pca_max, min(25, _pca_max), key="npca" + sfx,
                           help="The selected inputs are reduced to this many components before K-means.")
        _kmax = int(max(2, min(30, n_observations)))
        k = cc2.slider("Number of clusters (k)", 2, _kmax, min(4, _kmax), key="nclu" + sfx)
        try:
            if target == "Cells":
                cl = cc.cluster_cells(df, include_trace=include_trace,
                                      selected_features=selected_cell_features, threshold=peak_thr,
                                      frame_interval=frame_interval, n_pca=n_pca, n_clusters=k)
            else:
                cl = cc.cluster_tissue_states(df, selected_features=selected_tissue_features,
                                              threshold=peak_thr, n_pca=n_pca, n_clusters=k)
        except ValueError as exc:
            st.warning(f"Clustering cannot run yet: {exc}")
            return
        ids, labels, coords = cl["ids"], cl["labels"], cl["coords"]
        cluster_signature = (target, include_trace, tuple(selected_cell_features),
                             tuple(selected_tissue_features), n_pca, k, peak_thr, frame_interval)
        signature_key = "cluster_signature" + sfx
        if ss.get(signature_key) != cluster_signature:
            for stale in ("ai_story", "ai_story_full", "ai_err", "ai_chat", "ai_conn"):
                ss.pop(stale + sfx, None)
            ss[signature_key] = cluster_signature
        if sfx.endswith("_all"):
            # Do not leave a downloadable assignment table from a different
            # clustering target visible after the researcher switches modes.
            if target == "Cells":
                ss.pop("tissue_cluster_df", None)
            else:
                ss.pop("cluster_df", None)
        st.caption(f"Checked inputs used: **{', '.join(cl['feature_groups'])}**. PCA kept "
                   f"{cl['n_pca_used']} component{'s' if cl['n_pca_used'] != 1 else ''} "
                   f"({cl['explained_var']*100:.0f}% variance) · {cl['k']} clusters.")
        if cl["dropped_feature_groups"]:
            st.info("Not used because there was no variation in this recording: "
                    + ", ".join(cl["dropped_feature_groups"]) + ".")
        colors, cnames = cluster_palette(cl["k"])

        if target == "Tissue states (frames)":
            colL, colR = st.columns(2)
            tissue_table = cl["feature_table"].copy()
            tissue_table["tissue_state"] = labels
            tissue_table = tissue_table[["Frame", "tissue_state"] +
                                        [c for c in tissue_table if c not in ("Frame", "tissue_state")]]
            with colL:
                figt, axt = plt.subplots(figsize=(6, 4))
                axt.plot(tissue_table["Frame"], tissue_table["tissue_mean_dff0"], color="0.65", lw=1,
                         label="tissue mean ΔF/F0i")
                for lab in range(cl["k"]):
                    m = labels == lab
                    axt.scatter(np.asarray(ids)[m], tissue_table.loc[m, "tissue_mean_dff0"], s=28,
                                color=colors[lab] / 255, label=f"State {lab} (n={m.sum()})")
                axt.set_xlabel("Frame"); axt.set_ylabel("tissue mean ΔF/F0i")
                axt.grid(alpha=0.3); axt.legend(fontsize=8)
                axt.set_title("Tissue activity states")
                st.pyplot(figt)
            with colR:
                figp, axp = plt.subplots(figsize=(5.5, 4))
                for lab in range(cl["k"]):
                    m = labels == lab
                    axp.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.8,
                                color=colors[lab] / 255, label=f"State {lab} (n={m.sum()})")
                axp.set_xlabel("PC1"); axp.set_ylabel("PC2")
                axp.legend(fontsize=8); axp.grid(alpha=0.3)
                st.pyplot(figp)
            st.dataframe(tissue_table, width="stretch", height=280)
            if sfx.endswith("_all"):
                ss["tissue_cluster_df"] = tissue_table
            st.download_button("⬇ tissue-state assignments (CSV)", tissue_table.to_csv(index=False).encode(),
                               "tissue_state_assignments.csv", "text/csv", key="dltissueclu" + sfx)
            st.caption("Tissue-state clustering does not color individual cells or run the cell-cluster AI story. "
                       "Switch to **Cells** to group cells by the checked inputs.")
            return

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
        feature_columns = ["cell_id"] + [c for c in selected_cell_features
                                          if c in cl["feature_table"].columns]
        if len(feature_columns) > 1:
            cluster_df = cluster_df.merge(cl["feature_table"][feature_columns], on="cell_id", how="left")
        cluster_df["cluster_inputs"] = "; ".join(cl["feature_groups"])
        if sfx.endswith("_all"):
            ss["cluster_df"] = cluster_df      # the all-cell run feeds the Downloads tab
        if selected_cell_features and st.checkbox("Show selected per-cell clustering features", False,
                                                  key="clu_feature_table" + sfx):
            st.dataframe(cluster_df, width="stretch", height=280)
        st.download_button("⬇ cluster assignments (CSV)", cluster_df.to_csv(index=False).encode(),
                           "cluster_assignments.csv", "text/csv", key="dlclu" + sfx)

        if show_tissue_summary:
            st.subheader("Tissue-level summary")
            tissue_metrics, _ = cc.metrics(df, threshold=peak_thr)
            st.dataframe(pd.DataFrame([tissue_metrics]).T.rename(columns={0: "value"}), width="stretch")
            frac = (df[cells].to_numpy(float) > peak_thr).mean(axis=1)
            figaf, axaf = plt.subplots(figsize=(11, 2.8))
            axaf.plot(df["Frame"], frac, color="tab:purple", lw=1.8)
            axaf.set_ylim(0, 1); axaf.set_xlabel("Frame"); axaf.set_ylabel("active-cell fraction")
            axaf.grid(alpha=0.3); st.pyplot(figaf)

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
                       clustering_inputs=cl["feature_groups"],
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
                            clustering_inputs=cl["feature_groups"],
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
        tidx, trk_play, trk_speed = frame_controls("trk", nT, start=nT - 1)
        ftrk = frames[tidx]

        def _tracking_panels(frame_no):
            """Return the exact two images shown in the tracking playback."""
            _base = cc.stretch8(memsrc.get(frame_no, memsrc[frames[0]]), clahe=True)
            _tracked = reg["mask_per_frame"].get(frame_no, mask0)
            if link:                                  # stable global ID means stable colour
                _colored = np.dstack([_base] * 3).astype(float) * 0.4
                _colors = (plt.get_cmap("tab20")(np.linspace(0, 1, 20)) * 255)[:, :3]
                for _cell_id in np.unique(_tracked):
                    if _cell_id > 0:
                        _colored[_tracked == _cell_id] = _colors[int(_cell_id) % 20]
                _left = np.clip(_colored, 0, 255).astype(np.uint8)
                _right = np.dstack([_base] * 3)
                _right[find_boundaries(_tracked, mode="outer")] = [0, 255, 0]
            else:
                _left = np.dstack([_base] * 3)
                _left[find_boundaries(_tracked, mode="outer")] = [0, 255, 0]
                _right = np.dstack([_base] * 3)
                _right[find_boundaries(mask0, mode="outer")] = [255, 80, 0]
            return _left, _right

        left_panel, right_panel = _tracking_panels(ftrk)
        a, b = st.columns(2)
        if link:
            a.image(left_panel, caption=f"Tracked cells on frame {ftrk} (color = global id)",
                    width="stretch")
            b.image(right_panel, caption=f"Tracked boundaries on frame {ftrk}", width="stretch")
        else:
            a.image(left_panel, caption=f"TRACKED mask on frame {ftrk} (green)", width="stretch")
            b.image(right_panel, caption=f"STATIC frame-0 mask on frame {ftrk} (orange)",
                    width="stretch")

        def _tracking_gif_frames():
            movie = []
            for _frame_no in _gif_frame_ids(frames):
                _left, _right = _tracking_panels(_frame_no)
                _label = (f"Frame {_frame_no}: stable IDs" if link
                          else f"Frame {_frame_no}: tracked (green), static (orange)")
                movie.append(_label_gif_frame(_stack_gif_frames(_left, _right), _label))
            return movie

        gif_download_control("tracking", "cafin_tracking_playback.gif",
                             _tracking_gif_frames, trk_speed)
        frame_loop("trk", tidx, trk_play, trk_speed, nT)
    else:
        st.info("Pick **Cell tracking** in the sidebar to follow cells across frames.")

# ---------------------------------------------------------- Statistics
def render_stats(df, sfx):
    st_, dist_ = cc.metrics(df, threshold=peak_thr)

    stat_sections = st.multiselect(
        "Choose statistical results to display",
        ["Peak dynamics", "Tissue-level summary", "Active-cell fraction"],
        default=["Peak dynamics", "Tissue-level summary", "Active-cell fraction"],
        key="stat_sections" + sfx,
        help="Peak dynamics describes single-cell events. Tissue metrics summarize the field. "
             "Active-cell fraction shows recruitment through time.")

    # ---------- per-cell peak dynamics, one point per cell ----------
    fi = st.number_input("Frame interval (minutes per frame; leave at 1 to report in frames)",
                         0.001, 60.0, 1.0, step=0.1, key="fint" + sfx)
    feats = cc.peak_features(df, threshold=peak_thr, frame_interval=fi)
    unit = "min" if abs(fi - 1.0) > 1e-9 else "frames"
    panel_map = {
        "Number of peaks": ("n_peaks", "# peaks", ""),
        "Time to first peak": ("t_first_peak", "1$^{st}$ peak", unit),
        "Area under curve": ("auc", "A.U.C.", ""),
        "Peak amplitude": ("amplitude", "Amplitude ($\\Delta$F/F$_0$)", ""),
        "Peak width (FWHM)": ("fwhm", "F.W.H.M.", unit),
        "Time between peaks": ("dt_peak", "$\\Delta t_{peak}$", unit),
    }
    if "Peak dynamics" in stat_sections:
        st.subheader("Peak dynamics")
        selected_panels = st.multiselect(
            "Single-cell features to plot", list(panel_map), default=list(panel_map),
            key="stat_features" + sfx,
            help="Select only the event properties relevant to the biological question.")
        panels = [panel_map[name] for name in selected_panels]
        if panels:
            rng = np.random.default_rng(0)
            fig, axes = plt.subplots(1, len(panels), figsize=(max(4, 2.5 * len(panels)), 3.8))
            axes = np.atleast_1d(axes)
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
                ax.set_xticks([]); ax.yaxis.grid(True, ls="--", alpha=0.55)
                for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            n_ok = int(feats["n_peaks"].gt(0).sum())
            fig.suptitle(f"{len(feats)} cells ({n_ok} with a detected peak; "
                         f"threshold {peak_thr} ΔF/F0i)", fontsize=10)
            fig.tight_layout(); st.pyplot(fig)
        show_feature_table = st.checkbox("Show per-cell feature table", False,
                                         key="stat_table" + sfx)
        if show_feature_table:
            st.dataframe(feats, width="stretch", height=300)
        st.download_button("⬇ per-cell peak features (CSV)", feats.to_csv(index=False).encode(),
                           "peak_features.csv", "text/csv", key="dlfeat" + sfx)
        st.caption("One point per cell. Cells without a detected peak are excluded from "
                   "peak-shape features but remain in cell counts.")

    if "Tissue-level summary" in stat_sections:
        st.subheader("Tissue-level metrics")
        metric_table = pd.DataFrame([st_]).T.rename(columns={0: "value"})
        selected_metrics = st.multiselect(
            "Summary metrics", list(metric_table.index), default=list(metric_table.index),
            key="tissue_metrics" + sfx)
        st.dataframe(metric_table.loc[selected_metrics], width="stretch")

    if "Active-cell fraction" in stat_sections:
        st.subheader("Fraction of cells above threshold, per frame")
        cells = [c for c in df.columns if c.startswith("Cell_")]
        frac = (df[cells].to_numpy(float) > peak_thr).mean(axis=1)
        fig3, ax3 = plt.subplots(figsize=(11, 3.5))
        ax3.plot(df["Frame"], frac, "-o", ms=3)
        ax3.set_ylim(0, 1); ax3.set_xlabel("Frame"); ax3.set_ylabel("fraction active")
        ax3.grid(alpha=0.3); st.pyplot(fig3)


with T[TAB_STA]:
    roi_split(render_stats, "sta")

# ---------------------------------------------------------- Network Analysis
def render_pixel_network_legacy(df, sfx):
    """Retained only as historical code; the active tab uses cell networks below."""
    is_roi = "roi" in sfx and R.get("roi_box") is not None
    curr_roi_box = R.get("roi_box") if is_roi else None

    st.subheader("Network analysis (pixel correlations & k-clique communities)")
    st.caption(
        "Builds a pixel-level correlation network on registered calcium frames, filters nodes by "
        "positive correlation with the tissue-average signal, and detects k-clique percolation communities."
    )

    c1, c2, c3 = st.columns(3)
    n_samples = c1.number_input(
        "Sampled pixels",
        min_value=50,
        max_value=1000,
        value=250,
        step=50,
        key="net_samples" + sfx,
        help="Number of pixels randomly sampled within the tissue mask (default 250, capped at 1,000 for responsive community detection).",
    )
    seed = c2.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=0,
        step=1,
        key="net_seed" + sfx,
        help="Reproducible random seed for pixel sampling.",
    )
    k_clique = c3.number_input(
        "k-clique size (k)",
        min_value=3,
        max_value=15,
        value=6,
        step=1,
        key="net_k" + sfx,
        help="Minimum clique size for community percolation (default k=6).",
    )

    c4, c5 = st.columns(2)
    tissue_r_thresh = c4.slider(
        "Tissue-correlation threshold (positive direction)",
        min_value=0.00,
        max_value=0.90,
        value=0.30,
        step=0.05,
        key="net_tissue_r" + sfx,
        help="Only pixels whose calcium trace correlates positively with the tissue-average signal above this Pearson r are retained as network nodes.",
    )
    r2_thresh = c5.slider(
        "Pearson R² edge threshold",
        min_value=0.30,
        max_value=1.00,
        value=0.70,
        step=0.05,
        key="net_r2" + sfx,
        help="Pearson R² ≥ 0.70 corresponds to |Pearson r| ≥ 0.837. Edges connect pairs with R² at or above this threshold.",
    )

    c6, c7 = st.columns(2)
    restrict_mask = c6.checkbox(
        "Restrict to segmented tissue",
        value=True,
        key="net_mask" + sfx,
        help="Sample pixels only inside segmented cell/tissue regions.",
    )
    pos_edges = c7.checkbox(
        "Positive-only network edges",
        value=False,
        key="net_pos_edge" + sfx,
        help="When unchecked (default), high-R² edges include both strongly correlated and strongly anti-correlated pixel pairs, reproducing the referenced FocalPlane/NetworkX R² workflow. Check this to restrict edges to positively correlated signals.",
    )

    if not pos_edges:
        st.caption(
            "ℹ️ *Note: With positive-only network edges disabled (default), Pearson R² includes both strong positive (r ≥ +0.837) and strong negative (r ≤ -0.837) correlations.*"
        )

    # Parameter caching key
    dataset_name = os.path.basename(trial.rstrip("/\\")) if "trial" in locals() and trial else ""
    param_key = (
        dataset_name,
        tuple(curr_roi_box) if curr_roi_box else None,
        int(seed),
        int(n_samples),
        float(tissue_r_thresh),
        float(r2_thresh),
        bool(pos_edges),
        int(k_clique),
        bool(restrict_mask),
    )
    cache_state_key = "_net_cache_" + sfx
    result_state_key = "_net_result_" + sfx

    if st.button("🕸 Build network", type="primary", key="btn_net" + sfx):
        with st.spinner("Computing pixel correlations and k-clique communities…"):
            _bg_bxs = cc.auto_bg_boxes(reg["reg_ca"][frames[0]]) if R.get("do_bg") else None
            res = cc.analyze_pixel_network_legacy(
                reg_or_ca=reg,
                mask0=mask0,
                frames=frames,
                bg_boxes=_bg_bxs,
                do_bg=R.get("do_bg", False),
                roi_box=curr_roi_box,
                n_samples=int(n_samples),
                seed=int(seed),
                tissue_r_thresh=float(tissue_r_thresh),
                r2_thresh=float(r2_thresh),
                positive_edges_only=bool(pos_edges),
                k_clique=int(k_clique),
                restrict_to_mask=bool(restrict_mask),
                dataset_name=dataset_name,
            )
            ss[result_state_key] = res
            ss[cache_state_key] = param_key
            st.rerun()

    net_res = ss.get(result_state_key)
    if net_res is None:
        st.info("Click **🕸 Build network** above to compute pixel correlations and k-clique communities.")
        return

    if net_res.get("error"):
        if net_res.get("safety"):
            st.error(f"⚠️ **Safety Preflight Guard:** {net_res['error']}")
        else:
            st.warning(f"⚠️ **Network Analysis Notice:** {net_res['error']}")
        return

    # Metric summary row
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Retained Nodes", net_res["n_nodes"])
    m2.metric("Edges", net_res["n_edges"])
    m3.metric("Density", f"{net_res['density']:.3f}")
    m4.metric("Mean Degree", f"{net_res['mean_degree']:.1f}")
    m5.metric("Components", net_res["n_components"])
    m6.metric("Communities", net_res["n_communities"])
    m7.metric("Overlapping", net_res["n_overlapping"])

    if net_res["n_communities"] == 0:
        st.info(
            f"No k-clique communities were detected with the current settings (k={k_clique}, R²≥{r2_thresh:.2f}). "
            "Try lowering the R² threshold or k-clique size."
        )

    # Spatial Network Map
    st.subheader("Spatial community map")
    sc1, sc2 = st.columns([1, 2])
    show_edges = sc1.checkbox("Draw edges on map", value=False, key="net_show_edges" + sfx)
    edge_alpha = sc2.slider("Edge opacity", 0.05, 1.0, 0.25, 0.05, key="net_edge_alpha" + sfx) if show_edges else 0.25

    ref_img = reg["reg_ca"].get(frames[0], next(iter(reg["reg_ca"].values())))
    H, W = ref_img.shape
    fig, ax = plt.subplots(figsize=(8, max(4, int(8 * H / max(W, 1)))), dpi=120)
    ax.imshow(cc.stretch8(ref_img, clahe=True), cmap="gray")

    if curr_roi_box:
        x1, y1, x2, y2 = curr_roi_box
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color="yellow", lw=1.5, ls="--", label="ROI box")

    if show_edges and net_res["n_edges"] > 0 and len(net_res["edges_df"]) > 0:
        edf = net_res["edges_df"]
        for _, erow in edf.iterrows():
            ax.plot(
                [erow["source_x"], erow["target_x"]],
                [erow["source_y"], erow["target_y"]],
                color="cyan",
                alpha=float(edge_alpha),
                lw=0.75,
            )

    K = net_res["n_communities"]
    cols, cnames = cluster_palette(max(K, 1))

    # Plot unassigned nodes
    unassigned = net_res["nodes_df"][net_res["nodes_df"]["primary_community"] == -1]
    if len(unassigned) > 0:
        ax.scatter(
            unassigned["x"],
            unassigned["y"],
            c="gray",
            s=28,
            alpha=0.6,
            label=f"Unassigned ({len(unassigned)})",
            edgecolors="none",
            zorder=2,
        )

    # Plot community nodes
    for c in range(K):
        single = net_res["nodes_df"][
            (net_res["nodes_df"]["primary_community"] == c) & (net_res["nodes_df"]["overlap_count"] == 1)
        ]
        if len(single) > 0:
            ax.scatter(
                single["x"],
                single["y"],
                color=[cols[c] / 255.0],
                s=36,
                alpha=0.85,
                label=f"Community {c} ({len(single)})",
                edgecolors="none",
                zorder=3,
            )
        multi = net_res["nodes_df"][
            (net_res["nodes_df"]["primary_community"] == c) & (net_res["nodes_df"]["overlap_count"] > 1)
        ]
        if len(multi) > 0:
            ax.scatter(
                multi["x"],
                multi["y"],
                color=[cols[c] / 255.0],
                s=52,
                alpha=0.95,
                edgecolors="black",
                linewidths=1.8,
                label=f"Community {c} (overlapping, {len(multi)})",
                zorder=4,
            )

    ax.set_title("Spatial community map (black outline = overlapping community node)", fontsize=10)
    ax.axis("off")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        "Gray points: unassigned nodes (not part of any k-clique). Solid colors: single community membership. "
        "Black border rings: overlapping nodes that belong to multiple communities."
    )

    if net_res["n_overlapping"] > 0:
        with st.expander(f"Overlapping community nodes ({net_res['n_overlapping']} nodes)", expanded=False):
            over_df = net_res["nodes_df"][net_res["nodes_df"]["overlap_count"] > 1]
            st.dataframe(
                over_df[["node_id", "y", "x", "tissue_r", "degree", "community_ids", "overlap_count"]],
                width="stretch",
            )

    # Distribution plots
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        fig_d, ax_d = plt.subplots(figsize=(5, 3))
        ax_d.hist(net_res["nodes_df"]["degree"], bins=20, color="steelblue", edgecolor="black", lw=0.5)
        ax_d.set_xlabel("Node degree")
        ax_d.set_ylabel("Nodes")
        ax_d.set_title("Node degree distribution", fontsize=9)
        ax_d.grid(alpha=0.3)
        fig_d.tight_layout()
        st.pyplot(fig_d)
    with pcol2:
        fig_r, ax_r = plt.subplots(figsize=(5, 3))
        ax_r.hist(net_res["nodes_df"]["tissue_r"], bins=20, color="darkseagreen", edgecolor="black", lw=0.5)
        ax_r.set_xlabel("Tissue Pearson r")
        ax_r.set_ylabel("Nodes")
        ax_r.set_title("Tissue correlation distribution (retained nodes)", fontsize=9)
        ax_r.grid(alpha=0.3)
        fig_r.tight_layout()
        st.pyplot(fig_r)

    # Direct CSV Download Buttons
    st.divider()
    st.subheader("Download network results")
    dc1, dc2, dc3 = st.columns(3)
    dc1.download_button(
        "⬇ network_nodes.csv",
        net_res["nodes_df"].to_csv(index=False).encode(),
        "network_nodes.csv",
        "text/csv",
        key="dl_net_nodes" + sfx,
    )
    dc2.download_button(
        "⬇ network_edges.csv",
        net_res["edges_df"].to_csv(index=False).encode(),
        "network_edges.csv",
        "text/csv",
        key="dl_net_edges" + sfx,
    )
    dc3.download_button(
        "⬇ network_summary.csv",
        net_res["summary_df"].to_csv(index=False).encode(),
        "network_summary.csv",
        "text/csv",
        key="dl_net_summary" + sfx,
    )


def render_cell_network(df, sfx):
    """Render communication analysis where each graph node is one cell."""
    is_roi = "roi" in sfx and R.get("roi_box") is not None
    curr_roi_box = R.get("roi_box") if is_roi else None
    cell_cols = [c for c in df.columns if c.startswith("Cell_")]
    if len(cell_cols) < 2:
        st.info("At least two single-cell traces are needed for communication analysis.")
        return

    st.subheader("Communication analysis (single-cell calcium traces)")
    st.caption(
        "Each node is a segmented cell and each node signal is its extracted ΔF/F0i trace. "
        "Cells are filtered by correlation with the field-average cell trace, connected using "
        "Pearson R², and grouped with k-clique percolation."
    )
    c1, c2, c3 = st.columns(3)
    n_samples = c1.number_input(
        "Sampled cells", min_value=2, max_value=250, value=min(250, len(cell_cols)), step=1,
        key="cellnet_samples" + sfx,
        help="Randomly sample cells before pairwise correlation. Set this to the number of cells to use all available cells.",
    )
    seed = c2.number_input("Random seed", min_value=0, max_value=999999, value=0, step=1,
                           key="cellnet_seed" + sfx)
    k_clique = c3.number_input("k-clique size (k)", min_value=2, max_value=15, value=6, step=1,
                               key="cellnet_k" + sfx,
                               help="Minimum clique size for community percolation.")
    c4, c5 = st.columns(2)
    tissue_r_thresh = c4.slider(
        "Cell-to-tissue Pearson r threshold", 0.0, 0.95, 0.30, 0.05,
        key="cellnet_tissue_r" + sfx,
        help="Keep cells whose activity follows the mean activity of eligible cells.")
    r2_thresh = c5.slider(
        "Pearson R² edge threshold", 0.0, 1.0, 0.70, 0.05,
        key="cellnet_r2" + sfx,
        help="R² ≥ 0.70 corresponds to |Pearson r| ≥ 0.837.")
    c6, c7 = st.columns(2)
    tissue_positive_only = c6.checkbox(
        "Positive cell-to-tissue filter", value=True, key="cellnet_tissue_pos" + sfx,
        help="The original workflow keeps cells positively correlated with the average activity.")
    positive_edges = c7.checkbox(
        "Positive-only communication edges", value=False, key="cellnet_pos_edge" + sfx,
        help="Off reproduces the R² rule and includes strong anti-correlations as edges.")
    st.caption(
        "The default edge rule uses Pearson R², so strong negative correlations are retained too. "
        "Enable positive-only edges only when anti-correlated activity should be excluded."
    )

    dataset_name = os.path.basename(trial.rstrip("/\\")) if "trial" in locals() and trial else ""
    param_key = (dataset_name, tuple(curr_roi_box) if curr_roi_box else None,
                 tuple(cell_cols), int(n_samples), int(seed), int(k_clique),
                 float(tissue_r_thresh), float(r2_thresh), bool(tissue_positive_only),
                 bool(positive_edges))
    cache_key = "_cellnet_cache_" + sfx
    result_key = "_cellnet_result_" + sfx
    if ss.get(cache_key) != param_key:
        ss.pop(result_key, None)
    if st.button("🕸 Build cell communication network", type="primary", key="btn_cellnet" + sfx):
        with st.spinner("Computing cell correlations and k-clique communities…"):
            result = cc.analyze_cell_network(
                dff_df=df, mask0=mask0, roi_box=curr_roi_box, n_samples=int(n_samples),
                seed=int(seed), tissue_r_thresh=float(tissue_r_thresh),
                r2_thresh=float(r2_thresh), positive_edges_only=bool(positive_edges),
                tissue_positive_only=bool(tissue_positive_only), k_clique=int(k_clique),
                restrict_to_mask=True, dataset_name=dataset_name,
            )
            ss[result_key] = result
            ss[cache_key] = param_key
            st.rerun()

    result = ss.get(result_key)
    if result is None:
        st.info("Choose the settings and click **Build cell communication network**.")
        return
    if result.get("error"):
        message = result["error"]
        if result.get("safety"):
            st.error(f"⚠️ **Safety guard:** {message}")
        else:
            st.warning(f"⚠️ **Communication analysis notice:** {message}")
        return

    m = st.columns(7)
    m[0].metric("Cells", result["n_nodes"])
    m[1].metric("Edges", result["n_edges"])
    m[2].metric("Density", f"{result['density']:.3f}")
    m[3].metric("Mean degree", f"{result['mean_degree']:.1f}")
    m[4].metric("Components", result["n_components"])
    m[5].metric("Communities", result["n_communities"])
    m[6].metric("Overlapping cells", result["n_overlapping"])
    if result["n_communities"] == 0:
        st.info(f"No k-clique communities were found at k={k_clique} and R² ≥ {r2_thresh:.2f}.")

    st.subheader("Spatial cell-community map")
    show_edges = st.checkbox("Draw communication edges", value=False, key="cellnet_show_edges" + sfx)
    edge_alpha = st.slider("Edge opacity", 0.05, 1.0, 0.25, 0.05,
                           key="cellnet_edge_alpha" + sfx) if show_edges else 0.25
    ref = reg["reg_ca"].get(frames[0], reg["reg_mem"][frames[0]])
    h, w = ref.shape
    fig, ax = plt.subplots(figsize=(8, max(4, int(8 * h / max(w, 1)))), dpi=120)
    ax.imshow(cc.stretch8(ref, clahe=True), cmap="gray")
    if curr_roi_box:
        x1, y1, x2, y2 = curr_roi_box
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "--", color="yellow", lw=1.5)
    if show_edges and len(result["edges_df"]):
        for _, e in result["edges_df"].iterrows():
            ax.plot([e["source_x"], e["target_x"]], [e["source_y"], e["target_y"]],
                    color="cyan", alpha=float(edge_alpha), lw=0.8)
    nodes = result["nodes_df"]
    colors, _ = cluster_palette(max(1, result["n_communities"]))
    unassigned = nodes[nodes["primary_community"] < 0]
    if len(unassigned):
        ax.scatter(unassigned["x"], unassigned["y"], c="gray", s=42, label="Unassigned", zorder=2)
    for community in range(result["n_communities"]):
        single = nodes[(nodes["primary_community"] == community) & (nodes["overlap_count"] == 1)]
        overlap = nodes[(nodes["primary_community"] == community) & (nodes["overlap_count"] > 1)]
        if len(single):
            ax.scatter(single["x"], single["y"], color=colors[community] / 255.0, s=48,
                       label=f"Community {community} (n={len(single)})", zorder=3)
        if len(overlap):
            ax.scatter(overlap["x"], overlap["y"], color=colors[community] / 255.0, s=68,
                       edgecolors="black", linewidths=1.8,
                       label=f"Community {community}, overlapping", zorder=4)
    ax.set_title("Cell communication communities (black outline = overlap)", fontsize=10)
    ax.axis("off")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("One point is one cell. Gray cells are not in a k-clique; black outlines mark cells in multiple communities.")

    if result["n_overlapping"]:
        with st.expander(f"Overlapping cell memberships ({result['n_overlapping']})"):
            st.dataframe(nodes[nodes["overlap_count"] > 1], width="stretch")
    p1, p2 = st.columns(2)
    with p1:
        fig_d, ax_d = plt.subplots(figsize=(5, 3))
        ax_d.hist(nodes["degree"], bins=min(20, max(1, len(nodes))), color="steelblue", edgecolor="black")
        ax_d.set_xlabel("Cell degree"); ax_d.set_ylabel("Cells"); ax_d.set_title("Cell degree distribution")
        ax_d.grid(alpha=0.3); st.pyplot(fig_d)
    with p2:
        fig_r, ax_r = plt.subplots(figsize=(5, 3))
        ax_r.hist(nodes["tissue_r"], bins=min(20, max(1, len(nodes))), color="darkseagreen", edgecolor="black")
        ax_r.set_xlabel("Cell-to-tissue Pearson r"); ax_r.set_ylabel("Cells")
        ax_r.set_title("Cell-to-tissue correlation"); ax_r.grid(alpha=0.3); st.pyplot(fig_r)
    st.dataframe(nodes, width="stretch", height=260)
    st.download_button("⬇ cell network nodes (CSV)", nodes.to_csv(index=False).encode(),
                       "network_nodes.csv", "text/csv", key="dl_cellnet_nodes" + sfx)
    st.download_button("⬇ cell network edges (CSV)", result["edges_df"].to_csv(index=False).encode(),
                       "network_edges.csv", "text/csv", key="dl_cellnet_edges" + sfx)
    st.download_button("⬇ cell network summary (CSV)", result["summary_df"].to_csv(index=False).encode(),
                       "network_summary.csv", "text/csv", key="dl_cellnet_summary" + sfx)


with T[TAB_NET]:
    roi_split(render_cell_network, "net")

# ---------------------------------------------------------- Downloads
with T[TAB_DL]:
    # ------------------------------------------------ save straight to a folder
    st.subheader("Save results to a folder")
    if "save_dir" not in ss:
        ss["save_dir"] = trial + "_output"
    sd1, sd2 = st.columns([1, 3])
    if sd1.button("📂 Choose folder…", width="stretch"):
        _sp, _save_picker_error = pick_folder(ss["save_dir"], "Select a folder to save the results into")
        if _sp:
            ss["save_dir"] = os.path.normpath(_sp)
            st.rerun()
        elif _save_picker_error:
            st.warning("The folder picker could not open. Paste an output path in the box instead. "
                       f"Details: {_save_picker_error}")
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
    if ss.get("tissue_cluster_df") is not None:
        _avail["Tissue-state assignments (tissue_state_assignments.csv)"] = (
            "tissue_state_assignments.csv", "csv", ss["tissue_cluster_df"])
    if R.get("link") and reg.get("mask_per_frame"):
        _avail["Tracked masks (tracked_masks.tiff)"] = ("tracked_masks.tiff", "stack", None)

    # Network analysis outputs if present
    _net_res_active = (
        ss.get("_cellnet_result_roi")
        if (roi_ids and ss.get("_cellnet_result_roi") and
            not ss.get("_cellnet_result_roi", {}).get("error"))
        else ss.get("_cellnet_result_all")
    )
    if _net_res_active and not _net_res_active.get("error"):
        _avail["Network nodes (network_nodes.csv)"] = ("network_nodes.csv", "csv", _net_res_active["nodes_df"])
        _avail["Network edges (network_edges.csv)"] = ("network_edges.csv", "csv", _net_res_active["edges_df"])
        _avail["Network summary (network_summary.csv)"] = ("network_summary.csv", "csv", _net_res_active["summary_df"])

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
    if _net_res_active and not _net_res_active.get("error"):
        st.download_button("⬇ network nodes (CSV)", _net_res_active["nodes_df"].to_csv(index=False).encode(),
                           "network_nodes.csv", "text/csv", key="dl_ind_net_nodes")
        st.download_button("⬇ network edges (CSV)", _net_res_active["edges_df"].to_csv(index=False).encode(),
                           "network_edges.csv", "text/csv", key="dl_ind_net_edges")
        st.download_button("⬇ network summary (CSV)", _net_res_active["summary_df"].to_csv(index=False).encode(),
                           "network_summary.csv", "text/csv", key="dl_ind_net_summary")
