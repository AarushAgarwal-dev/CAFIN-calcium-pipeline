# CAFIN Calcium Analysis GUI

A Streamlit app that packs the whole pipeline (motion correction → Cellpose
segmentation → ΔF/F0 → statistics) into one window and shows every image and
graph inline.

## Install
```bash
pip install streamlit cellpose opencv-python scikit-image tifffile imageio pillow pandas scipy natsort SimpleITK
```
(`itk-elastix` is optional; on this machine it is blocked by Windows
Application Control, so the elastic mode uses **SimpleITK** B-spline instead —
the same B-spline registration described in the manuscript.)

## Run
```bash
cd CAFIN_Cleanedup_Code
streamlit run cafin_gui.py
```
A browser tab opens. In the sidebar:
1. Point **Trial folder** at a folder that contains `membrane/` and `ca2/`
   sub-folders of numbered `.tif` frames (base names are auto-detected).
2. Pick an **analysis mode** (below).
3. Open **Advanced parameters** to set cell diameter, frame sub-sampling
   (elastic/optical-flow are slow on CPU — raise "process every Nth frame"),
   background subtraction, ΔF/F0 baseline, and peak threshold.
4. Click **▶ Run analysis**.

## GPU / CUDA
The sidebar has a **"Use GPU (CUDA)"** toggle that accelerates the Cellpose
segmentation step (the registration backends stay on CPU). It auto-detects
whether a CUDA GPU is usable and shows the status; if you tick it without a
usable GPU it safely falls back to CPU.

CUDA works only with a CUDA build of PyTorch. The default install is CPU-only
(`torch ...+cpu`). To enable the GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
(match `cu124`/`cu121`/etc. to your installed CUDA driver). The standalone
scripts (`run_pipeline_fixed.py`, `cafin_pipeline.py`) auto-use the GPU when one
is present.

## The 7 modes
The manager's "4 options" are the **2×2 grid** of *registration × cell-handling*:

| # | Registration | Cells | Notes |
|---|--------------|-------|-------|
| 1 | Rigid (ECC) | static frame-0 mask | fast; good when the fish barely moves |
| 2 | Rigid (ECC) | **cell tracking** | ROI follows the tissue frame-to-frame |
| 3 | Elastic B-spline (SimpleITK) | static frame-0 mask | handles warping |
| 4 | Elastic B-spline (SimpleITK) | **cell tracking** | warping + ROI follows |

Plus three extra modes: optical-flow + static, optical-flow + tracking, and
no-registration + static (baseline sanity check).

**Static vs. tracking:** static reuses one Cellpose mask from frame 0 for every
frame (the paper's stated limitation — the ROI drifts off cells when the tissue
deforms). Tracking warps that frame-0 mask into every frame using the
per-frame transform, so each cell is followed as it moves/deforms.

## Tabs
- **Registration overlay** – heavy-contrast green/magenta overlay (green = frame 0,
  magenta = moving, white = aligned), before vs. after, with a frame slider and a
  "build GIF" button.
- **Segmentation** – frame-0 membrane with numbered cell ROIs.
- **Traces / ΔF/F0** – all per-cell traces + population mean, and a cells×frames heatmap.
- **Tracking** – tracked vs. static ROI position per frame (tracking modes only).
- **Statistics** – metrics table + violin plots (amplitude / interval / area) +
  fraction-of-active-cells curve.
- **Downloads** – raw/ΔF/F0/metrics CSVs and the overlay GIF.

## Files
- `cafin_gui.py`  – the Streamlit app (UI only).
- `cafin_core.py` – compute engine (importable/testable, no GUI).
