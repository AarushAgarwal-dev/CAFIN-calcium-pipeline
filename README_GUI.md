# CAFIN Calcium Analysis GUI

A Streamlit app that packs the whole pipeline (motion correction → Cellpose
segmentation → ΔF/F0 → statistics) into one window and shows every image and
graph inline.

## Install
```bash
python -m pip install -r requirements.txt
```
Use Python 3.10 or 3.11, preferably 3.11. This installs the pinned Cellpose 3
cyto3 environment used by CAFIN. `requirements-gui.txt` is the smaller GUI-only
list used by `install.py`; it expects PyTorch to be installed first by the installer.

## Run
```bash
cd CAFIN-calcium-pipeline
streamlit run cafin_gui.py
```
A browser tab opens. In the sidebar:
1. Point **Trial folder** at a folder that contains `membrane/` and `ca2/`
   sub-folders of numbered `.tif` frames (base names are auto-detected).
2. Pick an **analysis mode** (below).
3. Open **Advanced parameters** to set cell diameter, frame sub-sampling,
   background subtraction, ΔF/F0 baseline, and peak threshold.
4. Click **▶ Run analysis**.

## GPU backends
The sidebar's **Use GPU for segmentation** option accelerates Cellpose when a
usable backend is installed. Registration remains CPU-based. The GUI reports the
backend it found and falls back to CPU if the requested backend is unavailable.

The installer chooses a compatible backend when possible:

| Hardware | Backend |
|---|---|
| NVIDIA on Windows/Linux | CUDA |
| AMD or Intel on Windows | DirectML |
| AMD on Linux | ROCm |
| Apple Silicon | MPS |
| Intel macOS or unsupported GPU | CPU |

For a manual CUDA install:
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

The current GUI exposes three analysis methods: **Rigid (ECC)** with a static
reference mask, **Elastic (itk-elastix)** with a static reference mask, and
**Cell tracking**, which segments frames separately and links cell identities.
Tracking is available as its own method; it is not silently combined with the
registration choices. A static mask follows the paper's reference-frame
workflow, while tracking is the optional motion-aware alternative.

## Tabs
- **Registration overlay** – heavy-contrast green/magenta overlay (green = frame 0,
  magenta = moving, white = aligned), before vs. after, with a frame slider and a
  "build GIF" button.
- **Segmentation** – frame-0 membrane with numbered cell ROIs.
- **Traces / ΔF/F0** – all per-cell traces + population mean, and a cells×frames heatmap.
- **Clustering** – check whole traces and/or specific biological inputs before PCA + K-means: first-peak
  time, peak count, amplitude, width, area, inter-peak timing, per-cell activity, and tissue coupling.
  A separate tissue-state mode clusters frames from whole-field signal and active-cell fraction.
- **Network Analysis** – cell-level correlation network from the extracted ΔF/F0i traces and
  k-clique community detection based on NetworkX. One node is one segmented cell, with ROI-aware
  spatial mapping, overlapping-community tracking, degree and cell-to-tissue correlation plots,
  dense-graph safety guards, and direct CSV downloads (`network_nodes.csv`, `network_edges.csv`,
  `network_summary.csv`). Pearson R² includes strong negative relationships by default; a positive-only
  edge option is available.
- **Tracking** – tracked cell identities and motion diagnostics (tracking method only).
- **Statistics** – selectable metric tables and plots, including peak dynamics,
  tissue-level summaries, active-cell fraction, and regional comparisons.
- **Downloads** – raw/ΔF/F0/metrics/network CSVs and the overlay GIF.

## Files
- `cafin_gui.py`  – the Streamlit app (UI only).
- `cafin_core.py` – compute engine (importable/testable, no GUI).
- `tests/`        – unit test suite (`test_network_analysis.py`).
