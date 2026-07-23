# CAFIN — Quantitative Analysis of Calcium Transients in Epithelial Cell Layers *in vivo*

A reproducible image-analysis pipeline for single-cell **Ca²⁺ transient** analysis in the
zebrafish larval fin epithelium, before and after cytoskeletal disruption (Latrunculin).

It takes raw two-channel confocal time-lapse stacks (a **membrane** channel for structure and a
**calcium** channel for signal) and produces per-cell ΔF/F₀ traces, tissue-level statistics,
publication-style figures, and an auto-generated manuscript — plus an interactive GUI with
cell tracking, trace clustering, and AI-assisted interpretation.

---

## How it works

| Stage | What happens |
|---|---|
| **1. Registration** | Motion-corrects the time-lapse. Rigid = OpenCV **ECC**; non-rigid = **itk-elastix** B-spline. Registration is computed on the *membrane* channel and the transform applied to the *calcium* channel, preserving cross-channel correspondence. |
| **2. Segmentation** | **Cellpose** (`cyto3`, `channels=[2,0]`, diameter ≈ 15 px) on the reference membrane frame → an integer-labelled cell mask. |
| **3. Background subtraction** | Three signal-free reference regions per frame; values outside 1.5×IQR discarded, each region reduced to its median, the mean of the three medians subtracted and clipped at zero. |
| **4. ROI selection** | Optional rectangular region of interest — every cell whose mask intersects it is retained and exported separately. |
| **5. Trace extraction** | Mean pixel intensity inside each cell mask per frame → **ΔF/F₀ = (Fₜ − F₀)/F₀**, where F₀ is the mean of the lowest-activity frames. |
| **6. Cell tracking** *(optional)* | Segments **every** frame independently and links cells into stable global IDs — weighted IoU + size + centroid score, **Hungarian** assignment, forward/backward propagation, and gap closing. |
| **7. Analysis** | Per-cell metrics (peak ΔF/F₀, transient rate, per-frame AUC); tissue metrics — **spatial heterogeneity** (CV of peak amplitude) and **temporal synchronization** (mean pairwise correlation); spatial coordination vs inter-cell distance; time-dependence. |
| **8. Statistics** | Shapiro–Wilk normality → Welch *t*-test if normal, otherwise Mann–Whitney U; rank-biserial effect sizes; Kruskal–Wallis across regions. |

---

## How to run

### Install

```bash
pip install -r REPRODUCE/requirements.txt
```

Core: `numpy pandas scipy matplotlib scikit-image opencv-python tifffile cellpose torch`.
GUI extras: `streamlit plotly scikit-learn itk-elastix boto3`.

### A. Interactive GUI

```bash
streamlit run cafin_gui.py
```

In the sidebar set the **trial folder** (must contain `membrane/` and `ca2/` subfolders of numbered
`.tif` frames), pick a method, then click **Run analysis**.

**Analysis methods**
- **Rigid (ECC)** — global motion correction, fixed frame-0 mask.
- **Elastic (itk-elastix)** — non-rigid B-spline correction (`fast` / `balanced` / `accurate`).
- **Cell tracking** — segments every frame and links cells into stable IDs (no registration).

**Tabs**
- 🎞 **Registration + movie** — green/magenta before/after overlay, frame navigation (◀ ▶), auto-loop.
- 🧫 **Segmentation** — numbered cell ROIs.
- 🎯 **ROI** — drag a box directly on the calcium frame to restrict analysis; ROI-only CSV export.
- 📈 **Traces / ΔF/F₀** — per-cell traces and a cells × frames activity heatmap.
- 🧩 **Clustering** — PCA (5–30 components) + K-means; cells colored by cluster on the tissue,
  PC1/PC2 scatter, cluster-average traces, plus the AI narrative (below).
- 🎯 **Tracking** — stable global IDs colored across frames, with navigation + auto-loop.
- 📊 **Statistics** · ⬇ **Downloads** (CSV / GIF).

### B. Reproduce the full study (figures + paper)

```bash
cd REPRODUCE
python run_all.py
```

Runs, in order:
1. `recompute_from_raw.py` — regenerates ΔF/F₀ from the raw TIFFs end-to-end
   *(skipped if `regenerated/` exists; pass `--recompute` to force)*
2. `reproduce.py` — analysis, statistics, quantitative figures
3. `figures_paper.py` — paper-style figures from the real images
4. `generate_paper.py` — assembles the manuscript (`.docx`) and `PROOF.md`

Outputs land in `REPRODUCE/results/` (CSVs, `results.json`, 15 figures) plus
`CAFIN_reproduced_paper.docx`. A fixed random seed makes re-runs byte-identical.
Details: [`REPRODUCE/README_REPRODUCE.md`](REPRODUCE/README_REPRODUCE.md).

### C. AI interpretation (optional — Amazon Bedrock)

The Clustering tab can turn the clustering into a written "Findings" narrative using
**open-source models on Amazon Bedrock** (Llama 3.3 70B by default; Llama 3.1, DeepSeek-R1 and
Mixtral also selectable) through the model-agnostic Converse API.

1. Authenticate — `aws login` (or `aws configure`)
2. Enable model access for your chosen model in the Bedrock console, in your region
3. In the GUI: fill in **Background / context** (drug, concentration, protocol…), click
   **🔌 Test connection**, then either:
   - **✍ Story from clusters** — per-cluster summary statistics
   - **🎞 Full temporal analysis** — the entire time-course across all frames

The model only ever sees the numeric summaries shown in the *"data sent to the model"* expander.
Treat its output as hypotheses, not conclusions.

---

## Repository layout

```
cafin_core.py        registration (ECC / itk-elastix), segmentation, background, ΔF/F₀, ROI, clustering
cafin_track.py       cell tracking — per-frame segmentation linked by Hungarian matching + gap closing
cafin_pipeline.py    scripted end-to-end pipeline (no GUI)
cafin_gui.py         Streamlit application
cafin_ai.py          Amazon Bedrock (open-source models) narration of clustering results
REPRODUCE/           fully reproducible study: scripts, results, figures, generated paper
  ├── run_all.py             one command: recompute → analyse → figures → paper
  ├── reproduce.py           metrics, statistics, quantitative figures
  ├── figures_paper.py       paper-style figures from real images
  ├── generate_paper.py      builds the manuscript + PROOF.md
  ├── recompute_from_raw.py  regenerates ΔF/F₀ from raw TIFFs
  ├── WORKLOG.md             full record of the analysis and its caveats
  └── results/               CSVs, results.json, figures/
LATA1TRAIL/, LATA2TRIAL/     per-trial processed data (before / after drug)
```

Raw microscopy stacks (`.tif`, `.nd2`) and other large binaries are **not** tracked — only the
processed per-cell CSVs needed to re-run the analysis.

---

## Data

Each trial folder holds a **BEFOREDRUG** (baseline) and **AFTERDRUG** (Latrunculin) condition with:

| File | Contents |
|---|---|
| `all_cells_raw.csv` | raw per-cell mean intensity per frame |
| `all_cells_normalized.csv` | per-cell ΔF/F₀ |
| `roi_cells_normalized.csv` | ΔF/F₀ restricted to the ROI |
| `centroids_0.csv` | cell centroids (for spatial analysis) |

To run from raw images, a trial folder needs `membrane/` and `ca2/` subfolders of numbered `.tif`
frames (e.g. `..._0000.tif`, `..._0001.tif`, …).

---

## Notes and limitations

- **Per-frame Cellpose segmentation** (the tracking method) is slow on CPU (~30–60 s/frame). Use the
  **frame-step** control to subsample, or install a CUDA build of torch.
- **itk-elastix** non-rigid registration runs ≈ 0.5 / 3 / 5 s per frame for fast / balanced / accurate.
- **Statistical caveat**: pooled per-cell p-values treat individual cells as replicates. With only two
  biological trials per condition, the **per-trial direction of effect** is the primary evidence —
  see the generated paper and `REPRODUCE/WORKLOG.md`.

---

## Acknowledgements

Cell-tracking algorithm after **Linlin Li** (`Cell_Tracking_2D`), extended with optimal Hungarian
assignment and gap closing. Segmentation uses [Cellpose](https://github.com/MouseLand/cellpose);
non-rigid registration uses [ITKElastix](https://github.com/InsightSoftwareConsortium/ITKElastix).
