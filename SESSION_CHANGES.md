# Session changes — CAFIN pipeline fix + GUI

Everything created, fixed, and run in this working session, in one place.

---

## 1. Why the pipeline wasn't working (root causes)

The numbers previously in the manuscript could not have been produced by the
code, because the code failed before producing any per-cell data:

1. **Cellpose 4.x API break (main cause).** The notebook targeted old cellpose
   (`CellposeDenoiseModel`, `model_type="cyto3"`, `channels=[2,0]`). The installed
   version is **cellpose 4.2.1.1 (cpsam)**, where all three were removed →
   segmentation crashed → no mask → all downstream results were empty.
2. **Rigid registration never saved the calcium reference frame `0000`.** Only the
   membrane frame 0 was saved, so frame 0 — the ΔF/F0 baseline — was silently
   dropped in background subtraction, ROI, and analysis.
3. **Interactive-only steps** (background-region picking and ROI via `TkAgg`)
   could not run headless, so the pipeline could not complete unattended.
4. **Low-contrast overlay** — the registration comparison blended into dim
   blue-grey instead of the color overlay the manuscript asks for.
5. **Hardcoded paths** pointed at another user (`C:\Users\kvs62\...`).
6. Environment: `itk-elastix` is **blocked by Windows Application Control** on this
   machine (native DLL fails to load); `nd2` not installed. Elastic registration
   was therefore moved to **SimpleITK** B-spline (same method, works).

---

## 2. Files created

All under `CAFIN_Cleanedup_Code/`:

| File | Purpose |
|------|---------|
| `cafin_core.py` | Compute engine: rigid/elastic/optical-flow registration, Cellpose seg, static + tracking cell handling, background subtraction, ΔF/F0, metrics, plotting helpers, CUDA detection |
| `cafin_gui.py` | Streamlit GUI showing all images/graphs inline; 7 modes; GPU toggle |
| `run_pipeline_fixed.py` | Corrected single-trial headless run (rigid) + heavy-contrast overlay |
| `cafin_pipeline.py` | Parameterized engine for batch runs (corrected ΔF/F0 = raw + floor) |
| `run_all_conditions.py` | Runs all 4 conditions → `results_summary.json` |
| `results_summary.json` | Real metrics for the 4 conditions |
| `README.md` | Full project + how-to + GPU + vast.ai SSH instructions |
| `README_GUI.md` | GUI-specific detail |
| `SESSION_CHANGES.md` | This file |
| `CafinPaper1_BACKUP.ipynb` | Copy of the notebook before edits |

Output folders produced by runs: `lat_trial1_afterdrug_output/`,
`Total data/<trial>_output/` (registered stacks, masks, overlay GIF/PNG, CSVs).

Also, in `Downloads/`: a backup of the Word document was created before editing it.

---

## 3. Files modified

* **`CafinPaper1.ipynb`** — patched in place (backup kept):
  * cell 9 (config): local auto-detecting paths + frame count.
  * cell 11 (registration): also saves the calcium reference frame `0000`.
  * cell 13 (segmentation): rewritten for cellpose 4.x (cpsam; no
    `model_type`/`channels`/`DenoiseModel`); GPU auto-detect.
  * cells 15 & 18 (overlay): heavy-contrast green/magenta frame-0-vs-moving overlay.
* **The manuscript `.docx`** — the placeholder Results section was replaced with a
  real write-up + **Table 1** of per-condition metrics (backup saved in Downloads).

---

## 4. Real results (produced by the corrected pipeline)

ΔF/F0 computed from the registered raw calcium with an F0 floor.

| Metric | T1 before | T1 after | T2 before | T2 after |
|--------|-----------|----------|-----------|----------|
| Cells segmented | 501 | 542 | 506 | 475 |
| Frames analyzed | 31 | 90 | 31 | 55 |
| Active cells (%) | 97.4 | 76.6 | 89.5 | 90.9 |
| Mean peak ΔF/F0 | 2.94 | 0.95 | 1.28 | 0.99 |
| Median peak ΔF/F0 | 1.45 | 0.86 | 1.08 | 0.81 |
| Transient rate (peaks·cell⁻¹·frame⁻¹) | 0.076 | 0.069 | 0.058 | 0.090 |
| Spatial heterogeneity (CV) | 1.01 | 0.58 | 0.56 | 0.62 |
| Temporal synchronization (r) | 0.18 | 0.64 | 0.28 | 0.21 |

### Caveats you must weigh before using these in the paper
* The results are **mixed and do not cleanly support the current Discussion**
  (which claims LatA → increased heterogeneity, reduced synchronization). Trial 1
  shows the opposite; trial 2 is weak/mixed.
* **Trial 1 before-drug is a MAX-intensity projection** while its after-drug is an
  AVG projection → amplitudes not directly comparable. **Trial 2 (AVG vs AVG) is
  the fair comparison.**
* **Before (31 frames) vs after (55–90 frames)** — unequal windows bias peak
  counts and correlation estimates.
* Only two trials per condition → not statistically conclusive; add replicates
  with matched acquisition before drawing mechanistic conclusions.

---

## 5. What was verified (not just written)

* All **8 registration × tracking combinations** run on real data.
* Elastic (SimpleITK B-spline) registration and tracking run.
* Streamlit server boots (HTTP 200); a full **end-to-end run through the app**
  completes with **no exceptions** (6 tabs render, metrics compute, plots draw).
* GPU toggle renders, detection returns correct status + install hint, CPU
  fallback works.

---

## 6. Follow-ups offered (not yet done)

* Re-run with **matched frame windows** and AVG-only data for a fair
  control-vs-treatment comparison.
* Generate per-cell trace figures / violin plots as manuscript figures.
* Revise the Discussion to match the real findings.
* Wire the 7 modes into the notebook as well as the GUI.
