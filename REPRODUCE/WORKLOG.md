# CAFIN Reproduction — Research Work Log

A complete record of the work performed to reproduce *"Quantitatively Rapid Analysis of
Calcium Transients in Epithelial Cell Layer in Vivo upon Cytoskeleton Disruption"*
(Loo, Tan, Amiri Hezaveh, Ding, Buganza Tapole, Umulis, Deng, Li — draft manuscript).

---

## 1. Objective

The original document is a **draft** with placeholder methods and an outline Results section
(no numbers or figures of its own). The goal was to (a) assess how close the existing CAFIN
code was to the manuscript, (b) build a fully reproducible pipeline that generates the figures
and quantitative results the draft calls for, from the group's own data, and (c) produce a
written paper and proof of reproduction — while keeping the statistics and framing honest.

## 2. Data and provenance

| Trial | Condition | Cells | Frames | Provenance |
|---|---|---|---|---|
| LATA1 | Baseline (before drug) | 484 | 31 | **Recomputed end-to-end from raw TIFFs** |
| LATA1 | Latrunculin (after drug) | 517 | 90 | **Recomputed end-to-end from raw TIFFs** |
| LATA2 | Baseline | 479 | 31 | Existing pipeline CSVs (raw TIFFs unavailable) |
| LATA2 | Latrunculin | 452 | 55 | Existing pipeline CSVs (raw TIFFs unavailable) |

## 3. Work performed

### 3.1 Initial assessment (original draft vs existing code)
- Extracted and read the full manuscript; mapped each Methods subsection and each "Data need"
  / Results outline item to the existing code (`cafin_core.py`, `cafin_pipeline.py`, `cafin_gui.py`).
- Found the image-processing backbone (registration → segmentation → ΔF/F₀ traces) faithful,
  but flagged gaps: no baseline-vs-drug comparison, no statistics, no time-dependence, no spatial
  propagation analysis, ROI selection missing from the cleaned modules, Cellpose model mismatch.

### 3.2 Reproducible analysis package (`reproduce.py`)
- Loads per-cell ΔF/F₀ + centroids for both conditions of both trials.
- Per-cell metrics: peak ΔF/F₀, mean ΔF/F₀, transient count/rate (SciPy `find_peaks`), per-frame
  supra-threshold area (AUC), active flag.
- Tissue-level metrics: active fraction, **spatial heterogeneity** (CV of per-cell peak amplitude),
  **temporal synchronization** (mean pairwise Pearson r among active cells).
- **Spatial coordination**: pairwise correlation vs inter-cell centroid distance.
- **Time-dependence**: early/mid/late post-treatment windows.
- Deterministic (fixed seed). Outputs: `metrics_per_cell.csv`, `field_summary.csv`,
  `stats_tests.csv`, `results.json`.

### 3.3 Statistics
- **Shapiro–Wilk** normality test on each group → test selection: Welch t-test if both normal,
  else **Mann–Whitney U** (matches the draft's stated decision rule). All metrics were non-normal
  → Mann–Whitney used throughout. Rank-biserial effect sizes reported. Kruskal–Wallis for R1–R4.
- Corrected a **frame-count confound**: AUC normalized per frame (baseline 31 vs treated 55–90 frames).
- Addressed **pseudoreplication** (see §5): per-trial direction is the primary evidence; pooled
  per-cell p-values are reported as descriptive only.

### 3.4 Figures (15 total)
**Paper-style (reproduce the original figure layouts), from real LATA1 images:**
- `figP1` pipeline schematic; `figP2` baseline registration overlay (green/magenta, CLAHE);
  `figP3` rigid-vs-elastic registration comparison with residuals; `figP4` 4-panel processing row;
  `figP5` numbered cell mask; `figP6` Ca²⁺ event raster; `figP7` regional R1–R4 bars;
  `figP8` background reference regions; `figP9` ROI selection.

**Quantitative:** `fig1` traces, `fig2` heatmaps, `fig3` metric violins + p-values,
  `fig4` spatial coordination, `fig5` tissue pattern metrics, `fig6` time-dependence.

### 3.5 End-to-end recompute from raw (`recompute_from_raw.py`)
- Regenerated ΔF/F₀ for LATA1 straight from the raw TIFFs through the documented chain:
  rigid ECC registration (membrane → frame 0, applied to calcium) → Cellpose **cyto3**
  (channels=[2,0], diameter 15) → 3-region background subtraction (1.5×IQR, average of medians)
  → per-cell mean intensity → ΔF/F₀ → ROI export.
- Validation: regenerated baseline ΔF/F₀ (p50 = 0.06) closely matched the original CSVs (0.058),
  confirming the chain is faithful. LATA2 kept its existing CSVs (no raw TIFFs).

### 3.6 Generated paper (`generate_paper.py`)
- Writes `CAFIN_reproduced_paper.docx` (Abstract → Introduction → Methods → Results → Discussion
  → Reproducibility → Appendix), 15 embedded figures, Table 1 (per-trial), Table 2 (stats).
- **Auto figure-numbering registry** so captions and in-text references cannot drift out of sync.
- All numbers injected from `results.json` — the document cannot disagree with the data.
- Also emits `PROOF.md` (claim-by-claim reproduction evidence).

### 3.7 Adversarial verification (multi-agent workflow)
- Ran a 9-agent review: five independent reviewers (methods fidelity, figure completeness,
  statistical correctness, internal consistency, data provenance/honesty), each finding
  adversarially verified, then synthesized. Verdict: **GOOD_WITH_GAPS**.
- Confirmed: internal consistency clean, figure/data-need coverage complete, text↔code agree.
- Fixed every confirmed finding (see §5).

## 4. Key quantitative findings

**Cross-trial consistent (same direction in BOTH independent trials):**
- Transient frequency, mean ΔF/F₀, spatial heterogeneity (CV), temporal synchronization all
  **increased** after Latrunculin.
- Field-mean spatial heterogeneity CV: **0.72 → 1.13**; temporal synchronization r: **0.25 → 0.45**.
- Spatial coordination (pairwise r vs distance) elevated at all inter-cell distances after treatment.

**Per-trial (Table 1):**
- LATA1: peak ΔF/F₀ 2.14→5.40, CV 0.77→0.92, sync 0.26→0.53.
- LATA2: peak ΔF/F₀ 2.14→1.64 (**decreased**), CV 0.68→1.34, sync 0.24→0.37.
- ⇒ The robust signature is **increased frequency + heterogeneity**, *not* a uniform amplitude rise
  (peak amplitude was not consistent across trials).

**Pooled single cells (descriptive only — see §5):** all metrics Mann–Whitney U, p from 1.7e-27
to <1e-300; medians — transient rate 0.097→0.178, mean ΔF/F₀ 0.36→1.11, AUC 0.22→0.74, pairwise r 0.24→0.49.

## 5. Statistical caveat (pseudoreplication) and honesty fixes

The pooled per-cell comparisons pool ~960 cells from only **2 biological trials** per condition,
so those p-values overstate biological significance (cells are not independent replicates). The
paper therefore **leads with the per-trial direction of effect** and treats pooled p-values as
descriptive. Additional honesty fixes made after the verification workflow:
- "Two-way rigid+elastic" reworded — rigid was used for all analysis; elastic evaluated but not needed.
- Background subtraction stated as calcium-channel only, regions auto-selected (scripted proxy for
  the draft's manual selection).
- Peak-amplitude length bias disclosed; length-normalized metrics emphasized as the sound ones.
- Mixed provenance disclosed at the point the pooled numbers appear (not only in §5).
- Paper explicitly labeled a same-data/same-code reproduction of the authors' own draft, not an
  independent replication.
- Time-dependence claim softened (data don't extend far enough to establish recovery/adaptation).

## 6. Core code changes (`cafin_core.py`)
- `stretch8` + `two_color_overlay`: CLAHE local contrast + Otsu background gating (clean overlays).
- `segment`: uses the manuscript's `cyto3` model + `channels=[2,0]` (diameter 15).
- `bg_subtract`: averages the 3 region medians (per manuscript, was median-of-medians).
- `roi_cell_ids`: new — cells whose mask intersects a rectangular ROI (the missing ROI logic).

## 7. Deliverables (in `REPRODUCE/`)
- Code: `recompute_from_raw.py`, `reproduce.py`, `figures_paper.py`, `generate_paper.py`,
  `run_all.py`, `requirements.txt`.
- Docs: `README_REPRODUCE.md`, `PROOF.md`, this `WORKLOG.md`.
- Paper: `CAFIN_reproduced_paper.docx` (+ `.pdf`).
- Data: `results/` (CSVs, `results.json`, 15 figures), `regenerated/LATA1TRAIL/` (recomputed
  masks, traces, ROI CSVs, provenance).
- Plus updated `cafin_core.py` (one level up).

**One command reproduces everything:** `python run_all.py` (recompute → analysis → figures → paper).

## 8. Known limitations
- Only 2 biological replicates per condition; statistical power for biological significance is limited.
- LATA2 could not be recomputed end-to-end (raw TIFFs not in the dataset).
- Cellpose segmentation uses a single reference frame (no cross-time cell tracking).
- Written prose was AI-assisted and has not had a final human editing pass; data and figures verified.
