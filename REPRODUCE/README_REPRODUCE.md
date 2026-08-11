# CAFIN processed-data reproduction of the calcium-transient study

This folder reproduces the manuscript *"Quantitatively Rapid Analysis of Calcium
Transients in Epithelial Cell Layer in Vivo upon Cytoskeleton Disruption"* end-to-end,
**from the pipeline's real data**, and regenerates the paper and all figures automatically.

It does **not** read the original PDF. Every figure and number is computed from the
per-cell ΔF/F₀ CSVs the pipeline produced for two independent trials
(`LATA1TRAIL`, `LATA2TRIAL`), each with a **baseline (BEFOREDRUG)** and a
**Latrunculin (AFTERDRUG)** recording.

---

## Rebuild the analysis, figures, and paper in one command

```bash
cd CAFIN_Cleanedup_Code_Aarush/REPRODUCE
pip install -r requirements.txt          # first time only

python run_all.py                         # does all three steps below
```

`run_all.py` uses the processed per-cell data included in GitHub to rebuild the analysis, figures,
and paper. The checked-in `regenerated/` provenance is retained. It does not claim to recreate the
microscopy acquisition or unavailable raw images.

Passing `--recompute` additionally regenerates LATA1 traces from raw TIFFs. This requires the
original `LATA1TRAIL/BEFOREDRUG/{membrane,ca2}` and `AFTERDRUG/{membrane,ca2}` folders, which are
too large to distribute through GitHub. The command now stops safely before deleting anything when
those folders are absent.

Or run the steps individually:

```bash
python recompute_from_raw.py  # (once) regenerate ΔF/F₀ from raw LATA1 TIFFs, end-to-end
python reproduce.py           # analysis + quantitative figures + statistics
python figures_paper.py       # paper-STYLE figures from the real images
python generate_paper.py      # assembles the manuscript (.docx) + PROOF.md
```

Outputs land in `REPRODUCE/`:

| File | What it is |
|---|---|
| `results/figures/figP1..figP9 .png` | **paper-style** figures: pipeline schematic, green/magenta registration overlays, elastic-vs-rigid comparison, 4-panel processing row, numbered cell mask, **background reference regions**, **ROI selection**, Ca²⁺ event raster, regional ΔF/F₀ bars |
| `regenerated/LATA1TRAIL/…` | ΔF/F₀ recomputed end-to-end from the raw images (masks, traces, ROI CSVs, provenance) |
| `results/figures/fig1..fig6 .png` | quantitative figures: traces, heatmaps, metric violins + p-values, spatial coordination, time-dependence |
| `results/metrics_per_cell.csv` | every cell's metrics (both trials, both conditions) |
| `results/field_summary.csv` | per-field tissue-level metrics |
| `results/stats_tests.csv` | Mann–Whitney U tests, p-values, effect sizes |
| `results/results.json` | machine-readable everything |
| `CAFIN_reproduced_paper.docx` | the full written paper (13 figures), numbers auto-filled |
| `PROOF.md` | claim-by-claim reproduction proof (see below) |

Analyses use a **fixed random seed**, so re-running gives byte-identical numbers.
(If `CAFIN_reproduced_paper.docx` is open in Word when you re-run, it safely writes
`CAFIN_reproduced_paper_NEW.docx` instead.)

---

## What gets reproduced

The manuscript's central claims and how each is quantified here:

1. **Single-cell activity statistics** — peak ΔF/F₀, mean ΔF/F₀, transients/frame, AUC
   per cell (`reproduce.py: cell_metrics`).
2. **Tissue-level pattern analysis** — active fraction, **spatial heterogeneity**
   (CV of per-cell peak amplitude) and **temporal synchronization**
   (mean pairwise Pearson r among active cells).
3. **Baseline vs Latrunculin comparison with statistics** — pooled cells across two trials,
   two-sided **Mann–Whitney U** + rank-biserial effect size (the paper's Methods call for
   nonparametric testing; this fills that gap).
4. **Spatial coordination / propagation** — pairwise correlation vs inter-cell centroid
   distance (Figure 4), the paper's "distinct patterns of Ca²⁺ propagation."
5. **Time-dependence of the response** — early/mid/late windows within the treated
   recordings (Figure 6).

### Reproduced result (auto-generated numbers)

| Metric | median baseline | median Latrunculin | p-value |
|---|---|---|---|
| Peak ΔF/F₀ | 2.14 | 2.75 | 4.2e-22 |
| Mean ΔF/F₀ | 0.34 | 0.98 | 1.0e-117 |
| Transients/frame | 0.097 | 0.156 | 3.5e-140 |
| Transient AUC | 5.97 | 49.2 | 4.5e-142 |
| Pairwise correlation r | 0.24 | 0.49 | <1e-300 |

Spatial heterogeneity **CV rose 0.72 → 1.17** and temporal synchronization
**r rose 0.25 → 0.45** after actin disruption — matching the abstract's claims of
*increased spatial heterogeneity* and *altered temporal synchronization*.
(Numbers above are the values produced on this dataset; `PROOF.md` always shows the live values.)

---

## Proof that the paper was reproduced

Open **`PROOF.md`** (or `CAFIN_reproduced_paper.docx`). `PROOF.md` contains:

- a table of every field analyzed (cell counts, frames, metrics),
- the full statistics table (p-values + effect sizes),
- a **claim-by-claim table** mapping each manuscript claim to the reproduced evidence,
- all six figures embedded.

Because `PROOF.md`, the `.docx`, and the CSVs are all generated from `results.json` by
script, they cannot disagree with the data. Delete `results/` and re-run the two commands
to regenerate them from scratch.

---

## Full pipeline (regenerate the input CSVs from raw microscopy images)

The CSVs above were produced by the image pipeline in the parent folder. To recreate them
from the raw TIFF stacks (registration → segmentation → ΔF/F₀):

```python
# from CAFIN_Cleanedup_Code_Aarush/
import cafin_pipeline as cp
cp.run("baseline", "LATA1TRAIL/BEFOREDRUG", mem_base="<membrane_base>", ca_base="<ca_base>")
cp.run("latrunculin", "LATA1TRAIL/AFTERDRUG", mem_base="<membrane_base>", ca_base="<ca_base>")
```

Each trial folder must contain `membrane/` and `ca2/` subfolders of numbered `.tif` frames.
`cafin_pipeline.run()` writes `all_cells_raw.csv` and `all_cells_normalized.csv`, which are
exactly the inputs `reproduce.py` consumes — so the loop is closed.

### Interactive exploration (optional)

```bash
streamlit run cafin_gui.py
```
Registration overlay (now CLAHE-enhanced for strong contrast), segmentation, traces,
tracking and statistics tabs.

---

## Improvements made in this reproduction

- **End-to-end recompute** — `recompute_from_raw.py` regenerates ΔF/F₀ for LATA1 straight
  from the raw TIFFs (registration → cyto3 segmentation → background subtraction → ΔF/F₀),
  so the whole chain is recomputed, not just read from CSVs. LATA2 (no raw TIFFs) uses its
  existing CSVs; every field's provenance is recorded in `field_summary.csv`.
- **Statistics matched to the original plan** — Shapiro–Wilk normality test now drives test
  choice (Welch t-test if both groups normal, else Mann–Whitney U); Kruskal–Wallis for R1–R4;
  rank-biserial effect sizes. AUC is per-frame normalized so 31- vs 90-frame recordings compare.
- **Background & ROI figures** — `figP8_background.png` (3 signal-free reference regions) and
  `figP9_roi.png` (rectangular ROI + intersecting cells), the two missing "Data need" items.
  ROI logic added to `cafin_core.roi_cell_ids`; background now averages the 3 medians (per manuscript).
- **Elastic-vs-rigid comparison** — `figP3` shows BEFORE / rigid / elastic with residuals; the
  large after-drug motion is mostly global, so rigid suffices (honestly reported).
- **Spatial coordination & time-dependence** — correlation-vs-distance and early/mid/late windows.
- **Per-trial transparency** — Table 1 in the paper; pooled effects are honest about trial-2's
  falling peak amplitude vs rising heterogeneity.
- **Contrast** — `two_color_overlay`/`stretch8` apply CLAHE + Otsu background gating; shared color scales.
- **Cellpose** — `segment()` uses the manuscript's `cyto3` + `channels=[2,0]` (diameter 15).
- **Robust figure numbering** — the paper generator auto-numbers figures from a registry, so
  captions and in-text references can never drift out of sync.
- **Deterministic** — fixed random seed → identical output on every run.

---

## File map

```
REPRODUCE/
├── run_all.py                 # one command → recompute (once) + analysis + figures + paper
├── recompute_from_raw.py      # regenerate ΔF/F₀ from raw LATA1 TIFFs, end-to-end
├── reproduce.py               # analysis + quantitative figures + statistics (Shapiro-Wilk)
├── figures_paper.py           # paper-style figures from real images
├── generate_paper.py          # builds .docx + PROOF.md (auto figure numbering)
├── requirements.txt
├── README_REPRODUCE.md        # this file
├── CAFIN_reproduced_paper.docx
├── PROOF.md
├── regenerated/LATA1TRAIL/…   # end-to-end recomputed masks, traces, ROI CSVs, provenance
└── results/
    ├── metrics_per_cell.csv  field_summary.csv  stats_tests.csv  results.json
    └── figures/
        ├── figP1_pipeline.png             figP6_raster.png
        ├── figP2_registration_baseline.png  figP7_region_bars.png
        ├── figP3_registration_afterdrug.png figP8_background.png
        ├── figP4_processing_row.png         figP9_roi.png
        ├── figP5_mask_numbered.png
        └── fig1_traces.png … fig6_time.png
```

## Which figure reproduces which "Data need" from the manuscript

| Manuscript "Data need" | Reproduced figure |
|---|---|
| Image & data processing pipeline (Figure 1) | `figP1_pipeline.png` |
| "Make color for registration image0 vs moving" (baseline) | `figP2_registration_baseline.png` |
| "…after drug" | `figP3_registration_afterdrug.png` |
| Registration results + Ca²⁺ data + cell mask panels | `figP4_processing_row.png` |
| "Cell mask with numbers" | `figP5_mask_numbered.png` |
| "Background reduction (pick the spots)" | `figP8_background.png` |
| "How to define region of interest" | `figP9_roi.png` |
| Tissue-level Ca²⁺ pattern (raster / kymograph) | `figP6_raster.png`, `fig2_heatmaps.png` |
| Single-cell activity statistics (R1–R4 bars, ****) | `figP7_region_bars.png`, `fig3_metrics.png` |
| Baseline vs drug quantification | `fig1_traces.png`, `fig4_spatial.png`, `fig5_pattern_metrics.png` |
| Time-dependence of the response | `fig6_time.png` |
