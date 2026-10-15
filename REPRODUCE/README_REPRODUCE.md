# CAFIN processed-data reproduction of the calcium-transient study

This folder rebuilds the two-trial quantitative analysis for the draft manuscript
*"Quantitatively Rapid Analysis of Calcium Transients in Epithelial Cell Layer
in Vivo upon Cytoskeleton Disruption"* from the study's processed per-cell data.
It regenerates six quantitative figures and assembles an 18-figure draft paper
and evidence report. The other 12 figure assets include microscopy panels,
an active-cell fraction plot, and two spatial activity maps. The local run
regenerated those assets using trial-1 source TIFFs, matching masks, and
corrected traces; a checkout without those source images uses the checked-in
PNGs where needed.

It does **not** read the original manuscript or PDF. The quantitative results
come from stored per-cell raw-fluorescence CSVs for two trials
(`LATA1TRAIL`, `LATA2TRIAL`), each with a **baseline (BEFOREDRUG)** and a
**Latrunculin A (AFTERDRUG)** recording. Generated tables abbreviate the
compound to “Latrunculin.” Each cell gets its own baseline F₀ᵢ: the mean of
its ten lowest raw-fluorescence frames, floored at 1. Its normalized trace is
ΔF/F₀ᵢ = (Fₜᵢ − F₀ᵢ) / F₀ᵢ. The baseline frames can differ between cells.

---

## Rebuild the available analysis and paper in one command

```bash
cd CAFIN-calcium-pipeline/REPRODUCE       # use your actual checkout path
python -m pip install -r requirements.txt # first time only

python run_all.py
```

Without raw TIFFs, `run_all.py` first recalculates F₀ᵢ and ΔF/F₀ᵢ from the
stored per-cell raw-fluorescence tables, saving each field separately under
`normalized_per_cell/`. It then rebuilds the statistics, six quantitative
figures, `.docx`, and `PROOF.md`. It uses the included image-derived PNGs where
the source TIFFs and matching masks are unavailable. This processed-data run does not
repeat registration, Cellpose segmentation, or fluorescence extraction from
images. The included PDF is a static snapshot; the runner regenerates the
`.docx`, not the PDF.

Four source TIFF channel pairs (two trials × before/after, each with membrane
and calcium images) were found in a local external data folder; they are
**not bundled in GitHub**. With the trial-1 images available, `--recompute`
reruns registration, segmentation, trace
extraction, and per-cell F₀ᵢ normalization before the analysis and figures.
Point `CAFIN_RAW_DATA_ROOT` to the external “Total data” folder containing
`lat_trial1_beforedrug/{membrane,ca2}` and
`lat_trial1_afterdrug/{membrane,ca2}`, or place the four TIFF sequences in the
equivalent repository trial folders. A processed-data run can also regenerate
microscopy panels when matching masks are available: point `CAFIN_MASK_ROOT`
to a folder containing `BEFOREDRUG/mask_0.tiff` and
`AFTERDRUG/mask_0.tiff`. With raw TIFFs, install the image-processing
dependencies in `../requirements-gui.txt` as well as this folder's requirements.
The full raw recompute requires Cellpose 3.x cyto3 and stops on a newer model
version so segmentation does not silently change.

Or run the steps individually:

```bash
python recompute_from_raw.py   # requires original LATA1 TIFFs; optional
python normalize_f0_per_cell.py # per-cell baselines from stored raw-fluorescence CSVs
python reproduce.py            # analysis + quantitative figures + statistics
python figures_paper.py        # requires source LATA1 TIFFs and matching masks
python generate_paper.py       # assembles the manuscript (.docx) + PROOF.md
```

Outputs land in `REPRODUCE/`:

| File | What it is |
|---|---|
| `results/figures/figP*.png` | Nine paper-style figures; generated locally from trial-1 sources and reused from the repository when they are absent |
| `results/figures/figR4..figR6*.png` | Active-cell fraction by frame and two static-mask spatial activity maps |
| `regenerated/LATA1TRAIL/…` | Stored LATA1 traces previously extracted from TIFFs, centroid/ROI CSVs, and provenance; original TIFFs and masks are not included |
| `normalized_per_cell/{LATA1TRAIL,LATA2TRIAL}/…` | Corrected ΔF/F₀ᵢ, each cell's F₀ᵢ and selected baseline frames, ROI, centroids, and source checksums |
| `source_repairs/lata2_before_frame0.*` | Recovered trial-2 baseline frame-0 fluorescence row and TIFF/mask validation record |
| `results/figures/fig1..fig6 .png` | quantitative figures: traces, heatmaps, metric violins + p-values, spatial coordination, time-dependence |
| `results/metrics_per_cell.csv` | every cell's metrics (both trials, both conditions) |
| `results/field_summary.csv` | per-field tissue-level metrics |
| `results/stats_tests.csv` | Mann–Whitney U tests, p-values, effect sizes |
| `results/results.json` | machine-readable everything |
| `CAFIN_reproduced_paper_per_cell.docx` | current draft paper with 18 figures and per-cell F₀ᵢ results |
| `CAFIN_reproduced_paper_per_cell.pdf` | current PDF snapshot exported from the corrected DOCX |
| `CAFIN_reproduced_paper.docx/.pdf` | earlier version retained for comparison |
| `PROOF.md` | claim-by-claim reproduction proof (see below) |
| `SIX_TRIAL_AUDIT.md` | separate exploratory audit of six local before/after pairs; not the paper's primary result |

The analysis uses a **fixed random seed** for repeatable numerical results.
(If the current per-cell DOCX is open in Word when you re-run, it safely writes
`CAFIN_reproduced_paper_per_cell_NEW.docx` instead.)

---

## What gets reproduced

The manuscript's central claims and how each is quantified here:

1. **Single-cell activity statistics** — peak ΔF/F₀, mean ΔF/F₀, transients/frame, AUC
   per cell (`reproduce.py: cell_metrics`).
2. **Tissue-level pattern analysis** — active fraction, **spatial heterogeneity**
   (CV of per-cell peak amplitude) and **temporal synchronization**
   (mean pairwise Pearson r among active cells).
3. **Baseline vs Latrunculin A comparison with statistics** — Shapiro–Wilk selects
   Welch's t-test or a two-sided Mann–Whitney U test, with a rank-biserial
   effect size. Pooled cell-level p-values are descriptive because the two
   biological trials, not individual cells, are the independent replicates.
4. **Spatial coordination / propagation** — pairwise correlation vs inter-cell centroid
   distance (Figure 4), the paper's "distinct patterns of Ca²⁺ propagation."
5. **Time-dependence of the response** — early/mid/late windows within the treated
   recordings (Figure 6).

### Current two-trial results with corrected per-cell F₀ᵢ (pooled cells; descriptive)

| Metric | median baseline | median Latrunculin A | p-value |
|---|---|---|---|
| Peak ΔF/F₀ᵢ | 3.156 | 4.164 | 1.18e-27 |
| Mean ΔF/F₀ᵢ | 0.803 | 1.666 | 5.44e-127 |
| Transients/cell/frame | 0.0968 | 0.200 | 1.56e-215 |
| Supra-threshold AUC/frame | 0.496 | 1.235 | 1.66e-100 |
| Pairwise correlation r | 0.221 | 0.486 | <1e-300 |

Spatial heterogeneity **CV rose 0.65 → 1.04** and temporal synchronization
**r rose 0.23 → 0.44** after actin disruption — matching the abstract's claims of
*increased spatial heterogeneity* and *altered temporal synchronization*.
These are same-data, same-code results, not an independent replication.
Peak amplitude decreased in LATA2 (median 3.04 → 2.45) despite the pooled
increase, so consult the per-trial results in `PROOF.md` before interpreting
that metric. The pooled cell and cell-pair p-values do not represent tests
across independent animals.

The separate [six-trial audit](SIX_TRIAL_AUDIT.md) evaluates locally available
verification assets as an exploratory sensitivity analysis. Those assets use
all-rigid registration, whereas the reference method uses elastic registration
after treatment; their masks and traces may therefore differ. The two-trial
analysis above remains the paper's primary result. In the six-trial audit,
none of five tested metrics reaches false-discovery-adjusted q < 0.05 after
matching recordings to the first 30 frames. An elastic-registration pilot
had a severe background-region/F₀ floor problem, so it does not support a
causal claim that elastic registration performs worse than rigid registration.

---

## Evidence and limits of this reproduction

Open **`PROOF.md`** (or `CAFIN_reproduced_paper_per_cell.docx`). `PROOF.md` contains:

- a table of every field analyzed (cell counts, frames, metrics),
- the full statistics table (p-values + effect sizes),
- a **claim-by-claim table** mapping each manuscript claim to the reproduced evidence,
- all 18 figures linked, including the active-cell fraction plot and two
  static-mask activity maps.

The numerical tables in `PROOF.md` and the `.docx` are generated from
`results.json`. With processed data only, image-derived PNGs remain the
previously generated assets. Keep them in `results/figures/` when rerunning.
The six-trial audit is a separate local-data analysis and is not substituted
into the paper's tables or figures.

The trial-2 baseline raw-fluorescence table had an empty frame-0 row. That
row was recovered from its local source calcium TIFF with the saved 479-cell
mask and background subtraction. Extraction from the matching registered
frame 1 reproduced the stored values with maximum absolute error
2.27×10⁻¹³. The recovered row and file checksums are recorded in
`source_repairs/`; the original source CSV remains intact. This repair affects
the corrected F₀ᵢ and the current numbers above, but does not amount to a
full reprocessing of trial-2 images.

---

## Full pipeline (regenerate the input CSVs from raw microscopy images)

The stored raw-fluorescence CSVs were produced by an earlier image-processing
run. To recreate raw traces from available TIFF stacks (registration →
segmentation → background subtraction → fluorescence extraction), use
`python run_all.py --recompute` with the source folders described above. The
script then applies the corrected per-cell F₀ᵢ calculation and continues
through the two-trial analysis. For a general trial, the parent pipeline API is:

```python
# from the repository root
import cafin_pipeline as cp
cp.run("baseline", "LATA1TRAIL/BEFOREDRUG", mem_base="<membrane_base>", ca_base="<ca_base>")
cp.run("latrunculin_a", "LATA1TRAIL/AFTERDRUG", mem_base="<membrane_base>", ca_base="<ca_base>")
```

Each trial folder must contain `membrane/` and `ca2/` subfolders of numbered `.tif` frames.
`cafin_pipeline.run()` writes `all_cells_raw.csv` and `all_cells_normalized.csv` using
the per-cell F₀ᵢ method. For the paper, `normalize_f0_per_cell.py` writes its
own auditable copies under `normalized_per_cell/`, which `reproduce.py` uses.

### Interactive exploration (optional)

```bash
streamlit run cafin_gui.py
```
Registration overlay (now CLAHE-enhanced for strong contrast), segmentation, traces,
tracking and statistics tabs.

---

## Improvements made in this reproduction

- **Per-cell baseline correction** — the default processed-data run recalculates
  ΔF/F₀ᵢ separately for every cell from its stored raw-fluorescence trace and
  records F₀ᵢ, selected frames, and source checksums. This step does not rerun
  microscopy image processing.
- **Optional raw recompute** — when original LATA1 TIFFs are supplied,
  `recompute_from_raw.py` reruns rigid ECC registration → cyto3 segmentation →
  background subtraction → raw fluorescence extraction → per-cell ΔF/F₀ᵢ.
  The repository includes previously extracted LATA1 traces, but not its raw
  images or masks. LATA2 uses stored pipeline traces plus the validated
  frame-0 repair for its baseline field; `field_summary.csv` records each
  field's data source.
- **Additional figures** — active-cell fraction by frame (`figR4`) and peak
  and selected-timepoint activity maps (`figR5`, `figR6`) use the corrected
  traces. The maps place values on saved static cell masks; they do not
  establish longitudinal cell identities or calcium-wave propagation.
- **Statistics matched to the original plan** — Shapiro–Wilk normality test now drives test
  choice (Welch t-test if both groups normal, else Mann–Whitney U); Kruskal–Wallis for R1–R4;
  rank-biserial effect sizes. AUC is per-frame normalized so 31- vs 90-frame recordings compare.
- **Background & ROI figures** — `figP8_background.png` (3 signal-free reference regions) and
  `figP9_roi.png` (rectangular ROI + intersecting cells), the two missing "Data need" items.
  ROI logic added to `cafin_core.roi_cell_ids`; background now averages the 3 medians (per manuscript).
- **Elastic-vs-rigid comparison** — `figP3` shows before, rigid, and elastic
  overlays with residuals. The two-trial outputs here were processed with
  rigid registration; the exploratory six-trial assets are also all-rigid.
  These outputs do not establish equivalence to the reference paper's elastic
  after-treatment processing.
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
├── run_all.py                 # stored raw-trace normalization; optional raw-image recompute
├── recompute_from_raw.py      # regenerate ΔF/F₀ from raw LATA1 TIFFs, end-to-end
├── normalize_f0_per_cell.py   # per-cell F₀ᵢ from stored raw-fluorescence CSVs
├── recover_lata2_frame0.py    # validates and recovers trial-2 baseline frame 0
├── reproduce.py               # analysis + quantitative figures + statistics (Shapiro-Wilk)
├── figures_paper.py           # paper-style figures from real images
├── generate_paper.py          # builds .docx + PROOF.md (auto figure numbering)
├── SIX_TRIAL_AUDIT.md          # separate exploratory local-data sensitivity analysis
├── requirements.txt
├── README_REPRODUCE.md        # this file
├── CAFIN_reproduced_paper_per_cell.docx
├── PROOF.md
├── regenerated/LATA1TRAIL/…   # prior trace, ROI, centroid CSVs and provenance
├── normalized_per_cell/…       # corrected per-cell F₀ᵢ tables and provenance
├── source_repairs/…            # recovered frame-0 row and source validation
└── results/
    ├── metrics_per_cell.csv  field_summary.csv  stats_tests.csv  results.json
    └── figures/
        ├── figP1_pipeline.png             figP6_raster.png
        ├── figP2_registration_baseline.png  figP7_region_bars.png
        ├── figP3_registration_afterdrug.png figP8_background.png
        ├── figP4_processing_row.png         figP9_roi.png
        ├── figP5_mask_numbered.png
        ├── fig1_traces.png … fig6_time.png
        └── figR4_active_cell_fraction.png, figR5_peak_activity_static_masks.png,
            figR6_timepoint_activity_static_masks.png
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
| Active-cell fraction through the recording | `figR4_active_cell_fraction.png` |
| Spatial distribution of per-cell peak activity | `figR5_peak_activity_static_masks.png` |
| Activity at selected timepoints on static masks | `figR6_timepoint_activity_static_masks.png` |
