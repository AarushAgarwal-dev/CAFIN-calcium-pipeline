# Exploratory six-trial audit

This analysis uses six distinct named before/after pairs from the local
`Ca2Data_fin/verification_work/assets` folder. It is **not a reproduction of
the reference paper** and should not replace its figures or biological claims.
The `baseline` and `afterdrug` folders are byte-identical reprocessings of the
trial 3 source traces and are excluded to avoid counting that trial twice.

The twelve selected recordings each have a locally available original ND2 or
TIFF image, a background-corrected raw fluorescence table, saved per-cell
F0i and ΔF/F0i values, a 2D cell-label mask, and processing metadata. The
mask labels exactly match the trace cell columns. There are 30–92 frames and
355–617 cells per recording. Independent recomputation reproduced all saved
F0i and ΔF/F0i values within floating-point tolerance. F0i is the mean of
each cell's **own** ten lowest raw fluorescence frames, with a floor of 1.

## Trial-level comparison

The full-recording calculation preserves the stored F0i method. The paired
trial, rather than the cell, is the statistical unit (`n=6`); cell-level plots
are descriptive. Two-sided exact Wilcoxon signed-rank tests are exploratory.
The q values adjust the five displayed metrics within each analysis by the
Benjamini–Hochberg method.

| Metric | Full recording paired p | Full q | First-30-frame paired p | First-30 q |
|---|---:|---:|---:|---:|
| Mean detected peak ΔF/F0i | 0.03125 | 0.15625 | 0.4375 | 0.703125 |
| Transients per cell per frame | 0.0625 | 0.15625 | 0.5625 | 0.703125 |
| Temporal synchrony, mean r | 0.21875 | 0.273438 | 0.3125 | 0.703125 |
| Peak heterogeneity, CV | 0.4375 | 0.4375 | 0.5625 | 0.703125 |
| Active cell fraction | 0.09375 | 0.15625 | 1.0 | 1.0 |

Several after recordings are much longer than their before counterparts.
For the sensitivity analysis, each recording is limited to its first 30
frames and each cell's F0i is recomputed from its ten lowest frames in that
window. No metric reaches q < 0.05. This contrast shows that the apparent
full-recording peak change is sensitive to recording length and baseline
window choice. The first 30 frames are not guaranteed to represent the same
biological time without the original acquisition timing records.

The trial 4 after recording is also a strong outlier: 40 of 408 cells have
F0i fixed at the floor of 1. Its field mean of per-cell peak ΔF/F0i is 86.12
with those cells and 43.62 when they are excluded. Other cells with F0i above
1 can still have unusually large ratios. These values need raw-image and
background-subtraction review before interpretation.

A separate full 92-frame pilot applied balanced elastic registration to this
trial while retaining its saved frame-0 mask. It completed without registration
failures, but 247/408 cells then reached the F0 floor and the median peak rose
to 550.0 (versus 40/408 and 47.5 in the existing rigid output). This does not
isolate an effect of elastic registration: at a late frame, the three automatic
background boxes sampled bright tissue after elastic alignment (estimate 416),
whereas rigid black warp padding yielded an estimate of 0. The elastic
background subtraction clipped most cell pixels to zero. A stable background
selection and cell-quality review are needed before an elastic six-trial paper
run can be interpreted.

## Why the reference paper is not yet matched

- The locally processed verification assets used rigid ECC registration; the
  reference paper specifies elastic registration for after-treatment data.
- The source summaries name Cellpose `cyto3 (CPU)` but do not record the exact
  Cellpose package version.
- Before/after pairing is inferred from file names. Laboratory records are
  needed to verify specimen identity and acquisition timing.
- The raw ND2/TIFF images are present on the local computer but are not
  packaged with these GitHub outputs.

These differences can change masks, traces, and reported statistics. The
figures in `six_trial_results/` are explicitly labeled exploratory.

## Re-run

```powershell
python REPRODUCE/six_trial_analysis.py --assets "C:\path\to\Ca2Data_fin\verification_work\assets"
```

The script verifies each source image exists, audits mask labels and the
per-cell F0i formula, and writes source checksums, trial and cell summaries,
paired statistics, and comparison plots to `REPRODUCE/six_trial_results/`.
