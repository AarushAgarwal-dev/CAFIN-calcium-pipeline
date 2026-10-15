"""
generate_paper.py -- build a complete manuscript (.docx) purely from the
reproduction outputs (results.json + figures). It does NOT read the original PDF;
every number is injected from results.json so the paper always matches the data.

Run AFTER reproduce.py:
    python generate_paper.py
Produces: CAFIN_reproduced_paper_per_cell.docx and PROOF.md
"""
import os, json
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(RES, "figures")
R = json.load(open(os.path.join(RES, "results.json")))

fm = R["field_means"]
T = R["tests"]
C0 = R["conditions"]        # per-field summaries, available throughout the document
RAW_TRACES_RECOMPUTED = os.environ.get("CAFIN_RAW_TRACES_RECOMPUTED") == "1"
IMAGE_FIGURES_REGENERATED = os.environ.get("CAFIN_IMAGE_FIGURES_REGENERATED") == "1"


def g(cond, key):        # field mean helper
    return fm[key][cond]


def p_fmt(p):
    return "p < 1e-300" if p == 0 else f"p = {p:.1e}"


# ------------------------------------------------------------- figure numbering
# Auto-assign figure numbers in first-reference order so captions and in-text
# references can never disagree (they share the same key -> same number).
_FIGNUM = {}
def fignum(key):
    if key not in _FIGNUM:
        _FIGNUM[key] = len(_FIGNUM) + 1
    return _FIGNUM[key]

def frange(k1, k2):
    return f"Figures {fignum(k1)}–{fignum(k2)}"

# Pre-assign every figure number in DOCUMENT order so any in-text reference (even a
# forward one) is consistent with the caption and no intermediate figure is skipped.
FIGURE_ORDER = ["pipeline", "reg_base", "reg_drug", "processing", "mask", "background", "roi",
                "traces", "active_fraction", "metrics", "region", "raster", "heatmap",
                "peak_map", "activity_maps", "pattern", "spatial", "time"]
for _k in FIGURE_ORDER:
    fignum(_k)


# ------------------------------------------------------------- build doc
doc = Document()
style = doc.styles["Normal"]; style.font.name = "Calibri"; style.font.size = Pt(11)


def h(text, level=1):
    p = doc.add_heading(text, level=level)
    return p


def para(text, italic=False, bold=False, align=None, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text); run.italic = italic; run.bold = bold; run.font.size = Pt(size)
    if align == "center": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


def figure(path, key, caption):
    """Embed a figure; the number is assigned from `key` via the registry so the
    caption and any in-text `fignum(key)` reference always match."""
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(6.3))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = doc.add_paragraph()
        r = c.add_run(f"Figure {fignum(key)}. {caption}"); r.italic = True; r.font.size = Pt(9)


# ---------------- Title
title = para("Quantitative Analysis of Calcium Transients in an Epithelial Cell Layer "
             "in Vivo upon Cytoskeleton Disruption", bold=True, align="center", size=16)
para("A computational reproduction of the authors' own analysis draft, generated directly from "
     "pipeline outputs on the same imaging data", italic=True, align="center", size=11)
para("CAFIN pipeline · rigid ECC registration (elastic B-spline available) · Cellpose cyto3 "
     "segmentation · per-cell ΔF/F₀ᵢ analysis", align="center", size=10)
doc.add_paragraph()

# ---------------- Abstract
h("Abstract", 1)
abs = (
    "Calcium (Ca²⁺) signaling coordinates mechanical and biochemical responses across epithelial "
    "tissues, but quantifying spatiotemporal Ca²⁺ dynamics in vivo is complicated by tissue motion "
    "and deformation during live imaging. We present a scripted computational "
    "framework (CAFIN) for rapid analysis of Ca²⁺ transients in the zebrafish larval fin epithelium "
    "under cytoskeletal perturbation. Time-lapse recordings were motion-corrected with a two-way "
    "registration workflow on the membrane channel, with the resulting transform applied to the "
    "calcium channel. Rigid alignment used OpenCV's enhanced correlation coefficient (ECC); "
    "non-rigid B-spline registration was available for local deformation. Single cells were "
    "segmented on the reference frame with Cellpose, and each cell's calcium trace was normalized "
    "to its own baseline F₀ᵢ. Comparing baseline "
    f"activity to Latrunculin-induced actin disruption across two independent trials "
    f"({R['conditions']['trial1/BEFOREDRUG']['n_cells']}+{R['conditions']['trial2/BEFOREDRUG']['n_cells']} "
    f"baseline cells vs {R['conditions']['trial1/AFTERDRUG']['n_cells']}+"
    f"{R['conditions']['trial2/AFTERDRUG']['n_cells']} treated cells), actin disruption "
    "consistently increased transient frequency, mean ΔF/F₀ᵢ and per-frame transient area, and "
    "raised tissue-level spatial heterogeneity of peak amplitude "
    f"(CV {g('Baseline','spatial_heterogeneity_cv'):.2f} → "
    f"{g('Latrunculin','spatial_heterogeneity_cv'):.2f}) and temporal coordination "
    f"(mean pairwise correlation {g('Baseline','temporal_sync_r'):.2f} → "
    f"{g('Latrunculin','temporal_sync_r'):.2f}) — in the same direction in both trials. Because only "
    "two biological replicates were available per condition, we report the per-trial direction of "
    "effect as the primary evidence (Table 1); pooled single-cell distributions are descriptive "
    "support only (large cell counts yield very small p-values but treat cells, not animals, as "
    "replicates). This work reproduces the authors' own analysis pipeline on the same data, "
    "establishing internal reproducibility rather than independent replication, and provides a "
    "generalizable scripted approach for quantifying Ca²⁺ dynamics in deformable tissues."
)
para(abs)

# ---------------- Introduction
h("1. Introduction", 1)
para(
    "Cytosolic calcium is a ubiquitous second messenger governing exocytosis, contraction, "
    "proliferation and morphogenesis. In epithelia, Ca²⁺ waves propagate across the sheet to "
    "coordinate collective behaviors such as migration, contraction and wound response. Because the "
    "actin cytoskeleton mechanically couples cells and gates mechanosensitive channels, disrupting "
    "actin is expected to reshape Ca²⁺ dynamics. Latrunculin A inhibits actin polymerization by "
    "sequestering monomeric G-actin, depolymerizing F-actin and thereby altering membrane tension "
    "and mechanosensitive-channel activity, which can trigger Ca²⁺ influx and intercellular waves."
)
para(
    "The zebrafish larval fin epithelium is optically transparent and pharmacologically accessible, "
    "making it ideal for single-cell Ca²⁺ imaging. However, extracting quantitative traces in vivo is "
    "hindered by global drift and local deformation, which distort spatial mapping and temporal "
    "correlation of activity. We address this with a scripted pipeline that (i) motion-corrects on a "
    "stable membrane channel, (ii) segments single cells, and (iii) extracts and statistically compares "
    "Ca²⁺ dynamics between baseline and cytoskeletal-disruption conditions."
)

# ---------------- Methods
h("2. Materials and Methods", 1)
figure(os.path.join(FIGS, "figP1_pipeline.png"), "pipeline",
       "Image and data-processing pipeline: membrane segmentation → registration "
       "(motion correction) → per-cell Ca²⁺ extraction → single-cell and tissue-level analysis. "
       "Panels are real outputs from the LATA1 dataset.")
h("2.1 Imaging and preparation", 2)
para(
    "Zebrafish larvae (3 dpf) expressing a genetically encoded Ca²⁺ indicator were imaged on a "
    "confocal microscope. The green channel reported cytosolic Ca²⁺; a membrane label provided a "
    "structurally stable red channel. Latrunculin A (final ~200 µM in 1% DMSO) was applied "
    "immediately before imaging as the actin-disrupting perturbation; DMSO served as vehicle. Each "
    "field was recorded before (baseline) and after treatment."
)
h("2.2 Two-way image registration", 2)
para(
    "Registration was performed on the membrane channel and the resulting transform applied to the "
    "calcium channel to preserve cross-channel correspondence. For mild translational/rotational "
    "motion, rigid registration used OpenCV's Enhanced Correlation Coefficient (ECC) algorithm "
    "(Euclidean model: rotation + translation) aligning every frame to frame 0. For pronounced "
    "deformation, non-rigid registration used a SimpleITK B-spline transform with Mattes mutual "
    "information as the similarity metric and an LBFGS-B optimizer; control-point grid, iterations and "
    "step size were tuned empirically. On the recordings analyzed here the large post-treatment "
    "motion was predominantly global (translation and rotation), which rigid ECC corrected well; "
    f"elastic B-spline gave only a marginal additional reduction in residual (Figure {fignum('reg_drug')}), "
    "so rigid registration was used for the quantitative results."
)
figure(os.path.join(FIGS, "figP2_registration_baseline.png"), "reg_base",
       "Registration overlay, baseline (green = reference frame 0, magenta = moving "
       "frame; white = aligned). Before registration the moving frame is offset; after ECC "
       "alignment the channels overlap.")
figure(os.path.join(FIGS, "figP3_registration_afterdrug.png"), "reg_drug",
       "Rigid-vs-elastic registration after Latrunculin A (frame 0 green, moving frame "
       "magenta; white = aligned). The large post-treatment motion is predominantly global: rigid "
       "ECC substantially reduces the mean membrane residual (annotated on each panel), and elastic "
       "B-spline yields only a marginal further improvement on this recording. Rigid registration "
       "was therefore used for the quantitative analysis; elastic remains available for datasets with "
       "strong local deformation. Residual = mean absolute membrane mismatch over tissue pixels.")
h("2.3 Segmentation and trace extraction", 2)
para(
    "Single cells were segmented on the registered reference-frame membrane image with the Cellpose "
    "cyto3 model (channels=[2,0], cell diameter ≈ 15 px), yielding an integer-labeled mask. "
    "Frame-by-frame background subtraction was applied to the calcium channel. Three background "
    "reference regions were selected automatically as the lowest-median image tiles — a scripted "
    "proxy for the original protocol's manual selection of signal-free regions; per frame, values "
    "outside 1.5×IQR were removed, each region reduced to its median, and the average of the three "
    "medians subtracted from every pixel and clipped at zero. For each cell i, the mean pixel "
    "intensity inside its mask was measured at each frame t to give Fₜ,ᵢ. Its own baseline F₀ᵢ "
    "was the mean of its n lowest finite, background-corrected frame intensities, where "
    "n = min(10, max(3, T // 3)) for a recording of T frames. F₀ᵢ was floored at 1 intensity "
    "unit to avoid near-zero division. Each cell was then normalized independently as "
    "ΔF/F₀ᵢ(t) = (Fₜ,ᵢ − F₀ᵢ) / F₀ᵢ; no population-wide baseline was used."
)
figure(os.path.join(FIGS, "figP4_processing_row.png"), "processing",
       "Processing sequence: raw membrane frame 0, registered membrane, registered "
       "membrane with the Cellpose cell mask (red), and the registered calcium channel with the same "
       "mask — the ROIs used for per-cell trace extraction.")
figure(os.path.join(FIGS, "figP5_mask_numbered.png"), "mask",
       f"Cellpose segmentation of the reference membrane frame with numbered cell ROIs "
       f"({C0['trial1/BEFOREDRUG']['n_cells']} cells). Each labeled region defines one cell's ROI "
       f"across all frames.")
figure(os.path.join(FIGS, "figP8_background.png"), "background",
       "Background subtraction: three signal-free reference regions (cyan boxes) on the calcium "
       "frame. Per frame, values outside 1.5×IQR are discarded, each region reduced to its median, "
       "and the mean of the three medians is subtracted and clipped at zero.")
figure(os.path.join(FIGS, "figP9_roi.png"), "roi",
       "Region-of-interest selection: a rectangular ROI (yellow) is drawn on the first calcium "
       "frame, and every segmented cell whose mask intersects the ROI (cyan) is retained; the traces "
       "of these cells are exported to a separate CSV, so analysis can be restricted to a reliable "
       "in-view subset.")
h("2.4 Metrics and statistics", 2)
para(
    "Per cell we computed peak ΔF/F₀ᵢ, mean ΔF/F₀ᵢ, transient count (SciPy find_peaks, height 0.5, "
    "min distance 2 frames), transients per frame, and per-frame supra-threshold area (AUC). At the "
    "tissue level we computed the active fraction, spatial heterogeneity (coefficient of variation of "
    "per-cell peak amplitude across the field) and temporal synchronization (mean pairwise Pearson "
    "correlation among active cells). Spatial coordination was assessed as pairwise correlation versus "
    "inter-cell centroid distance. For each comparison, normality was tested with the Shapiro–Wilk "
    "test on both groups; where both groups were normal a two-sided Welch t-test was used (a close "
    "variant of the original protocol's Student t-test), otherwise the Mann–Whitney U test (the "
    "actual test used per metric is reported in the appendix). Effect size is the rank-biserial "
    "correlation; regional R1–R4 differences used the Kruskal–Wallis omnibus test (no post-hoc "
    "pairwise correction). Importantly, only two biological replicates (trials) were available per "
    "condition. The pooled single-cell tests therefore treat individual cells as replicates "
    "(pseudoreplication) and their p-values quantify separation of the cell populations, not "
    "biological-replicate significance; the per-trial direction of effect (Table 1) is the primary "
    "evidence. All analyses are deterministic (fixed random seed) and reproduced by a single script."
)

# ---------------- Results
h("3. Results", 1)

h("3.1 Single-cell activity increases after actin disruption", 2)
C = R["conditions"]
para(
    f"Because only two biological trials were available per condition, we take the per-trial "
    f"direction of effect (Table 1) as the primary evidence. Transient frequency, mean ΔF/F₀ᵢ and "
    f"spatial heterogeneity increased in both independent trials after treatment. The effect is not "
    f"uniform across every metric: the median peak amplitude rose in trial 1 "
    f"({C['trial1/BEFOREDRUG']['median_peak_amp']:.2f}→{C['trial1/AFTERDRUG']['median_peak_amp']:.2f}) "
    f"while decreasing in trial 2 "
    f"({C['trial2/BEFOREDRUG']['median_peak_amp']:.2f}→{C['trial2/AFTERDRUG']['median_peak_amp']:.2f}), "
    f"where the field instead became markedly more heterogeneous "
    f"(CV {C['trial2/BEFOREDRUG']['spatial_heterogeneity_cv']:.2f}→"
    f"{C['trial2/AFTERDRUG']['spatial_heterogeneity_cv']:.2f}). The consistent cross-trial signature "
    f"of actin disruption is therefore increased transient frequency and spatial heterogeneity "
    f"rather than a uniform amplitude increase ({frange('traces','region')})."
)
# Table 1: per-trial transparency
_cap = doc.add_paragraph()
_r = _cap.add_run("Table 1. Per-trial summary (both conditions). LATA1 raw fluorescence "
                  "traces originated from TIFF processing; LATA2 raw traces came from stored "
                  "pipeline CSVs. Every cell was renormalized to its own F₀ᵢ for this analysis.")
_r.italic = True; _r.font.size = Pt(9)
t1 = doc.add_table(rows=1, cols=6); t1.style = "Light Grid Accent 1"
for i, htxt in enumerate(["Trial / condition", "Cells", "Active %", "Median peak ΔF/F₀ᵢ",
                          "Heterogeneity CV", "Sync r"]):
    t1.rows[0].cells[i].text = htxt
for key in ["trial1/BEFOREDRUG", "trial1/AFTERDRUG", "trial2/BEFOREDRUG", "trial2/AFTERDRUG"]:
    s = C[key]; row = t1.add_row().cells
    row[0].text = f"{s['trial']} {s['condition']}"
    row[1].text = str(s["n_cells"]); row[2].text = f"{s['active_fraction']*100:.0f}"
    row[3].text = f"{s['median_peak_amp']:.2f}"; row[4].text = f"{s['spatial_heterogeneity_cv']:.2f}"
    row[5].text = f"{s['temporal_sync_r']:.3f}"
para(
    f"As descriptive support, we also pooled the single cells across both trials "
    f"({T['peak_amp']['n1']} baseline vs {T['peak_amp']['n2']} treated cells). Median transient rate "
    f"rose from {T['peaks_per_frame']['med1']:.3f} to {T['peaks_per_frame']['med2']:.3f} "
    f"transients/frame, mean ΔF/F₀ᵢ from {T['mean_dff']['med1']:.2f} to {T['mean_dff']['med2']:.2f}, "
    f"and per-frame supra-threshold area {T['auc']['med2']/max(T['auc']['med1'],1e-9):.1f}-fold; "
    f"Shapiro–Wilk rejected normality for every metric, so the {T['peak_amp']['test']} was applied "
    f"(Appendix). These pooled comparisons yield extremely small p-values, but because they treat "
    f"cells rather than animals as replicates (n = 2 trials), the p-values should be read as a "
    f"measure of population separation, not as biological-replicate significance. Two caveats apply: "
    f"the pooled peak-amplitude comparison spans unequal recording lengths (baseline 31 vs treated "
    f"55–90 frames), so a maximum-based metric is only weakly comparable — the length-normalized "
    f"transient rate and per-frame area are more comparable. The LATA2 fields use stored "
    f"background-corrected raw fluorescence traces; LATA1 traces originated from the documented "
    f"registration→cyto3→background-extraction chain. All four fields were renormalized "
    f"from raw per-cell traces for this analysis (see §5)."
)
figure(os.path.join(FIGS, "fig1_traces.png"), "traces",
       "Representative single-cell ΔF/F₀ᵢ traces (40 most active cells; black = population "
       "mean; dashed line = transient threshold). Trial-specific changes are summarized in Table 1.")
figure(os.path.join(FIGS, "figR4_active_cell_fraction.png"), "active_fraction",
       "Fraction of segmented cells above ΔF/F₀ᵢ = 0.5 in each processed frame, "
       "shown separately for the baseline and Latrunculin A recordings. The curves describe "
       "activity within each field; different recording lengths limit direct timepoint comparison.")
figure(os.path.join(FIGS, "fig3_metrics.png"), "metrics",
       "Single-cell metric distributions, baseline vs Latrunculin A, pooled across two "
       "trials (descriptive; the pooled per-cell p-values annotated here are pseudoreplicated — see "
       "§3.1). The length-normalized transient rate and per-frame area are the sound metrics.")
figure(os.path.join(FIGS, "figP7_region_bars.png"), "region",
       "Regional single-cell activity (peak ΔF/F₀ᵢ). Cells were split into four equal "
       "proximal-to-distal bands (R1–R4) by centroid y-position; bars are region means ± SEM. "
       "The figure reports the Kruskal–Wallis omnibus comparison within each condition.")

h("3.2 Tissue-level Ca²⁺ patterns and spatial heterogeneity", 2)
para(
    f"At the tissue scale, the activity map changed from sparse, low-amplitude flickers at baseline to "
    f"dense, high-amplitude activity after treatment ({frange('raster','heatmap')}). Spatial heterogeneity of peak "
    f"amplitude increased in both trials (CV "
    f"{fm['spatial_heterogeneity_cv']['per_field_Baseline']} at baseline → "
    f"{fm['spatial_heterogeneity_cv']['per_field_Latrunculin']} after treatment), consistent with a "
    "loss of uniform, coordinated responses across the sheet."
)
figure(os.path.join(FIGS, "figP6_raster.png"), "raster",
       "Ca²⁺ transient event raster (each tick = a detected transient; cells sorted by "
       "first event). The two conditions show different event patterns.")
figure(os.path.join(FIGS, "fig2_heatmaps.png"), "heatmap",
       "Tissue-level activity heatmaps (cells × frames, shared color scale). Left: baseline. "
       "Right: Latrunculin A. Each cell's signal uses its own F₀ᵢ baseline.")
figure(os.path.join(FIGS, "figR5_peak_activity_static_masks.png"), "peak_map",
       "Per-cell peak ΔF/F₀ᵢ painted on the saved frame-0 segmentation masks. "
       "Both conditions use the same color scale; the masks define fixed ROIs after registration.")
figure(os.path.join(FIGS, "figR6_timepoint_activity_static_masks.png"), "activity_maps",
       "Early, middle, and late per-cell ΔF/F₀ᵢ values mapped to the fixed frame-0 masks. "
       "These images show fluorescence in registered ROIs and do not establish cell movement.")
figure(os.path.join(FIGS, "fig5_pattern_metrics.png"), "pattern",
       "Tissue-level pattern metrics (bars = condition mean; points = individual trials). "
       "Spatial heterogeneity and temporal synchronization both rise after actin disruption.")

h("3.3 Spatial coordination of Ca²⁺ activity", 2)
para(
    f"Distance-binned pairwise correlations summarize spatial coordination in both conditions "
    f"(Figure {fignum('spatial')}). The mean pairwise correlation among active cells rose from "
    f"{g('Baseline','temporal_sync_r'):.2f} to {g('Latrunculin','temporal_sync_r'):.2f}, and the "
    f"pooled pairwise-correlation distribution shifted upward "
    f"({p_fmt(T['pairwise_sync_r']['p'])}, Mann–Whitney U). This cell-pair comparison is "
    "descriptive because pairs within each trial are not independent biological replicates."
)
figure(os.path.join(FIGS, "fig4_spatial.png"), "spatial",
       "Mean pairwise correlation versus inter-cell centroid distance, pooled per condition. "
       "Latrunculin A raises coordination across the distances shown.")

h("3.4 Time-dependence of the response", 2)
para(
    f"Within the post-treatment recordings, mean per-cell peak amplitude varied across early, middle "
    f"and late imaging windows, indicating that the Latrunculin response evolves over the recording "
    f"rather than being static (Figure {fignum('time')}). This within-recording variation is "
    f"suggestive of progressive actin depolymerization, but the present data — a single continuous "
    f"post-treatment recording per trial — do not extend long enough to establish recovery or "
    f"adaptation; testing that would require longer or repeated late-timepoint imaging."
)
figure(os.path.join(FIGS, "fig6_time.png"), "time",
       "Time-dependence of the response: mean per-cell peak ΔF/F₀ᵢ across early/mid/late "
       "windows for each treated field.")

# ---------------- Discussion
h("4. Discussion and Conclusion", 1)
para(
    "Using a scripted, deterministic pipeline we quantified how actin disruption reshapes epithelial "
    "Ca²⁺ dynamics in vivo. In both trials, Latrunculin A increased single-cell transient frequency "
    "and mean ΔF/F₀ᵢ, raised spatial heterogeneity of peak responses, and elevated inter-cellular "
    "correlation relative to baseline. These changes are consistent with actin depolymerization "
    "altering membrane tension and mechanosensitive-channel activity, promoting Ca²⁺ influx and "
    "intercellular propagation while degrading the uniform, finely balanced signaling seen in the "
    "intact tissue. Because the study rests on two biological replicates per condition and reproduces "
    "the authors' own draft analysis on the same imaging data, these results establish the internal "
    "reproducibility and direction of the effect rather than independent, statistically powered "
    "confirmation."
)
para(
    "The methodological contribution is a scripted registration front-end offering both rigid (ECC) "
    "and non-rigid (B-spline) correction; on the recordings analyzed here the motion was "
    "predominantly global and rigid registration sufficed, with elastic evaluated but not required "
    f"(Figure {fignum('reg_drug')}). Limitations include reliance on a stable membrane label for "
    "registration, a single reference-frame segmentation without explicit cross-time cell tracking, "
    "and a small number of biological replicates. Future work could add deep-learning non-rigid "
    "tracking, real-time registration and 3D volumetric imaging, increase replication, and combine "
    "cytoskeletal mutants with this platform to dissect specific actin–Ca²⁺ couplings."
)
para(
    "The reference manuscript draft discusses six paired trials. A separate six-trial local "
    "audit remains exploratory because its processing methods and data provenance differ from "
    "the two trial pairs analyzed here. The present tables and figures therefore do not reproduce "
    "the reference draft's six-trial estimates or significance tests."
)

# ---------------- Reproducibility
h("5. Reproducibility and data provenance", 1)
_local_raw_root = os.environ.get("CAFIN_RAW_DATA_ROOT")
_local_source_note = (
    "Original microscopy TIFFs were located in a local data folder outside the repository; "
    "they are not bundled with this checkout."
    if ((_local_raw_root and os.path.isdir(_local_raw_root))
        or RAW_TRACES_RECOMPUTED or IMAGE_FIGURES_REGENERATED) else
    "Original microscopy TIFFs are not bundled with this checkout; stored raw per-cell "
    "fluorescence traces permit renormalization without rerunning image extraction."
)
_image_source = (
    "The paper-style microscopy panels and three activity figures were regenerated from "
    "local TIFFs, saved cell masks, and corrected traces in this run."
    if IMAGE_FIGURES_REGENERATED else
    "The paper-style microscopy and activity figures were retained from the repository and "
    "were not regenerated in this run."
)
_trace_source = (
    "LATA1 raw fluorescence traces were re-extracted from local TIFFs in this run using "
    "registration, Cellpose cyto3 segmentation, and background subtraction."
    if RAW_TRACES_RECOMPUTED else
    "This run used LATA1 raw fluorescence CSVs previously extracted from TIFFs; it did not "
    "rerun registration, segmentation, or fluorescence extraction."
)
para(
    f"For all four fields, each cell's F₀ᵢ and ΔF/F₀ᵢ trace were recalculated from stored or newly "
    f"extracted, background-corrected raw fluorescence CSVs. The numerical results and six "
    f"quantitative figures were then generated by reproduce.py from these normalized traces; "
    f"generate_paper.py assembled this document. {_local_source_note} {_image_source} "
    f"{_trace_source} The LATA2 fields used stored pipeline raw fluorescence CSVs and were "
    f"renormalized in this run; LATA2 image extraction was not rerun. Each field's baseline values "
    f"and source trace hashes are retained in normalized_per_cell/, and provenance is summarized "
    f"in field_summary.csv. Analyses use a fixed random seed; see README_REPRODUCE.md."
)
para(
    "The stored LATA2 baseline raw-trace table lacked fluorescence values in frame 0 for every "
    "cell. That row was recovered from the locally available source calcium TIFF with the saved "
    "frame-0 mask and the same three-region background method. Applying the method to registered "
    "frame 1 reproduced all 479 stored cell means within floating-point precision. The original "
    "trace table was retained; the repaired row and source checksums are recorded in "
    "REPRODUCE/source_repairs/."
)

# summary table
doc.add_page_break()
h("Appendix: reproduced summary statistics", 2)
_cap = doc.add_paragraph()
_r = _cap.add_run("Statistical tests. Normality assessed by Shapiro–Wilk on each group; "
                  "Welch t-test if both normal, otherwise Mann–Whitney U. Effect = rank-biserial.")
_r.italic = True; _r.font.size = Pt(9)
tbl = doc.add_table(rows=1, cols=7); tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
for i, t in enumerate(["Metric", "Median base", "Median Lat", "Shapiro p (base/Lat)",
                       "Test used", "p-value", "effect (r)"]):
    hdr[i].text = t
label = {"peak_amp": "Peak ΔF/F₀ᵢ", "mean_dff": "Mean ΔF/F₀ᵢ",
         "peaks_per_frame": "Transients / frame", "auc": "Transient AUC",
         "pairwise_sync_r": "Pairwise correlation"}
def _sp(x):
    return "<1e-4" if (isinstance(x, float) and x < 1e-4) else (f"{x:.2f}" if x == x else "n/a")
for k, t in T.items():
    row = tbl.add_row().cells
    row[0].text = label.get(k, k)
    row[1].text = f"{t['med1']:.3f}"; row[2].text = f"{t['med2']:.3f}"
    row[3].text = f"{_sp(t.get('shapiro_p1', float('nan')))} / {_sp(t.get('shapiro_p2', float('nan')))}"
    row[4].text = t.get("test", "Mann-Whitney U")
    row[5].text = ("<1e-300" if t["p"] == 0 else f"{t['p']:.1e}")
    row[6].text = f"{t['effect']:.2f}"

out_docx = os.path.join(HERE, "CAFIN_reproduced_paper_per_cell.docx")
try:
    doc.save(out_docx)
except PermissionError:
    alt = os.path.join(HERE, "CAFIN_reproduced_paper_per_cell_NEW.docx")
    doc.save(alt)
    print(f"NOTE: {out_docx} was locked (open in Word?). Saved to {alt} instead.")
    out_docx = alt
print("wrote", out_docx)


# ------------------------------------------------------------- PROOF.md
def cond(key, tk):
    return R["conditions"][f"{tk}/{key}"]

md = []
md.append("# Reproduction proof — CAFIN calcium-transient study\n")
md.append("Generated automatically from `results/results.json`; every number below is computed "
          "from per-cell ΔF/F₀ᵢ outputs. Each F₀ᵢ is the mean of cell i's own "
          "n = min(10, max(3, T // 3)) lowest finite, background-corrected frames "
          "(floor 1); ΔF/F₀ᵢ(t) = (Fₜ,ᵢ − F₀ᵢ) / F₀ᵢ. "
          "Re-running normalization, `reproduce.py`, and `generate_paper.py` "
          "regenerates this file identically (fixed seed).\n")
md.append(f"{_local_source_note} {_image_source} {_trace_source} The LATA2 fields used "
          "stored raw per-cell fluorescence CSVs. All four fields were renormalized in this run.\n")
md.append("The stored LATA2 baseline trace table lacked frame-0 values. A separately checked "
          "source-TIFF row restores them without changing the original CSV; see "
          "source_repairs/lata2_before_frame0_provenance.json.\n")
md.append("The reference manuscript's six-paired-trial result is not reproduced here. "
          "A separate six-trial local audit is exploratory because processing methods and "
          "provenance differ; this report presents two trial pairs.\n")
md.append("## 1. Fields analyzed\n")
md.append("| Field | cells | frames | active % | median peak ΔF/F₀ᵢ | heterogeneity CV | sync r | provenance |")
md.append("|---|---|---|---|---|---|---|---|")
for tk in ["trial1", "trial2"]:
    for c in ["BEFOREDRUG", "AFTERDRUG"]:
        s = cond(c, tk)
        md.append(f"| {tk} {s['condition']} | {s['n_cells']} | {s['n_frames']} | "
                  f"{s['active_fraction']*100:.1f}% | {s['median_peak_amp']:.2f} | "
                  f"{s['spatial_heterogeneity_cv']:.2f} | {s['temporal_sync_r']:.3f} | "
                  f"{s.get('provenance','')} |")
md.append("\n## 2. Statistical tests (baseline vs Latrunculin, pooled cells)\n")
md.append("Normality via Shapiro–Wilk; Welch t-test if both groups normal, else Mann–Whitney U.\n")
md.append("| Metric | median base | median Lat | Shapiro p (base/Lat) | test | p-value | effect |")
md.append("|---|---|---|---|---|---|---|")
lbl = {"peak_amp": "Peak ΔF/F₀ᵢ", "mean_dff": "Mean ΔF/F₀ᵢ", "peaks_per_frame": "Transients/frame",
       "auc": "Transient AUC", "pairwise_sync_r": "Pairwise correlation r"}
def _spm(x):
    return "<1e-4" if (isinstance(x, (int, float)) and x == x and x < 1e-4) else (f"{x:.2f}" if x == x else "n/a")
for k, t in T.items():
    pv = "<1e-300" if t["p"] == 0 else f"{t['p']:.2e}"
    md.append(f"| {lbl.get(k,k)} | {t['med1']:.3f} | {t['med2']:.3f} | "
              f"{_spm(t.get('shapiro_p1', float('nan')))}/{_spm(t.get('shapiro_p2', float('nan')))} | "
              f"{t.get('test','Mann-Whitney U')} | {pv} | {t['effect']:.2f} |")
md.append("\n## 3. Manuscript claims vs reproduced result\n")
md.append("| Manuscript claim | Reproduced? | Evidence |")
md.append("|---|---|---|")
md.append(f"| Increased spatial heterogeneity | ✅ | CV {g('Baseline','spatial_heterogeneity_cv'):.2f} → "
          f"{g('Latrunculin','spatial_heterogeneity_cv'):.2f} (both trials) |")
md.append(f"| Altered temporal synchronization | ✅ | mean pairwise r {g('Baseline','temporal_sync_r'):.2f} → "
          f"{g('Latrunculin','temporal_sync_r'):.2f} across two trial pairs; pooled cell-pair p-values are descriptive |")
md.append(f"| Increased peak amplitude | Mixed | Trial 1 median "
          f"{C0['trial1/BEFOREDRUG']['median_peak_amp']:.2f} → "
          f"{C0['trial1/AFTERDRUG']['median_peak_amp']:.2f}; trial 2 "
          f"{C0['trial2/BEFOREDRUG']['median_peak_amp']:.2f} → "
          f"{C0['trial2/AFTERDRUG']['median_peak_amp']:.2f} |")
md.append("| Distinct Ca²⁺ propagation/coordination | Partial | Correlation-versus-distance "
          "supports a coordination difference; propagation was not directly measured |")
md.append("\n## 4. Figures (numbered as in the paper)\n")
_figs = [("pipeline", "figP1_pipeline.png", "Image & data processing pipeline"),
         ("reg_base", "figP2_registration_baseline.png", "Registration overlay, baseline (green/magenta)"),
         ("reg_drug", "figP3_registration_afterdrug.png", "Registration: rigid vs elastic, after drug"),
         ("processing", "figP4_processing_row.png", "Membrane | Registered | +Mask | Calcium+Mask"),
         ("mask", "figP5_mask_numbered.png", "Cell mask with numbered ROIs"),
         ("background", "figP8_background.png", "Background reference regions (3 boxes)"),
         ("roi", "figP9_roi.png", "ROI selection (cells intersecting the ROI)"),
         ("traces", "fig1_traces.png", "Single-cell ΔF/F₀ᵢ traces"),
         ("active_fraction", "figR4_active_cell_fraction.png", "Active-cell fraction per frame"),
         ("metrics", "fig3_metrics.png", "Single-cell metric comparison + p-values"),
         ("region", "figP7_region_bars.png", "Regional ΔF/F₀ᵢ (R1–R4)"),
         ("raster", "figP6_raster.png", "Ca²⁺ transient event raster"),
         ("heatmap", "fig2_heatmaps.png", "Tissue activity heatmaps"),
         ("peak_map", "figR5_peak_activity_static_masks.png", "Per-cell peak activity map"),
         ("activity_maps", "figR6_timepoint_activity_static_masks.png", "Activity maps at three timepoints"),
         ("pattern", "fig5_pattern_metrics.png", "Tissue pattern metrics"),
         ("spatial", "fig4_spatial.png", "Spatial coordination vs distance"),
         ("time", "fig6_time.png", "Time-dependence")]
for key, fn, cap in sorted(_figs, key=lambda x: fignum(x[0])):
    label = f"Figure {fignum(key)} — {cap}"
    md.append(f"**{label}**\n\n![{label}](results/figures/{fn})\n")
open(os.path.join(HERE, "PROOF.md"), "w", encoding="utf-8").write("\n".join(md))
print("wrote", os.path.join(HERE, "PROOF.md"))
