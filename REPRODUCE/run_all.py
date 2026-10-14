"""Rebuild the CAFIN analysis and manuscript from the available study data.

With the distributed processed CSVs, this regenerates the statistics, six
quantitative figures, DOCX, and PROOF.md. The nine paper-style microscopy PNGs
are rebuilt only when the original LATA1 TIFFs are available; otherwise every
required checked-in PNG must be present before the run starts.

Use --recompute to regenerate LATA1 traces from its original raw TIFFs first.
"""
import glob
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

force = "--recompute" in sys.argv
regen = os.path.join(HERE, "regenerated")
root = os.path.dirname(HERE)
raw_folders = [os.path.join(root, "LATA1TRAIL", cond, channel)
               for cond in ("BEFOREDRUG", "AFTERDRUG")
               for channel in ("membrane", "ca2")]
raw_ready = all(glob.glob(os.path.join(folder, "*.tif")) for folder in raw_folders)
paper_style_figures = (
    "figP1_pipeline.png",
    "figP2_registration_baseline.png",
    "figP3_registration_afterdrug.png",
    "figP4_processing_row.png",
    "figP5_mask_numbered.png",
    "figP6_raster.png",
    "figP7_region_bars.png",
    "figP8_background.png",
    "figP9_roi.png",
)

if force and not raw_ready:
    raise SystemExit("--recompute requires .tif frames in all four LATA1TRAIL "
                     "BEFOREDRUG/AFTERDRUG membrane/ and ca2/ folders. The raw TIFFs "
                     "are not included in this repository.")
if not raw_ready:
    missing_figures = [name for name in paper_style_figures
                       if not os.path.isfile(os.path.join(HERE, "results", "figures", name))
                       or not os.path.getsize(os.path.join(HERE, "results", "figures", name))]
    if missing_figures:
        raise SystemExit(
            "Cannot rebuild the manuscript without the original LATA1 TIFFs or all nine "
            "checked-in paper-style figures. Missing figure(s): "
            + ", ".join(missing_figures)
        )

# These flags describe work done by this invocation, not the history of the
# checked-in CSVs or PNGs. generate_paper.py records them in the manuscript.
os.environ["CAFIN_RAW_TRACES_RECOMPUTED"] = "0"
os.environ["CAFIN_IMAGE_FIGURES_REGENERATED"] = "0"

steps = []
regenerated_complete = all(
    os.path.isfile(os.path.join(regen, "LATA1TRAIL", cond, filename))
    for cond in ("BEFOREDRUG", "AFTERDRUG")
    for filename in ("all_cells_normalized.csv", "mask_0.tiff")
)
if force or (raw_ready and not regenerated_complete):
    steps.append("recompute_from_raw.py")
else:
    print("Using available processed per-cell CSVs; pass --recompute with raw LATA1 TIFFs "
          "to regenerate the traces.")
steps.append("reproduce.py")
if raw_ready:
    steps.append("figures_paper.py")
else:
    print("Original LATA1 TIFFs are unavailable; reusing all nine checked-in "
          "paper-style figure PNGs.")
steps.append("generate_paper.py")

for script in steps:
    print(f"\n=========== {script} ===========")
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    if script == "recompute_from_raw.py":
        os.environ["CAFIN_RAW_TRACES_RECOMPUTED"] = "1"
    elif script == "figures_paper.py":
        os.environ["CAFIN_IMAGE_FIGURES_REGENERATED"] = "1"
print("\nAll done. See CAFIN_reproduced_paper.docx, PROOF.md, and results/figures/.")
