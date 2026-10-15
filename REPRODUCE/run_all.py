"""Rebuild the CAFIN analysis and manuscript from the available study data.

The included raw per-cell fluorescence CSVs are normalized to each cell's own
F0, then used to regenerate statistics, figures, DOCX, and PROOF.md. Paper-style
microscopy and activity figures are rebuilt when source TIFFs and masks are supplied.
Use --recompute to rerun LATA1 image registration and segmentation first.
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
external_raw = os.environ.get("CAFIN_RAW_DATA_ROOT")


def raw_folder(cond, channel):
    if external_raw:
        original = os.path.join(external_raw, "LATA1TRAIL", cond, channel)
        if glob.glob(os.path.join(original, "*.tif")):
            return original
        recording = "lat_trial1_beforedrug" if cond == "BEFOREDRUG" else "lat_trial1_afterdrug"
        return os.path.join(external_raw, recording, channel)
    return os.path.join(root, "LATA1TRAIL", cond, channel)


raw_folders = [raw_folder(cond, channel)
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
    "figR4_active_cell_fraction.png",
    "figR5_peak_activity_static_masks.png",
    "figR6_timepoint_activity_static_masks.png",
)

if force and not raw_ready:
    raise SystemExit("--recompute requires .tif frames in all four LATA1TRAIL "
                     "BEFOREDRUG/AFTERDRUG membrane/ and ca2/ folders, or set "
                     "CAFIN_RAW_DATA_ROOT to the local Total data folder.")

# These flags describe work done by this invocation, not the history of the
# checked-in CSVs or PNGs. generate_paper.py records them in the manuscript.
os.environ["CAFIN_RAW_TRACES_RECOMPUTED"] = "0"
os.environ["CAFIN_IMAGE_FIGURES_REGENERATED"] = "0"

def run_step(script):
    print(f"\n=========== {script} ===========")
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")


if force:
    run_step("recompute_from_raw.py")
    os.environ["CAFIN_RAW_TRACES_RECOMPUTED"] = "1"
else:
    print("Using stored per-cell raw fluorescence traces. Image registration and "
          "segmentation are not rerun; pass --recompute to do so.")

run_step("normalize_f0_per_cell.py")
run_step("reproduce.py")

mask_root = os.environ.get("CAFIN_MASK_ROOT")
mask_ready = all(
    os.path.isfile(os.path.join(mask_root, cond, "mask_0.tiff"))
    if mask_root else (
        os.path.isfile(os.path.join(regen, "LATA1TRAIL", cond, "mask_0.tiff"))
        or os.path.isfile(os.path.join(root, "LATA1TRAIL", cond, "mask_0.tiff"))
    )
    for cond in ("BEFOREDRUG", "AFTERDRUG")
)
if raw_ready and mask_ready:
    run_step("figures_paper.py")
    os.environ["CAFIN_IMAGE_FIGURES_REGENERATED"] = "1"
else:
    missing_figures = [name for name in paper_style_figures
                       if not os.path.isfile(os.path.join(HERE, "results", "figures", name))
                       or not os.path.getsize(os.path.join(HERE, "results", "figures", name))]
    if missing_figures:
        raise SystemExit(
            "Cannot rebuild the manuscript without source TIFFs and matching masks "
            "or all checked-in paper-style/activity figures. Missing: "
            + ", ".join(missing_figures)
        )
    print("Source TIFFs or matching masks are unavailable for microscopy panels; "
          "reusing the checked-in paper-style and activity figure PNGs.")

run_step("generate_paper.py")
print("\nAll done. See CAFIN_reproduced_paper_per_cell.docx, PROOF.md, and results/figures/.")
