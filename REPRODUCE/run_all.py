"""
run_all.py -- one command to reproduce EVERYTHING end-to-end:
  0. recompute_from_raw.py  regenerate ΔF/F₀ from the raw LATA1 TIFFs (registration ->
     Cellpose cyto3 segmentation -> background subtraction -> ΔF/F₀).  SKIPPED if the
     regenerated/ folder already exists (it is slow; delete it to force a re-run).
  1. reproduce.py       analysis + quantitative figures + statistics (Shapiro-Wilk -> test choice)
  2. figures_paper.py   paper-style figures (registration overlays, mask, raster, background, ROI, schematic)
  3. generate_paper.py  assemble the manuscript (.docx) + PROOF.md

Usage:  python run_all.py            (skips step 0 if regenerated/ exists)
        python run_all.py --recompute  (force the end-to-end recompute first)
"""
import runpy, os, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

force = "--recompute" in sys.argv
regen = os.path.join(HERE, "regenerated")
if force and os.path.isdir(regen):
    shutil.rmtree(regen, ignore_errors=True)

steps = []
if force or not os.path.isdir(regen):
    steps.append("recompute_from_raw.py")
else:
    print("Skipping recompute_from_raw.py (regenerated/ exists; pass --recompute to force).")
steps += ["reproduce.py", "figures_paper.py", "generate_paper.py"]

for script in steps:
    print(f"\n=========== {script} ===========")
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")
print("\nAll done. See CAFIN_reproduced_paper.docx, PROOF.md, and results/figures/.")
