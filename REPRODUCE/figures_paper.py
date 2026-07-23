"""
figures_paper.py -- reproduce the ORIGINAL PAPER'S figure *styles* from real images:

  P1  pipeline schematic  (boxes + arrows + real mini-panels)         -> fig like slide 1
  P2  registration overlay, BASELINE  (green=frame0 / magenta=moving; before vs after)
  P3  registration overlay, AFTER DRUG
  P4  processing row: Membrane | Registered | Membrane+mask | Calcium+mask
  P5  numbered cell mask (Cellpose ROIs with IDs)
  P6  calcium event raster (cell x time tick plot), baseline vs after drug
  P7  regional ΔF/F0 bar chart (R1..R4 along the fin) with significance stars
  P8  background reference regions (3 signal-free boxes)
  P9  ROI selection (rectangle + cells that intersect it)

Uses the real LATA1 membrane/calcium TIFF stacks + Cellpose masks + centroids.
Run:  python figures_paper.py     (writes into results/figures/)
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow, Rectangle
import skimage.io as skio
import tifffile
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from scipy.signal import find_peaks
from scipy.stats import kruskal
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cafin_core as cc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "results", "figures")
REGEN = os.path.join(HERE, "regenerated", "LATA1TRAIL")   # end-to-end recomputed masks/CSVs
os.makedirs(FIGS, exist_ok=True)
TRIAL = "LATA1TRAIL"
THRESH = 0.5
# ROI rectangles reconstructed from the group's original roi_cells selection
ROI_BOX = {"BEFOREDRUG": (146, 205, 355, 325), "AFTERDRUG": (144, 195, 372, 310)}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10})


# ----------------------------------------------------------------- helpers
def base_of(folder):
    f = sorted(glob.glob(os.path.join(folder, "*.tif")))[0]
    return os.path.basename(f)[:-8]                    # strip NNNN.tif


def load_mask(cond):
    """Prefer the end-to-end recomputed mask, else the existing one."""
    rg = os.path.join(REGEN, cond, "mask_0.tiff")
    p = rg if os.path.exists(rg) else os.path.join(ROOT, TRIAL, cond, "mask_0.tiff")
    return tifffile.imread(p)


def load_dff_cen(cond):
    """Return (dff DataFrame, centroids DataFrame), preferring recomputed CSVs."""
    rg = os.path.join(REGEN, cond)
    base = rg if os.path.exists(os.path.join(rg, "all_cells_normalized.csv")) \
        else os.path.join(ROOT, TRIAL, cond)
    return (pd.read_csv(os.path.join(base, "all_cells_normalized.csv")),
            pd.read_csv(os.path.join(base, "centroids_0.csv")))


def load(cond, kind, i):
    folder = os.path.join(ROOT, TRIAL, cond, kind)
    b = base_of(folder)
    return skio.imread(os.path.join(folder, f"{b}{i:04d}.tif"))


def red_mask_overlay(gray, mask, color=(230, 30, 30), alpha_fill=0.35):
    """Membrane/calcium gray with cell mask filled semi-transparent red (paper look)."""
    g = cc.stretch8(gray, clahe=True)
    rgb = np.dstack([g, g, g]).astype(np.float32)
    fill = mask > 0
    for k in range(3):
        rgb[..., k][fill] = (1 - alpha_fill) * rgb[..., k][fill] + alpha_fill * color[k]
    rgb[find_boundaries(mask, mode="outer")] = color
    return np.clip(rgb, 0, 255).astype(np.uint8)


def moving_frame(cond):
    """Pick the frame with the LARGEST estimated motion vs frame 0, so the overlay
    actually shows registration doing something (baseline motion is small)."""
    folder = os.path.join(ROOT, TRIAL, cond, "membrane")
    b = base_of(folder)
    n = len(glob.glob(os.path.join(folder, "*.tif")))
    f0 = skio.imread(os.path.join(folder, f"{b}0000.tif"))
    best_i, best_disp = n - 1, -1.0
    for i in range(max(1, n // 6), n, max(1, n // 8)):     # sample ~8 candidates
        mov = skio.imread(os.path.join(folder, f"{b}{i:04d}.tif"))
        M = cc._ecc(f0, mov)
        disp = float(np.hypot(M[0, 2], M[1, 2]))
        if disp > best_disp:
            best_disp, best_i = disp, i
    return best_i


# ============================================================ P2/P3 registration
def _residual(fixed, reg):
    """Mean absolute membrane mismatch over tissue pixels (lower = better aligned)."""
    a = cc.stretch8(fixed).astype(float); b = cc.stretch8(reg).astype(float)
    fg = (cc.stretch8(fixed) > 40) | (cc.stretch8(reg) > 40)
    return float(np.mean(np.abs(a - b)[fg])) if fg.any() else np.nan


def registration_figure(cond, tag, method="rigid"):
    """method='rigid' -> 2-panel before/after (OpenCV ECC).
       method='compare' -> 3-panel before / rigid ECC / elastic B-spline, with the
       mean residual annotated on each (the manuscript's 'elastic vs rigid' comparison)."""
    k = moving_frame(cond)
    fixed = load(cond, "membrane", 0)
    mov = load(cond, "membrane", k)
    M = cc._ecc(fixed, mov)
    rigid = cc._warp_affine(mov, M, fixed.shape)
    before = cc.two_color_overlay(fixed, mov)          # green=frame0, magenta=moving
    after_rigid = cc.two_color_overlay(fixed, rigid)

    if method == "compare":
        tx = cc._bspline(fixed, mov, mesh=5, iters=60)  # coarse grid = smooth, constrained warp
        elastic = cc._bspline_apply(mov, tx).astype(mov.dtype)
        after_el = cc.two_color_overlay(fixed, elastic)
        panels = [(before, f"BEFORE  (residual {_residual(fixed, mov):.0f})"),
                  (after_rigid, f"AFTER — rigid ECC  (residual {_residual(fixed, rigid):.0f})"),
                  (after_el, f"AFTER — elastic B-spline  (residual {_residual(fixed, elastic):.0f})")]
        fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.9))
    else:
        panels = [(before, "overlap BEFORE registration"),
                  (after_rigid, "overlap AFTER registration (rigid ECC)")]
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.8))

    for a, (img, title) in zip(ax, panels):
        a.imshow(img); a.set_title(title, fontsize=10.5); a.axis("off")
    fig.suptitle(f"{tag}: frame 0 (green) vs frame {k} (magenta)  —  white = aligned",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIGS, f"figP{'2' if cond=='BEFOREDRUG' else '3'}_registration_{tag.lower().replace(' ','')}.png")
    fig.savefig(out); plt.close(fig)
    return before, after_rigid, k


# ============================================================ P4 processing row
def processing_row(cond="BEFOREDRUG"):
    k = moving_frame(cond)
    mem0 = load(cond, "membrane", 0)
    memk = load(cond, "membrane", k)
    M = cc._ecc(mem0, memk); reg = cc._warp_affine(memk, M, mem0.shape)
    ca0 = load(cond, "ca2", 0)
    mask = load_mask(cond)
    panels = [
        (cc.stretch8(mem0, clahe=True), "Membrane (frame 0)", "gray"),
        (cc.stretch8(reg, clahe=True), f"Registered membrane (frame {k})", "gray"),
        (red_mask_overlay(mem0, mask), "Reg. membrane + cell mask", None),
        (red_mask_overlay(ca0, mask), "Registered calcium + mask", None),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for a, (img, title, cmap) in zip(ax, panels):
        a.imshow(img, cmap=cmap); a.set_title(title, fontsize=10); a.axis("off")
    fig.suptitle(f"Image processing pipeline — {('Baseline' if cond=='BEFOREDRUG' else 'Latrunculin B')}",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIGS, "figP4_processing_row.png"); fig.savefig(out); plt.close(fig)
    return mask, mem0


# ============================================================ P5 numbered mask
def numbered_mask(cond="BEFOREDRUG"):
    mem0 = load(cond, "membrane", 0)
    mask = load_mask(cond)
    ov = cc.numbered_mask_overlay(mem0, mask)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(ov); ax.axis("off")
    ax.set_title(f"Cellpose cell mask with numbered ROIs — {int(mask.max())} cells", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figP5_mask_numbered.png")); plt.close(fig)
    return ov


# ============================================================ P6 event raster
def _events(A):
    """peak frame indices per cell."""
    ev = []
    for j in range(A.shape[1]):
        pk, _ = find_peaks(A[:, j], height=THRESH, distance=2)
        ev.append(pk)
    return ev


def raster_figure():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    for ax, cond, title in [(axes[0], "BEFOREDRUG", "Baseline"),
                            (axes[1], "AFTERDRUG", "Latrunculin B")]:
        dff, _ = load_dff_cen(cond)
        cells = [c for c in dff.columns if c.startswith("Cell_")]
        A = np.nan_to_num(dff[cells].to_numpy(float))
        ev = _events(A)
        order = np.argsort([e[0] if len(e) else A.shape[0] for e in ev])
        ev = [ev[o] for o in order]
        ax.eventplot(ev, colors="#1a9850", lineoffsets=np.arange(len(ev)),
                     linelengths=0.8, linewidths=0.8)
        ax.set_title(f"{title}  ({len(ev)} cells)", fontweight="bold")
        ax.set_xlabel("Frame"); ax.set_ylabel("Cell (sorted by first event)")
        ax.set_xlim(0, A.shape[0]); ax.set_facecolor("#f3faf3")
    fig.suptitle("Ca²⁺ transient event raster", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figP6_raster.png")); plt.close(fig)


# ============================================================ P8 background regions
def background_figure(cond="BEFOREDRUG"):
    """Show the 3 signal-free background reference regions on the calcium frame."""
    ca0 = load(cond, "ca2", 0)
    boxes = cc.auto_bg_boxes(ca0, k=3, box=40)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.imshow(cc.stretch8(ca0, clahe=True), cmap="gray")
    for j, (x1, y1, x2, y2) in enumerate(boxes, 1):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#00e5ff", lw=2.2))
        ax.text(x1, y1 - 5, f"bg{j}", color="#00e5ff", fontsize=11, fontweight="bold")
    ax.set_title("Background reference regions (3 signal-free boxes)\n"
                 "per frame: 1.5×IQR outlier rejection → median → mean of 3 → subtract",
                 fontweight="bold", fontsize=10)
    ax.axis("off"); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figP8_background.png")); plt.close(fig)


# ============================================================ P9 ROI selection
def roi_figure(cond="BEFOREDRUG"):
    """Show the rectangular ROI and the cells whose masks intersect it."""
    ca0 = load(cond, "ca2", 0)
    mask = load_mask(cond)
    box = ROI_BOX[cond]
    roi_ids = set(cc.roi_cell_ids(mask, box))
    inside = np.isin(mask, list(roi_ids))
    disp = np.dstack([cc.stretch8(ca0, clahe=True)] * 3).astype(float)
    disp[find_boundaries(mask, mode="outer")] = [190, 60, 60]        # all cells: faint red outline
    disp[inside] = 0.55 * disp[inside] + 0.45 * np.array([0, 200, 220])  # ROI cells: cyan fill
    disp[find_boundaries(inside, mode="outer")] = [0, 230, 255]
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.imshow(np.clip(disp, 0, 255).astype(np.uint8))
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", lw=2.5))
    ax.set_title(f"Region-of-interest selection — {len(roi_ids)} of {int(mask.max())} cells "
                 f"intersect the ROI (cyan)\ntraces for these cells are exported to a separate CSV",
                 fontweight="bold", fontsize=10)
    ax.axis("off"); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figP9_roi.png")); plt.close(fig)


# ============================================================ P7 regional bars
def region_bars():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, cond, title in [(axes[0], "BEFOREDRUG", "Baseline"),
                            (axes[1], "AFTERDRUG", "Latrunculin B")]:
        dff, cen = load_dff_cen(cond)
        cells = [c for c in dff.columns if c.startswith("Cell_")]
        A = np.nan_to_num(dff[cells].to_numpy(float))
        ids = np.array([int(c.split("_")[1]) for c in cells])
        peak = A.max(0) * 100.0                                   # ΔF/F0 in %
        ypos = cen.set_index("cell_number")["y_position"].reindex(ids).to_numpy()
        # 4 proximal->distal bands along the fin (y axis)
        edges = np.nanpercentile(ypos, [0, 25, 50, 75, 100])
        groups, means, sems = [], [], []
        for r in range(4):
            sel = (ypos >= edges[r]) & (ypos <= edges[r + 1] if r == 3 else ypos < edges[r + 1])
            vals = peak[sel & ~np.isnan(ypos)]
            groups.append(vals); means.append(np.mean(vals)); sems.append(np.std(vals) / max(1, np.sqrt(len(vals))))
        ax.bar(range(4), means, yerr=sems, capsize=4,
               color=["#4575b4", "#74add1", "#abd9e9", "#e0f3f8"], edgecolor="k")
        ax.set_xticks(range(4)); ax.set_xticklabels(["R1", "R2", "R3", "R4"])
        ax.set_ylabel("ΔF/F₀ [%]"); ax.set_title(title, fontweight="bold")
        try:
            H, p = kruskal(*groups)
            star = "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
        except Exception:
            p, star = np.nan, "n/a"
        y = max(means) + max(sems) * 1.2
        ax.plot([0, 3], [y, y], "k", lw=1); ax.text(1.5, y, star, ha="center", va="bottom")
        ax.set_ylim(top=y * 1.25)
    fig.suptitle("Regional single-cell Ca²⁺ activity (Kruskal–Wallis across R1–R4)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figP7_region_bars.png")); plt.close(fig)


# ============================================================ P1 pipeline schematic
def pipeline_schematic(before_ov, after_ov, mask_ov):
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.9, 2.4], hspace=0.15, wspace=0.25)
    stages = ["Membrane segmentation\n(first frame)", "Image registration\n(motion correction)",
              "Extract Ca²⁺ dynamics\n(per cell)", "Single-cell Ca²⁺\nactivity analysis",
              "Tissue-level Ca²⁺\npattern analysis"]
    # top: labeled boxes with arrows
    axtop = fig.add_subplot(gs[0, :]); axtop.axis("off"); axtop.set_xlim(0, 5); axtop.set_ylim(0, 1)
    for i, s in enumerate(stages):
        box = FancyBboxPatch((i + 0.06, 0.25), 0.82, 0.5, boxstyle="round,pad=0.02",
                             fc="#eaf2fb", ec="#2c6fbb", lw=1.6)
        axtop.add_patch(box)
        axtop.text(i + 0.47, 0.5, s, ha="center", va="center", fontsize=10, fontweight="bold")
        if i < 4:
            axtop.add_patch(FancyArrow(i + 0.9, 0.5, 0.14, 0, width=0.03, head_width=0.12,
                                       head_length=0.05, fc="#2c6fbb", ec="#2c6fbb"))
    # bottom row: real mini panels under each stage
    # 1 mask
    a = fig.add_subplot(gs[1, 0]); a.imshow(mask_ov); a.axis("off"); a.set_title("numbered ROIs", fontsize=8)
    # 2 registration before/after stacked
    a = fig.add_subplot(gs[1, 1]); a.imshow(np.vstack([before_ov, np.full((6, before_ov.shape[1], 3), 255, np.uint8), after_ov]))
    a.axis("off"); a.set_title("before / after", fontsize=8)
    # 3 traces
    a = fig.add_subplot(gs[1, 2])
    dff = pd.read_csv(os.path.join(ROOT, TRIAL, "AFTERDRUG", "all_cells_normalized.csv"))
    cells = [c for c in dff.columns if c.startswith("Cell_")][:40]
    for c in cells:
        a.plot(dff["Frame"], dff[c], lw=0.4, alpha=0.5)
    a.set_title("ΔF/F₀ traces", fontsize=8); a.set_xlabel("frame", fontsize=7); a.tick_params(labelsize=6)
    # 4 bar chart mini
    a = fig.add_subplot(gs[1, 3])
    A = np.nan_to_num(dff[[c for c in dff.columns if c.startswith('Cell_')]].to_numpy(float))
    q = np.array_split(np.argsort(A.max(0)), 4)
    a.bar(range(4), [A[:, idx].max(0).mean()*100 for idx in q][::-1],
          color=["#4575b4","#74add1","#abd9e9","#e0f3f8"], edgecolor="k")
    a.set_xticks(range(4)); a.set_xticklabels(["R1","R2","R3","R4"], fontsize=6)
    a.set_title("ΔF/F₀ [%]", fontsize=8); a.tick_params(labelsize=6)
    # 5 kymograph / heatmap mini
    a = fig.add_subplot(gs[1, 4])
    order = np.argsort(A.max(1))[::-1]
    a.imshow(A[:, order].T, aspect="auto", cmap="gray", vmin=0, vmax=np.percentile(A, 99))
    a.set_title("tissue kymograph", fontsize=8); a.axis("off")
    fig.suptitle("Figure 1. Image and data processing pipeline", fontweight="bold", fontsize=13)
    fig.savefig(os.path.join(FIGS, "figP1_pipeline.png")); plt.close(fig)


def main():
    print("P2/P3 registration overlays…")
    b_before, b_after, _ = registration_figure("BEFOREDRUG", "Baseline", method="rigid")
    registration_figure("AFTERDRUG", "After drug", method="compare")   # rigid vs elastic comparison
    print("P4 processing row…"); mask, mem0 = processing_row("BEFOREDRUG")
    print("P5 numbered mask…"); mask_ov = numbered_mask("BEFOREDRUG")
    print("P6 event raster…"); raster_figure()
    print("P7 regional bars…"); region_bars()
    print("P8 background regions…"); background_figure("BEFOREDRUG")
    print("P9 ROI selection…"); roi_figure("BEFOREDRUG")
    print("P1 pipeline schematic…"); pipeline_schematic(b_before, b_after, mask_ov)
    print("done ->", FIGS)


if __name__ == "__main__":
    main()
