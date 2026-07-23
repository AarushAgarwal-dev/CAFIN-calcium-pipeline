"""
reproduce.py -- 100% reproducible reproduction of the CAFIN calcium-transient study.

Reads the pipeline's processed per-cell dF/F0 traces + cell centroids for BOTH
conditions (BEFOREDRUG = baseline, AFTERDRUG = Latrunculin) across BOTH trials,
then computes every quantity the manuscript claims:

  * single-cell activity statistics
  * tissue-level Ca2+ pattern metrics (spatial heterogeneity CV, temporal sync r)
  * baseline-vs-drug comparison WITH statistical tests (Mann-Whitney U + effect size)
  * spatial coordination (correlation-vs-distance) using cell centroids
  * time-dependence of the response (early vs late windows post-treatment)

Outputs (all under REPRODUCE/results/):
  metrics_per_cell.csv   field_summary.csv   stats_tests.csv   results.json
  figures/fig1..fig6 .png

Run:  python reproduce.py
No original manuscript text is read; everything derives from the CSV outputs.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu, pearsonr, shapiro, ttest_ind

# ----------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # CAFIN_Cleanedup_Code_Aarush
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(RESULTS, "figures")
REGEN = os.path.join(HERE, "regenerated")          # end-to-end recomputed CSVs (LATA1)
os.makedirs(FIGS, exist_ok=True)

RNG = np.random.default_rng(0)                     # fixed seed => identical every run
THRESH = 0.5                                       # dF/F0 to count as a transient
MAX_PAIRS = 4000                                   # cap pairwise corr sample per field

TRIALS = {"trial1": "LATA1TRAIL", "trial2": "LATA2TRIAL"}
CONDITIONS = {"BEFOREDRUG": "Baseline", "AFTERDRUG": "Latrunculin"}
COND_COLOR = {"Baseline": "#2c7fb8", "Latrunculin": "#d95f0e"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


# ----------------------------------------------------------------- loading
def _field_dir(trial_dir, cond):
    """Prefer the end-to-end recomputed CSVs (regenerated/) when present, else the
    existing pipeline CSVs. Returns (dir, provenance_string)."""
    regen = os.path.join(REGEN, trial_dir, cond)
    if os.path.exists(os.path.join(regen, "all_cells_normalized.csv")):
        return regen, "recomputed from raw images"
    return os.path.join(ROOT, trial_dir, cond), "pipeline CSV (raw TIFFs unavailable)"


def load_field(trial_dir, cond):
    """Return (dff matrix frames x cells, cell_ids, centroids dict, provenance)."""
    base, prov = _field_dir(trial_dir, cond)
    dff = pd.read_csv(os.path.join(base, "all_cells_normalized.csv"))
    cells = [c for c in dff.columns if c.startswith("Cell_")]
    A = dff[cells].to_numpy(float)
    ids = np.array([int(c.split("_")[1]) for c in cells])
    cen = pd.read_csv(os.path.join(base, "centroids_0.csv"))
    cx = dict(zip(cen["cell_number"].astype(int), cen["x_position"].astype(float)))
    cy = dict(zip(cen["cell_number"].astype(int), cen["y_position"].astype(float)))
    centroids = {i: (cx.get(i, np.nan), cy.get(i, np.nan)) for i in ids}
    return np.nan_to_num(A, nan=0.0), ids, centroids, prov


# ----------------------------------------------------------------- per-cell metrics
def cell_metrics(trace):
    pk, _ = find_peaks(trace, height=THRESH, distance=2)
    peak_amp = float(trace[pk].max()) if len(pk) else float(trace.max())
    return {
        # NOTE: peak_amp is a max over frames -> grows weakly with recording length, so it is
        # only partially comparable between 31-frame baseline and 55-90-frame treated recordings.
        # The length-normalized peaks_per_frame and per-frame auc are the sound cross-length metrics.
        "peak_amp": peak_amp,                          # max dF/F0 (extreme-value; length-sensitive)
        "mean_dff": float(trace.mean()),               # mean dF/F0 (length-independent)
        "n_peaks": int(len(pk)),
        "peaks_per_frame": len(pk) / len(trace),       # normalized: baseline 31 vs drug 90 frames
        "active": bool(trace.max() > THRESH),
        # supra-threshold area PER FRAME (normalized) so recordings of different length compare
        "auc": float(np.clip(trace - THRESH, 0, None).sum()) / len(trace),
    }


def pairwise_corr_sample(A, active_idx, centroids, ids):
    """Return arrays (r, dist) for a capped random sample of active-cell pairs."""
    idx = active_idx
    if len(idx) < 2:
        return np.array([]), np.array([])
    pairs = [(a, b) for i, a in enumerate(idx) for b in idx[i + 1:]]
    if len(pairs) > MAX_PAIRS:
        sel = RNG.choice(len(pairs), MAX_PAIRS, replace=False)
        pairs = [pairs[s] for s in sel]
    rs, ds = [], []
    for a, b in pairs:
        va, vb = A[:, a], A[:, b]
        if va.std() < 1e-9 or vb.std() < 1e-9:
            continue
        r = np.corrcoef(va, vb)[0, 1]
        (xa, ya), (xb, yb) = centroids[ids[a]], centroids[ids[b]]
        if np.isnan(xa) or np.isnan(xb):
            continue
        rs.append(r); ds.append(float(np.hypot(xa - xb, ya - yb)))
    return np.array(rs), np.array(ds)


def field_summary(A, ids, centroids, label):
    n_frames, n_cells = A.shape
    rows = [cell_metrics(A[:, j]) for j in range(n_cells)]
    dfc = pd.DataFrame(rows)
    dfc["cell_id"] = ids
    active_mask = dfc["active"].to_numpy()
    active_idx = np.where(active_mask)[0]
    amps_active = dfc.loc[active_mask, "peak_amp"].to_numpy()
    het_cv = float(amps_active.std() / amps_active.mean()) if amps_active.size else np.nan
    rs, ds = pairwise_corr_sample(A, active_idx, centroids, ids)
    sync_r = float(np.nanmean(rs)) if rs.size else np.nan
    summ = {
        "label": label, "n_frames": int(n_frames), "n_cells": int(n_cells),
        "active_fraction": float(active_mask.mean()),
        "mean_peak_amp": float(dfc["peak_amp"].mean()),
        "median_peak_amp": float(dfc["peak_amp"].median()),
        "mean_peaks_per_frame": float(dfc["peaks_per_frame"].mean()),
        "spatial_heterogeneity_cv": het_cv,
        "temporal_sync_r": sync_r,
    }
    return summ, dfc, rs, ds


# ----------------------------------------------------------------- stats helpers
def _shapiro_p(x):
    """Shapiro-Wilk normality p-value (subsample to 5000, scipy's cap)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    if len(x) > 5000:
        x = RNG.choice(x, 5000, replace=False)
    try:
        return float(shapiro(x).pvalue)
    except Exception:
        return np.nan


def mwu(a, b):
    """Compare baseline (a) vs drug (b) following the manuscript's decision rule:
    Shapiro-Wilk tests normality of each group; if BOTH are normal (p>0.05) a
    two-sided Welch t-test is used, otherwise the Mann-Whitney U test. Always
    reports the rank-biserial effect size for interpretability."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return dict(p=np.nan, U=np.nan, effect=np.nan, n1=len(a), n2=len(b),
                    med1=np.nanmedian(a) if len(a) else np.nan,
                    med2=np.nanmedian(b) if len(b) else np.nan,
                    shapiro_p1=np.nan, shapiro_p2=np.nan, normal=False, test="n/a")
    sp1, sp2 = _shapiro_p(a), _shapiro_p(b)
    normal = (sp1 > 0.05) and (sp2 > 0.05)
    U, p_mwu = mannwhitneyu(a, b, alternative="two-sided")
    eff = 1 - 2 * U / (len(a) * len(b))            # rank-biserial correlation (always reported)
    if normal:
        _, p = ttest_ind(a, b, equal_var=False)     # Welch t-test
        test = "Welch t-test"
    else:
        p, test = p_mwu, "Mann-Whitney U"           # nonparametric (data non-normal)
    return dict(p=float(p), U=float(U), effect=float(eff),
                n1=len(a), n2=len(b),
                med1=float(np.median(a)), med2=float(np.median(b)),
                shapiro_p1=float(sp1) if not np.isnan(sp1) else np.nan,
                shapiro_p2=float(sp2) if not np.isnan(sp2) else np.nan,
                normal=bool(normal), test=test)


def stars(p):
    if np.isnan(p): return "n/a"
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


# ================================================================ MAIN
def main():
    fields = {}                 # key "trial1/AFTERDRUG" -> dict
    per_cell_frames = []
    for tk, tdir in TRIALS.items():
        for cond, clabel in CONDITIONS.items():
            A, ids, cen, prov = load_field(tdir, cond)
            summ, dfc, rs, ds = field_summary(A, ids, cen, clabel)
            key = f"{tk}/{cond}"
            summ["trial"] = tk; summ["condition"] = clabel; summ["provenance"] = prov
            fields[key] = dict(A=A, ids=ids, cen=cen, summ=summ, dfc=dfc, rs=rs, ds=ds)
            dfc2 = dfc.copy(); dfc2["trial"] = tk; dfc2["condition"] = clabel
            per_cell_frames.append(dfc2)
            print(f"{key:22s} cells={summ['n_cells']:4d} active={summ['active_fraction']*100:5.1f}%"
                  f"  peakAmp(med)={summ['median_peak_amp']:.2f}  hetCV={summ['spatial_heterogeneity_cv']:.2f}"
                  f"  sync_r={summ['temporal_sync_r']:.3f}")

    per_cell = pd.concat(per_cell_frames, ignore_index=True)
    per_cell.to_csv(os.path.join(RESULTS, "metrics_per_cell.csv"), index=False)
    field_df = pd.DataFrame([f["summ"] for f in fields.values()])
    field_df.to_csv(os.path.join(RESULTS, "field_summary.csv"), index=False)

    # ---- pooled baseline vs drug (cells across both trials)
    base = per_cell[per_cell.condition == "Baseline"]
    drug = per_cell[per_cell.condition == "Latrunculin"]
    tests = {}
    for metric in ["peak_amp", "mean_dff", "peaks_per_frame", "auc"]:
        tests[metric] = mwu(base[metric], drug[metric])
    # pooled pairwise correlations (synchronization distribution)
    base_r = np.concatenate([fields[k]["rs"] for k in fields if "BEFORE" in k]) if any("BEFORE" in k for k in fields) else np.array([])
    drug_r = np.concatenate([fields[k]["rs"] for k in fields if "AFTER" in k])
    tests["pairwise_sync_r"] = mwu(base_r, drug_r)

    stats_rows = []
    for m, t in tests.items():
        stats_rows.append({"metric": m, "median_baseline": t["med1"], "median_drug": t["med2"],
                           "shapiro_p_baseline": t["shapiro_p1"], "shapiro_p_drug": t["shapiro_p2"],
                           "normal": t["normal"], "test_used": t["test"],
                           "p_value": t["p"], "rank_biserial_effect": t["effect"],
                           "n_baseline": t["n1"], "n_drug": t["n2"], "sig": stars(t["p"])})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(RESULTS, "stats_tests.csv"), index=False)
    print("\nStatistical tests (baseline vs Latrunculin, pooled cells):")
    print(stats_df.to_string(index=False))

    # ---- field-level heterogeneity / sync means (descriptive, per condition)
    def cond_mean(col, cond):
        vals = [f["summ"][col] for f in fields.values() if f["summ"]["condition"] == cond]
        return float(np.nanmean(vals)), [round(v, 3) for v in vals]
    het_base, het_base_v = cond_mean("spatial_heterogeneity_cv", "Baseline")
    het_drug, het_drug_v = cond_mean("spatial_heterogeneity_cv", "Latrunculin")
    sync_base, sync_base_v = cond_mean("temporal_sync_r", "Baseline")
    sync_drug, sync_drug_v = cond_mean("temporal_sync_r", "Latrunculin")
    act_base, _ = cond_mean("active_fraction", "Baseline")
    act_drug, _ = cond_mean("active_fraction", "Latrunculin")

    results = {
        "conditions": {k: v["summ"] for k, v in fields.items()},
        "tests": tests,
        "field_means": {
            "spatial_heterogeneity_cv": {"Baseline": het_base, "Latrunculin": het_drug,
                                          "per_field_Baseline": het_base_v, "per_field_Latrunculin": het_drug_v},
            "temporal_sync_r": {"Baseline": sync_base, "Latrunculin": sync_drug,
                                 "per_field_Baseline": sync_base_v, "per_field_Latrunculin": sync_drug_v},
            "active_fraction": {"Baseline": act_base, "Latrunculin": act_drug},
        },
        "params": {"peak_threshold": THRESH, "max_pairs": MAX_PAIRS, "seed": 0},
    }
    with open(os.path.join(RESULTS, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ============================================================ FIGURES
    make_figures(fields, per_cell, tests, results)
    print(f"\nAll outputs written to: {RESULTS}")
    return results


# ----------------------------------------------------------------- figures
def make_figures(fields, per_cell, tests, results):
    # ---- FIG 1: representative dF/F0 traces, baseline vs drug (trial1) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, key, title in [(axes[0], "trial1/BEFOREDRUG", "Baseline"),
                           (axes[1], "trial1/AFTERDRUG", "Latrunculin B")]:
        A = fields[key]["A"]; f = np.arange(A.shape[0])
        # pick 40 most active cells for legibility, strong contrast
        order = np.argsort(A.max(0))[::-1][:40]
        for j in order:
            ax.plot(f, A[:, j], lw=0.5, alpha=0.35, color=COND_COLOR[fields[key]["summ"]["condition"]])
        ax.plot(f, A.mean(1), lw=2.6, color="black", label="population mean")
        ax.axhline(THRESH, color="grey", ls="--", lw=1, alpha=0.7)
        ax.set_title(f"{title}  (n={A.shape[1]} cells)"); ax.set_xlabel("Frame")
        ax.set_ylabel("ΔF/F0"); ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Single-cell Ca²⁺ transients: baseline vs actin disruption", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig1_traces.png")); plt.close(fig)

    # ---- FIG 2: activity heatmaps (high contrast, shared scale) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    vmax = np.percentile(np.concatenate([fields["trial1/BEFOREDRUG"]["A"].ravel(),
                                         fields["trial1/AFTERDRUG"]["A"].ravel()]), 99)
    for ax, key, title in [(axes[0], "trial1/BEFOREDRUG", "Baseline"),
                           (axes[1], "trial1/AFTERDRUG", "Latrunculin B")]:
        A = fields[key]["A"]
        order = np.argsort(A.max(1))[::-1]
        im = ax.imshow(A[:, order].T, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title); ax.set_xlabel("Frame"); ax.set_ylabel("Cell (sorted)")
        ax.grid(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="ΔF/F0", shrink=0.8)
    fig.suptitle("Tissue-level Ca²⁺ activity (cells × frames)", fontweight="bold")
    fig.savefig(os.path.join(FIGS, "fig2_heatmaps.png")); plt.close(fig)

    # ---- FIG 3: metric comparison with p-values ----
    metrics = [("peak_amp", "Peak ΔF/F0"), ("mean_dff", "Mean ΔF/F0"),
               ("peaks_per_frame", "Peaks / frame"), ("auc", "Transient AUC")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (m, lab) in zip(axes, metrics):
        b = per_cell[per_cell.condition == "Baseline"][m].to_numpy()
        d = per_cell[per_cell.condition == "Latrunculin"][m].to_numpy()
        parts = ax.violinplot([b, d], showmeans=True, showextrema=False)
        for pc, c in zip(parts["bodies"], [COND_COLOR["Baseline"], COND_COLOR["Latrunculin"]]):
            pc.set_facecolor(c); pc.set_alpha(0.6)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Baseline", "Latrunculin"])
        ax.set_title(lab)
        t = tests[m]
        y = np.nanpercentile(np.concatenate([b, d]), 97)
        ax.plot([1, 2], [y, y], color="k", lw=1)
        ax.text(1.5, y, f"{stars(t['p'])}\np={t['p']:.1e}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(top=y * 1.35 if y > 0 else None)
    fig.suptitle("Baseline vs Latrunculin B — single-cell metrics (Mann–Whitney U)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_metrics.png")); plt.close(fig)

    # ---- FIG 4: spatial coordination (correlation vs distance) ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for cond, keys in [("Baseline", [k for k in fields if "BEFORE" in k]),
                       ("Latrunculin", [k for k in fields if "AFTER" in k])]:
        rs = np.concatenate([fields[k]["rs"] for k in keys])
        ds = np.concatenate([fields[k]["ds"] for k in keys])
        if ds.size == 0:
            continue
        edges = np.linspace(0, np.percentile(ds, 95), 12)
        cent = 0.5 * (edges[:-1] + edges[1:])
        mean_r = [np.nanmean(rs[(ds >= edges[i]) & (ds < edges[i + 1])]) if np.any((ds >= edges[i]) & (ds < edges[i + 1])) else np.nan
                  for i in range(len(edges) - 1)]
        ax.plot(cent, mean_r, "-o", color=COND_COLOR[cond], label=cond, lw=2, ms=5)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Inter-cell distance (px)"); ax.set_ylabel("Mean pairwise correlation r")
    ax.set_title("Spatial coordination of Ca²⁺ activity", fontweight="bold")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_spatial.png")); plt.close(fig)

    # ---- FIG 5: heterogeneity + synchronization summary ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fm = results["field_means"]
    for ax, key, lab in [(axes[0], "spatial_heterogeneity_cv", "Spatial heterogeneity (CV of peak ΔF/F0)"),
                         (axes[1], "temporal_sync_r", "Temporal synchronization (mean pairwise r)")]:
        bv = fm[key]["per_field_Baseline"]; dv = fm[key]["per_field_Latrunculin"]
        ax.bar([0, 1], [np.mean(bv), np.mean(dv)],
               color=[COND_COLOR["Baseline"], COND_COLOR["Latrunculin"]], alpha=0.75, width=0.6)
        ax.scatter([0] * len(bv), bv, color="k", zorder=3, s=30)
        ax.scatter([1] * len(dv), dv, color="k", zorder=3, s=30)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "Latrunculin"])
        ax.set_title(lab, fontsize=10)
    fig.suptitle("Tissue-level pattern metrics (points = individual trials)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig5_pattern_metrics.png")); plt.close(fig)

    # ---- FIG 6: time-dependence within Latrunculin fields ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in [k for k in fields if "AFTER" in k]:
        A = fields[key]["A"]; n = A.shape[0]
        win = np.array_split(np.arange(n), 3)
        frac = [(A[w].max(0) > THRESH).mean() for w in win]        # active fraction per window
        amp = [np.nanmean(A[w].max(0)) for w in win]
        ax.plot([1, 2, 3], amp, "-o", label=f"{fields[key]['summ']['trial']} mean peak", lw=2)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Early", "Mid", "Late"])
    ax.set_xlabel("Post-treatment window"); ax.set_ylabel("Mean per-cell peak ΔF/F0")
    ax.set_title("Time-dependence of the Latrunculin response", fontweight="bold")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig6_time.png")); plt.close(fig)


if __name__ == "__main__":
    main()
