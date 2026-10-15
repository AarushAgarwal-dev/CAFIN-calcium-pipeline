"""Exploratory audit of six CAFIN before/after recordings at the trial level.

Run from the repository root::

    python REPRODUCE/six_trial_analysis.py --assets PATH_TO_VERIFICATION_ASSETS

The full-recording analysis preserves the pipeline's F0i: each cell's own ten
lowest raw fluorescence values across its recording, with a floor of 1. It is
duration-confounded because after recordings often have more frames. A separate
first-30-frame sensitivity check recomputes each cell's F0i in that window.
Recordings are paired by names; biological pairing is unverified. This script
does not reproduce the reference paper's elastic registration or its results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import wilcoxon
import tifffile

WINDOW_FRAMES = 30
N_BASE = 10
F0_FLOOR = 1.0
PEAK_THRESHOLD = 0.5
PEAK_DISTANCE = 2
MAX_CORRELATION_PAIRS = 4000
PAIR_NAMES = {
    i: (
        f"lata_trial{i}_beforedrug" if i <= 4 else f"trial_{i}_before",
        f"lata_trial{i}_afterdrug" if i <= 4 else f"trial_{i}_after",
    )
    for i in range(1, 7)
}
METRICS = {
    "mean_peak_dff0i": "Mean peak ΔF/F₀ᵢ",
    "transients_per_cell_frame": "Transients per cell per frame",
    "temporal_sync_r": "Temporal synchrony (r)",
    "spatial_heterogeneity_cv": "Peak heterogeneity (CV)",
    "active_cell_fraction": "Active cell fraction",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_matrix(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    cols = [col for col in df if col.startswith("Cell_")]
    if not cols or df.columns[0] != "Frame":
        raise ValueError("Expected Frame followed by Cell_* columns")
    ids = [int(col.split("_")[1]) for col in cols]
    if ids != sorted(set(ids)):
        raise ValueError("Cell labels are duplicated or out of order")
    frames = df["Frame"].to_numpy()
    if not np.array_equal(frames, np.arange(len(df))):
        raise ValueError("Frame numbers must start at zero and be consecutive")
    values = df[cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite per-cell fluorescence value")
    return cols, values


def normalize_each_cell(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply cafin_core's default F0i formula to a frames × cells matrix."""
    n_base = min(N_BASE, max(3, raw.shape[0] // 3))
    f0 = np.maximum(np.sort(raw, axis=0)[:n_base].mean(axis=0), F0_FLOOR)
    return (raw - f0) / f0, f0


def load_recording(folder: Path) -> dict:
    raw_path = folder / "cafin_raw_traces_bg_corrected.csv"
    f0_path = folder / "cafin_f0i.csv"
    dff_path = folder / "cafin_dff0i.csv"
    mask_path = folder / "cafin_mask.tif"
    summary_path = folder / "summary.json"
    for path in (raw_path, f0_path, dff_path, mask_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_df = pd.read_csv(raw_path)
    cols, raw = _cell_matrix(raw_df)
    dff_df = pd.read_csv(dff_path)
    dff_cols, saved_dff = _cell_matrix(dff_df)
    if cols != dff_cols or raw.shape != saved_dff.shape:
        raise ValueError(f"Raw and normalized cell/frame shape mismatch in {folder}")
    computed_dff, f0 = normalize_each_cell(raw)
    f0_df = pd.read_csv(f0_path).set_index("cell")
    saved_f0 = f0_df.loc[cols, "F0i"].to_numpy(dtype=float)
    f0_error = float(np.max(np.abs(saved_f0 - f0)))
    dff_error = float(np.max(np.abs(saved_dff - computed_dff)))
    if f0_error > 1e-8 or dff_error > 1e-8:
        raise ValueError(f"Per-cell F0i audit failed in {folder}: {f0_error}, {dff_error}")

    mask = tifffile.imread(mask_path)
    labels = np.unique(mask)
    labels = set(labels[labels > 0].astype(int).tolist())
    cell_ids = {int(col.split("_")[1]) for col in cols}
    if mask.ndim != 2 or labels != cell_ids:
        raise ValueError(f"Mask labels do not match trace columns in {folder}")
    meta = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(raw) != meta["frames"] or len(cols) != meta["n_cells"]:
        raise ValueError(f"Source metadata does not match traces in {folder}")
    if mask.shape != (meta["height"], meta["width"]):
        raise ValueError(f"Source metadata does not match mask in {folder}")
    source_path = Path(meta["source"]["file"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Original image absent: {source_path}")
    source = {
        "recording": folder.name,
        "source_file": source_path.name,
        "source_size_bytes": source_path.stat().st_size,
        "raw_trace_sha256": sha256(raw_path),
        "mask_sha256": sha256(mask_path),
        "frames": len(raw),
        "cells": len(cols),
        "projection": meta["projection"],
        "segmentation_method": meta["segmentation_method"],
        "f0i_method": meta["f0i_method"],
        "registration": meta["registration"],
        "cells_at_f0_floor": int(np.count_nonzero(np.isclose(f0, F0_FLOOR))),
        "f0i_max_abs_error": f0_error,
        "dff_max_abs_error": dff_error,
    }
    return {"source": source, "raw": raw, "cell_names": cols}


def temporal_synchrony(dff: np.ndarray, active: np.ndarray, seed: int) -> float:
    """Mean correlation of a capped, deterministic sample of active cell pairs."""
    indices = np.flatnonzero(active)
    if len(indices) < 2:
        return float("nan")
    upper_i, upper_j = np.triu_indices(len(indices), 1)
    rng = np.random.default_rng(seed)
    if len(upper_i) > MAX_CORRELATION_PAIRS:
        chosen = rng.choice(len(upper_i), MAX_CORRELATION_PAIRS, replace=False)
        upper_i, upper_j = upper_i[chosen], upper_j[chosen]
    a, b = dff[:, indices[upper_i]], dff[:, indices[upper_j]]
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    valid = denominator > 1e-12
    if not valid.any():
        return float("nan")
    correlations = (a[:, valid] * b[:, valid]).sum(axis=0) / denominator[valid]
    return float(correlations.mean())


def summarize_recording(raw: np.ndarray, cell_names: list[str], *, trial: int,
                        condition: str, window: str) -> tuple[dict, pd.DataFrame]:
    dff, f0 = normalize_each_cell(raw)
    peaks = [find_peaks(dff[:, i], height=PEAK_THRESHOLD,
                        distance=PEAK_DISTANCE)[0] for i in range(dff.shape[1])]
    counts = np.array([len(p) for p in peaks], dtype=int)
    active = counts > 0
    peak_means = np.array([float(dff[p, i].mean()) if len(p) else np.nan
                           for i, p in enumerate(peaks)])
    per_cell = pd.DataFrame({
        "trial": trial, "condition": condition, "window": window,
        "cell": cell_names, "f0i": f0, "mean_dff0i": dff.mean(axis=0),
        "mean_peak_dff0i": peak_means, "transient_count": counts,
        "transients_per_frame": counts / len(raw), "active": active,
        "at_f0_floor": np.isclose(f0, F0_FLOOR),
    })
    active_peaks = peak_means[active]
    mean_peak = float(active_peaks.mean()) if len(active_peaks) else float("nan")
    peak_cv = (float(active_peaks.std(ddof=1) / mean_peak)
               if len(active_peaks) > 1 and mean_peak > 0 else float("nan"))
    summary = {
        "trial": trial, "condition": condition, "window": window,
        "frames": int(len(raw)), "cells": int(len(cell_names)),
        "mean_peak_dff0i": mean_peak,
        "mean_peak_dff0i_no_f0_floor": (
            float(np.nanmean(peak_means[~np.isclose(f0, F0_FLOOR)]))
            if np.any(active & ~np.isclose(f0, F0_FLOOR)) else float("nan")
        ),
        "median_peak_dff0i": float(np.median(active_peaks)) if len(active_peaks) else float("nan"),
        "transients_per_cell_frame": float(counts.sum() / (len(raw) * len(cell_names))),
        "temporal_sync_r": temporal_synchrony(dff, active, seed=trial * 10 + (condition == "after")),
        "spatial_heterogeneity_cv": peak_cv,
        "active_cell_fraction": float(active.mean()),
        "mean_dff0i": float(dff.mean()),
    }
    return summary, per_cell


def paired_tests(fields: pd.DataFrame) -> pd.DataFrame:
    before = fields[fields.condition == "before"].set_index("trial").sort_index()
    after = fields[fields.condition == "after"].set_index("trial").sort_index()
    if list(before.index) != list(after.index) or len(before) != 6:
        raise ValueError("Expected exactly six paired trials")
    rows = []
    for metric in METRICS:
        b, a = before[metric].to_numpy(float), after[metric].to_numpy(float)
        if not (np.isfinite(b).all() and np.isfinite(a).all()):
            raise ValueError(f"Nonfinite paired metric: {metric}")
        delta = a - b
        if np.all(delta == 0):
            stat, p = 0.0, 1.0
        else:
            result = wilcoxon(b, a, alternative="two-sided", method="exact")
            stat, p = float(result.statistic), float(result.pvalue)
        rows.append({"metric": metric, "n_pairs": len(b),
                     "median_before": float(np.median(b)),
                     "median_after": float(np.median(a)),
                     "median_paired_difference": float(np.median(delta)),
                     "wilcoxon_statistic": stat, "p_two_sided_exact": p})
    stats = pd.DataFrame(rows)
    # Benjamini–Hochberg across the five exploratory primary outcomes.
    p = stats["p_two_sided_exact"].to_numpy(float)
    order = np.argsort(p)
    sorted_q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(sorted_q, 1.0)
    stats["q_bh_five_metrics"] = q
    return stats


def figures(fields: pd.DataFrame, cells: pd.DataFrame, output: Path, window_label: str) -> None:
    before = fields[fields.condition == "before"].sort_values("trial")
    after = fields[fields.condition == "after"].sort_values("trial")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    trial_colors = plt.get_cmap("tab10")(np.arange(6))
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        for b, a in zip(before.itertuples(index=False), after.itertuples(index=False)):
            y = [getattr(b, metric), getattr(a, metric)]
            ax.plot([0, 1], y, marker="o", color=trial_colors[b.trial - 1],
                    alpha=.85, zorder=2, label=f"Trial {b.trial}")
        ax.set_xticks([0, 1], ["Before", "After"])
        ax.set_xlim(-.2, 1.2)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", title="Named pairs", frameon=False)
    fig.suptitle(f"Exploratory six named pairs: {window_label.replace('_', ' ')}")
    fig.savefig(output / f"paired_metrics_{window_label}.png", dpi=180)
    plt.close(fig)

    positive = cells[cells.mean_peak_dff0i.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    data = [positive.loc[positive.condition == c, "mean_peak_dff0i"].to_numpy()
            for c in ("before", "after")]
    axes[0].boxplot(data, tick_labels=["Before", "After"], showfliers=False)
    axes[0].set_ylabel("Mean detected peak ΔF/F₀ᵢ per active cell")
    axes[0].set_title("Pooled cells (descriptive; nested within trial)")
    b = before.set_index("trial")["mean_peak_dff0i"]
    a = after.set_index("trial")["mean_peak_dff0i"]
    pct = 100 * (a - b) / b
    axes[1].bar(pct.index.astype(str), pct.to_numpy(), color="#746b9d")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_xlabel("Paired trial")
    axes[1].set_ylabel("Change in mean peak ΔF/F₀ᵢ (%)")
    axes[1].set_title("Six trial-level changes")
    fig.suptitle(f"Exploratory cell distribution and trial changes: {window_label.replace('_', ' ')}")
    fig.savefig(output / f"cell_distribution_{window_label}.png", dpi=180)
    plt.close(fig)


def run(assets: Path, output: Path, window_frames: int = WINDOW_FRAMES) -> dict:
    if window_frames < N_BASE:
        raise ValueError("Window must contain at least ten frames")
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    sources, matched_rows, full_rows, matched_cells, full_cells = [], [], [], [], []
    source_files = set()
    for trial, pair in PAIR_NAMES.items():
        for condition, name in zip(("before", "after"), pair):
            recording = load_recording(assets / name)
            source = recording["source"]
            if source["source_file"] in source_files:
                raise ValueError(f"Repeated original recording: {source['source_file']}")
            source_files.add(source["source_file"])
            if source["frames"] < window_frames:
                raise ValueError(f"{name} has fewer than {window_frames} frames")
            source.update(trial=trial, condition=condition)
            sources.append(source)
            raw, names = recording["raw"], recording["cell_names"]
            matched, per_cell_matched = summarize_recording(raw[:window_frames], names,
                trial=trial, condition=condition, window=f"first_{window_frames}_frames")
            full, per_cell_full = summarize_recording(raw, names, trial=trial,
                condition=condition, window="full_recording")
            matched_rows.append(matched)
            full_rows.append(full)
            matched_cells.append(per_cell_matched)
            full_cells.append(per_cell_full)

    fields = pd.DataFrame(matched_rows)
    full_fields = pd.DataFrame(full_rows)
    cells_matched = pd.concat(matched_cells, ignore_index=True)
    cells_full = pd.concat(full_cells, ignore_index=True)
    stats_matched = paired_tests(fields)
    stats_full = paired_tests(full_fields)
    stats = pd.concat([stats_full.assign(window="full_recording"),
                       stats_matched.assign(window=f"first_{window_frames}_frames")],
                      ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    fields.to_csv(output / f"fields_first_{window_frames}_frames_sensitivity.csv", index=False)
    full_fields.to_csv(output / "fields_full_recording_exploratory.csv", index=False)
    cells_matched.to_csv(output / f"cells_first_{window_frames}_frames_sensitivity.csv", index=False)
    cells_full.to_csv(output / "cells_full_recording_exploratory.csv", index=False)
    stats.to_csv(output / "exploratory_paired_trial_tests.csv", index=False)
    pd.DataFrame(sources).to_csv(output / "source_audit.csv", index=False)
    figures(full_fields, cells_full, output, "full_recordings")
    figures(fields, cells_matched, output, f"first_{window_frames}_frames")
    report = {
        "n_pairs": 6, "n_recordings": 12,
        "pairing_basis": "before/after names; verify same specimen using laboratory records",
        "sensitivity_window_frames": window_frames,
        "normalization": "Each cell's own 10 lowest raw-fluorescence frames within each recording or analysis window; F0 floor 1",
        "stored_f0_audit": "Full-recording F0i and ΔF/F0i independently recomputed and compared to stored files",
        "inference_unit": "paired trial, n=6 by names only; cells are nested observations, not replicates",
        "test": "exploratory two-sided exact Wilcoxon signed-rank; Benjamini-Hochberg q across five metrics within each window",
        "sampling": "First 30 frames may not correspond to identical biological time without timing metadata",
        "full_recording": "Preserves stored pipeline F0i; duration-confounded because treated recordings have longer durations in several pairs",
        "reference_match": "Not a reproduction of the reference paper: these assets used rigid ECC registration, while its methods specify elastic registration; segmentation software version and biological pairing are unverified",
        "raw_source_availability": "Original ND2/TIFF source images were confirmed locally but are not bundled with these outputs or the repository",
        "excluded_duplicate": "baseline/afterdrug asset folders duplicate trial 3 and are excluded",
        "source_audit": sources,
        "paired_tests": stats.to_dict(orient="records"),
    }
    (output / "exploratory_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                                  encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True,
                        help="Directory containing verified trial condition folders")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "six_trial_results")
    parser.add_argument("--window-frames", type=int, default=WINDOW_FRAMES)
    args = parser.parse_args()
    report = run(args.assets, args.output, args.window_frames)
    print(f"Validated {report['n_recordings']} recordings in {report['n_pairs']} named pairs")
    print(pd.DataFrame(report["paired_tests"]).to_string(index=False))
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
