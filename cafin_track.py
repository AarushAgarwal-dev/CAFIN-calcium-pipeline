"""
cafin_track.py  --  true cell tracking, ported verbatim from the wound-closure
pipeline's pipeline/track.py (Author: Linlin Li; improved by that project with
optimal Hungarian assignment + gap closing).

Algorithm (unchanged):
  * per-frame independent segmentation is linked across time;
  * weighted match score = IoU*w_overlap + size-sim*w_size + centroid-sim*w_centroid,
    gated by min pixel-overlap / max centroid-distance / min IoU;
  * optimal Hungarian assignment (scipy linear_sum_assignment) or greedy;
  * forward + backward propagation from a reference frame;
  * gap closing: fragments separated by <= gap frames are re-linked.

Usage in CAFIN:
    masks = cafin_core.segment_stack(reg_membrane_frames, diameter=15)   # (T,H,W)
    tracked, summary, stats = cafin_track.run(masks, TrackParams())
`tracked[t]` carries each cell's stable global id in every frame.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackParams:
    """The track_* knobs from the wound pipeline's Params (same defaults)."""
    track_method: str = "hungarian"       # "hungarian" (optimal) | "greedy"
    track_min_iou: float = 0.1            # min IoU to consider a match
    track_pixel_threshold: int = 50       # min absolute pixel overlap to accept
    track_max_distance_px: int = 50       # max centroid-centroid distance
    track_start_frame: int = 0            # reference frame; tracks fwd + bwd
    track_gap_frames: int = 2             # maximum missing source frames
    track_w_overlap: float = 0.5          # score weights (sum to 1)
    track_w_size: float = 0.3
    track_w_centroid: float = 0.2
    out_dir: str | None = None            # if set, save tracked_mask.tiff + summaries


def _features(mask):
    """label -> dict(centroid=(x,y), size) for all labels in a frame."""
    out = {}
    ids = np.unique(mask); ids = ids[ids != 0]
    for l in ids:
        ys, xs = np.where(mask == l)
        out[int(l)] = dict(centroid=(xs.mean(), ys.mean()), size=xs.size,
                           cx=xs.mean(), cy=ys.mean())
    return out


def _score(fa, fb, iou, p):
    size_sim = 1.0 / (1.0 + abs(fa["size"] - fb["size"]) / max(fa["size"], fb["size"]))
    dist = np.hypot(fa["centroid"][0] - fb["centroid"][0],
                    fa["centroid"][1] - fb["centroid"][1])
    cent_sim = 1.0 / (1.0 + dist / 100.0)
    return p.track_w_overlap * iou + p.track_w_size * size_sim + p.track_w_centroid * cent_sim


def match_frame(mask_curr, mask_prev, params):
    """Return dict curr_label -> prev_label using optimal/greedy matching."""
    fc = _features(mask_curr)
    fp = _features(mask_prev)
    if not fc or not fp:
        return {}
    cl = list(fc); pl = list(fp)
    S = np.zeros((len(cl), len(pl)))
    for i, c in enumerate(cl):
        a = fc[c]
        for j, p_ in enumerate(pl):
            b = fp[p_]
            if np.hypot(a["centroid"][0] - b["centroid"][0],
                        a["centroid"][1] - b["centroid"][1]) > params.track_max_distance_px:
                continue
            inter = int(np.logical_and(mask_curr == c, mask_prev == p_).sum())
            if inter < params.track_pixel_threshold:
                continue
            union = int(np.logical_or(mask_curr == c, mask_prev == p_).sum())
            iou = inter / union if union else 0
            if iou < params.track_min_iou:
                continue
            S[i, j] = _score(a, b, iou, params)

    matches = {}
    if params.track_method == "greedy":
        order = sorted(((S[i, j], i, j) for i in range(len(cl)) for j in range(len(pl)) if S[i, j] > 0),
                       reverse=True)
        uc, up = set(), set()
        for s, i, j in order:
            if i not in uc and j not in up:
                matches[cl[i]] = pl[j]; uc.add(i); up.add(j)
    else:  # hungarian (optimal)
        if S.max() > 0:
            row, col = linear_sum_assignment(-S)
            for i, j in zip(row, col):
                if S[i, j] > 0:
                    matches[cl[i]] = pl[j]
    return matches


def unified_tracking(masks, params, progress=None):
    n = len(masks)
    tracked = [np.zeros_like(masks[i], dtype=np.uint16) for i in range(n)]
    info = {"new_cells": {}, "lost_cells": {}}
    start = int(np.clip(params.track_start_frame, 0, n - 1))
    next_id = 1
    for lab in sorted(np.unique(masks[start])):
        if lab == 0:
            continue
        tracked[start][masks[start] == lab] = next_id
        next_id += 1

    def propagate(t_from, t_to):
        nonlocal next_id
        matches = match_frame(masks[t_to], masks[t_from], params)
        for c_label, r_label in matches.items():
            tids = np.unique(tracked[t_from][masks[t_from] == r_label])
            tids = tids[tids > 0]
            if tids.size:
                tracked[t_to][masks[t_to] == c_label] = tids[0]
        for c_label in np.unique(masks[t_to]):
            if c_label == 0 or c_label in matches:
                continue
            tracked[t_to][masks[t_to] == c_label] = next_id
            info["new_cells"].setdefault(t_to, []).append(next_id)
            next_id += 1

    done, total = 0, max(n - 1, 1)
    for t in range(start + 1, n):
        propagate(t - 1, t); done += 1
        if progress: progress(done / total, f"Tracking fwd {t + 1}/{n}")
    for t in range(start - 1, -1, -1):
        propagate(t + 1, t); done += 1
        if progress: progress(done / total, f"Tracking bwd {t + 1}/{n}")
    return tracked, info


def close_gaps(tracked, params, frame_idx=None, return_links=False):
    """Re-link mutual-nearest fragments across at most ``gap`` raw frames.

    ``frame_idx`` contains the source movie indices. This matters when the GUI
    processes every Nth frame: a gap of one processed frame may span many raw
    frames and should not be treated as a short interruption. Each fragment
    needs at least two observations before proximity-based relinking is tried.
    """
    gap = params.track_gap_frames
    if gap <= 0:
        return (tracked, []) if return_links else tracked
    n = len(tracked)
    if frame_idx is None:
        frame_idx = list(range(n))
    if len(frame_idx) != n or any(b <= a for a, b in zip(frame_idx, frame_idx[1:])):
        raise ValueError("frame_idx must contain one increasing source index per mask")
    ids = sorted({int(i) for f in tracked for i in np.unique(f) if i})
    span, head, tail, n_seen = {}, {}, {}, {}
    for i in ids:
        frames = [t for t in range(n) if np.any(tracked[t] == i)]
        if not frames:
            continue
        span[i] = (frames[0], frames[-1])
        n_seen[i] = len(frames)
        for store, fr in ((head, frames[0]), (tail, frames[-1])):
            ys, xs = np.where(tracked[fr] == i)
            store[i] = (xs.mean(), ys.mean(), xs.size)

    remap = {}
    def root(x):
        while x in remap:
            x = remap[x]
        return x

    candidates = {}
    for a in ids:
        if a not in span or n_seen[a] < 2:
            continue
        te = span[a][1]
        ax, ay, asz = tail[a]
        for b in ids:
            if b == a or b not in span or n_seen[b] < 2:
                continue
            ts = span[b][0]
            if ts <= te or frame_idx[ts] - frame_idx[te] - 1 > gap:
                continue
            bx, by, bsz = head[b]
            d = np.hypot(ax - bx, ay - by)
            if d > params.track_max_distance_px:
                continue
            if min(asz, bsz) / max(asz, bsz) < 0.5:
                continue
            candidates[(a, b)] = d

    # Both fragments must choose one another. The old loop let two ended
    # tracks claim the same new fragment; the later iteration silently won.
    best_head = {}
    best_tail = {}
    for (a, b), d in candidates.items():
        if a not in best_head or (d, b) < best_head[a]:
            best_head[a] = (d, b)
        if b not in best_tail or (d, a) < best_tail[b]:
            best_tail[b] = (d, a)
    links = []
    for a in ids:
        if a not in best_head:
            continue
        d, b = best_head[a]
        if best_tail[b][1] != a or root(b) == root(a):
            continue
        remap[b] = a
        links.append({"from_id": a, "to_id": b,
                      "last_frame": int(frame_idx[span[a][1]]),
                      "first_frame": int(frame_idx[span[b][0]]),
                      "distance_px": float(d)})

    if remap:
        for t in range(n):
            f = tracked[t]
            for b in list(remap):
                rb = root(b)
                if rb != b:
                    f[f == b] = rb
    return (tracked, links) if return_links else tracked


def run(masks, params, frame_idx=None, progress=None):
    """masks: (T,H,W) int array of per-frame independent segmentations.
    Frames must be consecutive in the original recording. Returns tracked
    global IDs, summary rows, and stats with provisional gap-link details."""
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("masks must be a nonempty (frames, height, width) stack")
    T = masks.shape[0]
    if frame_idx is None:
        frame_idx = list(range(T))
    if len(frame_idx) != T or any(b <= a for a, b in zip(frame_idx, frame_idx[1:])):
        raise ValueError("frame_idx must contain one increasing source index per mask")
    if any(b - a != 1 for a, b in zip(frame_idx, frame_idx[1:])):
        raise ValueError("Individual-cell tracking requires consecutive source frames")
    tracked_list, info = unified_tracking([masks[t] for t in range(T)], params, progress)
    tracked_list, gap_links = close_gaps(tracked_list, params, frame_idx, return_links=True)
    info["gap_links"] = gap_links
    tracked = np.stack(tracked_list).astype(np.uint16)

    ids = sorted({int(i) for t in range(T) for i in np.unique(tracked[t]) if i})
    total_coordinate, rows = [], []
    for i in ids:
        traj = np.zeros((T, 2), int)
        present = np.zeros(T, bool)
        for t in range(T):
            ys, xs = np.where(tracked[t] == i)
            if xs.size:
                traj[t] = [int(xs.mean()), int(ys.mean())]
                present[t] = True
        total_coordinate.append(traj)
        fr = np.where(present)[0]
        rows.append((i, int(frame_idx[fr[0]]), int(frame_idx[fr[-1]]),
                     int(present.sum()), round(float(present.mean()), 3)))

    if params.out_dir:                          # optional: persist like the original
        import tifffile
        os.makedirs(params.out_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(params.out_dir, "tracked_mask.tiff"), tracked)
        with open(os.path.join(params.out_dir, "track_summary.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["track_id", "first_frame", "last_frame", "n_frames", "coverage"])
            w.writerows(rows)
        np.save(os.path.join(params.out_dir, "traced_result_2D.npy"),
                {"total_coordinate": total_coordinate,
                 "total_traced_cell_mask": [tracked[t] for t in range(T)],
                 "track_ids": ids, "tracking_info": info}, allow_pickle=True)

    stats = dict(n_tracks=len(ids),
                 n_new=int(sum(len(v) for v in info["new_cells"].values())),
                 n_gap_links=len(gap_links),
                 gap_links=gap_links,
                 full_coverage=int(sum(1 for r in rows if r[4] == 1.0)))
    return tracked, rows, stats
