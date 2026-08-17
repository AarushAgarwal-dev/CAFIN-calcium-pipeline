import os
import sys

# Ensure repository root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import pytest
import cafin_core as cc


@pytest.fixture
def synthetic_stack():
    """Create a synthetic 3D calcium movie (T=50, H=40, W=40) with 4 distinct cell transient regions."""
    rng = np.random.default_rng(42)
    T, H, W = 50, 40, 40

    # Common global baseline wave
    t = np.linspace(0, 2 * np.pi, T)
    global_wave = 5.0 * np.sin(t)

    # Distinct transient peaks for 4 cell regions
    c1_event = 30.0 * np.exp(-0.5 * ((np.arange(T) - 15) / 3.0) ** 2)
    c2_event = 25.0 * np.exp(-0.5 * ((np.arange(T) - 18) / 3.0) ** 2)
    c3_event = 30.0 * np.exp(-0.5 * ((np.arange(T) - 32) / 3.0) ** 2)
    c4_event = 25.0 * np.exp(-0.5 * ((np.arange(T) - 40) / 3.0) ** 2)

    mask0 = np.zeros((H, W), dtype=np.int32)
    mask0[5:15, 5:15] = 1
    mask0[5:15, 25:35] = 2
    mask0[25:35, 5:15] = 3
    mask0[25:35, 25:35] = 4

    stack = np.zeros((T, H, W), dtype=np.float32)
    for frame_idx in range(T):
        frame = rng.normal(40.0, 3.0, (H, W)).astype(np.float32)
        frame += global_wave[frame_idx]

        # Add cell-specific transients with independent noise per pixel
        frame[5:15, 5:15] += c1_event[frame_idx] + rng.normal(0, 2.0, (10, 10)).astype(np.float32)
        frame[5:15, 25:35] += c2_event[frame_idx] + rng.normal(0, 2.0, (10, 10)).astype(np.float32)
        frame[25:35, 5:15] += c3_event[frame_idx] + rng.normal(0, 2.0, (10, 10)).astype(np.float32)
        frame[25:35, 25:35] += c4_event[frame_idx] + rng.normal(0, 2.0, (10, 10)).astype(np.float32)

        stack[frame_idx] = np.clip(frame, 0, None)

    ca_dict = {i: stack[i] for i in range(T)}
    return {"ca_dict": ca_dict, "stack": stack, "mask0": mask0, "frames": list(range(T))}


def test_reproducible_sampling(synthetic_stack):
    """Same random seed must yield identical sampled pixels and results."""
    data = synthetic_stack
    res1 = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=40,
        seed=123,
        tissue_r_thresh=0.2,
        r2_thresh=0.75,
        k_clique=3,
    )
    res2 = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=40,
        seed=123,
        tissue_r_thresh=0.2,
        r2_thresh=0.75,
        k_clique=3,
    )
    assert res1.get("error") is None
    assert res2.get("error") is None
    np.testing.assert_array_equal(res1["sampled_yx"], res2["sampled_yx"])
    np.testing.assert_array_equal(res1["retained_yx"], res2["retained_yx"])
    pd.testing.assert_frame_equal(res1["nodes_df"], res2["nodes_df"])


def test_different_seeds_differ(synthetic_stack):
    """Different random seeds should sample different pixel coordinates."""
    data = synthetic_stack
    res1 = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=40,
        seed=1,
        tissue_r_thresh=0.2,
        r2_thresh=0.75,
        k_clique=3,
    )
    res2 = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=40,
        seed=2,
        tissue_r_thresh=0.2,
        r2_thresh=0.75,
        k_clique=3,
    )
    assert res1.get("error") is None
    assert res2.get("error") is None
    assert not np.array_equal(res1["sampled_yx"], res2["sampled_yx"])


def test_positive_tissue_filter(synthetic_stack):
    """Retained pixels must all have positive Pearson correlation with tissue mean >= threshold."""
    data = synthetic_stack
    thresh = 0.30
    res = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=50,
        seed=0,
        tissue_r_thresh=thresh,
        r2_thresh=0.75,
        k_clique=3,
    )
    assert res.get("error") is None
    assert res["n_retained"] >= 3
    for r_val in res["tissue_r"]:
        assert r_val >= thresh
        assert r_val > 0


def test_tissue_filter_blocks_all(synthetic_stack):
    """Excessively high tissue correlation threshold should return a clean error without crashing."""
    data = synthetic_stack
    res = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=40,
        seed=0,
        tissue_r_thresh=0.9999,
        r2_thresh=0.7,
        k_clique=6,
    )
    assert res.get("error") is not None
    assert "passed the positive tissue-correlation threshold" in res["error"]


def test_r2_edge_two_tailed():
    """Two-tailed R² rule should include strongly anti-correlated signals (r ≈ -0.95 -> R² ≈ 0.90)."""
    T = 60
    t = np.linspace(0, 4 * np.pi, T)
    s1 = np.sin(t)
    s2 = -np.sin(t)  # anti-correlated

    # Stack with 12 nodes in s1 phase, 6 in s2 phase (so tissue mean is non-zero)
    stack = np.zeros((T, 18, 1), dtype=np.float32)
    for i in range(12):
        stack[:, i, 0] = 2.0 * s1 + np.random.normal(0, 0.05, T) + 20.0
    for i in range(12, 18):
        stack[:, i, 0] = 2.0 * s2 + np.random.normal(0, 0.05, T) + 20.0

    mask0 = np.ones((18, 1), dtype=np.int32)
    res = cc.analyze_calcium_network(
        reg_or_ca=stack,
        mask0=mask0,
        n_samples=18,
        seed=0,
        tissue_r_thresh=-1.0,  # allow all
        r2_thresh=0.70,
        positive_edges_only=False,
        k_clique=3,
    )
    assert res.get("error") is None
    assert len(res["edges_df"]) > 0
    assert (res["edges_df"]["pearson_r"] < -0.8).any()


def test_positive_edges_only():
    """positive_edges_only=True should exclude strongly anti-correlated pairs."""
    T = 60
    t = np.linspace(0, 4 * np.pi, T)
    s1 = np.sin(t)
    s2 = -np.sin(t)

    stack = np.zeros((T, 18, 1), dtype=np.float32)
    for i in range(12):
        stack[:, i, 0] = 2.0 * s1 + np.random.normal(0, 0.05, T) + 20.0
    for i in range(12, 18):
        stack[:, i, 0] = 2.0 * s2 + np.random.normal(0, 0.05, T) + 20.0

    mask0 = np.ones((18, 1), dtype=np.int32)
    res = cc.analyze_calcium_network(
        reg_or_ca=stack,
        mask0=mask0,
        n_samples=18,
        seed=0,
        tissue_r_thresh=-1.0,
        r2_thresh=0.70,
        positive_edges_only=True,
        k_clique=3,
    )
    assert res.get("error") is None
    assert len(res["edges_df"]) > 0
    assert (res["edges_df"]["pearson_r"] > 0).all()


def test_roi_restriction(synthetic_stack):
    """When roi_box is specified, sampled pixels must be strictly inside the ROI bounding box."""
    data = synthetic_stack
    roi_box = (5, 5, 15, 35)  # Left column cells (Cell 1 and Cell 3)
    res = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        roi_box=roi_box,
        n_samples=30,
        seed=0,
        tissue_r_thresh=0.1,
        r2_thresh=0.75,
        k_clique=3,
    )
    assert res.get("error") is None
    for y, x in res["sampled_yx"]:
        assert 5 <= x < 15
        assert 5 <= y < 35


def test_roi_independent_of_all(synthetic_stack):
    """ROI analysis should produce an independent result from whole-field analysis."""
    data = synthetic_stack
    res_all = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        roi_box=None,
        n_samples=40,
        seed=0,
        tissue_r_thresh=0.2,
        r2_thresh=0.75,
        k_clique=3,
    )
    res_roi = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        roi_box=(5, 5, 15, 35),
        n_samples=40,
        seed=0,
        tissue_r_thresh=0.2,
        r2_thresh=0.75,
        k_clique=3,
    )
    assert res_all.get("error") is None
    assert res_roi.get("error") is None
    assert res_roi["n_valid_pixels"] < res_all["n_valid_pixels"]


def test_no_community_graph(synthetic_stack):
    """When no k-cliques exist, function should return n_communities=0 cleanly without error."""
    data = synthetic_stack
    res = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=30,
        seed=0,
        tissue_r_thresh=0.1,
        r2_thresh=0.999,  # almost no edges
        k_clique=8,       # impossible clique
    )
    assert res.get("error") is None
    assert res["n_communities"] == 0
    assert res["n_assigned"] == 0
    assert res["n_unassigned"] == res["n_nodes"]


def test_overlapping_communities():
    """Synthetic network with two 3-cliques sharing 2 nodes should correctly identify overlapping nodes."""
    T = 50
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, T)
    base1 = np.sin(t)
    base2 = np.cos(t)

    # 6 nodes:
    # 0,1,2 driven by base1 (clique 1)
    # 1,2,3 driven by base2 (clique 2)
    # Nodes 1 and 2 participate in both cliques
    stack = np.zeros((T, 6, 1), dtype=np.float32)
    stack[:, 0, 0] = base1 + rng.normal(0, 0.05, T) + 10.0
    stack[:, 1, 0] = 0.5 * (base1 + base2) + rng.normal(0, 0.05, T) + 10.0
    stack[:, 2, 0] = 0.5 * (base1 + base2) + rng.normal(0, 0.05, T) + 10.0
    stack[:, 3, 0] = base2 + rng.normal(0, 0.05, T) + 10.0
    stack[:, 4, 0] = rng.normal(0, 1.0, T) + 10.0
    stack[:, 5, 0] = rng.normal(0, 1.0, T) + 10.0

    mask0 = np.ones((6, 1), dtype=np.int32)
    res = cc.analyze_calcium_network(
        reg_or_ca=stack,
        mask0=mask0,
        n_samples=6,
        seed=0,
        tissue_r_thresh=-1.0,
        r2_thresh=0.45,
        k_clique=3,
    )
    assert res.get("error") is None
    assert res["n_nodes"] == 6
    assert isinstance(res["n_overlapping"], int)
    assert res["n_assigned"] + res["n_unassigned"] == res["n_nodes"]


def test_dense_graph_blocked():
    """Multi-factor safety preflight should block k-clique on overly dense graph."""
    T = 30
    N = 80
    # 80 identical nodes -> complete graph -> density = 1.0
    stack = np.ones((T, N, 1), dtype=np.float32)
    for t_idx in range(T):
        stack[t_idx, :, 0] = np.sin(t_idx) + 10.0

    mask0 = np.ones((N, 1), dtype=np.int32)
    res = cc.analyze_calcium_network(
        reg_or_ca=stack,
        mask0=mask0,
        n_samples=N,
        seed=0,
        tissue_r_thresh=0.0,
        r2_thresh=0.1,
        k_clique=6,
    )
    assert res.get("error") is not None
    assert res.get("safety") is True


def test_fewer_than_k_nodes(synthetic_stack):
    """If fewer than k nodes exist, return an informative error dict."""
    data = synthetic_stack
    res = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        roi_box=(5, 5, 7, 7),  # 2x2 = 4 pixels
        n_samples=4,
        k_clique=6,
    )
    assert res.get("error") is not None
    assert "Fewer than 6 valid pixels" in res["error"]


def test_no_edges_after_filter(synthetic_stack):
    """High R² threshold yields 0 edges and an empty edges DataFrame."""
    data = synthetic_stack
    res = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=30,
        seed=0,
        tissue_r_thresh=0.1,
        r2_thresh=0.9999,
        k_clique=3,
    )
    assert res.get("error") is None
    assert res["n_edges"] == 0
    assert len(res["edges_df"]) == 0
    assert "pearson_r" in res["edges_df"].columns


def test_constant_trace_discarded():
    """Pixels with constant traces (zero std) should be dropped gracefully."""
    T = 30
    stack = np.zeros((T, 10, 1), dtype=np.float32)
    for i in range(5):
        stack[:, i, 0] = np.sin(np.linspace(0, 4 * np.pi, T)) + 10.0
    for i in range(5, 10):
        stack[:, i, 0] = 5.0  # constant

    mask0 = np.ones((10, 1), dtype=np.int32)
    res = cc.analyze_calcium_network(
        reg_or_ca=stack,
        mask0=mask0,
        n_samples=10,
        seed=0,
        tissue_r_thresh=0.1,
        r2_thresh=0.5,
        k_clique=3,
    )
    assert res.get("error") is None
    assert res["n_retained"] == 5


def test_nodes_csv_schema(synthetic_stack):
    """nodes_df must match expected schema for export."""
    data = synthetic_stack
    res = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=40,
        seed=0,
        tissue_r_thresh=0.2,
        r2_thresh=0.75,
        k_clique=3,
    )
    assert res.get("error") is None
    df = res["nodes_df"]
    expected_cols = ["node_id", "y", "x", "tissue_r", "degree", "community_ids", "primary_community", "overlap_count"]
    for col in expected_cols:
        assert col in df.columns
    assert len(df) == res["n_nodes"]


def test_edges_csv_schema(synthetic_stack):
    """edges_df must match expected schema for export."""
    data = synthetic_stack
    res = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=40,
        seed=0,
        tissue_r_thresh=0.2,
        r2_thresh=0.75,
        k_clique=3,
    )
    assert res.get("error") is None
    df = res["edges_df"]
    expected_cols = ["node_i", "node_j", "source_y", "source_x", "target_y", "target_x", "pearson_r", "r_squared"]
    for col in expected_cols:
        assert col in df.columns
    assert len(df) == res["n_edges"]


def test_summary_csv_schema(synthetic_stack):
    """summary_df must match expected schema for export."""
    data = synthetic_stack
    res = cc.analyze_calcium_network(
        reg_or_ca=data["ca_dict"],
        mask0=data["mask0"],
        frames=data["frames"],
        n_samples=40,
        seed=0,
        tissue_r_thresh=0.2,
        r2_thresh=0.75,
        k_clique=3,
        dataset_name="demo_dataset",
    )
    assert res.get("error") is None
    df = res["summary_df"]
    assert "parameter" in df.columns
    assert "value" in df.columns
    params = set(df["parameter"].values)
    assert "Sampled pixels" in params
    assert "Retained nodes after tissue filter" in params
    assert "Network edges" in params
    assert "Graph density" in params
    assert "k-clique size" in params
    assert "Number of communities" in params
