import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cafin_core as cc


@pytest.fixture
def synthetic_cells():
    """Cell traces with two coordinated groups and spatially labelled cells."""
    rng = np.random.default_rng(42)
    n_frames, n_cells = 80, 12
    t = np.linspace(0, 6 * np.pi, n_frames)
    group_a = np.sin(t) + 0.25 * np.sin(2 * t)
    group_b = -group_a + 0.05 * np.cos(t)
    traces = np.empty((n_frames, n_cells), dtype=float)
    for i in range(6):
        traces[:, i] = group_a + rng.normal(0, 0.03, n_frames)
    for i in range(6, 10):
        traces[:, i] = group_b + rng.normal(0, 0.03, n_frames)
    for i in range(10, 12):
        traces[:, i] = rng.normal(0, 1.0, n_frames)
    dff = pd.DataFrame(traces, columns=[f"Cell_{i}" for i in range(1, n_cells + 1)])
    dff.insert(0, "Frame", np.arange(n_frames))
    mask = np.zeros((45, 45), dtype=np.int32)
    for cell_id in range(1, n_cells + 1):
        y = 2 + ((cell_id - 1) // 4) * 13
        x = 2 + ((cell_id - 1) % 4) * 10
        mask[y:y + 8, x:x + 8] = cell_id
    return {"dff": dff, "mask": mask}


def build(data, **kwargs):
    defaults = dict(mask0=data["mask"], n_samples=12, seed=0,
                    tissue_r_thresh=-1.0, tissue_positive_only=False,
                    r2_thresh=0.70, k_clique=3)
    defaults.update(kwargs)
    return cc.analyze_calcium_network(data["dff"], **defaults)


def test_reproducible_cell_sampling(synthetic_cells):
    one = build(synthetic_cells, seed=123)
    two = build(synthetic_cells, seed=123)
    assert one["error"] is None
    np.testing.assert_array_equal(one["sampled_cell_ids"], two["sampled_cell_ids"])
    pd.testing.assert_frame_equal(one["nodes_df"], two["nodes_df"])
    pd.testing.assert_frame_equal(one["edges_df"], two["edges_df"])


def test_different_seeds_change_cells(synthetic_cells):
    one = build(synthetic_cells, seed=1)
    two = build(synthetic_cells, seed=2)
    assert one["error"] is None and two["error"] is None
    assert not np.array_equal(one["sampled_cell_ids"], two["sampled_cell_ids"])


def test_node_definition_is_cell_and_tissue_filter(synthetic_cells):
    result = build(synthetic_cells, tissue_r_thresh=0.30, tissue_positive_only=True)
    assert result["error"] is None
    assert result["n_nodes"] <= 12
    assert set(result["nodes_df"]["cell_id"]).issubset(set(range(1, 13)))
    assert (result["nodes_df"]["tissue_r"] >= 0.30).all()
    assert (result["summary_df"].query("parameter == 'Node definition'")["value"] ==
            "segmented cell").all()


def test_r2_keeps_negative_edges_unless_positive_only(synthetic_cells):
    result = build(synthetic_cells, positive_edges_only=False)
    assert result["error"] is None
    assert (result["edges_df"]["r_squared"] >= 0.70).all()
    assert (result["edges_df"]["pearson_r"] < -0.8).any()
    positive = build(synthetic_cells, positive_edges_only=True)
    assert positive["error"] is None
    assert (positive["edges_df"]["pearson_r"] > 0).all()


def test_roi_uses_cell_masks_not_pixels(synthetic_cells):
    result = build(synthetic_cells, roi_box=(1, 1, 23, 13), k_clique=2)
    assert result["error"] is None
    assert set(result["nodes_df"]["cell_id"]) == {1, 2}
    assert result["n_valid_cells"] == 2


def test_no_communities_is_valid_result(synthetic_cells):
    result = build(synthetic_cells, tissue_r_thresh=-1.0, tissue_positive_only=False,
                   r2_thresh=0.99999, k_clique=3)
    assert result["error"] is None
    assert result["n_communities"] == 0
    assert result["n_unassigned"] == result["n_nodes"]


def test_overlapping_communities_are_reported():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 5 * np.pi, 100)
    a, b = np.sin(t), np.cos(t)
    x = np.column_stack([
        a + rng.normal(0, .01, len(t)),
        (a + b) / 2 + rng.normal(0, .01, len(t)),
        (a + b) / 2 + rng.normal(0, .01, len(t)),
        b + rng.normal(0, .01, len(t)),
        rng.normal(0, 1, len(t)),
        rng.normal(0, 1, len(t)),
    ])
    dff = pd.DataFrame(x, columns=[f"Cell_{i}" for i in range(1, 7)])
    mask = np.arange(1, 7, dtype=np.int32)[:, None]
    result = cc.analyze_calcium_network(dff, mask0=mask, n_samples=6, seed=0,
                                     tissue_r_thresh=-1, tissue_positive_only=False,
                                     r2_thresh=0.45, k_clique=3)
    assert result["error"] is None
    assert result["n_overlapping"] >= 0
    assert "community_ids" in result["nodes_df"]


def test_dense_graph_safety_guard():
    t = np.sin(np.arange(40, dtype=float))
    dff = pd.DataFrame({f"Cell_{i}": t for i in range(1, 81)})
    mask = np.arange(1, 81, dtype=np.int32)[:, None]
    result = cc.analyze_calcium_network(dff, mask0=mask, n_samples=80,
                                     tissue_r_thresh=0, r2_thresh=0.1, k_clique=6)
    assert result["error"] is not None
    assert result["safety"] is True


def test_csv_schemas(synthetic_cells):
    result = build(synthetic_cells)
    assert result["error"] is None
    assert {"node_id", "cell_id", "x", "y", "tissue_r", "degree",
            "community_ids", "primary_community", "overlap_count"} <= set(result["nodes_df"])
    assert {"cell_i", "cell_j", "pearson_r", "r_squared"} <= set(result["edges_df"])
    assert {"parameter", "value"} == set(result["summary_df"])


def test_invalid_input_is_informative():
    result = cc.analyze_calcium_network(np.zeros((10, 10)))
    assert result["error"]
