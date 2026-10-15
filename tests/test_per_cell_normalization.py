"""Check the paper's F0i pass against the analysis core and source CSVs."""

import json

import numpy as np
import pandas as pd

import cafin_core as core
from REPRODUCE import normalize_f0_per_cell as norm


def test_per_cell_baselines_match_core_with_distinct_quiet_frames_and_floor():
    raw = pd.DataFrame(
        {
            "Frame": list(range(6)),
            "Cell_1": [0.1, 0.2, 0.3, 5, 6, 7],
            "Cell_2": [8, 7, 6, 0.4, 0.5, 0.6],
            "Cell_3": [np.nan, 5, 7, 2, 1, 4],
        }
    )
    actual, baselines = norm.normalize_traces(raw)
    expected, shared, expected_baselines = core.compute_dff0(
        raw, method="percell", floor=1.0, n_base=10, return_f0=True
    )

    pd.testing.assert_frame_equal(actual, expected)
    assert shared is None
    assert json.loads(baselines.loc[0, "baseline_row_indices"]) == [0, 1, 2]
    assert json.loads(baselines.loc[1, "baseline_row_indices"]) == [3, 4, 5]
    for row in baselines.itertuples():
        assert row.f0i == expected_baselines[row.cell][0]
        assert json.loads(row.baseline_row_indices) == expected_baselines[row.cell][1]


def test_source_normalization_preserves_input_and_roi_membership(tmp_path):
    source = norm.source_dir("LATA1TRAIL", "BEFOREDRUG")
    raw = pd.read_csv(source / "all_cells_raw.csv")
    old_normalized = pd.read_csv(source / "all_cells_normalized.csv")
    output = norm.normalize_field("LATA1TRAIL", "BEFOREDRUG", output_root=tmp_path)

    new_normalized = pd.read_csv(output / "all_cells_normalized.csv")
    baselines = pd.read_csv(output / "f0_per_cell.csv")
    expected, _, expected_f0 = core.compute_dff0(raw, method="percell", return_f0=True)
    pd.testing.assert_frame_equal(new_normalized, expected, check_exact=False, atol=1e-12)
    assert len(baselines) == len(expected_f0)
    assert baselines.loc[0, "f0i"] == expected_f0[baselines.loc[0, "cell"]][0]
    assert not np.allclose(
        new_normalized.filter(like="Cell_").to_numpy(),
        old_normalized.filter(like="Cell_").to_numpy(), equal_nan=True
    )

    roi = pd.read_csv(output / "roi_cells_normalized.csv")
    original_roi = pd.read_csv(source / "roi_cells_normalized.csv", nrows=0)
    assert list(roi.columns) == list(original_roi.columns)
    pd.testing.assert_frame_equal(roi, new_normalized[list(roi.columns)])
    assert (output / "centroids_0.csv").read_bytes() == (source / "centroids_0.csv").read_bytes()
    assert json.loads((output / "provenance.json").read_text())["normalization"] == "percell"
