"""Checks that the six-trial exploratory audit keeps cell and trial units distinct."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "REPRODUCE" / "six_trial_analysis.py"
spec = spec_from_file_location("six_trial_analysis", SCRIPT)
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_each_cell_gets_its_own_ten_lowest_frames():
    first = np.arange(2.0, 32.0)
    second = first[::-1] + 5.0
    dff, f0 = module.normalize_each_cell(np.stack([first, second], axis=1))
    assert f0.tolist() == pytest.approx([6.5, 11.5])
    assert dff[0, 0] == pytest.approx((2.0 - 6.5) / 6.5)
    assert dff[0, 1] == pytest.approx((36.0 - 11.5) / 11.5)


def test_f0_floor_and_active_cells_are_recorded_separately():
    raw = np.column_stack([np.full(30, 0.25), np.tile([2.0, 7.0], 15)])
    field, cells = module.summarize_recording(raw, ["Cell_1", "Cell_2"],
                                              trial=1, condition="after", window="first_30_frames")
    assert cells.at_f0_floor.tolist() == [True, False]
    assert cells.active.tolist() == [False, True]
    assert cells.f0i.tolist() == pytest.approx([1.0, 2.0])
    assert field["active_cell_fraction"] == pytest.approx(0.5)


def test_six_paired_trials_get_trial_level_exact_wilcoxon():
    rows = []
    for trial in range(1, 7):
        for condition, value in (("before", float(trial)),
                                 ("after", float(trial + 1))):
            row = {"trial": trial, "condition": condition}
            row.update({metric: value for metric in module.METRICS})
            rows.append(row)
    result = module.paired_tests(pd.DataFrame(rows))
    assert len(result) == len(module.METRICS)
    assert result.n_pairs.eq(6).all()
    assert result.p_two_sided_exact.eq(0.03125).all()
    assert result.q_bh_five_metrics.eq(0.03125).all()


def test_missing_pair_is_rejected():
    rows = []
    for trial in range(1, 7):
        for condition in ("before", "after"):
            row = {"trial": trial, "condition": condition}
            row.update({metric: float(trial) for metric in module.METRICS})
            rows.append(row)
    with pytest.raises(ValueError, match="six paired"):
        module.paired_tests(pd.DataFrame(rows[:-1]))
