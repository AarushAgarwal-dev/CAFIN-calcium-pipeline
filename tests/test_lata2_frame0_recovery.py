"""Guard the one missing LATA2 baseline frame against mismatched sources."""

import json

import numpy as np
import pandas as pd
import pytest
import tifffile

from REPRODUCE import normalize_f0_per_cell as norm
from REPRODUCE import recover_lata2_frame0 as recovery


def _synthetic_source(tmp_path):
    root = tmp_path / "source"
    frame0_path = root / recovery.FRAME_ZERO_RELATIVE
    frame1_path = root / recovery.FRAME_ONE_RELATIVE
    mask_path = root / recovery.MASK_RELATIVE
    raw_path = root / recovery.RAW_RELATIVE
    for path in (frame0_path, frame1_path, mask_path, raw_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.zeros((120, 120), dtype=np.uint16)
    frame0 = np.full(mask.shape, 194, dtype=np.uint16)
    frame1 = np.full(mask.shape, 194, dtype=np.uint16)
    for cell_id, start, value0, value1 in (
        (1, 70, 10, 11), (2, 85, 20, 22), (3, 100, 30, 33)
    ):
        area = (slice(start, start + 5), slice(start, start + 5))
        mask[area] = cell_id
        frame0[area] += value0
        frame1[area] += value1
    tifffile.imwrite(frame0_path, frame0)
    tifffile.imwrite(frame1_path, frame1)
    tifffile.imwrite(mask_path, mask)
    pd.DataFrame(
        {
            "Frame": [0, 1],
            "Cell_1": [np.nan, 11],
            "Cell_2": [np.nan, 22],
            "Cell_3": [np.nan, 33],
        }
    ).to_csv(raw_path, index=False)
    return root, raw_path


def test_recovery_validates_frame1_and_normalizer_uses_only_missing_row(tmp_path, monkeypatch):
    root, raw_path = _synthetic_source(tmp_path)
    output = root / "REPRODUCE/source_repairs"
    repair_csv, provenance_json = recovery.recover(root, output_dir=output, expected_cells=3)
    assert pd.read_csv(repair_csv).iloc[0].tolist() == [0, 10, 20, 30]
    provenance = json.loads(provenance_json.read_text(encoding="utf-8"))
    assert provenance["background_by_frame"]["0"]["mean"] == 194
    assert provenance["frame1_validation"]["max_absolute_error"] < 1e-6

    monkeypatch.setattr(norm, "ROOT", root)
    raw = pd.read_csv(raw_path)
    updated, info = norm.apply_frame_zero_repair(
        raw, "LATA2TRIAL", "BEFOREDRUG", raw_path, repair_root=output
    )
    assert updated.iloc[0].tolist() == [0, 10, 20, 30]
    assert updated.iloc[1].tolist() == [1, 11, 22, 33]
    assert raw.iloc[0, 1:].isna().all()
    assert info["frame1_validation"]["n_cells_compared"] == 3
    assert pd.read_csv(raw_path).iloc[0, 1:].isna().all()
    unchanged, repair = norm.apply_frame_zero_repair(
        raw, "LATA1TRAIL", "BEFOREDRUG", raw_path, repair_root=output
    )
    assert unchanged is raw and repair is None


def test_recovery_rejects_a_mismatched_registered_frame(tmp_path):
    root, raw_path = _synthetic_source(tmp_path)
    raw = pd.read_csv(raw_path)
    raw.loc[1, "Cell_1"] = 12
    raw.to_csv(raw_path, index=False)
    with pytest.raises(ValueError, match="Frame 1 does not reproduce"):
        recovery.recover(root, output_dir=root / "repairs", expected_cells=3)
    assert not (root / "repairs" / recovery.FRAME_ZERO_NAME).exists()


def test_committed_repair_matches_original_pipeline_csv(tmp_path):
    source_path = norm.source_dir("LATA2TRIAL", "BEFOREDRUG") / "all_cells_raw.csv"
    raw = pd.read_csv(source_path)
    updated, info = norm.apply_frame_zero_repair(
        raw, "LATA2TRIAL", "BEFOREDRUG", source_path
    )
    assert raw.loc[0, raw.columns[1:]].isna().all()
    assert np.isfinite(updated.loc[0, updated.columns[1:]].to_numpy(float)).all()
    assert len(updated.columns) - 1 == 479
    assert info["frame1_validation"]["max_absolute_error"] < 1e-6
    output = norm.normalize_field("LATA2TRIAL", "BEFOREDRUG", output_root=tmp_path)
    normalized = pd.read_csv(output / "all_cells_normalized.csv")
    assert np.isfinite(normalized.loc[0, normalized.columns[1:]].to_numpy(float)).all()
    assert json.loads((output / "provenance.json").read_text())["source_repair"]["recovered_frame"] == 0
