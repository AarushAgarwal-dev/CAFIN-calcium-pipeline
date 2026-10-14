"""Tracking checks for short movies, sparse sampling, and gap ambiguity."""

import numpy as np
import pytest

from cafin_track import TrackParams, close_gaps, match_frame, run


def square(label, x0, y0=10, side=10, shape=(50, 70)):
    mask = np.zeros(shape, dtype=np.uint16)
    mask[y0:y0 + side, x0:x0 + side] = label
    return mask


def test_two_frame_overlap_keeps_an_individual_id():
    masks = np.stack([square(7, 10), square(3, 12)])
    assert match_frame(masks[1], masks[0], TrackParams()) == {3: 7}
    tracked, rows, stats = run(masks, TrackParams())
    assert int(tracked[0].max()) == int(tracked[1].max()) == 1
    assert stats["full_coverage"] == 1
    assert rows[0][3] == 2


def test_sparse_source_frames_do_not_count_as_a_short_gap():
    first = square(1, 5)
    second = square(7, 6)
    third = square(9, 25)
    fourth = square(3, 26)
    masks = np.stack([first, second, third, fourth])
    assert match_frame(third, second, TrackParams()) == {}

    dense, _, dense_stats = run(masks, TrackParams(), frame_idx=[0, 1, 2, 3])
    assert dense[0].max() == dense[3].max() == 1
    assert dense_stats["n_gap_links"] == 1

    with pytest.raises(ValueError, match="consecutive source frames"):
        run(masks, TrackParams(), frame_idx=[0, 1, 10, 11])

    # The gap closer also uses source indices when called independently.
    prelinked = [square(1, 5), square(1, 6), square(2, 25), square(2, 26)]
    sparse, links = close_gaps(prelinked, TrackParams(),
                               frame_idx=[0, 1, 10, 11], return_links=True)
    assert int(sparse[3].max()) == 2
    assert links == []


def test_competing_gap_fragments_cannot_overwrite_the_closest_match():
    first = square(1, 10) + square(2, 30)
    empty = np.zeros_like(first)
    last = square(3, 16)
    linked, links = close_gaps([first, first.copy(), empty, last, last.copy()],
                              TrackParams(), frame_idx=[0, 1, 2, 3, 4],
                              return_links=True)
    assert int(linked[4].max()) == 1
    assert [(link["from_id"], link["to_id"]) for link in links] == [(1, 3)]


def test_two_frames_cannot_auto_confirm_a_proximity_only_identity():
    masks = np.stack([square(1, 5), square(9, 25)])
    tracked, _, stats = run(masks, TrackParams())
    assert int(tracked[0].max()) == 1
    assert int(tracked[1].max()) == 2
    assert stats["n_gap_links"] == 0


def test_large_deformation_is_not_mistaken_for_a_verified_link():
    masks = np.stack([square(1, 10, side=20), square(2, 17, side=6)])
    assert match_frame(masks[1], masks[0], TrackParams()) == {}
    tracked, _, stats = run(masks, TrackParams())
    assert tracked[0].max() == 1
    assert tracked[1].max() == 2
    assert stats["n_gap_links"] == 0


def test_source_frame_indices_must_be_complete_and_ordered():
    masks = np.stack([square(1, 10), square(1, 11)])
    with pytest.raises(ValueError, match="increasing source index"):
        run(masks, TrackParams(), frame_idx=[0, 0])
    with pytest.raises(ValueError, match="one increasing source index"):
        run(masks, TrackParams(), frame_idx=[0])


def test_demo_movie_motion_keeps_all_25_known_cell_identities():
    """Use the 12-frame motion in demo_data.py, with fresh labels per frame."""
    yy, xx = np.mgrid[:256, :256]
    centers = [(45 + 42 * col + (row % 2) * 20, 45 + 42 * row)
               for row in range(5) for col in range(5)]
    masks = []
    for t in range(12):
        dx = int(round(3 * np.sin(t / 3)))
        dy = int(round(2 * np.cos(t / 4)))
        frame = np.zeros((256, 256), dtype=np.uint16)
        for true_id, (cx, cy) in enumerate(centers, start=1):
            fresh_label = (true_id + 7 * t - 1) % 25 + 1
            frame[(xx - cx - dx) ** 2 + (yy - cy - dy) ** 2 <= 14 ** 2] = fresh_label
        masks.append(frame)

    tracked, _, stats = run(np.stack(masks), TrackParams())
    assert stats["n_tracks"] == stats["full_coverage"] == 25
    assert stats["n_gap_links"] == 0
    for true_id, (cx, cy) in enumerate(centers, start=1):
        known_track = int(tracked[0, cy + 2, cx])
        for t in range(12):
            dx = int(round(3 * np.sin(t / 3)))
            dy = int(round(2 * np.cos(t / 4)))
            assert int(tracked[t, cy + dy, cx + dx]) == known_track
