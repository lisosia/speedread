from speedread.extractor import (
    _expand_segment_starts,
    _select_analysis_indices,
    segment_by_two_refs,
)


def test_segment_by_two_refs_three_frames_boundary_in_middle() -> None:
    def sim(i: int, j: int) -> float:
        if i == j:
            return 1.0
        pair = {i, j}
        if pair == {0, 1}:
            return 0.6
        if pair == {1, 2}:
            return 0.6
        return 0.0

    boundaries = segment_by_two_refs(sim, n=3, thr=0.1, n_trans_min=1, n_trans_max=3)
    assert boundaries == [1]


def test_expand_segment_starts_keeps_boundaries() -> None:
    boundaries = [0, 1, 2, 3]
    segment_starts = _expand_segment_starts(boundaries, total=5, n_trans_max=2)
    assert segment_starts == [0, 1, 2, 3]


def test_segment_by_two_refs_adjacent_changes() -> None:
    def sim(i: int, j: int) -> float:
        return 1.0 if i == j else 0.0

    boundaries = segment_by_two_refs(sim, n=5, thr=0.2, n_trans_min=1, n_trans_max=10)
    assert boundaries == [1, 2, 3, 4]


def test_select_analysis_indices_single_frame() -> None:
    indices = _select_analysis_indices([], total=1, n_trans_max=5)
    assert indices == [0]
