from speedread.extractor import segment_by_two_refs


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
