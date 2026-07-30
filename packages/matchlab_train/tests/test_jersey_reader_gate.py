from matchlab_train.experiments.jersey_reader_gate import _even_stride, _legible_stats


def test_even_stride_keeps_first_and_spans_the_whole_list():
    paths = list(range(1000))
    sampled = _even_stride(paths, 100)
    assert len(sampled) == 100
    assert sampled[0] == 0
    assert sampled[-1] > 900  # spans toward the end, not just the first 100


def test_even_stride_returns_everything_when_under_budget():
    paths = list(range(10))
    assert _even_stride(paths, 100) == paths


def test_legible_stats_precision_and_coverage():
    counts = {"correct": 9, "wrong": 1, "abstained": 10, "illegible_gt": 5}
    stats = _legible_stats(counts)
    assert stats["legible_precision"] == 0.9
    assert stats["legible_coverage"] == 0.5


def test_legible_stats_handles_zero_denominator():
    counts = {"correct": 0, "wrong": 0, "abstained": 0, "illegible_gt": 0}
    stats = _legible_stats(counts)
    assert stats["legible_precision"] == 0.0
    assert stats["legible_coverage"] == 0.0
