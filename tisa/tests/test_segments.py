import numpy as np

from tisa.segments import Segment, segmentize


def test_segmentize_monotone():
    x = np.arange(10)
    segments = segmentize(x)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.start == 0 and seg.end == 9
    assert seg.direction == 1


def test_segmentize_with_turns_and_plateau():
    x = np.array([0, 1, 1, 2, 1, 0, 0, -1])
    segments = segmentize(x)
    # Expect upward then downward segments
    assert len(segments) == 2
    assert segments[0].direction == 1
    assert segments[1].direction == -1


def test_segmentize_respects_thresholds():
    x = np.array([0, 0.01, 0.02, 0.03, 0.01])
    segments = segmentize(x, tau_v=0.05)
    assert len(segments) == 1
