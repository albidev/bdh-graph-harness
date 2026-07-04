"""Tests for adaptive threshold (Phase 3.3)."""
import pytest
import harness


def test_adaptive_threshold_basic():
    """Adaptive threshold returns a value >= floor."""
    scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold = harness.compute_adaptive_threshold(scores, floor=0.15)
    assert threshold >= 0.15


def test_adaptive_threshold_few_scores_returns_floor():
    """With <3 scores, returns floor (statistics unreliable)."""
    scores = [0.5, 0.6]
    threshold = harness.compute_adaptive_threshold(scores, floor=0.15)
    assert threshold == 0.15


def test_adaptive_threshold_empty_returns_floor():
    """Empty scores returns floor."""
    threshold = harness.compute_adaptive_threshold([], floor=0.15)
    assert threshold == 0.15


def test_adaptive_threshold_high_scores():
    """When all scores are high, threshold should be high."""
    scores = [0.8, 0.85, 0.9, 0.95, 1.0, 0.88, 0.92, 0.87]
    threshold = harness.compute_adaptive_threshold(scores, floor=0.15)
    assert threshold > 0.5  # should be well above floor


def test_adaptive_threshold_low_scores():
    """When scores are low, threshold should still be at least floor."""
    scores = [0.1, 0.12, 0.15, 0.18, 0.2, 0.13, 0.16]
    threshold = harness.compute_adaptive_threshold(scores, floor=0.15)
    assert threshold >= 0.15


def test_adaptive_threshold_uses_max_of_q75_and_median_half_std():
    """Threshold = max(Q75, median+0.5std, floor)."""
    # Create scores where median+0.5std > Q75
    scores = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 1.0]
    threshold = harness.compute_adaptive_threshold(scores, floor=0.0)
    # Q75 is 0.1 (most scores), but median+0.5std should be higher due to outliers
    import statistics
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    expected = max(0.1, median + 0.5 * stdev, 0.0)
    assert abs(threshold - expected) < 0.001


def test_adaptive_threshold_never_below_floor():
    """Threshold never goes below floor even with very low scores."""
    scores = [0.01, 0.02, 0.03, 0.04, 0.05]
    threshold = harness.compute_adaptive_threshold(scores, floor=0.15)
    assert threshold >= 0.15


def test_adaptive_threshold_custom_floor():
    """Custom floor is respected."""
    scores = [0.1, 0.15, 0.2, 0.25, 0.3]
    threshold = harness.compute_adaptive_threshold(scores, floor=0.3)
    assert threshold >= 0.3