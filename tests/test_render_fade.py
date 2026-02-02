"""
Tests for video rendering fade: _alpha_fade and fade behavior.

- Alpha should be 1 only when event time exactly matches frame center.
- Alpha should fade linearly to 0 at both edges of the time window.
- Output brightness should reflect alpha (no normalizing so single events don't stay full brightness).
"""

import numpy as np
import pytest

from radiotrap.render import _alpha_fade


# ============== _alpha_fade ==============

def test_alpha_fade_at_center_is_one():
    """Alpha is 1 only when event time exactly equals center."""
    half = 100.0
    t_center = 1000.0
    # Single event at center
    alpha = _alpha_fade(np.array([t_center]), t_center, half)
    assert alpha.shape == (1,)
    assert alpha[0] == 1.0


def test_alpha_fade_at_edges_is_zero():
    """Alpha is 0 at window edges (center ± half_window)."""
    half = 100.0
    t_center = 1000.0
    # Events at edges
    alpha_lo = _alpha_fade(np.array([t_center - half]), t_center, half)
    alpha_hi = _alpha_fade(np.array([t_center + half]), t_center, half)
    assert alpha_lo[0] == 0.0
    assert alpha_hi[0] == 0.0


def test_alpha_fade_beyond_window_is_zero():
    """Alpha is 0 outside the window."""
    half = 100.0
    t_center = 1000.0
    alpha = _alpha_fade(np.array([t_center - half - 1]), t_center, half)
    assert alpha[0] == 0.0
    alpha = _alpha_fade(np.array([t_center + half + 1]), t_center, half)
    assert alpha[0] == 0.0


def test_alpha_fade_linear():
    """Alpha decreases linearly from center to edge."""
    half = 100.0
    t_center = 1000.0
    # At 1/4 and 3/4 of the way to the edge, alpha should be 0.75 and 0.25 (linear)
    t_quarter = t_center - half * 0.25  # 1/4 from center to left edge
    t_three_quarter = t_center - half * 0.75  # 3/4 from center to left edge
    alpha_quarter = _alpha_fade(np.array([t_quarter]), t_center, half)[0]
    alpha_three_quarter = _alpha_fade(np.array([t_three_quarter]), t_center, half)[0]
    assert abs(alpha_quarter - 0.75) < 1e-9
    assert abs(alpha_three_quarter - 0.25) < 1e-9


def test_alpha_fade_vectorized():
    """_alpha_fade works on arrays and returns correct values."""
    half = 50.0
    t_center = 500.0
    t_events = np.array([t_center, t_center - 25, t_center + 25, t_center - 50, t_center + 60])
    alpha = _alpha_fade(t_events, t_center, half)
    assert alpha.shape == (5,)
    assert alpha[0] == 1.0   # at center
    assert alpha[1] == 0.5   # halfway to left edge
    assert alpha[2] == 0.5   # halfway to right edge
    assert alpha[3] == 0.0   # at left edge
    assert alpha[4] == 0.0   # beyond right edge


def test_alpha_fade_zero_half_window_returns_one():
    """When half_window is 0 or negative, alpha is 1 (no fade)."""
    alpha = _alpha_fade(np.array([100.0]), 100.0, 0.0)
    assert alpha[0] == 1.0
    alpha = _alpha_fade(np.array([100.0]), 99.0, -1.0)
    assert alpha[0] == 1.0


# ============== Fade behavior: output = sum(color * alpha), no divide ==============

def test_single_event_faded_by_alpha():
    """
    A single event with alpha < 1 should contribute that fraction to the pixel,
    not full brightness. So we must NOT normalize by alpha (output = sum(color*alpha)).
    This test documents the expected behavior: we accumulate color*alpha and clip.
    """
    # Simulate: one event at (10, 10) with alpha 0.5, color [200, 100, 50]
    acc = np.zeros((256, 256, 3), dtype=np.float64)
    color = np.array([200.0, 100.0, 50.0])
    alpha = 0.5
    acc[10, 10] += color * alpha
    # Correct: output = color * alpha (no division)
    out = np.clip(acc, 0, 255).astype(np.uint8)
    assert out[10, 10, 0] == 100  # 200 * 0.5
    assert out[10, 10, 1] == 50
    assert out[10, 10, 2] == 25
    # Wrong would be: out = acc / alpha -> full [200, 100, 50]


def test_two_events_same_pixel_blend_by_alpha():
    """
    Two events at same pixel with alphas 0.5 and 0.3: output = 0.5*C1 + 0.3*C2 (no normalize).
    """
    acc = np.zeros((256, 256, 3), dtype=np.float64)
    acc[5, 5] += np.array([200.0, 0, 0]) * 0.5
    acc[5, 5] += np.array([0, 100.0, 0]) * 0.3
    out = np.clip(acc, 0, 255).astype(np.uint8)
    assert out[5, 5, 0] == 100  # 200*0.5
    assert out[5, 5, 1] == 30   # 100*0.3
    assert out[5, 5, 2] == 0
