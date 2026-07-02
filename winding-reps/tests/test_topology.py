import math

import numpy as np
import pytest

from winding.topology import (
    wrap, winding, round_winding, ang_c, winding_of_points,
    gate_events, conservation_violations,
)

PI = math.pi


# ---- wrap edge cases -------------------------------------------------------
def test_wrap_basic():
    # wrap(d) = ((d+pi) mod 2pi) - pi maps onto [-pi, pi); the boundary pi -> -pi
    assert abs(wrap(0.0)) < 1e-12
    assert abs(wrap(PI) - (-PI)) < 1e-9       # pi maps to -pi (boundary)
    assert abs(wrap(-PI) - (-PI)) < 1e-9
    assert abs(wrap(2 * PI)) < 1e-9
    assert abs(wrap(3 * PI) - (-PI)) < 1e-9
    assert abs(wrap(PI / 2) - PI / 2) < 1e-9


def test_wrap_vectorized():
    d = np.array([0.0, PI / 2, -PI / 2, 3 * PI])
    out = wrap(d)
    assert np.allclose(out, np.array([0.0, PI / 2, -PI / 2, -PI]), atol=1e-9)


# ---- winding on analytic circles, k = -2..2 --------------------------------
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
def test_winding_analytic_circle(k):
    n = 200
    t = np.arange(n) / n
    phi = 2 * PI * k * t                       # phase advances k full turns
    W = winding(phi)
    assert abs(W - k) < 1e-6
    assert round_winding(phi) == k


def test_winding_of_points_matches_circle():
    n = 128
    for k in (-2, -1, 1, 2):
        t = np.arange(n) / n
        ang = 2 * PI * k * t
        p = np.stack([1.5 * np.cos(ang), 1.5 * np.sin(ang)], axis=1)
        assert abs(winding_of_points(p) - k) < 1e-6


def test_winding_zero_for_non_encircling_loop():
    # a small loop that does not enclose the origin
    n = 100
    t = np.arange(n) / n
    cx, cy = 1.5, 0.0
    p = np.stack([cx + 0.2 * np.cos(2 * PI * t),
                  cy + 0.2 * np.sin(2 * PI * t)], axis=1)
    assert round_winding(ang_c(p)) == 0


def test_ang_c_around_center():
    p = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]])
    a = ang_c(p, c=(0.0, 0.0))
    assert np.allclose(a, np.array([0.0, PI / 2, PI]), atol=1e-9)


# ---- gate detection --------------------------------------------------------
def test_gate_events():
    series = np.array([0.5, 0.3, 0.01, 0.4, 0.005])
    mask, idx = gate_events(series, thresh=0.02)
    assert idx == [2, 4]
    assert mask.tolist() == [False, False, True, False, True]


def test_conservation_no_violation_when_gated():
    # winding class changes 1->0 but a gate event coincides -> not a violation
    W = np.array([1.0, 1.0, 0.0, 0.0])
    minr = np.array([0.4, 0.4, 0.01, 0.4])
    assert conservation_violations(W, minr, thresh=0.02) == []


def test_conservation_flags_ungated_jump():
    # class changes 1->0 with no gate event -> violation
    W = np.array([1.0, 1.0, 0.0, 0.0])
    minr = np.array([0.4, 0.4, 0.4, 0.4])
    assert conservation_violations(W, minr, thresh=0.02) == [2]
