"""Topology primitives: wrap, winding number, angle-around-center, gate events.

Pure numpy. This is the diagnostic backbone (§1). The integer winding number
is *monitored*, never optimized (warning W3).
"""
import math
import numpy as np

PI = math.pi


def wrap(d):
    """Wrap angular differences to (-pi, pi]:  wrap(d) = ((d + pi) mod 2pi) - pi."""
    return np.mod(np.asarray(d) + PI, 2 * PI) - PI


def winding(phases):
    """Winding number of an ordered *closed* sequence of phases phi_0..phi_{n-1}.

        W = (1/2pi) * sum_i wrap(phi_{i+1} - phi_i),  indices cyclic.

    Exact integer when consecutive gaps < pi. Returns a float; round() to get
    the integer class.
    """
    phi = np.asarray(phases, dtype=float)
    d = wrap(np.diff(phi))
    d = np.append(d, wrap(phi[0] - phi[-1]))  # close the loop
    return float(np.sum(d) / (2 * PI))


def round_winding(phases):
    """Integer winding class = round(winding(phases))."""
    return int(round(winding(phases)))


def ang_c(p, c=(0.0, 0.0)):
    """Angle of point(s) p around center c:  atan2(p_y - c_y, p_x - c_x).

    p: (..., 2) array. Returns (...,) angles in (-pi, pi].
    """
    p = np.asarray(p, dtype=float)
    c = np.asarray(c, dtype=float)
    dy = p[..., 1] - c[1]
    dx = p[..., 0] - c[0]
    return np.arctan2(dy, dx)


def winding_of_points(p, c=(0.0, 0.0)):
    """Oracle winding of a closed 2D loop `p` (shape (T,2)) around center c."""
    return winding(ang_c(p, c))


def gate_events(min_norm_series, thresh):
    """Indices where the min-norm series dips below `thresh` (gate touched).

    min_norm_series: 1D array of min_x ||f(x)|| logged per eval step.
    Returns boolean mask (per step) and the list of flagged step-indices.
    """
    s = np.asarray(min_norm_series, dtype=float)
    mask = s < thresh
    return mask, np.nonzero(mask)[0].tolist()


def conservation_violations(W_series, min_norm_series, thresh, w_jump=0.5):
    """Steps where the integer winding class changed *without* a gate event.

    A violation is a step where round(W) differs from the previous step's
    round(W) but min||f|| never dipped below `thresh` across the transition.
    Under the conservation law (§1) this set should be empty.
    """
    W = np.asarray(W_series, dtype=float)
    r = np.asarray(min_norm_series, dtype=float)
    cls = np.round(W).astype(int)
    viol = []
    for i in range(1, len(cls)):
        if cls[i] != cls[i - 1]:
            gated = (r[i] < thresh) or (r[i - 1] < thresh)
            if not gated:
                viol.append(i)
    return viol
