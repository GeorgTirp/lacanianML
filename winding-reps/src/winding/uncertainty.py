"""Epistemic-uncertainty estimation: the charge gets an empirical address (§3).

Train an ensemble of M small point-track classifiers (different seeds +
bootstrap resamples). The disagreement map over the generative square peaks
both *outside* the annulus (exterior OOD) and *inside the hole* (interior
singularity). We restrict to the convex hull of the training data and take the
disagreement-weighted centroid of the high-disagreement region as the estimated
center c_hat -- an empirical address for the epistemic singularity.
"""
import numpy as np
import torch
import torch.nn as nn

from .config import CFG
from .models import PointMLP
from .data import Embedding


# --------------------------------------------------------------------------- #
#  Lightweight 2D convex hull (Andrew's monotone chain) -- no scipy dependency #
# --------------------------------------------------------------------------- #
def convex_hull(points):
    """Convex hull of 2D points, returned CCW. points: (N,2)."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def in_hull(query, hull):
    """Boolean mask: which query points (M,2) lie inside convex `hull` (CCW)."""
    q = np.asarray(query)
    n = len(hull)
    inside = np.ones(len(q), dtype=bool)
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        edge = b - a
        # inside (CCW) => cross(edge, q-a) >= 0 for every edge
        cr = edge[0] * (q[:, 1] - a[1]) - edge[1] * (q[:, 0] - a[0])
        inside &= cr >= -1e-9
    return inside


# --------------------------------------------------------------------------- #
#  Ensemble                                                                    #
# --------------------------------------------------------------------------- #
def train_ensemble(point_track, cfg=CFG, seed=0, verbose=False):
    """Train M PointMLPs with distinct seeds + bootstrap resamples."""
    x = torch.tensor(point_track["x"], dtype=torch.float32)
    y = torch.tensor(point_track["y"], dtype=torch.long)
    n = len(y)
    models = []
    for m in range(cfg.n_ensemble):
        torch.manual_seed(seed * 100 + m)
        boot = np.random.default_rng(seed * 100 + m).integers(0, n, size=n)
        xb, yb = x[boot], y[boot]
        net = PointMLP(cfg)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        lossf = nn.CrossEntropyLoss()
        for _ in range(cfg.ens_epochs):
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
        if verbose:
            print(f"  ensemble member {m}: train loss {loss.item():.3f}")
        models.append(net)
    return models


def _softmax_stack(models, x):
    """Return (M, N, C) softmax probabilities."""
    with torch.no_grad():
        probs = [torch.softmax(net(x), dim=1).numpy() for net in models]
    return np.stack(probs)


def disagreement_map(models, cfg=CFG, kind="variance"):
    """Disagreement over a grid of the generative square [-lim, lim]^2.

    kind='variance': mean over classes of the across-ensemble variance of the
    softmax probability. kind='mi': mutual information (total - aleatoric).
    Returns (grid_pts (G,2), disagreement (G,), and the axis coordinate vector).
    """
    lim, gn = cfg.grid_lim, cfg.grid_n
    axis = np.linspace(-lim, lim, gn)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)          # (G, 2)
    emb = Embedding(cfg)
    xg = torch.tensor(emb(grid, noise=0.0), dtype=torch.float32)
    probs = _softmax_stack(models, xg)                          # (M, G, C)
    if kind == "variance":
        dis = probs.var(axis=0).mean(axis=1)                    # (G,)
    elif kind == "mi":
        mean_p = probs.mean(axis=0)                             # (G, C)
        ent = -(mean_p * np.log(mean_p + 1e-9)).sum(axis=1)
        exp_ent = -(probs * np.log(probs + 1e-9)).sum(axis=2).mean(axis=0)
        dis = ent - exp_ent
    else:
        raise ValueError(kind)
    return grid, dis, axis


def train_ensemble_regression(reg_track, cfg=CFG, seed=0):
    """v3 EU fix: ensemble of M regressors (seeds + bootstrap) on the regression
    point track. No class boundaries -> interior disagreement is coverage-driven.
    """
    from .models import PointRegressor
    x = torch.tensor(reg_track["x"], dtype=torch.float32)
    y = torch.tensor(reg_track["y"], dtype=torch.float32)
    n = len(y)
    models = []
    for m in range(cfg.n_ensemble):
        torch.manual_seed(seed * 100 + m)
        boot = np.random.default_rng(seed * 100 + m).integers(0, n, size=n)
        xb, yb = x[boot], y[boot]
        net = PointRegressor(cfg)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        lossf = nn.MSELoss()
        for _ in range(cfg.ens_epochs):
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
        models.append(net)
    return models


def disagreement_map_regression(models, cfg=CFG):
    """Disagreement = variance across the ensemble of the predicted regression
    mean, over the generative grid. Returns (grid, disagreement, axis)."""
    lim, gn = cfg.grid_lim, cfg.grid_n
    axis = np.linspace(-lim, lim, gn)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    emb = Embedding(cfg)
    xg = torch.tensor(emb(grid, noise=0.0), dtype=torch.float32)
    with torch.no_grad():
        preds = np.stack([net(xg).numpy() for net in models])   # (M, G)
    dis = preds.var(axis=0)                                      # (G,)
    return grid, dis, axis


def interior_peak(grid, dis, train_points, top_frac=0.1):
    """Estimate c_hat: disagreement-weighted centroid of the high-disagreement
    region *inside the convex hull* of the training data (isolates the interior
    singularity from exterior OOD).
    """
    hull = convex_hull(np.asarray(train_points))
    mask = in_hull(grid, hull)
    g_in, d_in = grid[mask], dis[mask]
    if len(d_in) == 0:
        raise RuntimeError("no grid points inside training hull")
    thr = np.quantile(d_in, 1.0 - top_frac)
    hi = d_in >= thr
    w = d_in[hi]
    c_hat = np.average(g_in[hi], axis=0, weights=w)
    return c_hat, dict(hull=hull, mask=mask, thr=float(thr),
                       n_high=int(hi.sum()))
