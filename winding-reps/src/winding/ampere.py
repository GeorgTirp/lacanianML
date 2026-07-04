"""v7 Ampère experiment: EU-as-lack tested quantitatively.

The charge law  ∮_γ ω_A = κ Σ_i Link(γ,A_i) q_i ,  q_i = ∫_{A_i} σ_f² dμ  — the
circulation quota per loop is set by the enclosed irreducible variance, read from
data. A LAW test in a clean world (two data-lacks with different epistemic
uncertainty), not a benchmark.

§1 discipline: additivity of the *designed* form and per-head drive coefficients
are TRUE BY CONSTRUCTION. What is tested is the emergent chain: data→charge (T1),
realization through a shared-trunk learned encoder (T2), coupled online dynamics
(T3), protection ordering (T4). This module builds the world, the charge
estimator, the model, and the charge-weighted drive; the parts test the law.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import CFG
from .data import Embedding
from .uncertainty import train_ensemble_regression, convex_hull, in_hull
from .losses import phase_of, gate_barrier
from .topology import ang_c, wrap, winding

PI = math.pi
DOMAIN = 3.0


@dataclass(frozen=True)
class World:
    r2: float = 0.8            # A2 radius (swept)
    rough: float = 2.5        # high-frequency amplitude near A1 (swept)
    c1: tuple = (-1.2, 0.0)
    r1: float = 1.2           # A1 large: genuine interior EU (small holes are
    c2: tuple = (1.6, 0.0)    #  pinned by their boundary -> no EU; verified in §2)
    seed: int = 0


def default_worlds():
    """>=5 variants spanning constructed charge ratios ~1.5-6x (§2 world sweep).
    A1 fixed-large (dominant lack); A2 radius + A1 roughness swept."""
    return [World(r2=r2, rough=ro) for r2, ro in
            [(1.00, 1.5), (0.90, 1.9), (0.80, 2.3), (0.70, 2.7), (0.62, 3.1)]]


# --------------------------------------------------------------------------- #
#  the world: target, support, sampling                                       #
# --------------------------------------------------------------------------- #
def target(p, w: World):
    """Smooth base + a high-frequency term localized near A1 (=> ensembles
    extrapolate wildly inside A1 => large sigma_f^2)."""
    p = np.asarray(p, float)
    d1 = np.linalg.norm(p - np.array(w.c1), axis=-1)
    a1 = np.arctan2(p[..., 1] - w.c1[1], p[..., 0] - w.c1[0])
    return (np.sin(p[..., 0]) + np.cos(1.3 * p[..., 1])
            + w.rough * np.sin(8 * a1) * np.exp(-d1 / 0.5))


def in_support(p, w: World):
    p = np.asarray(p, float)
    inside = np.all(np.abs(p) <= DOMAIN, axis=-1)
    d1 = np.linalg.norm(p - np.array(w.c1), axis=-1)
    d2 = np.linalg.norm(p - np.array(w.c2), axis=-1)
    return inside & (d1 >= w.r1) & (d2 >= w.r2)


def sample_support(n, w: World, rng):
    out = []
    while len(out) < n:
        q = rng.uniform(-DOMAIN, DOMAIN, size=(4 * n, 2))
        out.extend(q[in_support(q, w)].tolist())
    return np.array(out[:n])


def make_regression_data(w: World, cfg=CFG, seed=0, n=4000, extra=None):
    """Points on support (optionally + extra points inside A2 for the filling
    stream) with observed target; embedded to R^D."""
    rng = np.random.default_rng(seed)
    p = sample_support(n, w, rng)
    if extra is not None and len(extra):
        p = np.concatenate([p, extra], 0)
    y = target(p, w) + rng.normal(size=len(p)) * 0.1
    emb = Embedding(cfg)
    x = emb(p, noise=cfg.obs_noise, rng=rng)
    return dict(p=p, x=x, y=y.astype(np.float32))


# --------------------------------------------------------------------------- #
#  charge estimation: two interior components, q_hat = integrated excess var   #
# --------------------------------------------------------------------------- #
def disagreement_grid(models, w: World, cfg=CFG, gn=90):
    axis = np.linspace(-DOMAIN, DOMAIN, gn)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    emb = Embedding(cfg)
    xg = torch.tensor(emb(grid, noise=0.0), dtype=torch.float32)
    with torch.no_grad():
        preds = np.stack([m(xg).numpy() for m in models])       # (M, G)
    dis = preds.var(0).reshape(gn, gn)
    return grid.reshape(gn, gn, 2), dis, axis


def _components(mask):
    """Connected components of a boolean grid (4-neighbour flood fill)."""
    gn = mask.shape[0]; lab = -np.ones_like(mask, int); comps = []
    for i in range(gn):
        for j in range(gn):
            if mask[i, j] and lab[i, j] < 0:
                stack = [(i, j)]; cid = len(comps); cells = []
                lab[i, j] = cid
                while stack:
                    a, b = stack.pop(); cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < gn and 0 <= nb < gn and mask[na, nb] and lab[na, nb] < 0:
                            lab[na, nb] = cid; stack.append((na, nb))
                comps.append(cells)
    return comps


def estimate_charges(models, w: World, cfg=CFG, gn=90, top_frac=0.15):
    """Return the two largest interior high-disagreement components as
    [(c_hat, q_hat), ...] ordered by q_hat desc, plus the map for the figure.
    q_hat = Σ_cells (disagreement − background_median) · cell_area."""
    grid, dis, axis = disagreement_grid(models, w, cfg, gn)
    pts = grid.reshape(-1, 2)
    train = sample_support(1500, w, np.random.default_rng(w.seed))
    hull = convex_hull(train)
    inside = in_hull(pts, hull).reshape(gn, gn)
    bg = float(np.median(dis[inside]))
    thr = np.quantile(dis[inside], 1 - top_frac)
    mask = inside & (dis >= thr)
    cell_area = (2 * DOMAIN / (gn - 1)) ** 2
    comps = _components(mask)
    charged = []
    for cells in comps:
        if len(cells) < 3:
            continue
        idx = np.array(cells)
        if np.any(idx == 0) or np.any(idx == gn - 1):
            continue                                           # boundary/edge OOD, not a lack
        d = dis[idx[:, 0], idx[:, 1]]
        w_excess = np.clip(d - bg, 0, None)
        q = float(np.sum(w_excess) * cell_area)
        coords = grid[idx[:, 0], idx[:, 1]]
        c_hat = np.average(coords, axis=0, weights=w_excess + 1e-9)
        charged.append((c_hat, q))
    charged.sort(key=lambda t: -t[1])
    oq = oracle_charges_from_map(grid, dis, bg, w, cell_area)
    return charged[:2], dict(grid=grid, dis=dis, axis=axis, bg=bg, thr=thr,
                            hull=hull, oracle_q=oq)


def oracle_charges_from_map(grid, dis, bg, w: World, cell_area):
    """Oracle-ensemble charge: integrate excess disagreement over the TRUE disk
    regions (known centers/radii) — the anchor for P-A1b (measured within 2x)."""
    out = []
    for c, r in [(w.c1, w.r1), (w.c2, w.r2)]:
        m = np.linalg.norm(grid - np.array(c), axis=2) < r
        out.append(float(np.sum(np.clip(dis[m] - bg, 0, None)) * cell_area))
    return out


# --------------------------------------------------------------------------- #
#  model: shared trunk, one or two phase heads                                #
# --------------------------------------------------------------------------- #
class TwoHead(nn.Module):
    def __init__(self, cfg=CFG, n_heads=2, gate_init=0.02):
        super().__init__()
        h = cfg.enc_hidden
        self.trunk = nn.Sequential(nn.Linear(cfg.D, h), nn.Tanh(),
                                   nn.Linear(h, h), nn.Tanh())
        self.heads = nn.ModuleList([nn.Linear(h, 2) for _ in range(n_heads)])
        with torch.no_grad():
            for hd in self.heads:
                hd.weight.mul_(gate_init); hd.bias.mul_(gate_init)

    def f(self, x, i):
        return self.heads[i](self.trunk(x))

    def head_params(self, i):
        return list(self.trunk.parameters()) + list(self.heads[i].parameters())


def phase(model, x, i):
    return phase_of(model.f(x, i))[0]


# --------------------------------------------------------------------------- #
#  charge-weighted drive  Ω = κ Σ_i q̃_i D^i   (D^i normalized, q̃ normalized)  #
# --------------------------------------------------------------------------- #
def head_drive(model, x, i, margin):
    """v4 stop-gradient angular pullback for head i -> unit param-direction that
    advances head i's phase CCW (grads wrt trunk+head_i; trunk sharing is the
    cross-talk that T2 tests)."""
    f = model.f(x, i).reshape(-1, 2)
    u, v = f[:, 0], f[:, 1]
    denom = torch.clamp(u * u + v * v, min=margin ** 2)
    w_u = (-v / denom).detach(); w_v = (u / denom).detach()
    scalar = torch.mean(w_u * u + w_v * v)
    params = model.head_params(i)
    grads = torch.autograd.grad(scalar, params)
    flat = torch.cat([g.reshape(-1) for g in grads])
    return grads, params, float(flat.norm())


def apply_charge_drive(model, x, qhat, kappa, margin):
    """theta += kappa Σ_i q̃_i * normalize(D^i). q̃ = q/Σq (fixed scale across
    worlds). Returns per-head applied magnitudes."""
    qsum = sum(qhat) + 1e-12
    mags = []
    for i, q in enumerate(qhat):
        grads, params, gnorm = head_drive(model, x, i, margin)
        coef = kappa * (q / qsum) / (gnorm + 1e-12)
        with torch.no_grad():
            for p, g in zip(params, grads):
                p.add_(coef * g)
        mags.append(coef * gnorm)
    return mags


# --------------------------------------------------------------------------- #
#  rate measurement (v4 P8): mean wrapped phase advance per idle step          #
# --------------------------------------------------------------------------- #
def circ_mean(model, x, i):
    with torch.no_grad():
        phi = phase(model, x, i)
    return float(torch.atan2(torch.sin(phi).mean(), torch.cos(phi).mean()))


def install(model, w: World, chat, cfg=CFG, steps=800, seed=0, barrier_at=400):
    """Install winding +1 around chat_i on head i via full-domain oracle angular
    supervision (L_inst toward ang_chat_i, W1-compliant); barrier after warmup.
    Kill-switch is implicit: callers stop calling install before measurement."""
    torch.manual_seed(seed); np.random.seed(seed)
    emb = Embedding(cfg); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(steps):
        p = sample_support(96, w, rng)
        x = torch.tensor(emb(p, noise=cfg.obs_noise, rng=rng), dtype=torch.float32)
        loss = torch.zeros(())
        for i, c in enumerate(chat):
            ang = torch.tensor(ang_c(p, c), dtype=torch.float32)
            loss = loss + torch.mean(1.0 - torch.cos(phase(model, x, i) - ang))
            if step >= barrier_at:
                loss = loss + gate_barrier(model.f(x, i), cfg.barrier_margin, cfg.lam_bar)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def loop_around(c, r, n=200, phase0=0.0):
    t = np.linspace(0, 2 * PI, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(t + phase0), c[1] + r * np.sin(t + phase0)], 1)


def deformed_loops(kind, w: World, n_reps=5, T=256, seed=0):
    """Loop families with known linking (Part B). kind: 'A1','A2','both','neither'."""
    rng = np.random.default_rng(seed + hash(kind) % 1000)
    c1, c2 = np.array(w.c1), np.array(w.c2)
    loops = []
    for _ in range(n_reps):
        t = np.linspace(0, 2 * PI, T, endpoint=False)
        a = rng.normal(size=3) * 0.12
        wob = 1 + sum(a[k] * np.cos((k + 2) * t + rng.uniform(0, 2 * PI)) for k in range(3))
        if kind == "A1":
            c, rad = c1, (w.r1 + 0.35) * wob
        elif kind == "A2":
            c, rad = c2, (w.r2 + 0.3) * wob
        elif kind == "both":
            c = (c1 + c2) / 2
            rad = (np.linalg.norm(c1 - c2) / 2 + max(w.r1, w.r2) + 0.4) * wob
        else:                                          # neither: a loop enclosing no hole
            c, rad = np.array([0.0, 2.2]), 0.5 * wob
        loops.append(c + np.stack([rad * np.cos(t), rad * np.sin(t)], 1))
    return loops


def verify_winding(model, w: World, chat, cfg=CFG):
    """Decoded winding of head i around a probe loop enclosing chat_i."""
    emb = Embedding(cfg); out = []
    for i, c in enumerate(chat):
        r = (w.r1 if i == 0 else w.r2) + 0.25
        p = loop_around(c, r)
        p = p[in_support(p, w)]
        x = torch.tensor(emb(p, noise=0.0), dtype=torch.float32)
        with torch.no_grad():
            phi = phase(model, x, i).numpy()
        out.append(winding(phi))
    return out


def idle_rates(model, w: World, qhat, kappa, cfg=CFG, steps=2000, seed=0):
    """v4 P8 idle protocol with the charge-weighted drive. Returns per-head phase
    advance rate rho_i, ballisticity R^2, and traces."""
    emb = Embedding(cfg); rng = np.random.default_rng(seed + 3)
    probes = [probe_points(w, 0, cfg), probe_points(w, 1, cfg)]
    xdrive = torch.tensor(emb(sample_support(96, w, rng), noise=0.0), dtype=torch.float32)
    seqs = [[circ_mean(model, probes[i], i)] for i in (0, 1)]
    for _ in range(steps):
        apply_charge_drive(model, xdrive, qhat, kappa, cfg.barrier_margin)
        for i in (0, 1):
            seqs[i].append(circ_mean(model, probes[i], i))
    rho, r2, traces = [], [], []
    for i in (0, 1):
        y = np.unwrap(seqs[i]); t = np.arange(len(y))
        a = np.polyfit(t, y, 1); yh = np.polyval(a, t)
        rho.append(float(a[0]))
        r2.append(float(1 - np.sum((y - yh) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)))
        traces.append(y)
    return dict(rho=rho, r2=r2, traces=traces)


def probe_points(w: World, i, cfg=CFG, n=64, seed=0):
    """On-support points in an annulus hugging hole i (for the rate probe)."""
    c = np.array(w.c1 if i == 0 else w.c2); r = (w.r1 if i == 0 else w.r2)
    rng = np.random.default_rng(seed + 100 * i)
    ang = rng.uniform(0, 2 * PI, n * 4)
    rad = (r + 0.15) + rng.uniform(0, 0.4, n * 4)
    p = c + np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    p = p[in_support(p, w)][:n]
    return torch.tensor(Embedding(cfg)(p, noise=0.0), dtype=torch.float32)
