"""Data generation: annulus with an irreducible interior hole (§2).

Generative space is 2D. Support is the annulus r in [r_inner, r_outer]; the open
disk r < r_inner has ZERO support by construction (the oracle epistemic
singularity, centered at the origin). Models only ever see the 20-D embedding
x = psi(p) + noise.

Provides:
  - Embedding: frozen smooth psi: R^2 -> R^D (2-layer sin/tanh net).
  - Point track: annulus points, class label = angular sector (K sectors).
  - Trajectory track: closed smooth loops, label = winding k in {-1,0,+1}.
  - Perturbation operator + per-sample oracle quantities (winding, hole flag).
"""
import math
import numpy as np

from .config import CFG
from .topology import winding_of_points

PI = math.pi


# --------------------------------------------------------------------------- #
#  Frozen embedding psi: R^2 -> R^D                                            #
# --------------------------------------------------------------------------- #
class Embedding:
    """Fixed random smooth map R^2 -> R^D (frozen 2-layer net, sin then tanh)."""

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.embed_seed)
        H, D = cfg.embed_hidden, cfg.D
        # layer 1: 2 -> H  (sin activation);  layer 2: H -> D  (tanh activation)
        self.W1 = rng.normal(size=(2, H)) * math.sqrt(2.0 / 2)
        self.b1 = rng.normal(size=(H,)) * 0.5
        self.W2 = rng.normal(size=(H, D)) * math.sqrt(2.0 / H)
        self.b2 = rng.normal(size=(D,)) * 0.5

    def __call__(self, p, noise=0.0, rng=None):
        """Embed 2D points p (..., 2) -> (..., D). Adds Gaussian noise if >0."""
        p = np.asarray(p, dtype=float)
        h = np.sin(p @ self.W1 + self.b1)
        x = np.tanh(h @ self.W2 + self.b2)
        if noise and noise > 0.0:
            rng = rng or np.random.default_rng()
            x = x + rng.normal(size=x.shape) * noise
        return x


# --------------------------------------------------------------------------- #
#  Point track                                                                 #
# --------------------------------------------------------------------------- #
def sample_annulus_points(n, cfg=CFG, rng=None):
    """n points uniformly (area-weighted) on the annulus. Returns (p, sector)."""
    rng = rng or np.random.default_rng()
    u = rng.uniform(size=n)
    r = np.sqrt(u * (cfg.r_outer ** 2 - cfg.r_inner ** 2) + cfg.r_inner ** 2)
    theta = rng.uniform(0, 2 * PI, size=n)
    p = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    sector = np.floor(theta / (2 * PI / cfg.n_sectors)).astype(int) % cfg.n_sectors
    return p, sector


def make_point_track(cfg=CFG, seed=0):
    """Point-track dataset for EU estimation + calibration eval."""
    rng = np.random.default_rng(seed)
    p, y = sample_annulus_points(cfg.n_points, cfg, rng)
    emb = Embedding(cfg)
    x = emb(p, noise=cfg.obs_noise, rng=rng)
    return dict(p=p, x=x, y=y)


# --------------------------------------------------------------------------- #
#  Periodic (Fourier) noise -- guarantees loops close exactly                  #
# --------------------------------------------------------------------------- #
def _fourier_series(t, coeffs):
    """Sum_{n>=1} a_n sin(2 pi n t) + b_n cos(2 pi n t). coeffs: (N, 2) array.

    Period 1 in t, so value at t=0 equals value at t=1 (loops close exactly).
    """
    N = coeffs.shape[0]
    out = np.zeros_like(t)
    for n in range(1, N + 1):
        a, b = coeffs[n - 1]
        out = out + a * np.sin(2 * PI * n * t) + b * np.cos(2 * PI * n * t)
    return out


def _rand_fourier(rng, N, amp):
    """Random Fourier coeffs, overall scaled so typical amplitude ~ amp."""
    c = rng.normal(size=(N, 2))
    c = c / (np.sqrt(np.sum(c ** 2)) + 1e-8) * amp
    return c


# --------------------------------------------------------------------------- #
#  Trajectory track                                                           #
# --------------------------------------------------------------------------- #
def make_loop(k, cfg=CFG, rng=None):
    """One closed loop with winding label k in {-1,0,+1}. Returns p (T, 2).

    alpha(t) = alpha_0 + 2 pi k t + periodic-noise(t)
    r(t)     = 1.5 + periodic-noise(t)  clipped to [r_clip_lo, r_clip_hi]
    For k = 0 the angular noise amplitude keeps alpha within one sector so the
    loop wanders without encircling the origin.
    """
    rng = rng or np.random.default_rng()
    t = np.arange(cfg.T) / cfg.T            # endpoint excluded -> T distinct pts
    alpha0 = rng.uniform(0, 2 * PI)
    a_coeffs = _rand_fourier(rng, cfg.n_fourier, cfg.noise_amp_alpha)
    r_coeffs = _rand_fourier(rng, cfg.n_fourier, cfg.noise_amp_r)
    alpha = alpha0 + 2 * PI * k * t + _fourier_series(t, a_coeffs)
    r = 1.5 + _fourier_series(t, r_coeffs)
    r = np.clip(r, cfg.r_clip_lo, cfg.r_clip_hi)
    p = np.stack([r * np.cos(alpha), r * np.sin(alpha)], axis=1)
    return p


def make_traj_track(cfg=CFG, seed=0, n_per_class=None):
    """Balanced trajectory dataset. Returns dict with p (N,T,2), x (N,T,D),
    y (N,) winding label, and oracle_W (N,) the exact winding of the *clean*
    loop around the origin (should equal y)."""
    rng = np.random.default_rng(seed)
    emb = Embedding(cfg)
    n_per = n_per_class or cfg.n_traj_per_class
    ps, ys = [], []
    for k in (-1, 0, 1):
        for _ in range(n_per):
            ps.append(make_loop(k, cfg, rng))
            ys.append(k)
    p = np.stack(ps)                                   # (N, T, 2)
    y = np.array(ys, dtype=int)
    # embed pointwise with observation noise
    x = emb(p.reshape(-1, 2), noise=cfg.obs_noise, rng=rng).reshape(p.shape[0], cfg.T, cfg.D)
    oracle_W = np.array([winding_of_points(pp) for pp in p])
    return dict(p=p, x=x, y=y, oracle_W=oracle_W)


# --------------------------------------------------------------------------- #
#  Perturbation operator + oracle quantities (§2, §5)                          #
# --------------------------------------------------------------------------- #
def perturbation_field(cfg=CFG, seed=0):
    """A smooth periodic 2D deformation field, unit peak magnitude. Deterministic
    per seed so a given loop deforms progressively as eps grows."""
    rng = np.random.default_rng(seed)
    cx = _rand_fourier(rng, cfg.n_fourier, 1.0)
    cy = _rand_fourier(rng, cfg.n_fourier, 1.0)
    t = np.arange(cfg.T) / cfg.T
    fx = _fourier_series(t, cx)
    fy = _fourier_series(t, cy)
    field = np.stack([fx, fy], axis=1)                 # (T, 2)
    peak = np.max(np.linalg.norm(field, axis=1)) + 1e-8
    return field / peak                                # unit peak magnitude


def perturb_loop(p, eps, field):
    """Deform loop p (T,2) by eps * field (periodic => stays closed)."""
    return p + eps * field


def oracle_quantities(p_pert, cfg=CFG):
    """Oracle truth for a (possibly perturbed) loop: exact winding around the
    origin and whether it entered the hole r < r_inner (crossed-the-hole)."""
    W = winding_of_points(p_pert)                       # float, ~integer
    crossed = bool(np.any(np.linalg.norm(p_pert, axis=1) < cfg.r_inner))
    return dict(oracle_W=W, oracle_k=int(round(W)), crossed=crossed)


def embed_loop(p, cfg=CFG, noise=None, rng=None):
    """Embed a loop p (T,2) -> (T,D). Default uses observation noise."""
    emb = Embedding(cfg)
    n = cfg.obs_noise if noise is None else noise
    return emb(p, noise=n, rng=rng)
