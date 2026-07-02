"""Evaluation (§5): robustness grid, conditional analyses, calibration.

The PRIMARY metric (§7.5) is conditional-on-no-crossing consistency with the
original label. Everything else is secondary or exploratory. No metric shopping.
"""
import numpy as np
import torch

from .config import CFG
from .data import (make_traj_track, perturbation_field, perturb_loop,
                   oracle_quantities, Embedding, sample_annulus_points)
from .losses import phase_of


# --------------------------------------------------------------------------- #
#  Test set with per-loop perturbation fields                                 #
# --------------------------------------------------------------------------- #
def build_test_set(cfg=CFG, seed=7):
    traj = make_traj_track(cfg, seed=seed, n_per_class=cfg.n_test_per_class)
    N = traj["p"].shape[0]
    fields = np.stack([perturbation_field(cfg, seed=1000 + i) for i in range(N)])
    return dict(p=traj["p"], y=traj["y"], fields=fields)


def eps_grid(cfg=CFG):
    return np.linspace(0.0, cfg.eps_grid_max, cfg.eps_grid_n)


# --------------------------------------------------------------------------- #
#  Robustness sweep                                                            #
# --------------------------------------------------------------------------- #
def run_robustness(predict, test, cfg=CFG, noise_seed=0):
    """`predict`: callable(x_tensor (M,T,D)) -> integer class array (M,).

    For each eps and each test loop: perturb the loop, re-embed (with observation
    noise), predict; store prediction, oracle new winding, and crossed-hole flag.
    Returns dict of arrays shaped (n_eps, N).
    """
    emb = Embedding(cfg)
    grid = eps_grid(cfg)
    P, Y, F = test["p"], test["y"], test["fields"]
    N, T = P.shape[0], cfg.T

    preds = np.zeros((len(grid), N), dtype=int)
    oracle_k = np.zeros((len(grid), N), dtype=int)
    oracle_W = np.zeros((len(grid), N), dtype=float)
    crossed = np.zeros((len(grid), N), dtype=bool)

    for ei, eps in enumerate(grid):
        rng = np.random.default_rng(noise_seed * 131 + ei)
        p_pert = np.stack([perturb_loop(P[i], eps, F[i]) for i in range(N)])  # (N,T,2)
        for i in range(N):
            oq = oracle_quantities(p_pert[i], cfg)
            oracle_k[ei, i] = oq["oracle_k"]; oracle_W[ei, i] = oq["oracle_W"]
            crossed[ei, i] = oq["crossed"]
        x = emb(p_pert.reshape(-1, 2), noise=cfg.obs_noise, rng=rng).reshape(N, T, cfg.D)
        preds[ei] = predict(torch.tensor(x, dtype=torch.float32))
    return dict(eps=grid, preds=preds, oracle_k=oracle_k, oracle_W=oracle_W,
                crossed=crossed, orig=Y)


# --------------------------------------------------------------------------- #
#  Conditional analyses                                                        #
# --------------------------------------------------------------------------- #
def plateau_curve(res):
    """P2: per-eps consistency with the ORIGINAL label on the no-crossing subset.

    Returns eps, consistency (fraction pred==orig | not crossed), and n per eps.
    """
    eps = res["eps"]
    cons, ns = [], []
    for ei in range(len(eps)):
        mask = ~res["crossed"][ei]
        n = int(mask.sum())
        if n == 0:
            cons.append(np.nan); ns.append(0); continue
        cons.append(float((res["preds"][ei][mask] == res["orig"][mask]).mean()))
        ns.append(n)
    return dict(eps=eps, consistency=np.array(cons), n=np.array(ns))


def cliff_analysis(res, chance=1.0 / 3):
    """P2b: on CROSSING samples, does the new prediction track the oracle NEW
    winding? Returns per-eps fraction pred==oracle_k on the crossing subset and
    the chance level. Also the fraction that still equals the ORIGINAL label
    (should drop -> the discrete flip)."""
    eps = res["eps"]
    track, still_orig, ns = [], [], []
    for ei in range(len(eps)):
        mask = res["crossed"][ei]
        n = int(mask.sum())
        ns.append(n)
        if n == 0:
            track.append(np.nan); still_orig.append(np.nan); continue
        track.append(float((res["preds"][ei][mask] == res["oracle_k"][ei][mask]).mean()))
        still_orig.append(float((res["preds"][ei][mask] == res["orig"][mask]).mean()))
    return dict(eps=eps, track_oracle=np.array(track),
                still_orig=np.array(still_orig), n=np.array(ns), chance=chance)


def overall_accuracy_vs_oracle(res):
    """Secondary: per-eps accuracy against the oracle CURRENT winding (all samples)."""
    eps = res["eps"]
    acc = [float((res["preds"][ei] == res["oracle_k"][ei]).mean()) for ei in range(len(eps))]
    return dict(eps=eps, acc=np.array(acc))


def clean_accuracy(predict, test, cfg=CFG, noise_seed=0):
    """Accuracy at eps=0 against the original label (P1)."""
    emb = Embedding(cfg)
    rng = np.random.default_rng(noise_seed)
    P, Y = test["p"], test["y"]
    N = P.shape[0]
    x = emb(P.reshape(-1, 2), noise=cfg.obs_noise, rng=rng).reshape(N, cfg.T, cfg.D)
    pred = predict(torch.tensor(x, dtype=torch.float32))
    return float((pred == Y).mean())


# --------------------------------------------------------------------------- #
#  Calibration inside the hole (P4)                                            #
# --------------------------------------------------------------------------- #
def hole_points(cfg=CFG, n=1500, seed=5):
    """Points inside the hole r < r_inner (never seen in training) and matched
    on-support annulus points, embedded (no obs noise)."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, (cfg.r_inner * 0.9) ** 2, size=n))
    th = rng.uniform(0, 2 * np.pi, size=n)
    p_hole = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    p_supp, _ = sample_annulus_points(n, cfg, rng)
    emb = Embedding(cfg)
    return dict(p_hole=p_hole, x_hole=emb(p_hole, noise=0.0),
                p_supp=p_supp, x_supp=emb(p_supp, noise=0.0))


def phase_norm_calibration(encoder, hp, cfg=CFG):
    """P4: mean/median ||f_theta(x)|| inside the hole vs on support (a phase-norm
    abstention signal). Lower inside the hole is the desired free signal."""
    with torch.no_grad():
        _, n_hole = phase_of(encoder(torch.tensor(hp["x_hole"], dtype=torch.float32)))
        _, n_supp = phase_of(encoder(torch.tensor(hp["x_supp"], dtype=torch.float32)))
    n_hole, n_supp = n_hole.numpy(), n_supp.numpy()
    return dict(hole_mean=float(n_hole.mean()), hole_med=float(np.median(n_hole)),
                supp_mean=float(n_supp.mean()), supp_med=float(np.median(n_supp)),
                depressed=bool(np.median(n_hole) < np.median(n_supp)),
                n_hole=n_hole, n_supp=n_supp)


def ensemble_confidence_in_hole(models, hp):
    """P4: point-ensemble max-softmax confidence inside the hole (expected:
    confidently arbitrary -> a calibration failure the phase norm avoids)."""
    xh = torch.tensor(hp["x_hole"], dtype=torch.float32)
    with torch.no_grad():
        probs = np.stack([torch.softmax(m(xh), 1).numpy() for m in models])
    mean_p = probs.mean(0)
    return dict(mean_maxprob=float(mean_p.max(1).mean()),
                med_maxprob=float(np.median(mean_p.max(1))))
