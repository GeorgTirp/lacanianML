"""exp4_drive: Tier-3, the ACTIVE component (v4).

A deployed model need not freeze. The drive D (a closed non-exact term in the
update rule, src/winding/drive.py) keeps it moving along the data-unidentified
U(1) phase fiber — directed (ballistic) motion that conserves the protected
winding, versus the "just add noise" null (SGLD).

  exp4a  period test: audit the formalism (∮D≈2π, ∮∇L_ssl≈0)              -> P8a
  exp4b  idling: 15k label-free steps under FROZEN/SSL/SGLD/DRIVE at
         MATCHED per-step displacement; Φ(t) ballistic vs diffusive        -> P8
  exp4c  shift adaptation: does idling preserve adaptability?              -> P9
         plus P10 exploratory diagnostics.

Kill criteria K1/K2/K3 are reported with headline prominence.
Reuses exp2-quality arm-A models (3 seeds). Additive; exp0-exp3 untouched.
"""
import argparse
import copy
import dataclasses
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from _exp_common import fig_path, ROOT, result_path
from winding.config import CFG
from winding.data import (make_point_track, make_traj_track, sample_annulus_points,
                          Embedding)
from winding.losses import phase_of, gate_barrier
from winding.uncertainty import train_ensemble, disagreement_map, interior_peak
from winding.train import train_phase_arm, predict_phase, make_probes, batched_winding
from winding.topology import conservation_violations, gate_events
from winding.drive import (QHead, sample_pairs, lssl, drive_vector, apply_drive,
                           period_integrals, circular_mean_phase, plasticity_metrics)

REGIMES = ["FROZEN", "SSL", "SGLD_lo", "SGLD_hi", "DRIVE_lo", "DRIVE_hi"]


# --------------------------------------------------------------------------- #
#  build arm A (reproduce exp2 arm-A: classification EU c_hat + phase arm)     #
# --------------------------------------------------------------------------- #
def build_armA(cfg, seed):
    pt = make_point_track(cfg, seed=seed)
    ens = train_ensemble(pt, cfg, seed=seed)
    grid, dis, axis = disagreement_map(ens, cfg)
    c_hat, _ = interior_peak(grid, dis, pt["p"], top_frac=0.1)
    traj = make_traj_track(cfg, seed=seed)
    probes = make_probes(cfg)
    enc, _ = train_phase_arm(traj, c_hat, cfg, seed=seed, probes=probes)
    return enc, probes


def make_fixtures(cfg, seed):
    rng = np.random.default_rng(1234)
    emb = Embedding(cfg)
    # deployment stream pool (unlabeled, stationary)
    stream = make_traj_track(cfg, seed=100, n_per_class=200)
    stream_x = torch.tensor(stream["x"], dtype=torch.float32)
    # eval set (winding retention vs true labels)
    ev = make_traj_track(cfg, seed=7, n_per_class=120)
    evalx = torch.tensor(ev["x"], dtype=torch.float32); evaly = ev["y"]
    # fixed phase-probe points; fixed held-out point batch
    pp, _ = sample_annulus_points(cfg.n_phase_probe, cfg, rng)
    phase_probe = torch.tensor(emb(pp, noise=0.0), dtype=torch.float32)
    ho, _ = sample_annulus_points(512, cfg, rng)
    heldout = torch.tensor(emb(ho, noise=0.0), dtype=torch.float32)
    return dict(stream_x=stream_x, evalx=evalx, evaly=evaly, ev=ev,
                phase_probe=phase_probe, heldout=heldout, stream_traj=stream)


def _flat_norm(vs):
    return float(torch.sqrt(sum((v * v).sum() for v in vs)))


def _n_params(enc):
    return sum(p.numel() for p in enc.parameters())


# --------------------------------------------------------------------------- #
#  one deployment step + a full loop                                          #
# --------------------------------------------------------------------------- #
def _sample_traj_batch(stream_x, cfg, gen):
    idx = torch.randint(0, stream_x.shape[0], (cfg.batch,), generator=gen)
    return stream_x[idx]


def deploy_loop(enc, q, regime, cfg, fx, steps, eta_d, sgld_sigma, gen,
                log_disp_only=False, barrier=True):
    opt = None
    if regime != "FROZEN":
        opt = torch.optim.Adam(list(enc.parameters()) + list(q.parameters()), lr=cfg.deploy_lr)
    theta0 = [p.detach().clone() for p in enc.parameters()]
    heldout, probe, evalx, evaly = fx["heldout"], fx["phase_probe"], fx["evalx"], fx["evaly"]
    Dpts = heldout  # points used to evaluate the drive field

    log = {k: [] for k in ["step", "phi", "retention", "W", "minr", "minr_probe",
                           "gate", "disp", "lssl", "pr", "sat", "gradnorm"]}
    disp_first = []
    for step in range(steps):
        if regime != "FROZEN":
            xb = _sample_traj_batch(fx["stream_x"], cfg, gen)
            x_t, x_tk, k = sample_pairs(xb, cfg.drive_kmax, gen)
            before = [p.detach().clone() for p in enc.parameters()]
            f_flat = enc(xb.reshape(-1, cfg.D))
            loss = lssl(enc, q, x_t, x_tk, k)
            if barrier:
                loss = loss + gate_barrier(f_flat, cfg.barrier_margin, cfg.lam_bar)
            opt.zero_grad(); loss.backward(); opt.step()
            if regime.startswith("SGLD"):
                with torch.no_grad():
                    for p in enc.parameters():
                        p.add_(sgld_sigma * torch.randn(p.shape, generator=gen))
            elif regime.startswith("DRIVE"):
                apply_drive(enc, drive_vector(enc, xb.reshape(-1, cfg.D), cfg.barrier_margin), eta_d)
            if len(disp_first) < cfg.drive_calib_steps:
                disp_first.append(_flat_norm([p - b for p, b in zip(enc.parameters(), before)]))

        if step % cfg.drive_log_every == 0 or step == steps - 1:
            log["step"].append(step)
            log["phi"].append(circular_mean_phase(enc, probe))
            log["disp"].append(_flat_norm([p - t0 for p, t0 in zip(enc.parameters(), theta0)]))
            if log_disp_only:
                continue
            log["retention"].append(float((predict_phase(enc, evalx) == evaly).mean()))
            with torch.no_grad():
                phi_pr, norm_pr = phase_of(enc(_probes_x(fx)))
            W = batched_winding(phi_pr.numpy()); minr = norm_pr.numpy().min(axis=1)
            log["W"].append(W); log["minr_probe"].append(minr)
            log["minr"].append(float(minr.min()))
            log["gate"].append(int(minr.min() < cfg.gate_thresh))
            # held-out L_ssl + grad norm
            xt2, xtk2, k2 = sample_pairs(fx["stream_x"][:cfg.batch], cfg.drive_kmax, gen)
            l = lssl(enc, q, xt2, xtk2, k2)
            log["lssl"].append(float(l.detach()))
            g = torch.autograd.grad(l, list(enc.parameters()), retain_graph=False)
            log["gradnorm"].append(_flat_norm(g))
            pr, sat = plasticity_metrics(enc, heldout)
            log["pr"].append(pr); log["sat"].append(sat)
    out = {k: (np.array(v) if v else np.array([])) for k, v in log.items()}
    out["disp_per_step"] = float(np.mean(disp_first)) if disp_first else 0.0
    out["probe_k"] = _PROBE_K
    return out, enc, q


_PROBE_X = None
_PROBE_K = None


def _probes_x(fx):
    return _PROBE_X


def _init_probes(cfg):
    global _PROBE_X, _PROBE_K
    pr = make_probes(cfg)
    _PROBE_X = pr["x"]; _PROBE_K = pr["k"]


# --------------------------------------------------------------------------- #
#  calibration: eta_d for target phase advance; SGLD sigma for matched disp    #
# --------------------------------------------------------------------------- #
def calibrate(enc0, cfg, fx, gen):
    # eta_d: pure-drive advance per step scales linearly in eta_d. Measure using
    # the SAME batch source (streaming trajectory batches) that deployment uses,
    # so the calibrated advance matches the deployed advance.
    enc = copy.deepcopy(enc0)
    trial = 1e-3
    genc = torch.Generator().manual_seed(7)
    seq = [circular_mean_phase(enc, fx["phase_probe"])]
    for _ in range(cfg.drive_calib_steps):
        xb = _sample_traj_batch(fx["stream_x"], cfg, genc)
        apply_drive(enc, drive_vector(enc, xb.reshape(-1, cfg.D), cfg.barrier_margin), trial)
        seq.append(circular_mean_phase(enc, fx["phase_probe"]))
    adv = float(np.mean(np.diff(np.unwrap(seq))))         # per-step advance at trial
    eta_lo = cfg.drive_advance_lo * trial / adv
    eta_hi = cfg.drive_advance_hi * trial / adv
    # SGLD sigma: match the extra displacement the drive adds per step
    def drive_extra(eta_d):
        enc = copy.deepcopy(enc0)
        gen2 = torch.Generator().manual_seed(0)
        mags = []
        for _ in range(cfg.drive_calib_steps):
            xb = _sample_traj_batch(fx["stream_x"], cfg, gen2)
            grads = drive_vector(enc, xb.reshape(-1, cfg.D), cfg.barrier_margin)
            mags.append(eta_d * _flat_norm(grads))
        return float(np.mean(mags))
    N = _n_params(enc0)
    sigma_lo = drive_extra(eta_lo) / np.sqrt(N)
    sigma_hi = drive_extra(eta_hi) / np.sqrt(N)
    return dict(eta_lo=eta_lo, eta_hi=eta_hi, sigma_lo=float(sigma_lo),
                sigma_hi=float(sigma_hi), adv_per_trial=adv)


def _regime_params(regime, cal):
    if regime == "DRIVE_lo": return cal["eta_lo"], 0.0
    if regime == "DRIVE_hi": return cal["eta_hi"], 0.0
    if regime == "SGLD_lo": return 0.0, cal["sigma_lo"]
    if regime == "SGLD_hi": return 0.0, cal["sigma_hi"]
    return 0.0, 0.0


# --------------------------------------------------------------------------- #
#  exp4a period test                                                          #
# --------------------------------------------------------------------------- #
def exp4a(enc, cfg, fx):
    traj = fx["stream_traj"]
    xt = torch.tensor(traj["x"], dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)
    x_t, x_tk, k = sample_pairs(xt, cfg.drive_kmax, gen)
    q = QHead(cfg.drive_kmax)
    circ_D, circ_L = period_integrals(enc, fx["heldout"], (x_t, x_tk, k, q),
                                      margin=cfg.barrier_margin, n=200)
    return dict(circ_D=circ_D, circ_L=circ_L,
                D_ok=abs(circ_D - 2 * np.pi) < 0.05 * 2 * np.pi,
                L_ok=abs(circ_L) < 0.05 * 2 * np.pi)


# --------------------------------------------------------------------------- #
#  per-seed run: 4a + 4b + 4c                                                  #
# --------------------------------------------------------------------------- #
def run_seed(cfg, seed, fx):
    enc0, probes = build_armA(cfg, seed)
    a = exp4a(enc0, cfg, fx)
    cal = calibrate(enc0, cfg, fx, None)

    idle, final_enc, final_q = {}, {}, {}
    for regime in REGIMES:
        gen = torch.Generator().manual_seed(seed * 17 + REGIMES.index(regime))
        enc = copy.deepcopy(enc0); q = QHead(cfg.drive_kmax)
        eta_d, sigma = _regime_params(regime, cal)
        log, e, qq = deploy_loop(enc, q, regime, cfg, fx, cfg.drive_steps_idle,
                                 eta_d, sigma, gen)
        idle[regime] = log; final_enc[regime] = e; final_q[regime] = qq

    # exp4c: shift then adapt each regime's END state with SSL only
    shift = _make_shift(cfg, seed)
    shifted = _apply_shift(fx, shift, cfg)
    adapt = {}
    for regime in REGIMES:
        gen = torch.Generator().manual_seed(seed * 31 + REGIMES.index(regime))
        enc = copy.deepcopy(final_enc[regime]); q = copy.deepcopy(final_q[regime])
        log, _, _ = deploy_loop(enc, q, "SSL", cfg, shifted, cfg.drive_steps_shift,
                                0.0, 0.0, gen)
        adapt[regime] = log
    return dict(seed=seed, a=a, cal=cal, idle=idle, adapt=adapt,
                pre_shift_lssl={r: float(idle[r]["lssl"][-1]) for r in REGIMES})


# --------------------------------------------------------------------------- #
#  sensor-drift shift (exp4c)                                                  #
# --------------------------------------------------------------------------- #
def _make_shift(cfg, seed):
    """Fixed sensor-drift rotation Q = exp(alpha * S), S skew (=> Q orthogonal).
    alpha scaled by shift_strength; world topology (loops, winding) unchanged."""
    rng = np.random.default_rng(500 + seed)
    G = rng.normal(size=(cfg.D, cfg.D))
    S = (G - G.T)
    S = S / np.linalg.norm(S)                 # unit-norm skew-symmetric
    Q = torch.matrix_exp(torch.tensor(cfg.shift_strength * 2.0 * S, dtype=torch.float32))
    return Q.numpy().astype(np.float32)


def _apply_shift(fx, Q, cfg):
    rng = np.random.default_rng(9)
    sx = fx["stream_x"].numpy() @ Q.T + rng.normal(size=fx["stream_x"].shape) * cfg.obs_noise
    ex = fx["evalx"].numpy() @ Q.T + rng.normal(size=fx["evalx"].shape) * cfg.obs_noise
    new = dict(fx)
    new["stream_x"] = torch.tensor(sx, dtype=torch.float32)
    new["evalx"] = torch.tensor(ex, dtype=torch.float32)
    return new


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="re-run only the analysis/figures/RESULTS from saved logs")
    a = ap.parse_args()
    cfg = CFG
    if a.quick:
        cfg = dataclasses.replace(CFG, steps_install=500, steps_barrier=300, steps_kill=200,
                                  ens_epochs=25, n_points=1500, n_traj_per_class=120,
                                  drive_steps_idle=1500, drive_steps_shift=600,
                                  drive_calib_steps=100)
    _init_probes(cfg)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(cfg.seeds)
    save = result_path("exp4_per.npy")
    from exp4_report import make_figures_and_results
    if a.report_only:
        d = np.load(save, allow_pickle=True).item()
        per, runtime = d["per"], d["runtime"]
    else:
        t0 = time.time()
        fx = make_fixtures(cfg, seeds[0])
        per = []
        for seed in seeds:
            st = time.time()
            per.append(run_seed(cfg, seed, fx))
            print(f"[seed {seed}] 4a D={per[-1]['a']['circ_D']:.3f} L={per[-1]['a']['circ_L']:.3f} "
                  f"({time.time()-st:.0f}s)")
        runtime = time.time() - t0
        np.save(save, {"per": per, "runtime": runtime}, allow_pickle=True)
    verdict = make_figures_and_results(per, cfg, runtime)
    import json
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
