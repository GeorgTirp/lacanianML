"""exp6_ring: topology in the DYNAMICS — a ring attractor (v6).

The pivot: move S^1 from the static readout map (Tiers 2-5) into recurrent
dynamics. The class is held by an attractor manifold maintained by recurrence,
not by a function of the current input. The money prediction is P15 — REPAIR BY
RELAXATION: an attractor projects a shifted state back onto its manifold, so the
winding repairs itself with NO oracle and NO gradient updates. If it works, the
v5 P12 negative (relational objectives cannot repair a broken class) is overturned
by the dynamics.

Installation is oracle-assisted (standing fairness note): a relational-only
objective admits the W1-trivial stationary-bump solution (verified: acc ~0.20).
The novelty — P15 repair — uses no oracle and no gradients.

Predictions: P13 ring topology (b1=1), P14 winding tracking + conservation,
P15 repair, P16 intrinsic drive, P17 two registers. Kill: K-ring / K-repair /
K-drift, reported with headline prominence.
"""
import argparse
import dataclasses
import math
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from _exp_common import fig_path, ROOT, result_path
from winding.config import CFG
from winding.data import make_traj_track, radius_class_labels, Embedding
from winding.ring import (ring_weights, relax, decode_phase, decode_amplitude,
                          RingEncoder, settle_batch, decode_current, ring_certificate,
                          preferred_angles, _phi)
from winding.topology import winding, ang_c, wrap
import exp4_drive as e4

V6_MARKER = "\n<!-- V6 SECTION -->\n"
PI = math.pi


def batched_winding(psi):
    d = wrap(np.diff(psi, axis=1))
    close = wrap(psi[:, :1] - psi[:, -1:])
    return np.concatenate([d, close], axis=1).sum(1) / (2 * PI)


# --------------------------------------------------------------------------- #
#  install e_psi (oracle angular supervision, full loop — W1-compliant)        #
# --------------------------------------------------------------------------- #
def install(cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    W = ring_weights(cfg)
    traj = make_traj_track(cfg, seed=seed)
    X = torch.tensor(traj["x"], dtype=torch.float32)
    ANG = torch.tensor(ang_c(traj["p"]), dtype=torch.float32)     # oracle angle
    y = traj["y"]; N = X.shape[0]
    enc = RingEncoder(cfg)
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    for step in range(cfg.ring_install_steps):
        idx = np.random.randint(0, N, 16)
        psi, rho, _ = settle_batch(enc, W, X[idx], cfg, R=cfg.ring_R_train)
        loss = torch.mean(1.0 - torch.cos(psi - ANG[idx]))
        opt.zero_grad(); loss.backward(); opt.step()
    return dict(W=W, enc=enc, traj=traj, X=X, y=y)


def settled_states(enc, W, x_pts, cfg, R=60):
    with torch.no_grad():
        b = enc(x_pts); a = cfg.ring_dt / cfg.ring_tau
        r = torch.zeros(x_pts.shape[0], cfg.N_ring)
        for _ in range(R):
            r = r + a * (-r + _phi(r @ W.T + b + cfg.ring_h0))
    return r.numpy()


# --------------------------------------------------------------------------- #
#  P13 ring topology                                                          #
# --------------------------------------------------------------------------- #
def p13(inst, cfg, seed):
    W, enc, X = inst["W"], inst["enc"], inst["X"]
    pts = X.reshape(-1, cfg.D)
    sub = pts[np.random.default_rng(seed).choice(pts.shape[0], 500, replace=False)]
    st_input = settled_states(enc, W, sub, cfg)
    # spontaneous states: relax from random inits, no input
    gen = torch.Generator().manual_seed(seed)
    r = torch.rand(150, cfg.N_ring, generator=gen) * 0.1
    a = cfg.ring_dt / cfg.ring_tau
    with torch.no_grad():
        for _ in range(200):
            r = r + a * (-r + _phi(r @ W.T + 0.0 + cfg.ring_h0))
    states = np.concatenate([st_input, r.numpy()], 0)
    cert = ring_certificate(states)
    cert["states"] = states
    return cert


# --------------------------------------------------------------------------- #
#  P14 winding tracking + conservation under continued training               #
# --------------------------------------------------------------------------- #
def p14(inst, cfg, seed):
    W, enc, X, y = inst["W"], inst["enc"], inst["X"], inst["y"]
    with torch.no_grad():
        psi, rho, _ = settle_batch(enc, W, X, cfg, R=8)
    Wd = batched_winding(psi.numpy())
    acc = float(np.mean(np.round(Wd) == y))
    # continued training (relational L_ssl) — monitor b1, probe-W, min rho
    enc2 = RingEncoder(cfg); enc2.load_state_dict(enc.state_dict())
    q = nn.Embedding(cfg.drive_kmax, 1); nn.init.zeros_(q.weight)
    opt = torch.optim.Adam(list(enc2.parameters()) + list(q.parameters()), lr=1e-3)
    probe = X[:9]                                    # fixed probe loops (3 per class)
    b1_series, W_series, minrho_series = [], [], []
    for step in range(300):
        idx = np.random.randint(0, X.shape[0], 16)
        psi, rho, _ = settle_batch(enc2, W, X[idx], cfg, R=cfg.ring_R_train)
        B, T = psi.shape
        tk = torch.randint(0, T, (B,)); kk = torch.randint(1, cfg.drive_kmax + 1, (B,))
        ar = torch.arange(B)
        dp = psi[ar, (tk + kk) % T] - psi[ar, tk]
        loss = torch.mean(1.0 - torch.cos(dp - q(kk - 1).squeeze(-1)))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 30 == 0:
            with torch.no_grad():
                pp, pr, _ = settle_batch(enc2, W, probe, cfg, R=8)
            W_series.append(batched_winding(pp.numpy()))
            minrho_series.append(float(pr.min()))
            b1_series.append(p13(dict(W=W, enc=enc2, X=X), cfg, seed)["b1"])
    W_series = np.array(W_series)                    # (n, 9)
    b1_changes = int(np.sum([b != b1_series[0] for b in b1_series]))
    w_changes = int(np.sum(np.round(W_series[1:]) != np.round(W_series[:-1])))
    collapsed = bool(np.min(minrho_series) < 0.1)
    return dict(track_acc=acc, b1_changes=b1_changes, w_changes=w_changes,
                collapsed=collapsed, min_rho=float(np.min(minrho_series)))


# --------------------------------------------------------------------------- #
#  P15 REPAIR BY RELAXATION (money)                                           #
# --------------------------------------------------------------------------- #
def p15(inst, cfg, seed):
    W, enc, traj, y = inst["W"], inst["enc"], inst["traj"], inst["y"]
    emb = Embedding(cfg)
    P = traj["p"]; Np, T = P.shape[0], cfg.T
    clean = torch.tensor(emb(P.reshape(-1, 2), noise=0.0).reshape(Np, T, cfg.D), dtype=torch.float32)
    sel = np.random.default_rng(seed).choice(Np, 90, replace=False)     # subsample loops
    clean_s = clean[sel]; ys = y[sel]

    def surv(x):
        with torch.no_grad():
            ps = decode_current(enc, x, cfg)
            pd, _, _ = settle_batch(enc, W, x, cfg, R=cfg.ring_R_rep, carry=True)
        return (float(np.mean(np.round(batched_winding(ps.numpy())) == ys)),
                float(np.mean(np.round(batched_winding(pd.numpy())) == ys)))

    stat, dyn = [], []                                    # pre-registered: exp5 shift
    for i in range(cfg.ring_shifts):
        Q = torch.tensor(e4._make_shift(cfg, 1000 + i), dtype=torch.float32)
        rng = np.random.default_rng(3000 + i)
        xsh = clean_s @ Q.T + torch.tensor(rng.normal(size=clean_s.shape) * cfg.obs_noise, dtype=torch.float32)
        s, d = surv(xsh); stat.append(s); dyn.append(d)
    # mechanism contrast: HIGH-frequency off-manifold noise (no shift), sigma=1.0
    nst, ndy = [], []
    for i in range(cfg.ring_shifts):
        rng = np.random.default_rng(6000 + i)
        xn = clean_s + torch.tensor(rng.normal(size=clean_s.shape) * 1.0, dtype=torch.float32)
        s, d = surv(xn); nst.append(s); ndy.append(d)
    return dict(static=float(np.mean(stat)), dynamic=float(np.mean(dyn)),
                static_all=stat, dynamic_all=dyn,
                noise_static=float(np.mean(nst)), noise_dynamic=float(np.mean(ndy)))


# --------------------------------------------------------------------------- #
#  P16 intrinsic drive (asymmetric connectivity, zero input)                  #
# --------------------------------------------------------------------------- #
def p16(cfg, seed):
    th = preferred_angles(cfg.N_ring)
    out = {}
    for eta in (0.3, 0.6):
        W = ring_weights(cfg, eta=eta)
        r = relax(W, 0.5 * torch.cos(th - 0.0)[None], cfg, R=60)
        seq, amp, sat = [], [], []
        for _ in range(200):
            r = relax(W, torch.zeros(1, cfg.N_ring), cfg, R=2, r0=r)
            seq.append(float(decode_phase(r, th)[0]))
            amp.append(float(decode_amplitude(r, th)[0]))
            sat.append(float((r > 0.95).float().mean()))
        un = np.unwrap(seq)
        vel = float((un[-1] - un[0]) / len(un))
        # linear fit R^2 for ballisticity
        t = np.arange(len(un)); A = np.polyfit(t, un, 1); yh = np.polyval(A, t)
        r2 = float(1 - np.sum((un - yh) ** 2) / (np.sum((un - un.mean()) ** 2) + 1e-9))
        out[eta] = dict(vel=vel, r2=r2, amp_mean=float(np.mean(amp)),
                        amp_cv=float(np.std(amp) / (np.mean(amp) + 1e-9)),
                        sat=float(np.mean(sat)), seq=un.tolist())
    return out


# --------------------------------------------------------------------------- #
#  P17 two registers (full population vs 2-unit bottleneck)                    #
# --------------------------------------------------------------------------- #
def p17(inst, cfg, seed):
    W, enc, traj = inst["W"], inst["enc"], inst["traj"]
    X = inst["X"]; lab = torch.tensor(radius_class_labels(traj["p"], cfg), dtype=torch.long)
    # settled population per loop (mean over time of settled states)
    with torch.no_grad():
        _, _, _ = settle_batch(enc, W, X[:1], cfg, R=8)   # warmup
        feats = []
        for i in range(0, X.shape[0], 60):
            _, _, r = settle_batch(enc, W, X[i:i + 60], cfg, R=8)
            # use time-mean population: settle each timestep? use final state proxy
            feats.append(r)
        pop = torch.cat(feats, 0)                          # (N_loops, N_ring)

    def train_head(inp, in_dim):
        torch.manual_seed(seed); head = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 3))
        opt = torch.optim.Adam(head.parameters(), lr=3e-3); lossf = nn.CrossEntropyLoss()
        for _ in range(400):
            idx = np.random.randint(0, inp.shape[0], 32)
            loss = lossf(head(inp[idx]), lab[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = float((head(inp).argmax(1) == lab).float().mean())
        return acc
    ring_acc = train_head(pop.detach(), cfg.N_ring)
    # matched MLP baseline on raw embedding (mean over loop)
    base_in = X.mean(1)
    base_acc = train_head(base_in, cfg.D)
    return dict(ring_pop_acc=ring_acc, mlp_baseline_acc=base_acc)


# --------------------------------------------------------------------------- #
#  K-drift: bump drift under input noise (no shift)                           #
# --------------------------------------------------------------------------- #
def drift_rate(cfg, seed, noise=0.1, steps=300):
    th = preferred_angles(cfg.N_ring); W = ring_weights(cfg)
    r = relax(W, 0.5 * torch.cos(th - 1.0)[None], cfg, R=60)
    gen = torch.Generator().manual_seed(seed); angs = []
    for _ in range(steps):
        r = relax(W, torch.randn(1, cfg.N_ring, generator=gen) * noise, cfg, R=3, r0=r)
        angs.append(float(decode_phase(r, th)[0]))
    un = np.unwrap(angs)
    return float(abs(un[-1] - un[0]) / steps)


# =========================================================================== #
def run(cfg, seeds):
    t0 = time.time()
    per = []
    for seed in seeds:
        inst = install(cfg, seed)
        per.append(dict(seed=seed, p13=p13(inst, cfg, seed), p14=p14(inst, cfg, seed),
                        p15=p15(inst, cfg, seed), p16=p16(cfg, seed),
                        p17=p17(inst, cfg, seed), drift=drift_rate(cfg, seed)))
        print(f"[seed {seed}] b1={per[-1]['p13']['b1']} track={per[-1]['p14']['track_acc']:.2f} "
              f"repair static={per[-1]['p15']['static']:.2f}->dyn={per[-1]['p15']['dynamic']:.2f} "
              f"({time.time()-t0:.0f}s)")
    out = dict(per=per, runtime=time.time() - t0, seeds=list(seeds))
    np.save(result_path("exp6_data.npy"), out, allow_pickle=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    cfg = CFG
    if a.quick:
        cfg = dataclasses.replace(CFG, n_traj_per_class=100, ring_install_steps=150, ring_shifts=5)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(cfg.seeds)
    if a.report_only:
        out = np.load(result_path("exp6_data.npy"), allow_pickle=True).item()
    else:
        out = run(cfg, seeds)
    from exp6_report import make_figures_and_results
    import json
    print(json.dumps(make_figures_and_results(out, cfg), indent=2))


if __name__ == "__main__":
    main()
