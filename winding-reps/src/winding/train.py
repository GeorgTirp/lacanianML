"""Training schedules (§4): install -> barrier-on -> kill-switch.

Phase arms (A/B/D) share one schedule and one config; only the phase `center`
differs. Baseline C is a supervised GRU. During phase-arm training we monitor
the integer winding `W` and `min||f||` on a fixed set of probe loops every eval
step -- the joint log that tests the conservation law (P3).

The integer winding is only ever *monitored* (W3); the optimized surrogate is
L_inst (full-loop angular supervision, W1).
"""
import numpy as np
import torch

from .config import CFG
from .models import PhaseEncoder, GRUBaseline
from .losses import installation_loss, gate_barrier, phase_of
from .topology import wrap, ang_c

_CLASS_TO_IDX = {-1: 0, 0: 1, 1: 2}
_IDX_TO_CLASS = {0: -1, 1: 0, 2: 1}


# --------------------------------------------------------------------------- #
#  batched winding of phase sequences                                         #
# --------------------------------------------------------------------------- #
def batched_winding(phi):
    """phi: (B, T) array of angles along closed loops -> (B,) winding numbers."""
    d = wrap(np.diff(phi, axis=1))
    close = wrap(phi[:, :1] - phi[:, -1:])
    d = np.concatenate([d, close], axis=1)
    return d.sum(axis=1) / (2 * np.pi)


def predict_phase(encoder, x):
    """Rounded winding of the phase sequence: y_hat = round((1/2pi) Σ wrap(Δφ))."""
    with torch.no_grad():
        f = encoder(x)                       # (B,T,2)
        phi, _ = phase_of(f)
    W = batched_winding(phi.numpy())
    return np.round(W).astype(int)           # class in {-1,0,+1}


# --------------------------------------------------------------------------- #
#  probe loops for conservation monitoring                                     #
# --------------------------------------------------------------------------- #
def make_probes(cfg=CFG, seed=999, n_per_class=2):
    from .data import make_loop, embed_loop
    rng = np.random.default_rng(seed)
    ps, ks = [], []
    for k in (-1, 0, 1):
        for _ in range(n_per_class):
            ps.append(make_loop(k, cfg, rng)); ks.append(k)
    p = np.stack(ps)
    x = np.stack([embed_loop(pp, cfg, noise=0.0) for pp in p])   # noiseless probe
    return dict(p=p, x=torch.tensor(x, dtype=torch.float32), k=np.array(ks))


def _probe_stats(encoder, probes):
    with torch.no_grad():
        f = encoder(probes["x"])
        phi, norm = phase_of(f)
    W = batched_winding(phi.numpy())
    minr = norm.numpy().min(axis=1)
    return W, minr


# --------------------------------------------------------------------------- #
#  Phase arm trainer (A / B / D)                                              #
# --------------------------------------------------------------------------- #
def train_phase_arm(traj, center, cfg=CFG, seed=0, probes=None, verbose=False):
    """Train a phase encoder anchored at `center`. Returns (encoder, logs)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if probes is None:
        probes = make_probes(cfg)
    center = np.asarray(center, dtype=float)

    x = torch.tensor(traj["x"], dtype=torch.float32)         # (N,T,D)
    p = traj["p"]                                             # (N,T,2)
    ang_target = torch.tensor(ang_c(p, center), dtype=torch.float32)  # (N,T)
    y = traj["y"]
    N = x.shape[0]

    enc = PhaseEncoder(cfg)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)

    barrier_on = False
    logs = {k: [] for k in ["step", "phase", "loss_inst", "loss_bar",
                            "train_acc", "barrier", "probe_minr"]}
    probe_W_log, probe_minr_log = [], []

    for step in range(cfg.steps_total):
        if step < cfg.steps_install:
            phase = "install"
        elif step < cfg.steps_install + cfg.steps_barrier:
            phase = "barrier"
        else:
            phase = "kill"

        idx = np.random.randint(0, N, size=cfg.batch)
        xb, ab = x[idx], ang_target[idx]
        f = enc(xb)                                          # (B,T,2)
        phi, norm = phase_of(f)

        loss = torch.zeros(())
        li = torch.zeros(())
        lb = torch.zeros(())
        if phase in ("install", "barrier"):                 # L_inst on
            li = installation_loss(phi, ab)
            loss = loss + li
        if barrier_on:                                      # barrier stays once on
            lb = gate_barrier(f, cfg.barrier_margin, cfg.lam_bar)
            loss = loss + lb

        # In the kill phase with the barrier off there is no active loss term
        # (nothing to optimize); skip the step rather than backprop an empty loss.
        if loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps_total - 1:
            pW, pminr = _probe_stats(enc, probes)
            # W2: activate barrier once winding is installed on ALL probe loops
            if not barrier_on and np.all(np.abs(pW - probes["k"]) < cfg.barrier_W_tol):
                if phase != "install" or step >= cfg.steps_install - cfg.eval_every:
                    barrier_on = True
            # a cheap running train accuracy on a fixed eval slice
            acc = float((predict_phase(enc, x[:300]) == y[:300]).mean())
            logs["step"].append(step); logs["phase"].append(phase)
            logs["loss_inst"].append(float(li.detach())); logs["loss_bar"].append(float(lb.detach()))
            logs["train_acc"].append(acc); logs["barrier"].append(int(barrier_on))
            logs["probe_minr"].append(float(pminr.min()))
            probe_W_log.append(pW); probe_minr_log.append(pminr)
            if verbose and step % (cfg.eval_every * 20) == 0:
                print(f"  [{phase}] step {step} acc {acc:.3f} barrier {int(barrier_on)} minr {pminr.min():.3f}")

    logs = {k: np.array(v) for k, v in logs.items()}
    logs["probe_W"] = np.array(probe_W_log)          # (n_eval, n_probe)
    logs["probe_minr_all"] = np.array(probe_minr_log)
    logs["probe_k"] = probes["k"]
    return enc, logs


# --------------------------------------------------------------------------- #
#  Baseline C trainer (supervised GRU)                                        #
# --------------------------------------------------------------------------- #
def train_baseline(traj, cfg=CFG, seed=0, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.tensor(traj["x"], dtype=torch.float32)
    y_idx = torch.tensor([_CLASS_TO_IDX[int(k)] for k in traj["y"]], dtype=torch.long)
    N = x.shape[0]

    net = GRUBaseline(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    lossf = torch.nn.CrossEntropyLoss()
    logs = {k: [] for k in ["step", "loss", "train_acc"]}

    for step in range(cfg.steps_total):
        idx = np.random.randint(0, N, size=cfg.batch)
        logits = net(x[idx])
        loss = lossf(logits, y_idx[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps_total - 1:
            with torch.no_grad():
                acc = float((net(x[:300]).argmax(1) == y_idx[:300]).float().mean())
            logs["step"].append(step); logs["loss"].append(float(loss)); logs["train_acc"].append(acc)
            if verbose and step % (cfg.eval_every * 20) == 0:
                print(f"  [C] step {step} acc {acc:.3f}")
    return net, {k: np.array(v) for k, v in logs.items()}


def predict_baseline(net, x):
    with torch.no_grad():
        idx = net(x).argmax(1).numpy()
    return np.array([_IDX_TO_CLASS[int(i)] for i in idx])
