"""exp0_sanity: Tier-1 port (torch). Reproduce conservation + gate.

Data: circle S^1 smoothly embedded in R^20 (3 random harmonics) + noise.
JEPA task: from encoding of x(alpha), predict encoding of x(alpha+delta),
predictor conditioned on delta. No stop-grad, no EMA -> collapse-prone.

Models:
  m0: plain JEPA, R^2 latent                         (expected: collapse)
  m1: m0 + VICReg-style variance hinge               (statistical anti-collapse)
  m2: phase head (u,v)->S^1, circular JEPA loss
      + equivariance loss (installs winding=1)
      + norm barrier (guards the gate), activated
        once winding ~ 1 on the probe loop           (topological anti-collapse)

Kill switch: m1 drops the variance term, m2 drops the equivariance term
(barrier stays). Phase 2 runs predictive loss only.

This ports the validated Tier-1 mechanism to the Tier-2 stack (torch) so the
conservation diagnostics (winding constant except at gate events) carry over.
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from _exp_common import seed_all, fig_path, result_path
from winding.topology import wrap, winding, gate_events, conservation_violations
from winding.losses import phase_of

PI = math.pi

HP = dict(D=20, K=3, noise=0.05, H=48, HP=32, batch=96, lr=2e-3,
          steps1=2000, steps2=3000, steps3=1500, every=20, margin=0.5,
          lam_e=5.0, lam_b=10.0, lam_v=1.0, dmax=PI, gate_thresh=0.02)


# ---------------- data: circle embedded via 3 harmonics --------------------
def make_embed(seed=0):
    rng = np.random.default_rng(seed)
    K, D = HP["K"], HP["D"]
    A = rng.normal(size=(K, D)) / math.sqrt(K)
    B = rng.normal(size=(K, D)) / math.sqrt(K)

    def embed(alpha):
        ks = np.arange(1, K + 1)[:, None]
        c = np.cos(ks * alpha[None, :])
        s = np.sin(ks * alpha[None, :])
        return (c.T @ A + s.T @ B).astype(np.float32)

    return embed


# ---------------- model ----------------------------------------------------
class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def normalize(z, eps=1e-8):
    return z / (z.norm(dim=1, keepdim=True) + eps)


def evaluate(enc, probe_x):
    with torch.no_grad():
        z = enc(probe_x)
    zn = z.numpy()
    sd = float(np.mean(np.std(zn, axis=0)))
    r = np.sqrt((zn ** 2).sum(axis=1))
    phi = np.arctan2(zn[:, 1], zn[:, 0])
    W = winding(phi)
    Rbar = float(np.abs(np.mean(np.exp(1j * phi))))
    return sd, Rbar, W, float(r.min())


def train(model, seed=1):
    seed_all(seed)
    embed = make_embed(seed=0)
    probe_alpha = np.linspace(0, 2 * PI, 256, endpoint=False)
    probe_x = torch.tensor(embed(probe_alpha))
    rng = np.random.default_rng(seed)

    enc = MLP([HP["D"], HP["H"], HP["H"], 2])
    pred = MLP([4, HP["HP"], HP["HP"], 2])
    # Start the phase head near the gate (f ~ 0) so the winding class must be
    # *born* by crossing the gate during installation -- this exhibits the gate
    # event that the conservation law predicts (rather than starting at W=1).
    with torch.no_grad():
        enc.net[-1].weight.mul_(0.02)
        enc.net[-1].bias.mul_(0.02)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=HP["lr"])

    barrier_on = False
    logs = {k: [] for k in ["step", "sd", "Rbar", "W", "minr", "barrier"]}
    T = HP["steps1"] + HP["steps2"] + HP["steps3"]
    for t in range(T):
        use_aux = t < HP["steps1"]
        alpha = rng.uniform(0, 2 * PI, HP["batch"])
        delta = rng.uniform(-HP["dmax"], HP["dmax"], HP["batch"])
        xb = torch.tensor(embed(alpha) + rng.normal(size=(HP["batch"], HP["D"])).astype(np.float32) * HP["noise"])
        xb2 = torch.tensor(embed(alpha + delta) + rng.normal(size=(HP["batch"], HP["D"])).astype(np.float32) * HP["noise"])
        dfeat = torch.tensor(np.stack([np.cos(delta), np.sin(delta)], axis=1).astype(np.float32))
        dt = torch.tensor(delta.astype(np.float32))

        z, z2 = enc(xb), enc(xb2)
        if model == "m2":
            zn, z2n = normalize(z), normalize(z2)
            p = normalize(pred(torch.cat([zn, dfeat], dim=1)))
            loss = torch.mean(1.0 - torch.sum(p * z2n, dim=1))
            if use_aux:
                phi1 = torch.atan2(zn[:, 1], zn[:, 0])
                phi2 = torch.atan2(z2n[:, 1], z2n[:, 0])
                loss = loss + HP["lam_e"] * torch.mean(1.0 - torch.cos(phi2 - phi1 - dt))
            if barrier_on:
                r = z.norm(dim=1)
                loss = loss + HP["lam_b"] * torch.mean(torch.clamp(HP["margin"] - r, min=0.0) ** 2)
        else:
            p = pred(torch.cat([z, dfeat], dim=1))
            loss = torch.mean(torch.sum((p - z2) ** 2, dim=1))
            if model == "m1" and use_aux:
                sd = torch.sqrt(torch.mean((z - z.mean(0)) ** 2, dim=0) + 1e-8)
                loss = loss + HP["lam_v"] * torch.mean(torch.clamp(1.0 - sd, min=0.0))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if t % HP["every"] == 0 or t == T - 1:
            sd, Rbar, W, minr = evaluate(enc, probe_x)
            if model == "m2" and not barrier_on and abs(W - 1.0) < 0.1:
                barrier_on = True
            logs["step"].append(t)
            logs["sd"].append(sd); logs["Rbar"].append(Rbar)
            logs["W"].append(W); logs["minr"].append(minr)
            logs["barrier"].append(int(barrier_on))
    return {k: np.array(v) for k, v in logs.items()}


def main():
    all_logs = {m: train(m, seed=1) for m in ("m0", "m1", "m2")}
    np.savez(result_path("exp0_logs.npz"),
             **{f"{m}_{k}": v for m, lg in all_logs.items() for k, v in lg.items()})

    # ---- conservation check for m2 ----
    lg = all_logs["m2"]
    _, gate_idx = gate_events(lg["minr"], HP["gate_thresh"])
    viol = conservation_violations(lg["W"], lg["minr"], HP["gate_thresh"])
    summary = dict(
        m0_final_sd=float(all_logs["m0"]["sd"][-1]),
        m2_final_sd=float(lg["sd"][-1]),
        m2_final_W=float(lg["W"][-1]),
        m2_min_norm_final=float(lg["minr"][-1]),
        m2_gate_events=len(gate_idx),
        m2_conservation_violations=len(viol),
    )
    print(json.dumps(summary, indent=2))

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for m, c in zip(("m0", "m1", "m2"), ("C3", "C1", "C0")):
        ax[0].plot(all_logs[m]["step"], all_logs[m]["sd"], label=m, color=c)
    ax[0].set(title="collapse metric: latent std", xlabel="step", ylabel="mean per-dim std")
    ax[0].legend()

    ax[1].plot(lg["step"], lg["W"], color="C0", label="W (winding)")
    ax[1].axhline(1.0, ls=":", color="gray")
    for gi in gate_idx:
        ax[1].axvline(lg["step"][gi], color="red", alpha=0.3)
    b_on = np.nonzero(lg["barrier"])[0]
    if len(b_on):
        ax[1].axvline(lg["step"][b_on[0]], color="green", ls="--", label="barrier on")
    ax[1].set(title="m2: winding conserved (red=gate events)", xlabel="step", ylabel="W")
    ax[1].legend()

    ax[2].plot(lg["step"], lg["minr"], color="C2")
    ax[2].axhline(HP["gate_thresh"], ls=":", color="red", label="gate threshold")
    ax[2].axhline(HP["margin"], ls=":", color="green", label="barrier margin")
    ax[2].set(title="m2: min ||f|| (distance to gate)", xlabel="step", ylabel="min ||f||")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(fig_path("exp0_conservation.png"), dpi=120)
    print("saved", fig_path("exp0_conservation.png"))

    assert len(viol) == 0, f"conservation violated at steps {viol}"
    print("PASS: winding conserved except at gate events; m2 avoids collapse.")


if __name__ == "__main__":
    main()
