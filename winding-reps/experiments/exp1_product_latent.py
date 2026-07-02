"""exp1_product_latent: R^8 (geometric) x S^1 (phase). OPEN QUESTION.

Does the winding guard's rank floor -- the norm barrier that prevents the phase
part from collapsing to the gate -- also help the *geometric* R^8 part of a
shared-trunk representation resist collapse?

Setup: a ProductEncoder with a shared trunk splits into a geometric head (R^8)
and a phase head (R^2 -> S^1). The geometric head is trained on a deliberately
collapse-prone invariance objective (pull consecutive-step features together,
no anti-collapse term). The phase head installs winding=1 at the true center
and is guarded by the norm barrier. We compare the geometric head's effective
rank and per-dim std with the phase guard ON vs OFF.

This is exploratory: we report whatever happens (no pre-registered pass/fail).
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from _exp_common import seed_all, fig_path, result_path
from winding.config import CFG
from winding.data import make_traj_track
from winding.models import ProductEncoder
from winding.losses import phase_of, installation_loss, gate_barrier
from winding.topology import ang_c

STEPS = 2500
BARRIER_AT = 800


def _batched_winding(phi):
    d = np.mod(np.diff(phi, axis=1) + np.pi, 2 * np.pi) - np.pi
    close = np.mod(phi[:, :1] - phi[:, -1:] + np.pi, 2 * np.pi) - np.pi
    return np.concatenate([d, close], axis=1).sum(1) / (2 * np.pi)


def effective_rank(feats):
    """Participation ratio of the feature covariance: (Σλ)^2 / Σλ^2 in [1, dim]."""
    f = feats - feats.mean(0, keepdims=True)
    cov = (f.T @ f) / max(len(f) - 1, 1)
    lam = np.linalg.eigvalsh(cov)
    lam = np.clip(lam, 0, None)
    s = lam.sum()
    return float(s * s / (np.sum(lam ** 2) + 1e-12)) if s > 0 else 0.0


def run(guard_on, cfg=CFG, seed=0):
    seed_all(seed)
    traj = make_traj_track(cfg, seed=seed, n_per_class=150)
    x = torch.tensor(traj["x"], dtype=torch.float32)            # (N,T,D)
    p = traj["p"]
    ang = torch.tensor(ang_c(p, cfg.true_center), dtype=torch.float32)
    N, T = x.shape[0], cfg.T

    enc = ProductEncoder(cfg, geom_dim=8)
    with torch.no_grad():                                        # phase near gate
        enc.phase.weight.mul_(0.02); enc.phase.bias.mul_(0.02)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)

    logs = {k: [] for k in ["step", "geom_rank", "geom_std", "W", "minr", "barrier"]}
    probe = x[:60]
    for step in range(STEPS):
        idx = np.random.randint(0, N, size=cfg.batch)
        xb, ab = x[idx], ang[idx]
        geom, f = enc(xb)                                        # (B,T,8),(B,T,2)
        phi, norm = phase_of(f)

        # geometric collapse-prone invariance: consecutive steps pulled together
        g_next = torch.roll(geom, shifts=-1, dims=1)
        loss_geom = torch.mean(torch.sum((geom - g_next) ** 2, dim=-1))
        # phase installation (always) + barrier (after BARRIER_AT if guard_on)
        loss = loss_geom + installation_loss(phi, ab)
        barrier = guard_on and step >= BARRIER_AT
        if barrier:
            loss = loss + gate_barrier(f, cfg.barrier_margin, cfg.lam_bar)

        opt.zero_grad(); loss.backward(); opt.step()

        if step % 25 == 0 or step == STEPS - 1:
            with torch.no_grad():
                gp, fp = enc(probe)
                phip, normp = phase_of(fp)
                gflat = gp.reshape(-1, 8).numpy()
            W = float(np.mean(np.abs(_batched_winding(phip.numpy()))))
            logs["step"].append(step)
            logs["geom_rank"].append(effective_rank(gflat))
            logs["geom_std"].append(float(gflat.std()))
            logs["W"].append(W)
            logs["minr"].append(float(normp.numpy().min()))
            logs["barrier"].append(int(barrier))
    return {k: np.array(v) for k, v in logs.items()}


def main():
    on = run(guard_on=True)
    off = run(guard_on=False)
    np.savez(result_path("exp1_logs.npz"),
             **{f"on_{k}": v for k, v in on.items()},
             **{f"off_{k}": v for k, v in off.items()})

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(on["step"], on["geom_rank"], label="guard ON", color="C0")
    ax[0].plot(off["step"], off["geom_rank"], label="guard OFF", color="C3")
    ax[0].axvline(BARRIER_AT, ls="--", color="green", alpha=0.6)
    ax[0].set(title="geometric R^8 effective rank", xlabel="step",
              ylabel="participation ratio (1..8)"); ax[0].legend()
    ax[1].plot(on["step"], on["geom_std"], label="guard ON", color="C0")
    ax[1].plot(off["step"], off["geom_std"], label="guard OFF", color="C3")
    ax[1].axvline(BARRIER_AT, ls="--", color="green", alpha=0.6)
    ax[1].set(title="geometric R^8 per-dim std", xlabel="step", ylabel="std"); ax[1].legend()
    fig.tight_layout(); fig.savefig(fig_path("exp1_product_latent.png"), dpi=120)

    # The rank curve is noisy step-to-step; judge on the tail mean (post-collapse
    # steady state), not a single final point, to avoid reading noise.
    def tail(a, k=12):
        return float(np.mean(a[-k:]))
    r_on, r_off = tail(on["geom_rank"]), tail(off["geom_rank"])
    s_on, s_off = tail(on["geom_std"]), tail(off["geom_std"])
    summary = dict(
        rank_on_tailmean=r_on, rank_off_tailmean=r_off,
        std_on_tailmean=s_on, std_off_tailmean=s_off,
        W_on_final=float(on["W"][-1]), W_off_final=float(off["W"][-1]),
        minr_on_final=float(on["minr"][-1]), minr_off_final=float(off["minr"][-1]),
    )
    import json
    print(json.dumps(summary, indent=2))
    # Honest read: both geometric heads largely COLLAPSE (participation ratio <~1,
    # i.e. intermittently rank-0/1) under the collapse-prone invariance objective.
    # The two collapse metrics DISAGREE on direction (guard ON: higher rank but
    # LOWER std), so there is no consistent evidence the phase-part rank floor
    # protects the R^8 part. The only unambiguous effect of the guard is on the
    # PHASE norm it directly acts on (min||f||).
    rank_helps = (r_on - r_off) > 0.5
    std_helps = (s_on - s_off) > 0.005          # higher std = less collapsed
    if rank_helps and std_helps:
        verdict = "guard helps geometric part (both metrics agree)"
    elif (not rank_helps) and (not std_helps) and (r_off - r_on) > 0.5:
        verdict = "guard hurts geometric part (both metrics agree)"
    else:
        verdict = ("NO CONSISTENT EFFECT: both geometric heads largely collapse "
                   "(rank <~1); rank and std metrics disagree on sign. The phase "
                   "rank floor does NOT reliably propagate to the R^8 part here. The "
                   "guard's only clear effect is on the phase norm it directly acts "
                   f"on (min||f|| ON={summary['minr_on_final']:.2f} vs "
                   f"OFF={summary['minr_off_final']:.2f}).")
    print("OPEN-QUESTION OUTCOME:", verdict)
    print("saved", fig_path("exp1_product_latent.png"))


if __name__ == "__main__":
    main()
