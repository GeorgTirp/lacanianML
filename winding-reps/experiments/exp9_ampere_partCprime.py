"""exp9 Part C′ — the decay claim remeasured with a working instrument (v7 addendum).

Part C's circulation half was UNMEASURABLE (ρ₂ below the shared-trunk cross-talk
floor), not false. C′ re-tests the SAME locked prediction (P-C1) with a lock-in:
duty-cycle the dominant head's drive, measure ρ₂ only in the windows where the
cross-talk source is off. Three arms:
  C′-1 shared-trunk lock-in (primary), C′-2 milder world (ρ₂(0) above floor),
  C′-3 independent trunks (no cross-talk by construction — attribution control).
Bands unchanged (instrument change, not threshold change). §1 discipline holds.
"""
import argparse
import copy
import time

import numpy as np
import torch

from _exp_common import result_path
from winding.config import CFG
from winding import ampere as A
from winding.uncertainty import train_ensemble_regression
from winding.data import Embedding
from winding.losses import phase_of

KAPPA = 0.03
FILLS = np.linspace(0, 1, 9)
DEFAULT_W = A.default_worlds()[1]                    # ratio ~5
MILD_W = A.World(r2=1.0, rough=1.4)                  # ratio ~2 (ρ₂(0) above floor)


def charge_trajectory(w, seed):
    """q̂₁(fill), q̂₂(fill) via the oracle-region disagreement integral as A₂ fills."""
    rng = np.random.default_rng(seed + 50)
    q1s, q2s = [], []
    for frac in FILLS:
        n_extra = int(frac * 900)
        ang = rng.uniform(0, 2 * np.pi, n_extra); rr = w.r2 * np.sqrt(rng.uniform(0, 1, n_extra))
        extra = np.array(w.c2) + np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1)
        reg = A.make_regression_data(w, CFG, seed=seed, n=2500, extra=extra if n_extra else None)
        models = train_ensemble_regression(reg, CFG, seed=seed)
        _, mp = A.estimate_charges(models, w, CFG, gn=80)
        q1, q2 = mp["oracle_q"]
        q1s.append(q1); q2s.append(max(q2, 1e-6))
    return q1s, q2s


def _step_fn(model, xdrive, probes, qhat):
    def step(mask):
        A.apply_charge_drive(model, xdrive, qhat, KAPPA, CFG.barrier_margin, mask=mask)
        return A.circ_mean(model, probes[0], 0), A.circ_mean(model, probes[1], 1)
    return step


def run_arm(arm, w, seed, q_traj, steps=1000):
    ctor = A.IndepHeads if arm == "indep" else A.TwoHead
    m = ctor(CFG)
    # locate holes for install (use oracle centers; P1b: homotopy-correct suffices)
    A.install(m, w, [np.array(w.c1), np.array(w.c2)], CFG, steps=700, seed=seed)
    installed = copy.deepcopy(m.state_dict())        # reset point (isolates charge→rate map)
    emb = Embedding(CFG); rng = np.random.default_rng(seed + 7)
    probes = [A.probe_points(w, 0, CFG), A.probe_points(w, 1, CFG)]
    q1s, q2s = q_traj
    rho2w, rho2raw, rho1, minf2, gate = [], [], [], [], []
    for k, frac in enumerate(FILLS):
        m.load_state_dict(installed)                 # fresh, well-conditioned model each fill
        xdrive = torch.tensor(emb(A.sample_support(96, w, rng), noise=0.0), dtype=torch.float32)
        # A₁ stays empty forever (§2) => its charge is static; hold q̂₁ at its initial
        # measurement and re-estimate only the FILLING hole q̂₂ (re-estimating a static
        # lack each refit only injects estimator noise the drive would faithfully follow).
        out = A.lockin_measure(_step_fn(m, xdrive, probes, [q1s[0], q2s[k]]),
                               steps=steps, N=200, W=50)
        rho2w.append(out["rho2"]); rho2raw.append(out["rho2_raw"]); rho1.append(out["rho1"])
        with torch.no_grad():
            _, nrm = phase_of(m.f(probes[1], 1))
        minf2.append(float(nrm.min())); gate.append(int(nrm.min() < CFG.gate_thresh))
    return dict(frac=FILLS.tolist(), q1=q1s, q2=q2s, rho2_win=rho2w, rho2_raw=rho2raw,
                rho1=rho1, minf2=minf2, gate=gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        out = np.load(result_path("exp9_cprime.npy"), allow_pickle=True).item()
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
        t0 = time.time(); out = {"seeds": seeds, "arms": {}}
        for arm, w in [("shared", DEFAULT_W), ("mild", MILD_W), ("indep", DEFAULT_W)]:
            out["arms"][arm] = {}
            for s in seeds:
                qt = charge_trajectory(w, s)
                out["arms"][arm][s] = run_arm(arm, w, s, qt)
                r = out["arms"][arm][s]
                print(f"[{arm} s{s}] q̂₂ {r['q2'][0]:.3f}→{r['q2'][-1]:.4f} | "
                      f"ρ₂ᵂ {r['rho2_win'][0]:.4f}→{r['rho2_win'][-1]:.4f} "
                      f"(raw {r['rho2_raw'][0]:.4f}) ρ₁ {r['rho1'][0]:.4f}→{r['rho1'][-1]:.4f}", flush=True)
        out["runtime"] = time.time() - t0
        np.save(result_path("exp9_cprime.npy"), out, allow_pickle=True)
    from exp9_cprime_report import make_results
    import json
    print(json.dumps(make_results(out), indent=2))


if __name__ == "__main__":
    main()
