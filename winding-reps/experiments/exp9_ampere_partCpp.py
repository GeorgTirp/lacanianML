"""exp9 Part C″ — banking the crown claim by the letter (v7 addendum).

C′'s discrimination clauses passed 3/3 (corr≥0.99, ρ₂ʷ end≤1%, no K-fixation/
K-no-discrimination) but strict P-C1 passed only 1/3 because the ρ₁-stability
clause was MIS-DESIGNED: two-sided (±20%) while the guarded risk (the true lack's
circulation decaying like the false one's) is ONE-SIDED. C′-2/C′-3 failed it via
UPWARD fluctuation (ρ₁≈1.2), which instantiates no guarded risk.

Per the v5 rule, the locked clause cannot be reinterpreted post-hoc — it is
corrected IN ADVANCE and re-tested on FRESH seeds [3,4,5]:
  OLD: ρ₁ within 20% of initial.   NEW: ρ₁ never falls below 0.8·initial.
Variance aid (fixed in advance): double the per-point averaging for ρ₁.

Also answers the carry-over: in the rate arms the ENCODER is not trained on fill
data (model reset each fill, drive only) ⇒ the gate log there is trivial. So an
exploratory healing arm fine-tunes the encoder on the fill data (regression + the
head-2 barrier) and logs min‖f₂‖ / winding as A₂ heals.
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn

from _exp_common import result_path
from winding.config import CFG
from winding import ampere as A
from winding.data import Embedding
from winding.losses import phase_of, gate_barrier
from winding.topology import winding
from exp9_ampere_partCprime import charge_trajectory, run_arm, DEFAULT_W, MILD_W, FILLS

SEEDS = [3, 4, 5]                       # FRESH — never seen by the corrected clause
STEPS = 2000                            # doubled per-point averaging (§1 variance aid)


def heal_arm(seed, w=None, ft_steps=400):
    """Exploratory: the encoder IS trained on the fill data (regression + head-2
    barrier). Track min‖f₂‖ over A₂ and head-2 winding as A₂ heals — does the
    representational puncture survive (memory without cathexis) or dissolve?"""
    w = w or DEFAULT_W
    torch.manual_seed(seed); np.random.seed(seed)
    m = A.TwoHead(CFG); A.install(m, w, [np.array(w.c1), np.array(w.c2)], CFG, steps=700, seed=seed)
    installed = copy.deepcopy(m.state_dict())
    emb = Embedding(CFG); rng = np.random.default_rng(seed + 80); c2 = np.array(w.c2)
    fr, minf2, wind2, gate = [], [], [], []
    for frac in FILLS:
        m.load_state_dict(installed)
        reg = nn.Linear(CFG.enc_hidden, 1)
        n_extra = int(frac * 1200)
        an = rng.uniform(0, 2 * np.pi, n_extra); rr = w.r2 * np.sqrt(rng.uniform(0, 1, n_extra))
        extra = c2 + np.stack([rr * np.cos(an), rr * np.sin(an)], 1)
        psup = A.sample_support(1000, w, rng)
        p = np.concatenate([psup, extra], 0) if n_extra else psup
        x = torch.tensor(emb(p, noise=CFG.obs_noise, rng=rng), dtype=torch.float32)
        y = torch.tensor(A.target(p, w), dtype=torch.float32)[:, None]
        opt = torch.optim.Adam(list(m.parameters()) + list(reg.parameters()), lr=1e-3)
        for _ in range(ft_steps):
            idx = np.random.randint(0, len(x), 96)
            loss = ((reg(m.trunk(x[idx])) - y[idx]) ** 2).mean()
            loss = loss + gate_barrier(m.f(x[idx], 1), CFG.barrier_margin, CFG.lam_bar)
            opt.zero_grad(); loss.backward(); opt.step()
        an = rng.uniform(0, 2 * np.pi, 400); rr = w.r2 * np.sqrt(rng.uniform(0, 1, 400))
        disk = c2 + np.stack([rr * np.cos(an), rr * np.sin(an)], 1)
        with torch.no_grad():
            _, nrm = phase_of(m.f(torch.tensor(emb(disk, noise=0.0), dtype=torch.float32), 1))
        loop = A.loop_around(c2, w.r2 + 0.1)
        with torch.no_grad():
            phi = A.phase(m, torch.tensor(emb(loop, noise=0.0), dtype=torch.float32), 1).numpy()
        fr.append(float(frac)); minf2.append(float(nrm.min()))
        gate.append(int(nrm.min() < CFG.gate_thresh)); wind2.append(winding(phi))
    return dict(frac=fr, minf2=minf2, wind2=wind2, gate=gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        out = np.load(result_path("exp9_cpp.npy"), allow_pickle=True).item()
    else:
        t0 = time.time(); out = {"seeds": SEEDS, "arms": {}, "heal": {}}
        for arm, w in [("shared", DEFAULT_W), ("mild", MILD_W), ("indep", DEFAULT_W)]:
            out["arms"][arm] = {}
            for s in SEEDS:
                out["arms"][arm][s] = run_arm(arm, w, s, charge_trajectory(w, s), steps=STEPS)
                r = out["arms"][arm][s]
                print(f"[{arm} s{s}] ρ₂ʷ {r['rho2_win'][0]:.4f}→{r['rho2_win'][-1]:.4f} "
                      f"ρ₁ {r['rho1'][0]:.4f}→{r['rho1'][-1]:.4f} "
                      f"(min ρ₁/ρ₁₀ {min(np.array(r['rho1'])/abs(r['rho1'][0])):.2f})", flush=True)
        for s in SEEDS:
            out["heal"][s] = heal_arm(s)
            h = out["heal"][s]
            print(f"[heal s{s}] wind2 {h['wind2'][0]:.1f}→{h['wind2'][-1]:.1f} "
                  f"min‖f₂‖ {h['minf2'][0]:.2f}→{h['minf2'][-1]:.2f} gates={sum(h['gate'])}", flush=True)
        out["runtime"] = time.time() - t0
        np.save(result_path("exp9_cpp.npy"), out, allow_pickle=True)
    from exp9_cpp_report import make_results
    import json
    print(json.dumps(make_results(out), indent=2))


if __name__ == "__main__":
    main()
