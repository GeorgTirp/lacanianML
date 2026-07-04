"""exp9 — the Ampère experiment (v7): the charge law tested quantitatively.

Parts: A charge ordering + slope (P-A1), B superposition/additivity (P-B2),
C charge decay true-vs-false lack (P-C1), D protection ordering (secondary).
K-charge gates everything. §1 discipline enforced: the RESULTS §1 table assigns
every result to T1–T4 and marks by-construction facts as such.
"""
import argparse
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from _exp_common import fig_path, ROOT, result_path
from winding.config import CFG
from winding import ampere as A
from winding.uncertainty import train_ensemble_regression
from winding.data import Embedding
from winding.losses import phase_of, gate_barrier
from winding.topology import ang_c, wrap, winding

PI = np.pi
KAPPA = 0.03            # drive scale, set once, shared across worlds (§4)


def charge_of(w, seed, n=3000, gn=80):
    reg = A.make_regression_data(w, CFG, seed=seed, n=n)
    models = train_ensemble_regression(reg, CFG, seed=seed)
    charged, mp = A.estimate_charges(models, w, CFG, gn=gn)
    return charged, mp, models


# --------------------------------------------------------------------------- #
#  Part A — charge ordering + slope (T1 + T2, band-deciding P-A1)              #
# --------------------------------------------------------------------------- #
def part_A(seeds):
    pts = []          # (log q-ratio, log rho-ratio)
    rows = []
    for w in A.default_worlds():
        for s in seeds:
            charged, mp, _ = charge_of(w, s)
            if len(charged) < 2:
                continue
            (c1, q1), (c2, q2) = charged
            oq1, oq2 = mp["oracle_q"]
            m = A.TwoHead(CFG); A.install(m, w, [c1, c2], CFG, steps=700, seed=s)
            wind = A.verify_winding(m, w, [c1, c2], CFG)
            res = A.idle_rates(m, w, [q1, q2], KAPPA, CFG, steps=1500, seed=s)
            rho = res["rho"]
            if q2 > 0 and rho[1] != 0:
                pts.append((np.log(q1 / q2), np.log(abs(rho[0] / rho[1]))))
            near = np.linalg.norm(c1 - np.array(w.c1)) < 0.6
            rows.append(dict(r2=w.r2, seed=s, q1=q1, q2=q2, qr=q1 / max(q2, 1e-9),
                             oqr=oq1 / max(oq2, 1e-9), rho=rho, wind=wind,
                             r2fit=res["r2"], near_A1=bool(near)))
    P = np.array(pts)
    slope, inter = np.polyfit(P[:, 0], P[:, 1], 1)
    r = float(np.corrcoef(P[:, 0], P[:, 1])[0, 1])
    return dict(pts=P.tolist(), slope=float(slope), r=r, rows=rows)


# --------------------------------------------------------------------------- #
#  Part B — superposition + deformation invariance (T2, band-deciding P-B2)    #
# --------------------------------------------------------------------------- #

def _realized_transport(model, loops, cfg=CFG):
    """∮ dφ_learned around each loop (units of 2π). Single shared head (index 0)."""
    emb = Embedding(cfg); out = []
    for p in loops:
        x = torch.tensor(emb(p, noise=0.0), dtype=torch.float32)
        with torch.no_grad():
            phi = A.phase(model, x, 0).numpy()
        d = wrap(np.diff(phi)); d = np.append(d, wrap(phi[0] - phi[-1]))
        gap_ok = bool(np.all(np.abs(d) < PI))
        out.append((float(np.sum(d) / (2 * PI)), gap_ok))
    return out


def part_B(seeds):
    res = {}
    for s in seeds:
        w = A.default_worlds()[1]
        charged, mp, _ = charge_of(w, s)
        (c1, q1), (c2, q2) = charged
        # ONE shared head carrying BOTH windings: install toward ang1+ang2
        m = A.TwoHead(CFG, n_heads=1)
        torch.manual_seed(s); np.random.seed(s)
        emb = Embedding(CFG); rng = np.random.default_rng(s)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for step in range(900):
            p = A.sample_support(96, w, rng)
            x = torch.tensor(emb(p, noise=CFG.obs_noise, rng=rng), dtype=torch.float32)
            tgt = torch.tensor(ang_c(p, c1) + ang_c(p, c2), dtype=torch.float32)
            phi = A.phase(m, x, 0)
            loss = torch.mean(1 - torch.cos(phi - tgt))
            if step >= 450:
                loss = loss + gate_barrier(m.f(x, 0), CFG.barrier_margin, CFG.lam_bar)
            opt.zero_grad(); loss.backward(); opt.step()
        fam = {}
        for kind in ("A1", "A2", "both", "neither"):
            vals = _realized_transport(m, A.deformed_loops(kind, w, seed=s), CFG)
            fam[kind] = [v for v, ok in vals]
        res[s] = fam
    return res


# --------------------------------------------------------------------------- #
#  Part C — charge decay, true vs false lack (T3, band-deciding P-C1)          #
# --------------------------------------------------------------------------- #
def part_C(seeds, refits=12, idle_per=400):
    out = {}
    for s in seeds:
        w = A.default_worlds()[1]
        rng = np.random.default_rng(s + 50)
        # initial charges (A2 empty)
        charged, mp, _ = charge_of(w, s)
        (c1, q1_0), (c2, q2_0) = charged
        m = A.TwoHead(CFG); A.install(m, w, [c1, c2], CFG, steps=700, seed=s)
        emb = Embedding(CFG)
        q2_series, rho2_series, rho1_series, minf2_series, gate_series, dens = [], [], [], [], [], []
        for step in range(refits):
            frac = step / (refits - 1)                         # A2 fill fraction
            # extra points inside A2, density rising to full support density
            n_extra = int(frac * 900)
            ang = rng.uniform(0, 2 * PI, n_extra); rr = w.r2 * np.sqrt(rng.uniform(0, 1, n_extra))
            extra = np.array(w.c2) + np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1)
            reg = A.make_regression_data(w, CFG, seed=s, n=2500, extra=extra if n_extra else None)
            models = train_ensemble_regression(reg, CFG, seed=s)
            _, mp2 = A.estimate_charges(models, w, CFG, gn=80)
            # robust online charge = disagreement integrated over the KNOWN disks
            # (data-measured; avoids fragile component matching as A2 heals)
            q1, q2 = mp2["oracle_q"]
            res = A.idle_rates(m, w, [q1, max(q2, 1e-6)], KAPPA, CFG, steps=idle_per, seed=s + step)
            with torch.no_grad():
                _, nrm = phase_of(m.f(A.probe_points(w, 1, CFG), 1))
            q2_series.append(q2); rho1_series.append(res["rho"][0]); rho2_series.append(res["rho"][1])
            minf2_series.append(float(nrm.min())); gate_series.append(int(nrm.min() < CFG.gate_thresh))
            dens.append(frac)
        out[s] = dict(q1_0=q1_0, q2_0=q2_0, q2=q2_series, rho1=rho1_series, rho2=rho2_series,
                      minf2=minf2_series, gate=gate_series, frac=dens)
    return out


# --------------------------------------------------------------------------- #
#  Part D — protection ordering (T4, secondary)                                #
# --------------------------------------------------------------------------- #
def part_D(seeds):
    rows = []
    for s in seeds:
        w = A.default_worlds()[1]
        charged, mp, _ = charge_of(w, s)
        (c1, q1), (c2, q2) = charged
        m = A.TwoHead(CFG); A.install(m, w, [c1, c2], CFG, steps=700, seed=s)
        emb = Embedding(CFG); rng = np.random.default_rng(s + 9)
        task = torch.nn.Linear(CFG.enc_hidden, 1)
        opt = torch.optim.Adam(list(m.parameters()) + list(task.parameters()), lr=1e-3)
        first_gate = [None, None]
        for step in range(600):
            p = A.sample_support(96, w, rng)
            x = torch.tensor(emb(p, noise=CFG.obs_noise, rng=rng), dtype=torch.float32)
            y = torch.tensor(A.target(p, w), dtype=torch.float32)[:, None]
            loss = ((task(m.trunk(x)) - y) ** 2).mean()      # interfering task
            opt.zero_grad(); loss.backward(); opt.step()      # task step first
            A.apply_charge_drive(m, x, [q1, q2], KAPPA, CFG.barrier_margin)  # then drive
            with torch.no_grad():
                for i in (0, 1):
                    _, nrm = phase_of(m.f(A.probe_points(w, i, CFG), i))
                    if first_gate[i] is None and float(nrm.min()) < CFG.gate_thresh:
                        first_gate[i] = step
        wind = A.verify_winding(m, w, [c1, c2], CFG)
        rows.append(dict(seed=s, q1=q1, q2=q2, first_gate=first_gate, wind=wind))
    return rows


# =========================================================================== #
def eu_map_figure(seed=0):
    w = A.default_worlds()[1]
    charged, mp, _ = charge_of(w, seed, gn=100)
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.pcolormesh(mp["axis"], mp["axis"], mp["dis"], shading="auto", cmap="magma")
    fig.colorbar(im, ax=ax, label="ensemble disagreement (σ_f²)")
    for c, r, lab in [(w.c1, w.r1, "A₁"), (w.c2, w.r2, "A₂")]:
        th = np.linspace(0, 2 * PI, 100)
        ax.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th), "w--", lw=1)
        ax.scatter(*c, marker="+", c="lime", s=120)
    for (c, q), col in zip(charged, ["red", "cyan"]):
        ax.scatter(*c, marker="x", c=col, s=140, label=f"ĉ q̂={q:.3f}")
    ax.set(title="v7 EU map: two data-lacks with different charge", xlabel="x", ylabel="y",
           aspect="equal"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_path("exp9_eu_map.png"), dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=str, default="A,B,C,D")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        out = np.load(result_path("exp9_data.npy"), allow_pickle=True).item()
        from exp9_report import make_figures_and_results
        import json
        print(json.dumps(make_figures_and_results(out), indent=2)); return
    seeds = [int(s) for s in a.seeds.split(",")]
    parts = a.parts.split(",")
    t0 = time.time()
    out = {"seeds": seeds}
    eu_map_figure(seeds[0])
    if "A" in parts: out["A"] = part_A(seeds)
    if "B" in parts: out["B"] = part_B(seeds)
    if "C" in parts: out["C"] = part_C(seeds)
    if "D" in parts: out["D"] = part_D(seeds)
    out["runtime"] = time.time() - t0
    np.save(result_path("exp9_data.npy"), out, allow_pickle=True)
    from exp9_report import make_figures_and_results
    import json
    print(json.dumps(make_figures_and_results(out), indent=2))


if __name__ == "__main__":
    main()
