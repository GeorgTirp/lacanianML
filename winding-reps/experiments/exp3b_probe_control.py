"""exp3b: probe-recovery control — the outstanding P5 gate.

v3 flagship: under interfering fine-tuning, arm A kept winding acc 1.000 while
baseline C fell to 0.444 (measured through C's FROZEN original head). Two
hypotheses fit that:
  H-forgot: fine-tuning destroyed the winding info in C's trunk.
  H-drift : the info survives in the trunk; only the frozen readout's access broke.

This control retrains fresh probes on C's fine-tuned trunk and computes the
recovery ratio  R = (P_ft - P_rand) / (P_pre - P_rand).  R>=0.8 => H-drift, P5
downgrades to "readout drifted"; R<=0.3 => H-forgot, P5 stands; between => graded.

Pre-registration LOCKED 2026-07-04 (§4), do not alter thresholds post hoc.
C-only rebuild (v3 did not persist checkpoints); no phase arms / EU / barrier.
Required data deviation: r0 ~ U[1.15,1.85] per loop (the exp2 default r=1.5 makes
Task-2 mean-radius degenerate/non-interfering — the tell: v3 P7 had radius acc 1.0
for every arm). Flagged in RESULTS.
"""
import argparse
import copy
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from _exp_common import fig_path, ROOT, result_path
from winding.config import CFG
from winding.data import (Embedding, _fourier_series, _rand_fourier,
                          radius_class_labels, start_sector_labels)
from winding.models import GRUBaseline

PI = math.pi
V3B_MARKER = "\n<!-- V3B SECTION -->\n"
_C2I = {-1: 0, 0: 1, 1: 2}


# --------------------------------------------------------------------------- #
#  v3b data — loops with per-loop radius spread (required deviation §2.1)      #
# --------------------------------------------------------------------------- #
def make_v3b_traj(cfg, seed, n_per_class):
    rng = np.random.default_rng(seed)
    emb = Embedding(cfg)
    t = np.arange(cfg.T) / cfg.T
    ps, ys = [], []
    for k in (-1, 0, 1):
        for _ in range(n_per_class):
            alpha0 = rng.uniform(0, 2 * PI)
            a_c = _rand_fourier(rng, cfg.n_fourier, cfg.noise_amp_alpha)
            r_c = _rand_fourier(rng, cfg.n_fourier, cfg.noise_amp_r)
            r0 = rng.uniform(1.15, 1.85)                      # <-- the deviation
            alpha = alpha0 + 2 * PI * k * t + _fourier_series(t, a_c)
            r = np.clip(r0 + _fourier_series(t, r_c), 1.05, 1.95)
            ps.append(np.stack([r * np.cos(alpha), r * np.sin(alpha)], 1))
            ys.append(k)
    p = np.stack(ps); y = np.array(ys, dtype=int)
    x = emb(p.reshape(-1, 2), noise=cfg.obs_noise, rng=rng).reshape(p.shape[0], cfg.T, cfg.D)
    return dict(p=p, x=torch.tensor(x, dtype=torch.float32), y=y,
               yidx=torch.tensor([_C2I[int(v)] for v in y], dtype=torch.long))


# --------------------------------------------------------------------------- #
#  arm C: GRU trunk, train winding to the >=0.99 gate                          #
# --------------------------------------------------------------------------- #
def train_C(cfg, tr, te, seed, max_steps=3000):
    torch.manual_seed(seed); np.random.seed(seed)
    net = GRUBaseline(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    N = tr["x"].shape[0]
    for step in range(max_steps):
        idx = np.random.randint(0, N, 64)
        loss = lossf(net(tr["x"][idx]), tr["yidx"][idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                acc = float((net(te["x"]).argmax(1) == te["yidx"]).float().mean())
            if acc >= 0.995:
                break
    with torch.no_grad():
        acc = float((net(te["x"]).argmax(1) == te["yidx"]).float().mean())
    return net, acc


# --------------------------------------------------------------------------- #
#  interfering fine-tuning (exact v3 protocol) + frozen-head retention curve   #
# --------------------------------------------------------------------------- #
def finetune_C(net, cfg, ft, wind_te, seed):
    torch.manual_seed(seed + 777); np.random.seed(seed + 777)
    for p in net.head.parameters():                          # freeze original head
        p.requires_grad_(False)
    x = ft["x"]; N = x.shape[0]
    tasks = [("radius", torch.tensor(radius_class_labels(ft["p"], cfg), dtype=torch.long), 3),
             ("sector", torch.tensor(start_sector_labels(ft["p"], cfg), dtype=torch.long), cfg.n_sectors)]
    ret_steps, ret_acc = [], []
    g = 0
    for tname, lab, ncls in tasks:
        head = nn.Sequential(nn.Linear(cfg.gru_hidden, 32), nn.ReLU(), nn.Linear(32, ncls))
        opt = torch.optim.Adam(list(net.gru.parameters()) + list(head.parameters()), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        for step in range(1500):
            idx = np.random.randint(0, N, 64)
            feats = net.trunk_features(x[idx]).mean(1)        # mean-pooled hidden
            loss = lossf(head(feats), lab[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            g += 1
            if step % 25 == 0:
                with torch.no_grad():
                    ret_acc.append(float((net(wind_te["x"]).argmax(1) == wind_te["yidx"]).float().mean()))
                    ret_steps.append(g)
                    tacc = float((head(net.trunk_features(x[:300]).mean(1)).argmax(1) == lab[:300]).float().mean())
                if tacc >= 0.95:
                    break
    with torch.no_grad():
        frozen_final = float((net(wind_te["x"]).argmax(1) == wind_te["yidx"]).float().mean())
    return dict(ret_steps=ret_steps, ret_acc=ret_acc, frozen_final=frozen_final)


# --------------------------------------------------------------------------- #
#  probes                                                                     #
# --------------------------------------------------------------------------- #
def cache_feats(net, x):
    with torch.no_grad():
        h = net.trunk_features(x)                              # (N,T,H)
    return dict(final=h[:, -1, :], mean=h.mean(1), concat=torch.cat([h[:, -1, :], h.mean(1)], 1))


def train_probe(ftr, ytr, fte, yte, in_dim, mlp=False, seed=0, steps=2000):
    torch.manual_seed(seed)
    head = (nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 3))
            if mlp else nn.Linear(in_dim, 3))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(); N = ftr.shape[0]
    for _ in range(steps):
        idx = np.random.randint(0, N, 64)
        loss = lossf(head(ftr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((head(fte).argmax(1) == yte).float().mean())


def probe_all(net, cfg, ptr, pte, seed):
    ftr = cache_feats(net, ptr["x"]); fte = cache_feats(net, pte["x"])
    ytr, yte = ptr["yidx"], pte["yidx"]
    H = cfg.gru_hidden
    return dict(
        primary=train_probe(ftr["final"], ytr, fte["final"], yte, H, seed=seed),
        secondary=train_probe(ftr["mean"], ytr, fte["mean"], yte, H, seed=seed),
        exploratory=train_probe(ftr["concat"], ytr, fte["concat"], yte, 2 * H, mlp=True, seed=seed),
    )


# --------------------------------------------------------------------------- #
def run_seed(cfg, seed):
    tr = make_v3b_traj(cfg, seed, 200)                        # C winding train (600)
    te = make_v3b_traj(cfg, seed + 1, 100)                    # C winding test  (300)
    ft = make_v3b_traj(cfg, seed + 2, 200)                    # fine-tuning set
    ptr = make_v3b_traj(cfg, seed + 3, 200)                   # FRESH probe train
    pte = make_v3b_traj(cfg, seed + 4, 100)                   # FRESH probe test

    net, clean = train_C(cfg, tr, te, seed)
    net_pre = copy.deepcopy(net)                              # T_pre
    ftlog = finetune_C(net, cfg, ft, te, seed)                # net trunk -> T_ft
    net_rand = GRUBaseline(cfg); torch.manual_seed(seed + 999)  # T_rand fresh init
    for m in net_rand.modules():
        if isinstance(m, (nn.Linear, nn.GRU)):
            for p in m.parameters():
                nn.init.normal_(p, std=0.1)

    P_pre = probe_all(net_pre, cfg, ptr, pte, seed)
    P_ft = probe_all(net, cfg, ptr, pte, seed)
    P_rand = probe_all(net_rand, cfg, ptr, pte, seed)

    def R(fam):
        num = P_ft[fam] - P_rand[fam]; den = P_pre[fam] - P_rand[fam]
        return float(num / den) if abs(den) > 1e-6 else float("nan")

    return dict(seed=seed, clean=clean, frozen_final=ftlog["frozen_final"],
                ret_steps=ftlog["ret_steps"], ret_acc=ftlog["ret_acc"],
                P_pre=P_pre, P_ft=P_ft, P_rand=P_rand,
                R_primary=R("primary"), R_secondary=R("secondary"),
                task2_balance=np.bincount(radius_class_labels(ft["p"], cfg), minlength=3).tolist())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    cfg = CFG
    t0 = time.time()
    per = [run_seed(cfg, s) for s in seeds]
    for r in per:
        print(f"[seed {r['seed']}] clean={r['clean']:.3f} frozen_final={r['frozen_final']:.3f} "
              f"P_pre={r['P_pre']['primary']:.2f} P_ft={r['P_ft']['primary']:.2f} "
              f"P_rand={r['P_rand']['primary']:.2f} R={r['R_primary']:.2f}")
    out = dict(per=per, runtime=time.time() - t0, seeds=seeds)
    np.save(result_path("exp3b_data.npy"), out, allow_pickle=True)
    _figures(out)
    print(json.dumps(write_results(out, cfg), indent=2))


# --------------------------------------------------------------------------- #
def _figures(out):
    per = out["per"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for r in per:
        ax.plot(r["ret_steps"], r["ret_acc"], label=f"seed {r['seed']}")
    ax.axhline(1 / 3, ls=":", color="gray", label="chance"); ax.axhline(0.7, ls="--", color="red", label="repro gate")
    ax.set(title="v3b: C frozen-head winding accuracy during interfering fine-tuning",
           xlabel="fine-tune step", ylabel="winding acc", ylim=(0, 1.05)); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_path("exp3b_retention.png"), dpi=120); plt.close(fig)

    fams = ["primary", "secondary", "exploratory"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for j, fam in enumerate(fams):
        pre = np.mean([r["P_pre"][fam] for r in per]); ft = np.mean([r["P_ft"][fam] for r in per])
        rnd = np.mean([r["P_rand"][fam] for r in per])
        ax[j].bar([0, 1, 2], [pre, ft, rnd], color=["C2", "C0", "C7"])
        ax[j].axhline(1 / 3, ls=":", color="gray")
        ax[j].set(title=f"{fam}", xticks=[0, 1, 2], ylim=(0, 1.05))
        ax[j].set_xticklabels(["P_pre", "P_ft", "P_rand"])
        for i, v in enumerate([pre, ft, rnd]):
            ax[j].text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax[0].set_ylabel("held-out winding acc")
    fig.suptitle("v3b probe recovery: winding decodable from each trunk (3-seed mean)")
    fig.tight_layout(); fig.savefig(fig_path("exp3b_probes.png"), dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
def write_results(out, cfg):
    per = out["per"]; seeds = out["seeds"]
    frozen = np.array([r["frozen_final"] for r in per])
    Ppre = np.array([r["P_pre"]["primary"] for r in per])
    Pft = np.array([r["P_ft"]["primary"] for r in per])
    Prand = np.array([r["P_rand"]["primary"] for r in per])
    Rp = np.array([r["R_primary"] for r in per])
    R_mean = float(np.nanmean(Rp))

    repro_ok = bool(np.all(frozen < 0.7))
    sanity_ok = bool(np.all(Ppre >= 0.9))
    confound = bool(np.any(Prand > 0.8))
    # confound clause: if primary P_rand>0.8, shift to secondary
    fam = "primary"
    if confound:
        Prand_b = np.array([r["P_rand"]["secondary"] for r in per])
        if np.all(Prand_b <= 0.8):
            fam = "secondary"; Rp = np.array([r["R_secondary"] for r in per]); R_mean = float(np.nanmean(Rp))

    if R_mean >= 0.8:
        band = "R>=0.8 -> INFORMATION RETAINED: P5 downgrades to 'C's readout drifted'"
    elif R_mean <= 0.3:
        band = "R<=0.3 -> TRUNK LOST THE INVARIANT: P5 stands at full strength"
    else:
        band = "0.3<R<0.8 -> PARTIAL EROSION: P5 restated as graded trunk degradation"

    L = [V3B_MARKER, "# v3b — probe-recovery control (the outstanding P5 gate)\n",
         f"Seeds: {seeds}. Runtime: {out['runtime']:.0f}s ({out['runtime']/60:.1f} min) CPU. "
         f"Pre-registration LOCKED 2026-07-04.\n",
         "\n**Required data deviation (flagged):** loop radius r0 ~ U[1.15,1.85] per loop "
         "(exp2 default r=1.5 makes Task-2 mean-radius degenerate/non-interfering — the tell is "
         "v3 P7's radius acc 1.0 for every arm). Task-2 class balance (seed0): "
         f"{per[0]['task2_balance']} — all three classes populated.\n"]

    # gates first
    L.append("\n## Gates\n")
    L.append(f"- **Reproduction gate** (frozen-head winding acc after Task 3 must be <0.7): "
             f"frozen_final per seed = {np.round(frozen,3).tolist()} → "
             f"{'PASS — forgetting reproduced' if repro_ok else 'FAIL — forgetting did NOT reproduce; interpretation halted per §4'}.\n")
    L.append(f"- **Sanity anchor** (P_pre ≥ 0.9, else pipeline broken): P_pre per seed = "
             f"{np.round(Ppre,3).tolist()} → {'PASS' if sanity_ok else 'FAIL — probe pipeline broken, fix before interpreting'}.\n")
    L.append(f"- **Confound clause** (P_rand > 0.8 under primary → shift to secondary): "
             f"P_rand(primary) per seed = {np.round(Prand,3).tolist()} → "
             f"{'FIRED, using ' + fam if confound else 'clear'}.\n")

    if not repro_ok:
        L.append("\n> **HALT (§4):** the forgetting did not reproduce (frozen-head acc ≥ 0.7). This "
                 "control is inconclusive about P5 by its own pre-registered rule; no downgrade or "
                 "confirmation is claimed. Reported plainly.\n")
    elif not sanity_ok:
        L.append("\n> **BUG GATE (§4):** P_pre < 0.9 — the probe pipeline cannot recover winding even "
                 "from the pre-fine-tuning trunk where the info is demonstrably present. Fix the "
                 "pipeline before interpreting R.\n")
    else:
        L.append(f"\n## Verdict — {band}\n")
        L.append(f"> Recovery ratio **R = {R_mean:.2f}** (primary family '{fam}', per seed "
                 f"{np.round(Rp,2).tolist()}). "
                 f"P_pre={Ppre.mean():.2f} (info present), P_ft={Pft.mean():.2f} "
                 f"(fine-tuned trunk), P_rand={Prand.mean():.2f} (random-feature reservoir). "
                 f"Chance = 0.33.\n")
        if R_mean >= 0.8:
            L.append("> \n> **P5 CLAIM UPDATE (integrity rule 3):** the flagship statement 'C forgot "
                     "the invariant' is **retracted as stated** — the winding information SURVIVES in "
                     "C's fine-tuned trunk; only the frozen readout's access drifted. The surviving "
                     "contrast: recovery required fresh labeled data + optimization, whereas arm A's "
                     "parameter-free winding readout needed nothing and showed ZERO erosion. P5 is "
                     "restated as *readout-access* protection, not information protection.\n")
        elif R_mean <= 0.3:
            L.append("> \n> **P5 stands at full strength:** interfering fine-tuning erased the winding "
                     "invariant from C's trunk (a fresh, maximally-favourable probe cannot recover it "
                     "above the random-feature reservoir), while arm A retained it exactly. H-forgot "
                     "confirmed over H-drift.\n")
        else:
            L.append("> \n> **P5 restated (graded):** the trunk partially eroded the invariant; the "
                     "contrast with A's *zero* erosion survives but the 'C forgot entirely' framing is "
                     "weakened to graded degradation.\n")

    L.append("\n## All probe accuracies (held-out winding, 3-seed mean)\n")
    L.append("| trunk | primary (final-hidden) | secondary (mean-pool) | exploratory (MLP) |")
    L.append("|---|---|---|---|")
    for name, key in [("T_pre (upper anchor)", "P_pre"), ("T_ft (fine-tuned)", "P_ft"),
                      ("T_rand (reservoir)", "P_rand")]:
        L.append(f"| {name} | " + " | ".join(
            f"{np.mean([r[key][f] for r in per]):.3f}" for f in ["primary", "secondary", "exploratory"]) + " |")
    expl_rand = float(np.mean([r["P_rand"]["exploratory"] for r in per]))
    if expl_rand > 0.8:
        L.append(f"\n> **Exploratory family is NON-DIAGNOSTIC (reservoir confound realized).** The "
                 f"generous MLP probe reaches P_rand={expl_rand:.3f} on the RANDOM trunk — a random "
                 "GRU's features over the T×64 sequence already support near-perfect winding decoding "
                 "with a nonlinear reader. So its high T_ft recovery proves nothing about retention. "
                 "This is exactly why the band is decided by the LINEAR primary probe (reservoir "
                 f"P_rand={Prand.mean():.2f}, well clear of confound). The exploratory row confirms the "
                 "invariant is nonlinearly *present* in T_ft, but that reading cannot be separated from "
                 "the reservoir; the honest, confound-free verdict is the linear one.\n")

    L.append("\n## What would change our mind\n")
    L.append(
        "- If a *deeper* probe (family c) recovered winding from T_ft where the primary did not, the "
        "invariant is present but nonlinearly encoded — H-drift with a caveat on readout complexity.\n"
        "- If P_rand were > 0.8 (random-GRU reservoir already decodes winding over the T×64 sequence), "
        "the control is non-diagnostic — a fresh probe's success would prove nothing about retention.\n"
        "- If the reproduction gate had failed (frozen acc ≥ 0.7), the whole comparison is moot: no "
        "forgetting to explain.\n"
        "- The residual contrast is robust regardless of R: A's readout is parameter-free and needed no "
        "data; any C recovery needs labels + optimization. R only decides whether the words 'C forgot' "
        "are literally true of the trunk.\n")

    L.append("\n## Figures\n")
    for f, cap in [("exp3b_retention.png", "C frozen-head winding accuracy collapsing during fine-tuning."),
                   ("exp3b_probes.png", "Winding recoverable from each trunk by fresh probes (P_pre/P_ft/P_rand).")]:
        L.append(f"![{cap}](results/figures/{f})\n\n*{cap}*\n")

    txt = "\n".join(L) + "\n"
    path = f"{ROOT}/RESULTS.md"
    with open(path) as fh:
        base = fh.read().split(V3B_MARKER)[0].rstrip() + "\n"
    with open(path, "w") as fh:
        fh.write(base + txt)
    print("wrote v3b section to RESULTS.md")
    return dict(repro_ok=repro_ok, sanity_ok=sanity_ok, confound=confound,
                R=R_mean, band=band, family=fam)


if __name__ == "__main__":
    main()
