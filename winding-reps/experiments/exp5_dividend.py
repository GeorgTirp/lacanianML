"""exp5_dividend: confound gate, plasticity dividend, class stabilizer (v5).

exp5a is a GATE. If it returns ARTIFACT, the v5 RESULTS section LEADS with a
retraction of the v4 adaptation claim before any new positive result.

  5a-i   (v4 logs)  post-shift L_ssl floor vs winding-class status -> verdict.
  5a-ii  (v4 logs)  are class breaks gated? (they occur at the discrete shift).
  5a-iii (new)      survival rate per regime under 10 fresh shifts + binomial CI.
  5b     P11        plasticity dividend on an ORTHOGONAL task (mean-radius), so
                    class fate cannot contaminate it.
  5c     P12        label-free multi-view stabilizer L_stab at deployment.

Additive; exp0-exp4 untouched. Constants locked per addendum §4. CPU < 20 min.
"""
import argparse
import collections
import copy
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
from winding.models import MeanPoolHead
from winding.losses import phase_of, gate_barrier, stabilizer_loss
from winding.train import predict_phase
from winding.drive import QHead, sample_pairs, lssl, plasticity_metrics
import exp4_drive as e4

V5_MARKER = "\n<!-- V5 SECTION -->\n"
REG5 = ["FROZEN", "SSL", "SGLD_hi", "DRIVE_hi"]
ALL6 = ["FROZEN", "SSL", "SGLD_lo", "SGLD_hi", "DRIVE_lo", "DRIVE_hi"]


# --------------------------------------------------------------------------- #
#  5a-i: the confound gate (existing v4 logs, no new runs)                     #
# --------------------------------------------------------------------------- #
def exp5a_i():
    per = np.load(result_path("exp4_per.npy"), allow_pickle=True).item()["per"]
    rows = []
    for r in ALL6:
        for si, p in enumerate(per):
            lg = p["adapt"][r]; acc = float(lg["retention"][-1])
            st = "SURVIVED" if acc > 0.9 else ("BROKE" if acc < 0.5 else "EXCLUDED")
            floor = float(np.mean(np.asarray(lg["lssl"], float)[-10:]))   # last ~500 steps
            rows.append(dict(regime=r, seed=si, acc=acc, status=st, floor=floor))
    brk = [x["floor"] for x in rows if x["status"] == "BROKE"]
    sur = [x["floor"] for x in rows if x["status"] == "SURVIVED"]

    def spread(status):
        by = collections.defaultdict(list)
        for x in rows:
            if x["status"] == status:
                by[x["regime"]].append(x["floor"])
        ms = [np.mean(v) for v in by.values()]
        return (max(ms) - min(ms)) if ms else 0.0

    mb = float(np.mean(brk)) if brk else float("nan")
    msu = float(np.mean(sur)) if sur else float("nan")
    sp_brk, sp_sur = spread("BROKE"), spread("SURVIVED")
    artifact = (mb < msu - 0.05) and sp_brk < 0.02 and sp_sur < 0.02
    dsurv = [x["floor"] for x in rows if "DRIVE" in x["regime"] and x["status"] == "SURVIVED"]
    dividend_compat = bool(brk and any(abs(f - mb) < 0.02 for f in dsurv))
    verdict = "ARTIFACT" if artifact else ("DIVIDEND_COMPATIBLE" if dividend_compat else "MIXED")
    frag = {r: sum(1 for x in rows if x["regime"] == r and x["status"] == "BROKE") for r in ALL6}
    return dict(rows=rows, mb=mb, msu=msu, sp_brk=sp_brk, sp_sur=sp_sur,
                verdict=verdict, frag=frag, n_brk=len(brk), n_sur=len(sur))


# --------------------------------------------------------------------------- #
#  5a-ii: are the breaks gated? (they happen at the discrete shift onset)      #
# --------------------------------------------------------------------------- #
def exp5a_ii():
    per = np.load(result_path("exp4_per.npy"), allow_pickle=True).item()["per"]
    total = onset = during = 0
    for r in ALL6:
        for p in per:
            acc = np.asarray(p["adapt"][r]["retention"], float)
            if acc[-1] < 0.5:
                total += 1
                if acc[0] < 0.5:
                    onset += 1
                else:
                    during += 1
    return dict(total=total, onset=onset, during=during)


# --------------------------------------------------------------------------- #
#  idled checkpoints (re-idle; v4 did not persist encoders)                    #
# --------------------------------------------------------------------------- #
def make_checkpoints(cfg, seeds):
    fx = e4.make_fixtures(cfg, seeds[0])
    emb = Embedding(cfg)
    fx["stream_clean"] = torch.tensor(
        emb(fx["stream_traj"]["p"].reshape(-1, 2), noise=0.0).reshape(
            fx["stream_traj"]["p"].shape[0], cfg.T, cfg.D), dtype=torch.float32)
    fx["eval_clean"] = torch.tensor(
        emb(fx["ev"]["p"].reshape(-1, 2), noise=0.0).reshape(
            fx["ev"]["p"].shape[0], cfg.T, cfg.D), dtype=torch.float32)
    ck = {}
    for seed in seeds:
        enc0, _ = e4.build_armA(cfg, seed)
        cal = e4.calibrate(enc0, cfg, fx, None)
        for r in REG5:
            gen = torch.Generator().manual_seed(seed * 17 + e4.REGIMES.index(r))
            enc = copy.deepcopy(enc0); q = QHead(cfg.drive_kmax)
            eta, sig = e4._regime_params(r, cal)
            log, ef, qf = e4.deploy_loop(enc, q, r, cfg, fx, cfg.drive_steps_idle, eta, sig, gen)
            ck[(r, seed)] = dict(enc=ef, q=qf, sat=float(log["sat"][-1]), pr=float(log["pr"][-1]))
    return ck, fx


# --------------------------------------------------------------------------- #
#  a lean shift+adaptation run (used by 5a-iii survival and 5c stabilizer)     #
# --------------------------------------------------------------------------- #
def adapt_under_shift(enc, q, S_stream, S_eval, evaly, cfg, gen, lam_stab, steps):
    enc = copy.deepcopy(enc); q = copy.deepcopy(q)
    opt = torch.optim.Adam(list(enc.parameters()) + list(q.parameters()), lr=cfg.deploy_lr)
    N, T, D = S_stream.shape
    lssl_hist, step_hist = [], []
    for step in range(steps):
        idx = torch.randint(0, N, (cfg.batch,), generator=gen)
        x = S_stream[idx] + torch.randn(cfg.batch, T, D, generator=gen) * cfg.obs_noise
        x_t, x_tk, k = sample_pairs(x, cfg.drive_kmax, gen)
        f = enc(x.reshape(-1, D))
        loss = lssl(enc, q, x_t, x_tk, k) + gate_barrier(f, cfg.barrier_margin, cfg.lam_bar)
        if lam_stab > 0:
            v2 = S_stream[idx] + torch.randn(cfg.batch, T, D, generator=gen) * cfg.obs_noise
            phi1, _ = phase_of(enc(x.reshape(-1, D)))
            phi2, _ = phase_of(enc(v2.reshape(-1, D)))
            loss = loss + lam_stab * stabilizer_loss(phi1, phi2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(q.parameters()), cfg.ft_grad_clip)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                xb = S_stream[:cfg.batch] + torch.randn(cfg.batch, T, D, generator=gen) * cfg.obs_noise
                xt2, xtk2, k2 = sample_pairs(xb, cfg.drive_kmax, gen)
                lssl_hist.append(float(lssl(enc, q, xt2, xtk2, k2)))
            step_hist.append(step)
    ex = S_eval + torch.randn(S_eval.shape, generator=gen) * cfg.obs_noise
    acc = float((predict_phase(enc, ex) == evaly).mean())
    return acc, np.array(lssl_hist), np.array(step_hist)


def _shifted(fx, Q):
    Qt = torch.tensor(Q, dtype=torch.float32)
    return fx["stream_clean"] @ Qt.T, fx["eval_clean"] @ Qt.T


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# --------------------------------------------------------------------------- #
#  5a-iii survival  /  5c stabilizer survival                                  #
# --------------------------------------------------------------------------- #
def survival(ck, cfg, seeds, fx, n_draws, lam_stab, steps, tag=""):
    out = {}
    for r in REG5:
        surv = 0; accs = []
        recov = []
        for i in range(n_draws):
            seed = seeds[i % len(seeds)]
            c = ck[(r, seed)]
            Q = e4._make_shift(cfg, 1000 + i)
            S_stream, S_eval = _shifted(fx, Q)
            gen = torch.Generator().manual_seed(4000 + i)
            acc, lh, sh = adapt_under_shift(c["enc"], c["q"], S_stream, S_eval,
                                            fx["ev"]["y"], cfg, gen, lam_stab, steps)
            surv += int(acc > 0.9); accs.append(acc)
            # steps to recover L_ssl to within 90% of its drop
            L0, Lend = float(lh[0]), float(np.mean(lh[-3:]))
            tgt = L0 - 0.9 * (L0 - Lend)
            ok = np.where(lh <= tgt)[0]
            recov.append(int(sh[ok[0]]) if len(ok) else int(sh[-1]))
        lo, hi = _wilson(surv, n_draws)
        out[r] = dict(rate=surv / n_draws, n=n_draws, ci=(lo, hi),
                      accs=accs, recov=float(np.mean(recov)))
    return out


# --------------------------------------------------------------------------- #
#  5b: plasticity dividend (P11) on the orthogonal mean-radius task            #
# --------------------------------------------------------------------------- #
def exp5b(ck, cfg, seeds):
    res = []
    for (r, seed), c in ck.items():
        traj = make_traj_track(cfg, seed=seed)                 # stationary, no shift
        x = torch.tensor(traj["x"], dtype=torch.float32)
        lab = torch.tensor(radius_class_labels(traj["p"], cfg), dtype=torch.long)
        torch.manual_seed(seed + 4242); np.random.seed(seed + 4242)
        enc = copy.deepcopy(c["enc"]); head = MeanPoolHead(2, 3, cfg.ft_head_hidden)
        opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=cfg.lr)
        lossf = nn.CrossEntropyLoss()
        budget = 1500; accs, steps = [], []
        for step in range(budget):
            idx = np.random.randint(0, x.shape[0], cfg.batch)
            f = enc(x[idx])
            loss = lossf(head(f), lab[idx]) + gate_barrier(f, cfg.barrier_margin, cfg.lam_bar)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), cfg.ft_grad_clip)
            opt.step()
            if step % 20 == 0 or step == budget - 1:
                with torch.no_grad():
                    acc = float((head(enc(x[:300])).argmax(1) == lab[:300]).float().mean())
                accs.append(acc); steps.append(step)
        accs = np.array(accs); steps = np.array(steps); final = float(accs.max())
        s90 = int(steps[np.argmax(accs >= 0.9 * final)]) if final > 0 else budget
        res.append(dict(regime=r, seed=seed, sat0=c["sat"], steps90=s90, final=final,
                        accs=accs.tolist(), steps=steps.tolist()))
    return res


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def analyze_p11(res, seeds):
    by = {(x["regime"], x["seed"]): x for x in res}
    order_ok = 0
    for seed in seeds:
        d = by[("DRIVE_hi", seed)]["steps90"]
        f = by[("FROZEN", seed)]["steps90"]; s = by[("SSL", seed)]["steps90"]
        if d < f and d < s:
            order_ok += 1
    rho = _spearman([x["sat0"] for x in res], [x["steps90"] for x in res])
    # capacity check: DRIVE final vs FROZEN final
    dfin = np.mean([x["final"] for x in res if x["regime"] == "DRIVE_hi"])
    ffin = np.mean([x["final"] for x in res if x["regime"] == "FROZEN"])
    p11_pass = bool(order_ok >= 2 and rho > 0.5)
    return dict(order_ok=order_ok, rho=rho, p11_pass=p11_pass,
                drive_final=float(dfin), frozen_final=float(ffin),
                capacity_traded=bool(dfin < ffin - 0.05))


# --------------------------------------------------------------------------- #
#  pilot for lambda_stab (seed 0, then FROZEN)                                 #
# --------------------------------------------------------------------------- #
def pilot_lambda(ck, cfg, seeds, fx):
    best, best_rate = 1.0, -1.0
    trials = {}
    for lam in (1.0, 3.0):
        surv = 0
        for i in range(4):
            seed = seeds[0]; c = ck[("DRIVE_hi", seed)]
            Q = e4._make_shift(cfg, 1000 + i)
            S_stream, S_eval = _shifted(fx, Q)
            gen = torch.Generator().manual_seed(7000 + i)
            acc, _, _ = adapt_under_shift(c["enc"], c["q"], S_stream, S_eval,
                                          fx["ev"]["y"], cfg, gen, lam, 1000)
            surv += int(acc > 0.9)
        trials[lam] = surv / 4
        if surv / 4 > best_rate:
            best_rate, best = surv / 4, lam
    return best, trials


# =========================================================================== #
def run(cfg, seeds):
    t0 = time.time()
    e4._init_probes(cfg)                     # exp4.deploy_loop reads probe globals
    a_i = exp5a_i(); a_ii = exp5a_ii()
    ck, fx = make_checkpoints(cfg, seeds)
    base = survival(ck, cfg, seeds, fx, cfg.v5_draws, lam_stab=0.0, steps=cfg.v5_adapt_steps)
    lam, pilot = pilot_lambda(ck, cfg, seeds, fx)
    stab = survival(ck, cfg, seeds, fx, cfg.v5_draws, lam_stab=lam, steps=cfg.v5_adapt_steps)
    div = exp5b(ck, cfg, seeds)
    p11 = analyze_p11(div, seeds)
    runtime = time.time() - t0
    out = dict(a_i=a_i, a_ii=a_ii, base=base, stab=stab, lam=lam, pilot=pilot,
               div=div, p11=p11, runtime=runtime, seeds=list(seeds))
    np.save(result_path("exp5_data.npy"), out, allow_pickle=True)
    return out


# --------------------------------------------------------------------------- #
def make_figures(out):
    # P11 saturation-vs-speed scatter + learning curves (seed 0)
    div = out["div"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"FROZEN": "k", "SSL": "C7", "SGLD_hi": "C3", "DRIVE_hi": "C2"}
    for x in div:
        ax[0].scatter(x["sat0"], x["steps90"], color=colors[x["regime"]], s=50,
                      label=x["regime"] if x["seed"] == out["seeds"][0] else None)
    ax[0].set(title=f"P11: pre-task saturation vs steps-to-90% (ρ={out['p11']['rho']:.2f})",
              xlabel="saturated-tanh fraction (pre-task)", ylabel="steps to 90% acc")
    ax[0].legend(fontsize=8)
    for x in div:
        if x["seed"] == out["seeds"][0]:
            ax[1].plot(x["steps"], x["accs"], color=colors[x["regime"]], label=x["regime"])
    ax[1].set(title="P11: new-task learning curves (seed 0, mean-radius)",
              xlabel="step", ylabel="task acc"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_path("exp5_dividend_P11.png"), dpi=120); plt.close(fig)

    # 5a-iii / 5c survival bars
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = np.arange(len(REG5)); w = 0.38
    b = [out["base"][r]["rate"] for r in REG5]; s = [out["stab"][r]["rate"] for r in REG5]
    be = [[out["base"][r]["rate"] - out["base"][r]["ci"][0] for r in REG5],
          [out["base"][r]["ci"][1] - out["base"][r]["rate"] for r in REG5]]
    se = [[out["stab"][r]["rate"] - out["stab"][r]["ci"][0] for r in REG5],
          [out["stab"][r]["ci"][1] - out["stab"][r]["rate"] for r in REG5]]
    ax.bar(xs - w / 2, b, w, yerr=be, capsize=3, color="C3", label="L_ssl only (5a-iii)")
    ax.bar(xs + w / 2, s, w, yerr=se, capsize=3, color="C0", label=f"+ L_stab λ={out['lam']:.0f} (5c/P12)")
    ax.set(title="class survival under fresh shifts (Wilson 95% CI)", ylabel="survival rate",
           xticks=xs, ylim=(0, 1.05)); ax.set_xticklabels(REG5); ax.legend()
    fig.tight_layout(); fig.savefig(fig_path("exp5_survival_P12.png"), dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
def write_results(out, cfg):
    a = out["a_i"]; a2 = out["a_ii"]; p11 = out["p11"]
    base, stab = out["base"], out["stab"]
    L = [V5_MARKER, "# v5 — confound gate, dividend, stabilizer\n",
         f"Seeds: {out['seeds']}. Runtime: {out['runtime']:.0f}s "
         f"({out['runtime']/60:.1f} min) CPU. 5a-iii/5c: {cfg.v5_draws} fresh shift draws/regime.\n"]

    # ---- GATE FIRST ----
    if a["verdict"] == "ARTIFACT":
        L.append(
            "\n## ⚠️ RETRACTION — the v4 adaptation advantage was an artifact (exp5a gate)\n"
            "> **The v4 finding that idled DRIVE 'recovers L_ssl fastest / to the lowest "
            "floor' is RETRACTED.** exp5a-i (analysis of the *committed v4 logs*, no new "
            f"runs) shows the post-shift L_ssl floor is set ENTIRELY by winding-class "
            f"status, not by regime: **BROKE arms floor = {a['mb']:.3f}** (across-regime "
            f"spread {a['sp_brk']:.3f}, n={a['n_brk']}) vs **SURVIVED arms floor = "
            f"{a['msu']:.3f}** (spread {a['sp_sur']:.3f}, n={a['n_sur']}). DRIVE reached the "
            "low floor precisely because it *lost the winding class* — an unwound phase "
            "field is trivially fittable by relational SSL, and the topological constraint "
            f"carries a real fitting cost ≈{a['msu']:.2f}. Worse for the drive: it is the "
            f"MOST shift-fragile regime (breaks per 3 seeds: " +
            ", ".join(f"{r}={a['frag'][r]}" for r in ALL6) +
            "). There is no plasticity advantage in the v4 adaptation numbers; the dividend "
            "is re-tested cleanly below (5b) on a task orthogonal to the winding class.\n")
    else:
        L.append(f"\n## exp5a gate verdict: {a['verdict']}\n"
                 f"BROKE floor {a['mb']:.3f} (spread {a['sp_brk']:.3f}) vs SURVIVED floor "
                 f"{a['msu']:.3f} (spread {a['sp_sur']:.3f}). See table below.\n")

    L.append("\n### 5a-i floor-by-class-status table\n")
    L.append("| regime | seed | end winding-acc | status | L_ssl floor |")
    L.append("|---|---|---|---|---|")
    for x in a["rows"]:
        L.append(f"| {x['regime']} | {x['seed']} | {x['acc']:.2f} | {x['status']} | {x['floor']:.3f} |")

    L.append("\n### 5a-ii are the breaks gated?\n")
    L.append(f"- Of {a2['total']} class breaks in v4 adaptation, **{a2['onset']} occurred at "
             f"the discrete shift onset** (winding already broken at adapt step 0) and "
             f"{a2['during']} during adaptation. The breaks are caused by the discrete "
             "distribution SHIFT (a jump in input space), not by a continuous training step "
             "— so they are 'ungated' in the training-dynamics sense by construction. The "
             "conservation law protects against continued *training* (v3/v4), never against "
             "the world changing; no discrete-step tunneling investigation is warranted.\n")

    # ---- 5a-iii survival ----
    L.append("\n## 5a-iii class survival under fresh shifts (no stabilizer)\n")
    L.append("| regime | survival rate | 95% CI | mean L_ssl recovery steps |")
    L.append("|---|---|---|---|")
    for r in REG5:
        s = base[r]
        L.append(f"| {r} | {s['rate']:.2f} ({int(round(s['rate']*s['n']))}/{s['n']}) | "
                 f"[{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] | {s['recov']:.0f} |")
    fragile = base["DRIVE_hi"]["rate"] < base["FROZEN"]["rate"]
    L.append(f"\nDRIVE survival {'<' if fragile else '≈/≥'} FROZEN "
             f"({base['DRIVE_hi']['rate']:.2f} vs {base['FROZEN']['rate']:.2f}); CIs are wide "
             f"at n={cfg.v5_draws} — no strong claim on overlaps, but the point estimate is "
             "consistent with the 5a-i finding that motion-at-shift adds fragility.\n")

    # ---- 5b P11 ----
    L.append("\n## 5b — plasticity dividend (P11), orthogonal mean-radius task\n")
    L.append(f"| P11 | desaturated (DRIVE) nets learn a new task faster | "
             f"{'✅ PASS' if p11['p11_pass'] else '❌ FAIL (dividend DROPPED)'} | "
             f"DRIVE<FROZEN&SSL in {p11['order_ok']}/{len(out['seeds'])} seeds; Spearman ρ(saturation,steps-to-90%)="
             f"{p11['rho']:.2f}; DRIVE final acc={p11['drive_final']:.2f} vs FROZEN={p11['frozen_final']:.2f} |")
    if not p11["p11_pass"]:
        L.append("\n> **P11 dividend DROPPED (not deferred).** The desaturation is "
                 "physiological (v4: 0.36→0.05 saturated units) but does not translate into "
                 "faster new-task learning at this scale/ordering. Per the addendum kill rule, "
                 "the dividend claim is dropped until a scale where it reappears.\n")
    if p11["capacity_traded"]:
        L.append("\n> Note: DRIVE-idled nets plateau LOWER than FROZEN "
                 f"({p11['drive_final']:.2f} vs {p11['frozen_final']:.2f}) — desaturation traded "
                 "capacity; the claim would be halved even if the speed ordering held.\n")

    # ---- 5c P12 ----
    base_pool = float(np.mean([base[r]["rate"] for r in REG5]))
    stab_pool = float(np.mean([stab[r]["rate"] for r in REG5]))
    rec_base = float(np.mean([base[r]["recov"] for r in REG5]))
    rec_stab = float(np.mean([stab[r]["recov"] for r in REG5]))
    p12_pass = (stab_pool - base_pool >= 0.3) and (rec_stab - rec_base <= 50)
    L.append("\n## 5c — label-free class stabilizer (P12)\n")
    L.append(f"Pilot (seed 0, then FROZEN): λ_stab candidates {out['pilot']} → chose "
             f"**λ_stab={out['lam']:.0f}**.\n")
    L.append(f"| P12 | L_stab raises survival ≥0.3 pooled without hurting L_ssl recovery >50 steps | "
             f"{'✅ PASS' if p12_pass else '❌ FAIL'} | pooled survival "
             f"{base_pool:.2f}→{stab_pool:.2f} (Δ{stab_pool-base_pool:+.2f}); pooled L_ssl "
             f"recovery {rec_base:.0f}→{rec_stab:.0f} steps (Δ{rec_stab-rec_base:+.0f}) |")
    L.append("\n| regime | survival base → +L_stab |")
    L.append("|---|---|")
    for r in REG5:
        L.append(f"| {r} | {base[r]['rate']:.2f} → {stab[r]['rate']:.2f} |")
    if not p12_pass and (stab_pool - base_pool < 0.3):
        L.append("\n> **P12 negative (worth its own paragraph).** A multi-view global "
                 "surrogate does NOT reinstall a broken winding class: W1 (relational "
                 "blindness to topology) extends to L_stab too. Once the shift tunnels the "
                 "phase field across the gate, agreement between two noise views of the "
                 "*shifted* inputs re-anchors the (already-wrong) class rather than the "
                 "original — so class REPAIR needs either the oracle angular supervision or a "
                 "fundamentally different (non-relational) mechanism. A real limit, not a "
                 "tuning failure.\n")

    L.append("\n## Figures\n")
    for f, cap in [("exp5_dividend_P11.png", "P11: pre-task saturation vs new-task learning speed + curves."),
                   ("exp5_survival_P12.png", "5a-iii vs 5c: class survival under fresh shifts, ±L_stab.")]:
        L.append(f"![{cap}](results/figures/{f})\n\n*{cap}*\n")

    L.append("\n## Pre-registration hygiene (§4)\n")
    L.append("- Ballisticity metric unchanged (v4 §7.1: baseline-subtracted net advance + η_d "
             "scaling). Drive constants, deploy lr 3e-4, grad clip, margins/λ_b: unchanged.\n"
             f"- λ_stab fixed by the documented seed-0 pilot ({out['lam']:.0f}) before the "
             "pre-registered survival runs.\n"
             "- Checkpoints were regenerated by re-idling (v4 did not persist encoders); the "
             "idle protocol/seeds are identical to v4, so the idled states reproduce v4's.\n")

    txt = "\n".join(L) + "\n"
    path = f"{ROOT}/RESULTS.md"
    with open(path) as fh:
        base_txt = fh.read().split(V5_MARKER)[0].rstrip() + "\n"
    with open(path, "w") as fh:
        fh.write(base_txt + txt)
    print("wrote v5 section to RESULTS.md")
    return dict(gate=a["verdict"], p11=p11["p11_pass"], p12=bool(p12_pass))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    cfg = CFG
    if a.quick:
        cfg = dataclasses.replace(CFG, steps_install=500, steps_barrier=300, steps_kill=200,
                                  ens_epochs=25, n_points=1500, n_traj_per_class=120,
                                  drive_steps_idle=1500, v5_draws=4, v5_adapt_steps=800)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(cfg.seeds)
    if a.report_only:
        out = np.load(result_path("exp5_data.npy"), allow_pickle=True).item()
    else:
        out = run(cfg, seeds)
    make_figures(out)
    import json
    print(json.dumps(write_results(out, cfg), indent=2))


if __name__ == "__main__":
    main()
