"""valley-1: does patrolling the gauge orbit (exact ReLU-rescaling symmetry)
between training episodes have operational teeth — does WHERE you sit on a
loss-exact orbit change what the network can learn NEXT?

Protocol (§4.2): train to a minimum on task 0; then PLAIN sits at theta*,
GAUGE patrols the closed rescaling-gauge Lissajous loop for K idle steps
(zero function change, audited every step per §4.1), ISO spends the SAME
per-step displacement budget as isotropic noise (moves off the minimum,
unlike GAUGE). Then task 1 is presented; measure steps-to-threshold and
final loss.

Reports the kill-criteria verdict first (§4.2): K-noteeth (GAUGE ~= PLAIN,
a clean and important negative — the frozen point really is arbitrary but
inertly so) or K-noise (GAUGE ~= ISO, structure buys nothing over noise) or
P-G1 supported.
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from driveplast.data import PermutedMNIST
from driveplast.model import MLP, flat_params
from driveplast.gauge import GaugeDrive, relu_layers, gauge_mediators, gate_crossing_sweep

ARMS = ["PLAIN", "GAUGE", "ISO"]
HP = dict(width=64, depth=2, lr=0.3, bs=128, epochs0=3, K_idle=200, amp=0.3,
          epochs1=8, eval_every=4, threshold=0.80, n_tasks=2, n_train=2000, n_test=2000)


def _gauge_period(cfg):
    """The Lissajous period is set STRICTLY LONGER than the idle-patrol length
    K_idle (4x) so the patrol stops a QUARTER of the way around the loop --
    parked at a point genuinely different from theta*, not having closed the
    loop back onto it. (A full-period patrol would land GAUGE back on the
    IDENTICAL parameters as PLAIN, not just the identical function -- that
    would make the operational-teeth test vacuous by construction.)"""
    return 4 * cfg["K_idle"]


def _eval(model, x, y, lossf):
    with torch.no_grad():
        logits = model(x)
        acc = float((logits.argmax(1) == y).float().mean())
        loss = float(lossf(logits, y))
    return acc, loss


def _probe_out(model, x):
    with torch.no_grad():
        return model(x).clone()


def train_to_minimum(model, xtr, ytr, cfg, seed):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss()
    N = len(xtr)
    for _ in range(cfg["epochs0"]):
        perm = torch.randperm(N)
        for b in range(0, N, cfg["bs"]):
            idx = perm[b:b + cfg["bs"]]
            loss = lossf(model(xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()


def train_new_task(model, xtr, ytr, xte, yte, cfg, seed):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss()
    N = len(xtr)
    budget = cfg["epochs1"] * ((N + cfg["bs"] - 1) // cfg["bs"])
    steps_log, acc_log, loss_log = [], [], []
    step = 0
    for _ in range(cfg["epochs1"]):
        perm = torch.randperm(N)
        for b in range(0, N, cfg["bs"]):
            idx = perm[b:b + cfg["bs"]]
            loss = lossf(model(xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % cfg["eval_every"] == 0 or step == budget:
                acc, tl = _eval(model, xte, yte, lossf)
                steps_log.append(step); acc_log.append(acc); loss_log.append(tl)
    return dict(steps=steps_log, acc=acc_log, loss=loss_log, budget=budget)


def steps_to_threshold(curve, threshold):
    for s, a in zip(curve["steps"], curve["acc"]):
        if a >= threshold:
            return s
    return curve["budget"]           # censored: never reached within budget


def first_grad_norm(model, xtr, ytr, cfg):
    lossf = nn.CrossEntropyLoss()
    xb, yb = xtr[:cfg["bs"]], ytr[:cfg["bs"]]
    loss = lossf(model(xb), yb)
    g = torch.autograd.grad(loss, list(model.parameters()))
    return float(torch.cat([x.reshape(-1) for x in g]).norm())


def run_seed(seed, cfg):
    torch.manual_seed(seed); np.random.seed(seed)   # fix BEFORE any param init (reproducibility)
    stream = PermutedMNIST(cfg["n_tasks"], n_train=cfg["n_train"], n_test=cfg["n_test"], seed=seed)
    x0tr, y0tr, x0te, y0te = stream.task(0)
    x1tr, y1tr, x1te, y1te = stream.task(1)
    lossf = nn.CrossEntropyLoss()

    base = MLP(width=cfg["width"], depth=cfg["depth"])
    train_to_minimum(base, x0tr, y0tr, cfg, seed)
    acc0, _ = _eval(base, x0te, y0te, lossf)                # sanity: task0 actually learned

    dims = [l.out_features for l in relu_layers(base)[0]]
    probe_x = x0te[:256]
    f0 = _probe_out(base, probe_x)
    med_pre = gauge_mediators(base)

    # precompute GAUGE's realized per-step displacement trace ONCE (from a
    # throwaway copy) so ISO can be matched to the EXACT same per-step budget
    tmp = copy.deepcopy(base)
    disp_trace = [GaugeDrive(dims, period=_gauge_period(cfg), amp=cfg["amp"], seed=seed + 11).step(tmp)
                  for _ in range(cfg["K_idle"])]
    del tmp
    n_params = sum(p.numel() for p in base.parameters())

    out = {}
    for arm in ARMS:
        m = copy.deepcopy(base)
        audit = [0.0]
        if arm == "GAUGE":
            drive = GaugeDrive(dims, period=_gauge_period(cfg), amp=cfg["amp"], seed=seed + 11)
            for _ in range(cfg["K_idle"]):
                drive.step(m)
                audit.append(float((_probe_out(m, probe_x) - f0).abs().max()))
        elif arm == "ISO":
            gen = torch.Generator().manual_seed(seed * 97 + 3)
            for d in disp_trace:
                sigma = d / (n_params ** 0.5)
                with torch.no_grad():
                    for p in m.parameters():
                        p.add_(sigma * torch.randn(p.shape, generator=gen))
        # PLAIN: idle no-op

        audit_max = float(max(audit))
        if arm == "GAUGE" and audit_max >= 1e-4:
            raise RuntimeError(f"§4.1 invariance audit FAILED (max dev {audit_max:.2e}) — "
                               "the rescaling is coded wrong; this is a bug, not a finding.")

        disp_from_star = float((flat_params(m).detach() - flat_params(base).detach()).norm())
        acc_old_post, loss_old_post = _eval(m, x0te, y0te, lossf)  # old-task retention after patrol
        med_post = gauge_mediators(m)
        fgn = first_grad_norm(m, x1tr, y1tr, cfg)
        curve = train_new_task(m, x1tr, y1tr, x1te, y1te, cfg, seed)
        stt = steps_to_threshold(curve, cfg["threshold"])
        out[arm] = dict(arm=arm, audit_max=audit_max, acc_old_pre=acc0, acc_old_post=acc_old_post,
                        loss_old_post=loss_old_post, med_pre=med_pre, med_post=med_post,
                        first_grad_norm=fgn, curve=curve, steps_to_thr=stt,
                        final_acc=curve["acc"][-1], final_loss=curve["loss"][-1],
                        disp_from_star=disp_from_star)

    # §2 exploratory: parameter-space gate crossing on the trained base (bridge only)
    gate = gate_crossing_sweep(base, layer_idx=0, unit_idx=0)
    return out, gate, disp_trace


def run(cfg, seeds):
    out = {}
    t0 = time.time()
    for seed in seeds:
        r, gate, disp = run_seed(seed, cfg)
        out[seed] = r
        out.setdefault("gate", {})[seed] = gate
        out.setdefault("disp", {})[seed] = disp
        print(f"[seed {seed}] audit_max={r['GAUGE']['audit_max']:.2e} disp(G/I)="
              f"{r['GAUGE']['disp_from_star']:.3f}/{r['ISO']['disp_from_star']:.3f} | "
              f"steps-to-thr PLAIN={r['PLAIN']['steps_to_thr']} GAUGE={r['GAUGE']['steps_to_thr']} "
              f"ISO={r['ISO']['steps_to_thr']} | final_loss P={r['PLAIN']['final_loss']:.3f} "
              f"G={r['GAUGE']['final_loss']:.3f} I={r['ISO']['final_loss']:.3f}", flush=True)
    out["meta"] = dict(seeds=seeds, cfg=cfg)
    out["runtime"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    resdir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(resdir, exist_ok=True)
    save = os.path.join(resdir, "valley1_data.npy")
    if a.report_only:
        out = np.load(save, allow_pickle=True).item()
    else:
        cfg = dict(HP)
        seeds = [int(s) for s in a.seeds.split(",")]
        if a.quick:
            seeds = seeds[:2]; cfg["epochs0"] = 2; cfg["epochs1"] = 3; cfg["K_idle"] = 50
        out = run(cfg, seeds)
        np.save(save, out, allow_pickle=True)
    from valley1_report import make_results
    import json
    print(json.dumps(make_results(out), indent=2))


if __name__ == "__main__":
    main()
