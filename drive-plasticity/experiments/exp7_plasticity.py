"""exp7: does the drive pay rent? Directed vs isotropic perturbation on a real
loss-of-plasticity benchmark (Permuted-MNIST), testing the stability-plasticity
CONJUNCTION and the drift/erosion decomposition (§9).

Reports the KILL-criteria verdict first. Negatives with full prominence.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from driveplast.data import PermutedMNIST
from driveplast.model import MLP, flat_grad, add_flat_
from driveplast.curvature import top_k_subspace
from driveplast.drive import Drive
from driveplast.baselines import shrink_perturb_, l2_init_grad_, CBP
from driveplast.probe import probe_acc, random_trunk_probe

ARMS = ["PLAIN", "SHRINK", "L2INIT", "CBP", "DRIVE", "DRIVE_ISO", "DRIVE_UNDIR"]
HP = dict(lr=0.3, epochs=3, bs=128, k=5, tau_c=25, eta_d=0.05,
          shrink_lam=1e-3, shrink_sigma=1e-3, l2_lam=1e-3,
          cbp_period=100, cbp_frac=0.01, rotate_period=200, early=8)


def run_arm(arm, width, depth, stream, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    lossf = nn.CrossEntropyLoss()
    model = MLP(width=width, depth=depth)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    n = sum(p.numel() for p in model.parameters())
    init_params = [p.detach().clone() for p in model.parameters()]
    gen = torch.Generator().manual_seed(seed + 7)
    drive = Drive(n, mode={"DRIVE": "drive", "DRIVE_ISO": "iso", "DRIVE_UNDIR": "undir"}[arm],
                  rotate_period=cfg["rotate_period"], seed=seed) if arm.startswith("DRIVE") else None
    cbp = CBP(model, cfg["cbp_period"], cfg["cbp_frac"]) if arm == "CBP" else None

    per_task_acc, dead_tr, rank_tr, leak_tr = [], [], [], []
    anchors = {}                                        # P_i(t_i) for early tasks
    V = None; gstep = 0
    t0 = time.time()
    for ti in range(stream.n_tasks):
        xtr, ytr, xte, yte = stream.task(ti)
        N = len(xtr)
        for _ in range(cfg["epochs"]):
            for b in range(0, N, cfg["bs"]):
                xb, yb = xtr[b:b + cfg["bs"]], ytr[b:b + cfg["bs"]]
                loss = lossf(model(xb), yb)
                opt.zero_grad(); loss.backward()
                g_flat = flat_grad(model).detach()
                if drive is not None and gstep % cfg["tau_c"] == 0:
                    V = top_k_subspace(model, lossf, xb, yb, cfg["k"], iters=2, V0=V)
                if arm == "L2INIT":
                    l2_init_grad_(model, init_params, cfg["l2_lam"])
                opt.step()
                if drive is not None:
                    D, leak = drive.step(g_flat, V); add_flat_(model, D, cfg["eta_d"])
                    leak_tr.append(leak)
                elif arm == "SHRINK":
                    shrink_perturb_(model, cfg["shrink_lam"], cfg["shrink_sigma"], gen)
                elif arm == "CBP":
                    cbp.observe(model, xb); cbp.maybe_reinit(model, gstep, gen)
                gstep += 1
        with torch.no_grad():
            per_task_acc.append(float((model(xte).argmax(1) == yte).float().mean()))
        med = model.mediators(xte); dead_tr.append(med["dead"]); rank_tr.append(med["rank"])
        if ti < cfg["early"]:
            anchors[ti] = probe_acc(model, xtr, ytr, xte, yte, seed=seed)   # P_i(t_i)

    # ---- end-of-stream stability on the early set ----
    Hs, Ps, eros, drifts = [], [], [], []
    for ti in range(cfg["early"]):
        xtr, ytr, xte, yte = stream.task(ti)
        with torch.no_grad():
            H = float((model(xte).argmax(1) == yte).float().mean())
        P = probe_acc(model, xtr, ytr, xte, yte, seed=seed)
        Hs.append(H); Ps.append(P)
        eros.append(anchors[ti] - P)                    # true trunk erosion
        drifts.append(P - H)                            # readout-access drift
    return dict(arm=arm, acc=np.array(per_task_acc), dead=np.array(dead_tr),
                rank=np.array(rank_tr), leak=(np.mean(leak_tr) if leak_tr else 0.0),
                H=float(np.mean(Hs)), P=float(np.mean(Ps)), erosion=float(np.mean(eros)),
                drift=float(np.mean(drifts)), anchors=list(anchors.values()),
                Ps=Ps, eros=eros, wall=time.time() - t0)


def run(cfg, widths, seeds, n_tasks, arms, depth):
    stream_ref = None
    out = {}
    for width in widths:
        for seed in seeds:
            stream = PermutedMNIST(n_tasks, seed=seed)
            if stream_ref is None:
                x0tr, y0tr, x0te, y0te = stream.task(0)
                out["Prand"] = out.get("Prand", {})
            for arm in arms:
                r = run_arm(arm, width, depth, stream, cfg, seed)
                out.setdefault((width, arm), []).append(r)
                print(f"  w{width} s{seed} {arm:11s} plast(L/F)="
                      f"{r['acc'][-n_tasks//3:].mean():.3f}/{r['acc'][:n_tasks//3].mean():.3f} "
                      f"H={r['H']:.3f} eros={r['erosion']:+.3f} drift={r['drift']:+.3f} "
                      f"leak={r['leak']:.2f} ({r['wall']:.0f}s)", flush=True)
            # P_rand anchor once per (width, seed)
            xtr, ytr, xte, yte = stream.task(0)
            out["Prand"][(width, seed)] = random_trunk_probe(width, depth, xtr, ytr, xte, yte, seed=seed)
    out["meta"] = dict(widths=widths, seeds=seeds, n_tasks=n_tasks, arms=arms, depth=depth, cfg=cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=150)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--widths", type=str, default="100,256")
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    resdir = os.path.join(os.path.dirname(__file__), "..", "results")
    if a.report_only:
        out = np.load(os.path.join(resdir, "exp7_data.npy"), allow_pickle=True).item()
        from exp7_report import make_figures_and_results
        import json
        print(json.dumps(make_figures_and_results(out), indent=2))
        return
    cfg = dict(HP)
    seeds = [int(s) for s in a.seeds.split(",")]
    widths = [int(w) for w in a.widths.split(",")]
    arms = a.arms.split(",")
    tasks = a.tasks
    if a.quick:
        seeds, widths, tasks = seeds[:2], widths[:1], min(tasks, 40)
        cfg["epochs"] = 2
    t0 = time.time()
    out = run(cfg, widths, seeds, tasks, arms, a.depth)
    out["runtime"] = time.time() - t0
    resdir = os.path.join(os.path.dirname(__file__), "..", "results")
    np.save(os.path.join(resdir, "exp7_data.npy"), out, allow_pickle=True)
    from exp7_report import make_figures_and_results
    import json
    print(json.dumps(make_figures_and_results(out), indent=2))


if __name__ == "__main__":
    main()
