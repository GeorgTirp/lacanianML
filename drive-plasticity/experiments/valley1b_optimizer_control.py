"""valley-1b: does the optimizer explain valley-1's K-noteeth?

The rescaling gauge changes per-layer weight NORMS. Under plain SGD the
effective per-layer step size scales with the parameterization, so orbit
position can genuinely alter learning dynamics. Under Adam, per-coordinate
second-moment normalization makes updates ~invariant to exactly this
rescaling -- Adam would ABSORB the gauge position, pre-explaining K-noteeth
even if the phenomenon were real.

valley-1's original run already used PLAIN SGD (no momentum) for both
training phases -- confirmed by inspection of valley1_gauge_drive.py's
`train_to_minimum`/`train_new_task` (`torch.optim.SGD(..., lr=cfg["lr"])`,
no momentum, no adaptive optimizer). There was therefore no pre-existing
"adaptive run" to keep for comparison as the addendum's §1 assumed; this
script fills that gap by running SGD, SGD+momentum, AND Adam under one
unified, identical-otherwise protocol so all three are directly comparable.

Everything except the optimizer is byte-for-byte the same call into
valley1_gauge_drive.run_seed: same net, same K_idle, same period=4*K_idle
park, same matched-displacement ISO arm, same seeds, same locked P-G1 margin.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from valley1_gauge_drive import HP, run_seed, steps_to_threshold  # noqa: E402

OPTIMIZERS = ["sgd", "sgd_momentum", "adam"]


def run(cfg, seeds, optimizers):
    out = {}
    t0 = time.time()
    for opt_name in optimizers:
        out[opt_name] = {}
        for seed in seeds:
            r, gate, disp = run_seed(seed, cfg, opt_name=opt_name)
            out[opt_name][seed] = r
            print(f"[{opt_name:12s} seed {seed}] audit_max={r['GAUGE']['audit_max']:.2e} | "
                  f"steps-to-thr PLAIN={r['PLAIN']['steps_to_thr']} GAUGE={r['GAUGE']['steps_to_thr']} "
                  f"ISO={r['ISO']['steps_to_thr']} | final_loss P={r['PLAIN']['final_loss']:.3f} "
                  f"G={r['GAUGE']['final_loss']:.3f} I={r['ISO']['final_loss']:.3f}", flush=True)
    out["meta"] = dict(seeds=seeds, cfg=cfg, optimizers=optimizers)
    out["runtime"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--optimizers", type=str, default=",".join(OPTIMIZERS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    resdir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(resdir, exist_ok=True)
    save = os.path.join(resdir, "valley1b_data.npy")
    if a.report_only:
        out = np.load(save, allow_pickle=True).item()
    else:
        cfg = dict(HP)
        seeds = [int(s) for s in a.seeds.split(",")]
        optimizers = a.optimizers.split(",")
        if a.quick:
            seeds = seeds[:2]; cfg["epochs0"] = 2; cfg["epochs1"] = 3; cfg["K_idle"] = 50
        out = run(cfg, seeds, optimizers)
        np.save(save, out, allow_pickle=True)
    from valley1b_report import make_results
    import json
    print(json.dumps(make_results(out), indent=2))


if __name__ == "__main__":
    main()
