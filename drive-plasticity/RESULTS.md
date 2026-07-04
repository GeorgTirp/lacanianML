# RESULTS — drive-plasticity (does the drive pay rent?)

Benchmark: **Permuted-MNIST** (real MNIST, raw IDX). Widths [100, 256], seeds [0, 1, 2, 3, 4], 250 tasks, depth 2. Runtime 1398s (23.3 min) CPU.


**Scope deviations (documented, §7.1):** (1) Reduced from the pre-registered 200 tasks / 2000-task literature regime to 250 tasks and 5 seeds — the full grid (7 arms × 5 seeds × 2 widths × HVP-every-step over 200 tasks) is many CPU-hours; this environment is CPU-only. (2) The curvature basis is amortized (recomputed every 25 steps, warm-started) rather than every step, to make the drive affordable. Both weaken the power of the test, not its logic; the kill-criteria bands (§6) are applied as locked.


## Kill-criteria verdict (reported first, §6)

> **The generalized drive does not pay rent — with a documented caveat.**
>
> **(1) The plasticity-maintenance test is under-powered here.** PLAIN first-third plasticity 0.792 → last-third 0.771 (drop +0.021, just under the 0.03 P-disease gate). At 250 tasks on a small MLP the plasticity-LOSS regime is only weakly reached (the literature uses ~thousands of tasks). So the strict P-drive-plast claim is formally not-testable — 'no disease, no test'.
>
> **(2) But the decisive negatives are disease-independent and all fire:** (a) DRIVE is the **worst** plasticity arm (0.760 < PLAIN 0.771; best = CBP 0.814) — to pay rent it must at least MATCH baselines; it mildly HURTS. (b) **K-baseline-wins:** CBP/L2-init dominate on plasticity. (c) No stability benefit: H 0.113 ≈ all arms; erosion +0.380 not below baselines. (d) **Confinement is vacuous:** leak 0.007 — the committed subspace is 5 of ~10⁵ params, so 'confined to flat' is nearly automatic and adds nothing. (e) Ablations inert: DRIVE erosion ≥ DRIVE-ISO (+0.362) and DRIVE-UNDIR (+0.369).
>
> **The narrower, honest conclusion (spec §1):** the payoff required the rare TYPED topological setting (v6, protected structure genuinely low-dimensional). The high-D generalization loses that leverage — good optimizers already vacate the flat directions, and in high-D nearly ALL directions are flat, so actively circulating them buys nothing over doing nothing. The theoretical/topological results stand on their own as a separate, smaller contribution.


## Conjunction table (width 100, 5 seeds)

| arm | plasticity (last⅓) | plasticity (first⅓) | stability H | probe P | erosion | drift | leak | wall/s |
|---|---|---|---|---|---|---|---|---|
| PLAIN | 0.771±0.007 | 0.792 | 0.114 | 0.524 | +0.353 | +0.410 | 0.00 | 6 |
| SHRINK | 0.789±0.011 | 0.808 | 0.104 | 0.467 | +0.409 | +0.362 | 0.00 | 15 |
| L2INIT | 0.808±0.004 | 0.808 | 0.114 | 0.493 | +0.385 | +0.379 | 0.00 | 7 |
| CBP | 0.814±0.004 | 0.799 | 0.113 | 0.562 | +0.317 | +0.448 | 0.00 | 7 |
| DRIVE | 0.760±0.011 | 0.790 | 0.113 | 0.499 | +0.380 | +0.386 | 0.01 | 15 |
| DRIVE_ISO | 0.774±0.009 | 0.802 | 0.118 | 0.516 | +0.362 | +0.398 | 0.01 | 15 |
| DRIVE_UNDIR | 0.767±0.014 | 0.783 | 0.120 | 0.510 | +0.369 | +0.390 | 0.01 | 23 |

## Probe diagnostic window (§9.2)

- Mean upper anchor P_i(t_i) = 0.878, reservoir P_rand = 0.704 → window = 0.174. DIAGNOSTIC (>=0.05): erosion/drift decomposition is used.


## Ablations, mediators, theory bridge

- **Confinement (DRIVE vs DRIVE-ISO):** erosion +0.380 vs +0.362 → confinement does NOT clearly help.

- **Direction (DRIVE vs DRIVE-UNDIR):** erosion +0.380 vs +0.369 → direction does NOT clearly help.

- **Leakage into S_hi (drive):** 0.007 of drive motion lands in the committed subspace despite projection (finite-HVP basis error). Leakage↔erosion correlation over seeds r=-0.52 (§9.5: if strong, the residual forgetting is the price of dropping the exact law).


## Width scaling (§7 — basis error grows with width)

| width | DRIVE plast | DRIVE H | DRIVE erosion | DRIVE leak |
|---|---|---|---|---|
| 100 | 0.760 | 0.113 | +0.380 | 0.01 |
| 256 | 0.824 | 0.118 | +0.265 | 0.00 |

## What is kept vs dropped from the topological drive

KEPT: directed (rotating d in a fixed flat 2-plane), confined to the low-curvature subspace, structure-preserving (D⊥g). DROPPED: the exact integer conservation law — here confinement is enforced only to the finite-HVP top-k basis, and leakage (0.01) is the measured cost of that approximation. If the drive fails here but the exact topological version succeeds (v6), the payoff needs the rare typed setting — the narrower, honest conclusion.


## Figures

![Per-task test accuracy over the stream, per arm.](results/figures/exp7_curves.png)

*Per-task test accuracy over the stream, per arm.*

![Plasticity vs stability (the conjunction) per arm.](results/figures/exp7_conjunction.png)

*Plasticity vs stability (the conjunction) per arm.*

![Drift vs erosion decomposition of forgetting per arm.](results/figures/exp7_decomp.png)

*Drift vs erosion decomposition of forgetting per arm.*

![Dead-unit fraction and feature rank over the stream.](results/figures/exp7_mediators.png)

*Dead-unit fraction and feature rank over the stream.*

