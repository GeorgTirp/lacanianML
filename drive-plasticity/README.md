# drive-plasticity — does the drive pay rent?

The decisive, real-benchmark test of whether the deployment-time **drive** — the
active circulation of the subspace good optimizers vacate — is a genuine ML
mechanism or a mathematically-true mirage. Leaves the winding toy world.

## Thesis (unifies both papers)

Good optimizers *vacate* the sharp-curvature directions (paper 2, "Geometry of
Avoidance"); the sharp directions are the committed / low-EU ones, the flat
directions the unidentified / high-EU ones (paper 1, posterior variance). The
drive is the deployment-time **directed, structure-preserving circulation of
exactly the flat/high-EU complement** — motion that keeps a network plastic
without disturbing what it has learned. Does that buy the stability–plasticity
conjunction that isotropic perturbation cannot?

## What is kept vs dropped from the topological drive

- **KEPT:** (i) DIRECTED (deterministic rotation in a fixed flat 2-plane, not
  noise), (ii) CONFINED to the low-curvature subspace, (iii) STRUCTURE-PRESERVING
  (`D ⊥ g`, so to first order the drive does not change the task loss — the
  general-setting version of "winding conserved").
- **DROPPED:** the exact integer conservation law (no protected topology here);
  confinement is only to the finite-HVP top-k curvature basis, and the measured
  *leakage* into that basis is the honest cost of the approximation.

## The drive (`src/driveplast/drive.py`)

At step *t* with task gradient `g`:
1. top-k Hessian eigenvectors `{v_j}` via HVP subspace iteration (amortized,
   warm-started) → the committed subspace `S_hi`.
2. `D = normalize( P_flat(d) )`, `P_flat(x) = x − Σ_j(x·v_j)v_j − (x·ĝ)ĝ`, with
   `d` a slowly-rotating vector in a fixed flat 2-plane (the circulation
   surrogate for the topological limit cycle).
3. `θ ← θ − η g + η_d D`.

## Benchmark & arms

**Permuted-MNIST** (real MNIST, fetched as raw IDX — no torchvision), a
recognized loss-of-plasticity benchmark. Arms: PLAIN, SHRINK-PERTURB, L2-INIT,
CBP (SOTA plasticity), DRIVE, **DRIVE-ISO** (not confined to flat — isolates
confinement), **DRIVE-UNDIR** (fresh random d each step — isolates direction).

## Predictions & kill criteria (locked 2026-07-04)

- **P-disease:** PLAIN loses plasticity (else "no disease, no test").
- **P-drive-conjunction (THE CLAIM):** DRIVE is uniquely top on plasticity AND
  stability — matches perturbation baselines on plasticity while forgetting
  strictly less (≥0.05 early-task retention margin AND strictly less trunk
  **erosion**, §9).
- **Kills (reported first):** **K-inert** (DRIVE ≈ PLAIN plasticity — no rent
  paid), **K-no-conjunction** (retains plasticity but forgets as much as
  shrink-perturb), **K-baseline-wins** (an existing method already wins).

## §9 probe-recovery layer (from the v3b control)

Frozen-readout retention overstates forgetting. Each early-task loss is
decomposed into **drift** (readout-access, recoverable — `P_i − H_i`) and
**erosion** (true trunk loss — `P_i(t_i) − P_i`), via a fresh linear probe on the
frozen trunk, anchored against a random-trunk reservoir `P_rand`. The drive's
signature prediction is **near-zero erosion**. A `< 0.05` probe window on this
benchmark makes the metric non-diagnostic (pre-registered fallback to H).

## Run

```bash
pip install -r requirements.txt
pytest -q                                       # HVP, curvature, drive⊥g, probe
python experiments/exp7_plasticity.py           # full grid (writes RESULTS.md)
python experiments/exp7_plasticity.py --quick   # fast smoke
```

MNIST auto-downloads to `data/` on first run. Everything CPU.
```
src/driveplast/  data, model, curvature, drive, baselines, probe
experiments/     exp7_plasticity, exp7_report
tests/           HVP correctness, curvature, drive properties, probe
```
