# winding-reps — Topologically protected representations (Tier-2 toy)

## Thesis

We test whether **semantic identity can be organized as a winding (homotopy)
class around a data-identifiable uncertainty singularity**, and whether the
resulting robustness is *quantized* — exact stability until a discrete flip, a
**plateau–cliff** — rather than *graded* (smooth decay) as in standard
representations.

A representation is a circle-valued **phase head** `φ_θ(x) = f_θ(x)/‖f_θ(x)‖ ∈ S¹`.
A closed loop of inputs induces an integer **winding number**
`W = (1/2π) Σ wrap(Δφ)`. Because `W` is a homotopy invariant, it cannot change
under continuous deformation of the loop unless the loop is dragged through the
**gate** `f_θ(x)=0` (the point where the phase is undefined). That discreteness
is the hypothesized source of quantized robustness.

**Tier-1** (reference `../tier1_v0.py`) validated the mechanism on a circle:
winding is conserved during training except at detectable *gate events*, and a
norm barrier turns representational collapse from *penalized* into *prohibited*.

**Tier-2** (this repo) removes the oracle. The singularity is no longer supplied
by an augmentation group; it must be **located from data** via epistemic
uncertainty (EU). The payoff measured is **robustness structure**, not
anti-collapse.

## Setup

Generative space is 2D. Data lives on the annulus `r ∈ [1, 2]`; the open disk
`r < 1` has **zero support by construction** — an interior region no data ever
covers, i.e. the ground-truth epistemic singularity at the origin. Models see
only a frozen 20-D embedding `x = ψ(p) + ε`.

- **Point track**: annulus points, label = angular sector (K=4). Used to train
  an ensemble whose disagreement map locates the interior singularity `ĉ`.
- **Trajectory track** (main): closed loops, label = winding `k ∈ {−1,0,+1}`
  around the origin.

## Arms (identical architecture and budget; only the center differs)

| Arm | Center for the phase head | Role |
|-----|---------------------------|------|
| **A** | estimated `ĉ` (from EU map) | the method under test |
| **B** | wrong center `(1.5,0)` **on the data support** | placement ablation |
| **C** | — (GRU on embedded trajectory, supervised) | baseline, param-matched to A within ~2× |
| **D** | true center `(0,0)` | oracle upper bound |

A wrong center *inside the hole* is homotopy-equivalent to the oracle and is
therefore **not** used as B — that equivalence is prediction P1b.

Training schedule for A/B/D (retains the Tier-1 protocol): **install** (`L_inst`
supervises the phase toward the center angle over the *full loop*), then
**barrier-on** (activate `L_bar` only after `|W−target|<0.1` on probe loops, per
warning W2, then keep it on permanently), then **kill-switch** (turn `L_inst`
off, barrier stays) so conservation diagnostics carry over.

### Warnings honored (validated Tier-1 failure modes)
- **W1** locality is blind to topology → installation supervises the whole loop.
- **W2** install before guarding → barrier activates only after `W` is measured.
- **W3** the integer is gradient-dead → `W` is *monitored*, never in a loss.

## Pre-registered predictions (§7.1 — do not alter after seeing results)

- **P1** (placement): clean accuracy A ≈ D ≫ B. If B ≈ A the "anchored at the
  singularity" thesis is falsified.
- **P1b** (topological tolerance): any `ĉ` inside the hole performs like D;
  placement matters only up to the complementary component.
- **P2** (plateau — core claim): conditioned on *no oracle hole-crossing*, A/D
  consistency with the original label stays ≈ 100 % at all perturbation `ε`,
  while C degrades smoothly on the same no-crossing subset.
- **P2b** (cliff): on crossing samples A/D flip discretely and their new
  prediction tracks the *oracle new winding* above chance; C's errors are
  unstructured w.r.t. the oracle winding change.
- **P3** (conservation): during A/D training, probe-loop `W` is constant except
  at gate events (`min‖f‖ < 0.02`); reported as a joint log.
- **P4** (exploratory): inside the hole the raw norm `‖f_θ(x)‖` is depressed
  relative to on-support points — a free abstention signal; baseline / ensemble
  are expected to be confidently arbitrary there.

**Primary metric (no metric shopping, §7.5):** conditional-on-no-crossing
consistency. Everything else is secondary or exploratory.

**Kill criterion (§7.4):** if C matches A/D conditional stability under P2, the
topological claim adds nothing and is reported as such, with equal prominence.

## Fairness note

Arms A/B/D receive oracle *angular* supervision during installation — the
manipulated variable is the **center**, not the availability of supervision. C
receives the winding labels directly. The claim under test is about the
**robustness structure** of the resulting representation, not label efficiency
or fully self-supervised installation.

## Limitations (future work)

- Installation still uses oracle angular supervision; fully self-supervised
  installation of the winding class is out of scope here.
- 2D generative space with a single, convex, centered hole; multi-hole /
  higher-genus singularity structure is untested.
- EU localization uses a small deep ensemble; other estimators may move `ĉ`.

## Layout & running

```
src/winding/   data, models, losses, topology, uncertainty, train, eval
experiments/   exp0_sanity, exp1_product_latent, exp2_main
tests/         topology / data / uncertainty unit tests
results/       logs (.npz/.csv) and figures/
```

```bash
pip install -r requirements.txt
pytest -q                                   # unit tests
python experiments/exp0_sanity.py           # M1: conservation + gate reproduction
python experiments/exp1_product_latent.py   # M2: product-latent open question
python experiments/exp2_main.py             # M3–M5: EU map, four arms, full eval
```

Determinism: everything is seeded; exp2 runs ≥3 seeds and reports all. Every
experiment runs in < 10 min on CPU. Results and the pass/fail table are written
to `RESULTS.md`.
