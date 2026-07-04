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

## v3 — the training axis (exp3)

exp2 found no daylight between structural and learned winding invariants on the
**input axis** (fixed weights, near-manifold evaluation): when the label *is*
the invariant, a learned approximation and a structural function are the same
function. The conservation law, however, is about **training dynamics** — so v3
tests axes where the two can separate: continued training and weight-space noise.

**v3 claim:** structural invariants (typed phase head + permanent gate barrier)
are protected along the training axis; a learned invariant stored in ordinary
weights (baseline C) is not. Arms reuse exp2-quality models (all start at winding
accuracy 1.0). A = barrier active throughout; A_nb = barrier disabled during
fine-tuning (mechanism attribution); C = GRU with its winding head frozen for
evaluation.

- **P5 (retention under continued training):** fine-tune the shared trunk on two
  interfering tasks (quantized mean radius, then start-point sector). A's
  winding accuracy stays ≥ 0.99 absent gate events (any drop must coincide with
  a gate event); C's degrades. If A_nb ≈ A, the barrier is not load-bearing
  (typing alone protects) — reported.
  **[Refined by v3b, see below]:** a probe-recovery control shows C's frozen-head
  collapse *overstates* trunk information loss — a fresh linear probe recovers
  winding to ~0.79 from C's fine-tuned trunk (R=0.66, partial erosion). P5 is
  restated as *graded trunk degradation + total readout-access loss*, versus A's
  zero erosion and parameter-free readout. The strong "C forgot the invariant"
  wording is retracted as stated.
- **P6 (weight-noise plateau–cliff):** add per-layer-scaled Gaussian weight
  noise; A holds an exact plateau far past C, and A's per-loop failures are
  gate-mediated (failed loops sit closer to the gate); C degrades smoothly.
- **P7 (exploratory):** protection must not cost new-task plasticity — compare
  task learning curves.
- **v3 kill criterion (headline prominence):** if C retains as well as A under
  *both* continued training and weight noise, structural protection has no
  measurable advantage on any axis tested — stated plainly.

**EU fix (§3):** the exp2 disagreement peak at the origin is confounded (the four
sector boundaries meet there). v3 relocates ĉ from a **regression** point track
(`y = sin 2θ + 0.1ε`, no class boundaries), so the interior peak is
coverage-driven; its only requirement is landing inside the hole (per P1b).

**Scope:** v3 tests *passive* structural persistence (typing + barrier) under
ordinary training; it does not implement any active circulation/update-rule
change — a separate later tier.

## v4 — the drive (exp4, Tier-3)

The first *active* component: a **closed non-exact** term in the **update rule**
(not the loss — a real-valued loss cannot circulate, since dL/dt=−‖∇L‖²≤0). The
flow is `θ_{t+1} = θ_t − η∇(L_ssl+L_bar) + η_d·D(θ)`, where D is the pullback of
the angular 1-form dθ=(u dv−v du)/(u²+v²), advancing every sample's phase along
the data-unidentified U(1) fiber. The streaming loss `L_ssl` is **relational**
(depends only on phase *differences*), so the global phase is an exact symmetry —
the fiber the drive sweeps. Comparators: FROZEN, continual SSL, and **SGLD**
(SSL + isotropic noise — the "just add noise" null), calibrated to **matched
per-step displacement**.

- **P8a (period test):** ∮D≈2π (closed non-exact), ∮∇L_ssl≈0 (exact).
- **P8 (idling):** DRIVE Φ(t) ballistic (R²>0.99, slope∝η_d) while SGLD at matched
  displacement is diffusive; DRIVE conserves the winding at no L_ssl cost.
- **P9 (shift adaptation):** does idling preserve adaptability? DRIVE in the top
  group on adaptation AND retention.
- **Kill criteria (headline prominence):** K1 the drive doesn't drive / unstable;
  K2 the drive causes ungated winding loss; K3 the directed motion buys nothing
  measurable downstream (DRIVE≈SGLD on retention, ≈FROZEN on adaptation).

**Scope:** ONE global unidentified fiber (the U(1) phase) ⇒ one implicit charge;
the general EU-weighted multi-charge construction is out of scope. The drive is a
term in the *update rule*, not the loss. SGLD is matched by displacement so the
contrast is directed-vs-isotropic motion at equal energy.

## v5 — confound gate, dividend, stabilizer (exp5)

v5 audits v4's best downstream number and tests the follow-on claims. **exp5a is
a gate**: if the v4 "drive adapts better" result is an artifact of the winding
*class dying* (an unwound phase field is relationally easier to fit), the v5
RESULTS section leads with a **retraction** before any positive result. It does —
the post-shift L_ssl floor is set entirely by class status (BROKE≈0.00,
SURVIVED≈0.13), not regime, and the drive is the *most* shift-fragile arm.

- **exp5a (gate):** floor-vs-class-status verdict (ARTIFACT/DIVIDEND/MIXED);
  gated-breaks check (breaks are discrete-shift onset, not training events);
  survival rate per regime under fresh shifts (Wilson CIs).
- **P11 (dividend):** do desaturated (drive-idled) nets learn an *orthogonal*
  task (mean-radius) faster? Kill if no ordering (dropped, not deferred).
- **P12 (stabilizer):** a label-free multi-view phase-agreement loss `L_stab`
  during deployment; does it raise class survival ≥0.3 without hurting L_ssl
  recovery? A negative would show W1 (relational blindness to topology) extends
  to global surrogates — class *repair* needs the oracle or a non-relational
  mechanism.

## v6 — ring attractor: topology in the dynamics (exp6)

A pivot, not a patch. Tiers 2–5 painted S¹ onto the static readout map
`φ = f/‖f‖`; every deployment failure (uninstallable without oracle, unrepairable
after a shift, shattered by input jumps, 2-unit bottleneck) traced to that
staticness. v6 moves the circle into **fixed recurrent connectivity** — a
continuous ring attractor (N=64 units, cosine connectivity) — and only learns the
encoder `e_ψ` that injects input current onto it. The topology is supplied by the
recurrence; descent never has to create the hole. Biological existence proofs (not
evidence): head-direction ring attractors and grid-cell tori (Chaudhuri et al.
2019; Gardner et al. 2022).

- **P13** the settled-state cloud is a ring (b1=1, certified by angular coverage +
  radial concentration + loop-persistence; pre-committed thresholds).
- **P14** winding is tracked (acc ≥ 0.9) and conserved under continued training
  (probe-W / b1 constant absent a bump-collapse).
- **P15 (money) — repair by relaxation:** under the exp5 shift, does letting the
  ring *relax* (no oracle, no gradients) recover the winding a static decode
  loses? Pass = dynamic − static ≥ 0.3. **Kill K-repair** if dynamic ≈ static.
- **P16** intrinsic drive: an antisymmetric connectivity term circulates the bump
  spontaneously at zero input (velocity ∝ η) — the drive as a property of the
  module, not an optimizer term.
- **P17** (exploratory) reading the full N-dim population vs the 2-unit bottleneck.

Kills: **K-ring** (no attractor), **K-repair** (relaxation buys nothing — the
make-or-break), **K-drift** (continuous attractors are connectivity-fine-tuned;
measure bump drift before claiming repair). Installation is oracle-assisted
(standing fairness note): a relational-only objective admits the W1-trivial
stationary-bump solution. The novel claim (P15 repair) uses no oracle. The EU-as-
lack coupling (gain set by enclosed variance) is wired as a hook, tested in v7.

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
python experiments/exp3_retention.py        # v3: training-axis retention + weight noise
python experiments/exp4_drive.py            # v4: the drive (period test, idling, shift)
python experiments/exp5_dividend.py         # v5: confound gate, dividend, stabilizer
python experiments/exp6_ring.py             # v6: ring attractor (P13-P17, repair test)
```

Determinism: everything is seeded; exp2 runs ≥3 seeds and reports all. Every
experiment runs in < 10 min on CPU. Results and the pass/fail table are written
to `RESULTS.md`.
