# RESULTS — winding-reps Tier-2

Seeds: [0, 1, 2]. Total runtime: 169s (2.8 min) on CPU.

Parameter budget (§4 match): phase encoder 5634, GRU baseline 7563 (ratio 1.34×).

Estimated centers ĉ per seed: [-0.014, 0.007], [0.073, 0.049], [0.018, -0.028]


## Headline

The topological machinery behaves exactly as specified: arms A (estimated center) and D (oracle) show an *exact* robustness plateau on the no-crossing subset, flip discretely and track the oracle winding on crossings, conserve the integer winding except at gate events, and placement is topologically tolerant (P1, P1b, P3 pass). **However, the kill criterion is TRIGGERED:** a parameter-matched supervised baseline (C) is essentially as robust on the primary metric (no-crossing consistency; A/D−C gap 0.025) and also tracks the oracle winding on crossings (C=0.981). In this toy the topological structure is *sufficient but not advantageous* over a strong baseline. Reported as a negative result per §7.3–7.4; see caveats below for what would give the baseline room to break.


## Pass/fail table

| Prediction | Claim | Result | Detail |
|---|---|---|---|
| P1 | clean acc A≈D≫B | ✅ PASS | A=1.000±0.000, B=0.544±0.149, C=1.000±0.000, D=1.000±0.000 |
| P1b | centers inside hole ≈ D; outside poor | ✅ PASS | inside-hole clean acc=1.000, outside-hole=0.486 |
| P2 | no-crossing consistency: A/D flat ~1, C decays | ❌ FAIL | A/D flat-floor=1.000; at ε_max A=1.000, D=1.000, C=0.975 (C degraded 0.025, A/D−C gap=0.025) |
| P2b | crossing: A/D track oracle winding AND C unstructured | ❌ FAIL | pooled track A=1.000, D=1.000, C=0.981, chance=0.333 — C also tracks (contrast fails) |
| P3 | probe-loop W conserved except at gate events | ✅ PASS | conservation violations A=0, D=0 |
| P4 (exploratory) | ‖f‖ depressed inside hole | 🔬 OBSERVED | depressed in 100% of seeds; hole/supp median ratio =0.91 (weakly depressed); point-ensemble in-hole confidence=0.928 (confidently arbitrary — a calibration failure the phase norm partly avoids). Not a pass/fail claim. |

## Interpretation

> **KILL CRITERION TRIGGERED (§7.4):** on the no-crossing subset the supervised baseline C matches the A/D plateau (gap 0.025 < 0.03) — the topological framing adds nothing to conditional stability here.

- **Cliff (P2b) — negative contrast:** A/D flip discretely and track the oracle new winding (A=1.000, D=1.000), but the baseline C *also* tracks it (C=0.981). A well-trained winding classifier reads the true current winding of a crossed loop just as the phase head does, so the cliff does **not** by itself distinguish the topological representation. Reported per §7.3.


## Figures

![M3: EU disagreement map with ĉ, oracle center, data.](results/figures/exp2_eu_map.png)

*M3: EU disagreement map with ĉ, oracle center, data.*

![P2: no-crossing consistency vs ε (primary metric).](results/figures/exp2_plateau_P2.png)

*P2: no-crossing consistency vs ε (primary metric).*

![P2b: crossing-subset tracking of the oracle new winding.](results/figures/exp2_cliff_P2b.png)

*P2b: crossing-subset tracking of the oracle new winding.*

![P3: probe-loop W and min‖f‖ during training.](results/figures/exp2_conservation_P3.png)

*P3: probe-loop W and min‖f‖ during training.*

![P4: phase-norm depression inside the hole.](results/figures/exp2_calibration_P4.png)

*P4: phase-norm depression inside the hole.*

![exp0: Tier-1 conservation + gate reproduction.](results/figures/exp0_conservation.png)

*exp0: Tier-1 conservation + gate reproduction.*


## P1b center sweep (clean accuracy)

| center | region | clean acc |
|---|---|---|
| (0.0, 0.0) | inside | 1.000 |
| (0.4, 0.0) | inside | 1.000 |
| (0.0, 0.5) | inside | 1.000 |
| (1.5, 0.0) | outside | 0.643 |
| (0.0, 1.6) | outside | 0.478 |
| (1.8, 0.0) | outside | 0.335 |

## What would change our mind

- **P2 is the core claim.** If the no-crossing consistency of A/D decays with ε like C's (or if C stays flat too), the plateau is not real and the topological framing adds nothing — reported as the kill criterion.
- If **P1** showed B≈A, anchoring at the singularity would be irrelevant (any on-support center would do), falsifying the placement thesis.
- If **P3** logged winding changes without a coincident gate event (`min‖f‖<0.02`), the conservation law — the mechanism behind the plateau — would be violated.
- If **P2b** tracking sat at chance, the 'cliff' would be mere noise rather than a structured, oracle-predictable flip.
- **Baseline-breaking regime (the missing condition for a *positive* P2).** Here C stays robust because a smooth deformation that avoids the hole keeps the embedded loop near the training manifold, where the GRU generalizes. The topological advantage should appear only when robustness is demanded in a regime the baseline cannot memorize — e.g. perturbations that push far off-manifold without crossing the gate, out-of-distribution loop shapes, or much less winding-label supervision for C. A single-seed quick run showed a transient A/D−C gap (~0.08) that vanished under 3 seeds and the full test set: a reminder that the seed protocol (§6) exists precisely to kill such mirages.


## Fairness note & limitations

Arms A/B/D receive oracle *angular* supervision during installation; the manipulated variable is the phase **center**, not supervision availability. C receives winding labels directly. The claim under test is about robustness *structure*, not label efficiency or self-supervised installation (future work). See README for the full limitations list.

<!-- V3 SECTION -->

# v3 — the training axis

Seeds: [0, 1, 2]. Runtime: 143s (2.4 min) CPU.


**v3 claim:** structural invariants (typed phase head + permanent gate barrier) are protected along the *training* axis — continued training and weight-space noise — where a learned invariant stored in ordinary weights (baseline C) is not.


## Headline

On the training axis the two invariants **separate**. Under continued training on interfering tasks, structural winding retention stays 1.000 (floor 1.000) while the learned baseline C falls to 0.444. Under weight noise, A holds an exact plateau to σ≈1.47 — far past C's σ≈0.33 — and its per-loop winding failures are gate-mediated (AUC 0.74: failed loops sit systematically closer to the gate than intact ones), while C degrades smoothly with no such structure. Note: A_nb ≈ A here, so the *typing* (topological readout) does the protecting; the barrier is not decisive in this task regime.


## Pass/fail table (v3)

| Prediction | Claim | Result | Detail |
|---|---|---|---|
| P5 | struct. winding retained under continued training; learned degrades | ✅ PASS | A final=1.000 (floor 1.000, ungated drops 0), A_nb=1.000, C=0.444 |
| P6 | struct. weight-noise plateau far past learned; failures gate-mediated | ✅ PASS | A plateau end σ≈1.47 vs C σ≈0.33; gate-mediation AUC (cliff)=0.77, (band)=0.74; C degrades=True |
| EU-fix | regression ĉ lands inside the hole (its only requirement) | ✅ PASS | ĉ per seed: [0.078, 0.055], [0.058, 0.119], [0.041, 0.038] |

## Mechanism attribution & interpretation

- **Barrier load-bearing?** A final 1.000 vs A_nb 1.000: no clear gap — typing alone may already protect, barrier not decisive here.

- **Design notes (§7.1):** (i) the weight-noise σ ceiling was extended from the addendum's 1.0 to 4.0 because A's plateau exceeded 1.0 (A never fails within the pre-registered range — a stronger result than anticipated) and the sweep must reach A's cliff to test the failure mechanism. (ii) The gate-mediation of A's failures is reported as a graded AUC (do failed loops sit closer to the gate?) rather than a hard min‖f‖<0.02 count: under weight noise ‖f‖ is depressed toward the gate without always hitting the training floor, and deep in the broken regime the discrete winding also fails via phase-field scrambling (adjacent-step gaps >π) — a discretization mode distinct from a true gate crossing.

- Fine-tuning uses shared grad-norm clipping (ft_grad_clip) so the permanent barrier stays effective against the large initial task gradient; applied identically to A, A_nb and C.


## P7 (exploratory — cost of protection)

- Per-task learnability (final task acc): radius — A=1.000, C=1.000; sector — A=0.298, C=0.966.

- **There IS a plasticity cost.** The structural arm reads new tasks only through the *mean-pooled* 2-D phase output — a bottleneck that discards whatever the pooled phase does not carry. It struggles on the **sector** task while C (reading a 40-D order-aware GRU state) learns both. Note the mechanism: pooling over a full winding loop averages away per-step and start-point angle (so the start-sector task collapses to ~chance), while a global magnitude like mean radius survives pooling. Protecting the winding through this typed head is not free — it costs expressivity for tasks orthogonal to what the pooled phase preserves.


## v3 figures

![EU fix: regression disagreement map with ĉ inside the hole.](results/figures/exp3_eu_regression_map.png)

*EU fix: regression disagreement map with ĉ inside the hole.*

![P5: winding retention under continued training + A gate/min‖f‖ trace.](results/figures/exp3_retention_P5.png)

*P5: winding retention under continued training + A gate/min‖f‖ trace.*

![P6: weight-noise robustness + per-sample accuracy distributions.](results/figures/exp3_weightnoise_P6.png)

*P6: weight-noise robustness + per-sample accuracy distributions.*


## What would change our mind (v3)

- If C's continued-training retention stayed high (kill half A) or its weight-noise curve were as flat as A's (kill half B), structural protection would add nothing on the training axis either — the honest end of the program as framed.
- If A's retention dropped WITHOUT a coincident gate event, the conservation law (the claimed protection mechanism) would be false.
- If A_nb matched A, the *typing* would be doing the work and the barrier would be decorative — a different mechanism than claimed.
- Scope: this tests *passive* persistence (typing + barrier) under ordinary training; it does NOT implement any active circulation/update-rule change (a later tier).

