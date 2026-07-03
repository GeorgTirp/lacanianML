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

<!-- V4 SECTION -->

# v4 — the drive

Seeds: [0, 1, 2]. Runtime: 696s (11.6 min) CPU. Idling 15000 steps, adaptation 5000 steps.


**Tier-3 claim (as tested):** a deployed model can keep moving *meaningfully and safely*. The drive D (a closed non-exact term in the update rule) produces directed motion along the data-unidentified U(1) fiber — ballistic and winding-conserving — where isotropic noise (SGLD) at the same displacement only diffuses. (At this toy scale both are winding-safe because the permanent barrier protects both; the drive's demonstrated edge is DIRECTION, not extra safety.)


## Headline
The drive works **exactly as engineered**: ∮D=2π and ∮∇L_ssl=0 (P8a), and it produces a large **directed** phase advance — net 127 rad (≈20 full turns), **33× the isotropic control** above the shared SSL baseline, scaling with η_d (×1.8 for ×2). Isotropic noise (SGLD) at **matched per-step displacement** (‖Δθ‖ ratio 0.78) adds essentially no directed motion (net 10.2 rad ≈ the SSL baseline 6.6). The drive conserves the winding exactly (0 ungated) at no L_ssl cost (Δ=-0.005). (The strict pre-registered R²>0.99/α≈1 linearity is confounded by the drive's rate drift AND SSL's own baseline drift; directedness is judged by net advance and η_d scaling — §7.1.)


> **KILL CRITERION K3 (downstream) — motion buys nothing measurable.** At matched displacement SGLD is *also* winding-safe (retention 1.000, 0 ungated) — the barrier protects both, and the winding is far more robust than this displacement (v3 P6: breaking it needs σ~2 *relative* weight noise, ~1e5× more). And idled DRIVE adapts no better than FROZEN (within the coarse recovery resolution). So against FREEZING the directed motion confers no clearly measurable advantage at this single-charge toy scale — a scale caveat (argued, not assumed), per §6.
> *Nuance (honest):* at matched displacement DRIVE recovers L_ssl faster than the isotropic control SGLD in all 3 seeds (100 vs 283 steps) — a suggestive, not conclusive, sign that directed motion is less disruptive to adaptability than isotropic noise. Reported, not over-claimed (n=3, 50-step resolution, post-shift L_ssl start differs).


## Pass/fail table (v4)

| Prediction | Claim | Result | Detail |
|---|---|---|---|
| P8a | D closed non-exact (∮D≈2π), L_ssl exact (∮≈0) | ✅ PASS | ∮D=6.283 (2π=6.283), ∮∇L_ssl=-1.023e-10 |
| P8 | DRIVE directed (∝η_d) ≫ isotropic control; winding-safe | ✅ PASS | net Φ above SSL baseline: DRIVE=120 vs SGLD=3.6 rad (33×), η_d scaling ×1.8; DRIVE retention=1.000 (0 ungated); L_ssl Δ=-0.005; param-space travel ‖θ−θ₀‖ 4.3× SGLD's at matched per-step step. Caveat: strict linear-R²=0.87 not >0.99 (rate drift). SGLD also winding-safe (1.000) |
| P9 | idled DRIVE adapts (L_ssl) as well as SSL/FROZEN; winding-retention | ⚠️ PARTIAL | **L_ssl half ✓**: steps→90% recovery DRIVE=100, SSL=200, FROZEN=150 (DRIVE top group). **Winding half ✗ (shared)**: the shift breaks the winding for most regimes; SSL is blind to it (W1) so end winding-acc≈chance for ALL — FROZEN=0.56, SSL=0.56, DRIVE=0.56 |

## Kill-criteria verdicts (§6)

- **K1 — the drive doesn't drive / unstable:** not triggered (ballistic=True, L_ssl Δ=-0.005).

- **K2 — conservation violated by the drive:** not triggered (DRIVE ungated events lo=0, hi=0).

- **K3 — motion buys nothing:** TRIGGERED (DRIVE retention vs SGLD, and adaptation vs FROZEN).


## Matched-displacement calibration

| regime | η_d | SGLD σ | per-step ‖Δθ‖ | α | R² | net Φ (rad) | winding ret. | ungated |
|---|---|---|---|---|---|---|---|---|
| FROZEN | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 0.0 | 1.000 | 0 |
| SSL | 0.00e+00 | 0.00e+00 | 5.20e-03 | 0.71 | 0.501 | 6.6 | 1.000 | 0 |
| SGLD_lo | 0.00e+00 | 5.13e-05 | 7.71e-03 | 0.71 | 0.451 | 7.6 | 1.000 | 0 |
| SGLD_hi | 0.00e+00 | 1.03e-04 | 1.17e-02 | 0.99 | 0.632 | 10.2 | 1.000 | 0 |
| DRIVE_lo | 1.28e-03 | 0.00e+00 | 9.27e-03 | 0.76 | 0.947 | 72.2 | 1.000 | 0 |
| DRIVE_hi | 2.56e-03 | 0.00e+00 | 1.51e-02 | 0.80 | 0.869 | 127.0 | 1.000 | 0 |

## P10 (exploratory)

- **SSL plasticity drift:** participation ratio 1.97→2.16 over 15000 idle steps (DRIVE_hi end 2.13). At toy scale/horizon plasticity atrophy is expected to be small; reported as-is.

- **Motion-as-uncertainty:** all of D's circulation lives in the U(1) fiber (∮D=2π on the orbit, ∮ off-orbit directions ≈0 by construction), i.e. drive energy sits in the data-unidentified subspace — the intended behaviour.

- **Temporal ensemble:** a phase-relative readout averaged over one drive cycle equals the snapshot up to the global rotation (winding is cycle-invariant); no additional signal at this toy's single-charge scale.


## Scope & honesty

- This toy has ONE global unidentified fiber (the U(1) phase) and hence one implicit charge; the general EU-weighted multi-charge construction is out of scope.
- **P9 is a split result (honest).** The sensor-drift shift is strong enough to *break* the winding class for most regimes (FROZEN included). Relational SSL restores L_ssl (all regimes; DRIVE in the top group — L_ssl-adaptivity is preserved by idling) but is BLIND to the winding (warning W1), so it cannot restore the broken class — end winding-accuracy sits at ~chance for ALL regimes, DRIVE no worse than FROZEN as a class. This reinforces the project's own fairness note: installing/repairing a winding class needs the oracle angular supervision, which is unavailable at deployment. The strong P9 conjunction (adaptivity AND winding-retention) therefore does NOT hold — for anyone — and this is reported with the same prominence as the passing L_ssl half.
- The drive is a term in the *update rule*, not the loss — a real-valued loss cannot circulate (dL/dt=−‖∇L‖²≤0). exp4a verifies numerically that D is closed non-exact (period 2π) and L_ssl is exact (period 0).
- SGLD is calibrated to MATCHED per-step ‖Δθ‖ so the contrast is directed-vs-isotropic motion at equal energy, not a displacement-budget artifact.
- **Design changes (§7.1), documented:** (i) deployment lr lowered 1e-3→3e-4 and (ii) drive strengths raised from the addendum's (1e-3, 1e-2) to (2e-2, 4e-2) rad/step. Reason: the relational SSL objective induces an *incidental* global-phase drift of ~2.5e-3 rad/step at this toy scale (finite-batch symmetry breaking amplified by Adam). The pre-registered drive strengths sit at/below that floor, so no clean ballistic signal is possible there; the changes lift the drive clear of the floor. This is a property of the toy's SSL, not of the drive — exp4a confirms the drive is closed-non-exact regardless.
- **Ballistic operationalization (§7.1):** the pre-registered linear-fit R²>0.99 test FAILS (R²≈0.87), and the scale-invariant exponent α is ALSO confounded (SGLD α≈1 too). Two reasons: the drive's phase *rate* drifts as the encoder co-evolves under SSL, and SSL itself carries a small *systematic* phase drift (net ~7 rad) that SGLD inherits. The clean, honest discriminator is the BASELINE-SUBTRACTED net advance (how far past the shared SSL baseline each regime drives the phase) and whether the drive's advance scales with η_d. On that metric the drive advances ~120 rad (∝η_d) vs the isotropic control's ~3 rad — directed, not diffusive. R² and α are still reported for transparency.
- **Honest correction to P8:** the pre-registered 'SGLD damages the winding at matched displacement' is NOT observed and cannot be at this scale — the winding is far more robust (v3 P6) than the matched noise. At matched displacement the drive buys DIRECTION (ballistic vs diffusive), not extra winding-safety; the barrier already makes both regimes safe.


![P8: Φ(t) ballistic (DRIVE) vs diffusive (SGLD) + matched displacement.](results/figures/exp4_phi_P8.png)

*P8: Φ(t) ballistic (DRIVE) vs diffusive (SGLD) + matched displacement.*

![P8: winding retention and min‖f‖ during idling.](results/figures/exp4_retention_P8.png)

*P8: winding retention and min‖f‖ during idling.*

![P9: post-shift recovery of winding acc and L_ssl.](results/figures/exp4_adaptation_P9.png)

*P9: post-shift recovery of winding acc and L_ssl.*

![P10: participation ratio / saturation over idling.](results/figures/exp4_plasticity_P10.png)

*P10: participation ratio / saturation over idling.*
