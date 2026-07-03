"""Single shared configuration for all arms and experiments.

Scientific-integrity rule §7.2: shared hyperparameters across arms, no per-arm
tuning. Every experiment imports CFG from here. The only manipulated variable
between arms A/B/D is the phase center; C differs only in that it is a
supervised baseline. Nothing else may be tuned per arm.
"""
from dataclasses import dataclass, field, asdict
import math


@dataclass(frozen=True)
class Config:
    # ---- data / generative space ----
    D: int = 20                      # embedding dimension R^D
    r_inner: float = 1.0             # annulus inner radius (hole boundary)
    r_outer: float = 2.0             # annulus outer radius
    obs_noise: float = 0.05          # Gaussian observation noise sigma
    embed_hidden: int = 32           # frozen embedding net hidden width
    embed_seed: int = 12345          # fixed seed for the frozen embedding psi

    # ---- point track ----
    n_sectors: int = 4               # K angular sectors for point-track labels
    n_points: int = 4000             # point-track training set size

    # ---- trajectory track ----
    T: int = 64                      # steps per closed loop
    n_fourier: int = 3               # order of periodic (Fourier) noise
    noise_amp_alpha: float = 0.25    # angular periodic-noise amplitude
    noise_amp_r: float = 0.20        # radial periodic-noise amplitude
    r_clip_lo: float = 1.05
    r_clip_hi: float = 1.95
    n_traj_per_class: int = 400      # loops per winding class {-1,0,+1}

    # ---- phase head / losses ----
    enc_hidden: int = 64             # encoder MLP hidden width (2 layers)
    barrier_margin: float = 0.5      # m in L_bar
    lam_bar: float = 10.0            # lambda_b
    gate_thresh: float = 0.02        # min||f|| below this => gate event

    # ---- training schedule (per-arm identical) ----
    lr: float = 2e-3
    batch: int = 64
    steps_install: int = 1500        # phase 1: L_inst on
    steps_barrier: int = 1500        # phase 2: barrier on, L_inst still on
    steps_kill: int = 1000           # phase 3: L_inst off (kill switch), barrier stays
    eval_every: int = 25
    barrier_W_tol: float = 0.1       # |W - target| < tol before activating barrier

    # ---- EU ensemble ----
    n_ensemble: int = 8              # M classifiers
    ens_hidden: int = 32
    ens_epochs: int = 60
    grid_lim: float = 2.5            # generative square [-lim, lim]^2
    grid_n: int = 121                # grid resolution per axis

    # ---- baseline C (GRU) ----
    gru_hidden: int = 40

    # ---- v3: training-axis experiment (exp3) ----
    # Additive only; none of these change exp0-exp2 behaviour.
    ft_head_hidden: int = 32         # per-task head width (mean-pool -> MLP)
    ft_budget: int = 800             # max fine-tune steps per interfering task
    ft_target_acc: float = 0.95      # early-stop task accuracy
    ft_grad_clip: float = 1.0        # grad-norm clip so the barrier stays effective
    ft_log_every: int = 10           # fine-tune logging cadence (fine enough for gates)

    # ---- v4: the drive (exp4) ----  additive only
    deploy_lr: float = 3e-4          # SSL deployment lr (shared). Lowered from 1e-3
                                     # so SSL's incidental global-phase drift stays
                                     # below the drive signal (see RESULTS §7.1 note).
    drive_kmax: int = 8              # relational L_ssl step offsets k in {1..kmax}
    drive_steps_idle: int = 15000    # exp4b idling horizon
    drive_steps_shift: int = 5000    # exp4c adaptation horizon
    drive_log_every: int = 50
    # target mean per-step phase advance. Raised from the addendum's (1e-3, 1e-2)
    # to clear the SSL phase-noise floor (~2.5e-3 rad/step) at this toy scale so
    # the ballistic signal is clean; documented per §7.1.
    drive_advance_lo: float = 2e-2   # DRIVE-lo
    drive_advance_hi: float = 4e-2   # DRIVE-hi
    drive_calib_steps: int = 200     # steps used to calibrate eta_d and SGLD sigma
    n_phase_probe: int = 256         # fixed probe set for the cumulative-phase Phi(t)
    shift_strength: float = 3.0      # exp4c sensor-drift rotation (=> Q=exp(6*S));
                                     # calibrated to raise held-out L_ssl ~3x (forcing
                                     # SSL re-fitting) while the winding survives — the
                                     # encoder is robust until much larger shifts.

    # ---- v5: confound gate / dividend / stabilizer (exp5) ----  additive
    v5_draws: int = 10               # fresh shift draws per regime (5a-iii, 5c)
    v5_adapt_steps: int = 1500       # adaptation steps per shift draw

    # ---- v6: ring attractor (exp6) ----  additive
    # Topology lives in the FIXED recurrent weights; only e_psi is learned.
    N_ring: int = 64                 # ring units, preferred angles 2*pi*i/N
    ring_J0: float = -1.2            # uniform (inhibitory) recurrent term
    ring_J1: float = 4.0             # cosine (ring) recurrent term (> bump threshold)
    ring_h0: float = 0.10            # uniform background input
    ring_tau: float = 1.0
    ring_dt: float = 0.1
    ring_R: int = 25                 # relaxation steps to settle a bump per input
    ring_R_rep: int = 40             # extra relaxation steps for P15 repair
    ring_eta: float = 0.0            # asymmetric drive strength (P16 sets > 0)
    ring_input_gain: float = 1.0     # scales encoder current onto the ring
    ring_install_steps: int = 250    # oracle-assisted install of e_psi (fairness note)
    ring_R_train: int = 5            # relaxation steps per input during training
    ring_shifts: int = 10            # P15 fresh shift draws
    wn_sigma_lo: float = 0.01        # weight-noise relative sigma range
    wn_sigma_hi: float = 4.0         # extended 1.0->4.0: A's plateau exceeds the
                                     # addendum's pre-registered 1.0 (documented,
                                     # §7.1) so the sweep must reach A's cliff
    wn_n_sigma: int = 13
    wn_samples: int = 20             # noise draws per sigma
    reg_noise: float = 0.1           # regression EU target noise (0.1*eps)

    # ---- evaluation ----
    eps_grid_max: float = 1.2
    eps_grid_n: int = 13
    n_test_per_class: int = 200

    # ---- misc ----
    seeds: tuple = (0, 1, 2)         # >=3 seeds for exp2
    wrong_center: tuple = (1.5, 0.0) # B: wrong center ON the data support
    true_center: tuple = (0.0, 0.0)  # D: oracle center

    @property
    def steps_total(self) -> int:
        return self.steps_install + self.steps_barrier + self.steps_kill

    def to_dict(self):
        return asdict(self)


CFG = Config()
PI = math.pi
