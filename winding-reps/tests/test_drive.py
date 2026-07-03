import math

import numpy as np
import torch

from winding.config import CFG
from winding.models import PhaseEncoder
from winding.data import Embedding, sample_annulus_points, make_traj_track
from winding.drive import (QHead, sample_pairs, drive_vector, apply_drive,
                           period_integrals, circular_mean_phase, _final,
                           _rot, _rot_prime, _drive_scalar, lssl)

PI = math.pi


def _offgate_encoder(seed=0):
    """A phase encoder whose outputs sit off the gate (undo the near-gate init)
    so the clean 1-form identities hold without the m^2 clip biting."""
    torch.manual_seed(seed)
    enc = PhaseEncoder(CFG)
    with torch.no_grad():
        enc.net[-1].weight.mul_(40.0); enc.net[-1].bias.mul_(40.0)
    return enc


def _batch(n=128, seed=0):
    rng = np.random.default_rng(seed)
    p, _ = sample_annulus_points(n, CFG, rng)
    x = Embedding(CFG)(p, noise=0.0)
    return torch.tensor(x, dtype=torch.float32)


def test_period_D_is_2pi_and_Lssl_is_zero():
    enc = _offgate_encoder()
    x = _batch()
    traj = make_traj_track(CFG, seed=1, n_per_class=16)
    xt = torch.tensor(traj["x"], dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)
    x_t, x_tk, k = sample_pairs(xt, CFG.drive_kmax, gen)
    q = QHead(CFG.drive_kmax)
    circ_D, circ_L = period_integrals(enc, x, (x_t, x_tk, k, q), margin=1e-3, n=200)
    # D is closed non-exact: nonzero period ~ 2*pi (per-sample mean)
    assert abs(circ_D - 2 * PI) < 0.05 * 2 * PI, circ_D
    # L_ssl is exact: zero period around the U(1) orbit
    assert abs(circ_L) < 0.05 * 2 * PI, circ_L


def test_drive_orthogonal_to_lssl_on_orbit():
    # along the orbit tangent T (rigid final-row rotation), D.T ~ 1 per sample
    # while grad(L_ssl).T ~ 0.
    enc = _offgate_encoder()
    x = _batch()
    traj = make_traj_track(CFG, seed=2, n_per_class=16)
    xt = torch.tensor(traj["x"], dtype=torch.float32)
    gen = torch.Generator().manual_seed(1)
    x_t, x_tk, k = sample_pairs(xt, CFG.drive_kmax, gen)
    q = QHead(CFG.drive_kmax)
    W0 = _final(enc).weight.detach().clone(); b0 = _final(enc).bias.detach().clone()
    dW = _rot_prime(0.0) @ W0; db = _rot_prime(0.0) @ b0
    gW, gb = torch.autograd.grad(_drive_scalar(enc, x, 1e-3),
                                 [_final(enc).weight, _final(enc).bias])
    D_dot_T = float((gW * dW).sum() + (gb * db).sum())
    lW, lb = torch.autograd.grad(lssl(enc, q, x_t, x_tk, k),
                                 [_final(enc).weight, _final(enc).bias])
    L_dot_T = float((lW * dW).sum() + (lb * db).sum())
    assert abs(D_dot_T - 1.0) < 0.05, D_dot_T          # per-sample phase rate = 1
    assert abs(L_dot_T) < 0.05, L_dot_T                # L_ssl flat along the orbit


def test_drive_advances_phase_counterclockwise():
    enc = _offgate_encoder()
    x = _batch()
    seq = [circular_mean_phase(enc, x)]
    for _ in range(12):                      # small eta_d so total advance < 2*pi
        apply_drive(enc, drive_vector(enc, x, CFG.barrier_margin), eta_d=0.005)
        seq.append(circular_mean_phase(enc, x))
    un = np.unwrap(np.array(seq))
    incs = np.diff(un)
    assert un[-1] - un[0] > 0.0                # net counterclockwise advance
    assert np.all(incs > 0.0)                  # monotone (directed, not diffusive)
