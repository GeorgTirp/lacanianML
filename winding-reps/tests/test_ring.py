import math

import numpy as np
import torch

from winding.config import CFG
from winding import ring
from winding.topology import winding

PI = math.pi


def test_bump_forms_from_noise():
    W = ring.ring_weights(CFG)
    th = ring.preferred_angles(CFG.N_ring)
    torch.manual_seed(0)
    r0 = torch.rand(1, CFG.N_ring) * 0.1
    r = ring.relax(W, torch.zeros(1, CFG.N_ring), CFG, R=200, r0=r0)
    amp = float(ring.decode_amplitude(r, th)[0])
    assert amp > 0.5                       # a localized bump, not a flat state


def test_decode_inverts_injected_angle():
    W = ring.ring_weights(CFG)
    th = ring.preferred_angles(CFG.N_ring)
    for a in np.linspace(-PI, PI, 8, endpoint=False):
        b = 0.5 * torch.cos(th - float(a))[None]
        r = ring.relax(W, b, CFG, R=80)
        dec = float(ring.decode_phase(r, th)[0])
        d = (dec - a + PI) % (2 * PI) - PI
        assert abs(d) < 0.05, (a, dec)


def test_marginal_stability_drift_small():
    W = ring.ring_weights(CFG)
    th = ring.preferred_angles(CFG.N_ring)
    a0 = 1.0
    r = ring.relax(W, 0.5 * torch.cos(th - a0)[None], CFG, R=80)
    r = ring.relax(W, torch.zeros(1, CFG.N_ring), CFG, R=300, r0=r)  # remove input
    drift = abs(((float(ring.decode_phase(r, th)[0]) - a0 + PI) % (2 * PI)) - PI)
    assert drift < 0.05                    # marginally stable ring, low drift


def test_ring_certificate_positive_and_negative():
    W = ring.ring_weights(CFG)
    th = ring.preferred_angles(CFG.N_ring)
    ring_states = []
    for a in np.linspace(-PI, PI, 120, endpoint=False):
        r = ring.relax(W, 0.5 * torch.cos(th - float(a))[None], CFG, R=60)
        ring_states.append(r[0].numpy())
    assert ring.ring_certificate(np.array(ring_states))["b1"] == 1
    # a tight cluster (all near one angle) is NOT a ring -> b1 = 0
    blob = np.array(ring_states[:6] * 20) + np.random.default_rng(0).normal(scale=1e-3, size=(120, CFG.N_ring))
    assert ring.ring_certificate(blob)["b1"] == 0


def test_winding_tracked_around_loops():
    W = ring.ring_weights(CFG)
    th = ring.preferred_angles(CFG.N_ring)

    class Id(torch.nn.Module):
        def forward(self, x):
            return x

    for k in (-1, 0, 1):
        if k == 0:
            angs = 0.5 * np.sin(np.linspace(0, 2 * PI, 64, endpoint=False))
        else:
            angs = np.linspace(0, 2 * PI * k, 64, endpoint=False)
        xs = torch.stack([0.5 * torch.cos(th - float(a)) for a in angs])
        psi, rho, _ = ring.settle_sequence(Id(), W, xs, CFG, R=20, carry=True)
        assert round(winding(psi.numpy())) == k
        assert float(rho.min()) > 0.3       # bump stayed alive


def test_intrinsic_drive_circulates():
    # asymmetric connectivity -> spontaneous circulation at zero input (P16 core)
    th = ring.preferred_angles(CFG.N_ring)
    Wp = ring.ring_weights(CFG, eta=0.6)
    Wm = ring.ring_weights(CFG, eta=-0.6)
    a0 = 0.0
    seq_p, seq_m = [], []
    rp = ring.relax(Wp, 0.5 * torch.cos(th - a0)[None], CFG, R=60)
    rm = ring.relax(Wm, 0.5 * torch.cos(th - a0)[None], CFG, R=60)
    for _ in range(40):
        rp = ring.relax(Wp, torch.zeros(1, CFG.N_ring), CFG, R=2, r0=rp)
        rm = ring.relax(Wm, torch.zeros(1, CFG.N_ring), CFG, R=2, r0=rm)
        seq_p.append(float(ring.decode_phase(rp, th)[0]))
        seq_m.append(float(ring.decode_phase(rm, th)[0]))
    up, um = np.unwrap(seq_p), np.unwrap(seq_m)
    assert (up[-1] - up[0]) * (um[-1] - um[0]) < 0     # opposite signs
    assert abs(up[-1] - up[0]) > 0.1                    # actually moves
