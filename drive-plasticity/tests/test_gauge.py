import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from driveplast.model import MLP, flat_params
from driveplast.gauge import GaugeDrive, relu_layers, gauge_mediators, gate_crossing_sweep


def _probe(model, x):
    with torch.no_grad():
        return model(x).clone()


def test_rescale_invariance_random_net():
    """§4.1: the exact-symmetry audit. Patrolling the gauge orbit must leave
    outputs on a fixed probe set unchanged to floating-point tolerance -- an
    implementation-correctness gate, not a finding."""
    torch.manual_seed(0)
    model = MLP(width=48, depth=2)
    x = torch.randn(32, 784)
    f0 = _probe(model, x)
    dims = [l.out_features for l in relu_layers(model)[0]]
    drive = GaugeDrive(dims, period=40, amp=0.4, seed=1)
    max_dev = 0.0
    for _ in range(80):
        drive.step(model)
        f = _probe(model, x)
        max_dev = max(max_dev, float((f - f0).abs().max()))
    assert max_dev < 1e-4, max_dev


def test_gauge_orbit_closes_after_one_period():
    """A full period returns the network to its EXACT starting weights (the
    Lissajous loop is closed, s(period) = s(0) = 0)."""
    torch.manual_seed(0)
    model = MLP(width=32, depth=2)
    theta0 = flat_params(model).detach().clone()
    dims = [l.out_features for l in relu_layers(model)[0]]
    drive = GaugeDrive(dims, period=50, amp=0.3, seed=2)
    for _ in range(50):
        drive.step(model)
    theta1 = flat_params(model).detach()
    assert float((theta1 - theta0).abs().max()) < 1e-4


def test_matched_displacement_iso_can_replicate_gauge_magnitude():
    """The ISO control must be able to match GAUGE's measured per-step
    displacement (§3) -- sanity-check the calibration procedure itself:
    isotropic noise sized via sigma = disp/sqrt(n) produces comparable norms
    on average."""
    torch.manual_seed(0)
    model = MLP(width=32, depth=2)
    n = sum(p.numel() for p in model.parameters())
    dims = [l.out_features for l in relu_layers(model)[0]]
    drive = GaugeDrive(dims, period=200, amp=0.3, seed=3)
    disp = [drive.step(model) for _ in range(20)]

    gen = torch.Generator().manual_seed(0)
    iso_norms = []
    for d in disp:
        sigma = d / (n ** 0.5)
        noise = torch.randn(n, generator=gen) * sigma
        iso_norms.append(float(noise.norm()))
    # per-step, the EXPECTED iso norm equals the gauge displacement by
    # construction (E||noise|| ~ sigma*sqrt(n) = d); check the realized
    # sample is within a generous factor (single draw, so allow slack).
    for d, iso in zip(disp, iso_norms):
        assert 0.3 * d < iso < 3.0 * d, (d, iso)


def test_gate_crossing_sweep_diverges():
    """§2 exploratory bridge: driving one unit's incoming scale c -> 0 sends
    its outgoing weights (~1/c) to a large norm -- the parameter-space gate,
    demonstrated on request (not part of the primary bounded patrol)."""
    torch.manual_seed(0)
    model = MLP(width=16, depth=2)
    out = gate_crossing_sweep(model, layer_idx=0, unit_idx=0, n=40, c_min=0.02)
    assert out["outgoing_norm"][-1] > 5 * out["outgoing_norm"][0]
    assert out["incoming_norm"][-1] < out["incoming_norm"][0]


def test_gauge_mediators_run():
    torch.manual_seed(0)
    model = MLP(width=24, depth=2)
    med = gauge_mediators(model)
    assert len(med["cv"]) == 2 and len(med["cond"]) == 2
    assert all(c >= 0 for c in med["cv"])
