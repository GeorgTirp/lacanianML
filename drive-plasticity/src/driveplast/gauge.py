"""valley-1: the gauge-orbit drive — an EXACT, analytically-known symmetry
substrate, replacing exp7's vacuous finite-HVP curvature subspace.

For a ReLU hidden unit j, positive homogeneity gives an exact function-
preserving symmetry: multiply unit j's INCOMING weights+bias by c>0 and its
OUTGOING weights by 1/c, for every c>0 independently per unit. ReLU(c*z) =
c*ReLU(z) for c>0, and the next layer's contribution (1/c)*(c*relu_out) is
unchanged. The orbit per ReLU layer of width H is (R_+)^H; in log-coordinates
c = exp(s), it is the flat, CONTRACTIBLE group R^H (no winding here — that is
valley-2's permutation-gauge experiment, not this one).

GaugeDrive patrols a fixed closed Lissajous loop in a fixed 2-plane of the
combined log-scale space (concatenated across all ReLU layers), returning to
s=0 (identity gauge) after one period -- a closed, bounded, loss-exact orbit.
"""
import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  the rescaling gauge transform                                              #
# --------------------------------------------------------------------------- #
def relu_layers(model):
    """(lins, nexts): lins[l] is the Linear feeding ReLU layer l (its OUTPUT
    units are the gauge units); nexts[l] is the Linear consuming that ReLU
    layer's output (its INPUT columns are the outgoing weights)."""
    lins = [m for m in model.body if isinstance(m, nn.Linear)]
    nexts = lins[1:] + [model.head]
    return lins, nexts


def rescale_layer_(lin, nxt, delta_log_c):
    """theta_incoming *= exp(delta_log_c) (row-wise), theta_outgoing *=
    exp(-delta_log_c) (column-wise). Exact ReLU-homogeneity symmetry; leaves
    the network function bit-for-bit unchanged for any delta_log_c."""
    c = torch.exp(delta_log_c)
    with torch.no_grad():
        lin.weight.mul_(c.unsqueeze(1))
        lin.bias.mul_(c)
        nxt.weight.mul_((1.0 / c).unsqueeze(0))


class GaugeDrive:
    """Patrols a closed Lissajous loop s(t) in a fixed random 2-plane of the
    combined log-scale space, s(0) = 0. `.step(model)` advances t by one and
    applies the INCREMENTAL transform (s(t) - s(t-1)) so the cumulative
    absolute gauge state always equals s(t) exactly (closes to identity after
    one full `period`)."""

    def __init__(self, dims, period=200, amp=0.3, seed=0):
        self.dims = list(dims)          # [width_per_relu_layer, ...]
        self.n = sum(self.dims)
        self.period = period
        self.amp = amp
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(self.n, generator=g); a /= a.norm()
        b = torch.randn(self.n, generator=g); b -= (b @ a) * a; b /= b.norm()
        self.a, self.b = a, b
        self.t = 0
        self.cur = torch.zeros(self.n)

    def _target(self, t):
        ang = 2 * math.pi * t / self.period
        return self.amp * ((math.cos(ang) - 1.0) * self.a + math.sin(ang) * self.b)

    def step(self, model):
        """Advance the patrol by one step; returns the applied ||delta_theta||
        actually realized (measured, not assumed) for displacement matching."""
        theta_before = torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])
        self.t += 1
        target = self._target(self.t)
        delta = target - self.cur
        self.cur = target
        lins, nexts = relu_layers(model)
        for lin, nxt, d in zip(lins, nexts, torch.split(delta, self.dims)):
            rescale_layer_(lin, nxt, d)
        theta_after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        return float((theta_after - theta_before).norm())

    def at_start(self):
        return self.t == 0


# --------------------------------------------------------------------------- #
#  mediators (§4.3): weight-norm balance, layerwise Gram condition number     #
# --------------------------------------------------------------------------- #
def row_norm_cv(lin):
    """Coefficient of variation of incoming-weight-row norms (bias included)
    -- rescaling redistributes this; PLAIN can never change it."""
    with torch.no_grad():
        rows = torch.cat([lin.weight, lin.bias.unsqueeze(1)], dim=1)
        norms = rows.norm(dim=1)
    m = float(norms.mean())
    return float(norms.std() / (m + 1e-12))


def layer_condition_number(lin):
    with torch.no_grad():
        sv = torch.linalg.svdvals(lin.weight)
    sv = sv[sv > 1e-8 * sv[0]]
    return float(sv[0] / sv[-1]) if len(sv) > 1 else 1.0


def gauge_mediators(model):
    lins, _ = relu_layers(model)
    return dict(cv=[row_norm_cv(l) for l in lins],
                cond=[layer_condition_number(l) for l in lins])


# --------------------------------------------------------------------------- #
#  §2: the parameter-space gate (exploratory bridge, NOT part of the primary  #
#  test). A continuous rescale-toward-0 on one unit's incoming scale sends    #
#  its outgoing scale (1/c) to infinity -- a degenerate point on the gauge    #
#  orbit, structurally the same "f=0 gate" one level down from the phase-head #
#  winding story. The primary Lissajous patrol stays bounded away from this;  #
#  this function demonstrates/measures the crossing when explicitly driven   #
#  toward it, for a single unit of a single layer.                            #
# --------------------------------------------------------------------------- #
def gate_crossing_sweep(model, layer_idx=0, unit_idx=0, n=60, c_min=0.02):
    lins, nexts = relu_layers(model)
    lin, nxt = lins[layer_idx], nexts[layer_idx]
    w0 = lin.weight[unit_idx].detach().clone()
    b0 = lin.bias[unit_idx].detach().clone()
    v0 = nxt.weight[:, unit_idx].detach().clone()
    cs = torch.linspace(1.0, c_min, n)
    incoming_norm, outgoing_norm = [], []
    with torch.no_grad():
        for c in cs:
            lin.weight[unit_idx] = w0 * c
            lin.bias[unit_idx] = b0 * c
            nxt.weight[:, unit_idx] = v0 / c
            incoming_norm.append(float((lin.weight[unit_idx].norm() ** 2
                                        + lin.bias[unit_idx] ** 2) ** 0.5))
            outgoing_norm.append(float(nxt.weight[:, unit_idx].norm()))
        # restore exactly (c=1 point of this same path)
        lin.weight[unit_idx] = w0; lin.bias[unit_idx] = b0; nxt.weight[:, unit_idx] = v0
    return dict(c=cs.tolist(), incoming_norm=incoming_norm, outgoing_norm=outgoing_norm)
