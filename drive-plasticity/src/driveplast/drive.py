"""The generalized drive: directed, structure-preserving motion confined to the
flat / high-EU complement of the committed subspace.

Kept from the topological drive: DIRECTED (deterministic rotation, not noise),
CONFINED to the low-curvature subspace, STRUCTURE-PRESERVING (D ⊥ g, so to first
order the drive does not change the task loss — the general-setting version of
"winding conserved"). Dropped: the exact integer conservation law.

  D = normalize( P_flat(d) ),   P_flat(x) = x − Σ_j (x·v_j)v_j − (x·ĝ)ĝ

d is a slowly-rotating vector in a fixed 2-plane of the flat subspace (the
deterministic circulation surrogate for the topological limit cycle).
"""
import math

import torch


class Drive:
    """mode: 'drive' (confined+directed), 'iso' (directed, NOT confined to flat),
    'undir' (confined, fresh random d each step — no circulation)."""

    def __init__(self, n, mode="drive", rotate_period=200, seed=0):
        self.n = n
        self.mode = mode
        self.rotate_period = rotate_period
        self.t = 0
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(n, generator=g); a /= a.norm()
        b = torch.randn(n, generator=g); b -= (b @ a) * a; b /= b.norm()
        self.a, self.b, self.gen = a, b, g

    def _d(self):
        if self.mode == "undir":
            d = torch.randn(self.n, generator=self.gen)
            return d / d.norm()
        ang = 2 * math.pi * self.t / self.rotate_period
        return math.cos(ang) * self.a + math.sin(ang) * self.b

    def step(self, g_flat, V):
        """Return (unit drive direction, leakage into S_hi). V: (n,k) basis or None."""
        d = self._d()
        gn = g_flat / (g_flat.norm() + 1e-12)
        d = d - (d @ gn) * gn                            # project out g (all modes)
        leak = 0.0
        if V is not None:
            comp = V @ (V.T @ d)                         # component in S_hi
            leak = float(comp.norm() / (d.norm() + 1e-12))
            if self.mode != "iso":                       # 'drive'/'undir' confine to flat
                d = d - comp
        self.t += 1
        if d.norm() < 1e-6:                               # degenerate (d was ~in g/S_hi)
            return torch.zeros_like(d), leak
        return d / d.norm(), leak
