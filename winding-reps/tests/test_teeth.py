import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from winding.config import CFG
from teeth_demo import make_hole_loops, abstention_stats, _auroc_hole_lower


def test_hole_loops_stay_inside_the_irreducible_region():
    hole = make_hole_loops(CFG, seed=0, n=40)
    r = np.linalg.norm(hole["p"], axis=2)          # (N,T)
    assert r.max() < CFG.r_inner, r.max()


def test_auroc_hole_lower_separates_clean_case():
    rng = np.random.default_rng(0)
    supp = rng.normal(5.0, 0.5, size=500)          # confident on-support
    hole = rng.normal(1.0, 0.5, size=500)          # clearly less confident in-hole
    auc = _auroc_hole_lower(hole, supp)
    assert auc > 0.95, auc


def test_auroc_hole_lower_is_half_when_indistinguishable():
    rng = np.random.default_rng(1)
    supp = rng.normal(0, 1, size=2000)
    hole = rng.normal(0, 1, size=2000)
    auc = _auroc_hole_lower(hole, supp)
    assert abs(auc - 0.5) < 0.05, auc


def test_abstention_stats_matches_calibrated_acceptance():
    rng = np.random.default_rng(2)
    supp = rng.normal(1.0, 0.2, size=2000)
    hole = rng.normal(0.0, 0.2, size=2000)         # systematically lower confidence
    out = abstention_stats(supp, hole, accept_frac=0.90)
    assert abs(out["supp_abst"] - 0.10) < 0.02      # calibration is exact by construction
    assert out["hole_abst"] > out["supp_abst"]      # hole abstains more
    assert out["score"] > 0
    assert out["auroc"] > 0.9
