#!/usr/bin/env python3
"""Does `quantiles_` decide anything on mapie's LAC cross-validation path?

Why this probe exists
---------------------
The census recorded `lac.py:158`, `quantiles_ = (n + 1) * (1 - alpha_np)`, as a
resolution site on the crossval path. The census criterion requires that a site's
value decide an output threshold, interval, set or p-value. This one does
not: the branch of `get_prediction_sets` reached on that same path never reads
`quantiles_`. It compares an inclusion count against `_alpha * (n - 1)` instead,
which is a different rule carrying no (n+1) correction at all.

Reading the source says so. Reading has been wrong in this project before, so
this probe settles it by perturbation: scale `quantiles_` by 1000 and ask whether
a single returned set changes. If the value decided anything, it would.

APS is the control, and it is the reason this is a finding about LAC rather than
about the pattern. `aps.py:201` computes the identical expression on the identical
path -- and APS's `get_prediction_sets` DOES read it, so APS's sets move. Same
line, same value, opposite answer. Without the control this probe would only show
that one perturbation did nothing.

What follows for the census: the anchor moves to the expression that maps the
level, and the count is unchanged at one site on this path. The criterion was not
weakened to keep the site; the site was re-anchored to satisfy it.

Run:  python lac_crossval_dead_value.py
"""

import sys
import warnings

import numpy as np

import mapie
from mapie.conformity_scores.sets.aps import APSConformityScore
from mapie.conformity_scores.sets.lac import LACConformityScore

LINES = []
SEED = 20260806
N_CAL = 40
N_CLASS = 3
N_TEST = 12
ALPHA = 0.10
BLOWUP = 1000.0


def say(s=""):
    LINES.append(s)
    print(s)


def self_check():
    """The perturbation must be big enough that a live value could not survive it.

    A factor that leaves every comparison on the same side of its threshold would
    make the LAC result vacuous -- the set would be unchanged because nothing
    moved, not because nothing was read. The APS control is what rules that out,
    and it only rules it out if APS and LAC are driven with the same numbers.
    """
    assert BLOWUP >= 100.0, BLOWUP
    assert 0.0 < ALPHA < 1.0, ALPHA
    # the value under test is a count on the scale of n, so a x1000 perturbation
    # takes it far outside the range any count-scale comparison could ignore
    assert (N_CAL + 1) * (1 - ALPHA) * BLOWUP > 100 * N_CAL


def fixtures():
    rng = np.random.default_rng(SEED)
    scores = rng.random(N_CAL)
    proba = rng.dirichlet(np.ones(N_CLASS), size=N_TEST)[:, :, None]
    return scores, proba, np.array([ALPHA])


def main():
    self_check()
    say("=" * 92)
    say("Does mapie's LAC crossval `quantiles_` determine the returned set?")
    say(f"mapie {mapie.__version__}, numpy {np.__version__}")
    say("self_check() passed at import (the perturbation is larger than any")
    say("count-scale comparison on this path could absorb)")
    say("=" * 92)
    say("")
    say(f"  n_calib = {N_CAL}, classes = {N_CLASS}, test rows = {N_TEST}, "
        f"alpha = {ALPHA}")
    say(f"  cv = 5, agg_scores = 'crossval'  -- the branch reached only when cv is")
    say("  not 'prefit' AND agg_scores is not 'mean'")
    say("")
    say(f"{'score':<6}{'quantiles_ on this path':>26}{'(n+1)(1-alpha)':>17}"
        f"{'set changes under x1000?':>27}")

    scores, proba, alpha = fixtures()
    expected = (N_CAL + 1) * (1 - ALPHA)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, cls in (("LAC", LACConformityScore), ("APS", APSConformityScore)):
            cs = cls()
            q = np.asarray(cs.get_conformity_score_quantiles(
                scores, alpha, cv=5, agg_scores="crossval"), dtype=float)
            cs.quantiles_ = q.copy()
            base = cs.get_prediction_sets(proba, scores, alpha, cv=5,
                                          agg_scores="crossval")
            cs.quantiles_ = q * BLOWUP
            blown = cs.get_prediction_sets(proba, scores, alpha, cv=5,
                                           agg_scores="crossval")
            changed = not np.array_equal(base, blown)
            rows.append({"name": name, "q": float(np.ravel(q)[0]),
                         "changed": changed})
            say(f"{name:<6}{float(np.ravel(q)[0]):>26.4f}{expected:>17.4f}"
                f"{str(changed):>27}")
    say("")

    lac = [r for r in rows if r["name"] == "LAC"][0]
    aps = [r for r in rows if r["name"] == "APS"][0]
    assert abs(lac["q"] - expected) < 1e-9, lac
    assert abs(aps["q"] - expected) < 1e-9, aps
    assert not lac["changed"], (
        "LAC's set moved: the value is live and the census anchor was right")
    assert aps["changed"], (
        "APS's set did not move either -- the perturbation is too small to "
        "discriminate, so LAC's negative result proves nothing")

    say("-" * 92)
    say("VERDICT")
    say("-" * 92)
    say("  Both classes compute the SAME value on the SAME path. Only APS reads it.")
    say("")
    say("  LAC: `quantiles_` is stored, exposed as a documented public fitted")
    say("  attribute (classification.py:1141 copies it onto the estimator), and")
    say("  never consulted by the crossval branch of get_prediction_sets, which")
    say("  thresholds an inclusion count against alpha*(n-1) instead. So the value")
    say("  determines nothing returned, and a caller reading `quantiles_` on this")
    say("  path is reading a number the prediction did not use.")
    say("")
    say("  APS is the control and it moves, so the perturbation discriminates.")
    say("")
    say("  Consequence for the census: the LAC crossval site is re-anchored from")
    say("  lac.py:158 to the expression that actually resolves the level. One site")
    say("  on this path either way -- the total is unchanged.")

    out = "outputs/probe_output_lac_crossval_dead_value.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
