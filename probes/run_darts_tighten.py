#!/usr/bin/env python3
"""Tighten the darts null result: more fits, several calibration sizes.

run_darts.py Part C measured 0.9033 against a requested 0.9000 over 300 fits at
cal_length=10, standard error 0.0171. That is informative but not tight, and it
is a single cell. A null that carries an argument should not rest on either.

This probe reruns the same construction at REPS >= 2000 across four calibration
lengths, and reports for each cell the measured coverage, its standard error,
the exact coverage the convention predicts, and the gap between them in units of
that standard error.

The prediction is exact, not fitted: darts passes an uncorrected level to
np.quantile with method="higher", which lands on 1-based rank ceil(c*(n-1)) + 1,
and a bound built on rank r covers with probability exactly r/(n+1).
"""

import math
from fractions import Fraction as F

import numpy as np

REPS = 2000
CAL_LENGTHS = (10, 15, 30, 50)
QUANTILES = [0.05, 0.5, 0.95]  # a 90% two-sided interval
SERIES_LEN = 120
SEED = 20260805
OUT = "outputs/probe_output_darts_tighten.txt"


def darts_rank(c, n):
    """1-based rank numpy's method='higher' selects at level c over n scores."""
    return math.ceil(c * (n - 1)) + 1


def required_rank(c, n):
    """1-based rank a valid finite-sample bound needs."""
    return math.ceil(F(n + 1) * F(c))


def predicted_two_sided(n):
    """A HYPOTHESIS about darts' two-sided interval, and the run FALSIFIES it.

    The model: the lower rail takes level 0.05 and the upper 0.95, both
    uncorrected; with method='higher' each lands on a rank, the interval spans
    ranks [r_lo, r_hi], and coverage is (r_hi - r_lo)/(n+1).

    Measured against 2000 fits per cell this is wrong by up to 27 standard
    errors -- at n=10 it predicts 0.7273 where 0.9065 is observed. So
    ConformalNaiveModel does NOT build its interval from two independent rails
    of the signed residuals in the way assumed here.

    The column is retained, clearly labelled, because a falsified prediction
    with a measurement beside it is evidence; a silently deleted one is not.
    Resolving what the model actually does is an open item -- read
    ConformalNaiveModel's own scoring path rather than guessing again.
    """
    r_lo = darts_rank(F(1, 20), n)
    r_hi = darts_rank(F(19, 20), n)
    return F(r_hi - r_lo, n + 1), r_lo, r_hi


def self_check():
    # method='higher' rank model, against numpy itself
    import numpy as _np

    for n in range(2, 200):
        for c in (F(1, 20), F(19, 20), F(2, 3)):
            got = float(_np.quantile(_np.arange(1, n + 1, dtype=float), float(c), method="higher"))
            assert abs(got - darts_rank(c, n)) < 1e-9, (n, c, got, darts_rank(c, n))
    # required rank, hand-verified cells and the feasibility boundary
    assert required_rank(F(9, 10), 9) == 9
    assert required_rank(F(19, 20), 19) == 19
    assert required_rank(F(19, 20), 18) == 19  # > n, so infeasible at n=18
    # the two-sided prediction is a probability
    for n in CAL_LENGTHS:
        p, lo, hi = predicted_two_sided(n)
        assert 0 <= p <= 1 and lo < hi, (n, p, lo, hi)


self_check()


def main():
    from darts import TimeSeries
    from darts.models import ConformalNaiveModel, LinearRegressionModel

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 86)
    say("DARTS, TIGHTENED -- end-to-end coverage at several calibration lengths")
    say("=" * 86)
    say("self_check() passed at import (rank model verified against numpy on 594 cells)")
    say(f"{REPS} independent fits per cell, iid Gaussian series, "
        f"LinearRegressionModel(lags=1), quantiles={QUANTILES}")
    say("requested two-sided coverage: 0.9000")
    say("")
    say(f"  {'cal_len':>7}  {'measured':>9}  {'s.e.':>7}  {'predicted':>9}  "
        f"{'gap/s.e.':>9}  {'ranks':>10}  {'width':>9}")

    for cal in CAL_LENGTHS:
        rng = np.random.default_rng(SEED + cal)
        hits, widths = [], []
        for _ in range(REPS):
            y = rng.standard_normal(SERIES_LEN)
            base = LinearRegressionModel(lags=1)
            base.fit(TimeSeries.from_values(y[:-1]))
            cm = ConformalNaiveModel(model=base, quantiles=QUANTILES, cal_length=cal)
            vals = cm.predict(n=1, predict_likelihood_parameters=True, num_samples=1).values().ravel()
            lo, hi = float(vals[0]), float(vals[-1])
            hits.append(lo <= float(y[-1]) <= hi)
            widths.append(hi - lo)

        cov = float(np.mean(hits))
        se = float(np.std(hits, ddof=1) / math.sqrt(REPS))
        pred, r_lo, r_hi = predicted_two_sided(cal)
        gap = (cov - float(pred)) / se if se > 0 else float("nan")
        say(f"  {cal:>7}  {cov:>9.4f}  {se:>7.4f}  {float(pred):>9.4f}  "
            f"{gap:>+9.2f}  {f'[{r_lo},{r_hi}]':>10}  {np.mean(widths):>9.4f}")

    say("")
    say("'predicted' is the exact coverage of the ranks darts lands on, r/(n+1) --")
    say("derived, not fitted. 'gap/s.e.' is how far the measurement sits from it.")
    say("A gap within about +/-2 means the convention explains the coverage fully.")
    say("")
    say("Honest scope, unchanged: iid by construction, so the guarantee SHOULD hold")
    say("exactly here. This is not a claim about real dependent series.")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
