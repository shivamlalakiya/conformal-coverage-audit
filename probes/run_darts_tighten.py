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

The prediction column was WRONG in the first version of this file, and is fixed
--------------------------------------------------------------------------------
It modelled the interval as two independent uncorrected rails, one at level 0.05
and one at 0.95, spanning ranks [r_lo, r_hi]. The measurement falsified that by
up to 27 standard errors. `darts_scoring_path.py` read the source and
instrumented it: with `symmetric=True` (the default) darts uses ABSOLUTE-error
scores, applies ONE uncorrected level `interval_range_sym` = 0.90, and returns
centre +/- that single threshold. The corrected column below uses that model,
and every link in it is asserted per fit in the other probe.
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
    """1-based rank a valid finite-sample bound needs.

    Argument order is (level, n). Returns the raw ceil even past n, matching
    historical call sites in this probe.
    """
    from conformal_coverage import required_rank as _rr
    k = _rr(n, c)
    return k if k is not None else math.ceil(F(n + 1) * F(c))


def predicted_two_rail_FALSIFIED(n):
    """The original hypothesis, kept because a falsified prediction beside a
    measurement is evidence and a silently deleted one is not.

    The model: lower rail at level 0.05, upper at 0.95, both uncorrected, the
    interval spanning ranks [r_lo, r_hi], coverage (r_hi - r_lo)/(n+1).
    Wrong by up to 27 standard errors -- at n=10 it predicts 0.7273 where
    0.9065 is observed. darts does not build the interval this way.
    """
    r_lo = darts_rank(F(1, 20), n)
    r_hi = darts_rank(F(19, 20), n)
    return F(r_hi - r_lo, n + 1), r_lo, r_hi


def predicted_symmetric(n):
    """The verified model. conformal_models.py:1681 with symmetric=True uses
    |residual| (metrics.ae, :1717) and applies ONE uncorrected level,
    interval_range_sym = 0.90 (:165-167), with method='higher'. So the rank is
    ceil(0.90*(n-1)) + 1 and a symmetric bound there covers exactly k/(n+1).

    Asserted link by link, per fit, in darts_scoring_path.py.
    """
    k = darts_rank(F(9, 10), n)
    return F(k, n + 1), k, required_rank(F(9, 10), n)


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
    # both predictions are probabilities, and they disagree -- that is the point
    for n in CAL_LENGTHS:
        p, lo, hi = predicted_two_rail_FALSIFIED(n)
        assert 0 <= p <= 1 and lo < hi, (n, p, lo, hi)
        p2, k, k_req = predicted_symmetric(n)
        assert 0 <= p2 <= 1 and 1 <= k <= n, (n, p2, k)
        assert k_req - k in (0, 1), (n, k, k_req)
    assert predicted_symmetric(10)[0] == F(10, 11)
    assert predicted_symmetric(15)[0] == F(14, 16)


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
        f"{'gap/s.e.':>9}  {'k_np':>4}  {'k_req':>5}  {'width':>9}  {'2-rail (dead)':>13}")

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
        pred, k_np, k_req = predicted_symmetric(cal)
        dead = float(predicted_two_rail_FALSIFIED(cal)[0])
        gap = (cov - float(pred)) / se if se > 0 else float("nan")
        say(f"  {cal:>7}  {cov:>9.4f}  {se:>7.4f}  {float(pred):>9.4f}  "
            f"{gap:>+9.2f}  {k_np:>4}  {k_req:>5}  {np.mean(widths):>9.4f}  "
            f"{dead:>13.4f}")

    say("")
    say("'predicted' is the exact coverage of the rank darts lands on, k_np/(n+1) --")
    say("derived, not fitted. 'gap/s.e.' is how far the measurement sits from it.")
    say("A gap within about +/-2 means the convention explains the coverage fully.")
    say("'2-rail (dead)' is the falsified hypothesis, printed for the contrast.")
    say("")
    say("k_req - k_np is 0 at cal_length 10, 30 and 50 and 1 at 15, so three of these")
    say("four cells are coincidence cells where darts lands on the correct rank by")
    say("arithmetic accident. That is why this four-cell table looked non-monotonic.")
    say("darts_scoring_path.py samples both bands and separates the in-sample")
    say("residual bias that remains at cal_length=50 from the convention.")
    say("")
    say("Honest scope, unchanged: iid by construction, so the guarantee SHOULD hold")
    say("exactly here. This is not a claim about real dependent series.")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
