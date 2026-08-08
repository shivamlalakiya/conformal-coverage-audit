#!/usr/bin/env python3
"""What ConformalNaiveModel actually does -- read and instrumented, not derived.

Why this exists
---------------
run_darts_tighten.py carried a prediction column built by hand from the
assumption that darts forms a two-sided interval out of two independent
uncorrected rails, one at level 0.05 and one at 0.95. The measurement falsified
it by up to 27 standard errors. That script's docstring says the next step is to
read the scoring path rather than derive a third time. This is that step.

What the source says (darts 0.46.1)
-----------------------------------
conformal_models.py:1681 `ConformalNaiveModel._calibrate_interval`

    def q_hat_from_residuals(residuals_):
        return np.quantile(residuals_, q=self.interval_range_sym,
                           method="higher", axis=2)...
    if self.symmetric:
        q_hat = q_hat_from_residuals(residuals)
        return -q_hat, q_hat[:, :, ::-1]

and :1717 `_residuals_metric` returns `metrics.ae` when `symmetric` -- absolute
error. `interval_range_sym` is `interval_range` under `symmetric=True` (:165-167),
i.e. q_high - q_low = 0.90 for quantiles [0.05, 0.5, 0.95].

So there is ONE score set (|residual|), ONE level (0.90, uncorrected), and the
interval is centre +/- that single threshold. Not two rails. The falsified
column modelled a construction the library does not use.

Corrected prediction, and it is still exact rather than fitted:
    numpy method='higher' at level c over n scores selects 1-based rank
        k = ceil(c * (n - 1)) + 1
    a symmetric bound at the k-th of n exchangeable absolute scores covers with
    probability exactly k / (n + 1).

This probe verifies each link in that chain per fit -- the captured score set,
the rank, the returned q_hat, and the returned interval -- then measures
coverage against it, and adds the paired required-rank arm off the same scores
and the same centre (the design run_real_data.py uses).

Two blocks, because the residuals darts scores on are IN-SAMPLE by default
-------------------------------------------------------------------------
Block 1 fits the base model on the whole history, which is what a user does and
what run_darts_tighten.py did; its calibration residuals are in-sample and
therefore optimistically small.  Block 2 fits the base model on a prefix and
calibrates on the held-out remainder, so the scores are genuinely
out-of-sample and aligned with the test point.  The difference between the
two blocks is how much of the deficit is the level->rank map and how much is
in-sample residual bias -- a decomposition the earlier probe could not make.
"""

import math
from fractions import Fraction as F

import numpy as np

REPS = 2000
# k_req - k_np is 0 exactly at n = 9, 10, 19, 20, 29, 30, ... and 1 everywhere
# else (printed in full below). run_darts_tighten.py used 10/15/30/50, which is
# three coincidence cells and one deficit cell -- that accident is what made its
# result read as non-monotonic. These six sample both bands three times each.
CAL_LENGTHS = (10, 15, 30, 35, 50, 55)
QUANTILES = [0.05, 0.5, 0.95]  # a 90% two-sided interval
INTERVAL_RANGE = F(9, 10)
SERIES_LEN = 120
FIT_PREFIX = 60  # block 2: base model sees only y[:60]
SEED = 20260805
OUT = "outputs/probe_output_darts_scoring_path.txt"


def numpy_higher_rank(c, n):
    """1-based rank numpy's method='higher' selects at level c over n values."""
    return math.ceil(c * (n - 1)) + 1


def required_rank(c, n):
    """1-based rank a valid finite-sample symmetric bound needs. None if > n.

    Argument order is (level, n) for historical call sites; delegates to the
    package's (n, coverage) form.
    """
    from conformal_coverage import required_rank as _rr
    return _rr(n, c)


def exact_coverage(k, n):
    """Coverage of a symmetric bound at the k-th of n exchangeable scores."""
    return F(k, n + 1)


def self_check():
    # the rank model, against numpy itself, on every cell this probe uses and more
    for n in range(2, 200):
        for c in (F(9, 10), F(19, 20), F(1, 2), F(2, 3)):
            got = float(np.quantile(np.arange(1, n + 1, dtype=float), float(c),
                                    method="higher"))
            assert abs(got - numpy_higher_rank(c, n)) < 1e-9, (n, c, got)
    # hand-computed cells for the four calibration lengths in this probe
    assert numpy_higher_rank(INTERVAL_RANGE, 10) == 10  # ceil(9.0)+1
    assert numpy_higher_rank(INTERVAL_RANGE, 15) == 14  # ceil(12.6)+1
    assert numpy_higher_rank(INTERVAL_RANGE, 30) == 28  # ceil(26.1)+1
    assert numpy_higher_rank(INTERVAL_RANGE, 35) == 32  # ceil(30.6)+1
    assert numpy_higher_rank(INTERVAL_RANGE, 50) == 46  # ceil(44.1)+1
    assert numpy_higher_rank(INTERVAL_RANGE, 55) == 50  # ceil(48.6)+1
    # the deficit is 0 or 1 and never 2 -- the claim run_darts.py records
    for n in range(9, 2001):
        assert required_rank(INTERVAL_RANGE, n) - numpy_higher_rank(INTERVAL_RANGE, n) in (0, 1)
    # and 0 exactly on the coincidence band
    assert {n for n in range(9, 61)
            if required_rank(INTERVAL_RANGE, n) == numpy_higher_rank(INTERVAL_RANGE, n)} == {
        9, 10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60}
    # required rank, and the feasibility boundary for a 0.90 symmetric bound
    assert required_rank(INTERVAL_RANGE, 9) == 9
    assert required_rank(INTERVAL_RANGE, 8) is None
    assert required_rank(INTERVAL_RANGE, 10) == 10
    assert required_rank(INTERVAL_RANGE, 50) == 46
    # the map is what makes the two disagree, so they must differ somewhere
    assert any(numpy_higher_rank(INTERVAL_RANGE, n) != required_rank(INTERVAL_RANGE, n)
               for n in CAL_LENGTHS)
    for n in range(9, 400):
        k = required_rank(INTERVAL_RANGE, n)
        assert exact_coverage(k, n) >= INTERVAL_RANGE


self_check()


def install_spy(ConformalNaiveModel, sink):
    """Capture the score set and q_hat darts computes, without changing them."""
    original = ConformalNaiveModel._calibrate_interval

    def spy(self, residuals):
        out = original(self, residuals)
        sink.append((np.array(residuals, copy=True), out))
        return out

    ConformalNaiveModel._calibrate_interval = spy
    return original


def run_block(cal, reps, out_of_sample, sink, TimeSeries, ConformalNaiveModel,
              LinearRegressionModel):
    """One (cal_length, block) cell. Returns per-fit records."""
    rng = np.random.default_rng(SEED + cal + (10_000 if out_of_sample else 0))
    recs = []
    for _ in range(reps):
        y = rng.standard_normal(SERIES_LEN)
        hist, test = y[:-1], float(y[-1])
        base = LinearRegressionModel(lags=1)
        base.fit(TimeSeries.from_values(hist[:FIT_PREFIX] if out_of_sample else hist))
        cm = ConformalNaiveModel(model=base, quantiles=QUANTILES, cal_length=cal)

        sink.clear()
        vals = cm.predict(
            n=1,
            series=TimeSeries.from_values(hist) if out_of_sample else None,
            predict_likelihood_parameters=True,
            num_samples=1,
        ).values().ravel()
        assert len(sink) == 1, f"expected one calibration per forecast, got {len(sink)}"

        residuals, q_hat = sink[0]
        scores = np.asarray(residuals, dtype=float).ravel()
        n = scores.size
        lo, mid, hi = float(vals[0]), float(vals[1]), float(vals[2])

        # --- link 1: the score set is absolute errors (metrics.ae), so non-negative
        assert (scores >= 0).all(), "scores are not absolute errors"
        # --- link 2: q_hat is the rank numpy's method='higher' lands on
        k_np = numpy_higher_rank(INTERVAL_RANGE, n)
        expected = float(np.sort(scores)[k_np - 1])
        got = float(np.asarray(q_hat[1]).ravel()[0])
        assert math.isclose(got, expected, rel_tol=0, abs_tol=1e-9), (n, got, expected)
        # --- link 3: the interval is centre +/- that ONE threshold, not two rails
        assert math.isclose(hi - mid, got, rel_tol=0, abs_tol=1e-9)
        assert math.isclose(mid - lo, got, rel_tol=0, abs_tol=1e-9)

        k_req = required_rank(INTERVAL_RANGE, n)
        half_b = math.inf if k_req is None else float(np.sort(scores)[k_req - 1])
        recs.append({
            "n": n,
            "k_np": k_np,
            "k_req": k_req,
            "a_covered": lo <= test <= hi,
            "a_half": got,
            "b_covered": abs(test - mid) <= half_b,
            "b_half": half_b,
        })
    return recs


def main():
    from darts import TimeSeries
    from darts.models import ConformalNaiveModel, LinearRegressionModel

    sink = []
    install_spy(ConformalNaiveModel, sink)

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say("DARTS SCORING PATH -- instrumented, and it replaces a falsified prediction")
    say("=" * 100)
    say("self_check() passed at import (rank model verified against numpy on 792 cells)")
    say("")
    say("What the source does (conformal_models.py:1681, :1717, :165-167):")
    say("  scores = |residual|            (metrics.ae, symmetric=True is the default)")
    say("  level  = interval_range_sym    = q_high - q_low = 0.90, UNCORRECTED")
    say("  q_hat  = np.quantile(scores, 0.90, method='higher')")
    say("  interval = centre -/+ q_hat    -- ONE threshold, not two rails")
    say("")
    say("Every fit below asserts all three links: scores non-negative, q_hat equals")
    say("the order statistic method='higher' selects, and the interval is symmetric")
    say("about the median. Any failure aborts the run.")
    say("")
    say(f"{REPS} fits per cell, iid Gaussian series of {SERIES_LEN}, "
        f"LinearRegressionModel(lags=1), quantiles={QUANTILES}")
    say("requested two-sided coverage: 0.9000")
    say("")
    say("Deficit map, exact and complete over n = 9..60 -- k_req - k_np, no simulation:")
    row = "  "
    for n in range(9, 61):
        row += f"{required_rank(INTERVAL_RANGE, n) - numpy_higher_rank(INTERVAL_RANGE, n)}"
        row += " " if n % 10 else "  "
    say(row)
    say("  n=9 ......................................................... n=60")
    say(f"  coincidence cells (deficit 0): "
        f"{[n for n in range(9, 61) if required_rank(INTERVAL_RANGE, n) == numpy_higher_rank(INTERVAL_RANGE, n)]}")
    say("  the deficit is 0 or 1 and never 2, verified to n=2000 in self_check()")
    say(f"  below n=9 no valid finite 0.90 symmetric bound exists at all "
        f"(k_req > n for n <= 8)")

    for out_of_sample in (False, True):
        say("")
        say("-" * 100)
        if out_of_sample:
            say(f"BLOCK 2 -- base model fitted on y[:{FIT_PREFIX}], calibrated on the "
                f"held-out remainder")
            say("           calibration scores are OUT-OF-SAMPLE, so exchangeable with "
                "the test point")
        else:
            say("BLOCK 1 -- base model fitted on the whole history (what a user does)")
            say("           calibration scores are IN-SAMPLE, so optimistically small")
        say("-" * 100)
        say(f"  {'cal':>4}  {'n':>3}  {'k_np':>4}  {'k_req':>5}  {'A meas':>8}  "
            f"{'k/(n+1)':>8}  {'gap/s.e.':>9}  {'B meas':>8}  {'delta':>8}  "
            f"{'d s.e.':>7}  {'A half':>7}  {'B half':>7}")

        for cal in CAL_LENGTHS:
            recs = run_block(cal, REPS, out_of_sample, sink, TimeSeries,
                             ConformalNaiveModel, LinearRegressionModel)
            ns = {r["n"] for r in recs}
            assert len(ns) == 1, f"calibration size varied within a cell: {ns}"
            n = ns.pop()
            k_np, k_req = recs[0]["k_np"], recs[0]["k_req"]

            a = np.array([r["a_covered"] for r in recs], float)
            b = np.array([r["b_covered"] for r in recs], float)
            a_cov, b_cov = a.mean(), b.mean()
            se_a = a.std(ddof=1) / math.sqrt(a.size)
            d = b - a
            se_d = d.std(ddof=1) / math.sqrt(d.size)
            pred = float(exact_coverage(k_np, n))
            gap = (a_cov - pred) / se_a if se_a > 0 else float("nan")

            say(f"  {cal:>4}  {n:>3}  {k_np:>4}  {k_req if k_req else 'inf':>5}  "
                f"{a_cov:>8.4f}  {pred:>8.4f}  {gap:>+9.2f}  {b_cov:>8.4f}  "
                f"{d.mean():>+8.4f}  {se_d:>7.4f}  "
                f"{np.mean([r['a_half'] for r in recs]):>7.4f}  "
                f"{np.mean([r['b_half'] for r in recs]):>7.4f}")

    say("")
    say("Reading the table")
    say("-----------------")
    say("k_np    the rank darts lands on: ceil(0.90*(n-1)) + 1")
    say("k_req   the rank a valid 0.90 bound needs: ceil((n+1)*0.90)")
    say("A       darts as shipped;  B       same scores, same centre, required rank")
    say("k/(n+1) exact coverage of rank k_np under exchangeability -- derived, not fitted")
    say("gap     (A - k/(n+1)) in standard errors. Within about +/-2 means the")
    say("        level->rank map explains darts' coverage completely.")
    say("")
    say("The falsified column in run_darts_tighten.py predicted 0.7273 at cal_length=10")
    say("from a two-rail model. The one-threshold model above predicts 10/11 = 0.9091")
    say("for the same cell. Compare both against the measurement in that file.")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
