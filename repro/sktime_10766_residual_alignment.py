#!/usr/bin/env python3
"""sktime #10766: the residual diagonal read at a one-step horizon holds two-step errors.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
`ConformalIntervals` builds `residuals_matrix_` by expanding-window backtesting, then
`_predict_interval_series` reads one diagonal of it to calibrate the interval. At a
one-step horizon the diagonal it reads is offset by one from the one holding one-step
residuals, so the calibration set consists of TWO-step errors.

Two-step errors are larger than one-step errors for any process with positive
autocorrelation in its increments, so the interval is wider than the level asks for --
and, more importantly, the calibration scores are not exchangeable with the test
residual at any horizon, so the finite-sample guarantee does not apply whatever the data.

This script does not argue that. It reads both diagonals out of the library's own fitted
object and compares them to residuals it computes directly, which settles which diagonal
holds which horizon without appealing to the interval at all.
"""

import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")


def main():
    import pandas as pd
    import sktime
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.conformal import ConformalIntervals
    from sktime.forecasting.naive import NaiveForecaster

    print(f"sktime {sktime.__version__}, numpy {np.__version__}")
    print()
    rng = np.random.default_rng(0)
    y_vals = np.cumsum(rng.normal(size=60)) + 100.0
    y = pd.Series(y_vals, index=pd.RangeIndex(len(y_vals)))
    fh = ForecastingHorizon([1], is_relative=True)
    window = 20

    ci = ConformalIntervals(NaiveForecaster(strategy="last"), method="empirical",
                            initial_window=window)
    ci.fit(y, fh=fh)
    M = ci.residuals_matrix_.to_numpy()
    print(f"residuals_matrix_ shape {M.shape}, initial_window {window}")
    print()

    # NaiveForecaster(strategy="last") predicts y[t] for every horizon, so the h-step
    # error at origin t is y[t+h] - y[t]. Computed here directly, with no library call.
    one_step = np.array([y_vals[t + 1] - y_vals[t]
                         for t in range(window, len(y_vals) - 1)])
    two_step = np.array([y_vals[t + 2] - y_vals[t]
                         for t in range(window, len(y_vals) - 2)])

    print("Naive(last) forecasts y[t] at every horizon, so the h-step error at origin t")
    print("is y[t+h] - y[t]. Both sets below are computed from the series directly.")
    print()
    print(f"{'set':<28}{'n':>5}{'mean |e|':>11}{'sd':>10}")
    print("-" * 56)
    print(f"{'one-step, computed here':<28}{one_step.size:>5}"
          f"{np.abs(one_step).mean():>11.4f}{one_step.std(ddof=1):>10.4f}")
    print(f"{'two-step, computed here':<28}{two_step.size:>5}"
          f"{np.abs(two_step).mean():>11.4f}{two_step.std(ddof=1):>10.4f}")
    print()

    print(f"{'diagonal':<28}{'n':>5}{'mean |e|':>11}{'sd':>10}"
          f"{'matches':>22}")
    print("-" * 78)
    rows = []
    for off in (0, 1):
        d = np.diagonal(M, offset=off)
        d = d[np.isfinite(d)]
        if d.size < 3:
            print(f"{'offset ' + str(off):<28}{d.size:>5}{'too few finite entries':>43}")
            continue

        def closeness(ref):
            k = min(len(d), len(ref))
            return float(np.abs(np.sort(np.abs(d[:k])) - np.sort(np.abs(ref[:k]))).mean())

        c1, c2 = closeness(one_step), closeness(two_step)
        match = "one-step" if c1 < c2 else "TWO-step"
        rows.append({"off": off, "n": int(d.size), "match": match, "c1": c1, "c2": c2})
        print(f"{'offset ' + str(off):<28}{d.size:>5}{np.abs(d).mean():>11.4f}"
              f"{d.std(ddof=1):>10.4f}{match:>22}")

    print()
    # which diagonal does the shipped one-step path read? offset=1, per the source.
    shipped = [r for r in rows if r["off"] == 1]
    correct = [r for r in rows if r["off"] == 0]
    assert shipped and correct, (
        "both diagonals must be readable, or the comparison is not available")
    s, c = shipped[0], correct[0]
    print(f"offset 1 -- the diagonal the one-step path reads -- matches {s['match']}")
    print(f"offset 0 matches {c['match']}")
    print()
    if s["match"] == "TWO-step" and c["match"] == "one-step":
        print("REPRODUCES. The diagonal read at a one-step horizon holds two-step "
              "errors, and the correctly aligned set is the adjacent diagonal. The "
              "calibration scores are therefore not exchangeable with the test "
              "residual, at any horizon and for any data.")
        return 0
    print(f"does not reproduce: offset 1 matched {s['match']} and offset 0 matched "
          f"{c['match']}. If offset 1 now matches one-step, the alignment is fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
