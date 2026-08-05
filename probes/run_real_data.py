#!/usr/bin/env python3
"""Delivered coverage on REAL data: a library's shipped interval vs the interval
its OWN residuals support under the required order statistic.

Why this exists
---------------
Every other probe here runs on iid Gaussian draws, deliberately, so that the
guarantee should hold exactly and any miss is unambiguous. That is a clean
argument and a weak headline. This probe answers what a practitioner asks --
"does the interval my library hands me cover?" -- on real series.

The paired design, and why v1 was wrong
---------------------------------------
v1 of this probe built arm B from its own last-value residuals and its own
centre. That made the two arms incomparable: arm A's half-width landed on ranks
ABOVE n when scored against arm B's residual set, and arm B came out wider while
covering less, which is impossible for nested intervals. The delta measured the
difference between two harnesses, not the level-to-rank map.

v2 takes sktime's OWN residual matrix (`residuals_matrix_`, the offset-1
diagonal, exactly as run_sktime_river.py does) and sktime's OWN point forecast,
and changes ONE thing: how the level becomes a rank. Same data, same model, same
residuals, same centre.

Honest scope
------------
Real series are not exchangeable, so an absolute coverage miss is not
attributable to the convention on its own. The paired delta is the claim.
"""

import math
import sys
import warnings
from fractions import Fraction

import numpy as np

LEVELS = (0.90, 0.95)
OUT = "outputs/probe_output_real_data.txt"


# --------------------------------------------------------------------------
# arithmetic + self-check
# --------------------------------------------------------------------------
def required_rank(n, coverage):
    """1-based rank k = ceil((n+1) * coverage), exact. None when k > n."""
    k = math.ceil(Fraction(n + 1) * Fraction(coverage).limit_denominator(10**6))
    return k if k <= n else None


def delivered_coverage(k, n):
    return Fraction(k, n + 1)


def rank_of(threshold, scores):
    """Smallest 1-based rank of `scores` whose value is >= threshold."""
    s = np.sort(np.asarray(scores, dtype=float))
    return int(np.searchsorted(s, threshold, side="left") + 1)


def self_check():
    assert required_rank(9, 0.90) == 9
    assert required_rank(19, 0.95) == 19
    assert required_rank(39, 0.95) == 38
    assert required_rank(99, 0.99) == 99
    assert required_rank(6, Fraction(2, 3)) == 5
    for coverage, first_n in ((0.90, 9), (0.95, 19), (0.99, 99)):
        assert required_rank(first_n - 1, coverage) is None
        assert required_rank(first_n, coverage) is not None
    for n in range(2, 400):
        for coverage in (0.90, 0.95, Fraction(2, 3)):
            k = required_rank(n, coverage)
            if k is not None:
                assert delivered_coverage(k, n) >= Fraction(coverage).limit_denominator(10**6)
    assert rank_of(3.0, [1.0, 2.0, 3.0, 4.0]) == 3
    assert rank_of(9.0, [1.0, 2.0, 3.0]) == 4


self_check()


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_series(name, limit, min_len):
    from sktime.datasets import load_forecastingdata

    df, meta = load_forecastingdata(name)
    out = []
    for values in df["series_value"]:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size >= min_len:
            out.append(arr)
        if len(out) >= limit:
            break
    return out, meta


# --------------------------------------------------------------------------
# one series, one level: both arms off the SAME fitted object
# --------------------------------------------------------------------------
def run_cell(series, initial_window, coverage, method):
    import pandas as pd
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.conformal import ConformalIntervals
    from sktime.forecasting.naive import NaiveForecaster

    y_hist, y_test = series[:-1], series[-1]
    if len(y_hist) < initial_window + 4:
        return None
    y = pd.Series(y_hist, index=pd.RangeIndex(len(y_hist)))
    fh = ForecastingHorizon([1], is_relative=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci = ConformalIntervals(
                NaiveForecaster(strategy="last"),
                method=method,
                initial_window=initial_window,
            )
            ci.fit(y, fh=fh)
            interval = ci.predict_interval(fh=fh, coverage=[coverage])
            point = float(np.asarray(ci.predict(fh=fh)).ravel()[0])
            resid = np.diagonal(ci.residuals_matrix_.to_numpy(), offset=1)
    except Exception as exc:
        return {"error": type(exc).__name__}

    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    scores = np.abs(resid)
    n = scores.size
    if n < 2:
        return {"error": "too_few_residuals"}

    lo_a, hi_a = float(interval.iloc[0, 0]), float(interval.iloc[0, 1])
    half_a = (hi_a - lo_a) / 2.0

    # arm B: same residuals, same centre, only the level -> rank step differs
    k = required_rank(n, coverage)
    if k is None:
        half_b, cov_b, feasible = math.inf, True, False
    else:
        half_b = float(np.sort(scores)[k - 1])
        cov_b, feasible = abs(y_test - point) <= half_b, True

    return {
        "n": n,
        "required_rank": k,
        "feasible": feasible,
        "a_covered": bool(lo_a <= y_test <= hi_a),
        "a_width": hi_a - lo_a,
        "a_rank": rank_of(half_a, scores),
        "b_covered": bool(cov_b),
        "b_width": 2 * half_b if math.isfinite(half_b) else math.inf,
    }


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "m1_monthly_dataset"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    windows = (20, 40)

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 96)
    say("DELIVERED COVERAGE ON REAL DATA -- paired, both arms off the same fitted object")
    say("=" * 96)
    say("self_check() passed at import")
    say(f"dataset: {name}   series cap: {limit}")
    say("arm A: sktime ConformalIntervals via predict_interval, its own defaults")
    say("arm B: sktime's OWN residuals (residuals_matrix_, offset-1 diagonal) and")
    say("       its OWN point forecast, thresholded at rank ceil((n+1)*coverage)")
    say("")

    series, meta = load_series(name, limit, min_len=max(windows) + 6)
    say(f"series loaded: {len(series)}   frequency: {meta.get('frequency', '?')}")
    say("")

    for method in ("empirical", "conformal"):
        for coverage in LEVELS:
            for iw in windows:
                cells = [run_cell(s, iw, coverage, method) for s in series]
                good = [c for c in cells if c and "error" not in c]
                if not good:
                    errs = {c.get("error") for c in cells if c}
                    say(f"  {method:<10} {coverage:.2f} iw={iw:<3} -- no usable cells {errs}")
                    continue
                ns = [c["n"] for c in good]
                a_cov = np.mean([c["a_covered"] for c in good])
                b_cov = np.mean([c["b_covered"] for c in good])
                d = np.array([c["b_covered"] for c in good], float) - np.array(
                    [c["a_covered"] for c in good], float
                )
                se = d.std(ddof=1) / math.sqrt(d.size) if d.size > 1 else float("nan")
                infeas = sum(1 for c in good if not c["feasible"])
                say(f"  {method:<10} nominal {coverage:.2f}  initial_window={iw:<3} "
                    f"series={len(good):<4} n_resid median={int(np.median(ns))}")
                say(f"      arm A (shipped)        coverage {a_cov:.4f}   width "
                    f"{np.mean([c['a_width'] for c in good]):>12.3f}   "
                    f"lands on rank {int(np.median([c['a_rank'] for c in good]))} "
                    f"of {int(np.median(ns))}")
                say(f"      arm B (required rank)  coverage {b_cov:.4f}   width "
                    + ("     (+inf mixed)" if infeas else
                       f"{np.mean([c['b_width'] for c in good]):>12.3f}")
                    + f"   median required rank "
                      f"{int(np.median([c['required_rank'] for c in good if c['feasible']]))
                         if any(c['feasible'] for c in good) else 0}"
                      f" of {int(np.median(ns))}"
                    + (f"   [{infeas}/{len(good)} infeasible -> +inf]" if infeas else ""))
                say(f"      paired delta (B - A)   {d.mean():+.4f}  (s.e. {se:.4f})")
                if infeas:
                    # An infinite bound always covers, so mixing it into a mean width
                    # would compare arm B's FEASIBLE widths against arm A's ALL-cell
                    # widths -- different subsets, and arm B then looks narrower while
                    # covering more, which is the exact impossibility v1 tripped on.
                    # Restrict BOTH arms to the feasible subset instead.
                    f_ = [c for c in good if c["feasible"]]
                    if f_:
                        fd = np.array([c["b_covered"] for c in f_], float) - np.array(
                            [c["a_covered"] for c in f_], float
                        )
                        fse = fd.std(ddof=1) / math.sqrt(fd.size) if fd.size > 1 else float("nan")
                        say(f"      feasible cells only ({len(f_)}):  "
                            f"A {np.mean([c['a_covered'] for c in f_]):.4f} "
                            f"(width {np.mean([c['a_width'] for c in f_]):.3f})   "
                            f"B {np.mean([c['b_covered'] for c in f_]):.4f} "
                            f"(width {np.mean([c['b_width'] for c in f_]):.3f})   "
                            f"delta {fd.mean():+.4f} (s.e. {fse:.4f})")
                say("")

    say("A positive delta means the required rank covers more than the shipped call.")
    say("Real series are not exchangeable, so an absolute miss is not attributable to")
    say("the convention on its own -- the paired delta is what carries the claim.")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
