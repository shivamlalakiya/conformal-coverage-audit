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
import os
import sys
import warnings
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_report import format_cell, summarize  # noqa: E402

LEVELS = (0.90, 0.95)
OUT_TEMPLATE = "outputs/probe_output_real_data{suffix}.txt"

# Methods whose interval is two rails on the SIGNED residuals rather than a
# symmetric band on the absolute ones. `empirical` is absent from sktime's own
# ABS_RESIDUAL_BASED list (conformal.py:298) and takes np.quantile of `resids`
# at (1-c)/2 and (1+c)/2 with no sign flip (conformal.py:318-320), so its
# interval is asymmetric about the point forecast. Which arm B is the right one
# depends on this and on nothing else.
TWO_RAIL = ("empirical",)


# --------------------------------------------------------------------------
# arithmetic + self-check
# --------------------------------------------------------------------------
def out_path(template, dataset):
    """Dataset-suffixed output path, so a second dataset cannot clobber the first.

    m1_monthly is the primary and keeps the unsuffixed name that earlier commits
    and write-ups already reference.
    """
    if dataset == "m1_monthly_dataset":
        return template.format(suffix="")
    return template.format(suffix="_" + dataset.replace("_dataset", ""))


def required_rank(n, coverage):
    """1-based rank k = ceil((n+1) * coverage), exact. None when k > n."""
    k = math.ceil(Fraction(n + 1) * Fraction(coverage).limit_denominator(10**6))
    return k if k <= n else None


def delivered_coverage(k, n):
    return Fraction(k, n + 1)


def required_span(n, coverage):
    """The two-rail index pair delivering exactly ceil((n+1)*coverage)/(n+1).

    A one-rail bound needs a rank. A two-rail interval needs a SPAN. With signed
    scores R_(1) <= ... <= R_(n) and the conventions R_(0) = -inf and
    R_(n+1) = +inf, Pr(R_(a) <= R_(n+1) <= R_(b)) = (b - a)/(n + 1) exactly, so
    the requirement is b - a >= k = ceil((n+1)*coverage) and the smallest such
    span is k itself. The m = n + 1 - k excluded gaps are split as evenly as the
    arithmetic allows, which is what a correct implementation of the two-rail
    construction would do and does not depend on the interval being compared
    against.

    Both rails are finite exactly when m >= 2, i.e. n >= 2/alpha - 1. That is the
    single-rail floor with alpha halved, which is the general statement that any
    construction dividing alpha before resolving it multiplies the floor.

    Returns (a, b, k), 1-based, where a == 0 means the lower rail is -inf and
    b == n + 1 means the upper rail is +inf.
    """
    k = math.ceil(Fraction(n + 1) * Fraction(coverage).limit_denominator(10**6))
    m = n + 1 - k
    a = m // 2
    return a, a + k, k


def rank_of(threshold, scores):
    """Smallest 1-based rank of `scores` whose value is >= threshold."""
    s = np.sort(np.asarray(scores, dtype=float))
    return int(np.searchsorted(s, threshold, side="left") + 1)


def bracket_indices(t_lo, t_hi, scores_sorted):
    """The order-statistic indices an interval brackets, 1-based.

    j_lo counts the scores strictly below t_lo and j_hi those at or below t_hi,
    so {R_(j_lo+1) <= R_(n+1) <= R_(j_hi)} sits inside {t_lo <= R_(n+1) <= t_hi}
    and Proposition 2's distribution-free span is j_hi - j_lo - 1 gaps. Reported
    rather than asserted on, because ties move both counts and the property that
    matters -- arm B containing arm A -- is a statement about the values.
    """
    s = np.asarray(scores_sorted, dtype=float)
    return (int(np.searchsorted(s, t_lo, side="left")),
            int(np.searchsorted(s, t_hi, side="right")))


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
    assert bracket_indices(2.5, 3.5, [1.0, 2.0, 3.0, 4.0]) == (2, 3)
    assert bracket_indices(2.0, 4.0, [1.0, 2.0, 3.0, 4.0]) == (1, 4)

    # ---- the two-rail span, in exact arithmetic ---------------------------
    # The span delivers at least what was asked, and one gap less does not, so k
    # is minimal rather than merely sufficient.
    for n in range(2, 400):
        for coverage in (0.90, 0.95, Fraction(2, 3), Fraction(5, 7)):
            a, b, k = required_span(n, coverage)
            c = Fraction(coverage).limit_denominator(10**6)
            assert b - a == k and 0 <= a and b <= n + 1
            assert Fraction(k, n + 1) >= c > Fraction(k - 1, n + 1)
            assert abs((a) - (n + 1 - b)) <= 1, "excluded gaps not split evenly"
            # both rails finite exactly at the alpha/2 floor, not the alpha one
            assert (a >= 1 and b <= n) == (n + 1 - k >= 2)
            assert (a >= 1 and b <= n) == (n >= 2 / (1 - c) - 1)

    # ---- arm B contains arm A, over the whole range, before any data ------
    # `empirical` resolves both rails through numpy's default `linear`, whose
    # virtual index is h = 1 + q(n-1) clipped into [1, n]. Arm B nests arm A when
    # its lower index is at or below floor(h_lo) and its upper index at or above
    # ceil(h_hi). That is what makes the paired difference non-negative, and it is
    # a property of the two index rules rather than of the data -- so it is
    # checked here, exhaustively, rather than asserted on a sample. A symmetric
    # arm B built from |resid| has NO such property against a two-rail arm A, and
    # asserting it there is what this check exists to have prevented.
    for coverage in (0.90, 0.95, Fraction(2, 3), Fraction(5, 7)):
        c = Fraction(coverage).limit_denominator(10**6)
        for n in range(2, 2001):
            a, b, _ = required_span(n, coverage)
            if not (a >= 1 and b <= n):
                continue                      # a rail is infinite; nesting is trivial
            h_lo = min(max(1, 1 + (1 - c) / 2 * (n - 1)), n)
            h_hi = min(max(1, 1 + (1 + c) / 2 * (n - 1)), n)
            assert a <= math.floor(h_lo), (n, coverage, a, float(h_lo))
            assert b >= math.ceil(h_hi), (n, coverage, b, float(h_hi))


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
# one series: both arms off the SAME fitted object, at one or more levels
# --------------------------------------------------------------------------
def fit_series(series, initial_window, method, coverages, offset=1):
    """Fit once and read every level off the same fitted object.

    `ConformalIntervals(forecaster, method, initial_window)` does not take the
    coverage -- `predict_interval` does -- so the sliding-residual fit is shared
    across levels and refitting per level does the same work twice. The residual
    matrix, the point forecast and therefore both arms are identical either way;
    this is a compute change and not a measurement change, and the regression that
    holds it to that is reproducing a committed output byte for byte.

    Returns (per-level intervals, point forecast, residual diagonal, y_test) or a
    dict with an "error" key.
    """
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
            intervals = {c: ci.predict_interval(fh=fh, coverage=[c])
                         for c in coverages}
            point = float(np.asarray(ci.predict(fh=fh)).ravel()[0])
            # `offset` selects which diagonal of residuals_matrix_ is read. 1 is
            # what the shipped helper uses at a one-step horizon, and by the
            # alignment finding that diagonal holds TWO-step residuals -- so it is
            # not exchangeable with a one-step test residual whatever the data.
            # offset=0 is the correctly aligned set. Both are needed: the first is
            # what ships, the second is what makes an ABSOLUTE coverage number
            # attributable to the index convention rather than to the misalignment.
            resid = np.diagonal(ci.residuals_matrix_.to_numpy(), offset=offset)
    except Exception as exc:
        return {"error": type(exc).__name__}
    return intervals, point, resid, y_test


def run_cells(series, initial_window, method, coverages, offset=1):
    """One fit, one record per level. `run_cell` is the single-level wrapper."""
    fitted = fit_series(series, initial_window, method, coverages, offset=offset)
    if fitted is None or isinstance(fitted, dict):
        return {c: fitted for c in coverages}
    intervals, point, resid, y_test = fitted
    return {c: _score(intervals[c], point, resid, y_test, c, method)
            for c in coverages}


def run_cell(series, initial_window, coverage, method):
    return run_cells(series, initial_window, method, (coverage,))[coverage]


def _score(interval, point, resid, y_test, coverage, method):
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    n = resid.size
    if n < 2:
        return {"error": "too_few_residuals"}

    lo_a, hi_a = float(interval.iloc[0, 0]), float(interval.iloc[0, 1])
    a_covered = bool(lo_a <= y_test <= hi_a)

    if method in TWO_RAIL:
        # Arm A is two rails on the signed residuals. An arm B built symmetrically
        # from |resid| would differ from it in the SCORE SET and in the GEOMETRY as
        # well as in the rank -- three changes, not one -- and would not contain it,
        # so a paired difference across the two would carry the geometry rather than
        # the level-to-rank map. Arm B is therefore the same two-rail construction
        # with the required span substituted for the two levels.
        signed = np.sort(resid)
        a_idx, b_idx, k = required_span(n, coverage)
        lo_b = -math.inf if a_idx == 0 else point + float(signed[a_idx - 1])
        hi_b = math.inf if b_idx == n + 1 else point + float(signed[b_idx - 1])
        feasible = a_idx >= 1 and b_idx <= n
        j_lo, j_hi = bracket_indices(lo_a - point, hi_a - point, signed)
        # Containment, per fit, on the VALUES -- which is what makes the paired
        # difference non-negative. self_check() establishes the index inequality
        # behind it for every n before any data is read; this catches a series
        # whose residual set breaks an assumption that sweep did not model.
        assert lo_b <= lo_a + 1e-9 and hi_b >= hi_a - 1e-9, (
            n, coverage, lo_a, lo_b, hi_a, hi_b)
        return {
            "n": n,
            "required_rank": k,          # a span in gaps, not a rank
            "two_rail": True,
            "nests": True,               # asserted above, per fit, on the values
            "feasible": feasible,
            "a_covered": a_covered,
            "a_width": hi_a - lo_a,
            "a_rank": j_hi - j_lo - 1,   # the span arm A guarantees, in gaps
            "b_covered": bool(lo_b <= y_test <= hi_b),
            "b_width": hi_b - lo_b,
        }

    # Symmetric band on the absolute residuals: arm A is centre +/- one threshold,
    # so arm B is the same band at the required rank and contains it.
    scores = np.abs(resid)
    half_a = (hi_a - lo_a) / 2.0
    k = required_rank(n, coverage)
    if k is None:
        half_b, cov_b, feasible = math.inf, True, False
    else:
        half_b = float(np.sort(scores)[k - 1])
        cov_b, feasible = abs(y_test - point) <= half_b, True
        assert half_b >= half_a - 1e-9, (n, coverage, half_a, half_b)

    return {
        "n": n,
        "required_rank": k,
        "two_rail": False,
        "nests": True,
        "feasible": feasible,
        "a_covered": a_covered,
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
    say("       its OWN point forecast, at the required order statistic -- matched to")
    say("       arm A's GEOMETRY as well as to its scores and its centre:")
    say("         method=empirical  two rails on the signed residuals, so arm B is the")
    say("                           index pair spanning ceil((n+1)*coverage) gaps")
    say("         method=conformal  symmetric band on the absolute residuals, so arm B")
    say("                           is the rank ceil((n+1)*coverage) threshold")
    say("")

    series, meta = load_series(name, limit, min_len=max(windows) + 6)
    say(f"series loaded: {len(series)}   frequency: {meta.get('frequency', '?')}")
    say("")

    # summarize/format_cell live in paired_report.py and are shared with the
    # statsforecast, darts and tabular arms. This module printed its own copy of
    # them until the copy grew a bug the others did not have -- the count of units
    # that changed status, read off the delta instead of counted.
    # Fit once per (method, window) and keep every level's records. The emission
    # loop below then walks method -> coverage -> window, which is the order the
    # committed outputs already have: halving the fits must not reorder a file that
    # a fixture and three parsers read.
    cells_by = {}
    for method in ("empirical", "conformal"):
        for iw in windows:
            per = [run_cells(s, iw, method, LEVELS) for s in series]
            for coverage in LEVELS:
                cells_by[(method, coverage, iw)] = [p[coverage] for p in per]

    for method in ("empirical", "conformal"):
        for coverage in LEVELS:
            for iw in windows:
                cells = cells_by[(method, coverage, iw)]
                s = summarize(cells)
                if s is None:
                    errs = {c.get("error") for c in cells if c}
                    say(f"  {method:<10} {coverage:.2f} iw={iw:<3} -- no usable cells {errs}")
                    continue
                for ln in format_cell(f"{method:<10} nominal {coverage:.2f}  "
                                      f"initial_window={iw:<3}", s):
                    say(ln)
                say("")

    say("A positive delta means the required rank covers more than the shipped call.")
    say("`empirical` is two rails on the SIGNED residuals, so its arm B is the same")
    say("two-rail construction at the required SPAN and its landed figure is a span in")
    say("gaps. `conformal` is a symmetric band on the absolute ones and its arm B is a")
    say("rank. Arm B contains arm A in both, asserted per fit rather than argued.")
    say("Real series are not exchangeable, so an absolute miss is not attributable to")
    say("the convention on its own -- the paired delta is what carries the claim.")

    out = out_path(OUT_TEMPLATE, name)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
