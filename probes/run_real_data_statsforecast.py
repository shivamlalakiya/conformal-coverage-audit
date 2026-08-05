#!/usr/bin/env python3
"""W3, statsforecast arm: delivered coverage on real data, paired.

Same design as run_real_data.py, same dataset, same base model class, so the
two arms are comparable across libraries:

  arm A   statsforecast's shipped interval, its own API, its own defaults
  arm B   the SAME conformity scores and the SAME point forecast, thresholded
          at the required order statistic ceil((m+1) * coverage)

The scores are not recomputed. `_add_conformal_distribution_intervals` and
`_add_conformal_error_intervals` are wrapped so the probe records the exact
`cs` array and `fcst["mean"]` the library passed into its own quantile call,
and both arms are built from those. The only difference between the arms is how
the level becomes a rank.

Two things this arm shows that the sktime arm could not
------------------------------------------------------
1. `ConformalIntervals(n_windows=2)` is the DEFAULT, and at m=2 the required
   rank for a 0.90 interval is ceil(3*0.90) = 3 > 2. No valid finite bound
   exists at the default setting, for any choice of index. statsforecast
   returns a finite interval anyway.
2. The distribution method interpolates over a SYMMETRISED score set of size
   2m, so its half-width is not an order statistic of |cs| at all. The probe
   records the rank of |cs| that arm A's half-width lands on, which makes the
   interpolation visible in the table rather than argued.

Honest scope, identical to the sktime arm: real series are not exchangeable, so
the absolute coverage is not attributable to the convention. The paired delta
is the claim.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_report import format_cell, summarize  # noqa: E402
from run_real_data import rank_of, required_rank  # noqa: E402

LEVELS = (90, 95)
N_WINDOWS = (2, 5, 10, 20, 50)
METHODS = ("conformal_distribution", "conformal_error")
OUT_TEMPLATE = "outputs/probe_output_real_data_statsforecast{suffix}.txt"

CAPTURED = []


def install_spy():
    """Record the scores and centre statsforecast hands to its own quantile."""
    import statsforecast.models as M

    def wrap(fn):
        def inner(fcst, cs, level):
            CAPTURED.append((np.array(cs, copy=True),
                             np.array(fcst["mean"], copy=True)))
            return fn(fcst=fcst, cs=cs, level=level)
        return inner

    M._add_conformal_distribution_intervals = wrap(
        M._add_conformal_distribution_intervals)
    M._add_conformal_error_intervals = wrap(
        M._add_conformal_error_intervals)


def self_check():
    # the feasibility boundary that makes the default setting the headline
    assert required_rank(2, 0.90) is None, "m=2 must be infeasible at 0.90"
    assert required_rank(2, 0.95) is None
    assert required_rank(9, 0.90) == 9
    assert required_rank(10, 0.90) == 10
    assert required_rank(19, 0.95) == 19
    assert required_rank(20, 0.95) == 20
    assert required_rank(50, 0.90) == 46
    # the smallest m that admits a valid finite bound, per level
    for cov, first in ((0.90, 9), (0.95, 19)):
        assert required_rank(first - 1, cov) is None
        assert required_rank(first, cov) is not None


self_check()


def run_cell(series, n_windows, level, method):
    from statsforecast.models import Naive
    from statsforecast.utils import ConformalIntervals

    y_hist, y_test = series[:-1].astype(float), float(series[-1])
    coverage = level / 100.0
    # statsforecast itself caps n_windows at (n_samples - 1) // h, so a short
    # series silently calibrates on fewer windows. The probe reads the actual
    # count off the captured array rather than assuming the requested one.
    if y_hist.size < n_windows + 2:
        return {"error": "too_short"}

    CAPTURED.clear()
    try:
        model = Naive(prediction_intervals=ConformalIntervals(
            n_windows=n_windows, h=1, method=method))
        res = model.forecast(y=y_hist, h=1, level=[level])
    except Exception as exc:
        return {"error": type(exc).__name__}
    if not CAPTURED:
        return {"error": "no_conformal_call"}

    cs, mean = CAPTURED[-1]
    scores = np.abs(np.asarray(cs, dtype=float).ravel())
    m = scores.size
    if m < 2:
        return {"error": "too_few_windows"}
    centre = float(np.asarray(mean).ravel()[0])
    lo_a = float(np.asarray(res[f"lo-{level}"]).ravel()[0])
    hi_a = float(np.asarray(res[f"hi-{level}"]).ravel()[0])
    half_a = (hi_a - lo_a) / 2.0

    k = required_rank(m, coverage)
    if k is None:
        half_b, feasible = math.inf, False
    else:
        half_b, feasible = float(np.sort(scores)[k - 1]), True

    return {
        "n": m,
        "required_rank": k if k is not None else m + 1,
        "feasible": feasible,
        "a_covered": bool(lo_a <= y_test <= hi_a),
        "a_width": hi_a - lo_a,
        "a_rank": rank_of(half_a, scores),
        "b_covered": bool(abs(y_test - centre) <= half_b),
        "b_width": 2 * half_b if math.isfinite(half_b) else math.inf,
    }


def main():
    from run_real_data import load_series, out_path

    name = sys.argv[1] if len(sys.argv) > 1 else "m1_monthly_dataset"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    install_spy()

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say("W3, STATSFORECAST ARM -- delivered coverage on real data, paired")
    say("=" * 100)
    say("self_check() passed at import")
    say(f"dataset: {name}   series cap: {limit}   model: Naive (matches the sktime arm)")
    say("arm A: statsforecast Naive(prediction_intervals=ConformalIntervals(...)),")
    say("       its own API at its own defaults")
    say("arm B: the SAME captured conformity scores and the SAME point forecast,")
    say("       thresholded at rank ceil((m+1)*coverage)")
    say("")
    say("m is the number of calibration windows ACTUALLY used, read off the array")
    say("statsforecast passed to its own quantile call -- it caps n_windows at")
    say("(len(y) - 1) // h for short series.")
    say("")
    say("Feasibility, exact and prior to any measurement:")
    for cov in (0.90, 0.95):
        first = next(m for m in range(2, 200) if required_rank(m, cov) is not None)
        say(f"    coverage {cov:.2f}:  no valid finite bound exists below m={first}"
            f"   (default n_windows=2 -> "
            f"{'INFEASIBLE' if required_rank(2, cov) is None else 'feasible'})")
    say("")

    series, meta = load_series(name, limit, min_len=max(N_WINDOWS) + 6)
    say(f"series loaded: {len(series)}   frequency: {meta.get('frequency', '?')}")

    for method in METHODS:
        for level in LEVELS:
            say("")
            say(f"method={method}   nominal {level / 100:.2f}")
            for nw in N_WINDOWS:
                recs = [run_cell(s, nw, level, method) for s in series]
                s = summarize(recs)
                for ln in format_cell(f"n_windows={nw:<3}", s):
                    say(ln)

    say("")
    say("A positive delta means the required rank covers more than the shipped call.")
    say("Real series are not exchangeable, so an absolute miss is not attributable to")
    say("the convention on its own -- the paired delta is what carries the claim.")
    say("")
    say("The n_windows=2 rows are the default configuration. Where arm B reads")
    say("+inf-infeasible there, no index into a two-element score set supports a")
    say("0.90 or 0.95 bound, and arm A still returns a finite interval.")

    out = out_path(OUT_TEMPLATE, name)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
