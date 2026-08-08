#!/usr/bin/env python3
"""W3, darts arm: delivered coverage on real data, paired.

Same design as run_real_data.py and run_real_data_statsforecast.py:

  arm A   darts' shipped interval, its own API, its own defaults
  arm B   the SAME captured calibration scores and the SAME centre, thresholded
          at the required order statistic ceil((n+1) * coverage)

The scores are captured out of `ConformalNaiveModel._calibrate_interval`, so arm
B is built from exactly the array darts handed to its own `np.quantile` call.
darts_scoring_path.py establishes what that call does and asserts it per fit;
this probe takes it as given and measures on real series.

Reproduction, two commands, because darts and sktime cannot share a venv
-----------------------------------------------------------------------
    python probes/export_series.py m1_monthly_dataset 250 /tmp/m1_monthly.npz 70
    python probes/run_real_data_darts.py /tmp/m1_monthly.npz

Calibration lengths sample both deficit bands
---------------------------------------------
k_req - k_np is 0 at n = 9, 10, 19, 20, 29, 30, ... and 1 everywhere else, so a
cell list that happens to land on the coincidence band shows a zero delta for
arithmetic reasons rather than empirical ones. 10/30/50 are coincidence cells
and 15/35/55 are deficit cells, three of each.

Two departures from the sktime arm, both stated because they matter
------------------------------------------------------------------
1. `ConformalNaiveModel` wraps a GlobalForecastingModel, so the base model is
   `LinearRegressionModel(lags=1)` rather than a last-value naive forecaster.
   The point forecast therefore differs from the sktime arm's; the paired delta
   does not depend on it, since both arms share the same centre.
2. darts computes its calibration residuals from historical forecasts of a
   model already fitted on the whole input series, so they are in-sample.
   darts_scoring_path.py measures that bias separately on synthetic data; here
   it is simply part of what ships.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_report import format_cell, summarize  # noqa: E402
from run_real_data import out_path, rank_of, required_rank  # noqa: E402

CAL_LENGTHS = (10, 15, 30, 35, 50, 55)
LEVELS = (0.90, 0.95)
OUT_TEMPLATE = "outputs/probe_output_real_data_darts{suffix}.txt"

CAPTURED = []


def quantiles_for(coverage):
    """darts takes explicit quantiles; a two-sided interval is symmetric."""
    a = (1.0 - coverage) / 2.0
    return [round(a, 6), 0.5, round(1.0 - a, 6)]


def numpy_higher_rank(c, n):
    return math.ceil(c * (n - 1)) + 1


def self_check():
    assert quantiles_for(0.90) == [0.05, 0.5, 0.95]
    assert quantiles_for(0.95) == [0.025, 0.5, 0.975]
    # the coincidence band, which is why these cal lengths were chosen
    coincide = {n for n in CAL_LENGTHS
                if required_rank(n, 0.90) == numpy_higher_rank(0.90, n)}
    assert coincide == {10, 30, 50}, coincide
    deficit = {n for n in CAL_LENGTHS
               if required_rank(n, 0.90) == numpy_higher_rank(0.90, n) + 1}
    assert deficit == {15, 35, 55}, deficit
    # at 0.95 the bands sit elsewhere, so the same cells are not all coincidences
    assert required_rank(10, 0.95) is None  # ceil(11*0.95) = 11 > 10
    assert required_rank(20, 0.95) == 20


self_check()


def install_spy(ConformalNaiveModel):
    original = ConformalNaiveModel._calibrate_interval

    def spy(self, residuals):
        out = original(self, residuals)
        CAPTURED.append(np.array(residuals, copy=True))
        return out

    ConformalNaiveModel._calibrate_interval = spy


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    meta = [str(x) for x in z["meta"]]
    keys = sorted((k for k in z.files if k.startswith("s")),
                  key=lambda k: int(k[1:]))
    return [np.asarray(z[k], dtype=float) for k in keys], meta


def run_cell(series, cal, coverage, TimeSeries, ConformalNaiveModel,
             LinearRegressionModel):
    hist, test = series[:-1], float(series[-1])
    if hist.size < cal + 6:
        return {"error": "too_short"}

    CAPTURED.clear()
    try:
        base = LinearRegressionModel(lags=1)
        base.fit(TimeSeries.from_values(hist))
        cm = ConformalNaiveModel(model=base, quantiles=quantiles_for(coverage),
                                 cal_length=cal)
        vals = cm.predict(n=1, predict_likelihood_parameters=True,
                          num_samples=1).values().ravel()
    except Exception as exc:
        return {"error": type(exc).__name__}
    if not CAPTURED:
        return {"error": "no_calibration_call"}

    scores = np.asarray(CAPTURED[-1], dtype=float).ravel()
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n < 2:
        return {"error": "too_few_scores"}

    lo_a, centre, hi_a = float(vals[0]), float(vals[1]), float(vals[2])
    half_a = (hi_a - lo_a) / 2.0

    k = required_rank(n, coverage)
    if k is None:
        half_b, feasible = math.inf, False
    else:
        half_b, feasible = float(np.sort(scores)[k - 1]), True

    return {
        "n": n,
        "required_rank": k if k is not None else n + 1,
        "feasible": feasible,
        "a_covered": bool(lo_a <= test <= hi_a),
        "a_width": hi_a - lo_a,
        "a_rank": rank_of(half_a, scores),
        "b_covered": bool(abs(test - centre) <= half_b),
        "b_width": 2 * half_b if math.isfinite(half_b) else math.inf,
    }


def main():
    from darts import TimeSeries
    from darts.models import ConformalNaiveModel, LinearRegressionModel

    if len(sys.argv) < 2:
        print(__doc__)
        print("give the .npz path -- see the two commands above")
        return 2
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"no series cache at {path}\n"
              f"run: python probes/export_series.py m1_monthly_dataset 250 "
              f"{path} 70")
        return 2

    install_spy(ConformalNaiveModel)
    series, meta = load_npz(path)

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say("W3, DARTS ARM -- delivered coverage on real data, paired")
    say("=" * 100)
    say("self_check() passed at import")
    say(f"dataset: {meta[0]}   frequency: {meta[1]}   series: {len(series)}")
    say("arm A: darts ConformalNaiveModel(LinearRegressionModel(lags=1)), its own")
    say("       defaults, via predict(predict_likelihood_parameters=True)")
    say("arm B: the SAME captured calibration scores and the SAME centre,")
    say("       thresholded at rank ceil((n+1)*coverage)")
    say("")
    say("cal_length cells, and the exact deficit at each (k_req - k_np, no simulation):")
    for coverage in LEVELS:
        parts = []
        for cal in CAL_LENGTHS:
            kr = required_rank(cal, coverage)
            kn = numpy_higher_rank(coverage, cal)
            parts.append(f"{cal}:{'inf' if kr is None else kr - kn}")
        say(f"    coverage {coverage:.2f}   " + "   ".join(parts))
    say("    0 = darts lands on the required rank by arithmetic coincidence")
    say("    1 = it lands one rank short;  inf = no valid finite bound exists at that n")
    say("")

    for coverage in LEVELS:
        say("")
        say(f"nominal {coverage:.2f}   quantiles={quantiles_for(coverage)}")
        for cal in CAL_LENGTHS:
            recs = [run_cell(s, cal, coverage, TimeSeries, ConformalNaiveModel,
                             LinearRegressionModel) for s in series]
            s = summarize(recs)
            for ln in format_cell(f"cal_length={cal:<3}", s):
                say(ln)

    say("")
    say("A positive delta means the required rank covers more than the shipped call.")
    say("Raw archive series do not support assigning an absolute miss to the")
    say("convention on its own -- the paired delta is what carries the claim.")
    say("")
    say("Where the deficit above is 0 the two arms are the same interval and the delta")
    say("is exactly 0.0000 by construction, not by measurement. Those rows are the")
    say("control: they show the harness returns zero when there is nothing to find.")

    out = out_path(OUT_TEMPLATE, meta[0])
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
