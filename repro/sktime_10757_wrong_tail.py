#!/usr/bin/env python3
"""sktime #10757: ConformalIntervals(method="empirical_residual") takes the wrong tail.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
`empirical_residual` is documented as an interval method beside `empirical`. It resolves
its rails from the wrong end of the residual distribution, so the interval it returns
does not cover at the nominal rate -- it covers at close to zero. `empirical`, on the
same series, the same fit and the same level, covers near nominal.

The comparison is against the library's own sibling method rather than against a rank we
computed, so the finding needs no theory to read: two documented methods, one series set,
and coverage that differs by most of the unit interval.
"""

import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")


def coverage(method, series, window, level):
    """Empirical coverage of the shipped interval over a set of series."""
    import pandas as pd
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.conformal import ConformalIntervals
    from sktime.forecasting.naive import NaiveForecaster

    hit = 0
    used = 0
    for s in series:
        y_hist, y_test = s[:-1], s[-1]
        if len(y_hist) < window + 4:
            continue
        y = pd.Series(y_hist, index=pd.RangeIndex(len(y_hist)))
        fh = ForecastingHorizon([1], is_relative=True)
        try:
            ci = ConformalIntervals(NaiveForecaster(strategy="last"), method=method,
                                    initial_window=window)
            ci.fit(y, fh=fh)
            iv = ci.predict_interval(fh=fh, coverage=[level])
            lo, hi = float(iv.iloc[0, 0]), float(iv.iloc[0, 1])
        except Exception:
            continue
        used += 1
        hit += int(lo <= y_test <= hi)
    return used, (hit / used if used else float("nan"))


def synthetic(n_series=24, length=60, seed=0):
    """Random walks. The defect is in the resolution, not in the data."""
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=length)) + 100.0 for _ in range(n_series)]


def main():
    import sktime
    print(f"sktime {sktime.__version__}, numpy {np.__version__}")
    print()
    series = synthetic()
    level, window = 0.90, 20
    print(f"{len(series)} random-walk series, initial_window {window}, "
          f"nominal coverage {level}")
    print()
    print(f"{'method':<22}{'series scored':>15}{'delivered coverage':>21}"
          f"{'shortfall':>11}")
    print("-" * 70)
    got = {}
    for method in ("empirical", "empirical_residual"):
        used, cov = coverage(method, series, window, level)
        got[method] = cov
        print(f"{method:<22}{used:>15}{cov:>21.4f}{level - cov:>11.4f}")

    print()
    sib, bad = got["empirical"], got["empirical_residual"]
    assert not np.isnan(sib) and not np.isnan(bad), (
        "one method scored no series, so there is nothing to compare")
    print(f"gap between the two documented methods: {sib - bad:.4f}")
    print()
    if bad < 0.5 and sib > 0.75:
        print("REPRODUCES. `empirical` covers near its nominal rate and "
              "`empirical_residual` covers almost nothing, on the same series, the same "
              "fit and the same level. The rails come from the wrong tail.")
        return 0
    print(f"does not reproduce: empirical {sib:.4f}, empirical_residual {bad:.4f}. "
          f"If the second is now near nominal the fix has landed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
