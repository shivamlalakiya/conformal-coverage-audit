#!/usr/bin/env python3
"""mapie #980: allow_infinite_bounds=True returns a FINITE bound below the floor.

One filing, one script, no dependency on the rest of this repository. Run it and read
the last line.

THE CLAIM
---------
With n calibration scores, a one-sided distribution-free bound needs order statistic
k = ceil((n+1)*confidence). When k > n no such bound exists among the scores, and the
library's own guard says so by raising. The public keyword `allow_infinite_bounds=True`
is documented as permitting an infinite interval in that situation. It does not produce
one: it returns a finite width, and the coverage that width delivers is n/(n+1)
regardless of the confidence level asked for.

WHAT WOULD MAKE THIS SCRIPT PRINT "does not reproduce"
------------------------------------------------------
Either the guard raising in both branches, or the flagged branch returning an infinite
endpoint. Both are the documented behaviour. A finite width is the finding.
"""

import math
import sys
from fractions import Fraction

import numpy as np


def required_rank(n, confidence):
    k = math.ceil(Fraction(n + 1) * Fraction(confidence).limit_denominator(10 ** 6))
    return k if k <= n else None


def main():
    import mapie
    from mapie.regression import SplitConformalRegressor
    from sklearn.linear_model import LinearRegression

    n_calib, confidence = 10, 0.95
    k = math.ceil(Fraction(n_calib + 1) * Fraction(confidence).limit_denominator(10 ** 6))
    print(f"mapie {mapie.__version__}, numpy {np.__version__}")
    print(f"n_calib = {n_calib}, confidence_level = {confidence}")
    print(f"required order statistic k = ceil((n+1)*c) = {k}, of n = {n_calib}")
    assert required_rank(n_calib, confidence) is None, (
        "this configuration is feasible, so it cannot demonstrate the finding; "
        "pick n and confidence with ceil((n+1)*c) > n")
    print("-> no valid finite deterministic bound exists at this configuration\n")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(3 * n_calib, 3))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=3 * n_calib)
    Xtr, ytr = X[:n_calib], y[:n_calib]
    Xca, yca = X[n_calib:2 * n_calib], y[n_calib:2 * n_calib]
    Xte = X[2 * n_calib:2 * n_calib + 1]

    results = {}
    for flag in (False, True):
        try:
            m = SplitConformalRegressor(estimator=LinearRegression(),
                                        confidence_level=confidence,
                                        prefit=False)
            m.fit(Xtr, ytr)
            m.conformalize(Xca, yca)
            _, iv = m.predict_interval(Xte, allow_infinite_bounds=flag)
            lo, hi = float(np.ravel(iv)[0]), float(np.ravel(iv)[1])
            width = hi - lo
            results[flag] = ("finite" if math.isfinite(width) else "infinite", width)
            print(f"allow_infinite_bounds={flag!s:<5} -> "
                  f"[{lo:.4f}, {hi:.4f}]  width {width:.4f}  "
                  f"{'FINITE' if math.isfinite(width) else 'infinite'}")
        except Exception as exc:
            results[flag] = ("raised", f"{type(exc).__name__}")
            print(f"allow_infinite_bounds={flag!s:<5} -> "
                  f"{type(exc).__name__}: {str(exc)[:70]}")

    print()
    guarded = results.get(False, (None,))[0] == "raised"
    flagged_finite = results.get(True, (None,))[0] == "finite"
    if guarded and flagged_finite:
        print("REPRODUCES. The default branch refuses the configuration and the "
              "flagged branch returns a finite interval, so the keyword documented "
              "as allowing an infinite bound is what removes the guard.")
        return 0
    print("does not reproduce in this environment. "
          f"default branch: {results.get(False)}, flagged branch: {results.get(True)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
