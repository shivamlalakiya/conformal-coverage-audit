#!/usr/bin/env python3
"""crepes #47: the classifier is silent below the floor where the regressor warns.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
A one-sided distribution-free bound at confidence c needs order statistic
ceil((n+1)*c) among n calibration scores; when that exceeds n the confidence is not
reachable. `ConformalRegressor.predict_int` says so -- it emits a warning and returns
an infinite endpoint. `ConformalClassifier.predict_set` does neither. It returns
prediction sets with no warning and no exception, and at those sizes every set is the
full label set, which is valid and carries no information.

So the finding is not that the classifier is unsound. It is that two paths in the same
library disagree about whether an unreachable confidence is worth mentioning, and the
one that stays quiet is the one whose output looks ordinary.

WHAT WOULD MAKE THIS PRINT "does not reproduce"
-----------------------------------------------
The classifier warning or raising below the floor, or the regressor not warning. The
script asserts the grid straddles the floor, so a grid that could not show the
contrast fails rather than passing quietly.
"""

import math
import sys
import warnings
from fractions import Fraction

import numpy as np


def required_rank(n, c):
    return math.ceil(Fraction(n + 1) * Fraction(c).limit_denominator(10 ** 6))


def probe(fn):
    """(n_warnings, exception name or '', result) for one call."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # fn() FIRST, then count. `return len(caught), "", fn()` evaluates the count
        # before the call and reported zero warnings for a path that warns every time.
        try:
            out = fn()
        except Exception as exc:
            return len(caught), type(exc).__name__, None
        return len(caught), "", out


def main():
    import crepes
    from crepes import ConformalClassifier, ConformalRegressor
    print(f"crepes {crepes.__version__}, numpy {np.__version__}")

    c = 0.95
    sizes = (3, 5, 10, 18, 19, 20, 40)
    infeasible = [n for n in sizes if required_rank(n, c) > n]
    feasible = [n for n in sizes if required_rank(n, c) <= n]
    assert infeasible and feasible, (
        "the grid does not straddle the floor, so it cannot show the contrast")
    print(f"confidence {c}: sizes below the floor {infeasible}, above it {feasible}")
    print()
    print(f"{'n':>5}{'req k':>7}{'feasible':>10} | "
          f"{'regressor warns':>16}{'endpoint':>10} | "
          f"{'classifier warns':>17}{'set size':>10}")
    print("-" * 82)

    rows = []
    rng = np.random.default_rng(0)
    for n in sizes:
        k = required_rank(n, c)
        resid = np.abs(rng.normal(size=n)) + 1.0
        alphas_cal = rng.random(n)
        alphas_te = rng.random((4, 3))

        def reg():
            cr = ConformalRegressor()
            cr.fit(residuals=resid)
            return np.ravel(cr.predict_int(y_hat=np.array([0.0]), confidence=c))

        def clf():
            cc = ConformalClassifier()
            cc.fit(alphas_cal)
            return cc.predict_set(alphas_te, confidence=c, smoothing=False)

        wr, er, ir = probe(reg)
        wc, ec, sc = probe(clf)
        endpoint = ("raised" if er else
                    ("+inf" if not np.isfinite(ir[1]) else f"{ir[1]:.3f}"))
        setsize = ("raised" if ec else f"{float(np.mean(sc.sum(axis=1))):.2f}")
        rows.append({"n": n, "k": k, "feasible": k <= n,
                     "wr": wr, "er": er, "wc": wc, "ec": ec})
        print(f"{n:>5}{k:>7}{str(k <= n):>10} | {wr:>16}{endpoint:>10} | "
              f"{wc:>17}{setsize:>10}")

    print()
    below = [r for r in rows if not r["feasible"]]
    reg_signals = [r for r in below if r["wr"] > 0 or r["er"]]
    clf_silent = [r for r in below if r["wc"] == 0 and not r["ec"]]
    print(f"below the floor: {len(below)} size(s)")
    print(f"  regressor warned or raised at:      {len(reg_signals)}")
    print(f"  classifier warned or raised at:     {len(below) - len(clf_silent)}")
    print()
    if reg_signals and len(clf_silent) == len(below):
        print("REPRODUCES. Every size below the floor draws a signal from the "
              "regressor and none from the classifier. A caller of the classifier "
              "gets sets that look like any other sets.")
        return 0
    print("does not reproduce: "
          f"regressor signalled at {len(reg_signals)} of {len(below)}, "
          f"classifier silent at {len(clf_silent)} of {len(below)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
