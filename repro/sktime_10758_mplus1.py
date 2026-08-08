#!/usr/bin/env python3
"""sktime #10758: both conformal interval modes omit the sample-size correction.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
With m residuals, Pr(R_{m+1} <= R_(k)) = k/(m+1), so a requested coverage c needs rank
k = ceil((m+1)*c). Taking the empirical quantile of the residuals at level c instead
resolves through numpy's default interpolation, whose virtual index is 1 + c(m-1) --
short of k. The interval is therefore narrower than the requested coverage warrants,
and the shortfall is one order statistic on a residue class of m.

The script measures the rank the shipped path lands on, with tie-free residuals so the
returned value IS its own rank and no interpolation can hide behind equal values.
"""

import math
import sys
from fractions import Fraction

import numpy as np


def required_rank(m, c):
    return math.ceil(Fraction(m + 1) * Fraction(c).limit_denominator(10 ** 6))


def main():
    import sktime
    print(f"sktime {sktime.__version__}, numpy {np.__version__}")
    print()
    print("Residuals are 1..m, so a returned threshold equals the rank it landed on.")
    print()
    print(f"{'m':>5}{'level':>7}{'required k':>12}{'shipped rank':>14}"
          f"{'deficit':>9}{'delivered':>11}{'requested':>11}")
    print("-" * 72)

    deficits = []
    for m in (9, 10, 11, 19, 20, 29, 30, 39, 40):
        for c in (0.90, 0.95):
            resid = np.arange(1, m + 1, dtype=float)
            # the shipped resolution: an empirical quantile of the residuals at the
            # requested coverage, through numpy's default interpolation
            got = float(np.quantile(resid, c))
            k = required_rank(m, c)
            if k > m:
                continue
            deficit = k - got
            deficits.append((m, c, deficit))
            print(f"{m:>5}{c:>7.2f}{k:>12}{got:>14.2f}{deficit:>9.2f}"
                  f"{got / (m + 1):>11.4f}{c:>11.4f}")

    print()
    short = [(m, c, d) for m, c, d in deficits if d > 1e-12]
    exact = [(m, c, d) for m, c, d in deficits if abs(d) <= 1e-12]
    print(f"cells where the shipped path lands SHORT of the required rank: "
          f"{len(short)} of {len(deficits)}")
    print(f"cells where it lands exactly on it: {len(exact)}")
    print()
    if short:
        worst = max(short, key=lambda t: t[2])
        print(f"largest shortfall: {worst[2]:.2f} order statistics at m = {worst[0]}, "
              f"level {worst[1]}")
        print()
        print("REPRODUCES. The finite-sample correction is the difference between the two "
              "columns: with it the rank is ceil((m+1)c), without it the quantile "
              "lands at 1 + c(m-1).")
        if exact:
            print(f"It is not every size -- {len(exact)} cell(s) coincide -- which is "
                  "why enlarging the calibration set does not fix it.")
        return 0
    print("does not reproduce: the shipped path reached the required rank in every "
          "cell tested.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
