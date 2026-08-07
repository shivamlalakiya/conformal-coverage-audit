#!/usr/bin/env python3
"""mapie #978: the asymmetric score halves each rail's level, so the calibration-size
guard admits sizes at which the clamp fires.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
`conformity_scores/regression.py` splits the level when the score is asymmetric:

    beta_np  = alpha_np / 2
    alpha_low = alpha_np if self.sym else beta_np
    alpha_up  = 1 - alpha_np if self.sym else 1 - alpha_np + beta_np

So each rail is resolved at alpha/2. A rail at alpha/2 needs n >= 2/alpha - 1, not
n >= 1/alpha - 1 -- an interval attains its coverage through a SPAN of order statistics,
and halving the level per rail doubles the size the construction needs.

The calibration-size guard checks the one-sided floor. Between the two floors it
therefore passes, while the corrected level for the upper rail exceeds 1 and gets clipped
-- which is the same clamp as #974, reached by a path the guard has already approved.

That window is not narrow. At alpha = 1/10 it is every size from 9 to 18 inclusive.
"""

import math
import sys
import warnings
from fractions import Fraction

import numpy as np


def one_sided_floor(alpha):
    return math.ceil(1 / alpha - 1)


def two_rail_floor(alpha):
    return math.ceil(2 / alpha - 1)


def main():
    import mapie
    from mapie.conformity_scores import AbsoluteConformityScore

    print(f"mapie {mapie.__version__}, numpy {np.__version__}")
    print()
    alpha = 0.10
    lo, hi = one_sided_floor(alpha), two_rail_floor(alpha)
    print(f"alpha = {alpha}: one-sided floor n >= {lo}, two-rail floor n >= {hi}")
    print(f"the window where the guard passes and the asymmetric path cannot deliver: "
          f"n = {lo}..{hi - 1}")
    print()
    print("Upper rail level with sym=False is 1 - alpha + alpha/2 = "
          f"{1 - alpha + alpha / 2:.3f}. Scores are 1..n, so the returned threshold IS "
          "the rank.")
    print()
    print(f"{'n':>5}{'guard':>8}{'rail level':>12}{'corrected':>11}{'>1?':>6}"
          f"{'warns':>7}{'returned':>10}{'rail delivers':>14}{'short':>9}")
    print("-" * 84)

    score = AbsoluteConformityScore(sym=False)
    rail = Fraction(1) - Fraction(alpha).limit_denominator(10 ** 6) \
        + Fraction(alpha).limit_denominator(10 ** 6) / 2
    rows = []
    for n in range(lo - 1, hi + 3):
        v = np.arange(1, n + 1, dtype=float)
        corrected = Fraction(math.ceil(Fraction(n + 1) * rail), n)
        guard_passes = n >= lo
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                out = score.get_quantile(v.reshape(-1, 1), np.array([float(rail)]),
                                         axis=0, reversed=False, unbounded=False)
                got = float(np.ravel(out)[0])
                err = ""
            except Exception as exc:
                got, err = float("nan"), type(exc).__name__
            nw = len(caught)
        delivered = got / (n + 1) if math.isfinite(got) else float("nan")
        rows.append({"n": n, "guard": guard_passes, "corr": corrected,
                     "warn": nw, "got": got, "delivered": delivered, "err": err})
        print(f"{n:>5}{('passes' if guard_passes else 'blocks'):>8}"
              f"{float(rail):>12.3f}{float(corrected):>11.4f}"
              f"{str(corrected > 1):>6}{nw:>7}"
              f"{(err if err else f'{got:.1f}'):>10}{delivered:>14.4f}"
              f"{float(rail) - delivered:>9.4f}")

    print()
    window = [r for r in rows if r["guard"] and r["corr"] > 1]
    above = [r for r in rows if r["guard"] and r["corr"] <= 1]
    assert window and above, (
        "the grid must contain sizes the guard passes on both sides of the clamp")
    silent = [r for r in window if r["warn"] == 0 and not r["err"]]
    short = [r for r in window if r["delivered"] < float(rail) - 1e-12]
    print(f"sizes in the window where the guard passes AND the rail clamps: "
          f"{len(window)}  (n = {min(r['n'] for r in window)}"
          f"..{max(r['n'] for r in window)})")
    print(f"  of those, silent:                {len(silent)}")
    print(f"  of those, delivering short:      {len(short)}")
    print(f"sizes the guard passes above the two-rail floor: {len(above)}, "
          f"none of which clamp")
    print()
    if silent and short and len(window) == len(silent):
        print("REPRODUCES. Between the one-sided and two-rail floors the guard approves "
              "the size and the asymmetric rail clamps anyway, silently. The guard is "
              "checking the floor for a rank where the construction needs the floor for "
              "a span.")
        return 0
    print(f"does not reproduce: window {len(window)}, silent {len(silent)}, "
          f"short {len(short)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
