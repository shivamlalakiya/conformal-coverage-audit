#!/usr/bin/env python3
"""mapie #974: the corrected level is clipped to 1.0, silently, where no conformal
quantile exists.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
`BaseConformityScore.get_quantile` applies the finite-sample correction
alpha_cor = (n+1)(1-alpha)/n and then clips it into [0, 1]
(`conformity_scores/interface.py`, the `np.clip(alpha_cor, a_min=0, a_max=1)` line).

Where the correction pushes the level above 1 there is no order statistic that delivers
the requested coverage -- the required rank exceeds n. The clip does not report that. It
returns the level 1.0, so the helper returns max(scores), and the achieved coverage is
n/(n+1), independent of the requested level. No exception, no warning, and a returned number
that looks like any other threshold.

Direction of harm: **anti-conservative**. The interval is narrower than the request.

WHAT WOULD MAKE THIS PRINT "does not reproduce"
-----------------------------------------------
A warning or an exception at the clipped sizes, or a returned value that is not the
sample maximum. The script uses tie-free scores 1..n, so the returned threshold names
the helper's selected rank, and no interpolation can hide behind tied values.
"""

import math
import sys
import warnings
from fractions import Fraction

import numpy as np


def required_rank(n, coverage):
    k = math.ceil(Fraction(n + 1) * Fraction(coverage).limit_denominator(10 ** 6))
    return k if k <= n else None


def main():
    import mapie
    from mapie.conformity_scores import AbsoluteConformityScore

    print(f"mapie {mapie.__version__}, numpy {np.__version__}")
    print()
    print("Scores are 1..n, so a returned threshold equals the rank the helper selected.")
    print()
    print(f"{'n':>5}{'level':>7}{'corrected':>11}{'>1?':>6}{'req rank':>9}"
          f"{'warns':>7}{'returned':>10}{'delivered':>11}{'shortfall':>11}")
    print("-" * 82)

    score = AbsoluteConformityScore()
    rows = []
    for coverage in (0.90, 0.95):
        # get_quantile's alpha_np is the QUANTILE LEVEL, not the miscoverage:
        # alpha_cor = ceil(alpha_ref * (n+1)) / n with reversed=False and
        # alpha_ref = alpha_np. Passing 1 - coverage returns the lower tail and the
        # clamp is never reached -- which is how a first version of this script
        # reported "does not reproduce" against a defect that is there.
        for n in (8, 9, 10, 18, 19, 20):
            v = np.arange(1, n + 1, dtype=float)
            corrected = Fraction(
                math.ceil(Fraction(n + 1) * Fraction(coverage)
                          .limit_denominator(10 ** 6)), n)
            k = required_rank(n, coverage)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    out = score.get_quantile(
                        v.reshape(-1, 1), np.array([coverage]), axis=0,
                        reversed=False, unbounded=False)
                    got = float(np.ravel(out)[0])
                    err = ""
                except Exception as exc:
                    got, err = float("nan"), type(exc).__name__
                nw = len(caught)
            delivered = got / (n + 1) if math.isfinite(got) else float("nan")
            rows.append({"n": n, "cov": coverage, "corr": corrected, "k": k,
                         "warn": nw, "got": got, "delivered": delivered, "err": err})
            print(f"{n:>5}{coverage:>7.2f}{float(corrected):>11.4f}"
                  f"{str(corrected > 1):>6}{str(k):>9}{nw:>7}"
                  f"{(err if err else f'{got:.1f}'):>10}"
                  f"{delivered:>11.4f}{coverage - delivered:>11.4f}")

    print()
    clipped = [r for r in rows if r["corr"] > 1]
    fine = [r for r in rows if r["corr"] <= 1]
    assert clipped and fine, (
        "the grid must straddle the size where the correction passes 1, or there is "
        "nothing to contrast")
    silent = [r for r in clipped if r["warn"] == 0 and not r["err"]]
    at_max = [r for r in clipped if math.isfinite(r["got"]) and r["got"] == r["n"]]
    short = [r for r in clipped if r["delivered"] < r["cov"] - 1e-12]
    print(f"sizes where the corrected level exceeds 1: {len(clipped)}")
    print(f"  returning with no warning and no exception:   {len(silent)}")
    print(f"  returning the sample maximum:                 {len(at_max)}")
    print(f"  delivering LESS than the requested coverage:  {len(short)}")
    print(f"sizes where the correction stays in range: {len(fine)}, all of which also "
          f"return finitely")
    print()
    if silent and at_max and short:
        worst = max(short, key=lambda r: r["cov"] - r["delivered"])
        print(f"worst shortfall: {worst['cov'] - worst['delivered']:.4f} at n = "
              f"{worst['n']}, level {worst['cov']} -- delivered "
              f"{worst['delivered']:.4f}")
        print()
        print("REPRODUCES. Where the correction leaves the unit interval the clip "
              "returns level 1.0, the helper returns max(scores), and the delivered "
              "coverage is n/(n+1), independent of the input level. Nothing in the return "
              "value or the warning stream distinguishes it from a normal call, and the "
              "error is anti-conservative.")
        return 0
    print(f"does not reproduce: silent {len(silent)}, at max {len(at_max)}, "
          f"short {len(short)} of {len(clipped)} clipped sizes")
    return 1


if __name__ == "__main__":
    sys.exit(main())
