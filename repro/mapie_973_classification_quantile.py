#!/usr/bin/env python3
"""mapie #973: the classification quantile lands one order statistic above the required
rank on a residue class of calibration sizes.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
`mapie.utils._compute_quantiles` resolves the classification threshold as

    np.quantile(vector, ((n + 1) * (1 - alpha)) / n, method="higher")

The level handed to numpy already carries the finite-sample correction, and
`method="higher"` then resolves it as ceil(q(n-1)) + 1 -- a second rounding-up on top of
the first. At some sizes the two agree; at others the result is one order statistic above
k* = ceil((n+1)(1-alpha)), which is the smallest rank that delivers the request.

Direction of harm: **conservative**. The set is larger than the level asks for, so this
costs width and not validity. It is a defect because the width is not what was requested
and because which sizes are affected is a congruence in n rather than a constant, so it
does not shrink with data in the way a reader would assume.

The script computes both ranks in exact arithmetic and, separately, calls numpy on
tie-free scores so the returned value IS its own rank. Two routes, and they must agree.
"""

import math
import sys
from fractions import Fraction

import numpy as np


def required_rank(n, coverage):
    return math.ceil(Fraction(n + 1) * Fraction(coverage).limit_denominator(10 ** 6))


def landed_rank_exact(n, coverage):
    """ceil(q(n-1)) + 1 for q = (n+1)*coverage/n, in exact arithmetic."""
    q = Fraction(n + 1) * Fraction(coverage).limit_denominator(10 ** 6) / n
    return math.ceil(q * (n - 1)) + 1


def main():
    import mapie
    from mapie.utils import _compute_quantiles

    print(f"mapie {mapie.__version__}, numpy {np.__version__}")
    print()
    print("Two independent routes to the same rank: exact rational arithmetic, and")
    print("_compute_quantiles called on scores 1..n so its return value IS the rank.")
    print()
    print(f"{'level':>7}{'n':>5}{'required k*':>13}{'landed (exact)':>16}"
          f"{'landed (called)':>17}{'excess':>8}")
    print("-" * 70)

    rows = []
    for coverage in (0.90, 0.95):
        alpha = 1 - coverage
        for n in range(9, 41):
            k = required_rank(n, coverage)
            if k > n:
                continue
            exact = landed_rank_exact(n, coverage)
            v = np.arange(1, n + 1, dtype=float)
            called = float(np.ravel(_compute_quantiles(v, np.array([alpha])))[0])
            rows.append({"cov": coverage, "n": n, "k": k, "exact": exact,
                         "called": called, "excess": exact - k})
            if n <= 22:
                print(f"{coverage:>7.2f}{n:>5}{k:>13}{exact:>16}{called:>17.0f}"
                      f"{exact - k:>8}")
    print(f"  ... sizes to 40 computed, {len(rows)} feasible cells in total")
    print()

    # the two routes must agree, or one of them is wrong and neither is evidence
    disagree = [r for r in rows if abs(r["called"] - r["exact"]) > 1e-9]
    assert not disagree, (
        f"the exact arithmetic and the library call disagree in {len(disagree)} cells, "
        f"first {disagree[0]}. Fix the reproducer before reading anything into it.")
    print(f"the two routes agree in all {len(rows)} cells, so the rank below is the "
          f"library's own")
    print()

    high = [r for r in rows if r["excess"] > 0]
    exactly = [r for r in rows if r["excess"] == 0]
    low = [r for r in rows if r["excess"] < 0]
    print(f"{'outcome':<34}{'cells':>7}{'share':>9}")
    print("-" * 52)
    for label, g in (("lands one or more ranks HIGH", high),
                     ("lands exactly on k*", exactly),
                     ("lands BELOW k* (would be invalid)", low)):
        print(f"{label:<34}{len(g):>7}{len(g) / len(rows):>9.2%}")
    print()
    assert not low, (
        f"{len(low)} cells land below the required rank, which would be a validity "
        f"failure rather than the width defect this filing reports -- investigate "
        f"before publishing either reading")
    if high and exactly:
        ns = sorted({r["n"] for r in high})
        print(f"affected sizes: {ns[:12]}{' ...' if len(ns) > 12 else ''}")
        print(f"largest excess: {max(r['excess'] for r in high)} rank(s)")
        print()
        print("REPRODUCES. The level is corrected once and then rounded up again, so on "
              "a congruence class of sizes the threshold sits a rank above the smallest "
              "one that delivers. Conservative, so it costs width; periodic in n, so it "
              "does not wash out with more calibration data.")
        return 0
    print(f"does not reproduce: {len(high)} high, {len(exactly)} exact, {len(low)} low")
    return 1


if __name__ == "__main__":
    sys.exit(main())
