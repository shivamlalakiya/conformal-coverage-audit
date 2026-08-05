"""statsforecast 2.1.1, RUN: the clamp in ConformalSeasonalPool._oriented_index.

This is the second instance of the clamp idiom (MAPIE's np.clip is the first), and it
is the evidence behind the "clamp is a recurring idiom, not a MAPIE bug" claim, so it
gets run rather than read. _oriented_index is a @staticmethod, so its own code is
called directly with no fitting or data plumbing in the way.

Claim under test: when ceil((n+1)q) > n the returned level is clamped to exactly 1.0,
so np.quantile(R, level) becomes max(R) -- a finite bound where the conformal quantile
does not exist, with no warning.
"""

from math import ceil, floor

import numpy as np
from statsforecast.models import ConformalSeasonalPool

oi = ConformalSeasonalPool._oriented_index


def uncorrected_would_be(q, n):
    """What the level would be with no clamp at all."""
    return (ceil((n + 1.0) * q) / n) if q >= 0.5 else (floor((n + 1.0) * q) / n)


def _self_check():
    # Their own docstring-free arithmetic, restated: upper tail uses ceil, lower floor.
    assert oi(0.95, 100) == ceil(101 * 0.95) / 100 == 0.96
    assert oi(0.05, 100) == floor(101 * 0.05) / 100 == 0.05
    # n <= 0 short-circuits to the raw level.
    assert oi(0.95, 0) == 0.95
    print("self-check passed: _oriented_index matches ceil/floor((n+1)q)/n off the clamp")


def sweep(q, n_max=40):
    rows = []
    for n in range(1, n_max + 1):
        got = oi(q, n)
        raw = uncorrected_would_be(q, n)
        clamped = got != raw
        k = ceil((n + 1.0) * q)          # order statistic validity requires
        rows.append((n, k, raw, got, clamped, k > n))
    return rows


def coverage_when_clamped(n, q, draws, rng):
    """Empirical coverage of statsforecast's level vs the honest answer (+inf)."""
    s = rng.standard_normal((draws, n + 1))
    cal, test = s[:, :n], s[:, n]
    lvl = oi(q, n)
    thr = np.quantile(cal, lvl, axis=1)
    return (test <= thr).mean()


if __name__ == "__main__":
    _self_check()
    rng = np.random.default_rng(4082026)

    for q in (0.95, 0.975):
        print(f"\n{'=' * 74}\nupper-tail level q = {q}   "
              f"(statsforecast ConformalSeasonalPool)\n{'=' * 74}")
        print(f"{'n':>4} {'needs k':>8} {'unclamped lvl':>14} {'returned lvl':>13} "
              f"{'CLAMPED':>8} {'k>n':>5}")
        fired = []
        for n, k, raw, got, clamped, vac in sweep(q):
            if n <= 24 or clamped:
                flag = "yes" if clamped else ""
                print(f"{n:>4} {k:>8} {raw:>14.4f} {got:>13.4f} {flag:>8} "
                      f"{str(vac):>5}")
            if clamped:
                fired.append(n)
        print(f"\nclamp fires for n in {min(fired)}..{max(fired)}  "
              f"({len(fired)} sizes); k > n over the same range: "
              f"{all(ceil((n + 1.0) * q) > n for n in fired)}")

        print(f"\nmeasured one-sided coverage where the clamp fires "
              f"(requested {q}, 200k draws):")
        for n in fired[:6]:
            cov = coverage_when_clamped(n, q, 200_000, rng)
            print(f"  n={n:<3} level returned 1.0 -> threshold is max(cal)   "
                  f"coverage {cov:.4f}   exact n/(n+1) = {n / (n + 1):.4f}")
