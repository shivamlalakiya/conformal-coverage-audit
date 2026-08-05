#!/usr/bin/env python3
"""The rank map: which order statistic does a quantile definition deliver?

A finite-sample distribution-free bound is a statement about a RANK. For n
exchangeable scores, the interval built on rank r covers a fresh observation
with probability exactly r/(n+1). Quantile APIs do not accept a rank; they
accept a LEVEL, and each of the classical sample-quantile definitions maps that
level to a position differently.

Hyndman & Fan (The American Statistician, 1996) catalogued nine definitions --
three based on rounding, six on linear interpolation. NumPy implements all nine
by name and cites them in its own `quantile` docstring, plus four discontinuous
aliases. This probe asks, for each of them:

  1. At the UNCORRECTED level (1-alpha), which rank is delivered, and how far
     short of the required k = ceil((n+1)(1-alpha)) is it?
  2. Which levels recover rank k exactly, and for which definitions is that
     impossible at any level?
  3. What is the smallest n at which each definition delivers a valid,
     non-degenerate bound -- the DELIVERED n_min, next to the theoretical one?

The measurement trick: run the quantile on the scores [1, 2, ..., n]. The value
returned IS the rank it landed on. An integral result means the definition
selected an order statistic; a fractional result means it interpolated between
two, and then the only distribution-free guarantee it carries is the one from
the FLOOR -- the lower of the two order statistics it sits between.
"""

import math
from fractions import Fraction

import numpy as np

# the nine Hyndman & Fan definitions, in NumPy's naming, plus the four aliases
HF_NINE = [
    "inverted_cdf",              # H&F type 1
    "averaged_inverted_cdf",     # type 2
    "closest_observation",       # type 3
    "interpolated_inverted_cdf",  # type 4
    "hazen",                     # type 5
    "weibull",                   # type 6
    "linear",                    # type 7  <-- NumPy's default
    "median_unbiased",           # type 8
    "normal_unbiased",           # type 9
]
ALIASES = ["lower", "higher", "midpoint", "nearest"]
ALL_METHODS = HF_NINE + ALIASES

LEVELS = [Fraction(9, 10), Fraction(19, 20), Fraction(99, 100), Fraction(2, 3)]
TOL = 1e-9
OUT = "outputs/probe_output_rank_map.txt"


# --------------------------------------------------------------------------
# exact arithmetic
# --------------------------------------------------------------------------
def required_rank(n, coverage):
    """k = ceil((n+1) * coverage), exact. None when k > n (no rank suffices)."""
    k = math.ceil(Fraction(n + 1) * Fraction(coverage))
    return k if k <= n else None


def theoretical_n_min(coverage):
    """Smallest n for which some rank attains `coverage`: n >= 1/alpha - 1."""
    alpha = 1 - Fraction(coverage)
    n = math.ceil(1 / alpha) - 1
    assert required_rank(n, coverage) is not None
    assert required_rank(n - 1, coverage) is None
    return n


def landed(n, level, method):
    """Rank the definition lands on, run against the scores [1..n].

    Returns (position, is_order_statistic). `position` is the float the
    definition returns; when it is integral the definition selected that order
    statistic exactly.
    """
    scores = np.arange(1, n + 1, dtype=float)
    v = float(np.quantile(scores, float(level), method=method))
    return v, abs(v - round(v)) < TOL


def guaranteed_rank(n, level, method):
    """The rank whose guarantee the returned bound actually carries.

    An interpolated bound sits strictly between X_(j) and X_(j+1). It is >= X_(j),
    so it inherits j/(n+1) -- and nothing more, distribution-free. Hence floor.
    """
    v, exact = landed(n, level, method)
    return int(round(v)) if exact else int(math.floor(v + TOL))


def delivered_coverage(r, n):
    return Fraction(r, n + 1)


# --------------------------------------------------------------------------
# self-check, on a grid the bug chooses
# --------------------------------------------------------------------------
def self_check():
    # required_rank, hand-verified cells
    assert required_rank(9, Fraction(9, 10)) == 9
    assert required_rank(19, Fraction(19, 20)) == 19
    assert required_rank(39, Fraction(19, 20)) == 38
    assert required_rank(99, Fraction(99, 100)) == 99
    assert required_rank(6, Fraction(2, 3)) == 5
    # feasibility boundary
    for cov, first in ((Fraction(9, 10), 9), (Fraction(19, 20), 19), (Fraction(99, 100), 99)):
        assert theoretical_n_min(cov) == first, (cov, theoretical_n_min(cov))
    # the required rank always clears the level, exactly
    for n in range(2, 400):
        for cov in LEVELS:
            k = required_rank(n, cov)
            if k is not None:
                assert delivered_coverage(k, n) >= cov, (n, cov, k)
    # the measurement trick returns ranks: 1..n at level 0 and 1 are rank 1 and n
    for m in ALL_METHODS:
        assert guaranteed_rank(10, Fraction(0), m) == 1, m
        assert guaranteed_rank(10, Fraction(1), m) == 10, m
    # every method is monotone in the level
    for m in ALL_METHODS:
        seq = [guaranteed_rank(50, Fraction(i, 100), m) for i in range(101)]
        assert all(b >= a for a, b in zip(seq, seq[1:])), m
    # an interpolating method genuinely fails to land on a rank somewhere
    assert any(
        not landed(n, Fraction(19, 20), "linear")[1] for n in range(5, 60)
    ), "linear should interpolate off-rank somewhere"
    # a rounding method lands on a rank everywhere
    for n in range(2, 120):
        assert landed(n, Fraction(19, 20), "higher")[1], n
        assert landed(n, Fraction(19, 20), "inverted_cdf")[1], n


self_check()


# --------------------------------------------------------------------------
# the three tables
# --------------------------------------------------------------------------
def table_a(say, n_max=1000):
    """Uncorrected level: deficit distribution per method."""
    say("=" * 92)
    say("TABLE A -- delivered rank at the UNCORRECTED level, deficit vs required k")
    say("=" * 92)
    say("  deficit = k_required - rank_whose_guarantee_the_bound_carries")
    say("  cells counted over n = 2..%d where the level is feasible at all" % n_max)
    say("")
    say("  READ THIS BEFORE QUOTING A DEFICIT. Deficit > 0 does NOT mean invalid.")
    say("  k = ceil((n+1)(1-alpha)) is sometimes strictly larger than the level needs,")
    say("  so a bound one rank below k can still clear the level. Table A measures")
    say("  IDENTITY -- does the call reproduce the canonical index. TABLE C measures")
    say("  VALIDITY -- does the delivered rank r satisfy r/(n+1) >= 1-alpha. They are")
    say("  different questions and the answers differ; do not quote one as the other.")
    say("")
    for cov in LEVELS:
        say(f"  level {float(cov):.4f}  (alpha = {float(1 - cov):.4f})")
        say(f"    {'method':<26} {'lands on a rank?':<18} {'deficit distribution':<34} mean")
        for m in ALL_METHODS:
            defs, off_rank = [], 0
            for n in range(2, n_max + 1):
                k = required_rank(n, cov)
                if k is None:
                    continue
                v, exact = landed(n, cov, m)
                if not exact:
                    off_rank += 1
                defs.append(k - guaranteed_rank(n, cov, m))
            hist = {}
            for d in defs:
                hist[d] = hist.get(d, 0) + 1
            shape = "always" if off_rank == 0 else f"never ({off_rank}/{len(defs)} off)"
            dist = "  ".join(f"{d}:{c}" for d, c in sorted(hist.items()))
            say(f"    {m:<26} {shape:<18} {dist:<34} {np.mean(defs):+.3f}")
        say("")


def table_b(say, cells=((30, None), (50, None), (100, None), (200, None))):
    """Safe recipe: which level recovers rank k exactly, per method."""
    say("=" * 92)
    say("TABLE B -- the safe recipe: does the level recover the required rank k exactly?")
    say("=" * 92)
    say("  two candidate levels, both in common use:")
    say("      k/n        the 'finite-sample corrected' level")
    say("      (k-1)/(n-1)  the level whose virtual index is the 0-based rank")
    say("")
    cov = Fraction(19, 20)
    say(f"  level 1-alpha = {float(cov):.2f}")
    say(f"    {'method':<26} " + "  ".join(f"n={n}" for n, _ in cells))
    for m in ALL_METHODS:
        row_kn, row_km1 = [], []
        for n, _ in cells:
            k = required_rank(n, cov)
            if k is None:
                row_kn.append("  -  ")
                row_km1.append("  -  ")
                continue
            g1 = guaranteed_rank(n, Fraction(k, n), m)
            g2 = guaranteed_rank(n, Fraction(k - 1, n - 1), m)
            row_kn.append(" ok  " if g1 == k else f"{g1 - k:+d}   ")
            row_km1.append(" ok  " if g2 == k else f"{g2 - k:+d}   ")
        say(f"    {m:<26} " + "  ".join(row_kn) + "   <- level k/n")
        say(f"    {'':<26} " + "  ".join(row_km1) + "   <- level (k-1)/(n-1)")
    say("")


def table_c(say, n_max=2000):
    """Delivered n_min per method, next to the theoretical one."""
    say("=" * 92)
    say("TABLE C -- DELIVERED n_min: smallest n at which the shipped call is valid")
    say("=" * 92)
    say("  theoretical n_min: smallest n for which any rank attains the level")
    say("  delivered n_min:   smallest n at which this method, at the uncorrected")
    say("                     level, carries a guarantee >= the requested level")
    say("")
    say(f"    {'method':<26} " + "  ".join(f"{float(c):.2f}" for c in LEVELS))
    say(f"    {'theoretical n_min':<26} "
        + "  ".join(f"{theoretical_n_min(c):>4}" for c in LEVELS))
    for m in ALL_METHODS:
        cells = []
        for cov in LEVELS:
            hit = None
            for n in range(2, n_max + 1):
                k = required_rank(n, cov)
                if k is None:
                    continue
                if delivered_coverage(guaranteed_rank(n, cov, m), n) >= cov:
                    hit = n
                    break
            cells.append(f"{hit:>4}" if hit else "none")
        say(f"    {m:<26} " + "  ".join(cells))
    say("")
    say("  'none' means the uncorrected level never delivers the requested")
    say("  guarantee at any n <= %d -- the deficit does not close, it only shrinks." % n_max)


def main():
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("self_check() passed at import")
    say(f"numpy {np.__version__}; Hyndman & Fan cited in np.quantile's own docstring")
    say("")
    table_a(say)
    table_b(say)
    table_c(say)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
