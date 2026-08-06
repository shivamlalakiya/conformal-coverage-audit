#!/usr/bin/env python3
"""What an order-statistic interval can express, and what dividing alpha costs.

Why this probe exists
---------------------
The audit's paired design compares a library's shipped interval against the same
construction at the required order statistic. For a one-rail helper that is a
RANK. For a helper that resolves two levels on signed scores it is a SPAN, and
the arithmetic of spans is not the arithmetic of ranks:

  * the attainable coverages are the same grid, {j/(n+1)}, but a two-sided
    interval reaches j by the DIFFERENCE of two indices rather than by one index;
  * both endpoints are finite only when there are at least two excluded gaps to
    place outside them, which is n >= 2/alpha - 1 and not n >= 1/alpha - 1.

That second fact is the one-rail feasibility floor with alpha halved, and it is
the same statement as the Bonferroni-over-horizon floor with alpha divided by H.
Until this probe existed the two lived in different sections of the manuscript
and the two-rail case lived in a docstring.

An earlier version of the real-data harness built a SYMMETRIC arm B against an
asymmetric arm A, which is what happens when a span requirement is read as a rank
requirement. Everything below is the arithmetic that harness needed.

What it establishes
-------------------
(a) Pr(V_(a) < V_(n+1) <= V_(b)) = (b-a)/(n+1), by exhaustive enumeration over the
    n+1 equally likely ranks -- exact, no sampling.
(b) The minimal span is k = ceil((n+1)(1-alpha)); one gap fewer is insufficient.
(c) Both rails are finite iff n >= 2/alpha - 1, and more generally a construction
    resolving levels 1-alpha_1 .. 1-alpha_L with sum alpha_j = alpha is feasible
    iff n >= 1/min_j alpha_j - 1.
(d) The most nearly equal split of the excluded gaps attains the requested
    coverage exactly, for every admissible n.

Every check is exact rational arithmetic. The Monte Carlo at the end is a control
on the enumeration, not evidence for it.

    python probes/attainable_grid.py
"""

import math
import os
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_real_data import required_rank, required_span  # noqa: E402

OUT = "outputs/probe_output_attainable_grid.txt"

LEVELS = (F(9, 10), F(19, 20), F(2, 3), F(5, 7), F(99, 100))
# L=1 one-sided; L=2 symmetric two-sided; L=12 and L=24 Bonferroni horizons; and
# an ASYMMETRIC split, because min(alpha_j) and not L is what binds.
DIVISIONS = (("one-sided", (F(1),)),
             ("two-sided, equal", (F(1, 2), F(1, 2))),
             ("two-sided, 1:3", (F(1, 4), F(3, 4))),
             ("Bonferroni H=12", tuple([F(1, 12)] * 12)),
             ("Bonferroni H=24", tuple([F(1, 24)] * 24)))


# ---------------------------------------------------------------------------
# (a) the grid, by exhaustive enumeration rather than by sampling
# ---------------------------------------------------------------------------
def grid_coverage(n, a, b):
    """Pr(V_(a) < V_(n+1) <= V_(b)) from the rank of V_(n+1) being uniform.

    Exchangeability and continuity make the rank of the test score among the n+1
    values uniform on {1..n+1}. Rank r means the test score falls in the gap above
    V_(r-1), so it lies in (V_(a), V_(b)] exactly for r in {a+1, .., b}. Counting
    those ranks IS the probability, so this is enumeration and not simulation.
    """
    assert 0 <= a < b <= n + 1
    return F(len([r for r in range(1, n + 2) if a + 1 <= r <= b]), n + 1)


def feasible_division(n, alpha, shares):
    """Is every one of the divided levels expressible by a finite order statistic?

    Each share s_j takes alpha_j = s_j * alpha and needs a rank
    ceil((n+1)(1-alpha_j)) <= n, i.e. (n+1) alpha_j >= 1. The binding constraint
    is the SMALLEST alpha_j, which is why an asymmetric split is worse than an
    equal one at the same L.
    """
    return all(F(n + 1) * (s * alpha) >= 1 for s in shares)


def division_floor(alpha, shares):
    """Smallest n at which every divided level is expressible. Exact."""
    worst = min(shares) * alpha
    # (n+1) * worst >= 1  <=>  n >= 1/worst - 1
    return max(1, math.ceil(F(1, 1) / worst - 1))


def self_check():
    # ---- (a) the grid ----------------------------------------------------
    for n in (2, 3, 7, 20, 47):
        seen = set()
        for a in range(0, n + 1):
            for b in range(a + 1, n + 2):
                c = grid_coverage(n, a, b)
                assert c == F(b - a, n + 1), (n, a, b, c)
                seen.add(c)
        # the attainable set is the one-sided grid, reached by differences
        assert seen == {F(j, n + 1) for j in range(1, n + 2)}, (n, sorted(seen))
        # a one-sided bound is the a = 0 row, so the two agree where they overlap
        for r in range(1, n + 1):
            assert grid_coverage(n, 0, r) == F(r, n + 1)

    # ---- (b) minimality of the span --------------------------------------
    for n in range(2, 300):
        for cov in LEVELS:
            a, b, k = required_span(n, cov)
            assert b - a == k
            assert F(k, n + 1) >= cov, (n, cov, k)
            assert F(k - 1, n + 1) < cov, (n, cov, k)   # one gap fewer fails

    # ---- (c) the floor, and that it is 1/min(alpha_j) - 1 ----------------
    for cov in LEVELS:
        alpha = 1 - cov
        for _, shares in DIVISIONS:
            assert sum(shares) == 1, shares
            fl = division_floor(alpha, shares)
            assert feasible_division(fl, alpha, shares), (cov, shares, fl)
            assert not feasible_division(fl - 1, alpha, shares), (cov, shares, fl)
        # the one-sided floor is the textbook one, and the equal two-sided floor
        # is exactly the one-sided floor at alpha/2
        assert division_floor(alpha, (F(1),)) == math.ceil(F(1) / alpha - 1)
        assert (division_floor(alpha, (F(1, 2), F(1, 2)))
                == math.ceil(F(2) / alpha - 1))
        # an asymmetric split is bound by its SMALLER share, not by L
        assert (division_floor(alpha, (F(1, 4), F(3, 4)))
                == math.ceil(F(4) / alpha - 1))

    # ---- (d) both rails finite exactly at the two-sided floor ------------
    for cov in LEVELS:
        alpha = 1 - cov
        fl = division_floor(alpha, (F(1, 2), F(1, 2)))
        for n in range(2, min(fl + 40, 400)):
            a, b, k = required_span(n, cov)
            both_finite = (a >= 1 and b <= n)
            assert both_finite == (n >= fl), (cov, n, fl, a, b, k)
            if both_finite:
                assert grid_coverage(n, a, b) == F(k, n + 1) >= cov
                # most nearly equal split: the two outside counts differ by <= 1
                assert abs(a - (n + 1 - b)) <= 1, (n, cov, a, b)

    # ---- one-rail sanity: required_rank and required_span agree at L=1 ----
    for n in range(2, 300):
        for cov in LEVELS:
            k1 = required_rank(n, cov)
            _, _, k2 = required_span(n, cov)
            assert (k1 is None) == (k2 > n), (n, cov, k1, k2)
            if k1 is not None:
                assert k1 == k2


self_check()


# ---------------------------------------------------------------------------
# a Monte Carlo control on the enumeration -- not evidence for it
# ---------------------------------------------------------------------------
def monte_carlo(n, a, b, reps=200_000, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((reps, n + 1))
    cal = np.sort(x[:, :n], axis=1)
    test = x[:, n]
    lo = -np.inf if a == 0 else cal[:, a - 1]
    hi = np.inf if b == n + 1 else cal[:, b - 1]
    return float(np.mean((lo < test) & (test <= hi)))


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say("THE ATTAINABLE GRID, AND WHAT DIVIDING ALPHA COSTS")
    say("=" * 100)
    say("self_check() passed at import: the grid by exhaustive enumeration, the")
    say("minimality of the span, the floor for every division of alpha, and the")
    say("equality of required_rank and required_span in the one-rail case -- all in")
    say("exact rational arithmetic over n = 2..299 and five levels.")
    say("")

    say("(a) coverage of [V_(a), V_(b)] is (b-a)/(n+1): enumeration vs Monte Carlo")
    say(f"{'n':>4} {'a':>4} {'b':>4} {'exact':>10} {'monte carlo':>12} {'diff':>9}")
    say("-" * 100)
    worst = 0.0
    for n, a, b in ((20, 1, 19), (20, 0, 19), (47, 3, 45), (10, 2, 9), (10, 0, 10)):
        ex = float(grid_coverage(n, a, b))
        mc = monte_carlo(n, a, b)
        worst = max(worst, abs(ex - mc))
        say(f"{n:>4} {a:>4} {b:>4} {ex:>10.6f} {mc:>12.6f} {abs(ex - mc):>9.6f}")
    say(f"worst enumeration-vs-simulation gap: {worst:.6f}")
    say("")

    say("(b) the minimal span, and (c) the floor for each division of alpha")
    say("A construction resolving 1-alpha_1 .. 1-alpha_L with sum alpha_j = alpha")
    say("needs n >= 1/min_j(alpha_j) - 1. L is not what binds; the smallest share is.")
    say("")
    say(f"{'division':<20} {'nominal':>8} {'floor n':>9} {'vs 1-sided':>11} "
        f"{'min alpha_j':>12}")
    say("-" * 100)
    for cov in (F(9, 10), F(19, 20), F(99, 100)):
        alpha = 1 - cov
        base = division_floor(alpha, (F(1),))
        for label, shares in DIVISIONS:
            fl = division_floor(alpha, shares)
            say(f"{label:<20} {float(cov):>8.2f} {fl:>9} "
                f"{F(fl, base)!s:>11} {float(min(shares) * alpha):>12.6f}")
        say("")

    say("(d) the two-rail span at and around its floor, exact")
    say(f"{'nominal':>8} {'n':>5} {'k*':>5} {'a':>4} {'b':>4} {'both finite':>12} "
        f"{'delivered':>10}")
    say("-" * 100)
    for cov in (F(9, 10), F(19, 20)):
        fl = division_floor(1 - cov, (F(1, 2), F(1, 2)))
        for n in (fl - 2, fl - 1, fl, fl + 1, fl + 10):
            if n < 2:
                continue
            a, b, k = required_span(n, cov)
            bf = "yes" if (a >= 1 and b <= n) else "no"
            say(f"{float(cov):>8.2f} {n:>5} {k:>5} {a:>4} {b:>4} {bf:>12} "
                f"{float(F(k, n + 1)):>10.6f}")
        say("")

    say("The two-sided floor is the one-sided floor at alpha/2, and the Bonferroni")
    say("floor at horizon H is the one-sided floor at alpha/H. They are one")
    say("statement about dividing alpha before resolving it, not three results.")

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        OUT)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
