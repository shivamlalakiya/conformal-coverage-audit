#!/usr/bin/env python3
"""The one-rank gap against the spread of one calibration draw's coverage.

A threshold at rank k of n exchangeable continuous scores covers a fresh score with
probability k/(n+1) on average over calibration sets. One calibration set's own
coverage, conditional on that set, is Beta(k, n+1-k). So the gap this audit prices,
one rank or 1/(n+1), has to be read beside the standard deviation of that Beta law
and beside the chance that a caller's own calibration set leaves realised coverage
below the requested level L. This probe prints all three, at the required rank
k* = ceil((n+1)L) and one rank short of it, over a grid of n and L.

It also prints the tolerance-limit rank, the smallest k at which realised coverage
reaches L with probability at least 1 - delta: the (epsilon, delta) standard of
Wilks (1941), against which the marginal rank k* is the average-case answer.

Everything here is exact arithmetic through scipy's Beta and binomial laws; no
draw is simulated.

    python probes/beta_spread.py
"""

import math
import os
from fractions import Fraction

from scipy.stats import beta, binom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "probe_output_beta_spread.txt")

NS = (10, 20, 50, 100, 250, 1000, 5000)
LEVELS = (Fraction(4, 5), Fraction(9, 10), Fraction(19, 20), Fraction(99, 100))
DELTA = 0.10


def required_rank(n, L):
    k = math.ceil((n + 1) * L)          # exact: L is a Fraction
    return k if k <= n else None


def p_below(k, n, L):
    """P(realised coverage < L) for the rank-k threshold of n scores."""
    return float(beta.cdf(float(L), k, n + 1 - k))


def tolerance_rank(n, L, delta):
    for k in range(1, n + 1):
        if 1 - p_below(k, n, L) >= 1 - delta:
            return k
    return None


def self_check():
    # The Beta tail and the binomial tail are one quantity written two ways:
    # P(Beta(k, n+1-k) < L) = P(Binomial(n, L) >= k). The common slip, n-k for
    # n+1-k as the second shape, breaks the identity; the second assertion shows
    # the check can fail by running it on that slip.
    for n in (5, 50, 500):
        for L in LEVELS:
            slipped = 0
            for k in range(1, n):
                a = beta.cdf(float(L), k, n + 1 - k)
                b = binom.sf(k - 1, n, float(L))
                assert abs(a - b) < 1e-9, (n, L, k, a, b)
                if abs(beta.cdf(float(L), k, n - k) - b) > 1e-6:
                    slipped += 1
            assert slipped > 0, (n, L)
    # The mean of the rank-k coverage is k/(n+1).
    for n in (9, 99):
        for k in (1, n // 2, n):
            assert abs(beta.mean(k, n + 1 - k) - k / (n + 1)) < 1e-12
    # k* is the smallest k with k/(n+1) >= L.
    for n in range(1, 400):
        for L in LEVELS:
            k = math.ceil((n + 1) * L)
            assert Fraction(k, n + 1) >= L > Fraction(k - 1, n + 1)


self_check()


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say("THE ONE-RANK GAP AGAINST THE BETA SPREAD OF ONE CALIBRATION DRAW'S COVERAGE")
    say("=" * 100)
    say("self_check() passed at import: the Beta tail equals the binomial tail at every")
    say("k < n for n in {5, 50, 500}, the Beta mean is k/(n+1), and k* is the smallest")
    say("rank with k/(n+1) >= L, in exact rationals.")
    say("")
    say("gap          1/(n+1), one rank of predicted coverage")
    say("sd           sd of Beta(k*, n+1-k*), the spread of one calibration set's coverage")
    say("ratio        gap / sd")
    say("p_below_k    P(realised coverage < L) at the required rank k*")
    say("p_below_km1  the same one rank short, the deficit this audit measures")
    say(f"k_tol        smallest k with P(realised coverage >= L) >= {1 - DELTA:.2f} (Wilks)")
    say("A cell with k* > n is infeasible and printed as such.")
    say("")
    for L in LEVELS:
        for n in NS:
            k = required_rank(n, L)
            if k is None:
                say(f"CELL L={float(L):.2f} n={n} infeasible")
                continue
            gap = 1 / (n + 1)
            sd = float(beta.std(k, n + 1 - k))
            p_k = p_below(k, n, L)
            p_km1 = p_below(k - 1, n, L) if k > 1 else float("nan")
            kt = tolerance_rank(n, L, DELTA)
            say(f"CELL L={float(L):.2f} n={n} k={k} gap={gap:.4f} sd={sd:.4f} "
                f"ratio={gap / sd:.4f} p_below_k={p_k:.4f} p_below_km1={p_km1:.4f} "
                f"k_tol={kt if kt is not None else 'none'}")
    say("")
    say("Read: the gap is one rank of the AVERAGE over calibration sets. A caller holds")
    say("one calibration set, whose coverage spreads by sd; ratio is how much of that")
    say("spread one rank is, and it falls with n like 1/sqrt(n L (1-L)).")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
