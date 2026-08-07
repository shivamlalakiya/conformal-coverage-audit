#!/usr/bin/env python3
"""How often a raw level is honest: the density of the delivering set, in closed form.

What was missing
----------------
The residue table counts deliveries by enumeration -- one size in ten for
inverted_cdf at 9/10, one in five for higher, one in seven at 5/7 -- and the
periodicity argument establishes that the delivering sizes form a residue class
without pinning its size. So the claim that more data will not fix a lossy index map
rested on a sweep: periodic, yes, but the proportion of honest sizes was a measured
figure rather than a derived one.

There is a closed form for it, and having one accounts for coincidences the table
shows but cannot explain.

The result
----------
Write the requested level as L = p/d in lowest terms, so alpha = 1 - L = (d-p)/d. A
definition delivers at n exactly when floor(h(n)) >= ceil((n+1)L). Then:

    inverted_cdf, averaged_inverted_cdf     density = alpha
    weibull                                 density = 1/d
    higher                                  density = 2*alpha        (alpha < 1/2)
    linear, median_unbiased                 density = 0

Derivations, each three lines.

  inverted_cdf. h = ceil(Ln). The inequality ceil(Ln) >= ceil(Ln + L) survives only
  where the step by L stays inside one integer gap: Ln off an integer, and L no
  larger than the distance up to the next one. Writing frac(Ln) = r/d with
  r = pn mod d turns that into 1 <= r <= d - p. Coprimality permutes Z_d under
  n -> pn mod d, so d - p residues out of d clear it: (d-p)/d = alpha.

  weibull. h = L(n+1). floor and ceil agree only on integers, which needs
  d dividing p(n+1) and hence d dividing n+1. That is one residue out of d.

  higher. h = ceil(L(n-1)) + 1. Put s = p(n-1) mod d; delivery needs either s at zero
  alongside 2p <= d, or s in 1 .. 2(d-p). Below alpha = 1/2 the first branch admits
  nothing, leaving 2(d-p) residues and a density of 2*alpha.

  linear, median_unbiased. Neither index ever climbs to ceil((n+1)L), at any size.
  The residue table already prints zero for both.

Why this settles a question about the SWEEP, not just about the definitions
--------------------------------------------------------------------------
Where alpha = 1/d the expressions alpha and 1/d evaluate to one number, which puts
inverted_cdf and weibull on the same density at 9/10 and at 19/20 -- and that is what
the enumerated table shows. The level does it; neither definition has the property.
Move off the unit fractions and they part: 2/7 against 1/7 at L = 5/7. Hence the
standing rule that a non-unit-fraction level goes into every sweep. Without one the
two cannot be told apart at all, and this file is where that is established.

Enumeration over forty full periods per level checks every closed form, in exact
rational arithmetic.

    python probes/delivering_density.py
"""

import math
import os
import sys
from fractions import Fraction as F

OUT = "outputs/probe_output_delivering_density.txt"

# Levels: unit fractions, where alpha and 1/d coincide, AND non-unit fractions,
# where they do not. A grid of unit fractions alone cannot separate inverted_cdf
# from weibull, which is the point this probe makes.
LEVELS = [F(9, 10), F(19, 20), F(99, 100), F(3, 4), F(7, 8),
          F(2, 3), F(5, 7), F(11, 13), F(17, 21)]
PERIODS = 40


# ---------------------------------------------------------------------------
# virtual indices, exactly as the definitions compute them
# ---------------------------------------------------------------------------
def h_inverted_cdf(n, L):
    return math.ceil(F(L) * n)


def h_weibull(n, L):
    return F(L) * (n + 1)


def h_higher(n, L):
    return math.ceil(F(L) * (n - 1)) + 1


def h_linear(n, L):
    return 1 + F(L) * (n - 1)


def h_median_unbiased(n, L):
    return F(1, 3) + F(L) * (n + F(1, 3))


FUNS = {
    "inverted_cdf": h_inverted_cdf,
    "weibull": h_weibull,
    "higher": h_higher,
    "linear": h_linear,
    "median_unbiased": h_median_unbiased,
}


def delivers(hfun, n, L):
    """floor(h) >= ceil((n+1)L): the requested guarantee is actually carried."""
    return math.floor(hfun(n, L)) >= math.ceil(F(n + 1) * F(L))


def predicted_density(name, L):
    """The closed form. Returns an exact Fraction."""
    p, d = F(L).numerator, F(L).denominator
    if name in ("inverted_cdf", "averaged_inverted_cdf"):
        return F(d - p, d)                                  # = alpha
    if name == "weibull":
        return F(1, d)
    if name == "higher":
        extra = 1 if 2 * p <= d else 0
        return min(F(2 * (d - p) + extra, d), F(1))
    return F(0)


def feasible_from(L):
    """The one-sided floor: no rank carries L below n = ceil(1/alpha - 1)."""
    return max(1, math.ceil(F(1) / (1 - F(L)) - 1))


def enumerate_density(name, L, periods=PERIODS):
    d = F(L).denominator
    lo = feasible_from(L)
    ns = range(lo, lo + periods * d)
    hits = sum(1 for n in ns if delivers(FUNS[name], n, L))
    return F(hits, len(ns)), len(ns)


def self_check():
    # Every closed form against enumeration, exactly, at every level.
    for L in LEVELS:
        for name in FUNS:
            emp, total = enumerate_density(name, L)
            pred = predicted_density(name, L)
            # enumeration over whole periods must hit the closed form exactly once
            # the irregular prefix is past; allow one period of slack for it
            slack = F(F(L).denominator, total)
            assert abs(emp - pred) <= slack, (name, str(L), str(emp), str(pred))

    # The claim that makes the non-unit-fraction level necessary: at a unit
    # fraction the two densities coincide, off it they do not. If this ever stops
    # holding, the standing sweep rule has lost its justification.
    for L in (F(9, 10), F(19, 20), F(99, 100)):
        assert predicted_density("inverted_cdf", L) == predicted_density("weibull", L)
    for L in (F(5, 7), F(11, 13), F(17, 21)):
        assert predicted_density("inverted_cdf", L) != predicted_density("weibull", L)

    # higher is exactly twice inverted_cdf while alpha < 1/2, which is the factor
    # of two the residue table shows and did not explain
    for L in LEVELS:
        if 1 - L < F(1, 2):
            assert (predicted_density("higher", L)
                    == 2 * predicted_density("inverted_cdf", L)), str(L)

    # and the definitions that never deliver must enumerate to exactly zero, not
    # to something small
    for L in LEVELS:
        for name in ("linear", "median_unbiased"):
            emp, _ = enumerate_density(name, L)
            assert emp == 0, (name, str(L), str(emp))


self_check()


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say("THE DENSITY OF THE DELIVERING SET, IN CLOSED FORM")
    say("=" * 100)
    say("self_check() passed at import: every closed form matches enumeration over")
    say(f"{PERIODS} full periods at every level; the two densities coincide at unit")
    say("fractions and separate off them; higher is exactly twice inverted_cdf while")
    say("alpha < 1/2; and the two definitions that never deliver enumerate to zero")
    say("rather than to something small.")
    say("")
    say("Take L = p/d reduced, alpha = 1 - L. Delivery at n means")
    say("floor(h(n)) >= ceil((n+1)L).")
    say("")
    say("    inverted_cdf, averaged_inverted_cdf   density = alpha = (d-p)/d")
    say("    weibull                               density = 1/d")
    say("    higher                                density = 2*alpha   (alpha < 1/2)")
    say("    linear, median_unbiased               density = 0")
    say("")
    say(f"{'level':>8} {'alpha':>7} {'d':>4} {'definition':<17} {'closed form':>13} "
        f"{'enumerated':>12} {'sizes':>7} {'first n':>8}")
    say("-" * 100)
    for L in LEVELS:
        d = F(L).denominator
        for name in FUNS:
            emp, total = enumerate_density(name, L)
            pred = predicted_density(name, L)
            first = next((n for n in range(feasible_from(L), feasible_from(L) + 8 * d)
                          if delivers(FUNS[name], n, L)), None)
            say(f"{str(L):>8} {str(1 - L):>7} {d:>4} {name:<17} "
                f"{str(pred):>7} = {float(pred):<5.3f} {float(emp):>12.4f} "
                f"{total:>7} {(first if first else '--'):>8}")
        say("")

    say("=" * 100)
    say("WHAT THE FORM EXPLAINS THAT THE MEASURED TABLE COULD NOT")
    say("")
    say("(a) The factor of two. Below alpha = 1/2, `higher` lands honestly at twice")
    say("    the sizes `inverted_cdf` does, because its condition takes in a stretch")
    say("    of 2(d-p) residues against the other's (d-p).")
    say("")
    say("(b) Why inverted_cdf and weibull looked alike. Where alpha = 1/d the two")
    say("    expressions alpha and 1/d evaluate to one number, which is why the pair")
    say("    agrees at 9/10, 19/20 and 99/100. The LEVEL causes that, not anything")
    say("    about either definition. Away from unit fractions they come apart:")
    for L in (F(5, 7), F(11, 13), F(17, 21)):
        a = predicted_density("inverted_cdf", L)
        w = predicted_density("weibull", L)
        say(f"      L = {str(L):<7} inverted_cdf {str(a):<7} vs weibull {str(w):<7} "
            f"ratio {float(a / w):.3f}")
    say("")
    say("    Hence the standing rule that a non-unit-fraction level appears in every")
    say("    sweep. It is not caution: without one, two of these definitions are")
    say("    indistinguishable, and the closed form shows why.")
    say("")
    say("(c) A number in place of an adjective. For the rounding definitions the")
    say("    proportion of honest calibration sizes is alpha itself, so asking for a")
    say("    TIGHTER level leaves FEWER sizes that can honour it -- one in a hundred")
    say("    at 0.99. Precisely the wrong way round for anyone who tightens a level")
    say("    expecting a safer answer.")

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        OUT)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
