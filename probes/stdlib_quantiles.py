#!/usr/bin/env python3
"""W16: what Python's own standard library resolves a cut point through.

Why this probe exists
---------------------
The interface survey measures six ecosystem defaults, all of them functions that
take an arbitrary level. Python's standard library ships a quantile function too,
`statistics.quantiles`, and it takes no level at all. What it is handed is a number
of equal parts, and what it returns is the boundaries between them. It was left out of the survey for
that reason, and the sentence that survives the survey -- no default reaches the
sole convention that hands back the coverage it was asked for -- is therefore
stated over a set the standard library is not in.

That is a scope a reader cannot see, and the reader most likely to notice is a
Python programmer who has read the `statistics` documentation. This probe settles
what the omitted default actually does, so the scope sentence can name it instead
of quietly excluding it.

What is measured, and in what arithmetic
----------------------------------------
Block (i) prints the call signature, which is the whole reason for the exclusion:
there is no level parameter to pass a requested coverage to.

Block (ii) is EXACT and is the one the claim rests on. `statistics` computes in
whatever arithmetic its input carries, so the tie-free set 1..n is handed to it as
`Fraction`s and the cut points come back as `Fraction`s. On that set the values
are their own ranks, so the return IS the position the function computed, with no
rounding anywhere in the path. Each position is compared against the two closed
forms in exact rationals -- no tolerance, and none needed.

Block (iii) repeats the same grid in doubles against numpy, because that is what a
caller gets. It separates two disagreements that look alike and are not: numpy
clips a position to [1, n] and the standard library extrapolates past the sample,
which is a difference of behaviour; and where both stay inside the sample the two
differ only in the last bits, which is a difference of arithmetic order.

Reproducing it
--------------
    python probes/stdlib_quantiles.py

Needs numpy and nothing else; no audited library and no network. `statistics` is
the interpreter's own, so the version line below is the interpreter's.
"""

import os
import statistics
import sys
from fractions import Fraction as F
from inspect import signature

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_stdlib_quantiles.txt")

# The two methods the standard library documents, its default first, each beside
# the closed-form position it is being tested against and the numpy name that
# computes the same position. The forms are written out rather than looked up so
# that a reader can check them against Hyndman and Fan without leaving this file.
METHODS = (
    ("exclusive", "(k/m)(n+1)", lambda n, m, k: F(k, m) * (n + 1), "weibull"),
    ("inclusive", "1 + (k/m)(n-1)", lambda n, m, k: 1 + F(k, m) * (n - 1),
     "linear"),
)
SIZES = (2, 3, 5, 10, 20, 50, 99, 100, 250, 400)
GROUPS = (4, 10, 20, 100)

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def exact_scores(n):
    """The tie-free set whose values are their own ranks, in exact rationals."""
    return [F(i) for i in range(1, n + 1)]


def cells():
    """Every (n, m, k) the comparison runs over. k runs over the m-1 cut points
    the standard library returns; it returns no endpoint, and numpy resolves the
    endpoints identically for every definition anyway."""
    for n in SIZES:
        for m in GROUPS:
            for k in range(1, m):
                yield n, m, k


def self_check():
    """The four facts this probe rests on, each with an input that breaks it."""
    # (1) The arithmetic really is exact: handed Fractions, the standard library
    #     returns Fractions. If a release ever coerced to float, every "exact"
    #     count below would become a count of lucky roundings, so this is the
    #     load-bearing assertion of the file.
    got = statistics.quantiles(exact_scores(10), n=4)[0]
    assert isinstance(got, F), type(got)
    # (2) The instrument. On 1..n the return IS the position, so the middle cut
    #     point of an odd count sits at the middle rank.
    assert statistics.quantiles(exact_scores(9), n=2)[0] == 5
    # (3) The two documented methods are different calls, so a comparison that
    #     found them equal at every cell would be reading one of them twice.
    assert (statistics.quantiles(exact_scores(50), n=10)[8]
            != statistics.quantiles(exact_scores(50), n=10,
                                    method="inclusive")[8])
    # (4) The closed forms are tested against each other, not just against the
    #     library: they must disagree somewhere, or matching both would be free.
    ex, inc = METHODS[0][2], METHODS[1][2]
    assert ex(50, 10, 9) != inc(50, 10, 9)


self_check()


def main():
    grid = list(cells())
    say("=" * 104)
    say("W16 STDLIB QUANTILES: the default Python ships and the survey omits")
    say("=" * 104)
    say(f"    python  {sys.version.split()[0]}")
    say(f"    numpy   {np.__version__}")
    say(f"    cells   {len(grid)} = {len(SIZES)} sizes x "
        f"{sum(m - 1 for m in GROUPS)} cut points")
    say("")

    say("-" * 104)
    say("(i) THE SIGNATURE. There is no level parameter, which is why this")
    say("    entry point is not a row of the interface table.")
    say("-" * 104)
    say(f"    statistics.quantiles{signature(statistics.quantiles)}")
    say("    `n` here is how many equal parts, not the sample size, and the")
    say("    return is the m-1 cut points between them. A caller wanting a")
    say("    level has to choose m so that the level is k/m, so a level whose")
    say("    denominator is large costs a proportionally large m, and an")
    say("    irrational level cannot be asked for at all.")
    say("")

    say("-" * 104)
    say("(ii) EXACT. The set 1..n handed over as Fractions, so the cut point")
    say("     comes back as the position itself with no rounding in the path,")
    say("     and compared against the closed form in exact rationals.")
    say("-" * 104)
    exact_hits = {}
    for name, form, closed, npname in METHODS:
        hits = 0
        for n, m, k in grid:
            got = statistics.quantiles(exact_scores(n), n=m, method=name)[k - 1]
            hits += (got == closed(n, m, k))
        exact_hits[name] = hits
        default = " (the default)" if name == METHODS[0][0] else ""
        say(f"    method='{name}'{default}: position = {form} at "
            f"{hits} of {len(grid)} cells")
        say(f"        which is numpy's `{npname}` virtual index")
    say("")

    outside = [(n, m, k) for n, m, k in grid
               if not 1 <= METHODS[0][2](n, m, k) <= n]
    say(f"     the default's position falls outside the sample, below rank 1 or")
    say(f"     above rank n, at {len(outside)} of {len(grid)} cells; there the")
    say("     standard library extrapolates past the smallest and largest score")
    say("     while numpy clips to the sample. That is the one behavioural")
    say("     difference between the two, and it is not a difference of")
    say("     definition.")
    say("")

    say("-" * 104)
    say("(iii) IN DOUBLES, against numpy, which is what a caller gets. Only the")
    say("      cells whose position is inside the sample, so numpy's clip is")
    say("      not counted as a disagreement.")
    say("-" * 104)
    inside = [(n, m, k) for n, m, k in grid if (n, m, k) not in set(outside)]
    for name, _form, closed, npname in METHODS:
        same, worst = 0, 0.0
        for n, m, k in inside:
            got = statistics.quantiles([float(i) for i in range(1, n + 1)],
                                       n=m, method=name)[k - 1]
            ref = float(np.quantile(np.arange(1.0, n + 1.0), k / m,
                                    method=npname))
            same += (got == ref)
            worst = max(worst, abs(got - ref))
        say(f"    method='{name}' against numpy `{npname}`: identical double at "
            f"{same} of {len(inside)} cells,")
        say(f"        largest disagreement {worst:.3e}, which is the last bits "
            f"of a double and not a rank")
    say("")

    say("-" * 104)
    say("(iv) ONE CELL IN FULL, the running example of the manuscripts:")
    say("     50 calibration scores and a ninth decile.")
    say("-" * 104)
    for name, _form, _closed, npname in METHODS:
        pos = statistics.quantiles([float(i) for i in range(1, 51)],
                                   n=10, method=name)[8]
        ref = float(np.quantile(np.arange(1.0, 51.0), 0.9, method=npname))
        say(f"    statistics.quantiles(1..50, n=10, method='{name}')[8] = {pos}")
        say(f"    numpy.quantile(1..50, 0.9, method='{npname}')"
            f"{' ' * (12 - len(npname))} = {ref}")
    say("")
    say("    The default lands at (k/m)(n+1), the mean-rank plotting position,")
    say("    so the level a caller would have had to ask for equals the coverage")
    say("    the threshold is predicted to deliver. What carries the")
    say("    finite-sample guarantee is the order statistic below that position,")
    say("    and that is a strictly lower rank wherever the position is not an")
    say("    integer.")
    say("")

    ex_ok = exact_hits[METHODS[0][0]] == len(grid)
    say("=" * 104)
    say(f"MACHINE sq_cells={len(grid)} sq_sizes={len(SIZES)} "
        f"sq_groups={len(GROUPS)} sq_outside={len(outside)} "
        f"sq_default_is_weibull={'yes' if ex_ok else 'no'}")
    for name, form, _closed, npname in METHODS:
        say(f"MACHINE-METHOD {name} | form={form} | numpy={npname} | "
            f"exact_cells={exact_hits[name]}")
    say("=" * 104)

    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwritten -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
