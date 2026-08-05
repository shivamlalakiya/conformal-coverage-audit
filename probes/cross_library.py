#!/usr/bin/env python3
"""W16: the level-to-rank map across quantile APIs, not just numpy's.

Why this probe exists
---------------------
The manuscript claims the defect is "a property of the interfaces, not of any
release". It supports that claim entirely from `numpy.quantile`, and its Limitations
section concedes that a widely used statistical language's default type is
arithmetically identical to numpy's `linear` but that this was NOT EXECUTED. That is
a soft spot in a paper whose thesis is about interfaces. This probe closes as much of
it as the available toolchain allows, and states plainly what it cannot reach.

What it found that the taxonomy does not contain
-----------------------------------------------
`scipy.stats.mstats.mquantiles` exposes the Hyndman-Fan (alphap, betap)
parameterisation directly and defaults to alphap = betap = 0.4 -- the Cunnane
plotting position. That is NOT one of Hyndman and Fan's nine definitions, and it is
therefore not a row in the manuscript's rank map. A caller who reaches for scipy's
quantile function rather than numpy's gets a fourteenth convention, with its own
virtual index, its own delivered coverage and its own corrected level, none of which
the audit currently covers.

That is the sharper form of the interface claim: the taxonomy is incomplete with
respect to what is shipped, not merely under-used. The rank map is a map of the nine
definitions plus numpy's rounding variants; the space of shipped conventions is
larger, and the fractional-rank result covers all of it because it is stated in terms
of the virtual index rather than in terms of a named definition.

Method
------
The same instrument as everywhere else: quantile the tie-free score set 1..n, whose
values ARE their own ranks, so whatever an API returns IS its virtual index. No
algebra is trusted, and each API's affine coefficients (A, B) are FITTED at interior
levels rather than read from documentation -- the numpy clip at [1, n] makes an
endpoint fit return the wrong coefficients, which is recorded in W9.

What this probe cannot reach, stated rather than implied
-------------------------------------------------------
R, Julia, Octave/MATLAB and pyspark are not installed in this environment, so their
conventions are NOT executed here and this probe makes no measured claim about them.
For R specifically the manuscript's existing position stands unchanged: `type=7` is
the Hyndman-Fan definition with alpha = beta = 1, arithmetically identical to numpy's
`linear`, and we cite rather than claim. Block (iii) records the (alpha, beta) pair
each named type corresponds to so a reader with those toolchains can reproduce the
row, and marks every one of them as unexecuted.
"""

import math
import os
import sys
from fractions import Fraction as F

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_cross_library.txt")

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def scores(n):
    """Tie-free score set: the returned value IS the 1-indexed virtual index."""
    return np.arange(1, n + 1, dtype=float)


# ---------------------------------------------------------------------------
# the APIs actually installed here
# ---------------------------------------------------------------------------
def apis():
    import pandas as pd
    from scipy.stats import mstats

    out = {}
    for meth in ("linear", "higher", "inverted_cdf", "weibull", "median_unbiased"):
        out[f"numpy.quantile method='{meth}'"] = (
            lambda n, q, m=meth: float(np.quantile(scores(n), q, method=m)))
    # pandas exposes a SUBSET of numpy's methods under a different keyword name
    for interp in ("linear", "higher", "lower", "midpoint", "nearest"):
        out[f"pandas.Series.quantile interpolation='{interp}'"] = (
            lambda n, q, i=interp: float(pd.Series(scores(n)).quantile(
                q, interpolation=i)))
    # scipy exposes the (alphap, betap) family directly, and its DEFAULT is not
    # one of the nine Hyndman-Fan definitions
    out["scipy mstats.mquantiles DEFAULT (alphap=betap=0.4)"] = (
        lambda n, q: float(mstats.mquantiles(scores(n), prob=[q])[0]))
    for a, b, name in ((1, 1, "alphap=betap=1 (= numpy linear)"),
                       (0, 0, "alphap=betap=0 (= weibull)"),
                       (F(1, 3), F(1, 3), "alphap=betap=1/3 (= median_unbiased)"),
                       (0.5, 0.5, "alphap=betap=0.5 (= hazen)")):
        out[f"scipy mstats.mquantiles {name}"] = (
            lambda n, q, a=float(a), b=float(b): float(
                mstats.mquantiles(scores(n), prob=[q], alphap=a, betap=b)[0]))
    return out


# named conventions we can NAME but not RUN: recorded so a reader with the
# toolchain can reproduce the row, and marked unexecuted
NOT_RUN = [
    ("R quantile(type=7) [default]", F(1), F(1), "identical to numpy linear"),
    ("R quantile(type=4)", F(0), F(1), "interpolated_inverted_cdf"),
    ("R quantile(type=5)", F(1, 2), F(1, 2), "hazen"),
    ("R quantile(type=6)", F(0), F(0), "weibull"),
    ("R quantile(type=8)", F(1, 3), F(1, 3), "median_unbiased"),
    ("R quantile(type=9)", F(3, 8), F(3, 8), "normal_unbiased"),
    ("Julia Statistics.quantile [default]", F(1), F(1), "identical to numpy linear"),
    ("MATLAB/Octave quantile [default]", F(1, 2), F(1, 2), "hazen"),
]


def affine(fn, n):
    """(A, B) with h = A + Bq, fitted at INTERIOR levels.

    Not at the endpoints: every API clips the virtual index into [1, n], so an
    endpoint fit returns (1, n-1) regardless of the convention. W9 records the
    same trap.
    """
    q1, q2 = 3.0 / (n + 1), 1.0 - 3.0 / (n + 1)
    h1, h2 = fn(n, q1), fn(n, q2)
    B = (h2 - h1) / (q2 - q1)
    A = h1 - B * q1
    # Affinity must be checked on a GRID, not at a third point. A rounding
    # definition is a staircase whose treads are one apart and whose slope is
    # about n, so three chosen points can lie on a straight line by coincidence
    # -- `higher` at n=100 does exactly that, and self_check rejected the
    # three-point version for it.
    grid = np.linspace(q1, q2, 41)
    dev = max(abs(fn(n, float(q)) - (A + B * float(q))) for q in grid)
    return (A, B) if dev < 1e-9 else (None, None)


def q_needed(fn, n, alpha):
    """Smallest level delivering 1-alpha, by bisection on the virtual index."""
    target = (1.0 - alpha) * (n + 1)
    if fn(n, 1.0) + 1e-12 < target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fn(n, mid) + 1e-12 >= target:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
def self_check():
    A = apis()
    # (1) the instrument: on the tie-free set every API must return a value in
    #     [1, n], because that is what a virtual index is
    for name, fn in A.items():
        for n in (10, 50, 200):
            for q in (0.0, 0.25, 0.9, 1.0):
                h = fn(n, q)
                assert 1 - 1e-9 <= h <= n + 1e-9, (name, n, q, h)

    # (2) the correspondences this probe asserts must hold exactly, or the
    #     cross-API mapping is wrong
    for n in (10, 47, 200):
        for q in (0.9, 0.95, F(7, 11)):
            q = float(q)
            assert abs(A["pandas.Series.quantile interpolation='linear'"](n, q)
                       - A["numpy.quantile method='linear'"](n, q)) < 1e-9, (n, q)
            assert abs(A["scipy mstats.mquantiles alphap=betap=1 (= numpy linear)"](n, q)
                       - A["numpy.quantile method='linear'"](n, q)) < 1e-9, (n, q)
            assert abs(A["scipy mstats.mquantiles alphap=betap=0 (= weibull)"](n, q)
                       - A["numpy.quantile method='weibull'"](n, q)) < 1e-9, (n, q)
            assert abs(A["scipy mstats.mquantiles alphap=betap=1/3 (= median_unbiased)"](n, q)
                       - A["numpy.quantile method='median_unbiased'"](n, q)) < 1e-9

    # (3) scipy's DEFAULT must differ from every numpy method, or the claim that
    #     it is a fourteenth convention is false
    for n in (50, 200):
        d = A["scipy mstats.mquantiles DEFAULT (alphap=betap=0.4)"](n, 0.9)
        for name, fn in A.items():
            if name.startswith("numpy"):
                assert abs(d - fn(n, 0.9)) > 1e-6, (
                    f"scipy default coincides with {name} at n={n}; it is not a "
                    f"distinct convention and the claim must be withdrawn")
        # and it must match the (0.4, 0.4) Hyndman-Fan form
        want = 0.4 + 0.9 * (n + 1 - 0.4 - 0.4)
        assert abs(d - want) < 1e-9, (n, d, want)

    # (4) the affine fit must succeed for the continuous APIs and FAIL (return
    #     None) for the rounding ones, which are step functions
    for name, fn in A.items():
        Aa, Bb = affine(fn, 100)
        step = any(k in name for k in ("higher", "lower", "nearest",
                                       "inverted_cdf", "midpoint"))
        if step and "alphap" not in name:
            assert Aa is None, f"{name} fitted as affine but is a step function"
        else:
            assert Aa is not None, f"{name} is affine but the fit failed"


self_check()


def main():
    A = apis()
    n, alpha = 50, 0.10

    say("=" * 110)
    say("W16  THE LEVEL-TO-RANK MAP ACROSS QUANTILE APIs")
    say("=" * 110)
    say("")
    say("  Instrument: quantile the tie-free score set 1..n, so the returned value")
    say("  IS the virtual index. (A, B) with h = A + Bq are FITTED at interior")
    say(f"  levels. n = {n}, requested coverage {1 - alpha:.2f}.")
    say("")

    # ---------------- (i) every installed API ----------------------------
    say("-" * 110)
    say("(i) INSTALLED APIs, MEASURED")
    say("-" * 110)
    say(f"{'API':<52}{'A':>8}{'B':>8}{'h':>9}{'delivered':>11}"
        f"{'guarantee':>11}{'q needed':>10}")
    rows = []
    for name, fn in A.items():
        Aa, Bb = affine(fn, n)
        h = fn(n, 1 - alpha)
        qn = q_needed(fn, n, alpha)
        rows.append({"api": name, "A": Aa, "B": Bb, "h": h,
                     "delivered": h / (n + 1),
                     "guarantee": math.floor(h + 1e-12) / (n + 1),
                     "q_needed": qn})
        ab = f"{Aa:>8.4f}{Bb:>8.3f}" if Aa is not None else f"{'step':>8}{'step':>8}"
        say(f"{name:<52}{ab}{h:>9.3f}{h / (n + 1):>11.4f}"
            f"{math.floor(h + 1e-12) / (n + 1):>11.4f}"
            f"{('---' if qn is None else f'{qn:.4f}'):>10}")
    say("")

    # ---------------- (ii) the fourteenth convention ---------------------
    say("-" * 110)
    say("(ii) A CONVENTION THE TAXONOMY DOES NOT CONTAIN")
    say("-" * 110)
    d = [r for r in rows if "DEFAULT" in r["api"]][0]
    say(f"    scipy.stats.mstats.mquantiles defaults to alphap = betap = 0.4, the")
    say(f"    Cunnane plotting position. That pair is not among Hyndman and Fan's")
    say(f"    nine definitions, so it is not a row in the rank map.")
    say(f"      virtual index at the raw level : {d['h']:.4f}")
    say(f"      delivered coverage h/(n+1)     : {d['delivered']:.4f}"
        f"   against a requested {1 - alpha:.2f}")
    say(f"      distribution-free guarantee    : {d['guarantee']:.4f}")
    say(f"      corrected level it needs       : {d['q_needed']:.4f}")
    say("")
    say("    Its delivered coverage differs from every numpy method (asserted in")
    say("    self_check), so a caller who reaches for scipy rather than numpy gets a")
    say("    fourteenth convention with its own corrected level. The rank map is a")
    say("    map of the nine definitions; the space of SHIPPED conventions is larger.")
    say("    The fractional-rank result covers it anyway, because it is stated in")
    say("    terms of the virtual index rather than of a named definition -- which is")
    say("    the argument for stating it that way.")
    say("")

    # ---------------- (iii) named but not executed -----------------------
    say("-" * 110)
    say("(iii) NAMED BUT NOT EXECUTED. R, Julia and Octave/MATLAB are not installed")
    say("      in this environment. No measured claim is made about them. Their")
    say("      (alpha, beta) pairs are recorded so a reader with the toolchain can")
    say("      reproduce the row, and the delivered figure below is what our own")
    say("      instrument gives for that pair -- an ARITHMETIC transfer, not a run.")
    say("-" * 110)
    say(f"{'convention':<40}{'alpha':>8}{'beta':>8}{'h':>9}{'delivered':>11}"
        f"{'status':>10}   equals")
    for name, a, b, note in NOT_RUN:
        h = float(a) + (1 - alpha) * (n + 1 - float(a) - float(b))
        say(f"{name:<40}{str(a):>8}{str(b):>8}{h:>9.3f}{h / (n + 1):>11.4f}"
            f"{'NOT RUN':>10}   {note}")
    say("")
    say("    pyspark (approxQuantile, an approximate streaming estimator like the")
    say("    one in branch (g)) is also unavailable and unmeasured.")
    say("")

    # ---------------- (iv) what agrees, and what that means --------------
    say("-" * 110)
    say("(iv) AGREEMENT ACROSS APIs")
    say("-" * 110)
    by_h = {}
    for r in rows:
        by_h.setdefault(round(r["h"], 9), []).append(r["api"])
    say(f"    {len(rows)} installed API configurations produce {len(by_h)} distinct")
    say("    virtual indices. Configurations that coincide:")
    for h, names in sorted(by_h.items()):
        if len(names) > 1:
            say(f"      h = {h:.4f}  ({len(names)}): " + " | ".join(names))
    say("")
    say("    The defaults are what a caller actually gets, and they do NOT agree:")
    for r in rows:
        if "method='linear'" in r["api"] or "interpolation='linear'" in r["api"] \
                or "DEFAULT" in r["api"]:
            say(f"      {r['api']:<52} delivers {r['delivered']:.4f}")
    say("")
    say("=" * 110)
    say("SUMMARY")
    say("=" * 110)
    say(f"  {len(rows)} shipped API configurations across three packages, "
        f"{len(by_h)} distinct")
    say("  virtual indices, and one default convention -- scipy's -- that is not in")
    say("  the Hyndman-Fan taxonomy at all. numpy and pandas agree with each other by")
    say("  construction; scipy does not agree with either at its own default. The")
    say("  interface claim is therefore stronger than the manuscript states, and the")
    say("  right way to state it is in terms of the virtual index.")
    say("  R, Julia, Octave/MATLAB and pyspark are NOT measured here and no claim is")
    say("  made about them beyond the arithmetic transfer recorded in block (iii).")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
