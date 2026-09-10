#!/usr/bin/env python3
"""Pricing the two substitutions in span rather than in position.

Why this file exists
--------------------
Elsewhere here, two substitutions are put forward: keep the level and swap in
`weibull`, whose position turns the level into exactly that much coverage, or
divide the level and swap in `inverted_cdf`, whose position is the index the
guarantee demands. Both have so far been quoted in positions. Positions are the
right internal unit and the wrong unit for anybody weighing an edit to running
code, where the question is how much the band grows.

So this file runs the three calls over identical draws and divides. A band formed
by adding and subtracting the returned cutoff spans twice that cutoff, so a
quotient of cutoffs already equals the quotient of band spans; no further algebra
is applied to it.

Draws are non-negative throughout, matching how nonconformity is usually built --
an absolute residual, or a predicted probability negated and shifted. Five laws
cover the range of upper-tail weight used elsewhere in this repository, from
bounded support to a Pareto with unbounded variance, since the separation between
neighbouring positions high in the sample is governed by tail weight alone.

What gets measured
------------------
For every (law, size) pairing, across independent draws:

    default     numpy.quantile(scores, 1-alpha)                  method='linear'
    weibull     numpy.quantile(scores, 1-alpha, 'weibull')
    corrected   numpy.quantile(scores, (1-alpha)(n+1)/n, 'inverted_cdf')

Printed as the baseline averaged in whatever units the law carries, together with
each substitute over the baseline, divided draw by draw and then averaged. Dividing
two averages instead would answer a different question than the one somebody
editing a single call is asking.

Sign is settled before drawing anything and re-asserted after. At level 1-alpha
the three positions are

    linear        1 + (1-alpha)(n-1)
    weibull       (1-alpha)(n+1)
    inverted_cdf  ceil((1-alpha)(n+1))            at the divided level

These do not decrease down that list at any n the request is satisfiable at, so
against non-negative draws no quotient printed here drops under unity. Should one
do so the ordering has moved and execution halts.

    python probes/interval_width.py
"""

import math
import os

import numpy as np

OUT = "outputs/probe_output_interval_width.txt"

SEED = 20260906
REPS = 100_000
CHUNK = 10_000
SIZES = (20, 50, 100, 200)
LEVELS = (0.90, 0.95, 0.99)
HEADLINE = 0.90

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def laws(rng):
    """Non-negative draw laws, listed from lightest upper tail to heaviest."""
    return {
        "uniform": lambda s: rng.uniform(size=s),
        "half-normal": lambda s: np.abs(rng.standard_normal(s)),
        "exponential": lambda s: rng.exponential(size=s),
        "lognormal": lambda s: rng.lognormal(size=s),
        "pareto1.5": lambda s: 1.0 + rng.pareto(1.5, size=s),
    }


def virtual_index(n, q, method):
    """Whatever position numpy settles on, taken from it rather than reasoned to.

    On the ladder 1..n each entry coincides with its own index, so the value
    handed back states the position outright.
    """
    return float(np.quantile(np.arange(1, n + 1, dtype=float), q, method=method))


def corrected_level(n, q):
    return min(q * (n + 1) / n, 1.0)


def required_rank(n, q):
    return math.ceil((n + 1) * q)


def self_check():
    # (1) each of the three positions matches its published formula, taken from
    #     numpy at sizes picked near the satisfiability boundary, not round ones.
    for n in (9, 10, 11, 19, 20, 47, 100, 201):
        for q in (0.90, 0.95, 0.99):
            if math.ceil((n + 1) * q) > n:
                continue
            assert abs(virtual_index(n, q, "linear")
                       - (1 + q * (n - 1))) < 1e-8, (n, q, "linear")
            assert abs(virtual_index(n, q, "weibull")
                       - q * (n + 1)) < 1e-8, (n, q, "weibull")
            got = virtual_index(n, corrected_level(n, q), "inverted_cdf")
            assert abs(got - math.ceil(q * (n + 1))) <= 1.0 + 1e-8, (n, q, "invcdf")

    # (2) the order every cell below relies on, tested against the positions
    #     directly, so a break points at a definition and not at a draw.
    for n in SIZES:
        for q in LEVELS:
            a = virtual_index(n, q, "linear")
            b = virtual_index(n, q, "weibull")
            c = virtual_index(n, corrected_level(n, q), "inverted_cdf")
            assert a <= b + 1e-9 <= c + 1e-9, (n, q, a, b, c)

    # (3) a calibration with an answer known in advance: on the ladder 1..n what
    #     comes back IS the position, so the quotient formed below has to match
    #     the quotient of positions to the last bit. A break means the quotient
    #     names one thing and computes another.
    n, q = 100, 0.90
    v = np.arange(1, n + 1, dtype=float)
    d = float(np.quantile(v, q, method="linear"))
    w = float(np.quantile(v, q, method="weibull"))
    assert abs(w / d - virtual_index(n, q, "weibull")
               / virtual_index(n, q, "linear")) < 1e-12

    # (4) a case that must NOT be silently averaged away: at n = 9 and q = 0.90
    #     the divided level tops out at 1 and the call yields the largest draw.
    #     The sweep begins above that size, and this pins the boundary so that a
    #     later edit lowering SIZES trips here rather than printing a quotient
    #     whose numerator is that largest draw.
    assert corrected_level(9, 0.90) == 1.0
    assert min(SIZES) > 9


self_check()


def cell(name, law, n, q):
    """Average baseline cutoff, plus the two averaged per-draw span quotients."""
    qc = corrected_level(n, q)
    tot_d = tot_w = tot_c = 0.0
    sq_w = sq_c = 0.0
    done = 0
    while done < REPS:
        r = min(CHUNK, REPS - done)
        S = law((r, n))
        d = np.quantile(S, q, axis=1, method="linear")
        w = np.quantile(S, q, axis=1, method="weibull")
        c = np.quantile(S, qc, axis=1, method="inverted_cdf")
        rw, rc = w / d, c / d
        tot_d += float(d.sum())
        tot_w += float(rw.sum())
        tot_c += float(rc.sum())
        sq_w += float((rw ** 2).sum())
        sq_c += float((rc ** 2).sum())
        done += r
    mean_w, mean_c = tot_w / REPS, tot_c / REPS
    se_w = math.sqrt(max(sq_w / REPS - mean_w ** 2, 0.0) / REPS)
    se_c = math.sqrt(max(sq_c / REPS - mean_c ** 2, 0.0) / REPS)
    k = required_rank(n, q)
    return {"law": name, "n": n, "q": q, "k": k, "at_max": k == n,
            "default": tot_d / REPS,
            "weibull": mean_w, "corrected": mean_c, "se_w": se_w, "se_c": se_c}


def main():
    rng = np.random.default_rng(SEED)
    L = laws(rng)

    say("=" * 100)
    say("SPAN: THE PRICE OF EITHER SUBSTITUTE, MEASURED AGAINST WHAT SHIPS")
    say("=" * 100)
    say("self_check() cleared at import: positions taken from numpy, their order")
    say("pinned, and the size where the level tops out kept out of the averages")
    say("")
    say(f"numpy {np.__version__}   seed {SEED}   replicates per cell {REPS}")
    say("")
    say("A band spans twice its cutoff, so dividing cutoffs already divides spans.")
    say("Each quotient is averaged across draws; two averages are never divided.")
    say("Column `default` keeps whatever units the law has. The quotient columns")
    say("carry none, and may therefore be compared down the page.")
    say("")

    rows = []
    say("(1) AT THE LEVEL THIS REPOSITORY LEADS WITH")
    say("-" * 100)
    say(f"{'law':<14}{'n':>5}{'level':>7}{'k*':>5}{'default':>11}{'weibull /':>12}"
        f"{'s.e.':>9}{'invcdf@corr /':>16}{'s.e.':>9}")
    say(f"{'':<14}{'':>5}{'':>7}{'':>5}{'(units)':>11}{'default':>12}{'':>9}"
        f"{'default':>16}{'':>9}")
    say("-" * 100)
    for name, fn in L.items():
        for n in SIZES:
            r = cell(name, fn, n, HEADLINE)
            rows.append(r)
            say(f"{name:<14}{n:>5}{HEADLINE:>7.2f}{r['k']:>5}{r['default']:>11.4f}"
                f"{r['weibull']:>12.5f}{r['se_w']:>9.5f}"
                f"{r['corrected']:>16.5f}{r['se_c']:>9.5f}")
    say("")

    say("(2) TWO FURTHER LEVELS AT A SINGLE SIZE: THE PRICE TRACKS THE LEVEL")
    say("-" * 100)
    say(f"{'law':<14}{'n':>5}{'level':>7}{'k*':>5}{'default':>11}{'weibull /':>12}"
        f"{'s.e.':>9}{'invcdf@corr /':>16}{'s.e.':>9}")
    say("-" * 100)
    other = []
    for name, fn in L.items():
        for q in LEVELS:
            if q == HEADLINE:
                continue
            r = cell(name, fn, 100, q)
            other.append(r)
            say(f"{name:<14}{100:>5}{q:>7.2f}{r['k']:>5}{r['default']:>11.4f}"
                f"{r['weibull']:>12.5f}{r['se_w']:>9.5f}"
                f"{r['corrected']:>16.5f}{r['se_c']:>9.5f}")
    say("")

    allr = rows + other
    for r in allr:
        assert r["weibull"] >= 1.0, ("weibull came back narrower than the default, "
                                     "which reverses the index order", r)
        assert r["corrected"] >= r["weibull"], (
            "the corrected inverted_cdf call came back narrower than weibull, "
            "which reverses the index order", r)

    inter = [r for r in allr if not r["at_max"]]
    worst_w = max(allr, key=lambda r: r["weibull"])
    worst_c = max(allr, key=lambda r: r["corrected"])
    worst_i = max(inter, key=lambda r: r["corrected"])
    mean_w = sum(r["weibull"] for r in allr) / len(allr)
    mean_c = sum(r["corrected"] for r in allr) / len(allr)

    say("(3) THE BILL")
    say("-" * 100)
    say("Neither substitution shrinks anything; both enlarge. That follows from how")
    say("the three positions are ordered and needs no measurement. The k* column")
    say("splits two regimes. When k* coincides with n the requirement names the")
    say("largest draw, one step up from the region where no index whatsoever meets")
    say("the request, and the quotient then describes the uppermost spacing of a")
    say("heavy-tailed sample instead of describing the definitions. Elsewhere the")
    say("charge is a few per cent and shrinks with n, as the intervening positions")
    say("crowd together.")
    say("")
    say(f"  pairings                    {len(allr)}, with {len(inter)} under k* = n")
    say(f"  weibull, worst cell         {worst_w['weibull']:.5f}x  "
        f"({worst_w['law']}, n={worst_w['n']}, level {worst_w['q']:.2f}, "
        f"k*={worst_w['k']})")
    say(f"  weibull, mean cell          {mean_w:.5f}x")
    say(f"  corrected, worst cell       {worst_c['corrected']:.5f}x  "
        f"({worst_c['law']}, n={worst_c['n']}, level {worst_c['q']:.2f}, "
        f"k*={worst_c['k']})")
    say(f"  corrected, worst below k*=n {worst_i['corrected']:.5f}x  "
        f"({worst_i['law']}, n={worst_i['n']}, level {worst_i['q']:.2f}, "
        f"k*={worst_i['k']})")
    say(f"  corrected, mean cell        {mean_c:.5f}x")
    say("")
    say("A fixed line for whatever reads this file. Everything above is wording and")
    say("wording gets revised; anything keyed to a sentence snaps when it does.")
    say(f"MACHINE iw_cells={len(allr)} iw_interior={len(inter)} iw_laws={len(L)} "
        f"iw_sizes={len(SIZES)} iw_reps={REPS}")
    say(f"MACHINE iw_weibull_worst={worst_w['weibull']:.5f} "
        f"iw_weibull_mean={mean_w:.5f} "
        f"iw_corrected_worst={worst_c['corrected']:.5f} "
        f"iw_corrected_mean={mean_c:.5f} "
        f"iw_corrected_interior_worst={worst_i['corrected']:.5f}")
    say(f"MACHINE iw_worst_law={worst_c['law']} iw_worst_n={worst_c['n']} "
        f"iw_worst_level={worst_c['q']:.2f} iw_interior_law={worst_i['law']} "
        f"iw_interior_n={worst_i['n']} iw_interior_level={worst_i['q']:.2f}")

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        OUT)
    with open(path, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
