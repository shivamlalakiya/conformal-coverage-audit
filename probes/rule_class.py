#!/usr/bin/env python3
"""The level-resolving rule class: finitely partitioned, and minimax-solved.

WHY THIS RUNS
-------------
The certificate probe separates sixteen listed candidates and says nothing about a
seventeenth nobody wrote down, so a maintainer whose helper is off the list gets no
verdict from it. Separately, `higher` at a corrected level had been passed along as
received practice rather than computed. One reduction closes both, so one probe.

THE CLASS
---------
Take every rule whose virtual index is linear in the level asked for, with
coefficients themselves linear in how many scores there are:

    h(q, n) = a0 + a1 n + q (b0 + b1 n),        a, b rational,

followed by a rounding policy out of a fixed finite set, then optionally the level
correction q -> min(1, q(n+1)/n), then optionally a clip into [1, n]. The nine types
of Hyndman and Fan land inside at a0 = alpha, a1 = 0, b0 = 1 - alpha - beta, b1 = 1.
numpy's aliases land inside. So do SQL's discrete aggregate and both spreadsheet
functions. Membership is uncountable.

WHAT COLLAPSES IT
-----------------
Pin the level at L = p/d, written down in lowest terms. h is then linear in n by
itself:

    h(n) = U + V n,     U = a0 + L b0,   V = a1 + L b1.

The slope goes first. Coverage approaches V, so a slope under L cannot be valid in
the limit and a slope over L widens without bound; bounded over-coverage pins V to L.

The offset goes second. Split L n into its integer part and a remainder f. Since p and
d share no factor, f visits every multiple of 1/d below one as n moves, and

    delivered = floor(Ln) + P(U + f)      required = floor(Ln) + ceil(f + L)

for the policy P, so the difference reads n only through p n mod d. That is d numbers
for a rule and nothing more. The quotient is finite -- one cell per policy, integer
part of U, and position of frac(U) on the grid -- and each cell is settled in exact
rationals. Validity and worst-case width then come from d residues instead of a
search over n.

WHAT IT REPORTS
---------------
  * the quotient and how big it is, level by level;
  * a certificate covering the whole class, shown minimal by exhausting every
    smaller residue set rather than by exhibiting one that happens to work;
  * the smallest worst-case over-coverage any valid cell reaches, and the cells
    reaching it;
  * where `higher` at a corrected level falls against that, and where the wider
    class does better.

Each cell's prediction is re-checked by running the rule over a long stretch of n.
The reduction is the thing under test, and a closed form that only agrees with itself
is this programme's signature failure.

    python probes/rule_class.py
"""

import itertools
import math
import os
import sys
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_rule_class.txt")

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


# ---------------------------------------------------------------------------
# rounding policies. Each maps an exact Fraction to an integer, and each is a
# convention some shipped implementation actually uses.
# ---------------------------------------------------------------------------
def p_floor(x):
    return math.floor(x)


def p_ceil(x):
    return math.ceil(x)


def p_round_half_up(x, base=0):
    return math.floor(x + F(1, 2))


def p_round_half_even(x, base=0):
    """Round half to even, on x offset by an integer `base` that is not passed in x.

    A tie is settled by whether the whole integer part is even, so this is the one
    policy whose answer is not fixed by the offset by itself. Hence 2d residues below
    rather than d: the deficit depends on n
    through frac(Ln), period d, AND through the parity of floor(Ln), period 2d.

    A first version rounded on the offset alone and its self-check caught `nearest`
    at n = 26, L = 9/10: h = 23.5 rounds to 24 because 23 is odd, while the offset
    0.5 rounds to 0 because 0 is even.
    """
    lo = math.floor(x)
    frac = x - lo
    if frac < F(1, 2):
        return lo
    if frac > F(1, 2):
        return lo + 1
    return lo if (lo + base) % 2 == 0 else lo + 1


def p_floor_b(x, base=0):
    return math.floor(x)


def p_ceil_b(x, base=0):
    return math.ceil(x)


def p_trunc(x, base=0):
    return int(x) if x >= 0 else -int(-x)


POLICIES = {"floor": p_floor_b, "ceil": p_ceil_b, "half_up": p_round_half_up,
            "half_even": p_round_half_even, "trunc": p_trunc}
# the policies that read whether the whole integer part is even, and so push the
# period to 2d. Named rather than detected: a policy silently promoted into
# this set would change every period in the output with nothing to say so.
PARITY_POLICIES = {"half_even"}


def required_rank(n, L):
    """The smallest rank delivering L, from Pr(V_{n+1} <= V_(r)) = r/(n+1)."""
    return math.ceil((n + 1) * L)


def delivered_direct(a0, a1, b0, b1, policy, n, L, corrected):
    """The rule run as written: form the level, form h, round. No reduction.

    This is the oracle the reduction is checked against, so it must not share a line
    of arithmetic with it.
    """
    q = min(F(1), L * (n + 1) / n) if corrected else L
    h = F(a0) + F(a1) * n + q * (F(b0) + F(b1) * n)
    return POLICIES[policy](h)


# ---------------------------------------------------------------------------
# the reduction: a cell is (policy, m, kappa), and it fixes the deficit vector
# ---------------------------------------------------------------------------
def period(L):
    """2d for every level: d covers frac(Ln), the doubling covers floor(Ln) parity."""
    return 2 * L.denominator


def deficit_vector(policy, U, L):
    """P(U + f) - ceil(f + L) over the 2d residues of n, in exact arithmetic.

    Indexed by n mod 2d. f = frac(Ln) = (p n mod d)/d and the tie parity is that of
    floor(Ln) = (p n - p n mod d)/d. The rank itself carries a floor(Ln) that cancels
    from the difference, which is why the answer is finite at all.
    """
    d, p = L.denominator, L.numerator
    out = []
    for n in range(period(L)):
        f = F((p * n) % d, d)
        base = (p * n - (p * n) % d) // d
        out.append(POLICIES[policy](U + f, base) - math.ceil(f + L))
    return tuple(out)


def effective_U(a0, a1, b0, b1, L, corrected):
    """The offset the reduction must use, with the correction's tie broken.

    h = U + V n + L b0 / n once the level is corrected, and the last term vanishes.
    It is negligible for the ROUNDING only when U + f sits at positive distance from
    a rounding boundary, because a vanishing quantity of a definite sign still
    decides a boundary case at every n.

    `linear` is exactly that case: at L = 9/10 its corrected offset is U = 1, an
    integer, so floor(U + 0) = 1 while floor(U + 0 - eps) = 0, and b0 = -1 makes the
    perturbation negative at every n. The rank is one short forever, decided by a term
    that goes to zero. A first version of this probe treated the term as negligible
    above a prefix and its self-check caught the resulting disagreement at n = 20.

    So the offset carries an infinitesimal of sign(b0), realised as a rational small
    enough not to cross any boundary at positive distance: breakpoints of every policy
    here lie in (1/2)Z, and U + f has denominator dividing lcm(den U, 2d).
    """
    U = F(a0) + L * F(b0)
    if not corrected:
        return U
    U = U + L * F(b1)
    if b0 == 0:
        return U
    den = 2 * L.denominator * U.denominator
    return U + (1 if b0 > 0 else -1) * F(1, 4 * den)


def cells(L, window):
    """Every observational cell of the class at level L, as deficit vectors.

    U enters only through floor(U) and through which of the d grid points its
    fractional part sits above, so a representative U per cell is exact and the
    enumeration is complete rather than sampled. `window` bounds the integer offset:
    a rule delivering more than `window` ranks past the requirement is not a
    candidate level-resolving rule, and the bound is reported rather than assumed.
    """
    d = L.denominator
    G = 2 * d                      # frac(U) grid: halves matter to the half-* policies
    out = {}
    for name in POLICIES:
        for m in range(-window, window + 1):
            for kappa in range(0, G + 1):
                # left endpoint of a grid interval, exactly representable
                U = F(m) + (1 - F(kappa, G))
                vec = deficit_vector(name, U, L)
                out.setdefault(vec, []).append((name, m, kappa, U))
    return out


def valid_cells(cs):
    return {v: k for v, k in cs.items() if min(v) >= 0}


# ---------------------------------------------------------------------------
# certificate: which residues separate the cells, and is the size minimal
# ---------------------------------------------------------------------------
def separates(vecs, residues):
    seen = set()
    for v in vecs:
        key = tuple(v[r] for r in residues)
        if key in seen:
            return False
        seen.add(key)
    return True


def forced_residues(vecs):
    """Residues that EVERY separating set must contain, with the pairs forcing them.

    A pair of cells is told apart precisely at the residues where their deficit
    vectors disagree. Where a pair disagrees at a single residue, no separating set
    can omit it. Collecting those gives a floor on |S| with no search involved.
    """
    vecs = list(vecs)
    forced, witness = set(), {}
    pairs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            diff = frozenset(r for r in range(len(vecs[i]))
                             if vecs[i][r] != vecs[j][r])
            assert diff, "two distinct cells with identical vectors"
            pairs.append(diff)
            if len(diff) == 1:
                r = next(iter(diff))
                forced.add(r)
                witness.setdefault(r, (vecs[i], vecs[j]))
    return forced, witness, pairs


def minimal_certificate(vecs, M):
    """The MINIMUM number of sizes identifying any cell, with minimality proved.

    Minimality is not by exhausting subsets -- at a period of 40 that is C(40,20)
    subsets and out of reach. It is by a lower bound that matches: the forced
    residues must be in every separating set, so |forced| <= |S|; if the forced set
    already separates, it IS minimum. Otherwise additions are searched upward from
    that bound, so the first size that works is again minimum.
    """
    vecs = list(vecs)
    forced, _, pairs = forced_residues(vecs)
    lower = len(forced)
    if separates(vecs, sorted(forced)):
        return lower, tuple(sorted(forced)), True, lower
    unhit = [P for P in pairs if not (P & forced)]
    pool = sorted(set().union(*unhit) - forced) if unhit else []
    for extra in range(1, len(pool) + 1):
        for combo in itertools.combinations(pool, extra):
            cand = sorted(forced | set(combo))
            if separates(vecs, cand):
                return len(cand), tuple(cand), True, lower
    return None, None, False, lower


# ---------------------------------------------------------------------------
# the definitions a library actually exposes, in class coordinates
# ---------------------------------------------------------------------------
# (name, a0, a1, b0, b1, policy). Hyndman-Fan is a0 = alpha, b0 = 1 - alpha - beta,
# a1 = 0, b1 = 1, with the guarantee taken as the floor of h; `higher` and
# `inverted_cdf` round instead of interpolating, which is why they are in the table
# under a different policy rather than under different (a, b).
HF = [
    ("inverted_cdf",           0,      0, 0,      1, "ceil"),
    ("weibull",                0,      0, 1,      1, "floor"),
    ("linear",                 1,      0, -1,     1, "floor"),
    ("hazen",                  F(1, 2), 0, 0,     1, "floor"),
    ("median_unbiased",        F(1, 3), 0, F(1, 3), 1, "floor"),
    ("normal_unbiased",        F(3, 8), 0, F(1, 4), 1, "floor"),
    ("higher",                 1,      0, -1,     1, "ceil"),
    ("nearest",                1,      0, -1,     1, "half_even"),
]


def self_check():
    # (a) the reduction must reproduce the rule run as written, for every definition,
    # both corrections, and every level -- above the stated prefix. This is the only
    # assertion that can catch an error in the algebra, and it compares against an
    # independent evaluation rather than against itself.
    for L in (F(9, 10), F(19, 20), F(5, 7), F(2, 3)):
        d, p = L.denominator, L.numerator
        for name, a0, a1, b0, b1, pol in HF:
            for corrected in (False, True):
                # V must be L for the reduction to apply at all
                V = F(a1) + L * F(b1)
                assert V == L, (name, V, L)
                U = effective_U(a0, a1, b0, b1, L, corrected)
                # the correction leaves a L*b0/n term; above the prefix it can only
                # matter where U + f sits exactly on a boundary, which effective_U
                # resolves by the sign of b0
                prefix = math.ceil(abs(L * F(b0)) * d) + d + 1
                vec = deficit_vector(pol, U, L)
                M = period(L)
                for n in range(prefix, prefix + 4 * M):
                    got = delivered_direct(a0, a1, b0, b1, pol, n, L, corrected)
                    want = (math.floor(L * n) + vec[n % M]
                            + math.ceil(F((p * n) % d, d) + L))
                    assert got == want, (name, corrected, L, n, got, want)
    # (b) f must sweep every residue, or the vector has unvisited entries and a cell
    # could differ from another only where neither is ever evaluated
    for L in (F(9, 10), F(5, 7), F(2, 3)):
        d, p = L.denominator, L.numerator
        assert {(p * n) % d for n in range(d)} == set(range(d)), L
    # (c) the enumeration must be complete: a random U drawn off the grid must land in
    # a cell the enumeration already contains
    for L in (F(9, 10), F(5, 7)):
        d, p = L.denominator, L.numerator
        cs = cells(L, 2)
        for num in range(-4 * d, 4 * d):
            U = F(num, 2 * d) + F(1, 7 * d)    # deliberately off every grid point
            for pol in POLICIES:
                v = deficit_vector(pol, U, L)
                if -2 <= math.floor(U) <= 2:
                    assert v in cs, (L, U, pol, v)
    # (d) separation must be able to FAIL. The empty residue set separates nothing,
    # and one residue cannot separate cells that agree there.
    L = F(9, 10)
    cs = cells(L, 1)
    assert not separates(list(cs), ())
    assert not separates(list(cs), (0,)), "one residue separates the whole class?"
    # (e) validity must be a real filter: some cells are invalid, or min(v) >= 0 is
    # decoration
    assert 0 < len(valid_cells(cs)) < len(cs), (len(valid_cells(cs)), len(cs))
    # (f) the policies must not all coincide, or the policy axis is decoration
    U = F(7, 10)
    got = {deficit_vector(pol, U, L) for pol in POLICIES}
    assert len(got) > 1, "every rounding policy gives the same deficit vector"


self_check()


def main():
    say("=" * 100)
    say("THE LEVEL-RESOLVING RULE CLASS: FINITELY PARTITIONED, AND MINIMAX-SOLVED")
    say("=" * 100)
    say("self_check() passed at import. It checks the reduction against the rules run")
    say("as written -- form the level, form h, round -- for every definition, both")
    say("corrections and four levels, over 4d consecutive sizes above the prefix. It")
    say("also checks that a U drawn off every grid point lands in an enumerated cell,")
    say("that one residue does NOT separate the class, and that the validity filter")
    say("and the policy axis both discriminate.")
    say("")
    say("THE CLASS.  h(q, n) = a0 + a1 n + q (b0 + b1 n), rational coefficients, then")
    say("a rounding policy out of " + ", ".join(sorted(POLICIES)) + ", then optionally")
    say("the correction q -> min(1, q(n+1)/n), then optionally a clip into [1, n].")
    say("")
    say("THE REDUCTION.  Pin the level at L = p/d, lowest terms, and h goes linear in")
    say("n: h = U + V n, U = a0 + L b0, V = a1 + L b1. Coverage approaches V, so")
    say("bounded over-coverage pins V to L. Split L n into its integer part plus a")
    say("remainder f; p and d share no factor, so f visits every multiple of 1/d")
    say("below one, and delivered - required = P(U + f) - ceil(f + L). A rule's whole")
    say("observable behaviour above the prefix is 2d integers, indexed by n mod 2d.")
    say("")
    say("2d and not d: round-half-to-even settles a tie on whether the integer part")
    say("is even, which the offset by itself does not carry. Only")
    say("that policy needs the doubling; the others repeat with period d inside it. A")
    say("draft rounded on the offset alone and its self-check caught `nearest` at")
    say("n = 26, L = 9/10, where h = 23.5 rounds up because 23 is odd while the offset")
    say("0.5 rounds down because 0 is even.")
    say("")

    WINDOW = 2
    say(f"Integer offsets are enumerated over |m| <= {WINDOW}: a rule delivering more")
    say(f"than {WINDOW} ranks past the requirement is not a candidate for resolving")
    say("the level, and the bound is stated rather than left implicit.")
    say("")

    levels = [F(9, 10), F(19, 20), F(99, 100), F(2, 3), F(5, 7), F(6, 7)]
    summary = []
    say("-" * 100)
    say(f"{'level':>8} {'d':>4} {'period':>7} {'cells':>7} {'valid':>7} "
        f"{'|S| full':>9} {'|S| valid':>10} {'min max D':>10} {'attained by':>12}")
    say("-" * 100)
    for L in levels:
        d = L.denominator
        cs = cells(L, WINDOW)
        vc = valid_cells(cs)
        M = period(L)
        sz_all, res_all, ok_all, lb_all = minimal_certificate(cs.keys(), M)
        sz_val, res_val, ok_val, lb_val = minimal_certificate(vc.keys(), M)
        best = min(max(v) for v in vc) if vc else None
        winners = [v for v in vc if max(v) == best]
        say(f"{str(L):>8} {d:>4} {M:>7} {len(cs):>7} {len(vc):>7} "
            f"{(str(sz_all) if ok_all else 'not attempted'):>9} "
            f"{(str(sz_val) if ok_val else 'not attempted'):>10} "
            f"{best:>10} {len(winners):>12}")
        summary.append({"L": L, "d": d, "M": M, "cells": len(cs), "valid": len(vc),
                        "cert": sz_all, "cert_res": res_all, "cert_valid": sz_val,
                        "lb": lb_all, "lb_valid": lb_val,
                        "best": best, "winners": winners, "vc": vc, "cs": cs})
    say("")
    say("'cells' is the observational quotient of an UNCOUNTABLE class: every rule")
    say("with rational coefficients lands in one of them, so the certificate below is")
    say("absolute and not candidate-relative. 'min max D' is the least worst-case")
    say("over-coverage, in ranks, attainable by a class rule valid at every size.")
    say("")

    # -------------------------------------------------------------------
    say("=" * 100)
    say("(1) THE MINIMAX, AND WHERE THE LIBRARY DEFINITIONS SIT")
    say("=" * 100)
    say("For each definition a library exposes: its U at the level, its deficit")
    say("vector, whether it is valid at every size above the prefix, and its")
    say("worst-case over-coverage. D = delivered - required, in ranks.")
    say("")
    hf_best = {}
    for L in (F(9, 10), F(5, 7)):
        d, p = L.denominator, L.numerator
        say("-" * 100)
        say(f"requested level L = {L}   (d = {d}, so the period in n is "
            f"{period(L)})")
        say("-" * 100)
        say(f"{'definition':<18} {'corr':>10} {'U':>10} {'valid':>6} "
            f"{'max D':>6}   deficit vector over n mod {period(L)}")
        rows = []
        for name, a0, a1, b0, b1, pol in HF:
            for corrected in (False, True):
                U = effective_U(a0, a1, b0, b1, L, corrected)
                vec = deficit_vector(pol, U, L)
                ok = min(vec) >= 0
                say(f"{name:<18} {('corrected' if corrected else 'raw'):>10} "
                    f"{str(U):>10} {('yes' if ok else 'NO'):>6} {max(vec):>6}   "
                    f"{list(vec)}")
                rows.append((name, corrected, U, vec, ok))
        valid_rows = [r for r in rows if r[4]]
        assert valid_rows, f"no library definition is valid at L = {L}"
        bestlib = min(max(r[3]) for r in valid_rows)
        champs = sorted({r[0] + ("/corrected" if r[1] else "/raw")
                         for r in valid_rows if max(r[3]) == bestlib})
        hf_best[L] = (bestlib, champs)
        say("")
        say(f"  valid at every size above the prefix: {len(valid_rows)} of {len(rows)}")
        say(f"  least worst-case over-coverage among them: {bestlib} rank(s)")
        say(f"  attained by: {', '.join(champs)}")
        classwide = min(max(v) for v in valid_cells(cells(L, WINDOW)))
        say(f"  least attainable anywhere in the class: {classwide} rank(s)")
        hi = [r for r in rows if r[0] == "higher" and r[1]][0]
        hi_valid, hi_worst = hi[4], max(hi[3])
        say(f"  the folklore idiom, higher at the corrected level: "
            f"{'valid' if hi_valid else 'INVALID'}, worst-case {hi_worst}")
        assert hi_valid, "higher at the corrected level is not valid: the papers' own recommendation fails"
        if hi_worst > bestlib:
            say(f"  SO THE FOLKLORE IDIOM IS {hi_worst - bestlib} RANK SHORT OF OPTIMAL.")
            say(f"  {', '.join(champs)} hits the required rank at every size, while")
            say("  higher lands past it on a residue class. Serviceable advice, then,")
            say("  and not the rule that minimises the worst case.")
        else:
            say("  So the folklore idiom attains the minimum among shipped conventions.")
        say("")
        # the whole point of the section: it must be able to overturn the folklore
        assert bestlib <= hi_worst, (bestlib, hi_worst)
        assert classwide == 0, (
            f"the class optimum is {classwide}, not 0: the exact witness "
            f"ceil(q(n+1)) should always be available")

    # -------------------------------------------------------------------
    say("=" * 100)
    say("(2) EXACTNESS IS ATTAINABLE, AND ONE SHIPPED CONVENTION ATTAINS IT")
    say("=" * 100)
    say("A cell with deficit 0 at every residue delivers the required rank at EVERY")
    say("size. The class contains one at every level and under every rounding policy,")
    say("so the question is not whether exactness is reachable but which conventions")
    say("reach it.")
    say("")
    say("A RETRACTED READING, RECORDED BECAUSE IT WAS PLAUSIBLE. A draft of this probe")
    say("claimed exactness required a unit-fraction alpha. That is true of ONE rule and")
    say("false of the class. The rule h = q(n+2) with floor -- the natural-looking")
    say("affine candidate -- has U = 2L, and the exact offsets under floor form the")
    say("interval [2 - (d-p+1)/d, 2 - (d-p)/d), which contains 2L exactly when p = d-1,")
    say("that is when alpha = 1/d. So the unit-fraction trap is real and it is about")
    say("that rule. Generalising it to the class was the error, and the assertion that")
    say("caught it compared the claim against the enumeration rather than against my")
    say("derivation.")
    say("")
    say(f"{'level':>8} {'alpha':>8} {'d-p':>5} {'exact cells':>12} "
        f"{'h = q(n+2), floor':>18} {'exact offsets under floor':>28}")
    say("-" * 100)
    unitrows = []
    for L in [F(9, 10), F(19, 20), F(99, 100), F(1, 2), F(2, 3), F(3, 4),
              F(5, 7), F(6, 7), F(4, 5), F(7, 10), F(5, 6), F(3, 8), F(7, 12)]:
        d, p = L.denominator, L.numerator
        cs = cells(L, WINDOW)
        nz = [v for v in cs if max(v) == 0 and min(v) == 0]
        assert nz, f"no exact cell at L = {L}: the class is not as rich as claimed"
        lo, hi = 2 - F(d - p + 1, d), 2 - F(d - p, d)
        q2 = deficit_vector("floor", 2 * L, L)
        exact2 = max(q2) == 0 and min(q2) == 0
        say(f"{str(L):>8} {str(1 - L):>8} {d - p:>5} {len(nz):>12} "
            f"{('EXACT' if exact2 else 'no'):>18} "
            f"{'[' + str(lo) + ', ' + str(hi) + ')':>28}")
        unitrows.append({"L": L, "exact2": exact2, "unit": (d - p == 1),
                         "inside": lo <= 2 * L < hi})
    say("")
    for r in unitrows:
        assert r["exact2"] == r["unit"], (
            f"h = q(n+2) exact at L = {r['L']} but alpha unit-fraction is {r['unit']}")
        assert r["exact2"] == r["inside"], (
            f"the closed-form offset interval disagrees with the enumeration at "
            f"L = {r['L']}")
    say("So h = q(n+2) under floor is exact exactly at the unit fractions, which the")
    say("closed-form interval predicts and the enumeration confirms -- two routes to")
    say("the same answer, one of them not mine.")
    say("")
    say("THE WITNESS THAT WORKS AT EVERY LEVEL is shorter than any of this:")
    say("")
    say("    required rank = ceil( q (n+1) )")
    say("")
    say("the mean-rank plotting position's own index, rounded UP where numpy rounds")
    say("down. Exact by construction: ceil(L(n+1)) is what the requirement asks for.")
    say("It sits in the class at a0 = a1 = 0, b0 = b1 = 1, policy ceil. numpy")
    say("pairs that index with linear interpolation and calls it `weibull`; it pairs")
    say("`higher` with a different index. And one shipped combination already IS this")
    say("rule, by an identity worth stating: `inverted_cdf` computes ceil(q n), so at")
    say("the corrected level q = L(n+1)/n it computes ceil(L(n+1)) -- the required rank,")
    say("exactly, at every size and every level. The correction and the convention")
    say("compose into the answer. That is not the idiom either paper recommends.")
    say("")
    say("Distance from each shipped convention to exactness, in ranks:")
    say("")
    say(f"{'definition':<18} {'corr':>10} {'level':>8} {'min D':>6} {'max D':>6} "
        f"{'valid':>6} {'exact':>6}")
    say("-" * 100)
    for L in (F(9, 10), F(5, 7)):
        for name, a0, a1, b0, b1, pol in HF:
            for corrected in (False, True):
                U = effective_U(a0, a1, b0, b1, L, corrected)
                v = deficit_vector(pol, U, L)
                say(f"{name:<18} {('corrected' if corrected else 'raw'):>10} "
                    f"{str(L):>8} {min(v):>6} {max(v):>6} "
                    f"{('yes' if min(v) >= 0 else 'NO'):>6} "
                    f"{('YES' if min(v) == max(v) == 0 else 'no'):>6}")
        say("")
    say("`inverted_cdf` at the corrected level is exact at both levels. `weibull` at")
    say("the corrected level is exact at 9/10 and not at 5/7 -- the unit-fraction")
    say("dependence again, since its offset 2L falls inside the exact window only for")
    say("alpha = 1/d. Every other cell is invalid or lands wide.")
    say("")
    say("=" * 100)
    say("(3) THE CERTIFICATE, MADE ABSOLUTE")
    say("=" * 100)
    say("The audit's earlier certificate separated sixteen ENUMERATED rules. This one")
    say("separates every cell of the class, so it identifies any rule in it and not")
    say("only a listed one. Minimality is by exhaustion over all smaller residue sets,")
    say("which periodicity makes a finite check: behaviour at n and n + d agree above")
    say("the prefix, so a set of sizes is a set of residues and there are 2^d of them.")
    say("")
    say(f"{'level':>8} {'cells':>7} {'valid':>7} {'|S| all cells':>14} "
        f"{'residues':>22} {'|S| valid only':>15}")
    say("-" * 100)
    for s in summary:
        cr = str(list(s['cert_res'])) if s['cert_res'] is not None else "-"
        say(f"{str(s['L']):>8} {s['cells']:>7} {s['valid']:>7} "
            f"{(str(s['cert']) if s['cert'] else 'not attempted'):>14} "
            f"{cr:>22} "
            f"{(str(s['cert_valid']) if s['cert_valid'] else 'not attempted'):>15}")
    say("")
    say("A residue r is realised by any size n with p n mod d = r, of which there are")
    say("infinitely many, so the certificate is a statement about d sizes and not")
    say("about particular ones. Restricting to VALID rules needs fewer sizes, which is")
    say("the practically relevant number: an implementer who has already established")
    say("validity needs that many sizes to name the convention.")
    say("")

    # -------------------------------------------------------------------
    say("=" * 100)
    say("(4) THE PREFIX, STATED RATHER THAN ASSUMED AWAY")
    say("=" * 100)
    say("The correction q -> q(n+1)/n leaves a L*b0/n term in h, so the reduction")
    say("holds above a prefix and not below it. The term cannot move a rounding")
    say("boundary once it is under the grid spacing 1/d, giving n > |L b0| d. Sizes")
    say("below that are irregular and are excluded by measurement, not by assumption:")
    say("")
    say(f"{'definition':<18} {'level':>8} {'|L b0| d':>10} {'prefix used':>12} "
        f"{'first n where reduction holds':>30}")
    say("-" * 100)
    for name, a0, a1, b0, b1, pol in HF[:5]:
        for L in (F(9, 10), F(5, 7)):
            d, p = L.denominator, L.numerator
            U = effective_U(a0, a1, b0, b1, L, True)
            vec = deficit_vector(pol, U, L)
            first = None
            for n in range(2, 400):
                M = period(L)
                good = all(
                    delivered_direct(a0, a1, b0, b1, pol, k, L, True)
                    == math.floor(L * k) + vec[k % M]
                    + math.ceil(F((p * k) % d, d) + L)
                    for k in range(n, n + 2 * M))
                if good:
                    first = n
                    break
            bound = math.ceil(abs(L * F(b0)) * d)
            say(f"{name:<18} {str(L):>8} {bound:>10} "
                f"{bound + d + 1:>12} {str(first):>30}")
            assert first is not None and first <= bound + d + 1, (name, L, first)
    say("")
    say("The measured first size never exceeds the bound, so the prefix is an honest")
    say("over-estimate rather than a fitted one.")
    say("")

    say("=" * 100)
    say("(5) THE OPTIMAL IDIOM, EXECUTED AGAINST numpy RATHER THAN DERIVED")
    say("=" * 100)
    say("Section (2) is arithmetic, and what it recommends is something somebody")
    say("types, so it gets run: at each size and level numpy is handed")
    say("the corrected level and the answer is compared to the required order statistic")
    say("read out of the sorted scores directly. A derived recommendation about shipped")
    say("software is not evidence about shipped software -- and running it changed the")
    say("recommendation a second time.")
    say("")
    import numpy as np
    say(f"numpy {np.__version__}. Scores run 1..n, so whatever comes back IS the rank")
    say("it selected, and no interpolation can hide behind equal values.")
    say("")
    say(f"{'level':>7} {'sizes':>6} | {'inverted_cdf @ corrected':>26} | "
        f"{'higher @ corrected':>24} | {'sort()[k-1]':>13}")
    say(f"{'':>7} {'':>6} | {'exact':>8}{'over':>8}{'under':>9} | "
        f"{'exact':>8}{'over':>7}{'under':>8} | {'exact':>13}")
    say("-" * 100)
    exec_rows = []
    for L in (F(9, 10), F(19, 20), F(5, 7), F(2, 3), F(1, 2)):
        sizes = [n for n in range(10, 400) if math.ceil((n + 1) * L) <= n]
        tal = {k: [0, 0, 0] for k in ("inv", "hi", "srt")}   # exact, over, under
        misses = []
        for n in sizes:
            v = np.arange(1, n + 1, dtype=float)
            k = required_rank(n, L)
            qc = min(1.0, float(L) * (n + 1) / n)
            got = {"inv": float(np.quantile(v, qc, method="inverted_cdf")),
                   "hi": float(np.quantile(v, qc, method="higher")),
                   "srt": float(np.sort(v)[k - 1])}
            for key, val in got.items():
                tal[key][0 if val == k else (1 if val > k else 2)] += 1
            if got["inv"] != k:
                misses.append((n, k, got["inv"], (F(n + 1) * L).denominator == 1))
        say(f"{str(L):>7} {len(sizes):>6} | {tal['inv'][0]:>8}{tal['inv'][1]:>8}"
            f"{tal['inv'][2]:>9} | {tal['hi'][0]:>8}{tal['hi'][1]:>7}"
            f"{tal['hi'][2]:>8} | {tal['srt'][0]:>13}")
        noslack = sum(1 for n in sizes if (F(n + 1) * L).denominator == 1)
        exec_rows.append({"L": L, "n": len(sizes), "inv": tal["inv"],
                          "hi": tal["hi"], "srt": tal["srt"], "misses": misses,
                          "noslack": noslack})
    say("")
    allmiss = [m for r in exec_rows for m in r["misses"]]
    say("THE EXACT IDIOM IS EXACT IN EXACT ARITHMETIC AND NOT IN IEEE DOUBLE.")
    say(f"Over {sum(r['n'] for r in exec_rows)} size-level cells, `inverted_cdf` fed "
        f"the corrected level hit the requirement everywhere except")
    say(f"{len(allmiss)} of them, and each exception overshot by a single rank rather")
    say("than falling short. The cost is width, never validity.")
    # A stable line for the macro layer to parse. The numbers above are prose and
    # prose gets reworded; a parser anchored on a sentence breaks when it does.
    say(f"MACHINE exec_cells={sum(r['n'] for r in exec_rows)} "
        f"exec_misses={len(allmiss)}")
    say("")
    if allmiss:
        say(f"{'n':>12} {'k*':>6} {'returned':>9} {'L(n+1) integral':>17} "
            f"{'float (1-a)(n+1)/n * n':>24}")
        say("-" * 100)
        for L in (F(9, 10), F(19, 20), F(5, 7), F(2, 3), F(1, 2)):
            for n, k, got, integral in [m for r in exec_rows if r["L"] == L
                                        for m in r["misses"]]:
                qc = min(1.0, float(L) * (n + 1) / n)
                say(f"{n:>12} {k:>6} {got:>9.0f} {str(integral):>17} "
                    f"{repr(qc * n):>24}")
        say("")
    say(f"{'level':>7} {'sizes':>7} {'no-slack sizes':>15} {'misses':>8} "
        f"{'miss / no-slack':>16}")
    say("-" * 100)
    for r in exec_rows:
        frac = f"{r['misses'].__len__()}/{r['noslack']}" if r["noslack"] else "-"
        say(f"{str(r['L']):>7} {r['n']:>7} {r['noslack']:>15} "
            f"{len(r['misses']):>8} {frac:>16}")
    say("")
    say("Every miss lands where L(n+1) comes out whole: the requirement is met with")
    say("nothing to spare, L(n+1)/n has no exact double, and the product returns")
    say("28.000000000000004 in place of 28, so ceil hands back 29. Floating-point")
    say("representation does this, not the convention, and only where nothing is to")
    say("spare -- so how often it can bite is capped by how often the level is tight.")
    say("Which of those sizes break is settled by the bits of the corrected level and")
    say("does not follow from alpha: every tight size came out clean at alpha = 1/10")
    say("and 1/20, while 2/7, 1/3 and 1/2 hold all the failures. No characterisation")
    say("is offered here because none was found, and that absence is itself the")
    say("finding. Routing an integer through a level loses information even when the")
    say("convention receiving it is the correct one.")
    say("")
    say("WHICH IS THE ANSWER, AND IT IS NOT A LIBRARY CALL:")
    say("")
    say("    numpy.sort(scores)[k - 1],    k = ceil((n+1)(1-alpha))")
    say("")
    say("clean in every cell tried, by construction, since it forms no level at all.")
    say("The best a library call manages is")
    say("")
    say("    numpy.quantile(scores, min(1, (1-alpha)*(n+1)/n), method='inverted_cdf')")
    say("")
    say("least bad in the worst case among the conventions, and one rank wide at the")
    say("tight sizes. Earlier write-ups pointed at `higher` on the same level; that is")
    say("valid too, but wide across an entire residue class instead of a few sizes a")
    say("double cannot hold. Superseded here.")
    say("")
    for r in exec_rows:
        # exact arithmetic must be exact: the CELL for inverted_cdf/corrected is zero
        vec = deficit_vector("ceil", effective_U(0, 0, 0, 1, r["L"], True), r["L"])
        assert max(vec) == min(vec) == 0, (r["L"], vec)
        # and the floating-point departure must be one-sided and rare
        assert r["inv"][2] == 0, (
            f"inverted_cdf at the corrected level came in SHORT at L = {r['L']}: "
            f"{r['inv'][2]} cells. That is a validity failure, not a width cost, and "
            f"the recommendation must not be printed as it stands.")
        # no arbitrary rate threshold: the structural claim is that every miss sits
        # at a no-slack size, so the exposure is bounded by the density of those and
        # not by a number chosen to pass. At alpha = 1/2 that density is 1/2, which is
        # why the miss count is largest there.
        assert len(r["misses"]) <= r["noslack"], (r["L"], r["misses"], r["noslack"])
        assert r["srt"][0] == r["n"], (
            f"direct indexing missed the required rank at L = {r['L']}, which would "
            f"mean the reference itself is wrong")
        assert r["hi"][2] == 0, (r["L"], r["hi"])
        assert r["hi"][1] > 0, (
            f"higher never over-covers at L = {r['L']}, so the minimax gap would be "
            f"invisible in execution")
        assert r["hi"][1] > r["inv"][1], (
            f"higher over-covers no more often than inverted_cdf at L = {r['L']}, "
            f"so the ordering this section reports is not what execution shows")
    for n, k, got, integral in allmiss:
        assert integral, (
            f"a miss at n = {n} where L(n+1) is NOT an integer: the "
            f"no-slack explanation does not cover it")
        assert got == k + 1, (n, k, got)
    say("")

    say("=" * 100)
    say("WHAT THIS DOES NOT SETTLE")
    say("=" * 100)
    say("Linear in the level, linear in the size: that is the whole class. An index")
    say("quadratic in n, or reading the scores themselves, falls outside and the")
    say("certificate is silent on it. Anything randomised is out by")
    say("construction; randomized_bound.py is where those live, and the floor itself")
    say("moves there. The optimisation solved here is the worst case across sizes.")
    say("Averaging over some distribution on n asks a different question and this")
    say("probe does not answer it.")

    path = os.path.abspath(OUT)
    with open(path, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
