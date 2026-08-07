#!/usr/bin/env python3
"""W13: a minimal certificate -- which calibration sizes a test suite must use.

Why this probe exists
---------------------
Two results in this programme point at the same missing piece.

One result says a unit test nailed to a coincidence cell keeps passing while the
helper it guards is wrong at every other size, and names a library whose own fixture
values sit at exactly such a size. W9 then shows those cells are not scattered luck:
for every quantile definition they form a residue class with a density you can
compute. Together they make fixture choice a decidable question rather than a
preference. Either the sizes a suite probes tell the candidate rules apart, or they
do not, and which of those holds can be settled.

So decide it. This probe computes the smallest set S of calibration sizes such that
observing a helper's delivered rank at every n in S identifies which rule it
implements, and proves the identification extends to all n rather than holding only
on S.

The candidate set, and why the certificate is relative to it
------------------------------------------------------------
Candidates come from three choices multiplied out. Which rails: one at 1-alpha, or a
pair at 1-alpha/2. Whether the level is corrected and clipped at 1, or left raw. And
which family resolves it: linear, higher, inverted_cdf, or an index read straight off
the sorted scores with no level in between. Sixteen in total.

Where a helper matches none of them, what that reports is a gap in the candidate list
-- a fact about this instrument, not about that library. Scope inherited, and stated
here so it is not mistaken for a verdict. The certificate therefore reads "these
sizes separate the sixteen candidates", not "these sizes identify any conceivable
implementation". Stating it the weaker way is the honest form and it is still the
useful one, because the sixteen are what the audited libraries actually do.

Why the extension to all n is provable rather than hoped, and where it starts
----------------------------------------------------------------------------
Feed a rule the RAW level and the rank it lands on is a floor or ceiling of something
linear in n, sloped at the level itself. Rational level p/d, and the result repeats
modulo d beneath a constant trend: rank(n+d) - rank(n) does not move with n.

Correct the level first and that fails near the floor. An early version of this probe
claimed otherwise and self_check threw it out at n = 20, alpha = 1/20. The reason
is worth stating because it sets the threshold. The corrected level is L(n+1)/n =
L + L/n, so for `linear` the virtual index is

    1 + q(n-1) = 1 + Ln - L/n,

an affine function perturbed by a term that decays. The floor of that agrees with
the floor of the unperturbed 1 + Ln unless the fractional part of Ln is smaller than
L/n. For a rational L = p/d the fractional part of Ln takes only d values, the
smallest non-zero one being 1/d, so once

    L/n < 1/d,   i.e.   n > L*d,

the nudge can only bite where that fractional part sits at zero, and those sizes are
themselves periodic. Every rule then repeats above a threshold of order d that can be
computed, corrected level or not, and is simply ragged underneath.

That gives a complete finite check rather than a sample: exhaustively cover the
irregular prefix from the feasibility floor up to the threshold, plus two full
periods beyond it. Block (iii) does exactly that, block (iv) reports the per-rule
trend in the periodic regime, and self_check asserts the periodicity only where the
argument says it holds.

What falls out beyond the certificate
-------------------------------------
Some of the sixteen cannot be told apart at all. Reading an index off the sorted
scores forms no level, so there is nothing for a correction to act on and both
settings return the same thing. Collapsing those is part of the result: it says the audit's taxonomy has
fewer distinguishable behaviours than branches, which is worth knowing before
claiming to have classified a helper. Block (i) reports the classes.

And the blind sizes are quantified. Block (v) ranks every n by how many distinct
observations it produces: a size where many rules collapse is a size where a fixture
cannot discriminate, which is exactly the failure the audit reports anecdotally.
"""

import math
import os
import sys
from fractions import Fraction as F
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_certificate.txt")

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


# ---------------------------------------------------------------------------
# the sixteen candidate rules, in exact rational arithmetic
# ---------------------------------------------------------------------------
RAILS = {"one-sided": lambda a: 1 - a, "two-sided": lambda a: 1 - a / 2}
CORRECTIONS = {"raw": lambda L, n: L,
               "corrected": lambda L, n: min(F(1), L * (n + 1) / n)}
OUT_OF_RANGE = "X"          # delivered rank outside 1..n: the boundary case


def rank_linear(q, n):
    """numpy 'linear': virtual index 1 + q(n-1); the guarantee is its floor."""
    return math.floor(1 + q * (n - 1))


def rank_higher(q, n):
    """numpy 'higher': ceil(q(n-1)) + 1, an actual order statistic."""
    return math.ceil(q * (n - 1)) + 1


def rank_inverted_cdf(q, n):
    """numpy 'inverted_cdf': ceil(qn)."""
    return max(1, math.ceil(q * n))


FAMILIES = {"linear": rank_linear, "higher": rank_higher,
            "inverted_cdf": rank_inverted_cdf}


def rules():
    """Every candidate as (name, callable(n, alpha) -> observed rank or 'X')."""
    out = []
    for rail, Lf in RAILS.items():
        for fam, rf in FAMILIES.items():
            for corr, cf in CORRECTIONS.items():
                def make(Lf=Lf, cf=cf, rf=rf):
                    def f(n, alpha):
                        L = Lf(alpha)
                        q = cf(L, n)
                        r = rf(q, n)
                        return r if 1 <= r <= n else OUT_OF_RANGE
                    return f
                out.append((f"{rail}/{corr}/{fam}", make()))
        # reading the index directly forms no level, so a correction has nothing
        # to act on -- both settings are listed anyway, so the collapse gets
        # reported instead of quietly assumed
        for corr in CORRECTIONS:
            def make(Lf=Lf):
                def f(n, alpha):
                    r = math.ceil((n + 1) * Lf(alpha))
                    return r if 1 <= r <= n else OUT_OF_RANGE
                return f
            out.append((f"{rail}/{corr}/direct", make()))
    return out


RULES = rules()


def feasible_floor(alpha):
    return math.ceil(1 / alpha - 1)


def period_of(alpha):
    """Denominator of the coarsest requested level -- the period in n."""
    return max(RAILS[r](alpha).denominator for r in RAILS)


def periodic_from(alpha):
    """Smallest n above which every rule, corrected included, is periodic.

    The corrected level perturbs the virtual index by L/n, which can only move a
    floor once it exceeds the smallest non-zero fractional part 1/d. So the
    irregular prefix ends by n > L*d; take the ceiling and clear the floor.
    """
    d = period_of(alpha)
    L = max(RAILS[r](alpha) for r in RAILS)
    return max(feasible_floor(alpha) + 1, math.ceil(L * d) + 1)


def signature(rule_fn, sizes, alpha):
    return tuple(rule_fn(n, alpha) for n in sizes)


def classes(alpha, sizes):
    """Group rules by their observation vector over `sizes`."""
    groups = {}
    for name, fn in RULES:
        groups.setdefault(signature(fn, sizes, alpha), []).append(name)
    return groups


# ---------------------------------------------------------------------------
def self_check():
    # (1) the rank formulas agree with numpy on the tie-free score set, which is
    #     the instrument the audit already uses. Checked here so this probe does
    #     not quietly re-derive them differently.
    import numpy as np
    for n in (9, 10, 19, 20, 47, 100):
        for q in (F(9, 10), F(19, 20), F(2, 3), F(7, 11)):
            xs = np.arange(1, n + 1, dtype=float)
            h = float(np.quantile(xs, float(q), method="linear"))
            assert rank_linear(q, n) == math.floor(h + 1e-12), (n, q)
            assert rank_higher(q, n) == round(
                float(np.quantile(xs, float(q), method="higher"))), (n, q)
            assert rank_inverted_cdf(q, n) == round(
                float(np.quantile(xs, float(q), method="inverted_cdf"))), (n, q)

    # (2) the required rank and the feasibility floor, stated at the boundary
    for alpha, first in ((F(1, 10), 9), (F(1, 20), 19), (F(1, 100), 99)):
        assert feasible_floor(alpha) == first, alpha
        assert math.ceil((first + 1) * (1 - alpha)) <= first
        assert math.ceil(first * (1 - alpha)) <= first - 1 + 1

    # (3) Every rule is periodic in n modulo the level's denominator ABOVE the
    #     threshold n > L*d derived in the docstring, and only above it. Asserted
    #     both ways: periodic where the argument says so, and NOT assumed below.
    for alpha in (F(1, 10), F(1, 20), F(2, 7)):
        d = period_of(alpha)
        n0 = periodic_from(alpha)
        for name, fn in RULES:
            trend = None
            for n in range(n0, n0 + 4 * d):
                a, b = fn(n, alpha), fn(n + d, alpha)
                if a == OUT_OF_RANGE or b == OUT_OF_RANGE:
                    continue
                t = b - a
                if trend is None:
                    trend = t
                assert t == trend, (
                    f"{name}: rank(n+{d}) - rank(n) is not constant above the "
                    f"threshold n0={n0} ({t} vs {trend} at n={n}); the "
                    f"periodicity argument fails and a finite check cannot "
                    f"extend to all n")
        # the threshold is not vacuous: at least one corrected rule really is
        # irregular below it, or n0 is overcautious and should be lowered
        if feasible_floor(alpha) + 1 < n0:
            irregular = False
            for name, fn in RULES:
                if "corrected" not in name:
                    continue
                seen = set()
                for n in range(feasible_floor(alpha) + 1, n0):
                    a, b = fn(n, alpha), fn(n + d, alpha)
                    if a != OUT_OF_RANGE and b != OUT_OF_RANGE:
                        seen.add(b - a)
                if len(seen) > 1:
                    irregular = True
                    break
            assert irregular, (
                f"alpha={alpha}: no corrected rule is irregular below n0={n0}, "
                f"so the threshold is overcautious")

    # (4) the direct rules must collapse in pairs -- a correction cannot reach a
    #     level that is never formed. If this fails, the candidate set is wrong.
    for alpha in (F(1, 10), F(2, 7)):
        sizes = tuple(range(feasible_floor(alpha) + 1,
                            feasible_floor(alpha) + 40))
        for rail in RAILS:
            sigs = {corr: signature(dict(RULES)[f"{rail}/{corr}/direct"],
                                    sizes, alpha) for corr in CORRECTIONS}
            assert len(set(sigs.values())) == 1, (rail, sigs)


self_check()


# ---------------------------------------------------------------------------
def minimal_certificate(alpha, pool, target_classes, max_size=6):
    """Smallest subset of `pool` separating every distinguishable class."""
    for k in range(1, max_size + 1):
        for S in combinations(pool, k):
            if len(classes(alpha, S)) == target_classes:
                return list(S)
    return None


def main():
    say("=" * 100)
    say("W13  A MINIMAL CERTIFICATE -- the calibration sizes a suite must probe")
    say("=" * 100)
    say("")
    say(f"  {len(RULES)} candidate rules: rail x correction x family, the audit's")
    say("  own classifier candidate set, in exact rational arithmetic.")
    say("")

    results = []
    for alpha in (F(1, 10), F(1, 20), F(2, 7)):
        d = period_of(alpha)
        lo = feasible_floor(alpha) + 1
        n0 = periodic_from(alpha)
        # the irregular prefix [lo, n0) plus two full periods beyond n0 is a
        # COMPLETE check over all n, by the argument in the docstring and
        # self_check (3) -- not a sample
        wide = list(range(lo, n0 + 2 * d))
        period_pool = wide

        say("-" * 100)
        say(f"alpha = {alpha}   floor {feasible_floor(alpha)}   period {d}   "
            f"periodic from n = {n0}")
        say(f"  complete check: irregular prefix {lo}..{n0 - 1} "
            f"({max(0, n0 - lo)} sizes) + two periods {n0}..{n0 + 2 * d - 1}")
        say("-" * 100)

        # ---- (i) distinguishable classes ------------------------------
        full = classes(alpha, tuple(wide))
        say(f"(i) OBSERVATIONALLY DISTINCT CLASSES: {len(full)} of {len(RULES)} rules")
        for sig, names in sorted(full.items(), key=lambda kv: -len(kv[1])):
            if len(names) > 1:
                say(f"    collapsed ({len(names)}): " + "; ".join(names))
        singles = [n for s, n in full.items() if len(n) == 1]
        say(f"    {len(singles)} rules are uniquely identifiable; "
            f"{len(RULES) - len(singles)} share an observation with another")
        say("")

        # ---- (ii) the certificate --------------------------------------
        S = minimal_certificate(alpha, period_pool, len(full))
        say("(ii) MINIMAL CERTIFICATE")
        if S is None:
            say(f"     none of size <= 6 separates all {len(full)} classes")
        else:
            say(f"     |S| = {len(S)}   S = {S}")
            say(f"     Observing the delivered rank at these {len(S)} calibration")
            say(f"     sizes distinguishes all {len(full)} distinguishable classes.")
        say("")

        # ---- (iii) the extension to all n ------------------------------
        if S is not None:
            got = classes(alpha, tuple(S))
            ok_wide = len(got) == len(full)
            # and the PARTITIONS must match, not merely the counts
            same = ({frozenset(v) for v in got.values()}
                    == {frozenset(v) for v in full.values()})
            say("(iii) COMPLETENESS")
            say(f"      classes separated by S: {len(got)}; by the full checked "
                f"range: {len(full)}   -> {'YES' if ok_wide else 'NO'}")
            say(f"      identical partitions, not merely equal counts: "
                f"{'YES' if same else 'NO'}")
            assert ok_wide and same, "certificate does not reproduce the partition"
            say(f"      The range checked runs over the ragged prefix plus two whole")
            say(f"      periods, and rank(n+{d}) - rank(n) holds steady per rule above")
            say(f"      n = {n0} (asserted in self_check), so agreement there is")
            say("      agreement at every larger n. This is a proof, not a sample.")
        say("")

        # ---- (iv) the trend per rule -----------------------------------
        say(f"(iv) rank(n + {d}) - rank(n) in the periodic regime, per rule:")
        trends = {}
        for name, fn in RULES:
            for n in range(n0, n0 + d):
                a, b = fn(n, alpha), fn(n + d, alpha)
                if a != OUT_OF_RANGE and b != OUT_OF_RANGE:
                    trends.setdefault(b - a, []).append(name)
                    break
        for t, names in sorted(trends.items()):
            say(f"    +{t}: {len(names)} rules")
        say("")

        # ---- (v) the blind sizes ---------------------------------------
        say("(v) DISCRIMINATING POWER of a single calibration size. A fixture at a")
        say("    low-power n cannot separate the rules it collapses -- which is the")
        say("    failure the audit reports for one library's own test values.")
        per_n = sorted(((len(classes(alpha, (n,))), n) for n in wide),
                       key=lambda kv: (kv[0], kv[1]))
        worst = [n for c, n in per_n if c == per_n[0][0]]
        best = [n for c, n in per_n if c == per_n[-1][0]]
        say(f"    worst: {per_n[0][0]} distinct observations, at n = "
            f"{worst[:10]}{' ...' if len(worst) > 10 else ''}")
        say(f"    best:  {per_n[-1][0]} distinct observations, at n = "
            f"{best[:10]}{' ...' if len(best) > 10 else ''}")
        say(f"    no single n separates all {len(full)} classes"
            if per_n[-1][0] < len(full) else
            f"    a single n suffices: {best[0]}")
        results.append({"alpha": str(alpha), "floor": feasible_floor(alpha),
                        "period": d, "n0": n0, "rules": len(RULES),
                        "classes": len(full), "cert": S,
                        "worst_power": per_n[0][0], "best_power": per_n[-1][0],
                        "checked": len(wide)})
        say("")

    # ---------------- summary table ------------------------------------
    say("=" * 100)
    say("SUMMARY")
    say("=" * 100)
    say(f"{'alpha':>8}{'floor':>7}{'period':>8}{'periodic from':>14}"
        f"{'n checked':>11}{'rules':>7}{'classes':>9}{'|S|':>5}"
        f"{'worst n':>9}{'best n':>8}   certificate")
    for r in results:
        say(f"{r['alpha']:>8}{r['floor']:>7}{r['period']:>8}{r['n0']:>14}"
            f"{r['checked']:>11}{r['rules']:>7}{r['classes']:>9}"
            f"{(len(r['cert']) if r['cert'] else 0):>5}"
            f"{r['worst_power']:>9}{r['best_power']:>8}   {r['cert']}")
    say("")
    say("  Read `worst n power` against the claim that pinning a test to one of")
    say("  these sizes tells you nothing: there the sixteen candidates yield only")
    say("  that many distinct observations between them, so a suite fixed at such a")
    say("  size misses the difference by construction rather than by luck.")
    say("")
    say("  Scope: sixteen candidates, told apart. Where a helper matches none, the")
    say("  finding is a gap in this list -- a fact about the instrument and not a")
    say("  verdict on the library, which is the scope the classifier carries too.")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
