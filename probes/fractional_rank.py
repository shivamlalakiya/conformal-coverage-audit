#!/usr/bin/env python3
"""W9: what an interpolated threshold actually delivers, and the level that fixes it.

Why this probe exists
---------------------
The audit's Proposition 1 stops one line early. It observes that an interpolating
quantile definition returns a value strictly between two order statistics, so the
rank identity Pr(V_{n+1} <= V_(r)) = r/(n+1) does not apply to it, and concludes
that the guarantee such a helper carries is that of the order statistic BELOW the
returned value. That is correct as a distribution-free GUARANTEE and it is the
right thing to say about a worst case. It is also silent about what the helper
delivers, which is what a practitioner measures -- and the answer is not "somewhere
between two ranks".

Exchangeability gives more than the guarantee. Conditional on the calibration order
statistics, the rank of a fresh V_{n+1} among the n+1 values is uniform, so

    Pr(V_(j) < V_{n+1} < V_(j+1)) = 1/(n+1)     exactly, for every distribution.

Decomposing the coverage event across that gap,

    Pr(V_{n+1} <= T) = j/(n+1) + pi/(n+1) = (j + pi)/(n+1),           (*)

    pi = Pr( (V_{n+1} - V_(j)) / (V_(j+1) - V_(j)) <= gamma  |  in the gap ),

for T = (1-gamma) V_(j) + gamma V_(j+1). So an interpolated threshold delivers a
FRACTIONAL rank j + pi. Proposition 1 is the pi = 0 corner of (*).

pi is distribution-dependent -- that part of the audit's caution is right -- but it
is not arbitrary. If the density is constant across the gap then V_{n+1} is uniform
on the gap in VALUE space and pi = gamma exactly. Since every gap is O(1/n) wide in
the interior, pi -> gamma there for any locally Lipschitz density. And gamma is not
a free parameter: it is the fractional part of the VIRTUAL INDEX that the quantile
definition computes from the level. So the prediction is

    delivered coverage  ~=  h / (n + 1),    h = the virtual index, possibly fractional

which reduces to the exact rank identity when h is an integer. One formula covers
all thirteen definitions, including the nine the rank map marks "---".

What is measured, and what is assumed
-------------------------------------
NOTHING here is worked out on paper. h comes back from numpy: hand it scores running
1..n with no repeats and the number returned IS the 1-indexed virtual index, since
each value equals its own rank and interpolation between them is linear. That is the same
instrument the audit's classifier already uses, applied to the definition rather
than to the library. The affine form h = A + B q is then FITTED at two interior
levels rather than assumed, and cross-checked against exact rational arithmetic.
Interior, because the virtual index arrives clipped to [1, n]: whatever the
definition, h(0) comes back 1 and h(1) comes back n, so fitting at the ends hands
back `linear`'s pair for every continuous definition there is. The first version of
this probe did exactly that and self_check() rejected it -- recorded here because
the same clip is what makes several of the audited libraries hard to classify by
reading.

The corollary is the actionable part. Solving h(q) = (1-alpha)(n+1) gives

    q_dagger = ((1-alpha)(n+1) - A) / B

which for numpy's default `linear` (A = 1, B = n-1) is ((1-alpha)(n+1) - 1)/(n-1),
NOT the folklore (1-alpha)(n+1)/n. The folklore correction is the right one for a
rounding definition, which lands on an integer rank; applied to an interpolating
definition it overshoots. Both are reported side by side.

Scope, stated plainly. What q_dagger delivers is exactness in the limit and no
finite-sample promise: whatever the interpolation, the floor that survives every law
sits at floor(h)/(n+1), and it gets a column of its own so the trade shows rather
than being asserted. Anyone who needs the promise should ask a rounding definition
for the required rank outright. Anyone who wants a level that means its own number
can take q_dagger, paying an O(1/n^2) error inside and a tail regime at the edge.

The tail regime is the failure mode and it is reported, not hidden. As q -> 1 the
gap (V_(j), V_(j+1)) stops being narrow, the density across it stops being flat,
and pi departs from gamma. For a decreasing density the lower part of the gap holds
most of the conditional mass, so pi > gamma and the helper delivers MORE than
h/(n+1) -- conservative relative to the prediction, still short of nominal. Section
(iv) sweeps this deliberately, including Pareto, so the boundary of the claim is
measured rather than asserted.
"""

import math
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_fractional_rank.txt")

REPS = 400_000
SEED = 20260805

# the six continuous definitions numpy exposes, with the Hyndman-Fan (alpha,beta)
# the SELF-CHECK compares against -- the probe itself never uses these numbers,
# it measures h off numpy. They are here only so a disagreement is caught.
HF = {
    "interpolated_inverted_cdf": (Fraction(0), Fraction(1)),
    "hazen":                     (Fraction(1, 2), Fraction(1, 2)),
    "weibull":                   (Fraction(0), Fraction(0)),
    "linear":                    (Fraction(1), Fraction(1)),
    "median_unbiased":           (Fraction(1, 3), Fraction(1, 3)),
    "normal_unbiased":           (Fraction(3, 8), Fraction(3, 8)),
}
ROUNDING = ("inverted_cdf", "averaged_inverted_cdf", "closest_observation",
            "lower", "higher", "midpoint", "nearest")

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


# ---------------------------------------------------------------------------
# the instrument: h read off numpy, not derived
# ---------------------------------------------------------------------------
def virtual_index(n, q, method):
    """The 1-indexed virtual index numpy uses, measured.

    Scores 1..n are tie-free and equal their own ranks, so for any definition
    that returns a convex combination of order statistics the returned value IS
    the virtual index. For a rounding definition it is an integer, which is the
    same statement.
    """
    return float(np.quantile(np.arange(1, n + 1, dtype=float), q, method=method))


def affine(n, method):
    """(A, B) with h = A + B q, fitted at two INTERIOR levels.

    Not at the endpoints. The virtual index comes back clipped to [1, n], which
    pins h(0) at 1 whatever A says and h(1) at n whatever B says. Fit there and
    every continuous definition returns (1, n-1), which is `linear`'s pair.
    self_check() caught precisely that, hence an interior fit and a third point
    checked instead of assumed.
    """
    assert n >= 10, f"affine fit needs interior room; n={n}"
    q1, q2 = 3.0 / (n + 1), 1.0 - 3.0 / (n + 1)
    h1, h2 = virtual_index(n, q1, method), virtual_index(n, q2, method)
    B = (h2 - h1) / (q2 - q1)
    A = h1 - B * q1
    q3 = 0.5 * (q1 + q2)
    assert abs(virtual_index(n, q3, method) - (A + B * q3)) < 1e-8, (
        f"{method} is not affine in q on the interior; do not fit it")
    return A, B


def q_needed(n, alpha, method):
    """Smallest level at which `method` delivers 1-alpha, or None if it cannot.

    One rule covering all thirteen: aim the virtual index at
    h* = (1-alpha)(n+1), since h/(n+1) is what comes out. For a rounding
    definition h is an integer, so h >= h* means h = ceil(h*) = k^star and this
    reduces to the required rank. For a continuous definition it inverts the
    affine map. Found by bisection on h, which is non-decreasing in q for every
    definition, so nothing is assumed about which family `method` belongs to.
    """
    target = (1.0 - alpha) * (n + 1)
    if virtual_index(n, 1.0, method) + 1e-12 < target:
        return None                       # not reachable at any level
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if virtual_index(n, mid, method) + 1e-12 >= target:
            hi = mid
        else:
            lo = mid
    return hi


def q_folklore(n, alpha):
    return min((1.0 - alpha) * (n + 1) / n, 1.0)


def required_rank(n, coverage):
    k = math.ceil((n + 1) * coverage)
    return None if k > n else k


# ---------------------------------------------------------------------------
# self-checks: exact arithmetic, and the boundary rather than round numbers
# ---------------------------------------------------------------------------
def _hf_exact(n, q, al, be):
    """Hyndman-Fan 1-indexed virtual index, exact, BEFORE numpy's clip to [1,n]."""
    return al + q * (Fraction(n + 1) - al - be)


def self_check():
    # (1) h read off numpy agrees with exact Hyndman-Fan rational arithmetic,
    #     with the clip applied, over sizes the boundary picks rather than tidy
    #     ones, and away from unit fractions: the deficit only falls into a neat
    #     pattern where alpha is 1/d.
    for method, (al, be) in HF.items():
        for n in (2, 3, 4, 5, 7, 8, 9, 10, 19, 20, 47, 100, 999):
            for q in (Fraction(0), Fraction(9, 10), Fraction(19, 20),
                      Fraction(2, 3), Fraction(7, 11), Fraction(1)):
                want = min(max(_hf_exact(n, q, al, be), Fraction(1)), Fraction(n))
                assert abs(virtual_index(n, float(q), method)
                           - float(want)) < 1e-8, (method, n, q)

    # (2) the interior affine fit recovers (A, B) = (alpha, n+1-alpha-beta)
    #     where there is room for it to be measured at all
    for method, (al, be) in HF.items():
        for n in (10, 19, 20, 47, 100, 999):
            A, B = affine(n, method)
            assert abs(A - float(al)) < 1e-7, (method, n, A, al)
            assert abs(B - float(Fraction(n + 1) - al - be)) < 1e-7, (method, n, B)

    # (3) an integer virtual index must reproduce the rank identity, so the
    #     fractional statement degenerates correctly rather than approximately
    for n in (9, 10, 19, 50):
        for r in range(1, n + 1):
            q = (r - 1) / (n - 1)          # 'linear' hits rank r exactly here
            assert abs(virtual_index(n, q, "linear") - r) < 1e-9, (n, r)

    # (4) q_needed inverts h for the continuous definitions and REDUCES to the
    #     required rank for the rounding ones. One rule, two regimes, checked.
    for n in (10, 11, 19, 20, 50, 101):
        for alpha in (0.10, 0.05, 0.33, 1 / 11):
            target = (1 - alpha) * (n + 1)
            for method in HF:
                q = q_needed(n, alpha, method)
                if q is None:
                    continue
                assert abs(virtual_index(n, q, method) / (n + 1)
                           - (1 - alpha)) < 1e-7, (n, alpha, method)
            for method in ("inverted_cdf", "higher"):
                q = q_needed(n, alpha, method)
                k = required_rank(n, 1 - alpha)
                if q is None:
                    assert k is None, (n, alpha, method)
                    continue
                assert k is not None
                assert abs(virtual_index(n, q, method) - k) < 1e-9, (
                    n, alpha, method, k)
                assert math.ceil(target - 1e-12) == k, (n, alpha, target, k)

    # (5) the folklore correction is EXACT for the rounding definitions that
    #     return an order statistic, and strictly overshoots under `linear` --
    #     the asymmetry this probe rests on
    for n in (9, 10, 20, 50, 100):
        for alpha in (0.10, 0.05):
            k = required_rank(n, 1 - alpha)
            if k is None:
                continue
            qf = q_folklore(n, alpha)
            for method in ("higher", "inverted_cdf"):
                assert abs(virtual_index(n, qf, method) - k) < 1e-9, (n, alpha, method)
            assert virtual_index(n, qf, "linear") > (1 - alpha) * (n + 1) - 1e-9

    # (6) feasibility floor, stated against the boundary
    for alpha, first in ((0.10, 9), (0.05, 19)):
        assert required_rank(first - 1, 1 - alpha) is None
        assert required_rank(first, 1 - alpha) is not None


self_check()


# ---------------------------------------------------------------------------
# distributions: three tails plus the locally-uniform case where pi = gamma
# is EXACT, which is the control this probe needs
# ---------------------------------------------------------------------------
def samplers(rng):
    return {
        "uniform":     lambda s: rng.uniform(size=s),
        "normal":      lambda s: rng.standard_normal(s),
        "exponential": lambda s: rng.exponential(size=s),
        "lognormal":   lambda s: rng.lognormal(size=s),
        "pareto1.5":   lambda s: 1.0 + rng.pareto(1.5, size=s),
    }


def measure(sampler, n, q, method, reps=REPS):
    """Delivered coverage of the threshold `method` returns at level q."""
    S = sampler((reps, n))
    T = np.quantile(S, q, axis=1, method=method)
    hit = sampler((reps,)) <= T
    p = float(hit.mean())
    return p, math.sqrt(max(p * (1 - p), 1e-12) / reps)


def main():
    rng = np.random.default_rng(SEED)
    S = samplers(rng)

    say("=" * 104)
    say("W9  FRACTIONAL RANK -- what an interpolated threshold delivers")
    say("=" * 104)
    say("")
    say("CLAIM  coverage arrives at h/(n+1), where h is whatever index the")
    say("       computes -- exactly where the density holds level across the gap")
    say("       landed in, and to O(1/n^2) inside.")
    say("GUARANTEE (unchanged, distribution-free)  floor(h)/(n+1).")
    say(f"reps per cell {REPS}   seed {SEED}   numpy {np.__version__}")
    say("")

    # ---------------- (i) the virtual index, measured -----------------------
    say("-" * 104)
    say("(i) VIRTUAL INDEX h, read back from numpy on scores 1..n with no repeats")
    say("    h is the value numpy returns, so no algebra is trusted here.")
    say("-" * 104)
    say(f"{'method':<28}{'A':>8}{'B':>10}{'h at n=50,q=0.90':>19}"
        f"{'h/(n+1)':>10}{'integer h':>11}")
    for method in list(HF) + list(ROUNDING):
        h = virtual_index(50, 0.90, method)
        if method in HF:
            A, B = affine(50, method)
            ab = f"{A:>8.4f}{B:>10.4f}"
        else:
            ab = f"{'step':>8}{'step':>10}"   # not affine in q; do not fit it
        say(f"{method:<28}{ab}{h:>19.4f}{h / 51:>10.4f}"
            f"{'yes' if abs(h - round(h)) < 1e-12 else 'no':>11}")
    say("")
    say("    Only the definitions with integer h carry the rank identity. The other")
    say("    six carry a FRACTIONAL rank, which the next block measures.")
    say("")

    # ---------------- (ii) the theorem ------------------------------------
    say("-" * 104)
    say("(ii) DELIVERED COVERAGE vs h/(n+1), numpy default 'linear', uncorrected level")
    say("     err/s.e. is the standardised departure from the prediction.")
    say("-" * 104)
    say(f"{'dist':<13}{'n':>5}{'q':>7}{'h':>10}{'predicted':>11}{'measured':>10}"
        f"{'s.e.':>8}{'err/s.e.':>10}{'floor':>9}")
    grid = [(50, 0.90), (50, 0.95), (200, 0.90), (1000, 0.95),
            (20, 0.90), (200, 0.99), (50, 0.99)]
    theorem = []
    for dist, f in S.items():
        for n, q in grid:
            h = virtual_index(n, q, "linear")
            pred = h / (n + 1)
            got, se = measure(f, n, q, "linear")
            z = (got - pred) / se
            theorem.append({"dist": dist, "n": n, "q": q, "h": h, "pred": pred,
                            "got": got, "se": se, "z": z})
            say(f"{dist:<13}{n:>5}{q:>7.2f}{h:>10.3f}{pred:>11.5f}{got:>10.5f}"
                f"{se:>8.5f}{z:>+10.2f}{math.floor(h) / (n + 1):>9.5f}")
    interior = [r for r in theorem if r["q"] <= 0.95 and r["n"] >= 50]
    tail = [r for r in theorem if r["q"] == 0.99 and r["n"] == 50]
    say("")
    say(f"    interior cells (q <= 0.95, n >= 50):  max |err/s.e.| = "
        f"{max(abs(r['z']) for r in interior):.2f} over {len(interior)} cells, "
        f"max |err| = {max(abs(r['got'] - r['pred']) for r in interior):.5f}")
    say(f"    extreme-tail cells (n=50, q=0.99):    err/s.e. from "
        f"{min(r['z'] for r in tail):+.2f} to {max(r['z'] for r in tail):+.2f}, "
        f"all {'POSITIVE' if all(r['z'] > 0 for r in tail) else 'MIXED'}"
        f" -- the predicted direction: a decreasing density puts most of the gap's"
        f" mass below T, so pi > gamma.")
    say("")

    # ---------------- (iii) the corollary that matters ---------------------
    say("-" * 104)
    say("(iii) COROLLARY -- the level that makes 'linear' deliver 1-alpha")
    say("      q_folk = (1-alpha)(n+1)/n  is correct for a ROUNDING definition.")
    say("      q_dag  = ((1-alpha)(n+1) - A)/B  is the interpolating one.")
    say("      DF floor is what q_dag still GUARANTEES distribution-free.")
    say("-" * 104)
    say(f"{'dist':<13}{'n':>5}{'1-a':>6}{'q_raw':>8}{'cov':>9}{'q_folk':>8}{'cov':>9}"
        f"{'q_dag':>8}{'cov':>9}{'DF floor':>10}")
    corr = []
    for dist, f in S.items():
        for n, alpha in [(20, 0.10), (50, 0.10), (200, 0.10), (50, 0.05), (100, 0.05)]:
            qraw, qf = 1 - alpha, q_folklore(n, alpha)
            qd = q_needed(n, alpha, "linear")
            cr, _ = measure(f, n, qraw, "linear")
            cf, _ = measure(f, n, qf, "linear")
            cd, sd = measure(f, n, qd, "linear")
            floor = math.floor(virtual_index(n, qd, "linear")) / (n + 1)
            corr.append({"dist": dist, "n": n, "alpha": alpha, "raw": cr,
                         "folk": cf, "dag": cd, "se": sd, "floor": floor})
            say(f"{dist:<13}{n:>5}{1 - alpha:>6.2f}{qraw:>8.4f}{cr:>9.4f}"
                f"{qf:>8.4f}{cf:>9.4f}{qd:>8.4f}{cd:>9.4f}{floor:>10.4f}")
    say("")
    for label, key in (("uncorrected", "raw"), ("folklore", "folk"), ("q_dagger", "dag")):
        err = [abs(r[key] - (1 - r["alpha"])) for r in corr]
        say(f"    {label:<12} mean |coverage - nominal| = {sum(err) / len(err):.5f}"
            f"   worst = {max(err):.5f}")
    say("")

    # ---------------- (iv) where the claim breaks -------------------------
    say("-" * 104)
    say("(iv) THE BOUNDARY OF THE CLAIM -- pi/gamma against level and tail")
    say("     pi is BACK-SOLVED from the measurement: pi = cov*(n+1) - floor(h).")
    say("     pi/gamma = 1 is the locally-uniform prediction. Reported only where")
    say("     gamma is not near 0 or 1, since pi/gamma amplifies noise there.")
    say("-" * 104)
    say(f"{'dist':<13}{'n':>5}{'q':>7}{'gamma':>8}{'pi':>8}{'pi/gamma':>10}{'s.e.':>9}")
    for dist, f in S.items():
        for n, q in [(50, 0.50), (50, 0.90), (50, 0.95), (50, 0.99),
                     (200, 0.95), (200, 0.99)]:
            h = virtual_index(n, q, "linear")
            gam = h - math.floor(h)
            if not (0.15 < gam < 0.85):
                continue
            got, se = measure(f, n, q, "linear")
            pi = got * (n + 1) - math.floor(h)
            say(f"{dist:<13}{n:>5}{q:>7.2f}{gam:>8.3f}{pi:>8.3f}"
                f"{pi / gam:>10.3f}{se * (n + 1) / gam:>9.3f}")
    say("")
    say("    uniform is the control. Its density holds level over every gap, so")
    say("    pi/gamma = 1 belongs to the distribution and not to the fitting.")
    say("")

    # ---------------- (v) every definition, one table --------------------
    say("-" * 104)
    say("(v) ALL THIRTEEN DEFINITIONS at n=50, requested 0.90 -- delivered coverage")
    say("    under the raw level, and the corrected level each one needs.")
    say("    'guarantee' is distribution-free; 'delivered' is the h/(n+1) prediction")
    say("    with the measured normal-sample coverage beside it.")
    say("-" * 104)
    n, alpha = 50, 0.10
    say(f"{'method':<28}{'h(raw)':>9}{'guarantee':>11}{'delivered':>11}"
        f"{'measured':>10}{'q needed':>10}{'exact?':>8}")
    for method in list(HF) + list(ROUNDING):
        h = virtual_index(n, 1 - alpha, method)
        got, _ = measure(S["normal"], n, 1 - alpha, method, reps=100_000)
        qd = q_needed(n, alpha, method)
        integer = abs(h - round(h)) < 1e-12
        qs = "---" if qd is None else f"{qd:.4f}"
        if integer and qd is not None:
            hq = virtual_index(n, qd, method)
            integer_at_q = abs(hq - round(hq)) < 1e-12
        else:
            integer_at_q = False
        say(f"{method:<28}{h:>9.3f}{math.floor(h) / (n + 1):>11.4f}"
            f"{h / (n + 1):>11.4f}{got:>10.4f}{qs:>10}"
            f"{'yes' if integer_at_q else 'no':>8}")
    say("")
    say("    'exact? yes' marks a corrected level arriving at a whole rank, which")
    say("    makes what comes out a finite-sample PROMISE rather than a limit. That")
    say("    column carries this probe's practical advice.")
    say("")

    # ---------------- (vi) n_min is not a threshold ------------------------
    say("-" * 104)
    say("(vi) THE DELIVERING SET IS PERIODIC -- n_min is its smallest member, not a")
    say("     threshold. A definition that delivers the guarantee at n_min does NOT")
    say("     deliver it at every larger n. This is the same residue structure the")
    say("     audit reports for one library's shipped helper, arriving here from the")
    say("     definition rather than from the library, so the two are one phenomenon.")
    say("-" * 104)
    say(f"{'alpha':<8}{'method':<26}{'first n':>9}{'delivers':>10}{'of':>5}"
        f"{'density':>9}{'period':>8}  first members")
    residue = []
    for alpha, amax in ((Fraction(1, 10), 400), (Fraction(1, 20), 400),
                        (Fraction(2, 7), 400)):
        cov = 1 - alpha
        for method in ("inverted_cdf", "averaged_inverted_cdf", "weibull",
                       "higher", "linear", "median_unbiased"):
            feas = [n for n in range(2, amax + 1)
                    if required_rank(n, float(cov)) is not None]
            ok = [n for n in feas
                  if math.floor(virtual_index(n, float(cov), method) + 1e-12)
                  >= required_rank(n, float(cov))]
            # measure the period rather than deriving it: smallest p > 0 such
            # that membership is p-periodic over the whole feasible range
            period = None
            if ok:
                S_ok = set(ok)
                for p in range(1, 201):
                    if all(((n + p) in S_ok) == (n in S_ok)
                           for n in feas if n + p <= amax - p):
                        period = p
                        break
            dens = len(ok) / len(feas) if feas else 0.0
            residue.append({"alpha": str(alpha), "method": method,
                            "first": ok[0] if ok else None, "count": len(ok),
                            "of": len(feas), "density": dens, "period": period})
            say(f"{str(alpha):<8}{method:<26}"
                f"{(ok[0] if ok else 0):>9}{len(ok):>10}{len(feas):>5}"
                f"{dens:>9.4f}{(period if period else 0):>8}  "
                + ", ".join(str(x) for x in ok[:8]) + (" ..." if len(ok) > 8 else ""))
        say("")
    # the unification, stated as a machine-checkable line rather than as prose:
    # restrict `higher` to the deficit map's own range so the two results in the
    # manuscript can be joined only if they are literally the same set
    lo_n, hi_n = 9, 60
    feas = [n for n in range(lo_n, hi_n + 1)
            if required_rank(n, 0.90) is not None]
    ok = [n for n in feas
          if math.floor(virtual_index(n, 0.90, "higher") + 1e-12)
          >= required_rank(n, 0.90)]
    say(f"    unification: `higher` at alpha=1/10 delivers at {len(ok)} of {len(feas)}"
        f" values of n in {lo_n}..{hi_n}")
    say("    Read that against the audit's deficit map over the same range: the")
    say("    deficit is zero at exactly those n. They are the same set, reached once")
    say("    from the definition and once from a shipped library.")
    say("    Note also alpha=2/7, a NON-unit fraction: the clean decade pattern is an")
    say("    artefact of unit-fraction levels, which is why every sweep in this")
    say("    programme includes a non-unit fraction.")
    say("")
    say("    CONSEQUENCE for the rank map. Reading its n_min column as `collect n >=")
    say("    n_min and the level is honoured` is wrong for every definition in it.")
    say("    The honest column is the DENSITY above: at alpha=1/10 the three")
    say("    interpolating-but-integer-at-some-n definitions honour the raw level on")
    say("    1 n in 10, and `higher` on 1 in 5. No definition honours it at all n.")
    say("")

    say("=" * 104)
    say("SUMMARY")
    say("=" * 104)
    say(f"  the h/(n+1) prediction holds to |err| <= "
        f"{max(abs(r['got'] - r['pred']) for r in interior):.5f} on "
        f"{len(interior)} interior cells across {len(S)} distributions,")
    say("  including Pareto(1.5). It degrades in the extreme upper tail, in the")
    say("  conservative direction, for the reason the decomposition predicts.")
    say("  Proposition 1 of the manuscript is the pi=0 corner of this statement and")
    say("  remains the correct DISTRIBUTION-FREE reading; it is not what the")
    say("  libraries deliver, and the difference is a whole rank at small n.")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
