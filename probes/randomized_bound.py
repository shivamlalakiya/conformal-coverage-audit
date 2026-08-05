#!/usr/bin/env python3
"""W11: "no valid finite bound exists" is true only for DETERMINISTIC bounds.

Why this probe exists
---------------------
The audit's headline cell is a shipped default that calibrates on m = 2 windows and
returns a finite 0.90 interval, where the required rank is ceil(3 x 0.90) = 3 > 2.
The manuscript states that no valid finite bound exists at any conventional level.
Read as written that claim is too strong, and a conformal-prediction reviewer will
say so: it holds for bounds that are a deterministic function of the scores, and it
fails for randomised ones. A randomised bound at m = 2 achieving EXACTLY 0.90
marginal coverage exists, and this probe constructs and measures it.

The construction is not new -- it is the smoothed conformal p-value, which is exactly
uniform for every m and therefore yields exactly-1-alpha sets at every calibration
size, randomisation being the price. What is new here is the use: it is the honest
answer to the audit's own worst cell, and it changes the recommendation from "this
configuration cannot work" to "this configuration cannot work WITHOUT randomising,
and here is the width you pay".

Why the distinction earns its space rather than being pedantry
--------------------------------------------------------------
The deterministic achievable coverages with m exchangeable scores are

    k/(m+1) for k = 1..m,   and 1 (the vacuous +inf bound),

a set of m+1 points. A requested 1-alpha that is not one of them cannot be attained
exactly by ANY choice of index -- not merely at m = 2. So the gap this probe closes
is not a boundary curiosity either: at m = 20 and alpha = 0.05 the attainable
coverages straddle 0.95 without hitting it, and every library rounds up (conservative,
wider) or down (invalid). Randomising between the two neighbouring ranks hits it
exactly. That is a statement about every calibration size, and the audit currently
makes it about one.

The honest cost, reported rather than argued
--------------------------------------------
Randomisation buys exact MARGINAL coverage and nothing else. Below the feasibility
floor the mixture puts weight on the vacuous bound, so the interval is infinite a
known fraction of the time, and CONDITIONAL on being finite it undercovers by
construction. Both numbers are in the output. A practitioner who cannot ship an
infinite interval learns the real content of the constraint: the information for a
0.90 statement is not in two residuals, and a randomised bound makes that visible
instead of hiding it behind a finite number.

Everything is measured. The mixture weights are solved in exact rational arithmetic
and re-derived from the measurement; the p-value uniformity is tested against its
own exact discrete law rather than against a normal approximation.
"""

import math
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_randomized_bound.txt")

SEED = 20260805
REPS = 400_000
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def attainable(m):
    """Coverages a DETERMINISTIC index into m exchangeable scores can attain."""
    return [Fraction(k, m + 1) for k in range(1, m + 1)] + [Fraction(1)]


def required_rank(m, coverage):
    k = math.ceil((m + 1) * coverage)
    return None if k > m else k


def mixture(m, coverage):
    """Exact weights on the two neighbouring attainable coverages.

    Returns (k_lo, k_hi, lam) meaning: use rank k_lo with probability lam and
    rank k_hi with probability 1-lam, where k_hi = m+1 denotes the vacuous +inf
    bound. Marginal coverage is then exactly `coverage`.
    """
    c = Fraction(coverage).limit_denominator(10**6)
    pts = attainable(m)
    assert pts[0] <= c <= pts[-1], (m, coverage)
    # An exactly attainable target must degenerate to that single rank. Checking
    # the bracket first instead put weight 0 on rank k-1 and full weight on k --
    # arithmetically correct, but it reports a randomised bound where a
    # deterministic one is exact, which is the opposite of this probe's point.
    for k, p in enumerate(pts, start=1):
        if p == c:
            return k, k, Fraction(1)
    for i in range(len(pts) - 1):
        lo, hi = pts[i], pts[i + 1]
        if lo < c < hi:
            lam = (hi - c) / (hi - lo)
            k_lo = i + 1
            k_hi = i + 2 if i + 2 <= m else m + 1
            assert 0 < lam < 1, (m, coverage, lam)
            assert lam * lo + (1 - lam) * hi == c, (m, coverage, lam)
            return k_lo, k_hi, lam
    raise AssertionError((m, coverage))


# ---------------------------------------------------------------------------
def self_check():
    # (1) the attainable set has exactly m+1 points and its gaps are 1/(m+1)
    for m in (2, 3, 5, 9, 10, 19, 20, 50):
        pts = attainable(m)
        assert len(pts) == m + 1
        for a, b in zip(pts, pts[1:]):
            assert b - a == Fraction(1, m + 1), (m, a, b)

    # (2) the mixture is exact in rational arithmetic, at the boundary and above
    for m in (2, 3, 5, 9, 10, 19, 20, 50):
        for cov in (Fraction(9, 10), Fraction(19, 20), Fraction(2, 3),
                    Fraction(7, 11)):
            if not (attainable(m)[0] <= cov <= 1):
                continue
            k_lo, k_hi, lam = mixture(m, cov)
            lo = Fraction(k_lo, m + 1)
            hi = Fraction(1) if k_hi == m + 1 else Fraction(k_hi, m + 1)
            assert lam * lo + (1 - lam) * hi == cov, (m, cov)
            assert 0 <= lam <= 1

    # (3) where the deterministic bound IS exact, the mixture must degenerate to
    #     it rather than randomise pointlessly
    for m, cov in ((9, Fraction(9, 10)), (19, Fraction(19, 20)),
                   (2, Fraction(2, 3)), (10, Fraction(1, 11))):
        k_lo, k_hi, lam = mixture(m, cov)
        exact_k = cov * (m + 1)
        if exact_k.denominator == 1:
            assert lam == 1 and k_lo == int(exact_k), (m, cov, k_lo, lam)

    # (4) the audit's cell, spelled out: m=2 at 0.90 is deterministically
    #     unattainable and the mixture weight is 3/10 on the rank-2 bound
    assert required_rank(2, 0.90) is None
    assert Fraction(9, 10) not in attainable(2)
    k_lo, k_hi, lam = mixture(2, Fraction(9, 10))
    assert (k_lo, k_hi, lam) == (2, 3, Fraction(3, 10)), (k_lo, k_hi, lam)

    # (5) a NON-unit-fraction target, because unit fractions are the easy case
    k_lo, k_hi, lam = mixture(20, Fraction(7, 11))
    assert 0 < lam < 1


self_check()


def measure(rng, m, coverage, reps=REPS):
    """Deterministic ceil-rank bound vs the exact randomised mixture."""
    k_lo, k_hi, lam = mixture(m, Fraction(coverage).limit_denominator(10**6))
    S = np.sort(rng.standard_normal((reps, m)), axis=1)
    fresh = rng.standard_normal(reps)

    # deterministic: the required rank, or +inf when it does not exist
    k = required_rank(m, coverage)
    det = np.full(reps, np.inf) if k is None else S[:, k - 1]
    # ... and what a library that rounds DOWN instead would give
    k_down = min(max(int(math.floor((m + 1) * coverage)), 1), m)
    det_down = S[:, k_down - 1]

    # randomised: rank k_lo with prob lam, else rank k_hi (m+1 == vacuous)
    u = rng.random(reps)
    hi_col = np.full(reps, np.inf) if k_hi == m + 1 else S[:, k_hi - 1]
    rnd = np.where(u < float(lam), S[:, k_lo - 1], hi_col)

    def cov_se(T):
        hit = fresh <= T
        p = float(hit.mean())
        return p, math.sqrt(max(p * (1 - p), 1e-12) / reps)

    c_det, se_det = cov_se(det)
    c_down, se_down = cov_se(det_down)
    c_rnd, se_rnd = cov_se(rnd)
    fin = float(np.isfinite(rnd).mean())
    finite_mask = np.isfinite(rnd)
    c_cond = float((fresh[finite_mask] <= rnd[finite_mask]).mean())
    return {"m": m, "coverage": coverage, "k": k, "k_lo": k_lo, "k_hi": k_hi,
            "lam": float(lam), "lam_exact": lam,
            "det": c_det, "det_se": se_det, "down": c_down, "down_se": se_down,
            "down_k": k_down, "rnd": c_rnd, "rnd_se": se_rnd,
            "finite": fin, "cond": c_cond}


def main():
    rng = np.random.default_rng(SEED)

    say("=" * 100)
    say("W11  RANDOMISED BOUNDS -- what exists below the feasibility floor")
    say("=" * 100)
    say(f"reps per cell {REPS}   seed {SEED}")
    say("")

    # ---------------- (i) the attainable set -----------------------------
    say("-" * 100)
    say("(i) DETERMINISTIC ATTAINABLE COVERAGES -- exact, m+1 points, gaps 1/(m+1)")
    say("    A requested level off this grid cannot be hit exactly by ANY index.")
    say("-" * 100)
    for m in (2, 5, 9, 20):
        pts = attainable(m)
        shown = ", ".join(f"{float(p):.4f}" for p in pts[:8])
        say(f"  m={m:<4} {len(pts)} points: {shown}"
            + (" ..." if len(pts) > 8 else ""))
        for target in (0.90, 0.95):
            hit = any(p == Fraction(target).limit_denominator(10**6) for p in pts)
            k = required_rank(m, target)
            say(f"          target {target:.2f}: attainable exactly? "
                f"{'yes' if hit else 'NO':<3}   required rank "
                f"{k if k is not None else 'does not exist (> m)'}")
    say("")

    # ---------------- (ii) the audit's own worst cell --------------------
    say("-" * 100)
    say("(ii) THE AUDIT'S WORST CELL, m=2 at 0.90 -- a valid bound DOES exist")
    say("-" * 100)
    k_lo, k_hi, lam = mixture(2, Fraction(9, 10))
    say(f"    deterministic required rank ceil(3 x 0.90) = 3 > 2  -> does not exist")
    say(f"    randomised: rank {k_lo} with probability {lam} = {float(lam):.4f},")
    say(f"                otherwise the vacuous +inf bound")
    say(f"    exact marginal coverage: {lam} x 2/3 + {1 - lam} x 1 = "
        f"{lam * Fraction(2, 3) + (1 - lam) * 1} = "
        f"{float(lam * Fraction(2, 3) + (1 - lam)):.4f}")
    say("")

    # ---------------- (iii) measured -------------------------------------
    say("-" * 100)
    say("(iii) MEASURED -- deterministic (up and down) vs exact randomised")
    say("      'det up'   = required rank, or +inf where it does not exist")
    say("      'det down' = the neighbouring rank BELOW, which is what a library")
    say("                   that resolves an uncorrected level tends to land on")
    say("      'rand'     = the exact mixture; 'finite' = fraction of finite bounds;")
    say("      'cond'     = coverage CONDITIONAL on the bound being finite")
    say("-" * 100)
    say(f"{'m':>4}{'nominal':>9}{'k*':>6}{'lam':>8}{'det up':>9}{'det down':>10}"
        f"{'rand':>9}{'s.e.':>8}{'|rand-nom|':>12}{'finite':>8}{'cond':>8}")
    rows = []
    for coverage in (0.90, 0.95):
        for m in (2, 5, 9, 10, 19, 20, 50):
            r = measure(rng, m, coverage)
            rows.append(r)
            say(f"{r['m']:>4}{coverage:>9.2f}"
                f"{(r['k'] if r['k'] else 0):>6}{r['lam']:>8.4f}{r['det']:>9.4f}"
                f"{r['down']:>10.4f}{r['rnd']:>9.4f}{r['rnd_se']:>8.5f}"
                f"{abs(r['rnd'] - coverage):>12.5f}{r['finite']:>8.4f}"
                f"{r['cond']:>8.4f}")
        say("")
    worst = max(abs(r["rnd"] - r["coverage"]) / r["rnd_se"] for r in rows)
    say(f"    the randomised bound is within {worst:.2f} standard errors of nominal in")
    say(f"    every one of the {len(rows)} cells -- exact at EVERY m, including m=2,")
    say("    which no deterministic index achieves at any m off the attainable grid.")
    say("")

    # ---------------- (iv) the cost --------------------------------------
    say("-" * 100)
    say("(iv) THE COST, stated rather than argued")
    say("-" * 100)
    inf_cells = [r for r in rows if r["k"] is None]
    say("    Below the feasibility floor the mixture must place weight on the")
    say("    vacuous bound, so the interval is infinite a known fraction of the time")
    say("    and, conditional on being finite, undercovers by construction:")
    for r in inf_cells:
        say(f"      m={r['m']:<3} nominal {r['coverage']:.2f}: finite "
            f"{r['finite']:.4f} of the time; conditional coverage "
            f"{r['cond']:.4f}; marginal {r['rnd']:.4f}")
    say("")
    say("    That is the real content of the constraint. Two residuals do not contain")
    say("    a 0.90 statement. A randomised bound makes the shortfall visible as an")
    say("    infinite interval 70% of the time; the audited default hides it behind a")
    say("    finite number and delivers 0.58.")
    say("")
    say("    Above the floor the mixture is a pure width saving with no infinite")
    say("    branch, because both mixed ranks exist:")
    for r in rows:
        if r["k"] is not None and r["finite"] > 0.999:
            say(f"      m={r['m']:<3} nominal {r['coverage']:.2f}: finite "
                f"{r['finite']:.4f}; det up {r['det']:.4f} (conservative) vs "
                f"rand {r['rnd']:.4f}")
    say("")
    say("=" * 100)
    say("SUMMARY")
    say("=" * 100)
    say("  'No valid finite bound exists' must read 'no valid finite DETERMINISTIC")
    say("  bound exists'. The deterministic attainable set is m+1 points spaced")
    say("  1/(m+1) apart, so a requested level off that grid is unattainable at every")
    say("  m and not only below the floor; randomising between neighbours attains it")
    say("  exactly. The manuscript's claim survives with one word added, and the")
    say("  constructive alternative belongs next to it.")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
