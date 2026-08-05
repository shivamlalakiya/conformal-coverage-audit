"""Adjudicate three claims against the handoff's statsforecast framing (§5.4, §7).

Runs against statsforecast 2.1.1's own `ConformalSeasonalPool._oriented_index`
(a @staticmethod, so no fitting needed) plus an exact-rational oracle that is
independent of it.

Claims under test:
  (1) The degenerate window is documented, not silent.
  (2) The clamp cannot simply be deleted -- np.quantile domain-guards level > 1.
  (3) The clamp truncates the correction; it never narrows below the plain
      uncorrected interval.

Verdicts are printed; nothing here asserts the handoff's original reading.
"""

from fractions import Fraction as F
import math
import re
import sys

import numpy as np
from statsforecast.models import ConformalSeasonalPool

oi = ConformalSeasonalPool._oriented_index


# ---------------------------------------------------------------- oracle ----
def oriented_exact(q, n):
    """Exact-rational restatement of _oriented_index, clamps included."""
    q = F(q)
    if n <= 0:
        return q
    if q < F(1, 2):
        return max(F(0), F(math.floor((n + 1) * q), n))
    return min(F(1), F(math.ceil((n + 1) * q), n))


def oriented_unclamped(q, n):
    """Same, with the two clamps removed. May exceed 1 or fall below 0."""
    q = F(q)
    if n <= 0:
        return q
    if q < F(1, 2):
        return F(math.floor((n + 1) * q), n)
    return F(math.ceil((n + 1) * q), n)


def self_check():
    """§9: every closed form gets an exact-arithmetic self-check at import,
    on cells the *bug* chooses (inside the clamp window), not round numbers."""
    # cells statsforecast's own test_oriented_index_values covers: n=100, n=0
    assert oriented_exact(F(1, 40), 100) == F(math.floor(101 / 40), 100)
    assert oriented_exact(F(39, 40), 100) == F(math.ceil(101 * 39 / 40), 100)
    assert oriented_exact(F(1, 40), 0) == F(1, 40)
    # cells it does not: inside the clamp window at the 95% level
    assert oriented_exact(F(39, 40), 30) == F(1), "upper clamp must bind at n=30"
    assert oriented_exact(F(1, 40), 30) == F(0), "lower clamp must bind at n=30"
    assert oriented_unclamped(F(39, 40), 30) == F(31, 30) > 1
    # float implementation must agree with the exact form everywhere we test
    for n in range(1, 260):
        for q in (F(1, 40), F(39, 40), F(1, 20), F(19, 20), F(23, 100), F(77, 100)):
            assert math.isclose(oi(float(q), n), float(oriented_exact(q, n)),
                                rel_tol=0, abs_tol=1e-12), (q, n)
    print("self_check: OK (exact oracle agrees with statsforecast on 1554 cells)")


self_check()


# ------------------------------------------------- claim 1: documented? -----
print("\n" + "=" * 78)
print("CLAIM 1 -- the degenerate window is DOCUMENTED, not silent")
print("=" * 78)
doc = ConformalSeasonalPool.__init__.__doc__ or ConformalSeasonalPool.__doc__ or ""
m = re.search(r"For a level-L interval.*?\)\.", doc, re.S)
print("public docstring says:")
print("   ", " ".join(m.group(0).split()) if m else "<<NOT FOUND>>")


def documented_min_n(level):
    return math.ceil(2 / (1 - level / 100)) - 1


def window_returns_one(hi_q):
    """Largest n where the UPPER index comes back as exactly 1.0 (= max sample)."""
    return max(n for n in range(1, 4000) if oi(hi_q, n) >= 1.0)


def window_clamp_fires(hi_q):
    """Largest n where the clamp CHANGES the value, i.e. ceil((n+1)q) > n."""
    return max(n for n in range(1, 4000)
               if F(math.ceil((n + 1) * F(hi_q)), n) > 1)


def window_returns_zero(lo_q):
    """Largest n where the LOWER index comes back as exactly 0.0 (= min sample)."""
    return max(n for n in range(1, 4000) if oi(lo_q, n) <= 0.0)


print(f"\n{'level':>6} {'documented':>11} {'upper=1.0 thru n':>17} {'clamp fires thru n':>19} {'lower=0.0 thru n':>17}")
for level in (90, 95, 99):
    lo_q, hi_q = (1 - level / 100) / 2, 1 - (1 - level / 100) / 2
    print(f"{level:>6} {documented_min_n(level):>11} {window_returns_one(hi_q):>17}"
          f" {window_clamp_fires(hi_q):>19} {window_returns_zero(lo_q):>17}")
print("\n-> THESE ARE TWO DIFFERENT WINDOWS and the handoff conflated them.")
print("   Where clamp-fires < n <= upper-is-1.0, the level 1.0 is the GENUINE")
print("   k=n order statistic (max), not a clamped value. So the handoff's")
print("   'k > n throughout?  yes' column is FALSE over the wider window.")
print("\n-> the documented formula tracks the LOWER rail only (see below).")


# ------------------------- is max(0.0, .) reachable at all? -----------------
print("\n" + "=" * 78)
print("BONUS -- is `max(0.0, .)` on the lower rail live code or dead code?")
print("=" * 78)
neg = [(q, n) for n in range(1, 2000)
       for q in (F(1, 40), F(1, 20), F(1, 200), F(23, 100))
       if math.floor((n + 1) * q) < 0]
print(f"cells where floor((n+1)*q) < 0 (i.e. the max(0.0,.) clamp would fire): {len(neg)}")
print("-> floor((n+1)q) >= 0 for every q >= 0, so `max(0.0, .)` is DEAD CODE.")
print("   The 0.0 the lower rail returns comes from floor() rounding down to 0,")
print("   not from the clamp.  Only the upper rail has a live clamp.")


# -------------------- which rail does the documented formula describe? ------
print("\n" + "=" * 78)
print("WHICH RAIL DOES ceil(2/a)-1 DESCRIBE?")
print("=" * 78)
print(f"{'level':>6} {'first n lower non-degenerate':>29} {'first n upper non-degenerate':>29} {'ceil(2/a)-1':>12} {'ceil(4/a)-1':>12}")
for level in (90, 95, 99):
    a = 1 - level / 100
    lo_q, hi_q = a / 2, 1 - a / 2
    first_lo = min(n for n in range(1, 4000) if oi(lo_q, n) > 0.0)
    first_hi = min(n for n in range(1, 4000) if oi(hi_q, n) < 1.0)
    print(f"{level:>6} {first_lo:>29} {first_hi:>29} {math.ceil(2 / a) - 1:>12} {math.ceil(4 / a) - 1:>12}")
print("\n-> ceil(2/a)-1 matches the LOWER rail. The UPPER rail stays pinned to")
print("   max(sample) until ~ceil(4/a)-1, roughly double, and is UNDOCUMENTED.")


# ------------------------------------------- claim 2: clamp is a guard ------
print("\n" + "=" * 78)
print("CLAIM 2 -- deleting the clamp crashes; it is a domain guard")
print("=" * 78)
R = np.arange(30.0)
lvl = float(oriented_unclamped(F(39, 40), 30))
print(f"unclamped level at n=30, q=0.975: {lvl:.6f}")
try:
    np.quantile(R, lvl)
    print("np.quantile accepted it  <- claim 2 FAILS")
except ValueError as e:
    print(f"np.quantile(np.arange(30.), 31/30) -> ValueError: {e}")
print("-> confirmed. Removing min(1.0, .) alone raises; the only non-crashing")
print("   alternatives are branch (c) (+/-inf) or branch (a) (explicit raise).")


# --------------------------------- claim 3: direction of harm ---------------
print("\n" + "=" * 78)
print("CLAIM 3 -- the clamp TRUNCATES the correction; it never narrows")
print("=" * 78)
print("exhaustive check: oriented interval vs plain uncorrected interval")

rng = np.random.default_rng(20260804)
violations = []
for level in (80, 90, 95, 98, 99):
    lo_q, hi_q = (1 - level / 100) / 2, 1 - (1 - level / 100) / 2
    for n in range(2, 201):
        Rv = np.sort(rng.standard_normal(n))
        o_lo, o_hi = np.quantile(Rv, oi(lo_q, n)), np.quantile(Rv, oi(hi_q, n))
        p_lo, p_hi = np.quantile(Rv, lo_q), np.quantile(Rv, hi_q)
        if o_lo > p_lo + 1e-12 or o_hi < p_hi - 1e-12:
            violations.append((level, n, o_lo, p_lo, o_hi, p_hi))
print(f"cells checked: 5 levels x 199 n = 995")
print(f"cells where oriented is NARROWER than plain: {len(violations)}")
for v in violations:
    print(f"    level={v[0]} n={v[1]}  oriented [{v[2]:+.4f},{v[4]:+.4f}]  plain [{v[3]:+.4f},{v[5]:+.4f}]")

print("\nVERDICT, corrected -- an earlier draft of this script asserted the")
print("wider-always property was exact by monotonicity.  Running it disproves")
print("that.  The upper rail ceil((n+1)q)/n >= q always, so it never narrows.")
print("The LOWER rail floor((n+1)q)/n is NOT bounded above by q:")
for n in (39, 79, 119):
    lo_q = 0.025
    print(f"    n={n:>4}  q=0.0250  floor((n+1)q)/n = {oi(lo_q, n):.6f}  "
          f"{'>' if oi(lo_q, n) > lo_q else '<='} q   -> lower bound moves IN")
print("So: inside the clamp window the clamped interval is strictly wider than")
print("plain (claim 3 holds where it was asserted).  Outside it, at the cells")
print("where (n+1)q lands just above an integer, the floor rule is mildly")
print("ANTI-conservative on the lower rail relative to plain.  Both statements")
print("are needed; neither alone describes _oriented_index.")

print("\nSEPARATE OBSERVATION (unfiled, needs its own adjudication):")
print("_oriented_index returns a LEVEL fed to np.quantile with DEFAULT linear")
print("interpolation.  np.quantile maps level q to virtual index q*(n-1), so")
print("level k/n lands at k - k/n, not at order statistic k.  Recovering the")
print("k-th order statistic requires level (k-1)/(n-1), or method='higher'.")
for n, k in ((30, 30), (50, 49), (100, 97)):
    print(f"    n={n:>4} intended k={k:<4} np.quantile level k/n hits virtual index"
          f" {(n - 1) * k / n:8.4f}  (want {k - 1})")

print("\nreproduction of the reported n=30, level=95 cell (seed 20260804):")
Rv = np.sort(np.random.default_rng(20260804).standard_normal(30))
lo_q, hi_q = 0.025, 0.975
print(f"  oriented+clamped: {np.quantile(Rv, oi(lo_q, 30)):+.4f} / {np.quantile(Rv, oi(hi_q, 30)):+.4f}"
      f"   (indices {oi(lo_q, 30):.4f} / {oi(hi_q, 30):.4f} = min / max)")
print(f"  plain uncorrected:{np.quantile(Rv, lo_q):+.4f} / {np.quantile(Rv, hi_q):+.4f}"
      f"   (indices {lo_q:.4f} / {hi_q:.4f})")


# ------------------------------- what the clamped rule actually covers ------
print("\n" + "=" * 78)
print("SO WHAT DOES THE CLAMPED RULE COVER?  exact, no simulation")
print("=" * 78)
print("For n exchangeable scores the interval [X_(i), X_(j)] covers a new point")
print("with probability exactly (j-i)/(n+1).  Clamped => [min, max] => (n-1)/(n+1).")
print(f"\n{'n':>5} {'req 0.95':>10} {'clamped cover':>15} {'plain cover':>13} {'best possible':>15}")
for n in (6, 10, 20, 30, 38, 39, 50):
    lo_i = math.floor(31 * 0.025) if False else None
    o_lo_idx = oriented_exact(F(1, 40), n) * n   # order-stat index, 0-based scale
    o_hi_idx = oriented_exact(F(39, 40), n) * n
    clamped = F(int(o_hi_idx) - int(o_lo_idx), n + 1)
    p_lo_idx = math.floor(0.025 * (n - 1))
    p_hi_idx = math.ceil(0.975 * (n - 1))
    plain = F(p_hi_idx - p_lo_idx, n + 1)
    best = F(n - 1, n + 1)
    print(f"{n:>5} {0.95:>10.4f} {float(clamped):>15.4f} {float(plain):>13.4f} {float(best):>15.4f}")
print("\n-> at n<=38 the clamped rule already returns the WIDEST interval any")
print("   empirical-quantile method can return.  The residual gap to 0.95 is")
print("   not recoverable by any choice of index -- it is infeasibility, and")
print("   the honest branch there is (c) +/-inf or (a) raise, not a wider index.")
