#!/usr/bin/env python3
"""Re-derive the numbers in the audit's abstract. One environment, minutes not hours.

WHY A SECOND ENTRY POINT EXISTS. Full reproduction needs three pinned environments,
because the audited libraries disagree on numpy and pandas. A referee will not build
three environments. A referee spot-checks for ten minutes or not at all, so the
artifact needs a path that fits in ten minutes and needs nothing but numpy.

WHAT THIS DOES AND DOES NOT ESTABLISH. Every quantity below is recomputed here from
first principles -- exact rational arithmetic, or numpy driven directly -- and then
compared against the committed probe output. So a disagreement means the committed
output is wrong, which is the direction that matters. It does NOT re-run the audited
libraries: the boundary-behaviour counts and the real-data coverage need them, and
those are what the full reproduction is for. Each check says which kind it is.

    python verify_headline.py

Exit status is the number of checks that failed.
"""

import math
import os
import re
import sys
from fractions import Fraction as F

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
checks = []


def committed(fname, pattern, group=1):
    """Pull a figure out of a committed probe output, by regex."""
    with open(os.path.join(OUT, fname), encoding="utf-8", errors="replace") as fh:
        m = re.search(pattern, fh.read(), re.M)
    assert m, f"{fname}: nothing matched {pattern!r}"
    return m.group(group)


def check(name, kind, got, want, tol=0.0):
    ok = (abs(float(got) - float(want)) <= tol) if tol else (str(got) == str(want))
    checks.append((ok, name, kind, got, want))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}\n"
          f"         recomputed {got}   committed {want}   ({kind})")


def main():
    print(__doc__.splitlines()[0])
    print()
    print("(1) THE REQUIRED RANK AND THE FEASIBILITY FLOOR --- recomputed here")
    # The floor n >= 1/alpha - 1 is the whole boundary claim. Derived from the
    # required rank exceeding n, in exact rationals, not read from anywhere.
    for alpha in (F(1, 10), F(1, 20)):
        floor = min(n for n in range(1, 5000)
                    if math.ceil(F(n + 1) * (1 - alpha)) <= n)
        check(f"feasibility floor at alpha={alpha}", "recomputed",
              floor, int(1 / alpha - 1))

    print()
    print("(2) inverted_cdf AT THE CORRECTED LEVEL HITS THE REQUIRED RANK")
    # The paper's recommendation, run against numpy rather than argued. The claim
    # checked here is the one that does not depend on which grid is swept: every
    # departure is one rank PAST the requirement and never short. A count would
    # depend on the grid, and comparing a count against the probe's count over a
    # DIFFERENT grid is the error this project keeps having.
    over = short = cells = 0
    for alpha in (F(1, 10), F(1, 20), F(2, 7), F(1, 3), F(1, 2)):
        for n in range(int(1 / alpha), 400):
            # EXACT rationals for the requirement. Writing this as
            # ceil((n+1) * (1 - float(alpha))) gives 7 where the answer is 6 at
            # alpha = 1/3, n = 8, because 1 - 0.3333333333333333 lands ABOVE 2/3.
            # That is the paper's own subject biting inside its own check, and it
            # is why the requirement is never computed in floating point here.
            k = math.ceil(F(n + 1) * (1 - alpha))
            if k > n:
                continue
            cells += 1
            q = min(1.0, float((1 - alpha) * (n + 1) / n))
            got = int(round(float(np.quantile(np.arange(1, n + 1), q,
                                              method="inverted_cdf"))))
            over += int(got > k)
            short += int(got < k)
            assert got in (k, k + 1), f"departure of {got - k} ranks at n={n}"
    check("inverted_cdf ever falls short of the required rank", "recomputed",
          short, 0)
    print(f"         (over {cells} cells it lands past the requirement {over} "
          f"times, never below; the probe's own count is over its own grid)")

    print()
    print("(3) higher AT THE CORRECTED LEVEL IS EXACT OFF A CLASS OF DENSITY alpha")
    # The claim the audit's background section makes, and the one an earlier draft
    # got backwards. Recomputed against numpy at the level the paper quotes.
    L, over, exact = F(9, 10), 0, 0
    for n in range(10, 400):
        k = math.ceil(F(n + 1) * L)
        if k > n:
            continue
        q = min(1.0, float(L * (n + 1) / n))
        got = int(round(float(np.quantile(np.arange(1, n + 1), q,
                                          method="higher"))))
        over += int(got > k)
        exact += int(got == k)
        assert got >= k, f"higher fell SHORT at n={n}: {got} < {k}"
    check("higher over-covers at a fraction alpha of sizes", "recomputed",
          f"{over / (over + exact):.2f}", f"{float(1 - L):.2f}")

    print()
    print("(4) numpy's DEFAULT DELIVERS THE WRONG COVERAGE AT n=50, q=0.90")
    # h/(n+1) for method='linear', against the committed cross-library figure.
    n = 50
    h = float(np.quantile(np.arange(1, n + 1), 0.90))     # linear, the default
    got = h / (n + 1)
    want = committed(
        "probe_output_cross_library.txt",
        r"^numpy\.quantile method='linear'\s+\S+\s+\S+\s+\S+\s+([\d.]+)")
    check("numpy default delivered coverage", "recomputed",
          f"{got:.4f}", f"{float(want):.4f}", tol=5e-4)

    print()
    print("(5) THE SHIPPED DEFAULT BELOW ITS OWN FLOOR --- read, not re-run")
    # This one needs statsforecast and 250 real series, so it is READ from the
    # committed output and the arithmetic around it is what gets recomputed: at
    # m=2 the required rank for 0.90 is 3, which exceeds the two scores available.
    k = math.ceil(F(3) * F(9, 10))
    check("required rank at m=2, nominal 0.90", "recomputed", k, 3)
    check("...exceeds the calibration size", "recomputed", k > 2, True)
    # Anchored through the nominal-0.90 block and then the n_windows=2 cell. An
    # unanchored search picked up the 0.95 header and printed the LEVEL as though
    # it were the delivered coverage.
    delivered = committed(
        "probe_output_real_data_statsforecast.txt",
        r"nominal 0\.90\n\s*n_windows=2\s.*?\n\s*arm A \(shipped\)\s+"
        r"coverage ([\d.]+)")
    print(f"         the shipped call delivers {delivered} there "
          f"(read from the committed output; re-running it needs the archive)")

    print()
    bad = [c for c in checks if not c[0]]
    print(f"{len(checks) - len(bad)} of {len(checks)} checks reproduce.")
    if bad:
        print("FAILED:")
        for _, name, kind, got, want in bad:
            print(f"  {name}: recomputed {got}, committed {want} ({kind})")
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
