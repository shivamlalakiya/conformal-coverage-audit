#!/usr/bin/env python3
"""crepes #48: membership is decided by a floating-point comparison, so a label whose
p-value exactly equals the threshold is dropped.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
Membership is `p_values >= 1-confidence`. Both sides are doubles. A p-value from n
calibration scores is a multiple of 1/(n+1), and the threshold is formed as
`1 - confidence`. Those two routes to the same rational number do not land on the same
double:

    1/20        -> 0.05
    1 - 0.95    -> 0.050000000000000044

so `1/20 >= 1-0.95` is **False**, and a label whose p-value is exactly the significance
level is excluded. Under the definition of a conformal prediction set it belongs in.

Direction of harm: **anti-conservative**. The set is smaller than the confidence level
warrants, so this costs validity rather than width -- at exactly the sizes where the
p-value can equal the level, which is a congruence in n.

The script tests the arithmetic first, then drives the library on constructed scores so
the dropped label is observed and not inferred.
"""

import sys
from fractions import Fraction

import numpy as np


def main():
    import crepes
    from crepes import ConformalClassifier

    print(f"crepes {crepes.__version__}, numpy {np.__version__}")
    print()
    print("(1) THE ARITHMETIC. Exact equality, two float routes, three levels.")
    print()
    print(f"{'confidence':>12}{'1/(n+1) with n+1':>18}{'p as double':>26}"
          f"{'1-confidence as double':>26}{'p >= t':>8}{'exact':>7}")
    print("-" * 100)
    cases = []
    for confidence, denom in ((0.95, 20), (0.99, 100), (0.90, 10)):
        p = 1 / denom
        t = 1 - confidence
        exact = Fraction(1, denom) >= (Fraction(1)
                                       - Fraction(confidence)
                                       .limit_denominator(10 ** 6))
        cases.append({"c": confidence, "d": denom, "float": p >= t, "exact": exact})
        print(f"{confidence:>12}{denom:>18}{repr(p):>26}{repr(t):>26}"
              f"{str(p >= t):>8}{str(exact):>7}")
    print()
    wrong = [c for c in cases if c["float"] != c["exact"]]
    print(f"levels where the float comparison disagrees with exact arithmetic: "
          f"{len(wrong)} of {len(cases)}"
          + (f"  ({', '.join(str(c['c']) for c in wrong)})" if wrong else ""))
    assert any(c["exact"] for c in cases), (
        "no case has p exactly equal to the level, so the boundary is never tested")
    print()

    print("(2) THE LIBRARY. A calibration set of n scores and a test score placed so")
    print("    that its p-value is exactly 1/(n+1).")
    print()
    print(f"{'confidence':>12}{'n':>5}{'exact p':>10}{'threshold':>12}"
          f"{'label kept?':>13}{'should be':>11}")
    print("-" * 66)
    dropped = []
    for confidence, denom in ((0.95, 20), (0.99, 100), (0.90, 10)):
        n = denom - 1
        # calibration alphas 1..n; a test alpha above all of them has p = 1/(n+1)
        cal = np.arange(1, n + 1, dtype=float)
        test = np.array([[float(n) + 1.0]])
        cc = ConformalClassifier()
        cc.fit(cal)
        ps = cc.predict_set(test, confidence=confidence, smoothing=False)
        kept = bool(np.ravel(ps)[0])
        exact_p = Fraction(1, denom)
        should = exact_p >= (Fraction(1) - Fraction(confidence)
                            .limit_denominator(10 ** 6))
        if should and not kept:
            dropped.append(confidence)
        print(f"{confidence:>12}{n:>5}{str(exact_p):>10}{1 - confidence:>12.4f}"
              f"{str(kept):>13}{str(should):>11}")

    print()
    print(f"levels at which the label is dropped though it belongs in: {len(dropped)}"
          + (f"  ({', '.join(str(c) for c in dropped)})" if dropped else ""))
    print()
    if wrong and dropped:
        print("REPRODUCES. The threshold and the p-value reach the same rational number "
              "by different float routes and do not meet, so a label at exactly the "
              "significance level is excluded. Anti-conservative, and it depends on the "
              "calibration size through 1/(n+1).")
        return 0
    print(f"does not reproduce: {len(wrong)} arithmetic disagreements, "
          f"{len(dropped)} labels dropped")
    return 1


if __name__ == "__main__":
    sys.exit(main())
