"""Is branch (d) — no (n+1)/n correction — a real coverage defect, or a rounding curiosity?

Branch (d) is what sktime's ConformalIntervals and river's RegressionJackknife do:
take np.quantile(scores, 1 - alpha) directly, with no finite-sample inflation.
The valid split-conformal threshold is order statistic ceil((1-alpha)(n+1)) of n.

Exact part: coverage of order statistic k out of n against a fresh exchangeable
point is exactly k/(n+1). Checked in rationals, no floats.
Monte Carlo part: np.quantile interpolates, so its threshold is not an order
statistic and needs measuring rather than deriving.
"""

from fractions import Fraction as F
from math import ceil

import numpy as np


def correct_k(n, alpha):
    """Order statistic the valid conformal threshold needs. Integer arithmetic only."""
    a = F(alpha).limit_denominator(10**6)
    return ceil((1 - a) * (n + 1))


def exact_coverage(k, n):
    """P(Y_new <= X_(k)) for exchangeable draws = k/(n+1). Vacuous if k > n."""
    return None if k > n else F(k, n + 1)


def self_check():
    """The closed forms, against values derived by hand. Fails loudly, per §6."""
    # Feasibility boundary: k > n exactly when n <= 1/alpha - 1, i.e. n <= 18 at
    # alpha = 1/20. So n=18 is the LAST vacuous cell and n=19 the first feasible
    # one -- an earlier version of this check asserted 18 was feasible and was wrong.
    assert correct_k(18, F(1, 20)) == 19, correct_k(18, F(1, 20))
    assert exact_coverage(correct_k(18, F(1, 20)), 18) is None  # 19 > 18 -> vacuous
    assert correct_k(19, F(1, 20)) == 19
    assert exact_coverage(correct_k(19, F(1, 20)), 19) == F(19, 20)
    # n_y = 98/99 one-index result from §3.3, in exact rationals.
    assert correct_k(98, F(1, 100)) == 99 and exact_coverage(99, 98) is None
    assert correct_k(99, F(1, 100)) == 99 and exact_coverage(99, 99) == F(99, 100)
    # A non-unit-fraction alpha, because §6 says a unit-fraction-only grid lies.
    assert correct_k(20, F(23, 100)) == ceil(F(77, 100) * 21) == 17
    print("self-check passed")


def measured_coverage(n, alpha, draws, rng):
    """Coverage of the branch-(d) threshold vs the corrected one, same draws.

    Scores are exchangeable standard-normal magnitudes: n calibration, 1 test.
    Returns (branch_d_coverage, corrected_coverage).
    """
    s = np.abs(rng.standard_normal((draws, n + 1)))
    cal, test = s[:, :n], s[:, n]
    a = float(alpha)
    q_d = np.quantile(cal, 1 - a, axis=1)  # branch (d): uncorrected level
    k = correct_k(n, alpha)
    if k > n:  # corrected threshold does not exist -> +inf, coverage 1
        q_c = np.full(draws, np.inf)
    else:
        q_c = np.sort(cal, axis=1)[:, k - 1]  # order statistic k, no interpolation
    return (test <= q_d).mean(), (test <= q_c).mean()


if __name__ == "__main__":
    self_check()
    rng = np.random.default_rng(20260804)
    DRAWS = 200_000
    for alpha in (F(1, 10), F(1, 20)):
        target = 1 - float(alpha)
        print(f"\nalpha = {alpha}  requested coverage {target:.2f}   {DRAWS} draws")
        print(f"{'n':>5} {'branch(d)':>10} {'correct':>9} {'exact k/(n+1)':>14} "
              f"{'(d) deficit':>12}")
        for n in (10, 15, 20, 30, 50, 100, 200, 500):
            cov_d, cov_c = measured_coverage(n, alpha, DRAWS, rng)
            k = correct_k(n, alpha)
            ex = exact_coverage(k, n)
            ex_s = "vacuous" if ex is None else f"{float(ex):.4f}"
            print(f"{n:>5} {cov_d:>10.4f} {cov_c:>9.4f} {ex_s:>14} "
                  f"{target - cov_d:>+12.4f}")
        # The corrected arm must match its exact closed form wherever it exists.
        for n in (20, 50, 200):
            _, cov_c = measured_coverage(n, alpha, DRAWS, rng)
            ex = float(exact_coverage(correct_k(n, alpha), n))
            assert abs(cov_c - ex) < 0.004, (n, alpha, cov_c, ex)
    print("\ncorrected arm matches k/(n+1) at every checked cell")
