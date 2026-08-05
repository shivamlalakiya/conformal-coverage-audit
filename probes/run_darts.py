"""darts 0.46.1, RUN: closing the last "read, not run" row in the §5.3 map.

§9 requires verification by running.  darts was the one library in the empirical
map classified from source alone, in a paper whose method claim IS run-verification.

Three parts, each with an oracle independent of darts:
  A  index arithmetic -- which order statistic does darts' np.quantile(..., method="higher")
     actually return, versus the required ceil(c(n+1)), in exact rationals?
  B  the convention in isolation -- paired Monte Carlo coverage, darts' rule vs correct,
     on identical draws.
  C  end-to-end -- fit a real ConformalNaiveModel and measure interval coverage.

The handoff's hand-derived claim "short by ~1-2 order stats" is NOT assumed here;
Part A computes it.  §9: a hand-derived assertion has already been caught wrong once.
"""

from fractions import Fraction as F
import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ oracle --
def required_rank(c, n):
    """1-based order statistic a valid split-conformal upper bound needs."""
    return math.ceil(F(c) * (n + 1))


def darts_rank_theory(c, n):
    """1-based rank numpy's method='higher' lands on for level c."""
    return math.ceil(F(c) * (n - 1)) + 1


def self_check():
    """§9: exact-arithmetic self-check at import, on cells the *bug* picks."""
    # vacuity boundary: required rank exceeds n exactly when c(n+1) > n
    assert required_rank(F(9, 10), 9) == 9, required_rank(F(9, 10), 9)
    assert required_rank(F(9, 10), 10) == 10
    assert required_rank(F(19, 20), 19) == 19
    assert required_rank(F(19, 20), 20) == 20
    # numpy's method="higher" must agree with darts_rank_theory on real arrays
    rng = np.random.default_rng(0)
    for n in range(2, 120):
        vals = np.arange(1.0, n + 1.0)          # value == rank
        for c in (F(9, 10), F(19, 20), F(4, 5), F(77, 100), F(2, 3)):
            got = np.quantile(vals, float(c), method="higher")
            assert int(got) == darts_rank_theory(c, n), (c, n, got)
    # a non-unit-fraction level must be in the grid -- §9
    assert darts_rank_theory(F(2, 3), 10) == math.ceil(F(2, 3) * 9) + 1 == 7
    print("self_check: OK (rank model matches numpy on 590 cells, exact rationals)")


self_check()


LEVELS = [F(9, 10), F(19, 20), F(4, 5), F(77, 100), F(2, 3)]
LEVEL_NAMES = {F(9, 10): "0.90", F(19, 20): "0.95", F(4, 5): "0.80",
               F(77, 100): "0.77", F(2, 3): "2/3"}


# ------------------------------------------------- PART A: index arithmetic --
print("\n" + "=" * 78)
print("PART A -- which order statistic does darts return vs the one it needs?")
print("=" * 78)
print("darts: np.quantile(residuals, c, method='higher'), c UNCORRECTED")
print("       (conformal_models.py:1686 and :1837; c = interval_range_sym, :161-170,")
print("        which carries no (n+1)/n factor anywhere in the file)")
print(f"\n{'level':>6} {'n':>5} {'darts rank':>11} {'needs rank':>11} {'deficit':>8} {'valid at all?':>14}")
deficit_hist = {}
for c in LEVELS:
    for n in (5, 9, 10, 15, 20, 30, 50, 100, 500):
        d, r = darts_rank_theory(c, n), required_rank(c, n)
        feasible = "vacuous" if r > n else "yes"
        print(f"{LEVEL_NAMES[c]:>6} {n:>5} {d:>11} {r:>11} {r - d:>8} {feasible:>14}")
    print()

for c in LEVELS:
    for n in range(2, 1001):
        deficit_hist[c] = deficit_hist.get(c, {})
        k = required_rank(c, n) - darts_rank_theory(c, n)
        deficit_hist[c][k] = deficit_hist[c].get(k, 0) + 1

print("deficit distribution over n = 2..1000 (required rank minus darts rank):")
for c in LEVELS:
    dist = ", ".join(f"{k}: {v}" for k, v in sorted(deficit_hist[c].items()))
    print(f"  level {LEVEL_NAMES[c]:>5}   {dist}")
print("\n-> the handoff's hand-derived '~1-2 order stats' is NOT what the arithmetic")
print("   gives.  Read the row above before quoting a deficit.")


# ------------------------------------- PART B: the convention in isolation --
print("\n" + "=" * 78)
print("PART B -- paired coverage, darts' convention vs correct, identical draws")
print("=" * 78)
DRAWS = 200_000
rng = np.random.default_rng(4082026)
print(f"{DRAWS:,} draws/cell, one-sided upper bound, standard normal scores\n")
print(f"{'level':>6} {'n':>5} {'darts':>9} {'correct':>9} {'deficit':>9} {'paired s.e.':>12}")
for c in (F(9, 10), F(19, 20)):
    for n in (9, 10, 15, 20, 30, 50, 100):
        s = rng.standard_normal((DRAWS, n + 1))
        cal, test = np.sort(s[:, :n], axis=1), s[:, n]
        d_rank, r_rank = darts_rank_theory(c, n), required_rank(c, n)
        hit_d = test <= cal[:, d_rank - 1]
        hit_r = (np.ones(DRAWS, dtype=bool) if r_rank > n
                 else test <= cal[:, r_rank - 1])
        diff = hit_r.astype(float) - hit_d.astype(float)
        se = diff.std(ddof=1) / math.sqrt(DRAWS)
        tag = " (vacuous)" if r_rank > n else ""
        print(f"{LEVEL_NAMES[c]:>6} {n:>5} {hit_d.mean():>9.4f} "
              f"{hit_r.mean():>9.4f}{tag} {diff.mean():>9.4f} {se:>12.5f}")
    print()


# --------------------------------------------------- PART C: end-to-end -----
print("=" * 78)
print("PART C -- end-to-end through darts' real ConformalNaiveModel")
print("=" * 78)
from darts import TimeSeries                                    # noqa: E402
from darts.models import ConformalNaiveModel, LinearRegressionModel  # noqa: E402

REPS, CAL = 300, 10
QUANTILES = [0.05, 0.5, 0.95]          # 90% interval
print(f"{REPS} independent fits, cal_length={CAL}, quantiles={QUANTILES}")
print("iid Gaussian series, LinearRegressionModel(lags=1) as the base model --")
print("deliberately near-exchangeable residuals, so the guarantee SHOULD hold.\n")

rng = np.random.default_rng(20260804)
hits, widths = [], []
for rep in range(REPS):
    y = rng.standard_normal(80)
    series = TimeSeries.from_values(y[:-1])
    base = LinearRegressionModel(lags=1)
    base.fit(series)
    cm = ConformalNaiveModel(model=base, quantiles=QUANTILES, cal_length=CAL)
    pred = cm.predict(n=1, predict_likelihood_parameters=True, num_samples=1)
    vals = pred.values().ravel()
    lo, hi = float(vals[0]), float(vals[-1])
    truth = float(y[-1])
    hits.append(lo <= truth <= hi)
    widths.append(hi - lo)

cov = float(np.mean(hits))
se = float(np.std(hits, ddof=1) / math.sqrt(REPS))
print(f"requested two-sided coverage : 0.9000")
print(f"measured                     : {cov:.4f}  (s.e. {se:.4f}, {REPS} fits)")
print(f"mean interval width          : {np.mean(widths):.4f}")
print(f"\nconvention-only prediction for n={CAL} at the 0.95 upper rail:")
print(f"  darts rank {darts_rank_theory(F(19, 20), CAL)} of {CAL}, "
      f"needs rank {required_rank(F(19, 20), CAL)} "
      f"({'VACUOUS -- no valid finite bound exists at this n' if required_rank(F(19, 20), CAL) > CAL else 'feasible'})")
print("\nNOTE the honest-scope caveat that §5.5 already carries for sktime applies")
print("here too: this harness is iid by construction so the guarantee should hold")
print("exactly.  It is NOT a claim about real dependent time series.")
