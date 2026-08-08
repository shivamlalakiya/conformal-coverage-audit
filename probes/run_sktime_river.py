"""Branch (d), RUN rather than read: sktime 1.1.0 and river 0.25.0 through public APIs.

§3.5 gap 1. The 6.7 pp figure in branch_d_check.py is a property of the *convention*
(np.quantile at an uncorrected level). This script asks whether the two libraries
classified as branch (d) by source reading actually undercover when driven normally.

Part A  sktime, mechanism: compare sktime's own threshold against the valid conformal
        order statistic, computed from sktime's own residuals matrix. Exact, no sampling.
Part B  sktime, coverage: fit ConformalIntervals on iid data, predict_interval, test a
        held-out future point. Repeated over independent seeds.
Part C  river, mechanism + coverage: RegressionJackknife driven with a non-learning
        regressor so residuals are exactly the iid targets, hence exchangeable.

Design note for Part B: residuals come from expanding-window refits, so they are only
approximately exchangeable. initial_window is kept LARGE while the residual count m is
kept SMALL, which makes the training windows nearly identical in size and the residual
variance nearly constant. Any remaining effect inflates residuals from smaller windows,
i.e. biases toward CONSERVATISM -- so undercoverage measured here is not an artifact.
"""

from math import ceil

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- oracle (independent)
def valid_k(n, alpha):
    """Order statistic a valid split-conformal threshold needs. None if it exceeds n."""
    k = ceil((1 - alpha) * (n + 1))
    return None if k > n else k


def valid_threshold(scores, alpha):
    """Order statistic k of scores, or inf when no finite conformal bound exists."""
    s = np.sort(np.asarray(scores))
    k = valid_k(len(s), alpha)
    return np.inf if k is None else s[k - 1]


def _self_check():
    # n=10, alpha=0.1 -> ceil(0.9*11)=10 <= 10, so the max is the valid threshold.
    assert valid_k(10, 0.1) == 10
    assert valid_threshold(np.arange(1.0, 11.0), 0.1) == 10.0
    # n=9, alpha=0.1 -> ceil(0.9*10)=9 <= 9 -> feasible, the max again.
    assert valid_k(9, 0.1) == 9
    # n=8, alpha=0.1 -> ceil(0.9*9)=9 > 8 -> vacuous.
    assert valid_k(8, 0.1) is None and valid_threshold(np.arange(8.0), 0.1) == np.inf
    # alpha=0.05 boundary from the handoff: n=18 last vacuous, n=19 first feasible.
    assert valid_k(18, 0.05) is None and valid_k(19, 0.05) == 19
    print("oracle self-check passed")


# --------------------------------------------------------------------------- Part A + B
def sktime_probe(n_resid, alpha, reps, seed, method="conformal"):
    """Drive ConformalIntervals normally; return (coverage, mechanism_rows)."""
    from sktime.forecasting.conformal import ConformalIntervals
    from sktime.forecasting.naive import NaiveForecaster

    initial_window = 60          # large, so refit windows are near-identical in size
    m = n_resid + 1              # diagonal at offset 1 has length m - 1 = n_resid
    n_obs = initial_window + m
    coverage_req = 1 - alpha
    rng = np.random.default_rng(seed)

    hits, mech = 0, []
    for r in range(reps):
        y_all = rng.standard_normal(n_obs + 1)
        idx = pd.period_range("2000-01", periods=n_obs + 1, freq="M")
        y = pd.Series(y_all[:-1], index=idx[:-1])
        y_future = y_all[-1]

        f = ConformalIntervals(
            NaiveForecaster(strategy="mean"),
            initial_window=initial_window,
            method=method,
        )
        f.fit(y, fh=[1])
        pi = f.predict_interval(coverage=[coverage_req])
        lo = float(pi.iloc[0, 0])
        hi = float(pi.iloc[0, 1])
        hits += lo <= y_future <= hi

        if r < 3:  # mechanism: sktime's threshold vs the valid one, same residuals
            resids = np.diagonal(f.residuals_matrix_.to_numpy(), offset=1)
            resids = resids[~np.isnan(resids)]
            abs_r = np.abs(resids)
            sk = float(np.quantile(abs_r, coverage_req))   # what sktime computes
            ok = valid_threshold(abs_r, alpha)             # what validity requires
            mech.append((len(abs_r), sk, ok, valid_k(len(abs_r), alpha)))
    return hits / reps, mech


# ------------------------------------------------------------------------------- Part C
def river_probe(n_cal, alpha, reps, seed):
    """RegressionJackknife with a non-learning regressor: residuals ARE the targets."""
    from river import base
    from river.conf import RegressionJackknife

    class Zero(base.Regressor):
        """Predicts 0.0 always and never learns, so residual == y exactly."""

        def learn_one(self, x, y, **kw):
            return None

        def predict_one(self, x, **kw):
            return 0.0

    rng = np.random.default_rng(seed)
    hits, degenerate = 0, 0
    for _ in range(reps):
        model = RegressionJackknife(Zero(), confidence_level=1 - alpha)
        ys = rng.standard_normal(n_cal)
        for v in ys:
            model.learn_one({}, float(v))
        iv = model.predict_one({}, with_interval=True)
        lo, hi = float(iv.lower), float(iv.upper)
        if lo == hi:
            degenerate += 1
        y_new = float(rng.standard_normal())
        hits += lo <= y_new <= hi
    return hits / reps, degenerate / reps


def river_attribution(n_cal, alpha, reps, seed):
    """Split river's deficit into (i) P2 estimator error and (ii) missing correction.

    Compares river's streaming P2 quantile against the exact empirical quantile at the
    same level, and against the valid conformal order statistic, on identical samples.
    """
    from river import stats

    rng = np.random.default_rng(seed)
    p2_lo, p2_hi, emp_lo, emp_hi, val_hi = [], [], [], [], []
    for _ in range(reps):
        ys = rng.standard_normal(n_cal)
        lo, hi = stats.Quantile(alpha / 2), stats.Quantile(1 - alpha / 2)
        for v in ys:
            lo.update(float(v))
            hi.update(float(v))
        p2_lo.append(lo.get() if lo.get() is not None else np.nan)
        p2_hi.append(hi.get() if hi.get() is not None else np.nan)
        emp_lo.append(np.quantile(ys, alpha / 2))
        emp_hi.append(np.quantile(ys, 1 - alpha / 2))
        # one-sided valid threshold on the signed scores, for scale reference
        k = valid_k(n_cal, alpha / 2)
        val_hi.append(np.inf if k is None else np.sort(ys)[k - 1])
    f = lambda a: float(np.nanmean(a))  # noqa: E731
    return dict(p2=(f(p2_lo), f(p2_hi)), empirical=(f(emp_lo), f(emp_hi)),
                valid_upper=f(val_hi), truth=(-1.6449, 1.6449))


if __name__ == "__main__":
    _self_check()
    ALPHA = 0.10

    print("\n" + "=" * 78)
    print("PART A/B  sktime 1.1.0  ConformalIntervals  (alpha=0.10, requested 0.90)")
    print("=" * 78)
    for method in ("conformal", "empirical"):
        print(f"\nmethod={method!r}")
        print(f"{'n_resid':>8} {'coverage':>9} {'reps':>6}   mechanism (first rep)")
        for n_resid in (9, 15, 30):
            reps = 700
            cov, mech = sktime_probe(n_resid, ALPHA, reps, seed=4082026 + n_resid,
                                     method=method)
            n, sk, ok, k = mech[0]
            oks = "inf" if ok == np.inf else f"{ok:.4f}"
            print(f"{n_resid:>8} {cov:>9.4f} {reps:>6}   n={n} sktime_thr={sk:.4f} "
                  f"valid_thr={oks} (needs order stat k={k} of {n})")

    print("\n" + "=" * 78)
    print("PART C  river 0.25.0  RegressionJackknife  (alpha=0.10, requested 0.90)")
    print("=" * 78)
    print(f"{'n_cal':>6} {'coverage':>9} {'degenerate':>11} {'reps':>6}")
    for n_cal in (3, 5, 10, 20, 50, 200, 1000):
        reps = 4000
        cov, deg = river_probe(n_cal, ALPHA, reps, seed=4082026 + n_cal)
        print(f"{n_cal:>6} {cov:>9.4f} {deg:>11.2%} {reps:>6}")

    print("\nATTRIBUTION — mean interval endpoints, N(0,1) targets, truth = +/-1.6449")
    print(f"{'n_cal':>6} {'P2 (river)':>22} {'exact empirical':>22} {'valid k-th':>11}")
    for n_cal in (5, 10, 20, 50, 200, 1000):
        a = river_attribution(n_cal, ALPHA, 2000, seed=99 + n_cal)
        print(f"{n_cal:>6} {str('(%.3f, %.3f)' % a['p2']):>22} "
              f"{str('(%.3f, %.3f)' % a['empirical']):>22} {a['valid_upper']:>11.3f}")
