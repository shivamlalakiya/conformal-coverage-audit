#!/usr/bin/env python3
"""W10: the feasibility floor is not a small-n curiosity -- it scales with the horizon.

Why this probe exists
---------------------
The biggest shortfall measured anywhere in this artifact comes from a shipped default
calibrating on two windows, a size at which no finite bound is valid for any level
anyone requests. The easy dismissal: an edge case. Two points is nobody's calibration
set, nine is cheap, so one careless default and nothing more.

That rebuttal is wrong, and this probe is the reason. The feasibility floor is

    n  >=  1/alpha_eff - 1

in the level actually resolved, not in the level the user requested. Any construction
that DIVIDES alpha before resolving it multiplies the floor by the same factor. A
Bonferroni split across a forecast horizon of length H resolves alpha/H, so

    n  >=  H/alpha - 1,

which at H = 12 and alpha = 0.05 is 239 calibration windows. Multi-step forecasting
is the normal case, not a corner, and no library default comes close to 239.

There is a real site behind it. Inside `_predict_interval_series`, sktime's
`ConformalIntervals` under `method="conformal_bonferroni"` computes

    alphas    = 1 - coverage
    quantiles = 1 - alphas / len(fh)
    pred_int_row = np.quantile(abs_resids, quantiles)

-- so: no correction, the miscoverage cut by however many steps are forecast, and
numpy's default interpolation to resolve the result. conformal.py:326 in the census.
What that combination hands back is what this probe measures.

What is measured, and what is assumed
-------------------------------------
Nothing about sktime's internals is modelled. `np.quantile` is wrapped inside the
sktime module namespace, so the score-set SIZE and the LEVEL are recorded off the
library's own call rather than reconstructed from its documentation. That is the same
instrument the statsforecast arm uses, for the same reason: three of this programme's
retractions came from reasoning correctly about source that did something else.

Why the data are synthetic here, when the rest of the audit uses real series
---------------------------------------------------------------------------
Deliberately. The audit's real-data arms can only claim a PAIRED difference, because
real series are not exchangeable and an absolute coverage miss is therefore not
attributable to the convention. Here the generating process is chosen so that
exchangeability holds, which makes the ABSOLUTE coverage attributable and admits two
claims the real-data arms cannot support:

  * per-step marginal coverage against the nominal 1 - alpha/H, and
  * SIMULTANEOUS coverage over all H steps against the nominal 1 - alpha,

both against the h/(n+1) prediction of the fractional-rank probe.

The process is a RANDOM WALK, not i.i.d. noise, and the difference matters. A
last-value forecaster on i.i.d. data has residuals y_t - y_{t-1}, which are first
differences: identically distributed but 1-dependent, because adjacent residuals
share a value. The first version of this probe used i.i.d. data and would have
claimed exchangeability it did not have. On a random walk the same forecaster's
step-1 residuals ARE the innovations, so they are i.i.d.

A SECOND defect, found by the structural check that was written to verify the first
--------------------------------------------------------------------------------
The check pitted the score set sktime resolves against first differences of the
input, taken absolutely, and it failed. Reading entries back to the index pairs
behind them explains that. `residuals_matrix_` holds

    A[i, j] = y[j] - y[origin_i - 1],

whose main diagonal therefore holds ONE-step residuals already. The slice taken is

    resids = np.diagonal(residuals_matrix, offset=offset)      # offset = relative fh

so a forecast h steps out calibrates against diagonal h, and the entries there span
h+1 steps. Block (0) pins this down twice over: once by reading entries back to their
index pairs, once by variance. On a random walk diagonal k must carry (k+1) sigma^2
where the misalignment is genuine, and k sigma^2 where it is not.

That is an off-by-one in the horizon, a different fault from the index arithmetic and
pointing the opposite way: residuals reaching further ahead are bigger on an
integrated series, so the band comes out too WIDE. Which is likely why it has lasted.
Nothing that errs wide will fail a coverage test. It also puts the shipped scores out
of exchange with the test residual at every horizon, for any input, so no single cause
owns the absolute coverage seen through the public API. Four arms, attributed
separately, is the response.

Four arms, one fit
------------------
  A  sktime shipped: its own scores, its own Bonferroni level, its own quantile call
  B  sktime's OWN scores at the required rank      -> isolates the level-to-rank map
  C  scores on the RIGHT diagonal, at the required rank -> isolates the misalignment
  D  correctly aligned scores at sktime's level    -> tests the h/(n+1) prediction

At step 1 arm C is looking at the innovations themselves, so they are i.i.d. and
exchange with the test residual exactly; absolute coverage is claimed there only.
Beyond step 1 even a correctly aligned residual is a sum of overlaps and depends on
the step, and those cells are marked approximate. None of that bears on feasibility,
which is arithmetic: either some index into n scores reaches 1 - alpha/H or none
does.
"""

import math
import os
import sys
import warnings
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from conformal_coverage import required_rank  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_horizon_feasibility.txt")

SEED = 20260805
SERIES = 300
LINES = []
CALLS = []


def say(s=""):
    print(s)
    LINES.append(s)


def floor_n(alpha_eff):
    """Smallest n admitting a valid finite bound at effective miscoverage alpha_eff.

    Exact. Pass a Fraction; floats are converted through their shortest decimal
    form, which is what 0.10 and 0.05 mean here but is a guess in general -- a float
    cannot say whether it stands for 1/3 or for 0.3333333333333333.

    This used to be `math.ceil(1.0 / alpha_eff - 1.0)`, and the float subtraction
    cost a whole rank. Block (ii) recovered alpha_eff as `1 - level`, and
    `1 - 0.90` is 0.09999999999999998, whose floor genuinely IS 10 -- so the probe
    printed 10 in block (ii) against 9 in block (i) for the same quantity, and the
    two manuscripts' tables each picked up a different one. 0.95 happened to round
    the other way, which is why only some rows disagreed.
    """
    a = alpha_eff if isinstance(alpha_eff, Fraction) else Fraction(str(alpha_eff))
    return math.ceil(Fraction(1) / a - 1)


def alpha_eff_exact(coverage, H):
    """(1 - coverage)/H as a rational, so no caller has to subtract in floating point."""
    return (Fraction(1) - Fraction(str(coverage))) / H


def virtual_index(n, q, method="linear"):
    return float(np.quantile(np.arange(1, n + 1, dtype=float), q, method=method))


# ---------------------------------------------------------------------------
def self_check():
    # the floor, stated against its own boundary rather than at round numbers
    # Fraction(1, 3), not 1/3: the float is not one third, and the floor of the
    # float really is 3. Writing it exactly is the point of the exercise.
    for alpha, first in ((0.10, 9), (0.05, 19), (0.01, 99), (Fraction(1, 3), 2)):
        assert floor_n(alpha) == first, (alpha, floor_n(alpha), first)
        assert required_rank(first - 1, 1 - alpha) is None
        assert required_rank(first, 1 - alpha) is not None
    # the Bonferroni floor is the plain floor in the DIVIDED level, not a new law
    for alpha in (0.10, 0.05):
        for H in (1, 2, 6, 12, 24):
            fl = floor_n(Fraction(str(alpha)) / H)
            # The floor recovered from the LEVEL must equal the floor computed from
            # alpha. This assertion used to read `floor_n(alpha / H) ==
            # floor_n(alpha / H)` -- both sides identical, so it could not fail, and
            # the +1 it exists to catch shipped underneath it for that reason. A
            # probe about tests that cannot catch anything had one of its own.
            level = Fraction(1) - Fraction(str(alpha)) / H
            assert floor_n(Fraction(1) - level) == fl, (alpha, H, fl)
            assert required_rank(fl, 1 - alpha / H) is not None
            assert required_rank(fl - 1, 1 - alpha / H) is None
    # H=1 must reduce to the ordinary floor, or the parameterisation is wrong
    assert floor_n(0.10 / 1) == 9 and floor_n(0.05 / 1) == 19
    # and the divided level goes through `linear`, which never lands on an integer
    # rank at these levels -- the rank map's result, re-checked here so this probe
    # does not silently assume it
    for H in (6, 12):
        for n in (50, 119, 239, 500):
            h = virtual_index(n, 1 - 0.05 / H)
            assert abs(h - round(h)) > 1e-9 or n in (), (H, n, h)


self_check()


# ---------------------------------------------------------------------------
def install_spy():
    """Record (size, level) of every quantile call sktime's conformal module makes."""
    import sktime.forecasting.conformal as mod

    real = mod.np.quantile

    def spy(a, q, *args, **kw):
        arr = np.asarray(a, dtype=float)
        CALLS.append((int(arr.size),
                      np.atleast_1d(np.asarray(q, dtype=float)).tolist(),
                      kw.get("method", "linear"), np.sort(arr.ravel())))
        return real(a, q, *args, **kw)

    mod.np.quantile = spy


def run_cell(rng, H, initial_window, n_cal_target, coverages, n_series=SERIES):
    """Four arms off ONE fit. See the module docstring for what each isolates.

    One fit serves every requested coverage: sktime builds the rolling-origin
    residual matrix in fit(), which dominates the cost and does not depend on the
    level. Refitting per level doubled the runtime for nothing.
    """
    import pandas as pd
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.conformal import ConformalIntervals
    from sktime.forecasting.naive import NaiveForecaster

    K, ARMS = len(coverages), ("A", "B", "C", "D")
    step1 = {a: np.zeros(K) for a in ARMS}
    sim = {a: np.zeros(K) for a in ARMS}
    finite = {a: np.zeros(K) for a in ARMS}
    used, aligned_n, levels = [], [], [[] for _ in range(K)]
    total = 0
    # the calibration count is set by (length - initial_window); parameterise by
    # the count wanted rather than by the window, or cells intended to differ in
    # n come out identical
    length = initial_window + n_cal_target + 1
    checked = [False]

    for _ in range(n_series):
        # random walk: the correctly aligned step-1 residuals are then the
        # innovations, i.i.d. and exchangeable with the test residual
        innov = rng.standard_normal(length + H)
        vals = np.cumsum(innov)
        y = pd.Series(vals[:length], index=pd.RangeIndex(length))
        future = vals[length:length + H]
        fh = ForecastingHorizon(list(range(1, H + 1)), is_relative=True)
        CALLS.clear()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ci = ConformalIntervals(NaiveForecaster(strategy="last"),
                                        method="conformal_bonferroni",
                                        initial_window=initial_window)
                ci.fit(y, fh=fh)
                iv = ci.predict_interval(fh=fh, coverage=list(coverages))
                point = np.asarray(ci.predict(fh=fh), dtype=float).ravel()
                M = ci.residuals_matrix_.to_numpy()
        except Exception as exc:
            return {"error": type(exc).__name__}
        if not CALLS:
            return {"error": "no_quantile_call"}
        arr = np.asarray(iv, dtype=float)
        if arr.shape != (H, 2 * K) or point.size != H:
            return {"error": f"shape {arr.shape}, point {point.size}"}

        # sktime issues one quantile call per horizon step, each with the same
        # vector, and repeats each coverage once per RAIL. Indexing the
        # concatenation of all calls read rail two of coverage one as coverage
        # two's level, which mislabelled an infeasible cell as feasible.
        first = CALLS[0][1]
        rails = len(first) // K
        assert rails in (1, 2), f"{len(first)} levels per call for {K} coverages"
        for c in CALLS:
            assert np.allclose(c[1], first), "per-step levels differ; not Bonferroni"
        lv = []
        for j, coverage in enumerate(coverages):
            q = first[j * rails]
            want = 1.0 - (1.0 - coverage) / H
            assert abs(q - want) < 1e-9, (
                f"level {q} but documented rule 1-(1-{coverage})/{H} = {want}")
            levels[j].append(q)
            lv.append(q)

        # sktime's own score set per step: diag(offset=h). Correctly ALIGNED set:
        # diag(offset=h-1) -- see block (0).
        shipped, alignedd = [], []
        for h in range(1, H + 1):
            for off, acc in ((h, shipped), (h - 1, alignedd)):
                d = np.diagonal(M, offset=off)
                acc.append(np.abs(d[~np.isnan(d)]))
        if not checked[0]:
            # STRUCTURAL: the aligned step-1 set must be the absolute innovations
            g = np.sort(alignedd[0])
            w = np.sort(np.abs(np.diff(vals[:length])))
            dist = np.abs(g[:, None] - w[None, :]).min(axis=1)
            assert dist.max() < 1e-9, (
                "aligned step-1 scores are not the input's absolute first "
                f"differences (max distance {dist.max():.2e}); exchangeability "
                "is not established")
            # and sktime's own set must NOT be, which is the off-by-one
            g2 = np.sort(shipped[0])
            d2 = np.abs(g2[:, None] - w[None, :]).min(axis=1)
            assert d2.max() > 1e-9, "shipped step-1 scores match the innovations"
            checked[0] = True

        used.append(min(a.size for a in shipped))
        aligned_n.append(min(a.size for a in alignedd))

        for j, coverage in enumerate(coverages):
            for arm in ARMS:
                lo = np.empty(H)
                hi = np.empty(H)
                for h in range(H):
                    if arm == "A":
                        lo[h], hi[h] = arr[h, 2 * j], arr[h, 2 * j + 1]
                        continue
                    sc = shipped[h] if arm == "B" else alignedd[h]
                    n = sc.size
                    if arm == "D":
                        w = float(np.quantile(sc, lv[j], method="linear"))
                    else:
                        k = required_rank(n, lv[j])
                        w = np.inf if k is None else float(np.sort(sc)[k - 1])
                    lo[h], hi[h] = point[h] - w, point[h] + w
                inside = (future >= lo) & (future <= hi)
                step1[arm][j] += bool(inside[0])
                sim[arm][j] += bool(inside.all())
                finite[arm][j] += bool(np.isfinite(hi).all())
        total += 1

    n_cal, n_al = int(np.median(used)), int(np.median(aligned_n))
    out = []
    for j, coverage in enumerate(coverages):
        level = float(np.median(levels[j]))
        k = required_rank(n_al, level)
        h = virtual_index(n_al, level)
        out.append({"H": H, "n_cal": n_cal, "n_aligned": n_al, "level": level,
                    "total": total, "coverage": coverage, "req_rank": k,
                    "h": h, "pred": h / (n_al + 1),
                    "feasible": k is not None,
                    # from the nominal coverage and H, not from `level` -- `level` is
                    # a median of measured levels and 1 - level costs a rank
                    "floor": floor_n(alpha_eff_exact(coverage, H)),
                    **{f"step1_{a}": step1[a][j] / total for a in ARMS},
                    **{f"sim_{a}": sim[a][j] / total for a in ARMS},
                    **{f"finite_{a}": finite[a][j] / total for a in ARMS}})
    return {"cells": out}


def alignment_block(rng, say, reps=300, L=80):
    """Establish the residual-alignment off-by-one, structurally and by variance."""
    import pandas as pd
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.conformal import ConformalIntervals
    from sktime.forecasting.naive import NaiveForecaster

    say("-" * 100)
    say("(0) RESIDUAL ALIGNMENT -- an off-by-one in the horizon, not in the level")
    say("    residuals_matrix_ has A[i,j] = y[j] - y[origin_i - 1], which puts")
    say("    one-step residuals on the main diagonal already. The slice taken for")
    say("    step h is diag(offset=h), whose entries span h+1 steps.")
    say("-" * 100)

    # structural: read each diagonal entry back to the indices behind it
    vals = np.cumsum(rng.standard_normal(L))
    y = pd.Series(vals, index=pd.RangeIndex(L))
    fh = ForecastingHorizon([1, 2], is_relative=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = ConformalIntervals(NaiveForecaster(strategy="last"),
                                method="conformal", initial_window=30)
        ci.fit(y, fh=fh)
    A = ci.residuals_matrix_.to_numpy()
    origins = list(ci.residuals_matrix_.index)
    say("    structural, first four entries of each diagonal:")
    for off in (0, 1, 2):
        d = np.diagonal(A, offset=off)
        d = d[~np.isnan(d)]
        pairs = []
        for k in range(min(4, d.size)):
            hit = [(pp, qq) for pp in range(L) for qq in range(L)
                   if abs((vals[pp] - vals[qq]) - d[k]) < 1e-9]
            pairs.append(hit[0] if hit else None)
        spans = [pp - qq for (pp, qq) in pairs if pairs and pairs[0] is not None]
        say(f"      diag(offset={off}): index pairs {pairs}   spans {spans}"
            f"   -> {off + 1}-step residuals")
        assert all(sp == off + 1 for sp in spans), (off, spans)

    # statistical: on a random walk the variance must be (offset+1) sigma^2
    var = {0: [], 1: [], 2: []}
    for _ in range(reps):
        v = np.cumsum(rng.standard_normal(L))
        yy = pd.Series(v, index=pd.RangeIndex(L))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c2 = ConformalIntervals(NaiveForecaster(strategy="last"),
                                    method="conformal", initial_window=30)
            c2.fit(yy, fh=fh)
        M = c2.residuals_matrix_.to_numpy()
        for off in var:
            d = np.diagonal(M, offset=off)
            d = d[~np.isnan(d)]
            var[off].append(float(np.var(d)))
    say("")
    say(f"    statistical, random walk with unit innovation variance, {reps} series:")
    say(f"{'diagonal':>18}{'mean Var':>12}{'if (k+1)-step':>15}{'if k-step':>12}")
    for off in sorted(var):
        mv = float(np.mean(var[off]))
        say(f"{'diag(offset=' + str(off) + ')':>18}{mv:>12.3f}"
            f"{off + 1:>15}{(off if off else 0):>12}")
        assert abs(mv - (off + 1)) < 0.25, (off, mv)
    say("")
    say("    Every offset matches the (k+1)-step column and misses the k-step one.")
    say("    The misalignment is real. Which way it errs: on an integrated series a")
    say("    residual reaching further ahead is bigger, so the shipped band is too")
    say("    WIDE -- and nothing erring wide will fail a coverage test.")
    say("")
    return {"var": {k: float(np.mean(v)) for k, v in var.items()}}


def main():
    rng = np.random.default_rng(SEED)
    install_spy()

    say("=" * 100)
    say("W10  HORIZON FEASIBILITY -- the floor scales with what you divide alpha by")
    say("=" * 100)
    say("")
    alignment_block(rng, say)

    # ---------------- (i) exact arithmetic --------------------------------
    say("-" * 100)
    say("(i) THE FLOOR once alpha is cut H ways, one per forecast step")
    say("    n >= 1/alpha_eff - 1 with alpha_eff = alpha/H. Exact arithmetic.")
    say("-" * 100)
    say(f"{'nominal':>8}{'H':>5}{'alpha_eff':>11}{'floor n':>9}"
        f"{'x vs H=1':>10}   comment")
    for alpha in (0.10, 0.05, 0.01):
        base = floor_n(alpha)
        for H in (1, 3, 6, 12, 24, 52):
            fl = floor_n(Fraction(str(alpha)) / H)
            note = ""
            if H == 1:
                note = "the audit's stated floor"
            elif fl > 200:
                note = "no library default is within an order of magnitude"
            say(f"{1 - alpha:>8.2f}{H:>5}{alpha / H:>11.5f}{fl:>9}"
                f"{fl / base:>10.1f}   {note}")
        say("")
    say("    Twelve months ahead at 95% wants 239 calibration windows before the")
    say("    band is feasible at all. Same wall the two-window default runs into,")
    say("    arrived at from a setting nobody would file under edge case.")
    say("")

    # ---------------- (ii) four arms ------------------------------------
    say("-" * 100)
    say("(ii) FOUR ARMS on sktime ConformalIntervals(method='conformal_bonferroni')")
    say("     A shipped   B own scores at required rank   C aligned scores at")
    say("     required rank   D aligned scores at sktime's level")
    say("     step1 is the step-1 rate, where arm C's scores are the innovations and")
    say("     exchangeability is EXACT. sim is the all-H simultaneous rate, which is")
    say(f"     what a Bonferroni band targets. {SERIES} series per cell.")
    say("-" * 100)
    say(f"{'nom':>5}{'H':>4}{'n':>5}{'level':>9}{'floor':>7}{'feas':>6}"
        f"{'h/(n+1)':>9}{'A':>8}{'B':>8}{'C':>8}{'D':>8}"
        f"{'simA':>8}{'simC':>8}{'finA':>6}")
    rows = []
    for H, target in ((1, 14), (1, 60), (6, 30), (6, 140),
                      (12, 48), (12, 260), (24, 90)):
        res = run_cell(rng, H, 30, target, (0.90, 0.95))
        if "error" in res:
            say(f"{H:>4}   ERROR {res['error']}")
            continue
        for r in res["cells"]:
            rows.append(r)
            say(f"{r['coverage']:>5.2f}{r['H']:>4}{r['n_aligned']:>5}"
                f"{r['level']:>9.5f}{r['floor']:>7}"
                f"{('yes' if r['feasible'] else 'NO'):>6}{r['pred']:>9.4f}"
                f"{r['step1_A']:>8.4f}{r['step1_B']:>8.4f}{r['step1_C']:>8.4f}"
                f"{r['step1_D']:>8.4f}{r['sim_A']:>8.4f}{r['sim_C']:>8.4f}"
                f"{r['finite_A']:>6.2f}")
        say("")

    # ---------------- (iii) three failures, separated --------------------
    say("-" * 100)
    say("(iii) THREE DISTINCT FAILURES, separated")
    say("-" * 100)
    infeas = [r for r in rows if not r["feasible"]]
    feas = [r for r in rows if r["feasible"]]
    say(f"  1. INFEASIBILITY ({len(infeas)} of {len(rows)} cells). n below")
    say("     H/alpha - 1, so no index into the score set supports the divided")
    say("     level. Arithmetic, needs no exchangeability. A finite interval is")
    say("     returned regardless:")
    for r in infeas:
        say(f"       nominal {r['coverage']:.2f} H={r['H']:<3} n={r['n_aligned']:<4}"
            f" needs {r['floor']:<4} simultaneous(A) {r['sim_A']:.4f}"
            f"  finite(A) {r['finite_A']:.2f}")
    say("")
    say("  2. ALIGNMENT (arm A vs arm C at step 1, feasible cells). sktime's scores")
    say("     are (h+1)-step, so they are inflated and the interval is too wide:")
    for r in feas:
        say(f"       nominal {r['coverage']:.2f} H={r['H']:<3} A {r['step1_A']:.4f}"
            f"   C {r['step1_C']:.4f}   A-C {r['step1_A'] - r['step1_C']:+.4f}")
    say("")
    say("  3. LEVEL-TO-RANK (arm D vs arm C at step 1, feasible cells). Same scores,")
    say("     same alignment; D resolves sktime's level through `linear`, C takes the")
    say("     required rank. D should sit at h/(n+1):")
    for r in feas:
        say(f"       nominal {r['coverage']:.2f} H={r['H']:<3} predicted"
            f" {r['pred']:.4f}   D {r['step1_D']:.4f}"
            f"   gap {r['step1_D'] - r['pred']:+.4f}   C {r['step1_C']:.4f}")
    say("")
    if feas:
        say(f"  The h/(n+1) prediction of the fractional-rank result tracks arm D to"
            f" within {max(abs(r['step1_D'] - r['pred']) for r in feas):.4f} on the"
            f" feasible cells,")
        say("  measured on scores that do exchange, at a level the library chose.")
    say("")
    say("=" * 100)
    say("SUMMARY")
    say("=" * 100)
    say("  The feasibility floor is a property of the level RESOLVED, not the level")
    say("  REQUESTED. Any construction that divides alpha -- Bonferroni over a")
    say("  horizon, over output dimensions, over groups -- multiplies the floor.")
    say("  At H=12, alpha=0.05 the floor is 239 calibration windows; the audited")
    say("  default is 60 or fewer. The audit's two-window finding is the H=1 corner")
    say("  of this, not a curiosity.")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
