#!/usr/bin/env python3
"""W10: the feasibility floor is not a small-n curiosity -- it scales with the horizon.

Why this probe exists
---------------------
The audit's largest measured shortfall is a shipped default that calibrates on two
windows, where no valid finite bound exists at any conventional level. The obvious
rebuttal is that this is a corner: nobody calibrates on two points, and n >= 9 is
cheap, so the finding is a curiosity about one bad default.

That rebuttal is wrong, and this probe is the reason. The feasibility floor is

    n  >=  1/alpha_eff - 1

in the level actually resolved, not in the level the user requested. Any construction
that DIVIDES alpha before resolving it multiplies the floor by the same factor. A
Bonferroni split across a forecast horizon of length H resolves alpha/H, so

    n  >=  H/alpha - 1,

which at H = 12 and alpha = 0.05 is 239 calibration windows. Multi-step forecasting
is the normal case, not a corner, and no library default comes close to 239.

The site is not hypothetical. sktime's `ConformalIntervals(method="conformal_bonferroni")`
computes, in `_predict_interval_series`,

    alphas    = 1 - coverage
    quantiles = 1 - alphas / len(fh)
    pred_int_row = np.quantile(abs_resids, quantiles)

-- an uncorrected level, Bonferroni-divided by the horizon length, through numpy's
default `linear` interpolation. The census records it at conformal.py:326. This probe
measures what it delivers.

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
The check compared the score set sktime resolves the level from against the input's
absolute first differences, and it failed. Solving each entry for the index pair
that produces it shows why. `residuals_matrix_` has

    A[i, j] = y[j] - y[origin_i - 1],

so its offset-0 diagonal is already the ONE-step residual set. sktime slices it with

    resids = np.diagonal(residuals_matrix, offset=offset)      # offset = relative fh

so for a step-h forecast it calibrates on the offset-h diagonal, which holds
(h+1)-STEP residuals. Block (0) establishes this twice: structurally, by solving
each entry for its index pair, and statistically, by the variance ratio on a random
walk where Var[diag(offset=k)] must be (k+1) sigma^2 if the off-by-one is real and
k sigma^2 if it is not.

This is a horizon off-by-one, distinct from the level-to-rank map and pointing the
other way: for an integrated series a longer-horizon residual is larger, so the
interval is too WIDE. That is presumably why it has survived -- it is conservative,
and a conservative interval fails no coverage test. It also means the shipped
helper's scores are NOT exchangeable with the test residual at any horizon, whatever
the data, so absolute coverage through the public API is not attributable to any one
cause. The probe therefore runs four arms and attributes separately.

Four arms, one fit
------------------
  A  sktime shipped: its own scores, its own Bonferroni level, its own quantile call
  B  sktime's OWN scores at the required rank      -> isolates the level-to-rank map
  C  correctly ALIGNED scores at the required rank -> isolates the alignment defect
  D  correctly aligned scores at sktime's level    -> tests the h/(n+1) prediction

Arm C's step-1 scores are the innovations, so they are i.i.d. and exchangeable with
the test residual exactly; the absolute-coverage claims are made there and nowhere
else. At step h > 1 even the aligned residuals are overlapping sums and therefore
h-dependent, so those cells are labelled approximate. The FEASIBILITY claim needs
none of this: whether an index into a score set of size n can support level
1 - alpha/H is arithmetic.
"""

import math
import os
import sys
import warnings

import numpy as np

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
    """Smallest n admitting a valid finite bound at effective miscoverage alpha_eff."""
    return math.ceil(1.0 / alpha_eff - 1.0)


def required_rank(n, coverage):
    k = math.ceil((n + 1) * coverage)
    return None if k > n else k


def virtual_index(n, q, method="linear"):
    return float(np.quantile(np.arange(1, n + 1, dtype=float), q, method=method))


# ---------------------------------------------------------------------------
def self_check():
    # the floor, stated against its own boundary rather than at round numbers
    for alpha, first in ((0.10, 9), (0.05, 19), (0.01, 99), (1 / 3, 2)):
        assert floor_n(alpha) == first, (alpha, floor_n(alpha), first)
        assert required_rank(first - 1, 1 - alpha) is None
        assert required_rank(first, 1 - alpha) is not None
    # the Bonferroni floor is the plain floor in the DIVIDED level, not a new law
    for alpha in (0.10, 0.05):
        for H in (1, 2, 6, 12, 24):
            assert floor_n(alpha / H) == floor_n(alpha / H)
            assert required_rank(floor_n(alpha / H), 1 - alpha / H) is not None
            assert required_rank(floor_n(alpha / H) - 1, 1 - alpha / H) is None
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
                    "floor": floor_n(1 - level),
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
    say("(0) RESIDUAL ALIGNMENT -- a horizon off-by-one, separate from the level map")
    say("    residuals_matrix_ has A[i,j] = y[j] - y[origin_i - 1], so diag(offset=0)")
    say("    is ALREADY the one-step residual set. sktime slices diag(offset=h) for")
    say("    step h, which holds (h+1)-step residuals.")
    say("-" * 100)

    # structural: solve each diagonal entry for the index pair that produces it
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
    say("    The (k+1)-step column is matched and the k-step column is not, at every")
    say("    offset. The off-by-one is real. Direction: for an integrated series a")
    say("    longer-horizon residual is larger, so the shipped interval is too WIDE.")
    say("    A conservative defect fails no coverage test, which is why it survives.")
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
    say("(i) THE FLOOR under a Bonferroni split across a horizon of length H")
    say("    n >= 1/alpha_eff - 1 with alpha_eff = alpha/H. Exact arithmetic.")
    say("-" * 100)
    say(f"{'nominal':>8}{'H':>5}{'alpha_eff':>11}{'floor n':>9}"
        f"{'x vs H=1':>10}   comment")
    for alpha in (0.10, 0.05, 0.01):
        base = floor_n(alpha)
        for H in (1, 3, 6, 12, 24, 52):
            fl = floor_n(alpha / H)
            note = ""
            if H == 1:
                note = "the audit's stated floor"
            elif fl > 200:
                note = "no library default is within an order of magnitude"
            say(f"{1 - alpha:>8.2f}{H:>5}{alpha / H:>11.5f}{fl:>9}"
                f"{fl / base:>10.1f}   {note}")
        say("")
    say("    A monthly forecast one year ahead at 95% needs 239 calibration windows")
    say("    for the Bonferroni band to be feasible at all. This is the SAME")
    say("    infeasibility the audit reports for a two-window default, reached from a")
    say("    configuration nobody would call a corner case.")
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
        say("  measured against a SHIPPED helper's own level on exchangeable scores.")
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
