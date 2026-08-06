#!/usr/bin/env python3
"""Remove the convenience-sample hedge by measuring it instead of stating it.

The forecasting arms in `run_real_data.py`, `run_real_data_statsforecast.py` and
`run_real_data_darts.py` take the first 250 series above a length floor, in
archive order, and give each series one test point at one horizon step. Two
objections follow, and neither is answerable by argument:

  (a) "you chose the series"    -- archive order is a convenience sample
  (b) "one point per series"    -- caps the resolution on the small deltas

This probe answers both by running the same cells on **every eligible series in
the archive** (no sample, so there is no selection left to object to) with a
**rolling origin** (K test points per series, K distinct fits). It does not
replace the committed arms and does not touch their code: their outputs stay
byte-reproducible, and this is a separate measurement that either agrees with
them or does not.

Why a rolling origin needs no change to any arm
-----------------------------------------------
All three arms share one contract: they take a 1-D array, fit on `s[:-1]` and
test on `s[-1]`. So origin `j` is just the truncation `s[:len(s)-j]` -- a
shorter history and an earlier test point, through the arm's own code path,
unmodified. That is the whole trick, and it is why this file adds no forecasting
logic of its own.

Why the standard error must be clustered
----------------------------------------
The K origins of one series share almost all of their history, so their paired
outcomes are not independent. Treating 4372 origin-points as 4372 independent
draws would understate the standard error and manufacture significance. Every
cell here is aggregated **series first**: the K origin-level paired differences
of a series are averaged into one series-level difference, and the standard
error is taken across series. The naive unclustered figure is printed beside it
so the size of that mistake is visible rather than asserted.

Length floor, and what it excludes
----------------------------------
One floor is used for all three arms so the comparison is between arms and not
between eligibility rules. It is set by the strictest requirement, the darts
arm's 70-point cache floor, plus K-1 for the rolling origin: 73 points. That
excludes 335 of the archive's 1428 monthly series as too short, reported here
rather than dropped silently.

    .venv-probe/bin/python probes/sample_robustness.py
    # the darts arm runs in its own venv, on the same series, single origin:
    .venv-probe/bin/python probes/sample_robustness.py --export-npz /tmp/m3_full.npz
    ../.venv-darts/bin/python probes/run_real_data_darts.py /tmp/m3_full.npz
"""

import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_real_data as SK  # noqa: E402
import run_real_data_statsforecast as SF  # noqa: E402

DATASET = "m3_monthly_dataset"
ORIGINS = 4                      # test points per series; K in the docstring
MIN_LEN = 73                     # darts' 70-point floor + (ORIGINS - 1)
SK_WINDOWS = (20, 40)            # identical to the committed sktime arm
OUT = "outputs/probe_output_sample_robustness.txt"


# --------------------------------------------------------------------------
# the rolling origin, and the aggregation that has to go with it
# --------------------------------------------------------------------------
def origins(s, k):
    """The k rolling origins of one series, longest history first.

    Origin j has history s[:len(s)-1-j] and test point s[len(s)-1-j].
    """
    return [s[: s.size - j] for j in range(k)]


def clustered(per_series):
    """Mean and standard error of a paired difference, clustered by series.

    `per_series` is one mean paired difference per series. Independence is
    claimed at the series level only, which is the level at which it holds.
    """
    a = np.asarray(per_series, dtype=float)
    if a.size < 2:
        return float(a.mean()) if a.size else float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(a.size))


def self_check():
    s = np.arange(10.0)

    # the truncations are prefixes, one step shorter each, and origin j's test
    # point is the archive's (len-1-j)th observation
    os_ = origins(s, 4)
    assert [o.size for o in os_] == [10, 9, 8, 7], [o.size for o in os_]
    for j, o in enumerate(os_):
        assert o[-1] == s[s.size - 1 - j], (j, o[-1])
        assert np.array_equal(o[:-1], s[: s.size - 1 - j])
    assert os_[0].size - os_[-1].size == 3

    # every origin of the shortest admissible series still clears both arms'
    # own floors, so the floor is not doing silent filtering inside a cell
    shortest_hist = MIN_LEN - (ORIGINS - 1) - 1
    assert shortest_hist >= max(SK_WINDOWS) + 4, shortest_hist
    assert shortest_hist >= max(SF.N_WINDOWS) + 6, shortest_hist

    # clustering with one origin per series must reproduce the plain paired s.e.
    d = [0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    m, se = clustered(d)
    plain = np.array(d)
    assert abs(m - plain.mean()) < 1e-12
    assert abs(se - plain.std(ddof=1) / math.sqrt(plain.size)) < 1e-12

    # and clustering must WIDEN, not narrow, when origins are correlated:
    # two series, 3 perfectly-correlated origins each
    per_origin = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    naive = np.array(per_origin).std(ddof=1) / math.sqrt(6)
    _, cl = clustered([1.0, 0.0])
    assert cl > naive, (cl, naive)

    # exact arithmetic on the identity the arms' required rank uses
    from fractions import Fraction
    for n in (10, 19, 20, 47, 100):
        for cov in (0.90, 0.95):
            k = SK.required_rank(n, cov)
            if k is not None:
                assert Fraction(k, n + 1) >= Fraction(cov).limit_denominator(100)
                assert Fraction(k - 1, n + 1) < Fraction(cov).limit_denominator(100)


self_check()


# --------------------------------------------------------------------------
# data: every eligible series, which is the point
# --------------------------------------------------------------------------
def eligible_pool():
    from sktime.datasets import load_forecastingdata

    df, meta = load_forecastingdata(DATASET)
    kept, total = [], 0
    for values in df["series_value"]:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        total += 1
        if arr.size >= MIN_LEN:
            kept.append(arr)
    return kept, total, meta


# --------------------------------------------------------------------------
# one series, all its origins, all sktime cells -- the parallel unit
# --------------------------------------------------------------------------
def sktime_series(s):
    """Every sktime cell for one series, over its origins.

    `run_cells` fits once per (origin, window, method) and reads every level off
    that fit, because ConformalIntervals takes the coverage at predict time and
    not at fit time. Refitting per level did the same sliding-residual work twice
    and made this the longest probe in the deposit; the numbers are identical
    either way and run_real_data.py's committed output is the regression that
    holds it to that.
    """
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("empirical", "conformal"):
            for iw in SK_WINDOWS:
                per = [SK.run_cells(o, iw, method, SK.LEVELS)
                       for o in origins(s, ORIGINS)]
                for cov in SK.LEVELS:
                    out[(method, cov, iw)] = [
                        p[cov] for p in per if p[cov] and "error" not in p[cov]
                    ]
    return out


def statsforecast_series(s):
    out = {}
    for method in SF.METHODS:
        for level in SF.LEVELS:
            for nw in SF.N_WINDOWS:
                recs = [SF.run_cell(o, nw, level, method) for o in origins(s, ORIGINS)]
                out[(method, level, nw)] = [r for r in recs if r and "error" not in r]
    return out


def fold(per_series_cells, key):
    """Collapse one cell across series into the reported record."""
    deltas, a_cov, b_cov, ns, a_rank, req, infeas, points = [], [], [], [], [], [], 0, 0
    for cells in per_series_cells:
        recs = cells.get(key, [])
        if not recs:
            continue
        d = [float(r["b_covered"]) - float(r["a_covered"]) for r in recs]
        deltas.append(float(np.mean(d)))
        points += len(recs)
        for r in recs:
            a_cov.append(float(r["a_covered"]))
            b_cov.append(float(r["b_covered"]))
            ns.append(r["n"])
            a_rank.append(r["a_rank"])
            if r["feasible"]:
                req.append(r["required_rank"])
            else:
                infeas += 1
    if not deltas:
        return None
    mean, se = clustered(deltas)
    flat = np.array([
        float(r["b_covered"]) - float(r["a_covered"])
        for cells in per_series_cells for r in cells.get(key, [])
    ])
    se_naive = (flat.std(ddof=1) / math.sqrt(flat.size)) if flat.size > 1 else float("nan")
    # Counted, not read off the delta. A clustered mean of per-series means is not
    # even delta x points, so the count has to come from the points themselves --
    # and a single loss where arm B contains arm A means the two arms are not the
    # constructions this probe thinks they are.
    gains, losses = int(np.sum(flat > 0)), int(np.sum(flat < 0))
    nests = all(r.get("nests", True) for cells in per_series_cells
                for r in cells.get(key, []))
    assert not (nests and losses), (
        f"{key}: {losses} of {flat.size} points lost coverage under arm B while arm B "
        f"is meant to contain arm A")
    two_rail = all(r.get("two_rail", False) for cells in per_series_cells
                   for r in cells.get(key, []))
    return {
        "series": len(deltas), "points": points,
        "n_med": int(np.median(ns)),
        "a_cov": float(np.mean(a_cov)), "b_cov": float(np.mean(b_cov)),
        "delta": mean, "se": se, "se_naive": float(se_naive),
        "a_rank_med": int(np.median(a_rank)),
        "req_med": int(np.median(req)) if req else 0,
        "gains": gains, "losses": losses, "changed": gains + losses,
        "unit": "span" if two_rail else "rank",
        "infeasible": infeas,
    }


def cell_line(prefix, r):
    """One self-describing CELL line. Written once because the sktime and the
    statsforecast blocks carried separate copies of it, and a field added to one
    copy is a field the other silently lacks."""
    if r is None:
        return f"{prefix} -- no usable cells"
    ratio = (r["se"] / r["se_naive"]) if r["se_naive"] > 0 else 0
    return (f"{prefix} series={r['series']} points={r['points']} n_med={r['n_med']} "
            f"a_cov={r['a_cov']:.4f} b_cov={r['b_cov']:.4f} "
            f"delta={r['delta']:+.4f} se={r['se']:.4f} se_naive={r['se_naive']:.4f} "
            f"se_ratio={ratio:.4f} "
            f"a_rank_med={r['a_rank_med']} req_med={r['req_med']} unit={r['unit']} "
            f"gains={r['gains']} losses={r['losses']} changed={r['changed']} "
            f"infeasible={r['infeasible']}")


def main():
    export = None
    if "--export-npz" in sys.argv:
        export = sys.argv[sys.argv.index("--export-npz") + 1]

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    pool, total, meta = eligible_pool()

    if export:
        np.savez_compressed(
            export,
            meta=np.array([f"{DATASET.replace('_dataset', '')}_full_dataset",
                           str(meta.get("frequency", "?")), str(MIN_LEN)]),
            **{f"s{i}": s for i, s in enumerate(pool)},
        )
        print(f"{len(pool)} series -> {export}")
        return 0

    say("=" * 104)
    say("SAMPLE ROBUSTNESS -- the whole eligible archive, with a rolling origin")
    say("=" * 104)
    say("self_check() passed at import")
    say("")
    say(f"dataset: {DATASET}")
    say(f"series in archive: {total}   eligible at min_len={MIN_LEN}: {len(pool)}"
        f"   excluded as too short: {total - len(pool)}")
    say(f"selection: NONE -- every eligible series is used, so archive order cannot")
    say(f"           bias the result and there is no sample to have chosen")
    say(f"rolling origin: {ORIGINS} test points per series, {ORIGINS} distinct fits")
    say(f"           origin j uses history s[:len(s)-1-j] and tests on s[len(s)-1-j]")
    say(f"test points per cell: up to {len(pool) * ORIGINS}"
        f"   (committed arms: 250, one per series, archive order)")
    say("")
    lens = np.array([s.size for s in pool])
    say(f"eligible length min/median/max: {lens.min()} / {int(np.median(lens))} / {lens.max()}")
    say("")
    say("Standard errors are CLUSTERED BY SERIES: the rolling origins of one series")
    say("share their history, so each series contributes one mean paired difference")
    say("and the s.e. is taken across series. se_naive treats every origin as an")
    say("independent draw; it is printed so the dependence is visible rather than")
    say("assumed. se exceeds se_naive where origins within a series are positively")
    say("correlated and falls below it where they are not, so se_ratio goes BOTH ways")
    say("and this output used to claim the two coincide off the positive side. They do")
    say("not: read se_ratio per cell. The sharp narrowing sits in the cells where arm")
    say("B is vacuous at every point, where the paired difference is 1 minus a")
    say("coverage indicator rather than a comparison of two index rules.")
    say("")
    say("req_med=0 marks a cell with no feasible required rank at all -- read it with")
    say("the infeasible count on the same line, never as a rank of zero.")
    say("")
    say("unit=rank for a symmetric band on absolute scores; unit=span for a helper")
    say("resolving TWO levels on signed ones, whose arm B is the index pair spanning")
    say("the required number of gaps. Half of an asymmetric width is not an order")
    say("statistic of anything, so the two figures are not interchangeable.")
    say("gains/losses/changed are counted over the test points. `changed` is NOT")
    say("delta x points: a clustered mean of per-series means does not equal that, and")
    say("gains - losses is the net rather than the count. losses must read 0 wherever")
    say("arm B contains arm A, which each arm asserts per fit.")
    say("")

    # ---- sktime arm, parallel across series -------------------------------
    say("-" * 104)
    say("(i) sktime ConformalIntervals -- arm A shipped, arm B its own residuals at the required rank")
    say("-" * 104)
    # Each worker holds a fitted sktime object plus its residual matrix, so the
    # pool's footprint scales with the count. Overlapping this run with another
    # probe exhausted swap and the OS killed both, silently, mid-cell. PROBE_WORKERS
    # caps it; the default leaves four cores and is what a dedicated run wants.
    workers = int(os.environ.get("PROBE_WORKERS") or max(1, (os.cpu_count() or 4) - 4))
    assert workers >= 1, f"PROBE_WORKERS={workers} is not a usable pool size"
    with ProcessPoolExecutor(max_workers=workers) as ex:
        sk_cells = list(ex.map(sktime_series, pool, chunksize=8))

    for method in ("empirical", "conformal"):
        for cov in SK.LEVELS:
            for iw in SK_WINDOWS:
                r = fold(sk_cells, (method, cov, iw))
                say(cell_line(f"CELL arm=sktime method={method} level={cov:.2f} "
                              f"window={iw}", r))

    # ---- statsforecast arm ----------------------------------------------
    say("")
    say("-" * 104)
    say("(ii) statsforecast ConformalIntervals -- same construction, its own captured scores")
    say("-" * 104)
    SF.install_spy()
    sf_cells = [statsforecast_series(s) for s in pool]
    for method in SF.METHODS:
        for level in SF.LEVELS:
            for nw in SF.N_WINDOWS:
                r = fold(sf_cells, (method, level, nw))
                say(cell_line(f"CELL arm=statsforecast method={method} "
                              f"level={level / 100:.2f} n_windows={nw}", r))

    say("")
    say("A positive delta means the required rank covers more than the shipped call.")
    say("Real series are not exchangeable, so an absolute coverage figure is still not")
    say("attributable to the convention -- the paired delta is what carries the claim.")
    say("What this probe removes is the selection and resolution objections to that")
    say("delta, not the exchangeability caveat, which no amount of data can remove.")
    say("")
    say("The darts arm runs the same series in its own venv at a single origin, so its")
    say("standard error needs no clustering; see probe_output_real_data_darts_m3_full.txt.")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), OUT)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
