#!/usr/bin/env python3
"""PR1-17: one row per test point for the statsforecast distribution method, so
the paired delta of Table 4 is attributed rather than declined.

Table 4 reported a predicted delta near zero beside a measured one two orders of
magnitude larger, at every feasible statsforecast cell. Two candidate causes: the
archive is not exchangeable and the prediction cannot apply, or the predicted
column is wrong. This probe separates them by instrumenting the unit.

Three sections, each executed:

  (1) the library expression, read out of the installed file at run time with its
      line numbers, so the mechanism below is quoted and not remembered.

  (2) the index arithmetic, checked by calling `np.quantile` on a marker set whose
      values encode their own position. `_add_conformal_distribution_intervals`
      stacks `mean - cs` and `mean + cs`, so the array it hands `np.quantile` has
      2n rows for n calibration windows, and `cs` is already absolute
      (`models.py:_conformity_scores`, `np.abs`). Under the default
      `interpolation='linear'` the upper cut 1 - alpha/200 lands at 1-based
      position 1 + (1 - alpha/200)(2n - 1) of the stacked set. Above position n
      that set is `mean` plus the ascending absolute scores, so the half-width
      lands at position 1 + (1 - alpha/200)(2n - 1) - n among the n scores. There
      is nowhere an (n + 1) factor in what the library computes.

  (3) the ledger. For each cell: T_A (the shipped half-width), T_B (the score at
      the required rank), the position T_A lands on, its floor, the required rank,
      the absolute test score, and whether that score fell in (T_A, T_B]. The
      delta is then a count of units in that half-open gap, which is checked
      against the mean of the per-unit difference.

Two predicted arm A coverages are printed for every cell and neither is derived
from the other. The floor one, floor(position)/(n + 1), is what the rank law gives
with no assumption on the score distribution: a threshold at or above the
floor(position)-th score covers at least that often. The interpolated one,
position/(n + 1), reads the fractional position through the same law and is exact
only for uniform scores. The delta each implies is printed beside it.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_report import format_cell, landed, summarize  # noqa: E402
from run_real_data import rank_of, required_rank  # noqa: E402

LEVEL = 90
METHOD = "conformal_distribution"
N_WINDOWS = (2, 5, 10, 20, 50)
LEDGER_CELL = 10
LEDGER_ROWS = 15
SWEEP_MAX = 400
OUT = "outputs/probe_output_sf_unit_ledger.txt"

CAPTURED = []


def install_spy():
    import statsforecast.models as M

    fn = M._add_conformal_distribution_intervals

    def inner(fcst, cs, level):
        CAPTURED.append((np.array(cs, copy=True),
                         np.array(fcst["mean"], copy=True)))
        return fn(fcst=fcst, cs=cs, level=level)

    M._add_conformal_distribution_intervals = inner


def source_lines():
    """The expression, with the line numbers of the installed copy."""
    import statsforecast
    import statsforecast.models as M

    path = M.__file__
    with open(path) as fh:
        src = fh.readlines()
    first = next(i for i, ln in enumerate(src)
                 if ln.startswith("def _add_conformal_distribution_intervals"))
    end = next(i for i in range(first, len(src))
               if src[i].strip() == "return fcst")
    body = [(i + 1, src[i].rstrip()) for i in range(first, end + 1)]
    scores = next(i for i, ln in body if ln.strip().startswith("scores = np.vstack"))
    quant = next(i for i, ln in body if ln.strip().startswith("quantiles = np.quantile"))
    # If either of these two lines is gone the arithmetic in section (2) is about
    # a version that is no longer installed, and every number below is void.
    assert "mean - cs" in dict(body)[scores] and "mean + cs" in dict(body)[scores]
    return statsforecast.__version__, path, body, scores, quant


def landed_position(n, level, cuts=None):
    """Executed, not asserted: which position of the n scores the cut lands on.

    Builds a marker score set 1..n, runs the library's own stack-and-quantile on
    it, and inverts the result through the marker values. The markers are equally
    spaced, so linear interpolation between neighbours returns the position itself
    and the return value IS the 1-based fractional position among the n scores.
    """
    alpha = 100 - level
    cut = 1 - alpha / 200
    mean = np.zeros((1, 1))
    cs = np.arange(1, n + 1, dtype=float).reshape(n, 1)
    stacked = np.vstack([mean - cs, mean + cs])
    got = float(np.quantile(stacked, cut, axis=0).ravel()[0])
    stacked_pos = 1 + cut * (2 * n - 1)
    return got, stacked_pos


def run_unit(series, n_windows, level):
    from statsforecast.models import Naive
    from statsforecast.utils import ConformalIntervals

    y_hist, y_test = series[:-1].astype(float), float(series[-1])
    coverage = level / 100.0
    if y_hist.size < n_windows + 2:
        return {"error": "too_short"}

    CAPTURED.clear()
    try:
        model = Naive(prediction_intervals=ConformalIntervals(
            n_windows=n_windows, h=1, method=METHOD))
        res = model.forecast(y=y_hist, h=1, level=[level])
    except Exception as exc:
        return {"error": type(exc).__name__}
    if not CAPTURED:
        return {"error": "no_conformal_call"}

    cs, mean = CAPTURED[-1]
    scores = np.sort(np.abs(np.asarray(cs, dtype=float).ravel()))
    n = scores.size
    if n < 2:
        return {"error": "too_few_windows"}
    centre = float(np.asarray(mean).ravel()[0])
    lo_a = float(np.asarray(res[f"lo-{level}"]).ravel()[0])
    hi_a = float(np.asarray(res[f"hi-{level}"]).ravel()[0])
    t_a = (hi_a - lo_a) / 2.0
    test = abs(y_test - centre)

    k = required_rank(n, coverage)
    t_b = math.inf if k is None else float(scores[k - 1])

    # The figure the paired report used to predict on: the smallest rank whose
    # score is at or above T_A. When the helper interpolated, that score is ABOVE
    # T_A and crediting T_A with it is what put the predicted delta at zero.
    at_or_above = rank_of(t_a, scores)
    # The rank T_A reaches and where inside the gap above it T_A sits, from the
    # one implementation both the arms and this ledger read.
    at_or_below, frac_pos = landed(t_a, scores)

    return {
        "n": n,
        "t_a": t_a,
        "t_b": t_b,
        "test": test,
        "required_rank": k if k is not None else n + 1,
        "feasible": k is not None,
        # a_rank is the rank T_A reaches, which is what paired_report predicts on.
        # a_rank_above is the figure the shipped Table 4 column was formed from,
        # kept so the two can be printed side by side.
        "a_rank": at_or_below,
        "a_rank_above": at_or_above,
        "landed_frac": frac_pos,
        "in_gap": bool(t_a < test <= t_b),
        "a_covered": bool(lo_a <= y_test <= hi_a),
        "a_width": hi_a - lo_a,
        "b_covered": bool(test <= t_b),
        "b_width": 2 * t_b if math.isfinite(t_b) else math.inf,
    }


def self_check():
    assert required_rank(2, 0.90) is None
    assert required_rank(10, 0.90) == 10
    assert required_rank(20, 0.90) == 19
    assert required_rank(50, 0.90) == 46
    # The two landed-rank figures must disagree on an interpolated threshold and
    # agree on an exact one, or the fix below is not measuring anything.
    s = np.array([1.0, 2.0, 3.0, 4.0])
    assert rank_of(2.5, s) == 3 and int(np.searchsorted(s, 2.5, "right")) == 2
    assert rank_of(3.0, s) == 3 and int(np.searchsorted(s, 3.0, "right")) == 3
    for n in (2, 5, 10, 20, 50):
        got, stacked = landed_position(n, 90)
        assert abs(got - (stacked - n)) < 1e-9, (n, got, stacked)


self_check()


def main():
    from run_real_data import load_series

    name = sys.argv[1] if len(sys.argv) > 1 else "m1_monthly_dataset"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    install_spy()

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say("PR1-17 -- per-unit ledger, statsforecast conformal_distribution")
    say("=" * 100)
    say("self_check() passed at import")
    say("")

    version, path, body, scores_ln, quant_ln = source_lines()
    say("-" * 100)
    say("(1) THE EXPRESSION, read from the installed file")
    say("-" * 100)
    say(f"statsforecast {version}")
    say(f"file: {path}")
    say(f"stack line: {scores_ln}   quantile line: {quant_ln}")
    for i, ln in body:
        say(f"  {i:5d}  {ln}")
    say("")
    say("cs is already absolute -- _conformity_scores ends in np.abs(fcst - y_test).")
    say("So the stacked array is mean +- the n absolute scores: 2n rows, symmetric")
    say("about mean, and above row n it is mean plus the ascending scores.")
    say("interpolation is the np.quantile default, 'linear'. No (n + 1) appears.")
    say("")

    say("-" * 100)
    say(f"(2) WHERE THE CUT LANDS, at level {LEVEL}, by executing the same call")
    say("-" * 100)
    say("    n   pos in 2n stacked   pos in n scores   floor   k* = ceil((n+1)*0.90)"
        "   short by")
    cut_lines = []
    for n in N_WINDOWS:
        got, stacked = landed_position(n, LEVEL)
        k = required_rank(n, LEVEL / 100.0)
        fl = math.floor(got)
        short = "infeasible" if k is None else f"{k - fl}"
        kt = "n/a" if k is None else str(k)
        say(f"  {n:3d}   {stacked:17.4f}   {got:15.4f}   {fl:5d}   {kt:>21}"
            f"   {short:>8}")
        cut_lines.append(
            f"CUT n={n} level={LEVEL} stacked_pos={stacked:.6f} pos={got:.6f} "
            f"floor={fl} kstar={kt} short={short}")
    say("")
    say("")
    for ln in cut_lines:
        say(ln)
    say("")
    say("The floor of the landed position is 0.9n for every n divisible by 10 and")
    say("k* is 0.9n + 1 there, so the shipped threshold reaches one rank fewer than")
    say("the level asks for, at every such n. The 0.05 remainder is where in the")
    say("gap above that position the threshold sits, not a rank it reaches.")
    say("")
    say("Swept rather than asserted from five cells, because a shortfall true at")
    say("the sizes this probe runs is not a shortfall true at every size:")
    by_short = {}
    for n in range(2, SWEEP_MAX + 1):
        got, _ = landed_position(n, LEVEL)
        k = required_rank(n, LEVEL / 100.0)
        if k is None:
            continue
        by_short.setdefault(k - math.floor(got), []).append(n)
    tens = sorted(n for n in range(10, SWEEP_MAX + 1, 10)
                  if required_rank(n, LEVEL / 100.0) is not None)
    # The claim the body states, over every size in the sweep and not over the
    # five the archive arm happens to run.
    for n in tens:
        got, _ = landed_position(n, LEVEL)
        k = required_rank(n, LEVEL / 100.0)
        assert k - math.floor(got) == 1, (n, got, k)
    feasible_ns = sum(len(v) for v in by_short.values())
    say(f"    feasible sizes to n={SWEEP_MAX}   {feasible_ns}")
    for short in sorted(by_short):
        ns = by_short[short]
        say(f"    short by {short}: {len(ns):4d} sizes"
            f"   first {ns[0]}   last {ns[-1]}")
    say(f"    every n divisible by 10 is short by exactly 1: {len(tens)} sizes,")
    say(f"    {tens[0]} to {tens[-1]}. Those are the mildest feasible sizes there")
    say(f"    are, and they are the three the archive arm runs.")
    say(f"SWEEP max_short={max(by_short)} min_short={min(by_short)} "
        f"feasible={feasible_ns} to_n={SWEEP_MAX} tens={len(tens)} "
        f"tens_first={tens[0]} tens_last={tens[-1]} "
        + " ".join(f"n_short{k}={len(v)}" for k, v in sorted(by_short.items())))
    say("")
    say("-" * 100)
    say("(3) THE LEDGER")
    say("-" * 100)
    series, meta = load_series(name, limit, min_len=max(N_WINDOWS) + 6)
    say(f"dataset: {name}   series: {len(series)}   cap: {limit}"
        f"   frequency: {meta.get('frequency', '?')}")
    say(f"model: Naive   method: {METHOD}   nominal {LEVEL / 100:.2f}")
    say("")

    cell_stats = []
    for nw in N_WINDOWS:
        recs = [run_unit(s, nw, LEVEL) for s in series]
        good = [r for r in recs if "error" not in r]
        s = summarize(recs)
        if s is None:
            say(f"  n_windows={nw}  -- no usable units")
            continue

        if nw == LEDGER_CELL:
            say(f"  first {LEDGER_ROWS} units at n_windows={nw}, one row each:")
            say("      unit    n        T_A        T_B   landed  floor   k*"
                "   |test score|  in gap   A   B")
            for i, r in enumerate(good[:LEDGER_ROWS]):
                tb = "inf" if not math.isfinite(r["t_b"]) else f"{r['t_b']:.4f}"
                say(f"      {i + 1:4d} {r['n']:4d} {r['t_a']:10.4f} {tb:>10}"
                    f" {r['landed_frac']:8.4f} {r['a_rank']:6d}"
                    f" {r['required_rank']:4d} {r['test']:14.4f}"
                    f" {'yes' if r['in_gap'] else ' no':>7}"
                    f"   {'Y' if r['a_covered'] else 'n'}"
                    f"   {'Y' if r['b_covered'] else 'n'}")
            say("")

        for ln in format_cell(f"n_windows={nw:<3}", s):
            say(ln)

        feas = [r for r in good if r["feasible"]]
        gap = sum(1 for r in good if r["in_gap"])
        # Arm A's own standard error across units, so a predicted arm A coverage
        # can be read against the measurement in the same units as the delta is.
        acov = np.array([r["a_covered"] for r in good], float)
        a_se = float(acov.std(ddof=1) / math.sqrt(acov.size)) if acov.size > 1 else 0.0
        # The delta as a COUNT. Arm B contains arm A here, so every unit that
        # changes status is a unit whose absolute score fell in (T_A, T_B], and
        # the two figures must agree exactly.
        say(f"      units in the gap (T_A, T_B]   {gap} of {len(good)}"
            f"   = {gap / len(good):+.4f}")
        assert abs(gap / len(good) - s["delta"]) < 1e-12, (nw, gap, s["delta"])
        assert s["gains"] == gap and s["losses"] == 0, (nw, s["gains"], gap)

        if not feas:
            # Every unit infeasible: arm B is +inf and covers by construction, so
            # a predicted delta against it is not a rank statement. The predicted
            # arm A coverage still is.
            pos = np.mean([r["landed_frac"] for r in good])
            nn = np.mean([r["n"] for r in good])
            say(f"      landed position (mean)        {pos:.4f} of {nn:.1f}"
                f"   floor {np.mean([r['a_rank'] for r in good]):.4f}")
            say(f"      pred arm A, floor rank        "
                f"{np.mean([r['a_rank'] / (r['n'] + 1) for r in good]):.4f}")
            pa_floor = float(np.mean([r["a_rank"] / (r["n"] + 1) for r in good]))
            pa_frac = float(np.mean([r["landed_frac"] / (r["n"] + 1) for r in good]))
            say(f"      pred arm A, interpolated      {pa_frac:.4f}")
            assert abs(s["pred_a"] - pa_frac) < 1e-12, (s["pred_a"], pa_frac)
            assert abs(s["pred_a_floor"] - pa_floor) < 1e-12
            say(f"      arm A s.e.                    {a_se:.4f}")
            za = {}
            for label, p in (("floor", pa_floor), ("interpolated", pa_frac)):
                z = (abs(s["a_cov"] - p) / a_se) if a_se > 0 else float("inf")
                za[label] = z
                say(f"      |arm A - pred A| / s.e., {label:<12}   {z:.2f}")
            say(f"      pred delta                    n/a"
                f"   (required rank exceeds n in {len(good)} of {len(good)} units)")
            say(f"CELL n_windows={nw} level={LEVEL} method={METHOD} "
                f"units={len(good)} feasible=0 gap={gap} "
                f"n_med={int(np.median([r['n'] for r in good]))} "
                f"a_cov={s['a_cov']:.6f} a_se={a_se:.6f} b_cov={s['b_cov']:.6f} "
                f"delta={s['delta']:.6f} se={s['se']:.6f} "
                f"pos_mean={np.mean([r['landed_frac'] for r in good]):.6f} "
                f"floor_mean={np.mean([r['a_rank'] for r in good]):.6f} "
                f"pred_a_floor={pa_floor:.6f} pred_a_frac={pa_frac:.6f} "
                f"z_a_floor={za['floor']:.6f} z_a_frac={za['interpolated']:.6f}")
            cell_stats.append({"n": nw, "z_a_floor": za["floor"],
                               "z_a_frac": za["interpolated"]})
            say("")
            continue

        pa_floor = float(np.mean([r["a_rank"] / (r["n"] + 1) for r in feas]))
        pa_frac = float(np.mean([r["landed_frac"] / (r["n"] + 1) for r in feas]))
        pb = float(np.mean([r["required_rank"] / (r["n"] + 1) for r in feas]))
        pd_floor = float(np.mean([(r["required_rank"] - r["a_rank"])
                                  / (r["n"] + 1) for r in feas]))
        pd_frac = float(np.mean([(r["required_rank"] - r["landed_frac"])
                                 / (r["n"] + 1) for r in feas]))
        pd_above = float(np.mean([(r["required_rank"] - r["a_rank_above"])
                                  / (r["n"] + 1) for r in feas]))
        say(f"      landed position (mean)        "
            f"{np.mean([r['landed_frac'] for r in feas]):.4f} of "
            f"{np.mean([r['n'] for r in feas]):.1f}"
            f"   floor {np.mean([r['a_rank'] for r in feas]):.4f}"
            f"   at-or-above {np.mean([r['a_rank_above'] for r in feas]):.4f}")
        say(f"      pred arm A, floor rank        {pa_floor:.4f}")
        say(f"      pred arm A, interpolated      {pa_frac:.4f}")
        say(f"      arm A s.e.                    {a_se:.4f}")
        za = {}
        for label, p in (("floor", pa_floor), ("interpolated", pa_frac)):
            z = (abs(s["a_cov"] - p) / a_se) if a_se > 0 else float("inf")
            za[label] = z
            say(f"      |arm A - pred A| / s.e., {label:<12}   {z:.2f}")
        say(f"      pred arm B, required rank     {pb:.4f}")
        # The shared summary computes the same two figures from the same records
        # through paired_report.landed. Printing both and comparing neither would
        # leave the ledger and the arms free to disagree.
        assert abs(s["pred_a"] - pa_frac) < 1e-12, (s["pred_a"], pa_frac)
        assert abs(s["pred_a_floor"] - pa_floor) < 1e-12
        assert abs(s["pred_delta"] - pd_frac) < 1e-12, (s["pred_delta"], pd_frac)
        assert abs(s["pred_delta_floor"] - pd_floor) < 1e-12
        assert abs(pa_frac + pd_frac - pb) < 1e-12, (pa_frac, pd_frac, pb)
        say(f"      pred delta, floor rank        {pd_floor:+.4f}")
        say(f"      pred delta, interpolated      {pd_frac:+.4f}")
        # Whether the measurement exceeds the rank-only bound. Reported per cell,
        # because the bound is the reason the article quotes the other form and
        # "a bound the measurement exceeds" has to be a cell someone can name.
        say(f"      measured delta above the floor bound   "
            f"{'yes' if s['delta'] > pd_floor else ' no'}")
        say(f"      pred delta, at-or-above rank  {pd_above:+.4f}"
            f"   (the figure Table 4 printed)")
        # How many standard errors each prediction sits from the measurement, off
        # the unrounded values, because the manuscript may not divide two of its
        # own rounded macros.
        zs = {}
        for label, p in (("floor", pd_floor), ("interpolated", pd_frac),
                         ("at-or-above", pd_above)):
            z = (abs(s["delta"] - p) / s["se"]) if s["se"] > 0 else float("inf")
            zs[label] = z
            say(f"      |measured - pred| / s.e., {label:<12}  {z:.2f}")
        say(f"CELL n_windows={nw} level={LEVEL} method={METHOD} "
            f"units={len(good)} feasible={len(feas)} gap={gap} "
            f"n_med={int(np.median([r['n'] for r in good]))} "
            f"a_cov={s['a_cov']:.6f} a_se={a_se:.6f} b_cov={s['b_cov']:.6f} "
            f"delta={s['delta']:.6f} se={s['se']:.6f} "
            f"pos_mean={np.mean([r['landed_frac'] for r in feas]):.6f} "
            f"floor_mean={np.mean([r['a_rank'] for r in feas]):.6f} "
            f"req_mean={np.mean([r['required_rank'] for r in feas]):.6f} "
            f"pred_a_floor={pa_floor:.6f} pred_a_frac={pa_frac:.6f} "
            f"pred_b={pb:.6f} pred_d_floor={pd_floor:.6f} "
            f"pred_d_frac={pd_frac:.6f} pred_d_above={pd_above:.6f} "
            f"z_d_floor={zs['floor']:.6f} z_d_frac={zs['interpolated']:.6f} "
            f"z_d_above={zs['at-or-above']:.6f} "
            f"z_a_floor={za['floor']:.6f} z_a_frac={za['interpolated']:.6f} "
            f"over_floor={1 if s['delta'] > pd_floor else 0}")
        cell_stats.append({"n": nw, "z_d_frac": zs["interpolated"],
                           "z_d_floor": zs["floor"], "z_d_above": zs["at-or-above"],
                           "z_a_frac": za["interpolated"],
                           "z_a_floor": za["floor"],
                           "over_floor": 1 if s["delta"] > pd_floor else 0})
        say("")

    # The worst cell per prediction form, printed rather than left to a reader to
    # find. The body quotes one figure for "at the worst of the three cells" and
    # taking a maximum over four-decimal fields in the manuscript is arithmetic the
    # build refuses.
    say("The worst feasible cell for each prediction form, in standard errors of")
    say("the measurement, and for arm A over every cell including the infeasible:")
    for field, label in (("z_d_frac", "pred delta at h"),
                         ("z_d_floor", "pred delta at floor(h)"),
                         ("z_d_above", "pred delta at ceil(h)"),
                         ("z_a_frac", "pred arm A at h"),
                         ("z_a_floor", "pred arm A at floor(h)")):
        pool = [c for c in cell_stats if field in c]
        if not pool:
            continue
        w = max(pool, key=lambda c: c[field])
        say(f"    {label:<26} {w[field]:.2f}  at n={w['n']}")
        say(f"WORST {field}={w[field]:.6f} at_n={w['n']}")
    say("")
    over = [c["n"] for c in cell_stats if c.get("over_floor")]
    say(f"OVERFLOOR cells={len(over)} of={len([c for c in cell_stats if 'z_d_frac' in c])}"
        f" first_n={over[0] if over else 0}")
    say("")
    say("Arm B contains arm A at every unit, so the paired delta is exactly the")
    say("share of units whose absolute score fell in (T_A, T_B]. That gap is one")
    say("score wide wherever the required rank is one above the landed floor, and")
    say("the count above says how much archive mass sat in it.")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
