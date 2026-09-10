#!/usr/bin/env python3
"""Shared summary and formatting for the paired real-data arms.

run_real_data.py established the design for sktime: arm A is the library's
shipped interval, arm B is the interval the SAME scores and the SAME centre
support under the required order statistic, and the paired delta is the claim.
The statsforecast, darts and tabular arms repeat that design, so the summary
arithmetic and the reporting subtleties live here once rather than four times.

The subtleties, each learned from a failure of this harness:

  1. An infinite bound always covers, so mixing infeasible cells into a mean
     width compares arm B's feasible subset against arm A's full set. Widths
     are reported for both arms on the feasible subset only.
  2. Medians of two quantities that vary per unit (the required rank and n) can
     print a rank above n without any cell having one. Both are labelled as
     medians for that reason.
  3. `Delta x units` is an integer for ANY pair of 0/1 indicators, nested or
     not, so it cannot tell you the arms are paired as claimed and it is not the
     count of units with status changes. The gains and the losses are counted
     separately here and printed, because their SUM is the count the prose wants
     and their DIFFERENCE is the delta. Reading the count off the delta hid a
     reversal in three cells of the sktime arm through two adversarial reads.
  4. Whether arm B contains arm A is a property of the two constructions, not
     something to be assumed from the word "paired". A record carrying
     nests=True asserts here that no unit went the other way; a probe comparing
     genuinely different constructions must set nests=False and say so.
  5. An interpolating helper returns a threshold BETWEEN two scores, so it
     reaches the lower of the two and no more. Crediting it with the one above
     -- which "the smallest rank at or above the threshold" does -- puts the
     predicted delta at the wrong end of a whole gap, and at these calibration
     sizes a whole gap is the entire effect: the statsforecast n=10 cell
     predicted +0.0007 against a measured +0.1080 for that reason alone.
     `a_rank` is therefore the rank arm A REACHES, counted at or below its
     threshold, and `landed_frac` says where inside the gap above it the
     threshold sits. The two give a distribution-free floor and a point
     prediction, and both are printed.

A record is a dict with keys: n, required_rank, feasible, a_covered, a_width,
a_rank, landed_frac, b_covered, b_width, and optionally two_rail and nests.
"""

import math

import numpy as np


# A threshold that IS one of the scores arrives off it by float drift: darts'
# calibration path returns the 10th of 10 scores and misses it by one part in 10^15
# at 28 of 250 series, and reading that as "reaches the 9th" is a whole gap of error
# in the direction that flatters the library. Relative, so it scales with the
# score's own magnitude, and twelve orders of magnitude below any gap these score
# sets have. self_check() pins both ends: a near-hit reaches, a real near-miss does
# not.
LANDED_TOL = 1e-9


def landed(threshold, scores):
    """Where a threshold lands in a score set: (rank reached, fractional position).

    The rank reached is how many scores sit AT OR BELOW the threshold. That is the
    rank whose coverage the threshold inherits. An exchangeable draw sits under the
    r-th of n order statistics with probability r/(n+1) and no other value; a
    threshold placed between the r-th and the (r+1)-th is under the (r+1)-th, so it
    inherits the r-th's guarantee and not the next one's.

    The fractional position places the threshold inside the gap above that rank,
    linearly, and equals the rank exactly when the threshold IS one of the scores.
    Reading it through r/(n+1) estimates the coverage the interpolation delivers:
    exact for uniform scores, and the companion gives the order-1/n^2 error for a
    smooth density. Clamped to [0, n], because a threshold outside the score range
    has no gap to sit in.

    LANDED_TOL closes a near-hit from below, so a threshold that should equal a
    score is credited with the rank that score carries.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    if not math.isfinite(threshold):
        return n, float(n)
    t = threshold + LANDED_TOL * max(1.0, abs(threshold))
    r = int(np.searchsorted(s, t, side="right"))
    if r >= n:
        return n, float(n)
    if r == 0:
        return 0, 0.0
    lo, hi = float(s[r - 1]), float(s[r])
    # From the threshold as given, not from the tolerated one: `tol` exists to
    # decide the RANK when a near-hit misses by drift, and letting it into the
    # fraction as well would print a position of 3.000000003 for a threshold that
    # is one of the scores.
    frac = (threshold - lo) / (hi - lo) if hi > lo else 0.0
    return r, r + float(min(1.0, max(0.0, frac)))


def summarize(records):
    """Paired statistics over one cell. Returns None if the cell is empty."""
    good = [r for r in records if r is not None and "error" not in r]
    if not good:
        return None
    # Subtlety 5. A probe that has not been updated to report where its threshold
    # landed inside the gap would otherwise fall back to the rank silently, and the
    # predicted delta it prints would be the old wrong-end one.
    missing = [r for r in good if "landed_frac" not in r]
    assert not missing, (
        f"{len(missing)} of {len(good)} records carry no 'landed_frac'; the "
        f"predicted coverage cannot be formed from the rank alone")
    a = np.array([r["a_covered"] for r in good], float)
    b = np.array([r["b_covered"] for r in good], float)
    d = b - a
    feas = [r for r in good if r["feasible"]]
    # The three counts the delta cannot give you. gains - losses is the delta
    # times the unit count; gains + losses is how many units changed status.
    gains = int(np.sum(d > 0))
    losses = int(np.sum(d < 0))
    nests = all(r.get("nests", True) for r in good)
    # WHY a cell does not nest, decided from the index figures rather than named by
    # the caller. If arm A already reaches or passes the required rank or span then
    # arm B is the MINIMAL sufficient interval and the narrower one, so a loss is
    # arm A's conservatism showing and not two constructions being compared. Getting
    # this backwards is how a one-gap rounding-outward became "splitting alpha across
    # two tails lands wider than the symmetric bound arm B builds".
    a_meets = all(r["a_rank"] >= r["required_rank"] for r in good
                  if r.get("required_rank")) if good else False
    # Mean of per-unit (req - landed)/(n+1) over feasible units. Medians of the two
    # index figures are not a substitute when n varies inside the cell -- that
    # construction printed +0.0000 at statsforecast m=10 against a measured miss.
    # Two forms, and the difference between them is one gap: `landed_frac` is where
    # arm A's threshold actually sits and gives a point prediction, `a_rank` is the
    # rank it reaches and gives the distribution-free end of the bracket. Neither is
    # derived from the other here; each is a mean across the identical unit set.
    preds = [
        (r["required_rank"] - r["landed_frac"]) / (r["n"] + 1)
        for r in feas
        if r.get("required_rank") is not None
    ]
    preds_floor = [
        (r["required_rank"] - r["a_rank"]) / (r["n"] + 1)
        for r in feas
        if r.get("required_rank") is not None
    ]
    # Predicted arm A coverage, over EVERY unit and not only the feasible ones:
    # arm A returns a threshold below the floor too, and what it lands on there is
    # the whole of reading (1). Arm B is +inf there and has no rank, which is why
    # the predicted delta above cannot be formed at those cells.
    pred_a = [r["landed_frac"] / (r["n"] + 1) for r in good]
    pred_a_floor = [r["a_rank"] / (r["n"] + 1) for r in good]
    out = {
        "cells": len(good),
        "n_median": int(np.median([r["n"] for r in good])),
        "a_cov": float(a.mean()),
        "b_cov": float(b.mean()),
        "delta": float(d.mean()),
        "se": float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else float("nan"),
        "pred_delta": float(np.mean(preds)) if preds else float("nan"),
        "pred_delta_floor": (float(np.mean(preds_floor)) if preds_floor
                             else float("nan")),
        "pred_a": float(np.mean(pred_a)),
        "pred_a_floor": float(np.mean(pred_a_floor)),
        "gains": gains,
        "losses": losses,
        "changed": gains + losses,
        "nests": nests,
        "a_meets": a_meets,
        "two_rail": all(r.get("two_rail", False) for r in good),
        "infeasible": len(good) - len(feas),
        "a_rank_median": int(np.median([r["a_rank"] for r in good])),
        "req_rank_median": int(np.median([r["required_rank"] for r in feas])) if feas else 0,
        "errors": {},
    }
    # A nesting claim is checked, not carried. If arm B contains arm A for every
    # unit then no unit can lose coverage by moving to arm B, so a single loss
    # means the two arms are not the constructions the caller thinks they are.
    assert not (nests and losses), (
        f"{losses} of {len(good)} units lost coverage under arm B while the records "
        f"claim arm B contains arm A -- the arms differ in more than the rank")
    errs = [r.get("error") for r in records if r is not None and "error" in r]
    for e in errs:
        out["errors"][e] = out["errors"].get(e, 0) + 1
    if feas:
        fa = np.array([r["a_covered"] for r in feas], float)
        fb = np.array([r["b_covered"] for r in feas], float)
        fd = fb - fa
        out.update({
            "f_cells": len(feas),
            "f_a_cov": float(fa.mean()),
            "f_b_cov": float(fb.mean()),
            "f_a_width": float(np.mean([r["a_width"] for r in feas])),
            "f_b_width": float(np.mean([r["b_width"] for r in feas])),
            "f_delta": float(fd.mean()),
            "f_se": float(fd.std(ddof=1) / math.sqrt(fd.size)) if fd.size > 1 else float("nan"),
        })
    return out


def exceedance_multiple(delivered, nominal):
    """The delivered exceedance rate as a multiple of the requested one.

    A risk or alerting report states 1 - coverage, so a shortfall in coverage is
    read there as a multiple: 0.5840 against a requested 0.90 is not "0.316 low",
    it is 4.16 times the exceedance the caller asked for. The article's Figure 1
    argues that this is the number such a report carries, and then prints only the
    absolute pair beside it.

    Computed here from the unrounded mean coverage, and printed by the probe, for
    the reason the whole build exists: (1 - 0.5840) / (1 - 0.90) off two printed
    fields is 4.16 and off the unrounded values it can differ in the last digit,
    and the manuscript may not do arithmetic on parsed values.
    """
    room = 1.0 - nominal
    if room <= 0:
        return float("inf")
    return (1.0 - delivered) / room


def format_cell(header, s):
    """Return the lines for one summarized cell."""
    if s is None:
        return [f"  {header}  -- no usable cells"]
    # A two-rail helper resolves two levels and what it delivers is a SPAN in
    # gaps, not a rank. Half of an asymmetric width does not correspond to an order statistic of
    # anything, so the word changes with the construction.
    unit = "span" if s["two_rail"] else "rank"
    lines = [
        f"  {header}   series={s['cells']:<4} median n={s['n_median']}"
        + (f"   [{', '.join(f'{k} x{v}' for k, v in s['errors'].items())}]"
           if s["errors"] else ""),
        f"      arm A (shipped)        coverage {s['a_cov']:.4f}   "
        f"lands on median {unit} {s['a_rank_median']} of {s['n_median']}",
        f"      arm B (required rank)  coverage {s['b_cov']:.4f}   "
        + (f"median required {unit} {s['req_rank_median']} of {s['n_median']}"
           if s["req_rank_median"] else f"required {unit} exceeds n in every cell")
        + (f"   [{s['infeasible']}/{s['cells']} infeasible -> +inf]"
           if s["infeasible"] else ""),
        f"      paired delta (B - A)   {s['delta']:+.4f}  (s.e. {s['se']:.4f})"
        f"  {stars(s['delta'], s['se'])}",
        f"      predicted arm A        {s['pred_a']:.4f}"
        f"   (rank floor {s['pred_a_floor']:.4f})",
        (
            f"      predicted delta        {s['pred_delta']:+.4f}"
            f"   (rank floor {s['pred_delta_floor']:+.4f})"
            if not math.isnan(s["pred_delta"])
            else "      predicted delta        n/a (no feasible unit)"
        ),
        # Printed for every cell, including the zero ones, so the count in the
        # prose is a parsed field rather than delta x units.
        f"      status changed in {s['changed']} of {s['cells']} units: "
        f"gains={s['gains']} losses={s['losses']}"
        + ("   [arm B contains arm A: losses must be 0]" if s["nests"]
           else "   [arm A already reaches the required index, so arm B is the "
                "MINIMAL sufficient interval and the narrower one: losses are arm "
                "A's conservatism]" if s["a_meets"]
           else "   [arms are different constructions: losses are expected]"),
    ]
    if s["infeasible"] == s["cells"]:
        # Arm B is +inf everywhere here, so it covers by construction. The delta
        # is then a measure of how far the shipped FINITE interval falls short of
        # a nominal level that no finite interval can reach -- not a comparison
        # of two ranks. Saying so in the output keeps the number from being
        # quoted as if it were the convention's effect size.
        lines.append("      ^ arm B is vacuous (+inf) in every cell: this delta measures"
                     " infeasibility,")
        lines.append("        not the level->rank map. 1 - A is the shortfall against a"
                     " level no finite")
        lines.append("        interval can deliver at this n.")
    if "f_cells" in s and s["infeasible"]:
        lines.append(
            f"      feasible only ({s['f_cells']}):  A {s['f_a_cov']:.4f} "
            f"(width {s['f_a_width']:.4g})   B {s['f_b_cov']:.4f} "
            f"(width {s['f_b_width']:.4g})   delta {s['f_delta']:+.4f} "
            f"(s.e. {s['f_se']:.4f})")
    elif "f_cells" in s:
        lines.append(f"      widths:  A {s['f_a_width']:.4g}   B {s['f_b_width']:.4g}")
    return lines


def stars(delta, se):
    """How many standard errors the delta is from zero, as a short tag."""
    if not se or math.isnan(se) or se == 0:
        return "(exact 0)" if delta == 0 else ""
    z = abs(delta) / se
    return f"{z:.1f} s.e." + ("  <- >=2 s.e." if z >= 2 else "")


def self_check():
    # a cell where B covers strictly more, all feasible
    recs = [dict(n=10, required_rank=10, feasible=True, a_covered=(i > 1),
                 a_width=1.0, a_rank=9, landed_frac=9.0, b_covered=True,
                 b_width=1.5)
            for i in range(10)]
    s = summarize(recs)
    assert s["a_cov"] == 0.8 and s["b_cov"] == 1.0
    assert abs(s["delta"] - 0.2) < 1e-12
    # (10 - 9)/(10+1) = 1/11 on every feasible unit
    assert abs(s["pred_delta"] - 1 / 11) < 1e-12, s["pred_delta"]
    assert s["infeasible"] == 0 and s["f_cells"] == 10
    # Medians of the two ranks can agree while the per-unit mean does not: two
    # units at n=10 (pred 1/11) and two at n=20 with equal ranks (pred 0) give
    # medians that look like a zero prediction and a mean that is not.
    split = [
        dict(n=10, required_rank=10, feasible=True, a_covered=True,
             a_width=1.0, a_rank=9, landed_frac=9.0, b_covered=True, b_width=1.5),
        dict(n=10, required_rank=10, feasible=True, a_covered=True,
             a_width=1.0, a_rank=9, landed_frac=9.0, b_covered=True, b_width=1.5),
        dict(n=20, required_rank=19, feasible=True, a_covered=True,
             a_width=1.0, a_rank=19, landed_frac=19.0, b_covered=True, b_width=1.5),
        dict(n=20, required_rank=19, feasible=True, a_covered=True,
             a_width=1.0, a_rank=19, landed_frac=19.0, b_covered=True, b_width=1.5),
    ]
    sp = summarize(split)
    median_pred = (sp["req_rank_median"] - sp["a_rank_median"]) / (sp["n_median"] + 1)
    assert abs(median_pred) < 1e-12, median_pred
    assert abs(sp["pred_delta"] - ((1 / 11) + (1 / 11) + 0 + 0) / 4) < 1e-12, sp["pred_delta"]
    assert abs(sp["pred_delta"] - median_pred) > 1e-6, (
        "the median-of-indices construction that must not reach the table")
    # gains and losses are counted, and their sum is NOT delta x units when a
    # unit goes the other way. This is the case that shipped undetected.
    assert (s["gains"], s["losses"], s["changed"]) == (2, 0, 2)
    mixed = [dict(r, nests=False) for r in recs]
    # unit 2 covered under BOTH arms above, so flipping arm B alone turns a zero
    # into a loss and leaves the two gains standing
    mixed[2] = dict(mixed[2], a_covered=True, b_covered=False)
    m = summarize(mixed)
    assert round(m["delta"] * m["cells"]) == 1, m["delta"]
    assert (m["gains"], m["losses"], m["changed"]) == (2, 1, 3), m
    assert m["changed"] != round(m["delta"] * m["cells"]), (
        "the case the old delta x units check could not see")
    # and the reason for not nesting is read off the index figures, not declared
    assert not m["a_meets"], "arm A lands at rank 9 of a required 10 here"
    above = summarize([dict(r, nests=False, a_rank=12, landed_frac=12.0,
                            required_rank=10) for r in mixed])
    assert above["a_meets"] and "conservatism" in "\n".join(
        format_cell("h", above)), "an over-covering arm A is reported as conservatism"
    # and the nesting claim is enforced rather than printed
    try:
        summarize([dict(r, nests=True) for r in mixed])
    except AssertionError:
        pass
    else:
        raise AssertionError("a coverage loss passed under nests=True")
    # the word follows the construction
    assert "median span" in "\n".join(
        format_cell("h", summarize([dict(r, two_rail=True) for r in recs])))
    assert "median rank" in "\n".join(format_cell("h", s))
    # an infeasible cell must not contaminate the feasible-only widths
    recs2 = recs + [dict(n=5, required_rank=6, feasible=False, a_covered=True,
                         a_width=1.0, a_rank=5, landed_frac=5.0, b_covered=True,
                         b_width=math.inf)]
    s2 = summarize(recs2)
    assert s2["infeasible"] == 1
    assert math.isfinite(s2["f_b_width"]) and s2["f_cells"] == 10
    # errors are counted, not dropped silently
    s3 = summarize(recs + [{"error": "too_short"}])
    assert s3["errors"] == {"too_short": 1} and s3["cells"] == 10
    assert summarize([]) is None
    # landed(): the rank REACHED, and where inside the gap above it the threshold
    # sits. An exact hit must give the rank itself, or the two prediction forms
    # coincide everywhere and the distinction below is decoration.
    sc = np.array([10.0, 20.0, 30.0, 40.0])
    assert landed(30.0, sc) == (3, 3.0)
    r, f = landed(30.5, sc)
    assert (r, round(f, 4)) == (3, 3.05), (r, f)
    assert landed(5.0, sc) == (0, 0.0)
    assert landed(40.0, sc) == (4, 4.0) and landed(99.0, sc) == (4, 4.0)
    assert landed(math.inf, sc) == (4, 4.0)
    # the drift tolerance closes a near-hit from below and nothing wider
    assert landed(30.0 - 1e-12, sc) == (3, 3.0)
    assert landed(30.0 - 1e-6, sc)[0] == 2, (
        "the tolerance is wide enough to credit a rank the threshold misses")
    assert landed(29.0, sc)[0] == 2
    # An interpolating helper: the threshold sits 5% of the way up the last gap,
    # so it reaches rank 9 and the required rank is 10. The point prediction is
    # 0.95/11 and the distribution-free floor is 1/11; the figure the harness used
    # to print, crediting rank 10, is 0. All three differ, which is the whole of
    # PR1-17.
    interp = [dict(n=10, required_rank=10, feasible=True, a_covered=True,
                   a_width=1.0, a_rank=9, landed_frac=9.05, b_covered=True,
                   b_width=1.5) for _ in range(4)]
    si = summarize(interp)
    assert abs(si["pred_delta"] - 0.95 / 11) < 1e-12, si["pred_delta"]
    assert abs(si["pred_delta_floor"] - 1 / 11) < 1e-12, si["pred_delta_floor"]
    assert abs(si["pred_a"] - 9.05 / 11) < 1e-12, si["pred_a"]
    assert abs(si["pred_a_floor"] - 9 / 11) < 1e-12, si["pred_a_floor"]
    assert abs(si["pred_a"] + si["pred_delta"] - 10 / 11) < 1e-12, (
        "the printed arm A and delta predictions must sum to arm B's exact "
        "coverage, or the two table columns cannot be read against each other")
    # a record that has not been updated must fail loudly rather than fall back
    try:
        summarize([{k: v for k, v in interp[0].items() if k != "landed_frac"}])
    except AssertionError:
        pass
    else:
        raise AssertionError("a record with no landed_frac was summarized")
    assert "0.0 s.e." not in stars(0.0, 0.0)


self_check()
