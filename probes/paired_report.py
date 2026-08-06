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
     number of units that changed status. The gains and the losses are counted
     separately here and printed, because their SUM is the count the prose wants
     and their DIFFERENCE is the delta. Reading the count off the delta hid a
     reversal in three cells of the sktime arm through two adversarial reads.
  4. Whether arm B contains arm A is a property of the two constructions, not
     something to be assumed from the word "paired". A record carrying
     nests=True asserts here that no unit went the other way; a probe comparing
     genuinely different constructions must set nests=False and say so.

A record is a dict with keys: n, required_rank, feasible, a_covered, a_width,
a_rank, b_covered, b_width, and optionally two_rail and nests.
"""

import math

import numpy as np


def summarize(records):
    """Paired statistics over one cell. Returns None if the cell is empty."""
    good = [r for r in records if r is not None and "error" not in r]
    if not good:
        return None
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
    out = {
        "cells": len(good),
        "n_median": int(np.median([r["n"] for r in good])),
        "a_cov": float(a.mean()),
        "b_cov": float(b.mean()),
        "delta": float(d.mean()),
        "se": float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else float("nan"),
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


def format_cell(header, s):
    """Return the lines for one summarized cell."""
    if s is None:
        return [f"  {header}  -- no usable cells"]
    # A two-rail helper resolves two levels and what it delivers is a SPAN in
    # gaps, not a rank. Half of an asymmetric width is not an order statistic of
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
                 a_width=1.0, a_rank=9, b_covered=True, b_width=1.5)
            for i in range(10)]
    s = summarize(recs)
    assert s["a_cov"] == 0.8 and s["b_cov"] == 1.0
    assert abs(s["delta"] - 0.2) < 1e-12
    assert s["infeasible"] == 0 and s["f_cells"] == 10
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
    above = summarize([dict(r, nests=False, a_rank=12, required_rank=10)
                       for r in mixed])
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
                         a_width=1.0, a_rank=5, b_covered=True, b_width=math.inf)]
    s2 = summarize(recs2)
    assert s2["infeasible"] == 1
    assert math.isfinite(s2["f_b_width"]) and s2["f_cells"] == 10
    # errors are counted, not dropped silently
    s3 = summarize(recs + [{"error": "too_short"}])
    assert s3["errors"] == {"too_short": 1} and s3["cells"] == 10
    assert summarize([]) is None
    assert "0.0 s.e." not in stars(0.0, 0.0)


self_check()
