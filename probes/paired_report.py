#!/usr/bin/env python3
"""Shared summary and formatting for the paired real-data arms.

run_real_data.py established the design for sktime: arm A is the library's
shipped interval, arm B is the interval the SAME scores and the SAME centre
support under the required order statistic, and the paired delta is the claim.
The statsforecast, darts and tabular arms repeat that design, so the summary
arithmetic and the two reporting subtleties live here once rather than three
times.

The two subtleties, both learned from the sktime arm's v1 failure:

  1. An infinite bound always covers, so mixing infeasible cells into a mean
     width compares arm B's feasible subset against arm A's full set. Widths
     are reported for both arms on the feasible subset only.
  2. Medians of two quantities that vary per unit (the required rank and n) can
     print a rank above n without any cell having one. Both are labelled as
     medians for that reason.

A record is a dict with keys: n, required_rank, feasible, a_covered, a_width,
a_rank, b_covered, b_width.
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
    out = {
        "cells": len(good),
        "n_median": int(np.median([r["n"] for r in good])),
        "a_cov": float(a.mean()),
        "b_cov": float(b.mean()),
        "delta": float(d.mean()),
        "se": float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else float("nan"),
        "infeasible": len(good) - len(feas),
        "a_rank_median": int(np.median([r["a_rank"] for r in good])),
        "req_rank_median": int(np.median([r["required_rank"] for r in feas])) if feas else 0,
        "errors": {},
    }
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
    lines = [
        f"  {header}   series={s['cells']:<4} median n={s['n_median']}"
        + (f"   [{', '.join(f'{k} x{v}' for k, v in s['errors'].items())}]"
           if s["errors"] else ""),
        f"      arm A (shipped)        coverage {s['a_cov']:.4f}   "
        f"lands on median rank {s['a_rank_median']} of {s['n_median']}",
        f"      arm B (required rank)  coverage {s['b_cov']:.4f}   "
        + (f"median required rank {s['req_rank_median']} of {s['n_median']}"
           if s["req_rank_median"] else "required rank exceeds n in every cell")
        + (f"   [{s['infeasible']}/{s['cells']} infeasible -> +inf]"
           if s["infeasible"] else ""),
        f"      paired delta (B - A)   {s['delta']:+.4f}  (s.e. {s['se']:.4f})"
        f"  {stars(s['delta'], s['se'])}",
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
