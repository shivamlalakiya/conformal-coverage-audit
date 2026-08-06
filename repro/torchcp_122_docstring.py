#!/usr/bin/env python3
"""torchcp #122: the docstring names a threshold the code no longer computes.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
`calculate_conformal_value` documents the quantile level it uses. The level the code
computes is the finite-sample-corrected one, ceil((n+1)(1-alpha))/n, which is not the
level the documentation names -- and above the feasibility boundary the corrected level
exceeds 1, at which point the function warns and returns an infinite threshold rather
than the quantity either description implies.

The script reads the docstring out of the installed module and runs the function, so
the comparison is between what ships and what ships beside it. It does not paraphrase
the docstring: the relevant lines are printed verbatim.
"""

import inspect
import math
import re
import sys
import warnings
from fractions import Fraction

import numpy as np


def main():
    import torch
    import torchcp
    from torchcp.classification.utils.metrics import Metrics  # noqa: F401 (import check)
    from torchcp.utils.common import calculate_conformal_value

    print(f"torchcp {getattr(torchcp, '__version__', '?')}, torch {torch.__version__}")
    print()
    doc = inspect.getdoc(calculate_conformal_value) or ""
    src = inspect.getsource(calculate_conformal_value)
    print("--- docstring, verbatim ---")
    for ln in doc.splitlines():
        print(f"  {ln}")
    print("--- end docstring ---")
    print()

    # what the code actually forms, read off the source rather than described
    lvl = [ln.strip() for ln in src.splitlines()
           if "1 - alpha" in ln or "alpha" in ln and ("ceil" in ln or "/" in ln)]
    print("lines in the body that form a level:")
    for ln in lvl[:6]:
        print(f"  {ln}")
    print()

    print("Executed: scores 1..n, so a returned threshold IS the rank it landed on.")
    print(f"{'n':>5}{'alpha':>7}{'corrected level':>17}{'>1?':>6}"
          f"{'warns':>7}{'returned':>12}")
    print("-" * 56)
    rows = []
    for n in (8, 9, 10, 19, 20):
        for alpha in (0.1, 0.05):
            scores = torch.arange(1, n + 1, dtype=torch.float32)
            q = Fraction(math.ceil(Fraction(n + 1) * Fraction(1 - alpha)
                                   .limit_denominator(10 ** 6)), n)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    out = calculate_conformal_value(scores, alpha)
                    nw = len(caught)
                    val = float(out)
                except Exception as exc:
                    nw = len(caught)
                    val = float("nan")
                    print(f"{n:>5}{alpha:>7}{str(q):>17}{str(q > 1):>6}"
                          f"{nw:>7}  {type(exc).__name__}")
                    continue
            rows.append({"n": n, "alpha": alpha, "q": q, "over": q > 1,
                         "warn": nw, "val": val})
            print(f"{n:>5}{alpha:>7}{str(q):>17}{str(q > 1):>6}{nw:>7}"
                  f"{('+inf' if not math.isfinite(val) else f'{val:.3f}'):>12}")

    print()
    over = [r for r in rows if r["over"]]
    under = [r for r in rows if not r["over"]]
    assert over and under, (
        "the grid does not straddle the boundary where the corrected level exceeds 1")
    inf_over = [r for r in over if not math.isfinite(r["val"])]
    mentions_corrected = bool(re.search(r"n\s*\+\s*1|ceil|finite[- ]sample", doc, re.I))
    print(f"sizes where the corrected level exceeds 1: {len(over)}, of which "
          f"{len(inf_over)} return an infinite threshold")
    print(f"does the docstring mention the (n+1) correction or the boundary? "
          f"{'yes' if mentions_corrected else 'NO'}")
    print()
    if inf_over and not mentions_corrected:
        print("REPRODUCES. The function forms a corrected level, returns an infinite "
              "threshold once that level passes 1, and the docstring describes "
              "neither.")
        return 0
    print("does not reproduce: "
          f"{len(inf_over)} infinite returns above the boundary, docstring mentions "
          f"the correction: {mentions_corrected}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
