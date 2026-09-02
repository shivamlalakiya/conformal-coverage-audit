#!/usr/bin/env python3
"""Wilson 95% intervals on the paired differences, from the arms' own counts.

WHY THIS IS A SEPARATE PROBE AND NOT A COLUMN IN THE ARMS THEMSELVES
The real-data arms print a standard error beside every paired difference and no
interval. A standard error is not an interval, and the two are not
interchangeable at the sizes here. Take 250 units and a difference of 0.0120: the
symmetric form reaches below zero, the score form stops short of it, and the reason
is that the proportion sits close enough to the boundary for symmetry to be the
wrong assumption. Empirical-software-engineering referees ask
for intervals as a matter of course, and the honest way to supply them is to
compute them from the counts rather than from the printed standard error.

The counts are exact integers. Each arm prints `gains=G losses=L` over `units=N`
and the paired difference is (G-L)/N, which the reader of this file re-derives
and asserts against the printed value rather than trusting either. Nothing here
re-rounds a rounded field: this probe never reads the printed delta as an input
to an interval, only as a check on the counts it did read. That is the rule the
manuscript's third gate enforces one layer up, and it is worth respecting here
too even though this file sits outside that gate's boundary.

WHY WILSON
Every arm in the family below is nested, so `losses` is 0 and the difference is the
plain binomial proportion G/N. Wilson is the score interval for that proportion.
It is chosen over Wald because Wald's coverage collapses for small p -- and small
p is exactly this paper's regime, where a one-rank deficit moves nine units in
250 -- and over Clopper-Pearson because Clopper-Pearson is conservative by
construction and would overstate the width of a result this paper is arguing is
real. A non-nested arm would need an interval for a difference of correlated
proportions instead, so this file refuses one rather than printing a Wilson
interval that does not apply: the two non-nested cells in the paper are in the
tabular asymmetric configuration and keep their standard errors.

    python probes/interval_estimates.py > outputs/probe_output_interval_estimates.txt
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiplicity_and_reimpl import FAMILY, read_cell        # noqa: E402

Z95 = 1.959963984540054          # the two-sided normal quantile, not a measurement

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def wilson(successes, n, z=Z95):
    """The score interval for a binomial proportion. Returns (lo, hi)."""
    assert 0 <= successes <= n and n > 0, (successes, n)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _self_check():
    """Two properties that a wrong implementation fails, and one that Wald passes.

    The interval must contain the point estimate, and it must NOT be symmetric
    about it near the boundary -- if it is, what has been implemented is Wald
    under a different name, which is the mistake this file exists to avoid.
    """
    lo, hi = wilson(9, 250)
    p = 9 / 250
    assert lo < p < hi, (lo, p, hi)
    assert abs((p - lo) - (hi - p)) > 1e-4, (
        "the interval is symmetric about the estimate, so this is Wald and not "
        "Wilson; near a boundary that is the difference the file is for")
    # zero successes: a proper score interval still has width, and starts at 0.
    # Not `== 0.0`: centre and half agree to the last bit rather than exactly, so
    # the subtraction lands a few times 1e-19 above zero and an equality here
    # fails on arithmetic rather than on the interval being wrong.
    lo0, hi0 = wilson(0, 250)
    assert lo0 < 1e-12 and hi0 > 0.0, (lo0, hi0)
    # and it narrows as n grows, which a constant would not
    assert (wilson(90, 2500)[1] - wilson(90, 2500)[0]) < (hi - lo)


_self_check()


def main():
    say("=" * 96)
    say("WILSON 95% INTERVALS ON THE PAIRED DIFFERENCES")
    say("=" * 96)
    say("self_check() passed at import (Wilson, not Wald: asymmetry near 0 asserted)")
    say()
    say("Read from each arm's own committed output. The paired difference is")
    say("(gains - losses)/units on integer counts; every arm below is nested, so")
    say("losses is 0 and the difference is the binomial proportion gains/units.")
    say("The printed delta is re-derived from the counts and asserted, never used")
    say("as an input. z = 1.959963984540054.")
    say()
    say(f"{'arm':<32} {'units':>6} {'gains':>6} {'delta':>9} "
        f"{'wilson lo':>10} {'wilson hi':>10} {'s.e.':>8}")
    say("-" * 96)

    rows = []
    for name, path, nominal, key in FAMILY:
        c = read_cell(path, nominal, key)
        assert c["losses"] == 0, (
            f"{name}: losses={c['losses']}, so the arms are not nested here and a "
            f"binomial interval on the difference does not apply. This probe "
            f"refuses rather than printing one")
        lo, hi = wilson(c["gains"], c["units"])
        delta = (c["gains"] - c["losses"]) / c["units"]
        assert abs(delta - c["delta"]) < 5e-5, (name, delta, c["delta"])
        say(f"{name:<32} {c['units']:>6} {c['gains']:>6} {delta:>9.4f} "
            f"{lo:>10.4f} {hi:>10.4f} {c['se']:>8.4f}")
        rows.append((name, lo, hi, delta))

    say("-" * 96)
    say(f"{len(rows)} arms")
    say()
    excludes = [r for r in rows if r[1] > 0.0]
    say(f"intervals excluding zero: {len(excludes)} of {len(rows)}")
    say("A lower end sitting above zero says the arm's difference is not")
    say("attributable to sampling. It says nothing about the MECHANISM, which is")
    say("what the predicted-delta column beside each cell is for.")
    say()
    for name, lo, hi, delta in rows:
        say(f"  {name:<32} delta {delta:.4f}  95% CI [{lo:.4f}, {hi:.4f}]"
            + ("" if lo > 0 else "   <- includes zero"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
