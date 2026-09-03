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
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiplicity_and_reimpl import FAMILY, read_cell        # noqa: E402

Z95 = 1.959963984540054          # the two-sided normal quantile, not a measurement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The cells that reach a typeset table, named by the file that measured each and
# by the selectors that pick one block out of it. Selectors, not measurements:
# every number comes back from read_block, which asserts it found exactly the
# block asked for. The Holm family above is a subset of this list; it is kept
# separate because it answers a different question and its membership is fixed by
# what the permutation test can run on, not by what a table prints.
#
# Two kinds of cell appear here that the family excludes on purpose, and both get
# an interval because the arithmetic applies to them unchanged. A cell whose arm B
# is vacuous everywhere has a difference that is one minus arm A's coverage --
# still a count of units over units, so still a binomial proportion, and the
# interval is an interval on infeasibility rather than on the level-to-rank map.
# A coincidence cell has gains of zero, and the interval that comes back starts at
# zero and has width, which is the correct statement about a difference that is
# zero by arithmetic rather than zero by measurement.
TABLE_CELLS = [
    # archive, library, cell, path, nominal, method-line, block-line
    ("m1", "sktime", "20", "outputs/probe_output_real_data.txt",
     "0.90", None, "empirical  nominal 0.90  initial_window=20"),
    ("m1", "sktime", "40", "outputs/probe_output_real_data.txt",
     "0.90", None, "empirical  nominal 0.90  initial_window=40"),
    ("m3", "sktime", "20", "outputs/probe_output_real_data_m3_monthly.txt",
     "0.90", None, "empirical  nominal 0.90  initial_window=20"),
    ("m3", "sktime", "40", "outputs/probe_output_real_data_m3_monthly.txt",
     "0.90", None, "empirical  nominal 0.90  initial_window=40"),

    ("m1", "statsforecast", "2", "outputs/probe_output_real_data_statsforecast.txt",
     "0.90", "conformal_distribution", "n_windows=2 "),
    ("m1", "statsforecast", "5", "outputs/probe_output_real_data_statsforecast.txt",
     "0.90", "conformal_distribution", "n_windows=5 "),
    ("m1", "statsforecast", "10", "outputs/probe_output_real_data_statsforecast.txt",
     "0.90", "conformal_distribution", "n_windows=10 "),
    ("m1", "statsforecast", "20", "outputs/probe_output_real_data_statsforecast.txt",
     "0.90", "conformal_distribution", "n_windows=20 "),
    ("m1", "statsforecast", "50", "outputs/probe_output_real_data_statsforecast.txt",
     "0.90", "conformal_distribution", "n_windows=50 "),
    ("m3", "statsforecast", "2",
     "outputs/probe_output_real_data_statsforecast_m3_monthly.txt",
     "0.90", "conformal_distribution", "n_windows=2 "),
    ("m3", "statsforecast", "10",
     "outputs/probe_output_real_data_statsforecast_m3_monthly.txt",
     "0.90", "conformal_distribution", "n_windows=10 "),

    ("m1", "darts", "10", "outputs/probe_output_real_data_darts.txt",
     "0.90", None, "cal_length=10 "),
    ("m1", "darts", "15", "outputs/probe_output_real_data_darts.txt",
     "0.90", None, "cal_length=15 "),
    ("m1", "darts", "30", "outputs/probe_output_real_data_darts.txt",
     "0.90", None, "cal_length=30 "),
    ("m1", "darts", "35", "outputs/probe_output_real_data_darts.txt",
     "0.90", None, "cal_length=35 "),
    ("m1", "darts", "50", "outputs/probe_output_real_data_darts.txt",
     "0.90", None, "cal_length=50 "),
    ("m1", "darts", "55", "outputs/probe_output_real_data_darts.txt",
     "0.90", None, "cal_length=55 "),
    ("m3", "darts", "15", "outputs/probe_output_real_data_darts_m3_monthly.txt",
     "0.90", None, "cal_length=15 "),
    ("m3", "darts", "30", "outputs/probe_output_real_data_darts_m3_monthly.txt",
     "0.90", None, "cal_length=30 "),
]

_NOMINAL = re.compile(r"nominal (\d\.\d+)")
_METHOD = re.compile(r"method=(conformal_\w+)")
_DELTA = re.compile(r"paired delta \(B - A\)\s+([-+][\d.]+)\s+\(s\.e\. ([\d.]+)\)")
_COUNTS = re.compile(
    r"status changed in \d+ of (\d+) units: gains=(\d+) losses=(\d+)")

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


def read_block(path, nominal, method, blockline):
    """One cell's units, gains, losses, delta and s.e., by three selectors.

    read_cell in the multiplicity probe tracks the nominal level and nothing else,
    which is enough for the eight arms it was written for. It is not enough here:
    one statsforecast output carries two methods at the same nominal level, and a
    block line naming only the window count matches the first of them whichever it
    is. So the method is tracked as well and asserted, and the block line has to
    match after both selectors already hold. A file that stops carrying the block
    this asks for raises rather than returning the neighbouring one.
    """
    with open(os.path.join(ROOT, path)) as fh:
        lines = fh.read().splitlines()
    cur_nom = cur_meth = None
    for idx, ln in enumerate(lines):
        n = _NOMINAL.search(ln)
        if n:
            cur_nom = n.group(1)
        mm = _METHOD.search(ln)
        if mm:
            cur_meth = mm.group(1)
        if cur_nom != nominal:
            continue
        if method is not None and cur_meth != method:
            continue
        if blockline not in ln:
            continue
        delta = se = counts = None
        for nxt in lines[idx:idx + 12]:
            d = _DELTA.search(nxt)
            if d and delta is None:
                delta, se = float(d.group(1)), float(d.group(2))
            c = _COUNTS.search(nxt)
            if c and counts is None:
                counts = tuple(int(x) for x in c.groups())
        assert counts is not None and delta is not None, (path, blockline, nominal)
        units, gains, losses = counts
        assert abs(delta - (gains - losses) / units) < 5e-5, (
            f"{path} {blockline}: printed delta {delta} is not "
            f"(gains-losses)/units; this block is not one point per unit")
        return {"units": units, "gains": gains, "losses": losses,
                "delta": delta, "se": se}
    raise AssertionError(
        f"no {nominal} block matching {blockline!r} in {path}"
        + (f" under method={method}" if method else ""))


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

    # The method selector has to be load-bearing or it is decoration. One
    # statsforecast output carries the same window count under two methods at the
    # same nominal level, and the two disagree; if they ever stop disagreeing this
    # check fails and says so, rather than letting the selector quietly stop
    # mattering. This is the failing input for read_block's third selector.
    sf = "outputs/probe_output_real_data_statsforecast.txt"
    a = read_block(sf, "0.90", "conformal_distribution", "n_windows=10 ")
    b = read_block(sf, "0.90", "conformal_error", "n_windows=10 ")
    assert a["gains"] != b["gains"], (
        "the two statsforecast methods return the same discordant count at "
        "n_windows=10, so the method selector no longer distinguishes them")
    try:
        read_block(sf, "0.90", "conformal_distribution", "n_windows=7 ")
    except AssertionError:
        pass
    else:                                            # pragma: no cover
        raise AssertionError("read_block returned a block that is not in the file")


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

    say()
    say("=" * 96)
    say("EVERY CELL THAT REACHES A TYPESET TABLE")
    say("=" * 96)
    say("The eight arms above are the family a correction is applied across. The")
    say("rows below are the cells a reader sees in a table, which is a larger set:")
    say("it adds the two archives' vacuous cells, where the difference is one minus")
    say("arm A's coverage and the interval is an interval on infeasibility, and the")
    say("coincidence cells, where gains is zero and a proper score interval still")
    say("has width. Same arithmetic on both, from the same integer counts.")
    say()
    say("A cell is picked by three selectors and never by position: the archive's")
    say("own output file, the nominal level, the method where one output carries")
    say("more than one, and the block line. Machine-readable, one line per cell.")
    say()
    n_zero = 0
    for arch, lib, cell, path, nominal, method, blockline in TABLE_CELLS:
        c = read_block(path, nominal, method, blockline)
        assert c["losses"] == 0, (
            f"{arch} {lib} {cell}: losses={c['losses']}, so the arms are not "
            f"nested here and a binomial interval on the difference does not "
            f"apply. This probe refuses rather than printing one")
        lo, hi = wilson(c["gains"], c["units"])
        delta = (c["gains"] - c["losses"]) / c["units"]
        assert abs(delta - c["delta"]) < 5e-5, (arch, lib, cell, delta, c["delta"])
        # The containment check needs a tolerance at exactly one end, and the
        # reason is arithmetic rather than statistics: at zero discordant units
        # the centre and the half-width agree to the last bit instead of exactly,
        # so the lower end lands a few times 1e-19 above the zero it should be and
        # a bare `lo <= delta` fails on a cell whose interval is right. The
        # self_check above pins the same behaviour at the boundary.
        assert lo - 1e-12 <= delta <= hi + 1e-12, (arch, lib, cell, lo, delta, hi)
        if c["gains"] == 0:
            n_zero += 1
        say(f"CELL arch={arch} lib={lib} cell={cell} units={c['units']} "
            f"gains={c['gains']} delta={delta:+.4f} lo={lo:.4f} hi={hi:.4f} "
            f"se={c['se']:.4f}")
    say()
    say(f"{len(TABLE_CELLS)} cells; {n_zero} with a discordant count of zero, whose")
    say("interval starts at zero because the difference is zero by arithmetic.")
    say()
    say("Every interval above contains its own cell's paired difference, asserted")
    say("per row. A row whose interval did not would be an interval read off a")
    say("neighbouring block, which is the failure the three selectors exist for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
