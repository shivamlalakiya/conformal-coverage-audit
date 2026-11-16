#!/usr/bin/env python3
"""Run torchcp's two weighted resolution sites rather than read them.

A second reader found two torchcp expressions that map the requested level onto an
index of the calibration scores and were missing from the census:

  WeightedPredictor.predict      classification/predictor/weight.py:120
      i_T = argmax(cumsum(normalised weights, test weight appended) >= 1 - alpha),
      gathered from the sorted scores with +inf appended
  NAPSPredictor._get_weighted_quantile   graph/predictor/naps.py:214-221
      brentq root of sum(w/(sum w + 1) * [score <= q]) - (1 - alpha) on [-1000, 1000]

Both reduce to split conformal when every weight is one, so the score set 1..n with
unit weights is the case where the required rank k* = ceil((n+1)(1-alpha)) is known
and either expression can be scored against it. This probe calls the NAPS method
itself and replays weight.py:115-121 line for line (the method needs an image
encoder to reach that line, which is beside the point of the expression).

What reading suggested and running shows differ at the floor n = 1/alpha - 1, where
k* = n: the weighted predictor's float cumulative sum lands a hair under 1 - alpha
at index n-1 and returns +inf, and the NAPS root-finder meets f(1000) == 0 and
returns its bracket edge, 1000, rather than the n-th score. Below the floor the
first returns +inf and the second raises ValueError.

    .venv-tabular/bin/python probes/torchcp_weighted_sites.py   # torch lives there
"""

import math
import os
import warnings

import torch
import torchcp
from torchcp.graph.predictor.naps import NAPSPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "probe_output_torchcp_weighted_sites.txt")

ALPHAS = (0.1, 0.05)
NS = tuple(range(2, 61))


def required_rank(n, alpha):
    k = math.ceil(round((n + 1) * (1 - alpha), 9))
    return k if k <= n else None


def naps_site(n, alpha):
    s = torch.arange(1, n + 1, dtype=torch.float64)
    w = torch.ones(n, dtype=torch.float64)
    try:
        return float(NAPSPredictor._get_weighted_quantile(None, s, w, alpha))
    except ValueError:
        return "raise"


def weighted_site(n, alpha):
    # weight.py:115-121 with every calibration weight and the test weight equal to one
    w_sorted = torch.ones(1, n, dtype=torch.float64)
    w_new = torch.ones(1, dtype=torch.float64)
    w_sorted = torch.cat([w_sorted, w_new.unsqueeze(1)], 1)
    p_sorted = w_sorted / w_sorted.sum(1, keepdim=True)
    p_sorted_acc = p_sorted.cumsum(1)
    i_T = torch.argmax((p_sorted_acc >= 1.0 - alpha).int(), dim=1, keepdim=True)
    scores_sorted = torch.cat([torch.arange(1, n + 1, dtype=torch.float64),
                               torch.tensor([math.inf], dtype=torch.float64)])
    return float(scores_sorted.unsqueeze(0).gather(1, i_T))


def classify(got, k, n):
    """Score a threshold by its position h on the score set 1..n, against k*.

    On 1..n a threshold's value is its virtual index, and by the fractional-rank
    identity a threshold at h delivers h/(n+1) on a flat density, so h - k* is the
    excess in ranks. A root-finder returns a point within its tolerance of a jump,
    on either side, so a difference under 1e-9 is reported as exact; anything larger
    is printed with its sign."""
    if got == "raise":
        return "raise"
    if math.isinf(got):
        return "inf"
    if got >= 999:
        return "bracket"
    if k is None:
        return f"finite h={got:.3f}"
    d = got - k
    return "exact" if abs(d) < 1e-9 else f"h-k={d:+.3f}"


def self_check():
    # The scorer can fail: one rank short, a fractional excess and the bracket
    # edge are each told apart from exact.
    assert classify(45.0, 46, 50) == "h-k=-1.000"
    assert classify(46.0 - 1e-12, 46, 50) == "exact"
    assert classify(46.355, 46, 50) == "h-k=+0.355"
    assert classify(1000.0, 9, 9) == "bracket"
    assert required_rank(9, 0.1) == 9 and required_rank(8, 0.1) is None
    assert required_rank(50, 0.1) == 46


self_check()


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 96)
    say("TORCHCP'S TWO WEIGHTED RESOLUTION SITES, RUN ON UNIT WEIGHTS")
    say("=" * 96)
    say(f"torchcp {torchcp.__version__}; score set 1..n, every weight one, so k* is known")
    say("exact   = returns the k*-th score;  inf = returns +inf;  raise = ValueError;")
    say("bracket = returns the root-finder's bracket edge (1000), above every score")
    say("h-k=d   = the threshold sits d ranks from k*; a positive d over-covers by d/(n+1)")
    say("A row is printed wherever either site is not exact.")
    say("")
    tally = {}
    for alpha in ALPHAS:
        floor = math.ceil(round(1 / alpha - 1, 9))
        for n in NS:
            k = required_rank(n, alpha)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                qa, qb = naps_site(n, alpha), weighted_site(n, alpha)
                a, b = classify(qa, k, n), classify(qb, k, n)
            region = "below" if k is None else ("at" if n == floor else "above")
            for site, got in (("naps", a), ("weighted", b)):
                tally[(alpha, site, region, got)] = tally.get(
                    (alpha, site, region, got), 0) + 1
            if region != "above" or a != "exact" or b != "exact":
                say(f"ROW alpha={alpha} n={n} region={region} k={k if k else 'none'} "
                    f"naps={a} weighted={b} naps_q={qa if qa == 'raise' else repr(qa)} "
                    f"warnings={len(caught)}")
    say("")
    say("TALLY over n in 2..60 (alpha, site, region relative to the floor, result: count)")
    for key in sorted(tally, key=str):
        alpha, site, region, got = key
        say(f"TALLY alpha={alpha} site={site} region={region} result={got} "
            f"count={tally[key]}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
