"""The rank arithmetic of a finite-sample distribution-free bound.

For ``n`` scores exchangeable with a fresh one, the bound at rank ``r`` covers with
probability exactly ``r / (n + 1)``. Nothing in a numerical library is indexed that way. A quantile
function takes a probability, and each convention maps it to a position
differently, so the guarantee is lost in the mapping.

Four functions, no dependencies, exact arithmetic. Vendor the file if that is
easier than adding a requirement.

    >>> required_rank(100, 0.9)
    91
    >>> delivered_coverage(90, 100)
    0.8910891089108911
    >>> feasibility_floor(0.9)
    9
    >>> required_rank(8, 0.9) is None      # below the floor, no valid bound
    True

Everything here is computed with ``fractions.Fraction``. That is not decoration:
the same programme found a shipped helper that computed a feasibility floor as
``ceil(1/alpha - 1)`` in floating point, where ``1 - 0.90`` is
``0.09999999999999998`` and the floor came out one too high.
"""

import math
from fractions import Fraction

__all__ = ["required_rank", "delivered_coverage", "feasibility_floor",
           "conformal_threshold"]
__version__ = "0.1.0"


def _exact(x):
    """A level as an exact rational, via its shortest decimal form."""
    return x if isinstance(x, Fraction) else Fraction(str(x))


def required_rank(n, coverage):
    """Smallest rank delivering at least ``coverage`` from ``n`` scores.

    Returns the 1-based rank ``ceil((n + 1) * coverage)``, or ``None`` when that
    exceeds ``n`` -- in which case no valid finite *deterministic* bound exists at
    this size, and returning a finite threshold anyway is the defect this package
    exists to make easy to avoid.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    k = math.ceil((n + 1) * _exact(coverage))
    return None if k > n else k


def delivered_coverage(rank, n):
    """Coverage of the bound at 1-based ``rank``: exactly ``rank / (n + 1)``."""
    if not 0 <= rank <= n:
        raise ValueError(f"rank {rank} outside 0..{n}")
    return rank / (n + 1)


def feasibility_floor(coverage):
    """Smallest ``n`` at which a valid finite deterministic bound exists.

    ``ceil(1/alpha - 1)`` with ``alpha = 1 - coverage``, computed exactly.
    """
    alpha = 1 - _exact(coverage)
    if alpha <= 0:
        raise ValueError("coverage must be below 1")
    return math.ceil(Fraction(1) / alpha - 1)


def conformal_threshold(scores, alpha):
    """The conformal threshold at miscoverage ``alpha``, or ``+inf``.

    Lands on the required order statistic by construction rather than by asking a
    quantile function for a level, so no interpolation convention can move it.
    Returns ``float('inf')`` where no valid finite bound exists, which is the
    honest answer at that size -- a finite number there would be a bound the data
    cannot support.
    """
    ordered = sorted(scores)
    n = len(ordered)
    if n == 0:
        raise ValueError("no calibration scores")
    k = required_rank(n, 1 - _exact(alpha))
    return float("inf") if k is None else ordered[k - 1]


def _self_check():
    """Run me. Cheap, and each assertion here failed something once."""
    # the identity the whole package is about
    assert required_rank(100, 0.9) == 91
    assert delivered_coverage(91, 100) == 91 / 101

    # the floor, and that required_rank agrees with it at the boundary
    for cov in (Fraction(9, 10), Fraction(19, 20), Fraction(99, 100), Fraction(2, 3)):
        fl = feasibility_floor(cov)
        assert required_rank(fl, cov) is not None, cov
        assert required_rank(fl - 1, cov) is None, cov

    # exactness: a float level must not move the answer
    for n in range(2, 400):
        for cov in (0.9, 0.95, 0.99, 2 / 3):
            assert required_rank(n, cov) == required_rank(n, Fraction(str(cov)).limit_denominator(10**6))

    # the drop-in returns +inf exactly below the floor, and an order statistic above
    scores = [float(i) for i in range(1, 11)]           # n = 10
    assert conformal_threshold(scores, 0.05) == float("inf")   # needs rank 11 of 10
    assert conformal_threshold(scores, 0.10) == 10.0           # needs rank 10 of 10
    assert conformal_threshold(scores, 0.5) == 6.0             # ceil(11*0.5) = 6

    # a threshold equal to max(scores) is NOT evidence of a clamp: where the
    # required rank is n, the maximum is the correct answer. This cost a retraction.
    assert conformal_threshold(scores, 0.10) == max(scores)

    print("conformal_coverage self-check passed")


if __name__ == "__main__":
    _self_check()
