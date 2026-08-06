# conformal-coverage

Exact rank arithmetic for finite-sample distribution-free prediction bounds. Four
functions, no dependencies, stdlib only.

```bash
pip install conformal-coverage
```

```python
from conformal_coverage import (
    required_rank, delivered_coverage, feasibility_floor, conformal_threshold,
)

required_rank(100, 0.9)          # 91  -- the order statistic you need
delivered_coverage(90, 100)      # 0.8910...  -- what rank 90 actually gives
feasibility_floor(0.9)           # 9   -- below this, no valid finite bound exists
required_rank(8, 0.9)            # None
conformal_threshold(scores, 0.1) # the threshold, or +inf where none is valid
```

## Why this exists

For `n` scores exchangeable with a fresh one, the bound at rank `r` covers with
coverage of exactly `r / (n + 1)`, and no numerical library indexes that way. A quantile
function takes a probability, and each convention maps it to a position by its own
rule — so `numpy.quantile(scores, 0.9)` and
`numpy.quantile(scores, 0.9, method="higher")` land on different order statistics,
and neither is guaranteed to be the one the coverage claim requires.

`conformal_threshold` sidesteps the conversion. It indexes the sorted scores
directly, so no interpolation convention can move it.

## Two behaviours worth knowing before you use it

**It returns `+inf` instead of a number when no valid bound exists.** At `n = 10`
and `alpha = 0.05` the required rank is 11 of 10. There is no order statistic that
delivers 95% here, and any finite threshold would be a bound the data cannot
support. Returning infinity is the honest answer; widen the calibration set, lower
the confidence, or use a randomised bound.

**A threshold equal to `max(scores)` is not a bug.** Where the required rank is `n`,
returning the sample maximum *is* right. Treating that as evidence of a clamped level is
a mistake — one this package's authors made and had to retract.

## Exactness

Levels are converted with `Fraction(str(x))`, not `Fraction(x)`. The difference
matters: `Fraction(0.9)` is `8106479329266893/9007199254740992`, which is strictly
greater than `9/10`, and `ceil(10 * Fraction(0.9))` is 10 where the correct required
rank at `n = 9` is 9. Passing a `Fraction` directly bypasses the conversion.

The same programme found a shipped helper computing a feasibility floor as
`ceil(1/alpha - 1)` in floating point, where `1 - 0.90` is `0.09999999999999998` and
the floor came out one too high at every horizon.

## Checking it

```bash
python -m conformal_coverage
```

Runs the self-check: the coverage identity, agreement between `required_rank` and
`feasibility_floor` at the boundary from both sides, invariance to the float
spelling of a level across `n = 2..399`, and the `+inf` behaviour below the floor.

MIT licensed.
