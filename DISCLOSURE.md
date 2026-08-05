# Disclosure manifest

Upstream reports for every defect this audit claims, one row per site, with the date
filed. The manuscript is **anonymised for review**, so it cites this file rather than
inlining the numbers — a public issue identifies its author. The camera-ready inlines
them.

This file is the timestamped record: the filing dates can be checked against the
submission date to confirm every defect was reported before it was published.

## Filed

| Date | Package | Site | Report | Status |
|---|---|---|---|---|
| 2026-08-05 | `sktime` | `ConformalIntervals._compute_sliding_residuals` / `_predict_interval_series` — residual-alignment off-by-one: a step-*h* forecast is calibrated on (h+1)-step residuals | [sktime/sktime#10766](https://github.com/sktime/sktime/issues/10766) | open |

### sktime#10766 — what was reported

The consumer slices `np.diagonal(residuals_matrix, offset=h)` for relative horizon
step *h*, which is correct **given the matrix's documented contract**:

> `[i,j]`-th entry is signed residual of forecasting `y.loc[j]` from `y.loc[:i]`

`y.loc[:i]` is inclusive of `i`. But the producer builds rows with
`get_slice(y, start=None, end=id)`, which is **end-exclusive**, so row `id` trains on
`y[:id)` with last training observation `y[id-1]`. That makes `diag(offset=0)` already
the one-step residual set, and `diag(offset=h)` the **(h+1)-step** set.

Verified three independent ways, all in `probes/horizon_feasibility.py` block (0) and
in the reproducer attached to the issue:

1. **Index spans.** Solving each diagonal entry for the index pair that produces it
   gives horizon spans 1, 2, 3 for offsets 0, 1, 2.
2. **Variance.** On a unit-innovation random walk an *h*-step last-value residual has
   variance *h*. Measured over 300 series: 0.993, 1.884, 2.762 for offsets 0, 1, 2 —
   matching *k+1*, not *k*.
3. **Direct spy.** Wrapping `np.quantile` inside `_predict_interval_series` shows
   `predict_interval(fh=[1])` resolving its level from `|diag(offset=1)|`.

Direction: **conservative.** For an integrated series a longer-horizon residual is
larger, so intervals are too wide — roughly `sqrt(2)` too wide at `fh=[1]` on a random
walk. Coverage tests therefore pass, which is why it survived. It is orthogonal to the
level-to-rank map and points the opposite way, so in the shipped helper the two defects
partly cancel; that is why the manuscript separates them into four arms rather than
reporting one number.

Two fixes were offered — `offset-1` in the consumer (smaller diff), or starting
`y_test` at `id+1` in the producer (makes the docstring true) — with an offer to open a
PR for whichever the maintainers prefer.

## Not yet filed

The level-to-rank findings in the census are pre-existing and, where already public,
are cited by their existing issue or pull-request numbers in the census output rather
than re-filed. Sites this audit newly located and has **not** yet reported upstream
should be listed here before submission. Retracted findings are not filed at all and
are recorded in the manuscript's retraction section instead.

- [ ] Review the \cenNew{} newly located sites and decide, per site, whether it is a
      defect worth a maintainer's time or a documentation gap. Not every uncorrected
      level is a bug report; several are deliberate and documented.
- [ ] `statsforecast` `ConformalIntervals(n_windows=2)` default: no valid finite
      deterministic bound exists at that size for any conventional level. Worth
      reporting as a default-choice issue rather than a code defect.
