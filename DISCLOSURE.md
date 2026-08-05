# Disclosure manifest

Every defect this audit reports as a defect, as filed upstream, with the date filed.
The manuscript is **anonymised for review**, so it cites this file rather than inlining
the numbers — a public issue identifies its author. The camera-ready inlines them.

This file is the timestamped record. Each row's date is verifiable upstream, which is what
makes the ordering — report first, publish second — a checkable fact rather than an
assurance. It is generated to match the write-up's own table, and the build fails if the
two disagree.

⚠️ **An unlisted site is not a site certified correct.** 35 resolution sites were
surveyed and far fewer are reportable defects, by design. Two reasons keep a site off
this table: its measured behaviour is documented and deliberate, or it was newly found
here and filing it would spend a maintainer's attention for no fix. Which of the two
applies is stated per site in the next section.

## Filed

| Date | Package | Site | Report | Status |
|---|---|---|---|---|
| 2026-08-02 | `mapie` | classification quantile lands one order statistic too high | [scikit-learn-contrib/MAPIE#973](https://github.com/scikit-learn-contrib/MAPIE/pull/973) (PR) | open |
| 2026-08-02 | `mapie` | corrected level silently clipped to `1.0` where no conformal quantile exists | [scikit-learn-contrib/MAPIE#974](https://github.com/scikit-learn-contrib/MAPIE/issues/974) | open |
| 2026-08-04 | `mapie` | asymmetric quantile score cancels the calibration-size guard | [scikit-learn-contrib/MAPIE#978](https://github.com/scikit-learn-contrib/MAPIE/pull/978) (PR) | open |
| 2026-07-10 | `mapie` | eight inline findings raised in review on a third party's PR; merged with the blocking one unfixed, which #978 addresses | [scikit-learn-contrib/MAPIE#958](https://github.com/scikit-learn-contrib/MAPIE/pull/958) (review) | merged |
| 2026-08-02 | `crepes` | docstring inverts the implemented set-membership condition | [henrikbostrom/crepes#46](https://github.com/henrikbostrom/crepes/issues/46) | open |
| 2026-08-02 | `crepes` | no warning on a too-small calibration set | [henrikbostrom/crepes#47](https://github.com/henrikbostrom/crepes/issues/47) | open |
| 2026-08-02 | `crepes` | membership compared against a float threshold | [henrikbostrom/crepes#48](https://github.com/henrikbostrom/crepes/issues/48) | open |
| 2026-08-02 | `crepes` | fix for #46 | [henrikbostrom/crepes#49](https://github.com/henrikbostrom/crepes/pull/49) (PR) | open |
| 2026-08-02 | `crepes` | fix for #47 | [henrikbostrom/crepes#50](https://github.com/henrikbostrom/crepes/pull/50) (PR) | open |
| 2026-08-04 | `sktime` | `ConformalIntervals method="empirical_residual"` takes the wrong tail | [sktime/sktime#10757](https://github.com/sktime/sktime/issues/10757), fix in [#10765](https://github.com/sktime/sktime/pull/10765) (PR, 2026-08-05) | open |
| 2026-08-04 | `sktime` | `conformal` and `conformal_bonferroni` omit the `(m+1)` correction | [sktime/sktime#10758](https://github.com/sktime/sktime/issues/10758) | open |
| 2026-08-05 | `sktime` | `ConformalIntervals._compute_sliding_residuals` / `_predict_interval_series` — residual-alignment off-by-one: a step-*h* forecast is calibrated on (h+1)-step residuals | [sktime/sktime#10766](https://github.com/sktime/sktime/issues/10766) | open |
| 2026-08-04 | `statsforecast` | `ConformalSeasonalPool`: documented sufficiency rule covers the lower rail only, and its worked example is off by one | [Nixtla/statsforecast#1202](https://github.com/Nixtla/statsforecast/issues/1202) | open |
| 2026-08-04 | `torchcp` | `calculate_conformal_value` docstring names a threshold the code no longer computes | [ml-stat-Sustech/torchcp#122](https://github.com/ml-stat-Sustech/torchcp/pull/122) (PR) | merged 2026-08-05 |

Filing window: **2026-08-02 to 2026-08-05**, other than the #958 review, which is
2026-07-10 and is a set of findings raised on someone else's pull request rather than a
report filed by us.

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
   variance *h*. Measured over 300 series: **0.978, 1.899, 2.782** for offsets 0, 1, 2 —
   matching *k+1*, not *k*.
3. **Direct spy.** Wrapping `np.quantile` inside `_predict_interval_series` shows
   `predict_interval(fh=[1])` resolving its level from `|diag(offset=1)|`.

Direction: **conservative.** Residuals grow with the horizon on an integrated series, so
the intervals come out too wide — by roughly `sqrt(2)` at `fh=[1]` on a random walk.
Coverage tests therefore pass, which is why it survived. It is orthogonal to the
level-to-rank map and its sign is reversed, so the two defects partly cancel inside the
shipped helper. Hence the four measurement arms in the write-up instead of a single
number.

Two fixes were offered — `offset-1` in the consumer (smaller diff), or starting
`y_test` at `id+1` in the producer (makes the docstring true) — with an offer to open a
PR for whichever the maintainers prefer.

⚠️ The variance figures above read `0.993, 1.884, 2.762` when the issue was first filed.
Those came from an earlier run and did not match the committed probe output. The issue
body was corrected on 2026-08-05, with the change recorded in a comment rather than
edited silently, and this manifest now carries the committed values.

## Not filed, and why — stated as policy, not left open

The level-to-rank findings in the census are overwhelmingly **pre-existing and
documented**: a library resolving an uncorrected level through a rounding definition is
doing something its own documentation describes. Where such a site is already public, the
census output cites its existing issue or pull-request number rather than re-filing it.

For the 16 sites this audit newly located, the rule applied is: **file where a maintainer
would have to change code or documentation to make the shipped behaviour match its own
stated contract; do not file where the behaviour is intended, documented, and merely
lossy.** Every filing in the table above meets the first test. The remainder do not, and
are reported in the manuscript as measurements rather than as defects.

One site is worth naming because it is the paper's largest measured shortfall and is
nonetheless **not** filed as a defect:

- `statsforecast` `ConformalIntervals(n_windows=2)`. At two calibration windows no valid
  finite *deterministic* bound exists at any conventional level, and the library returns a
  finite interval regardless — delivering 0.5840 against a nominal 0.90. This is a
  **default-choice** question rather than a code defect: the arithmetic is correct for the
  configuration, and the configuration is the problem. `Nixtla/statsforecast#1202` covers
  the documentation defect in the same helper. The manuscript's §"What exists below the
  floor" gives the constructive answer — a randomised bound that is exact at every window
  count — rather than treating the configuration as a bug report.

Retracted findings are not filed at all, and are recorded in the manuscript's retraction
section instead.
