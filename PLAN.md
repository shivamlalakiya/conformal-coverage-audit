# Research plan

**Status: measurement complete, no prose written yet.** This file is the scientific plan — the question,
the method, the protocol, and what is finished versus what is not. It is deliberately explicit about what
has *not* been established.

---

## 1. The question

Finite-sample distribution-free intervals are indexed by **order statistics**, and the index fixes the
coverage: with `n` calibration scores drawn exchangeably, index `r` carries `r / (n + 1)` exactly. That
identity is standard.

What libraries call to get such a bound is not indexed that way. It takes a **probability level**, and each
quantile convention turns that level into a position by its own rule. Hence:

> **Which rank does an implementation actually land on, and what coverage does that rank deliver?**

## 2. The claim being tested

> Whether a library applies the finite-sample `(n+1)/n` correction does not predict whether its intervals
> cover. The level→rank map does.

Supporting observation already measured on synthetic draws: two libraries that both omit the correction
sit **0.0000** and **0.1643** from nominal at comparable `n`. One passes an uncorrected level through a
rounding-based method whose landing point coincides with the required rank in a fifth to two-thirds of cells;
the other interpolates two separate quantiles of signed residuals, which lands between order statistics
rather than on one.

## 3. Method

| | Component | What it produces |
|---|---|---|
| **M1** | **The rank map.** For each quantile definition and each API, the delivered rank `r̂` as a closed form, and the deficit `Δ = ⌈(n+1)(1−α)⌉ − r̂` | A table. Non-ageing: a property of the APIs, not of any release |
| **M2** | **Delivered `n_min`.** The smallest `n` at which an implementation's bound is (a) valid and (b) non-degenerate, beside the theoretical minimum | A lookup for sizing a calibration split. One-sided and two-sided reported separately |
| **M3** | **Paired coverage on real data.** Library arm versus required-rank arm, same data, same base model, same residual set | The headline: delivered coverage against each library's own nominal level |
| **M4** | **Branch identification.** Non-smoothed conformal p-values lie exactly on the lattice `{j/(n+1)}`, which a quantile implementation cannot produce; threshold extraction separates a clamped level from a missing correction | Classification of a helper without curve fitting |

**Reference convention for M1 and M2:** the nine sample-quantile definitions catalogued by Hyndman and Fan
(*The American Statistician*, 1996) — three rounding-based, six interpolation-based — examined here **not
as estimators of a population quantile but as carriers of a coverage guarantee.** For each: can it express
`⌈(n+1)(1−α)⌉` at all, and if not, what does it deliver instead?

## 4. Unit of analysis

**The quantile helper, not the library.** One package can reach the bound by three unrelated code paths — a p-value path, a quantile path, and a path that never forms an order
statistic. One row per library cannot express that, and several published comparisons take
that shape.

## 5. Protocol for M3

Per series or dataset:

1. Split chronologically (or by a stated rule, for tabular data) into train / calibration / test.
2. **Arm A — the library**: called through its documented entry point, at the nominal level it names, with
   every default left alone. What ships is the object under study, not what could be configured.
3. **Arm B — the required rank**, using **the library's own point forecast and the library's own residual
   set**, changing only the level→rank step.
4. Record: covered or not, interval width, and the rank of the calibration scores that arm A's bound
   actually landed on.
5. Aggregate coverage across units, with paired standard errors.

**Width is reported alongside coverage.** A wider interval that covers is not the same result as a
correctly indexed one.

⚠️ **Attribution requirement, and it is the whole reason arm B exists.** Series drawn from a real archive
break exchangeability, so no absolute coverage figure can be laid at the convention's door by itself. Only the **paired delta**
between arms carries the claim. If the two arms differ in their residual set, their centre, or their base
model, the delta is meaningless — see §8.

## 6. Data

| Domain | Source | Rule |
|---|---|---|
| Forecasting | Monash Time Series Forecasting Archive — **M1 monthly and M3 monthly, 250 series each** | Selection rule stated before running; minimum series length fixed by the largest calibration window. Two independent collections, so archive-specific selection cannot explain a result |
| Tabular | **OpenML-CC18** (classification) and **OpenML-CTR23** (regression) | ≤ 5000 rows, ≤ 100 features after one-hot encoding, no missing values, first *N* by dataset id. Every skipped dataset is reported with its reason |
| Synthetic | iid draws | Retained deliberately: where the guarantee *should* hold exactly, any miss is unambiguous |

## 7. Reporting standard proposed by this work

Three items, short enough to be asked for in review:

1. **The delivered coverage**, measured, not the nominal level.
2. **The rank** the implementation lands on, or the quantile definition and level used to obtain it.
3. **The calibration-set size**, with one-sided and two-sided distinguished.

## 8. Phase status

| Phase | State |
|---|---|
| Convention in isolation, synthetic, exact-rational oracle | ✅ Complete |
| Structural branch identification and threshold extraction (M4) | ✅ Complete |
| Per-helper survey of thirteen packages, versions pinned, read at source | ✅ Complete |
| End-to-end synthetic runs for four libraries | ✅ Complete |
| **M1 rank map over the nine definitions** | ✅ Complete. Only 4 of 13 definitions deliver the requested guarantee at an uncorrected level; numpy's default `linear` and Hyndman–Fan's own recommendation deliver it at **no n ≤ 2000** |
| **M2 delivered `n_min` table** | ✅ Complete |
| **M3 paired real-data coverage** | ✅ Complete, four library arms on two independent Monash collections plus two OpenML suites. v1's attribution was invalid — its two arms did not share a residual set or a centre — and that failure is documented in the probe that replaced it rather than deleted |
| Tightening the one null result to ≥2000 fits across several calibration sizes | ✅ Complete. **There is no null**: the deficit alternates between 0 and 1 on a residue pattern, and a four-cell table that lands on the coincidence band shows zeros for arithmetic reasons |
| Conformance suite and the §7 checklist | ✅ Complete, `probes/conformance_suite.py` |
| Whether the mechanism generalises beyond conformal prediction | ✅ Complete, `probes/w8_falsification.py`. Reproduced under a value-at-risk framing and under a Wilks framing; under bootstrap resampling the index term is swamped |
| Per-helper count under a stated criterion | ✅ Complete, `probes/helper_census.py` |
| **Whether M3 depends on the series chosen or on one point per series** | ✅ Complete, `probes/sample_robustness.py`. Re-run over all 1093 usable monthly series of 1428 (the rest are shorter than the darts arm's floor) with a rolling origin, standard errors clustered by series. Selection and resolution are now measured rather than conceded; **exchangeability still is not, and no archive size can repair it** |

### 8.1 What the measurement found, in one paragraph

The level→rank map, not the presence of the `(n+1)/n` correction, predicts whether an interval covers.
**Seven tabular implementations, each given identical scores and an identical split, reach the same bound
with a paired delta of exactly zero in six cases**; the seventh sits above nominal. The forecasting libraries are where
the deficit lives, and there the size of it is set by which order statistic the level lands on — one
library's deficit alternates between zero and one rank on a residue pattern in `n`, so the same code path
hits the required rank at some calibration sizes and falls one short at others. Separately, one library's **default**
configuration calibrates on two residuals, a size admitting no valid finite bound at any conventional level,
and returns a finite interval regardless.

## 9. Verification rules applied to everything here

These are not aspirations; they have each already caught a real error in this work.

- **Each formula is re-derived under `fractions.Fraction` the moment its module loads.** A failing check aborts
  the run. Two such checks caught errors in their author's own hand-derived assertions.
- **A grid is chosen by the bug, not by the author.** Every sweep runs on well past the first boundary,
  to at least double it. Stopping a grid just beyond the first boundary is worse than stopping well short of
  it, because the short grid does not look finished and the other one does.
- **Include a level that is not a unit fraction in every sweep.** Claims of the form "this only happens on that residue
  class" are usually artifacts of sweeping only `1/10`, `1/20`, `1/100`.
- **Never check an implementation against something that shares its convention.** And a fixture pinned to a
  cell with no discriminating power is worse than no fixture at all.
- **Label one-sided and two-sided on every number, in the table, not only in the script.** `n/(n+1)` and
  `(n−1)/(n+1)` differ by fourteen points at `n = 6`.
- **Establish the direction of harm before calling anything a defect**, and check that the minimal patch
  you have in mind does not itself raise.
- **Verify by running.** Reading has produced confident false claims in this work more than once.

## 10. Known limitations

- Synthetic harnesses use iid draws with a deliberately simple base model. They establish that a miss
  occurs where the guarantee should hold exactly; they say nothing about dependent real-world series.
  That is what M3 is for.
- Non-exchangeability generally biases toward **over**coverage, so a synthetic result understates rather
  than overstates.
- On real data the **absolute** coverage is not attributable to the convention. Only the paired delta is.
  Every real-data table here reports the delta and its standard error for that reason. This is the one
  limitation the robustness probe explicitly does **not** remove: running every series in the archive
  answers "you chose the series", and a rolling origin answers "one point per series", but neither makes
  a real series exchangeable.
- The rolling origins of one series share their history, so they are not independent test points.
  `sample_robustness.py` clusters every standard error by series for that reason and prints the naive
  figure beside it, so the size of the dependence is visible instead of argued.
- Findings are version-pinned. The rank map (M1) is not, which is why it is the theory core.
- One library's calibration residuals are computed from a model already fitted on the whole input series,
  so they are in-sample and optimistically small. That bias is measured **separately** from the level→rank
  map, in `probes/darts_scoring_path.py`, rather than being folded into a single number.
- Under bootstrap resampling the generalisation check found the index term **swamped** by percentile
  intervals. The mechanism generalises; the effect sizes measured here do not.
- R, Julia and Octave **are** executed, not transferred: `quantile(type = 7)` (R's default, identical to
  numpy's `linear`), `Statistics.quantile(alpha=, beta=)` and Octave's nine methods all agree with the
  instrument on every documented `(α, β)` pair. Octave's **default** is method 5 (hazen), not linear, so
  a fourth ecosystem default sits off the common one. MATLAB proper is not run — Octave implements the
  same nine methods — and pyspark is out of scope for the reason given in §11.
- One finding previously asserted in this work was **retracted** after three independent checks
  contradicted it. The probe that adjudicated it is in this repository. Several later claims were
  retracted the same day they were made, by exact-arithmetic checks written to test them; the most
  instructive is that a returned threshold equal to `max(scores)` is **not** evidence of a clamped level,
  because where the required rank *is* `n`, returning the sample maximum is right.

## 11. Out of scope

- Closed-source or vendor implementations: not inspectable, and not claimed about.
- Packages that never index into a sorted calibration set — learned-quantile models, and probability
  intervals from isotonic calibration. Excluded with the reason stated rather than silently omitted.
- Sample-path simulators that produce trajectories rather than a split-conformal interval.
- Which method is *best*. This is a correctness audit, not a benchmark of predictive performance.

## 11a. A warning for anyone extending this to a literature census

A search over conformal-prediction papers was run and is deliberately not reported here or in
the write-up: the counts come from a network sweep over a corpus that cannot be pinned, so they
could not carry the guarantee every committed output in this repository carries. Two things
found during it are worth passing on.

- One paper in the corpus (arXiv:2407.06658v3) carries a prompt injection inside its Methods
  text, instructing an AI reader to give a positive review. Treated as data and ignored. A
  regex sweep over local text is structurally immune; an extraction pipeline that puts a
  language model in the loop is not.
- Fetch-and-summarise tooling produced a false negative and a *fabricated* derivation on this
  corpus. Every figure that survived was re-verified against raw source.

## 12. What is in this repository

The probes and their committed outputs, listed by purpose in `README.md` — a count is deliberately not
quoted here, because it went stale twice and `README.md` is the list that has to be right. Each formula is
re-derived under `fractions.Fraction` at module load, and a failure stops execution there.

Not included, each for a stated reason: third-party library sources (pinned in
`probe-requirements.txt` instead of redistributed), and the `.npz` series cache one probe writes (the two
commands that regenerate it are in `README.md`).
