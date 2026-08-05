# Research plan

**Status: measurement in progress. No prose written yet.** This file is the scientific plan — the
question, the method, the protocol, and what is finished versus what is not. It is deliberately explicit
about what has *not* been established.

---

## 1. The question

A finite-sample distribution-free prediction interval is a statement about an **order statistic**. For `n`
exchangeable calibration scores, an interval built on rank `r` covers a fresh observation with probability
exactly `r / (n + 1)`. That identity is standard.

The APIs libraries use to obtain that bound do not accept a rank. They accept a **level**, and every
quantile function maps a level to a position under some interpolation convention. So:

> **Which rank does an implementation actually land on, and what coverage does that rank deliver?**

## 2. The claim being tested

> Whether a library applies the finite-sample `(n+1)/n` correction does not predict whether its intervals
> cover. The level→rank map does.

Supporting observation already measured on synthetic draws: two libraries that both omit the correction
sit **0.0000** and **0.1643** from nominal at comparable `n`. One passes an uncorrected level through a
rounding-based quantile method that coincides with the required rank in a fifth to two-thirds of cells;
the other takes two separate interpolated quantiles of signed residuals and lands on no order statistic
at all.

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

**The quantile helper, not the library.** A single package can resolve the bound three different ways in
three different code paths — a p-value path, a quantile path, and a path that never forms an order
statistic. Any table with one row per library cannot express that, and several published comparisons have
that shape.

## 5. Protocol for M3

Per series or dataset:

1. Split chronologically (or by a stated rule, for tabular data) into train / calibration / test.
2. **Arm A — the library**, through its public API, at its own nominal level, with its own defaults.
   The subject is what ships, not what is achievable.
3. **Arm B — the required rank**, using **the library's own point forecast and the library's own residual
   set**, changing only the level→rank step.
4. Record: covered or not, interval width, and the rank of the calibration scores that arm A's bound
   actually landed on.
5. Aggregate coverage across units, with paired standard errors.

**Width is reported alongside coverage.** A wider interval that covers is not the same result as a
correctly indexed one.

⚠️ **Attribution requirement, and it is the whole reason arm B exists.** Real data is not exchangeable, so
an absolute coverage miss cannot be attributed to the convention on its own. Only the **paired delta**
between arms carries the claim. If the two arms differ in their residual set, their centre, or their base
model, the delta is meaningless — see §8.

## 6. Data

| Domain | Source | Rule |
|---|---|---|
| Forecasting | Monash Time Series Forecasting Archive; M4 | Selection rule stated before running; minimum series length fixed by the largest calibration window |
| Tabular | OpenML-CC18, or a documented subset | Same |
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
| Per-helper survey of thirteen packages, versions pinned, read at source | ✅ Complete; two counts pending re-audit against a stated criterion |
| End-to-end synthetic runs for four libraries | ✅ Complete |
| **M1 rank map over the nine definitions** | 🔲 Not started |
| **M2 delivered `n_min` table** | 🔲 Arithmetic exists, table not assembled |
| **M3 paired real-data coverage** | ⚠️ **Harness written and running; v1's attribution is INVALID and its numbers are not reportable.** The two arms did not share a residual set or a centre, which showed up as arm A landing on ranks above `n` and as a negative paired delta between intervals that are not nested. Fix identified: build arm B from the library's own residual matrix and point forecast, as an existing probe in this repository already does. **Not committed until that holds** |
| Tightening the one null result to ≥2000 fits across several calibration sizes | 🔲 Not started |
| Conformance suite and the §7 checklist as installable tooling | 🔲 Not started |

## 9. Verification rules applied to everything here

These are not aspirations; they have each already caught a real error in this work.

- **Every closed form self-checks against exact rational arithmetic at import.** A failing check aborts
  the run. Two such checks caught errors in their author's own hand-derived assertions.
- **A grid is chosen by the bug, not by the author.** Every sweep extends to at least twice the first
  boundary it finds. A grid that stops just past a boundary is worse than one that stops far short,
  because it looks converged.
- **Sweep at least one non-unit-fraction level.** Claims of the form "this only happens on that residue
  class" are usually artifacts of sweeping only `1/10`, `1/20`, `1/100`.
- **An oracle must be independent of the implementation it checks.** A fixture placed at a cell that
  cannot discriminate is worse than no fixture.
- **Label one-sided and two-sided on every number, in the table, not only in the script.** `n/(n+1)` and
  `(n−1)/(n+1)` differ by fourteen points at `n = 6`.
- **Establish the direction of harm before calling anything a defect**, and check that the minimal patch
  you have in mind does not itself raise.
- **Verify by running.** Reading has produced confident false claims in this work more than once.

## 10. Known limitations

- Synthetic harnesses use iid draws with a deliberately simple base model. They establish that a miss
  occurs where the guarantee should hold exactly; they say nothing about dependent real-world series.
  That is what M3 is for, and M3 is not finished.
- Non-exchangeability generally biases toward **over**coverage, so a synthetic result understates rather
  than overstates.
- Findings are version-pinned. The rank map (M1) is not, which is why it is the theory core.
- One finding previously asserted in this work was **retracted** after three independent checks
  contradicted it. The probe that adjudicated it is in this repository.

## 11. Out of scope

- Closed-source or vendor implementations: not inspectable, and not claimed about.
- Packages that form no order statistic of a calibration set — learned-quantile models, and probability
  intervals from isotonic calibration. Excluded with the reason stated rather than silently omitted.
- Sample-path simulators that produce trajectories rather than a split-conformal interval.
- Which method is *best*. This is a correctness audit, not a benchmark of predictive performance.

## 12. What is in this repository

Five probes and three committed outputs. Two outputs are intentionally absent: one prints a mechanism that
has not yet been reported to the maintainers of the package concerned, and both regenerate exactly by
running the committed probes under the pinned environment.
