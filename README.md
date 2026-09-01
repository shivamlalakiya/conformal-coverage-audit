# Conformal coverage audit

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21799203-1682D4)](https://doi.org/10.5281/zenodo.21799203)
[![CI](https://github.com/shivamlalakiya/conformal-coverage-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/shivamlalakiya/conformal-coverage-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

Measurement harnesses for auditing how Python conformal-prediction implementations
resolve the conformal quantile at finite sample sizes, plus a tiny stdlib package
that does the rank arithmetic without asking a quantile function.

**Cite the archive:** [`10.5281/zenodo.21799203`](https://doi.org/10.5281/zenodo.21799203)
(all-versions DOI; always resolves to the latest release).

| | |
|---|---|
| **Package** | [`conformal_coverage`](README-package.md) — four functions, zero dependencies |
| **Probes** | `probes/` — one harness per question; committed transcripts in `outputs/` |
| **Ten-minute check** | `python -m conformal_coverage` and `python verify_headline.py` |
| **Disclosure record** | [`DISCLOSURE.md`](DISCLOSURE.md) — upstream reports with filing dates |
| **Research plan** | [`PLAN.md`](PLAN.md) — question, method, protocol, what is *not* established |
| **License** | MIT |

---

## About

Distribution-free intervals at finite sample size are indexed by **order statistics**.
The APIs libraries use to obtain one accept a **level**, and the map from level to
rank depends on an interpolation convention. These probes measure which rank each
implementation actually lands on, and what coverage that rank delivers.

This repository is the public measurement artifact: pinned library versions, scripts
that re-derive their own formulas with `fractions.Fraction` at import, and plain-text
outputs committed beside each script so a third party can re-run or re-parse without
guessing.

It is **not** a conformal-prediction library. For the arithmetic alone, install
`conformal-coverage` (or vendor `conformal_coverage/__init__.py`).

---

## Quick start

```bash
# package only — stdlib, no third-party deps
pip install -e .
python -m conformal_coverage

# headline quantities from first principles (needs numpy)
pip install numpy
python verify_headline.py
# or: make verify
```

```python
from conformal_coverage import (
    required_rank, delivered_coverage, feasibility_floor, conformal_threshold,
)

required_rank(100, 0.9)           # 91   the order statistic the guarantee needs
delivered_coverage(90, 100)       # Fraction(90, 101)   what rank 90 actually delivers
feasibility_floor(0.9)            # 9    below this no valid finite bound exists
required_rank(8, 0.9)             # None
conformal_threshold(scores, 0.1)  # the threshold, or +inf where none is valid
```

`conformal_threshold` indexes the sorted scores directly, so no interpolation
convention can move it, and it returns `+inf` rather than a number where no valid
bound exists — which is the honest answer at that size. Levels are converted with
`Fraction(str(x))`, not `Fraction(x)`: the latter is
`8106479329266893/9007199254740992` for `0.9`, strictly greater than `9/10`, and
gives the wrong rank at `n = 9`.

See [`README-package.md`](README-package.md) for the rest of the API surface.

---

## Layout

```
conformal_coverage/   installable package (rank arithmetic)
probes/               measurement harnesses
outputs/              committed output of each harness, one file per script
repro/                reproduction helpers
verify_headline.py    ten-minute referee path (numpy only)
DISCLOSURE.md         filed upstream reports
PLAN.md               research plan and phase status
PREREGISTRATION.md    pre-registration note
CITATION.cff          citation metadata
.zenodo.json          Zenodo deposit metadata
```

---

## Probes

### Synthetic and structural

| Probe | What it measures |
|---|---|
| `probes/branch_d_check.py` | The convention in isolation: an uncorrected level versus the required order statistic on identical draws. 200k draws per cell, `fractions.Fraction` oracle |
| `probes/convention_probe.py` | Structural branch identification, coverage sweeps with paired standard errors, threshold extraction |
| `probes/rank_map.py` | The nine Hyndman–Fan quantile definitions plus four aliases, read for what guarantee each can carry rather than for estimation accuracy: which express `⌈(n+1)(1−α)⌉` at all, and what each delivers instead |
| `probes/rule_class.py` | Affine level→index rules reduced by residue arithmetic: numpy's thirteen method names scored for validity at every size and for worst-case over-coverage, plus how many calibration sizes separate the classes. Exact rationals; executed against numpy |
| `probes/run_sktime_river.py` | sktime `ConformalIntervals` via `predict_interval`, river `RegressionJackknife` via `predict_one`, with an oracle independent of both |
| `probes/run_darts.py` | Exact rank arithmetic, paired Monte Carlo on identical draws, and an end-to-end run through a real `ConformalNaiveModel` |
| `probes/run_darts_tighten.py` | The same construction at 2000 fits per cell across four calibration lengths, with the exact coverage the convention predicts beside each measurement |
| `probes/darts_scoring_path.py` | What `ConformalNaiveModel` actually does, instrumented: the captured score set, the rank, the returned threshold and the returned interval, each asserted per fit. Separates the level→rank map from in-sample residual bias |
| `probes/verify_statsforecast_rebuttal.py` | `ConformalSeasonalPool._oriented_index`: window mapping, and the adjudication of a finding of this author's own that turned out to be wrong |

### Real data — paired arms

Each real-data probe reports a matched pair. **Arm A** calls the package as shipped, defaults untouched.
**Arm B** takes the order statistic the guarantee requires, built from **that same package's scores and
that same package's point prediction** — the only step that differs is how the level becomes an index.
Archive series are not exchangeable, so no absolute coverage number can be charged to the convention;
only the **paired delta** supports a claim. See [`PLAN.md`](PLAN.md) §5.

| Probe | What it measures |
|---|---|
| `probes/run_real_data.py` | sktime `ConformalIntervals` on Monash series, both arms off the same fitted object |
| `probes/run_real_data_statsforecast.py` | statsforecast `ConformalIntervals`, with the scores captured out of the library's own interval call |
| `probes/run_real_data_darts.py` | darts `ConformalNaiveModel`, calibration lengths chosen to sample both the coincidence band and the deficit band |
| `probes/run_real_data_tabular.py` | Seven tabular implementations over OpenML data, each handed identical scores and an identical split |
| `probes/export_series.py` | Caches Monash series as `.npz`, because the darts probe cannot share an environment with the loader |
| `probes/sample_robustness.py` | Whether the arms above depend on which series were chosen or on one test point per series. Re-runs the same cells on **every eligible series in the archive** — settling the question of *which* series, though neither the archive nor the eligibility cutoff — with a rolling origin, and clusters the standard error by series because rolling origins of one series share their history. Adds no forecasting logic: origin `j` is the truncation `s[:len(s)-j]` through each arm's own unmodified code path |

### Tooling and generalisation

| Probe | What it measures |
|---|---|
| `probes/attainable_grid.py` | What an order-statistic interval can express. The coverage grid by exhaustive enumeration over the `n+1` equally likely ranks; the minimality of the required **span**; and the smallest usable calibration size once `alpha` is shared out among several resolved levels, which turns out to be governed by the smallest share rather than by how many shares there are. Exact rational arithmetic; the Monte Carlo at the end is a control on the enumeration, not evidence for it |
| `probes/beta_optimize_floor.py` | Whether mapie's width-minimising `beta` search selects a rail level no order statistic of the calibration scores can carry. The grid runs down to `alpha/(n+1)`, where a finite lower rail would need `alpha >= 1`, so unlike the one-rail floor this does not close as `n` grows. Driven three ways: the arithmetic, the composed site the census anchors, and the public `predict_interval(minimize_interval_width=True)` |
| `probes/conformance_suite.py` | Given any `(scores, level) → threshold` callable: the branch, the rank it lands on, the delivered coverage, the least calibration size that honours the level as asked, and whether a boundary case raises a warning. Self-checks at import by classifying a reference implementation of each branch and aborting if one comes back mislabelled |
| `probes/helper_census.py` | Counts the level→rank resolution sites across the audited packages under a stated criterion, verifying each site's file, line and anchor text on disk. Fails loudly once an anchor shifts |
| `probes/oos_frame.py` | The population behind the out-of-census rows. Files one verdict per package on the unaudited side of the download denominator, pinning each to a path and line number inside a published source distribution, and re-opens that location where this tree keeps a copy. Self-checks that its population is the one `install_weight.py` printed and that the packages it files as carrying a site are the packages the three out-of-census runs name. Prints no site count |
| `probes/frame_sweep.py` | Whether the audited list could have been reached by querying the index rather than by naming distributions. Two sweeps built to fail in opposite directions: every distribution name on PyPI matched against a stem pattern and then the metadata behind each hit, and separately the most-downloaded distributions above a monthly floor read the same way. Both are scored against the audited list, which is known-positive, so each sweep's recall is measured instead of assumed. Bounds the gap in the other direction too: conformal-first distributions the list never held, and their monthly downloads. Reaches the network; needs nothing installed except the one distribution it **executes** rather than describes |
| `probes/oos_prevalence.py` | What a machine tally of the resolution sites would cost, calibrated before it is spent. The manifest carries every audited site at a file and a line, so the enumerator is pointed at the same packages and its recall is a number. The sites it loses are attributed to named mechanisms, each anchored at the site that shows it, rather than to a residual. Prints no rate: one carrying that loss would sit beside a hand-made count and invite the comparison the loss invalidates |
| `probes/second_coder.py` | Independent re-classification made possible rather than claimed. One person authored the criterion, the classifier and every verdict, so this ships the half somebody else has to supply. `--build` picks sites under a fixed seed, emits the surrounding source for each, and lists the permitted branch names; `--score` reads a completed file back and reports raw agreement together with Cohen's kappa. Redaction is verified, not assumed: the emitted file is parsed again after writing and the run aborts on any surviving verdict or key term. A synthetic completed file drives the scorer end to end, so it is known good before real input reaches it. Concealment here defeats accident only — every label is public in the manifest |
| `probes/w8_falsification.py` | Whether the level→rank map matters outside conformal prediction: empirical value-at-risk, nonparametric tolerance bounds, and bootstrap percentile intervals. Includes a setting chosen because it was likely to refute the general claim |
| `probes/mapie_clip_reachability.py` | Whether mapie's clip on the corrected level is reachable through the public API, and what the path delivers. Walks every size at which the requested rank cannot exist, under exact rational arithmetic, keeping the first *feasible* size as a control that has to score zero, confirms it end to end at 50k draws per cell, counts the signatures carrying `allow_infinite_bounds` with `ast`, and drives every public regressor class that exposes it. Filed as MAPIE#980; **withdraws a retraction of this audit's own** |
| `probes/lac_crossval_dead_value.py` | Whether mapie's LAC `quantiles_` decides anything on the cross-validation path. Scales it by 1000 and checks whether a single returned set moves. APS computes the identical expression on the identical path and *is* read, so it is the control that makes the negative result mean something |
| `probes/paired_report.py` | Shared summary arithmetic for the paired arms, including the two reporting subtleties an earlier version of this work got wrong |

**The research plan is in [`PLAN.md`](PLAN.md)** — the question, the method, the protocol, phase status,
and what has *not* been established.

---

## Running the probes

Three environments are needed: the packages under audit pin incompatible numpy and pandas releases.
`probe-requirements.txt` documents all three and names which probe needs which.

```bash
# [1] forecasting: sktime, statsforecast, river
python3 -m venv .venv-probe
.venv-probe/bin/pip install -r probe-requirements.txt
.venv-probe/bin/python probes/convention_probe.py
.venv-probe/bin/python probes/branch_d_check.py
.venv-probe/bin/python probes/rank_map.py
.venv-probe/bin/python probes/rule_class.py
.venv-probe/bin/python probes/run_sktime_river.py
.venv-probe/bin/python probes/verify_statsforecast_rebuttal.py
.venv-probe/bin/python probes/run_real_data.py m1_monthly_dataset 250
.venv-probe/bin/python probes/run_real_data_statsforecast.py m1_monthly_dataset 250
.venv-probe/bin/python probes/w8_falsification.py
# what an interval can express, and the floor for any division of alpha.
# Exact arithmetic only, so it needs no library and runs in seconds.
.venv-probe/bin/python probes/attainable_grid.py
# selection/resolution robustness: every eligible series, rolling origin.
# ~30 min; parallel across series, so it wants a few free cores AND free memory --
# each worker holds a fitted forecaster and its residual matrix. Run it alone.
# PROBE_WORKERS caps the pool; overlapping it with another probe exhausted swap
# here and the OS killed both runs mid-cell without writing an output.
.venv-probe/bin/python probes/sample_robustness.py
# PROBE_WORKERS=4 .venv-probe/bin/python probes/sample_robustness.py   # low-memory
.venv-probe/bin/python probes/conformance_suite.py \
    --out outputs/probe_output_conformance_forecasting.txt

# [2] darts pins pandas 3.x, which sktime 1.1.0 rejects
python3 -m venv .venv-darts
.venv-darts/bin/pip install "darts==0.46.1" "numpy==2.4.6" "scikit-learn==1.9.0"
.venv-darts/bin/python probes/run_darts.py
.venv-darts/bin/python probes/run_darts_tighten.py
.venv-darts/bin/python probes/darts_scoring_path.py
# the darts real-data arm reads a cache the loader in [1] writes:
.venv-probe/bin/python probes/export_series.py m1_monthly_dataset 250 /tmp/m1.npz 70
.venv-darts/bin/python probes/run_real_data_darts.py /tmp/m1.npz
# and the same arm on every eligible series, for the robustness check:
.venv-probe/bin/python probes/sample_robustness.py --export-npz /tmp/m3_full.npz
.venv-darts/bin/python probes/run_real_data_darts.py /tmp/m3_full.npz

# [3] tabular: mapie, crepes, puncc, torchcp, nonconformist, openml
python3 -m venv .venv-tabular
.venv-tabular/bin/pip install "mapie==1.4.1" "openml==0.15.1" \
    "scikit-learn==1.7.2" "crepes==0.9.1" "puncc==0.9.3" \
    "nonconformist==2.1.0" torch torchcp
.venv-tabular/bin/python probes/run_real_data_tabular.py 15
# whether the width-minimising beta search selects a level no rank can carry
.venv-tabular/bin/python probes/beta_optimize_floor.py
.venv-tabular/bin/python probes/conformance_suite.py \
    --out outputs/probe_output_conformance_tabular.txt
# both mapie-only; the second takes a few minutes at 50k draws a cell
.venv-tabular/bin/python probes/lac_crossval_dead_value.py
.venv-tabular/bin/python probes/mapie_clip_reachability.py
```

Each script writes into `outputs/`. Every real-data forecasting probe takes a dataset name and a series
cap as arguments and writes a **dataset-suffixed** output file, so a second dataset cannot overwrite the
first — `m1_monthly_dataset` keeps the unsuffixed name.

`probes/helper_census.py` reads source text only, so it runs under any of the three environments. Point it
at a directory of unpacked package sources with `--root`, or run it with the packages installed; either
way it **reports** which packages it could not locate rather than quietly counting fewer sites.

---

## Two conventions worth knowing before reading the code

**Each script re-derives its own formulas with `fractions.Fraction` the moment it loads.** A failing
self-check aborts the run. This is not decoration: several of these self-checks caught errors in their own
author's hand-derived assertions, including ones that invalidated claims already written down.

**A returned threshold equal to `max(scores)` is not evidence of a clamped level.** Where the required
rank *is* `n`, returning the sample maximum is right. Separating those two cases needs the level→rank rule
identified over several `n`, not a single probe — `conformance_suite.py` does it by fitting the rule and
probing the boundary, and it is written that way because the single-probe version mislabelled a library.

---

## Continuous integration

GitHub Actions runs on every push and pull request to `main`:

1. `pip install -e .` and run `python -m conformal_coverage` (self-check) on Python 3.9–3.13
2. install numpy and run `python verify_headline.py` (re-derives the abstract quantities against committed outputs)

Full probe reproduction is **not** in CI: it needs three pinned environments and, for some arms, archive downloads measured in tens of minutes. Use the commands above for that path.

---

## Not included

- Third-party library sources are **not** redistributed here. `probe-requirements.txt` pins the exact
  versions instead.
- The `.npz` series cache `export_series.py` writes. The two commands that regenerate it are above.

---

## Zenodo version DOIs

Archived on Zenodo. **Cite `10.5281/zenodo.21799203`** — the all-versions DOI, which always resolves to
the latest release.

To pin an exact reproduction, cite a version DOI. Resolve it by **which tree it
archives**, not by the version string Zenodo shows: `.zenodo.json`'s version field
was bumped after tagging rather than in the tagged commit, so two records carry a
label one release behind their contents. The labels are permanent per record; this
table is the mapping, each row checked against the Zenodo API.

| Version DOI | Zenodo label | Archives |
|---|---|---|
| `10.5281/zenodo.21799204` | v1.0.0 | tree `v1.0.0`, predating every real-data arm |
| `10.5281/zenodo.21811491` | v1.1.0 | tree `v1.1.0` |
| `10.5281/zenodo.21814982` | v1.1.0 | tree **`v1.2.0`** — whole-archive robustness arm, exact feasibility floor, the `conformal_coverage` package |
| `10.5281/zenodo.21816837` | v1.2.0 | tree **`v1.3.0`** — probes print the ratios the write-up quotes |
| `10.5281/zenodo.21816871` | v1.3.1 | tree `v1.3.1` — first record whose label matches its contents |

`v1.3.1` and later are labelled correctly; `paperlib/check_release_version.py` in the
write-up repository fails a release whose `.zenodo.json` and `CITATION.cff` disagree
with the tag.

---

## Citing

```bibtex
@software{lalakiya_conformal_coverage_audit,
  author       = {Lalakiya, Shivam},
  title        = {Conformal quantile convention probes},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21799203},
  url          = {https://doi.org/10.5281/zenodo.21799203}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff). GitHub's "Cite this repository" widget reads the same file.

---

## License

MIT — see [`LICENSE`](LICENSE).
