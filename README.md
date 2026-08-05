# Conformal quantile convention probes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21799203.svg)](https://doi.org/10.5281/zenodo.21799203)

Archived on Zenodo. **Cite `10.5281/zenodo.21799203`** — the all-versions DOI, which always resolves to
the latest release. To pin an exact reproduction, cite the version DOI instead
(`10.5281/zenodo.21799204` is v1.0.0).

Measurement harnesses for auditing how Python conformal-prediction implementations resolve the conformal
quantile at finite sample sizes.

A finite-sample distribution-free interval is a statement about an **order statistic**. The APIs libraries
use to obtain one accept a **level**, and the map from level to rank depends on an interpolation
convention. These probes measure which rank each implementation actually lands on, and what coverage that
rank delivers.

## Layout

```
probes/       the harnesses
outputs/      committed output of each harness, one file per script
```

### Synthetic and structural

| Probe | What it measures |
|---|---|
| `probes/branch_d_check.py` | The convention in isolation: an uncorrected level versus the required order statistic on identical draws. 200k draws per cell, `fractions.Fraction` oracle |
| `probes/convention_probe.py` | Structural branch identification, coverage sweeps with paired standard errors, threshold extraction |
| `probes/rank_map.py` | The nine Hyndman–Fan quantile definitions plus four aliases, read not as estimators of a population quantile but as carriers of a coverage guarantee: which can express `⌈(n+1)(1−α)⌉` at all, and what each delivers instead |
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
| `probes/sample_robustness.py` | Whether the arms above depend on which series were chosen or on one test point per series. Re-runs the same cells on **every eligible series in the archive** — no sample, so there is no selection to object to — with a rolling origin, and clusters the standard error by series because rolling origins of one series share their history. Adds no forecasting logic: origin `j` is the truncation `s[:len(s)-j]` through each arm's own unmodified code path |

### Tooling and generalisation

| Probe | What it measures |
|---|---|
| `probes/conformance_suite.py` | Given any `(scores, level) → threshold` callable: the branch, the rank it lands on, the delivered coverage, the least calibration size that honours the level as asked, and whether a boundary case raises a warning. Validated at import against reference implementations of every branch |
| `probes/helper_census.py` | Counts the level→rank resolution sites across the audited packages under a stated criterion, verifying each site's file, line and anchor text on disk. Fails loudly if an anchor has moved |
| `probes/w8_falsification.py` | Whether the level→rank map matters outside conformal prediction: empirical value-at-risk, nonparametric tolerance bounds, and bootstrap percentile intervals. Includes a setting chosen because it was likely to refute the general claim |
| `probes/paired_report.py` | Shared summary arithmetic for the paired arms, including the two reporting subtleties an earlier version of this work got wrong |

**The research plan is in [`PLAN.md`](PLAN.md)** — the question, the method, the protocol, phase status,
and what has *not* been established.

## Running them

Three environments are needed: the packages under audit pin incompatible numpy and pandas releases.
`probe-requirements.txt` documents all three and names which probe needs which.

```bash
# [1] forecasting: sktime, statsforecast, river
python3 -m venv .venv-probe
.venv-probe/bin/pip install -r probe-requirements.txt
.venv-probe/bin/python probes/convention_probe.py
.venv-probe/bin/python probes/branch_d_check.py
.venv-probe/bin/python probes/rank_map.py
.venv-probe/bin/python probes/run_sktime_river.py
.venv-probe/bin/python probes/verify_statsforecast_rebuttal.py
.venv-probe/bin/python probes/run_real_data.py m1_monthly_dataset 250
.venv-probe/bin/python probes/run_real_data_statsforecast.py m1_monthly_dataset 250
.venv-probe/bin/python probes/w8_falsification.py
# selection/resolution robustness: every eligible series, rolling origin.
# ~30 min; parallel across series, so it wants a few free cores.
.venv-probe/bin/python probes/sample_robustness.py
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
.venv-tabular/bin/python probes/conformance_suite.py \
    --out outputs/probe_output_conformance_tabular.txt
```

Each script writes into `outputs/`. Every real-data forecasting probe takes a dataset name and a series
cap as arguments and writes a **dataset-suffixed** output file, so a second dataset cannot overwrite the
first — `m1_monthly_dataset` keeps the unsuffixed name.

`probes/helper_census.py` reads source text only, so it runs under any of the three environments. Point it
at a directory of unpacked package sources with `--root`, or run it with the packages installed; either
way it **reports** which packages it could not locate rather than quietly counting fewer sites.

## Two conventions worth knowing before reading the code

**Every script self-checks its closed forms against exact rational arithmetic at import.** A failing
self-check aborts the run. This is not decoration: several of these self-checks caught errors in their own
author's hand-derived assertions, including ones that invalidated claims already written down.

**A returned threshold equal to `max(scores)` is not evidence of a clamped level.** Where the required
rank *is* `n`, the maximum is the correct answer. Separating those two cases needs the level→rank rule
identified over several `n`, not a single probe — `conformance_suite.py` does it by fitting the rule and
probing the boundary, and it is written that way because the single-probe version mislabelled a library.

## Not included

- Third-party library sources are **not** redistributed here. `probe-requirements.txt` pins the exact
  versions instead.
- The `.npz` series cache `export_series.py` writes. The two commands that regenerate it are above.

## Citing

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
