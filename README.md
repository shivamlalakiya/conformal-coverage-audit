# Conformal quantile convention probes

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21799203-1682D4)](https://doi.org/10.5281/zenodo.21799203)

Archived on Zenodo. **Cite `10.5281/zenodo.21799203`** — the all-versions DOI, which always resolves to
the latest release. To pin an exact reproduction, cite the version DOI instead
(`10.5281/zenodo.21811491` is v1.1.0, the current release; `10.5281/zenodo.21799204` is
v1.0.0, which predates every real-data arm).

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

Every real-data probe runs two arms. **Arm A** is the library through its public API at its own defaults.
**Arm B** is the required order statistic computed from **the library's own scores and the library's own
point prediction**, changing only the level→rank step. Real data is not exchangeable, so an absolute
coverage miss is not attributable to the convention — only the **paired delta** carries a claim. See
[`PLAN.md`](PLAN.md) §5.

| Probe | What it measures |
|---|---|
| `probes/run_real_data.py` | sktime `ConformalIntervals` on Monash series, both arms off the same fitted object |
| `probes/run_real_data_statsforecast.py` | statsforecast `ConformalIntervals`, with the scores captured out of the library's own interval call |
| `probes/run_real_data_darts.py` | darts `ConformalNaiveModel`, calibration lengths chosen to sample both the coincidence band and the deficit band |
| `probes/run_real_data_tabular.py` | Six tabular implementations on OpenML data, resolving the same bound from the same scores on the same split |
| `probes/export_series.py` | Caches Monash series as `.npz`, because the darts probe cannot share an environment with the loader |

### Tooling and generalisation

| Probe | What it measures |
|---|---|
| `probes/conformance_suite.py` | Given any `(scores, level) → threshold` callable: the branch, the rank it lands on, the delivered coverage, the smallest `n` at which the requested level is delivered, and whether it warns at the boundary. Validated at import against reference implementations of every branch |
| `probes/helper_census.py` | Counts the level→rank resolution sites across the audited packages under a stated criterion, verifying each site's file, line and anchor text on disk. Fails loudly if an anchor has moved |
| `probes/w8_falsification.py` | Whether the level→rank map matters outside conformal prediction: empirical value-at-risk, nonparametric tolerance bounds, and bootstrap percentile intervals. Includes a setting chosen because it was likely to refute the general claim |
| `probes/paired_report.py` | Shared summary arithmetic for the paired arms, including the two reporting subtleties an earlier version of this work got wrong |

**The research plan is in [`PLAN.md`](PLAN.md)** — the question, the method, the protocol, phase status,
and what has *not* been established.

## Running them

Three environments, because the audited libraries do not agree on numpy and pandas versions.
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
