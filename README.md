# Conformal quantile convention probes

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

| Probe | What it measures |
|---|---|
| `probes/branch_d_check.py` | The convention in isolation: an uncorrected level versus the required order statistic on identical draws. 200k draws per cell, `fractions.Fraction` oracle |
| `probes/convention_probe.py` | Structural branch identification, coverage sweeps with paired standard errors, threshold extraction |
| `probes/run_sktime_river.py` | sktime `ConformalIntervals` via `predict_interval`, river `RegressionJackknife` via `predict_one`, with an oracle independent of both |
| `probes/run_darts.py` | Exact rank arithmetic, paired Monte Carlo on identical draws, and an end-to-end run through a real `ConformalNaiveModel` |
| `probes/verify_statsforecast_rebuttal.py` | `ConformalSeasonalPool._oriented_index`: window mapping, and the adjudication of a finding of this author's own that turned out to be wrong |

**The research plan is in [`PLAN.md`](PLAN.md)** — the question, the method, the protocol, phase status,
and what has *not* been established.

## Running them

```bash
python3 -m venv .venv-probe
.venv-probe/bin/pip install -r probe-requirements.txt

.venv-probe/bin/python probes/convention_probe.py
.venv-probe/bin/python probes/branch_d_check.py
.venv-probe/bin/python probes/run_sktime_river.py
.venv-probe/bin/python probes/run_statsforecast.py
.venv-probe/bin/python probes/verify_statsforecast_rebuttal.py

# darts needs its own environment (numpy 2.4.6, scikit-learn 1.9.0)
python3 -m venv .venv-darts && .venv-darts/bin/pip install "darts==0.46.1"
.venv-darts/bin/python probes/run_darts.py
```

Environment: Python 3.13 · sktime 1.1.0 · river 0.25.0 · statsforecast 2.1.1 · numpy 2.4.6 · pandas 2.3.3
· scikit-learn 1.7.2. Each script writes into `outputs/`.

Two outputs — `probe_output_sf_rebuttal.txt` and `probe_output_v5.txt` — are not committed. Run
`probes/verify_statsforecast_rebuttal.py` and `probes/convention_probe.py` to generate them; both are
deterministic under the pinned environment. One of them prints a mechanism that has not yet been reported
to the maintainers of the package concerned, which is the only reason it is absent.

## One convention worth knowing before reading the code

**Every script self-checks its closed forms against exact rational arithmetic at import.** A failing
self-check aborts the run. This is not decoration: two of these self-checks caught errors in their own
author's hand-derived assertions, including one that invalidated a claim already written down.

## Not included

Third-party library sources are **not** redistributed here. `probe-requirements.txt` pins the exact
versions instead.

## Citing

See `CITATION.cff`. Deposit metadata and the reproduction manifest are in `ARTIFACT.md`.

## License

MIT — see `LICENSE`.
