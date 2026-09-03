# Environments

`probe-requirements.txt` names four environments and says which probe needs which.
This directory records three of them as `pip freeze --all`: two taken from the
interpreters that produced the committed outputs, and one reconstructed from the pins
and validated by reproducing them.

| | environment | recorded here | why |
|---|---|---|---|
| [1] | forecasting: sktime, statsforecast, river | `forecasting.lock.txt` | the interpreter most outputs came from |
| [2] | darts | `darts.lock.txt` | pins pandas 3.x, which [1] rejects |
| [3] | tabular: mapie, crepes, puncc, openml, torchcp, nonconformist | `tabular.lock.txt` | reconstructed from the pins on 2 Sept 2026 and validated by reproduction, below; no freeze was taken from the interpreter that ran the outputs |
| [4] | out-of-census: crepes-weighted, skforecast, conformal-tights | not recorded | no freeze was taken, none has been reconstructed, and each of the three ran in an environment of its own |

The two files that are here are records rather than curated lists. They say what
ran; `probe-requirements.txt` says what was asked for. Where the two disagree the
freeze is the fact.

**No container image is shipped.** One was considered and not built, because it
could not be built and run here, and an unexecuted `Dockerfile` in an artifact is a
claim about reproducibility rather than a demonstration of it. The two lockfiles and
`probe-requirements.txt` are what this deposit can stand behind.

## Environment [3], reconstructed and checked, 2 September 2026

`tabular.lock.txt` is not a record of the interpreter that ran the tabular outputs;
none was kept. It is `pip freeze --all` of an environment built from the [3] block of
`probe-requirements.txt` on CPython 3.12.13 (numpy 1.26.4, pandas 3.0.5, mapie 1.4.1,
scikit-learn 1.7.2, torch 2.14.0) and then made to earn the name by re-running the
five tabular probes against the committed outputs, with a clean bytecode cache:

| output | under `tabular.lock.txt` |
|---|---|
| `probe_output_beta_optimize_floor.txt` | byte-identical |
| `probe_output_lac_crossval_dead_value.txt` | byte-identical; its header names numpy 1.26.4 |
| `probe_output_real_data_tabular.txt` | every committed line reproduced; the probe has since gained one `predicted delta` line per cell and reworded one bracketed annotation, so the fresh output is 55 lines longer |
| `probe_output_conformance_tabular.txt` | every committed line reproduced except the column padding of one row label, which the probe's formatting changed after the output was committed |
| `probe_output_mapie_clip_reachability.txt` | every line a macro reads reproduced; two explanatory sentences were reworded in the probe after the output was committed, and two interval widths in the class-reachability listing differ between runs and are read by nothing |

So three probe sources changed after their outputs were last committed. Re-running them
under this environment and committing the fresh outputs is the pending repair; until
then the committed files are numerically current and textually behind their probes.

Two further observations, recorded so nobody chases them twice. The machine also
holds a Python 3.14.4 environment with numpy 2.5.2 and puncc 0.9.3 installed side by
side, against the pin comment that says puncc forces numpy 1.26.x; under it
`run_real_data_tabular.py` and `conformance_suite.py` reproduce every committed numeric
line, while `beta_optimize_floor.py` and `mapie_clip_reachability.py` stop with
`TypeError: only 0-dimensional arrays can be converted to Python scalars`, numpy 2.x
refusing `float()` on a one-element array. The manuscript's tabular numbers are
therefore insensitive to the numpy major version wherever the probe runs at all, and
numpy 1.26.4 is the version under which all five run. And running probes under two
interpreters at once, sharing `probes/__pycache__`, produced two assertion failures
(`paired_report.self_check` and the span rule in `run_real_data.py`) that vanished with
`PYTHONPYCACHEPREFIX` pointed at an empty directory; they were the cache, not the
arithmetic.

## The short path, which needs none of this

```bash
python3 -m venv .venv && .venv/bin/pip install numpy
make verify
```

`verify_headline.py` re-derives the quantities in the manuscript's abstract from the
committed probe outputs and prints each beside the output line it was read from. It
needs numpy and nothing else. The full reproduction needs the environments above.
