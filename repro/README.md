# One script per upstream filing

Every script in this directory targets a single report listed in `../DISCLOSURE.md`,
imports nothing but the library it is about, and finishes in seconds. None of them
touches `../probes/`, so a maintainer can copy one file out and run it.

## The contract

- **Exit 0 means the finding reproduces.** Exit 1 with `does not reproduce` on stdout
  means it no longer does. A red row is the point of filing, so the polarity is that
  way round on purpose.
- **Each script states its own claim** in the module docstring, then prints the numbers
  it based that claim on, then a verdict. Nothing is asserted that the printed table
  does not show.
- **Each script says what would falsify it.** A reproducer that cannot come out
  negative is a demonstration, not a check.
- **The grid is asserted, not assumed.** Where a finding depends on straddling a
  feasibility boundary, the script fails if its own sizes do not straddle it, rather
  than reporting a contrast it could not have seen.

## Environments

The audited libraries pin incompatible `numpy` versions and cannot share one virtual
environment; `../probe-requirements.txt` records which needs which. `run_all.py`
matches each script to the interpreter running it by the library in the filename and
reports anything unimportable as **NOT RUN** rather than as passing.

```
python repro/run_all.py
```

## Coverage, stated rather than implied

This directory does not yet cover every filing. What is here:

| Filing | Script |
|---|---|
| `mapie` — `allow_infinite_bounds=True` skips the calibration-size guard | `mapie_980_allow_infinite_bounds.py` |
| `crepes` — the classifier is silent below the floor where the regressor warns | `crepes_47_small_calibration.py` |
| `sktime` — `conformal` and `conformal_bonferroni` omit the `(m+1)` correction | `sktime_10758_mplus1.py` |
| `torchcp` — the docstring names a threshold the code no longer computes | `torchcp_122_docstring.py` |

Four of the fifteen filings. The remaining eleven are reproduced by the probes in
`../probes/` and by the conformance suites, and do **not** yet have a standalone script
here. They are listed as absent rather than left to be counted: a directory that looks
complete and is not is worse than one that says where it stops.

## One error worth recording

The first version of `crepes_47_small_calibration.py` reported zero warnings from a
path that warns on every call. The helper returned `len(caught), "", fn()` — Python
evaluates the warning count *before* the call that produces the warnings. The
reproducer would have concluded that the regressor is as silent as the classifier, and
the contrast it exists to show would have disappeared in the direction that makes the
library look worse.
