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

## Coverage

Twelve scripts for twelve reproducible filings. Three of the fifteen rows in
`../DISCLOSURE.md` are not reproducible defects and are listed below with the reason,
rather than padded with a script that would assert nothing.

| Filing | Script | Direction of harm |
|---|---|---|
| `mapie` #973 — classification quantile lands a rank high | `mapie_973_classification_quantile.py` | conservative |
| `mapie` #974 — corrected level clipped to 1.0 | `mapie_974_clamped_level.py` | anti-conservative |
| `mapie` #978 — asymmetric score cancels the size guard | `mapie_978_asymmetric_guard.py` | anti-conservative |
| `mapie` #980 — `allow_infinite_bounds` skips the guard | `mapie_980_allow_infinite_bounds.py` | anti-conservative |
| `crepes` #46 — docstring inverts the membership condition | `crepes_46_docstring_inverted.py` | documentation |
| `crepes` #47 — classifier silent below the floor | `crepes_47_small_calibration.py` | silent |
| `crepes` #48 — membership on a float comparison | `crepes_48_float_threshold.py` | anti-conservative |
| `sktime` #10757 — `empirical_residual` takes the wrong tail | `sktime_10757_wrong_tail.py` | anti-conservative |
| `sktime` #10758 — omitted `(m+1)` correction | `sktime_10758_mplus1.py` | anti-conservative |
| `sktime` #10766 — residual diagonal off by one horizon | `sktime_10766_residual_alignment.py` | exchangeability |
| `statsforecast` #1202 — sufficiency rule names one rail | `statsforecast_1202_sufficiency_rule.py` | documentation |
| `torchcp` #122 — docstring names a threshold not computed | `torchcp_122_docstring.py` | documentation |

### The three without a script, and why

- **`crepes` #49 and #50** are the pull requests *fixing* #46 and #47. A reproducer for a
  fix is the reproducer for the defect it fixes, which is already here. When either lands,
  the corresponding script flips to "does not reproduce" — which is the signal, and
  duplicating it under a second filename would only make the directory look busier.
- **`mapie` #958** is eight inline findings raised in review on a third party's pull
  request, merged with the blocking one unfixed. There is no released behaviour to run:
  the artifact is a review thread. Reproducing it would mean checking out an unmerged
  branch, which is not what "standalone, seconds to run, no harness" describes.

## Direction of harm is a column for a reason

Four of these are anti-conservative — the interval or set is smaller than the requested
level warrants — and one is conservative, costing width. Three are documentation defects
where the shipped behaviour is defensible and the sentence beside it is not. That
distinction decides how a maintainer should prioritise, so each script states it in its
own docstring and prints the evidence for it rather than asserting it.

## Errors worth recording

Four, because each is a way a reproducer can lie.

The first version of `crepes_47_small_calibration.py` reported zero warnings from a
path that warns on every call. The helper returned `len(caught), "", fn()` — Python
evaluates the warning count *before* the call that produces the warnings. The
reproducer would have concluded that the regressor is as silent as the classifier, and
the contrast it exists to show would have disappeared in the direction that makes the
library look worse.

**A first `mapie` #974 script passed `1 - coverage` as the quantile level.** `get_quantile`
takes the level itself — `alpha_cor = ceil(alpha_ref * (n+1)) / n` — so the call returned
the lower tail, the clamp was never reached, and the script reported "does not reproduce"
against a defect that is there. Corroboration that the fix is right: it now reproduces the
figure the filing states, 0.9091 against 0.95 at n = 10.

**A first `crepes` #46 script read only `ConformalClassifier.predict_set`.** The sentence
sits on a sibling method that takes the online arguments. It now scans every public
docstring and asserts it found the sentence somewhere, rather than concluding from one
place that it is gone.

**A first `statsforecast` #1202 script matched the sentence with `[^.]*\.`** — which stops
inside "e.g." and drops the worked example the filing is about.
