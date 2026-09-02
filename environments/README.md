# Environments

`probe-requirements.txt` names four environments and says which probe needs which.
This directory records two of them exactly, as `pip freeze --all` of the
interpreters that produced the committed outputs.

| | environment | recorded here | why |
|---|---|---|---|
| [1] | forecasting: sktime, statsforecast, river | `forecasting.lock.txt` | the interpreter most outputs came from |
| [2] | darts | `darts.lock.txt` | pins pandas 3.x, which [1] rejects |
| [3] | tabular: mapie, crepes, puncc, openml, torchcp, nonconformist | not recorded | no freeze is offered because none was taken from the interpreter that ran it, and a reconstructed one would be a guess presented as a record |
| [4] | out-of-census: crepes-weighted, skforecast, conformal-tights | not recorded | as [3], and each of the three ran in an environment of its own |

The two files that are here are records rather than curated lists. They say what
ran; `probe-requirements.txt` says what was asked for. Where the two disagree the
freeze is the fact.

**No container image is shipped.** One was considered and not built, because it
could not be built and run here, and an unexecuted `Dockerfile` in an artifact is a
claim about reproducibility rather than a demonstration of it. The two lockfiles and
`probe-requirements.txt` are what this deposit can stand behind.

## The short path, which needs none of this

```bash
python3 -m venv .venv && .venv/bin/pip install numpy
make verify
```

`verify_headline.py` re-derives the quantities in the manuscript's abstract from the
committed probe outputs and prints each beside the output line it was read from. It
needs numpy and nothing else. The full reproduction needs the environments above.
