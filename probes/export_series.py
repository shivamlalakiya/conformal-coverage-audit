#!/usr/bin/env python3
"""Cache Monash series as .npz, because the darts arm runs in a different venv.

darts pins numpy/pandas versions that sktime 1.1.0 does not accept, so the two
real-data arms cannot share one environment (probe-requirements.txt records
both). The loader lives with sktime, so it runs once here and writes a plain
.npz that any venv can read. The .npz is a CACHE, not an artifact -- it is not
part of the deposit, and the command that produces it is recorded in the darts
probe's docstring.

    python probes/export_series.py m1_monthly_dataset 250 /tmp/m1_monthly.npz 61
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_real_data import load_series  # noqa: E402

name, limit, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
min_len = int(sys.argv[4]) if len(sys.argv) > 4 else 60

series, meta = load_series(name, limit, min_len=min_len)
np.savez_compressed(
    out,
    meta=np.array([name, str(meta.get("frequency", "?")), str(min_len)]),
    **{f"s{i}": s for i, s in enumerate(series)},
)
print(f"{len(series)} series -> {out}  (freq {meta.get('frequency', '?')}, "
      f"min_len {min_len})")
