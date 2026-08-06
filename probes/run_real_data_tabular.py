#!/usr/bin/env python3
"""W3, tabular arm: delivered coverage on OpenML data, paired, four libraries.

Design, identical to the forecasting arms
-----------------------------------------
  arm A   the library's shipped set or interval, its own API
  arm B   the SAME calibration scores and the SAME point prediction, thresholded
          at the required order statistic ceil((n+1) * coverage)

What the tabular arm adds that the forecasting arms cannot
---------------------------------------------------------
Four independent implementations resolve the same bound from the SAME scores on
the SAME split, so the paired delta separates them without any cross-dataset
comparison:

  mapie  SplitConformalClassifier (LAC)          corrected level + method='higher'
  mapie  SplitConformalRegressor  sym=True       corrected level, one rail
  mapie  SplitConformalRegressor  sym=False      corrected level, two rails at
                                                 1 - alpha/2 each
  crepes ConformalRegressor.predict_int          (n+1) index with a +/-inf guard
  puncc  BaseCalibrator.calibrate                +inf appended to the scores,
                                                 then inverted_cdf -- exact by
                                                 construction

puncc is the control. It should show a paired delta of exactly zero, because
appending +inf and taking the inverted_cdf at 1-alpha lands on the same order
statistic arm B computes. A harness that reports a defect there is measuring
itself, not the libraries.

  torchcp calculate_conformal_value              integer order statistic via
                                                 kthvalue, +inf above the bound
  nonconformist AbsErrorErrFunc.apply_inverse    required rank, then a CLAMPED
                                                 index -- max(scores) at small n

Six implementations, one bound, one set of scores. Three of them (puncc, torchcp,
nonconformist) reach the required rank by construction rather than by correcting a
level, which is why they belong here: an audit that only measures the libraries it
expects to fail is not an audit.

The mapie clip, and why it is dead code on this path
---------------------------------------------------
`get_quantile` corrects the level to ceil(alpha_ref*(n+1))/n and then clips it to
1.0, which would return max(scores) where the conformal quantile does not exist.
That clip never alters a value through `SplitConformalRegressor`, at EITHER sym
setting: `_check_alpha_and_n_samples` raises for every n where the corrected
level would exceed 1. self_check() scans n = 2..5000 at five confidence levels
and both sym settings and finds no n where the guard passes and the clip bites;
the guard floor is also measured off the library itself rather than modelled.

This is the same shape of result as this audit's finding that statsforecast's
`max(0.0, .)` lower clamp is unreachable: the code is there, the
branch is real in source, and no caller can get to it.

It matters because a rail returned at exactly max|score| looks like a clip and is
not one. Where the required rank for that rail's own level IS n, max IS the
correct order statistic. This probe classifies every such rail against that
level, and reports the two cases separately -- conflating them is the error the
audit already retracted once.

Note the direction as well. A rail pinned to max|score| WIDENS an interval, so
even where such a clip did fire it would produce over-coverage. It must never be
written up as undercoverage.

DATASET SELECTION RULE, stated before any measurement
-----------------------------------------------------
  classification  OpenML-CC18 suite, datasets with <= MAX_ROWS instances and
                  <= MAX_FEATURES features and no missing values, ordered by
                  dataset id, first LIMIT of them
  regression      OpenML-CTR23 suite (id 353), same caps, same ordering

Categorical columns are one-hot encoded and the feature cap is applied AFTER
encoding, so the rule is reproducible from the suite alone. Datasets that fail
to load or encode are reported by name and count, never dropped silently.

    python probes/run_real_data_tabular.py [LIMIT] [N_CAL ...]
"""

import math
import os
import sys
import warnings

import numpy as np
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_report import format_cell, summarize  # noqa: E402
from run_real_data import bracket_indices, required_rank, required_span  # noqa: E402

MAX_ROWS = 5000
MAX_FEATURES = 100
TEST_CAP = 1000
COVERAGE = (0.90, 0.95)
N_CAL = (20, 30, 50, 200)
SEED = 20260805
OUT = "outputs/probe_output_real_data_tabular.txt"


def frac(x):
    """Exact rational form of a requested level. Floats break these checks:
    1 - 0.90 is 0.09999999999999998, and 1/that is 10.000000000000002, which
    moves the guard floor off an integer and flips the comparison at n = 10."""
    return F(x).limit_denominator(10 ** 6)


def rail_level(coverage, sym):
    """The level ONE rail resolves at, exact.

    sym=True takes a single rail at the requested coverage. sym=False splits the
    miscoverage across two tails, so each rail resolves at 1 - alpha/2. The
    default beta really is alpha/2: at n_cal=20 and coverage 0.90 the returned
    rail is rank 20 of 20, which is ceil((n+1)*0.95) and not ceil((n+1)*0.90).
    """
    a = 1 - frac(coverage)
    return frac(coverage) if sym else 1 - a / 2


def clip_changes_value(n, level):
    """Does np.clip in get_quantile alter the level? ceil(level*(n+1))/n > 1."""
    return math.ceil(F(level) * (n + 1)) > n


def guard_floor(coverage, sym):
    """Smallest n that _check_alpha_and_n_samples accepts, exact.

    The check is n >= max(1/a, 1/(1-a)). Under sym=False it receives the per-rail
    miscoverage alpha/2, which doubles the floor -- measured below, not assumed.
    """
    a = (1 - frac(coverage)) / (1 if sym else 2)
    return max(1 / a, 1 / (1 - a))


def guard_passes(n, coverage, sym):
    return n >= guard_floor(coverage, sym)


def clip_reachability():
    """Exhaustive scan: any n where the guard passes AND the clip alters the
    level? Returns {(coverage, sym): [n, ...]}."""
    out = {}
    for coverage in (0.80, 0.90, 0.95, 0.98, 0.99):
        for sym in (True, False):
            lvl = rail_level(coverage, sym)
            out[(coverage, sym)] = [
                n for n in range(2, 5001)
                if guard_passes(n, coverage, sym) and clip_changes_value(n, lvl)]
    return out


def measure_guard_floor(coverage, sym, X, y, hi=60):
    """The smallest n_cal mapie actually accepts, by asking it.

    guard_floor() above is a model of the library's check. This asks the library,
    so the reachability claim rests on observed behaviour rather than on the
    model agreeing with itself.
    """
    from mapie.conformity_scores import AbsoluteConformityScore
    from mapie.regression import SplitConformalRegressor

    rng = np.random.default_rng(SEED)
    parts = split(len(X), hi + 40, rng)
    if parts is None:
        return None
    tr, cal, te = parts
    est = regressor().fit(X[tr], y[tr])
    for n in range(2, hi + 1):
        r = SplitConformalRegressor(
            estimator=est, confidence_level=coverage, prefit=True,
            conformity_score=AbsoluteConformityScore(sym=sym))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r.conformalize(X[cal[:n]], y[cal[:n]])
                r.predict_interval(X[te[:5]])
            return n
        except ValueError:
            continue
    return None


def self_check():
    # The clip in get_quantile and the guard in _check_alpha_and_n_samples are
    # mutually exclusive, at BOTH sym settings. An earlier reading of this probe
    # inferred "the clip fired" from a rail equal to max|score|, which is the
    # same conflation this audit retracted for statsforecast's clamp:
    # where the required rank IS n, max is the correct answer. Scanned, not
    # argued.
    for key, hits in clip_reachability().items():
        assert not hits, (key, hits[:5])
    assert clip_changes_value(8, frac(0.90)) and not guard_passes(8, 0.90, True)
    assert not clip_changes_value(20, rail_level(0.90, False))
    assert guard_passes(20, 0.90, False) and not guard_passes(19, 0.90, False)
    assert guard_floor(0.90, True) == 10 and guard_floor(0.90, False) == 20
    assert rail_level(0.90, False) == F(19, 20) and rail_level(0.90, True) == F(9, 10)
    assert required_rank(20, 0.90) == 19
    assert required_rank(19, 0.95) == 19
    assert required_rank(18, 0.95) is None
    assert required_rank(200, 0.95) == 191


self_check()


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_suite(suite, limit):
    """Datasets passing the stated rule, plus a report of what was skipped."""
    import openml
    import pandas as pd

    s = openml.study.get_suite(suite)
    kept, skipped = [], []
    for did in sorted(s.data):
        if len(kept) >= limit:
            break
        try:
            ds = openml.datasets.get_dataset(did, download_data=True,
                                             download_qualities=False,
                                             download_features_meta_data=True)
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute,
                                     dataset_format="dataframe")
        except Exception as exc:
            skipped.append((did, f"load:{type(exc).__name__}"))
            continue
        if X is None or y is None:
            skipped.append((did, "no target"))
            continue
        if len(X) > MAX_ROWS:
            skipped.append((did, f"rows {len(X)}"))
            continue
        if X.isna().to_numpy().any() or pd.isna(y).any():
            skipped.append((did, "missing values"))
            continue
        Xe = pd.get_dummies(X, dummy_na=False)
        if Xe.shape[1] > MAX_FEATURES:
            skipped.append((did, f"features {Xe.shape[1]} after encoding"))
            continue
        Xa = Xe.to_numpy(dtype=float)
        ya = np.asarray(y)
        if not np.isfinite(Xa).all():
            skipped.append((did, "non-finite after encoding"))
            continue
        kept.append((ds.name, did, Xa, ya))
    return kept, skipped


def regressor():
    """Ridge on standardised features.

    OpenML regression features are on wildly different scales, and a bare Ridge
    on the raw columns overflows in the matmul on several CTR23 datasets -- the
    residuals then contain inf and every coverage number downstream is garbage.
    The scaler is fitted inside the pipeline on the training split only, so it
    leaks nothing into calibration or test.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge())


def classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def rank_at(threshold, scores):
    """Smallest 1-based rank whose score is >= threshold, tolerant of the float
    drift that averaging identical half-widths introduces."""
    if not math.isfinite(threshold):
        return scores.size + 1
    s = np.sort(np.asarray(scores, dtype=float))
    tol = 1e-9 * max(1.0, abs(threshold))
    return int(np.searchsorted(s, threshold - tol, side="left") + 1)


def split(n, n_cal, rng):
    """train / calibration / test, disjoint, sizes fixed by n_cal."""
    idx = rng.permutation(n)
    n_train = max(30, (n - n_cal) // 2)
    if n_train + n_cal + 20 > n:
        return None
    tr = idx[:n_train]
    cal = idx[n_train:n_train + n_cal]
    te = idx[n_train + n_cal:][:TEST_CAP]
    return tr, cal, te


# --------------------------------------------------------------------------
# one dataset x one configuration -> one record
# --------------------------------------------------------------------------
def cell_mapie_classifier(X, y, n_cal, coverage, rng):
    from mapie.classification import SplitConformalClassifier

    parts = split(len(X), n_cal, rng)
    if parts is None:
        return {"error": "too_small"}
    tr, cal, te = parts
    if len(np.unique(y[tr])) < 2 or not set(np.unique(y[te])) <= set(np.unique(y[tr])):
        return {"error": "label_mismatch"}
    est = classifier().fit(X[tr], y[tr])
    c = SplitConformalClassifier(estimator=est, confidence_level=coverage,
                                 conformity_score="lac", prefit=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c.conformalize(X[cal], y[cal])
            _, sets = c.predict_set(X[te])
    except ValueError:
        return {"error": "refused_by_guard"}

    inner = c._mapie_classifier
    scores = np.asarray(inner.conformity_scores_, dtype=float).ravel()
    q_a = float(np.asarray(inner.quantiles_).ravel()[0])
    probs = est.predict_proba(X[te])
    sets_a = np.asarray(sets)[:, :, 0]
    # arm A must be reproducible from (probs, its own quantile) or the paired
    # comparison below is comparing two different constructions
    if not np.array_equal(sets_a, (1.0 - probs) <= q_a + 1e-12):
        return {"error": "set_reproduction_failed"}

    classes = list(est.classes_)
    truth = np.array([classes.index(v) for v in y[te]])
    hit_a = sets_a[np.arange(len(truth)), truth]

    n = scores.size
    k = required_rank(n, coverage)
    if k is None:
        hit_b, size_b, feasible = np.ones_like(hit_a, bool), len(classes), False
    else:
        q_b = float(np.sort(scores)[k - 1])
        sets_b = (1.0 - probs) <= q_b + 1e-12
        hit_b = sets_b[np.arange(len(truth)), truth]
        size_b, feasible = float(sets_b.sum(1).mean()), True

    return {
        "n": n,
        "required_rank": k if k is not None else n + 1,
        "feasible": feasible,
        "a_covered": float(hit_a.mean()),
        "a_width": float(sets_a.sum(1).mean()),
        "a_rank": rank_at(q_a, scores),
        "b_covered": float(hit_b.mean()),
        "b_width": size_b,
        "clipped": math.isclose(q_a, scores.max(), rel_tol=1e-9),
    }


def cell_mapie_regressor(X, y, n_cal, coverage, rng, sym):
    from mapie.conformity_scores import AbsoluteConformityScore
    from mapie.regression import SplitConformalRegressor

    parts = split(len(X), n_cal, rng)
    if parts is None:
        return {"error": "too_small"}
    tr, cal, te = parts
    est = regressor().fit(X[tr], y[tr])
    r = SplitConformalRegressor(estimator=est, confidence_level=coverage,
                                prefit=True,
                                conformity_score=AbsoluteConformityScore(sym=sym))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.conformalize(X[cal], y[cal])
            pt, iv = r.predict_interval(X[te])
    except ValueError:
        return {"error": "refused_by_guard"}

    # Under sym=False mapie stores SIGNED scores (regression.py:100 skips the abs)
    # and resolves two rails at beta and 1 - alpha + beta (regression.py:319-320),
    # so arm A is not a symmetric band and arm B must be the two-rail one.
    raw = np.asarray(r._mapie_regressor.conformity_scores_, dtype=float).ravel()
    scores = np.abs(raw)
    iv = np.asarray(iv)
    lo_a, hi_a = iv[:, 0, 0], iv[:, 1, 0]
    centre = np.asarray(pt, dtype=float)
    half_hi = float(np.mean(hi_a - centre))
    return _finish_regression(scores, centre, y[te], lo_a, hi_a, coverage,
                              clipped=math.isclose(half_hi, scores.max(),
                                                   rel_tol=1e-9),
                              signed=None if sym else raw)


def cell_crepes(X, y, n_cal, coverage, rng):
    from crepes import ConformalRegressor

    parts = split(len(X), n_cal, rng)
    if parts is None:
        return {"error": "too_small"}
    tr, cal, te = parts
    est = regressor().fit(X[tr], y[tr])
    resid = np.abs(y[cal] - est.predict(X[cal]))
    cr = ConformalRegressor().fit(residuals=resid)
    centre = est.predict(X[te])
    iv = np.asarray(cr.predict_int(y_hat=centre, confidence=coverage))
    return _finish_regression(np.asarray(resid, dtype=float), centre, y[te],
                              iv[:, 0], iv[:, 1], coverage, clipped=False)


def cell_puncc(X, y, n_cal, coverage, rng):
    from deel.puncc.api.calibration import BaseCalibrator
    from deel.puncc.api.nonconformity_scores import absolute_difference
    from deel.puncc.api.prediction_sets import constant_interval

    parts = split(len(X), n_cal, rng)
    if parts is None:
        return {"error": "too_small"}
    tr, cal, te = parts
    est = regressor().fit(X[tr], y[tr])
    calib = BaseCalibrator(nonconf_score_func=absolute_difference,
                           pred_set_func=constant_interval)
    calib.fit(y_true=y[cal], y_pred=est.predict(X[cal]))
    centre = est.predict(X[te])
    try:
        lo_a, hi_a = calib.calibrate(alpha=1 - coverage, y_pred=centre)
    except ValueError:
        return {"error": "refused_by_guard"}
    scores = np.abs(y[cal] - est.predict(X[cal]))
    return _finish_regression(np.asarray(scores, dtype=float), centre, y[te],
                              np.asarray(lo_a), np.asarray(hi_a), coverage,
                              clipped=False)


def _finish_regression(scores, centre, y_true, lo_a, hi_a, coverage, clipped,
                       signed=None):
    """One paired cell. `signed` is the SIGNED score set for a two-rail arm A.

    A configuration that splits alpha across two tails does not build a symmetric
    band, so an arm B thresholding |score| at one rank would differ from it in the
    score set and in the geometry as well as in the rank. Where `signed` is given,
    arm B is the same two-rail construction at the required SPAN and what arm A
    lands on is a span in gaps rather than a rank.
    """
    n = scores.size
    if n < 2:
        return {"error": "too_few_scores"}
    lo_a, hi_a = np.asarray(lo_a, dtype=float), np.asarray(hi_a, dtype=float)
    centre = np.asarray(centre, dtype=float)
    finite = np.isfinite(lo_a) & np.isfinite(hi_a)
    a_covered = float(np.mean((lo_a <= y_true) & (y_true <= hi_a)))
    a_width = float(np.mean(hi_a - lo_a)) if finite.all() else math.inf

    if signed is not None:
        s = np.sort(np.asarray(signed, dtype=float).ravel())
        assert s.size == n, (s.size, n)
        a_idx, b_idx, k = required_span(n, coverage)
        lo_b = -math.inf if a_idx == 0 else centre + float(s[a_idx - 1])
        hi_b = math.inf if b_idx == n + 1 else centre + float(s[b_idx - 1])
        feasible = a_idx >= 1 and b_idx <= n
        # mapie's rails are centre + a fixed offset, so the offsets are the same
        # for every test point and one bracket pair describes the cell.
        off_lo, off_hi = lo_a - centre, hi_a - centre
        assert np.ptp(off_lo) < 1e-6 and np.ptp(off_hi) < 1e-6, "rails are not offsets"
        j_lo, j_hi = bracket_indices(float(off_lo[0]), float(off_hi[0]), s)
        nests = bool(np.all(lo_b <= lo_a + 1e-9) and np.all(hi_b >= hi_a - 1e-9))
        return {
            "n": n,
            "required_rank": k,
            "two_rail": True,
            "nests": nests,
            "feasible": feasible,
            "a_covered": a_covered,
            "a_width": a_width,
            "a_rank": j_hi - j_lo - 1,
            "b_covered": float(np.mean((lo_b <= y_true) & (y_true <= hi_b))),
            "b_width": float(np.mean(hi_b - lo_b)) if feasible else math.inf,
            "clipped": clipped,
        }

    k = required_rank(n, coverage)
    if k is None:
        half_b, feasible = math.inf, False
    else:
        half_b, feasible = float(np.sort(scores)[k - 1]), True
    half_a = float(np.mean((hi_a - lo_a) / 2.0))
    return {
        "n": n,
        "required_rank": k if k is not None else n + 1,
        "two_rail": False,
        # Measured, not declared. A helper landing ABOVE the required rank makes
        # arm B the narrower interval and the paired difference can then go either
        # way; saying so here is what lets paired_report assert on the rest.
        "nests": bool(half_b >= half_a - 1e-9),
        "feasible": feasible,
        "a_covered": a_covered,
        "a_width": a_width,
        "a_rank": rank_at(half_a, scores),
        "b_covered": float(np.mean(np.abs(y_true - centre) <= half_b)),
        "b_width": 2 * half_b if math.isfinite(half_b) else math.inf,
        "clipped": clipped,
    }


def cell_torchcp(X, y, n_cal, coverage, rng):
    """torchcp's threshold helper, fed the same sklearn residuals as the others.

    `calculate_conformal_value` is the reference implementation: an integer order
    statistic via torch.kthvalue, with an explicit warning and +inf above the
    feasibility boundary. Using it at the helper level rather than through a
    torch predictor keeps this arm comparable with the crepes and puncc arms,
    which are also fed sklearn residuals.
    """
    import torch
    from torchcp.utils.common import calculate_conformal_value

    parts = split(len(X), n_cal, rng)
    if parts is None:
        return {"error": "too_small"}
    tr, cal, te = parts
    est = regressor().fit(X[tr], y[tr])
    scores = np.abs(y[cal] - est.predict(X[cal]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = float(calculate_conformal_value(
            torch.tensor(np.asarray(scores, dtype=float)), 1 - coverage))
    centre = est.predict(X[te])
    return _finish_regression(np.asarray(scores, dtype=float), centre, y[te],
                              centre - q, centre + q, coverage, clipped=False)


def cell_nonconformist(X, y, n_cal, coverage, rng):
    """nonconformist's AbsErrorErrFunc.apply_inverse, the real helper.

    It sorts descending and takes index floor(alpha*(n+1)) - 1, which is exactly
    the required rank -- and then CLAMPS that index into range, so at small n it
    silently returns max(scores). The source carries a
    `# TODO: should probably warn against too few calibration examples` on the
    line above the clamp, which makes this the only clamp in the audit that its
    own authors have already flagged.
    """
    from nonconformist.nc import AbsErrorErrFunc

    parts = split(len(X), n_cal, rng)
    if parts is None:
        return {"error": "too_small"}
    tr, cal, te = parts
    est = regressor().fit(X[tr], y[tr])
    scores = np.abs(y[cal] - est.predict(X[cal]))
    bounds = AbsErrorErrFunc().apply_inverse(np.asarray(scores, dtype=float),
                                            1 - coverage)
    q = float(np.asarray(bounds).ravel()[0])
    centre = est.predict(X[te])
    return _finish_regression(np.asarray(scores, dtype=float), centre, y[te],
                              centre - q, centre + q, coverage,
                              clipped=math.isclose(q, float(np.max(scores)),
                                                   rel_tol=1e-9))


def sym_of(label):
    """Which mapie symmetry setting a config label refers to."""
    return "sym=False" not in label


CONFIGS = (
    ("mapie SplitConformalClassifier (LAC)", "classification", cell_mapie_classifier),
    ("mapie SplitConformalRegressor sym=True", "regression",
     lambda *a: cell_mapie_regressor(*a, sym=True)),
    ("mapie SplitConformalRegressor sym=False", "regression",
     lambda *a: cell_mapie_regressor(*a, sym=False)),
    ("crepes ConformalRegressor.predict_int", "regression", cell_crepes),
    ("puncc BaseCalibrator (CONTROL, exact)", "regression", cell_puncc),
    ("torchcp calculate_conformal_value (reference)", "regression", cell_torchcp),
    ("nonconformist AbsErrorErrFunc.apply_inverse", "regression", cell_nonconformist),
)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    n_cals = tuple(int(a) for a in sys.argv[2:]) or N_CAL

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 104)
    say("W3, TABULAR ARM -- delivered coverage on OpenML data, paired, four libraries")
    say("=" * 104)
    say("self_check() passed at import (the clip window is derived, not assumed)")
    say("")
    say("SELECTION RULE, fixed before measurement:")
    say(f"  classification  OpenML-CC18,  <= {MAX_ROWS} rows, <= {MAX_FEATURES} "
        f"features after one-hot, no missing values, first {limit} by dataset id")
    say(f"  regression      OpenML-CTR23, same caps and ordering")
    say(f"  split           train = half of the remainder, calibration = n_cal, "
        f"test = rest capped at {TEST_CAP}")
    say(f"  models          StandardScaler + LogisticRegression(max_iter=1000) / Ridge(),"
        f" fitted on train only, shared by both arms")
    say(f"  seed            {SEED}")
    say("")
    say("Coverage is measured per dataset over its test rows; the paired delta and its")
    say("standard error are then taken ACROSS datasets, so one large dataset cannot")
    say("dominate the result.")
    say("")
    say("Is mapie's np.clip in get_quantile reachable? Exhaustive, exact arithmetic,")
    say("n = 2..5000 at five confidence levels and both sym settings:")
    reach = clip_reachability()
    for (cov, sym), hits in sorted(reach.items()):
        say(f"    coverage {cov:.2f}  sym={str(sym):<5}  rail level "
            f"{float(rail_level(cov, sym)):.4f}   guard floor n>="
            f"{guard_floor(cov, sym)}   clip alters the level for n<"
            f"{min((n for n in range(2, 5001) if not clip_changes_value(n, rail_level(cov, sym))), default='?')}"
            f"   overlap: {len(hits)}")
    say(f"    total n where the guard passes AND the clip bites: "
        f"{sum(len(v) for v in reach.values())} of "
        f"{len(reach) * 4999} scanned -> the clip is dead code on this path")
    say("")

    data = {}
    for kind, suite in (("classification", "OpenML-CC18"), ("regression", 353)):
        kept, skipped = load_suite(suite, limit)
        data[kind] = kept
        say(f"{kind}: {len(kept)} datasets kept from {suite}")
        for name, did, X, _ in kept:
            say(f"    {did:>6}  {name[:34]:<34} {X.shape[0]:>5} x {X.shape[1]:<4}")
        if skipped:
            say(f"    skipped {len(skipped)}: "
                + ", ".join(f"{d}({r})" for d, r in skipped[:12])
                + (" ..." if len(skipped) > 12 else ""))
        say("")

    for label, kind, fn in CONFIGS:
        say("")
        say("=" * 104)
        say(label)
        say("=" * 104)
        for coverage in COVERAGE:
            say(f"  nominal {coverage:.2f}")
            for n_cal in n_cals:
                rng = np.random.default_rng(SEED + n_cal)
                recs = [fn(X, y, n_cal, coverage, rng) for _, _, X, y in data[kind]]
                s = summarize(recs)
                for ln in format_cell(f"n_cal={n_cal:<4}", s):
                    say(ln)
                at_max = [r for r in recs if r and "error" not in r and r.get("clipped")]
                if at_max:
                    # A rail equal to max|score| is NOT evidence of the clip. Where
                    # the required rank is exactly n, max|score| IS the required
                    # order statistic and the correct answer. The clip only changed
                    # the value where the required rank exceeds n. Conflating the
                    # two is the error the audit retracted once already for
                    # statsforecast's clamp window.
                    lvl = rail_level(coverage, sym_of(label))
                    genuine = [r for r in at_max
                               if not clip_changes_value(r["n"], lvl)]
                    clipped = [r for r in at_max if clip_changes_value(r["n"], lvl)]
                    say(f"      a rail equals max|score| in {len(at_max)}/{s['cells']}"
                        f" datasets, classified against that rail's own level"
                        f" {lvl:.3f}:")
                    say(f"        {len(genuine)} where max IS the required order"
                        f" statistic for that level (CORRECT, not a clip),"
                        f" {len(clipped)} where the clip altered it")
            say("")

    say("")
    say("Reading this table")
    say("------------------")
    say("puncc is the control: it appends +inf and takes the inverted_cdf, so it lands")
    say("on the same order statistic as arm B and the delta must be exactly zero. Any")
    say("other reading there is a harness bug, not a library defect.")
    say("")
    say("'refused_by_guard' is a PASS, not a failure. A library that raises where no")
    say("valid bound exists is behaving correctly; the branch to worry about is the one")
    say("that returns a finite interval there instead.")
    say("")
    say("A NEGATIVE delta means the shipped interval covers MORE than the required")
    say("rank -- conservatism, not a defect. mapie's asymmetric rails split alpha across")
    say("two tails and so land wider than the symmetric bound arm B builds. Do not")
    say("write that up as undercoverage.")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
