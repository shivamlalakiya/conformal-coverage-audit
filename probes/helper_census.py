#!/usr/bin/env python3
"""Count the quantile helpers, with the criterion stated first and every site
located mechanically rather than tallied by hand.

Why this exists
---------------
The abstract used to say "seven have a distinct quantile/p-value helper". That
number named no criterion, so it was not reproducible -- the same failure the
twelve-of-thirteen library count had before it was recounted against C1/C2/C3.
This script is the per-helper equivalent: it states the unit, enumerates every
site under it, and verifies each site still exists where it is recorded.

THE UNIT
--------
A **resolution site**: a single expression converting a nominated coverage or
miscoverage rate into either

  (i)  a position or rank within a calibration nonconformity score array, or
  (ii) a p-value denominator over that array,

and whose output controls a delivered bound, region, prediction set or p-value.

When two public methods share a single expression it is tallied once. A method housing two
expressions under distinct rules is tallied twice. An expression that only feeds a
diagnostic, a metric or an internal clustering step does not count -- those are
listed separately as `determines_output=False` so the exclusion is auditable
rather than silent.

THREE COUNTS, and any write-up must name which one it uses
----------------------------------------------------------
  P1  sites this audit had already documented before the census was built
  P2  every site in the pinned versions that determines a returned bound
      -- the honest total, and the one to use
  P3  public API entry points that route through at least one P2 site
      -- the user-facing surface, always the largest number

Scope, stated because it changes the count
------------------------------------------
Counted over the versions pinned in probe-requirements.txt, which are the
versions the audit read. mapie's `_compute_classification_quantile` and
`QuantileRegressionScore` are master-only -- self_check() asserts they are
absent here, which is why the two mapie master-branch sites are excluded
from P1/P2/P3 and reported on their own line.

Running it
----------
    python probes/helper_census.py [--root DIR]

--root points at a directory holding unpacked package sources, one directory
per distribution (the layout of the audit's own `cp-src/`, which is not
redistributed -- see the deposit README). Without it, each package is located
through its installed module, so `pip install -r probe-requirements.txt` is the
other way to run this.
"""

import argparse
import glob
import importlib.util
import os
import re
import sys

WINDOW = 6  # lines of drift tolerated before a site is reported as MOVED

# ---------------------------------------------------------------------------
# The manifest. `anchor` is verified against the file, so a version bump that
# moves or rewrites a site fails loudly instead of silently changing the count.
#   branch    the taxonomy letter from section 3, or "?" where this audit has
#             located the site but not yet classified it
#   in_map    already documented by this audit before the census
#   output    determines a returned bound/set/p-value (counts toward P2)
#   entries   public API a user calls to reach it (counts toward P3)
# ---------------------------------------------------------------------------
def S(lib, path, line, symbol, anchor, rule, branch, in_map, output, entries):
    return dict(lib=lib, path=path, line=line, symbol=symbol, anchor=anchor,
                rule=rule, branch=branch, in_map=in_map, output=output,
                entries=tuple(entries))


MANIFEST = [
    # ---------------------------------------------------------------- mapie --
    S("mapie 1.4.1", "mapie/conformity_scores/interface.py", 137,
      "BaseConformityScore.get_quantile",
      "alpha_cor = np.ceil(alpha_ref * (n_calib + 1)) / n_calib",
      "corrected level, then np.clip(..., 0, 1) -- the clip is the (b) branch",
      "b", True, True, ["SplitConformalRegressor.predict_interval"]),
    S("mapie 1.4.1", "mapie/utils.py", 766,
      "_compute_quantiles",
      "((n + 1) * (1 - _alpha)) / n,",
      "corrected level, method='higher', NO clip -- np.quantile raises above 1",
      "a", False, True, ["SplitConformalClassifier.predict_set"]),
    S("mapie 1.4.1", "mapie/conformity_scores/sets/lac.py", 156,
      "LACConformityScore.get_conformity_score_quantiles [prefit/mean]",
      "quantiles_ = _compute_quantiles(conformity_scores, alpha_np)",
      "delegates to _compute_quantiles, so it inherits branch (a): the corrected "
      "level is unclipped and np.quantile raises above 1. CLASSIFIED BY RUNNING "
      "in conformance_suite.py, which sees it raise at n=8",
      "a", False, True, ["SplitConformalClassifier.predict_set"]),
    # RE-ANCHORED. This site used to point at lac.py:158, `quantiles_ =
    # (n + 1) * (1 - alpha_np)`, on the grounds that it is where a level becomes a
    # numeric value on the crossval path. It is -- but that value decides nothing. It is
    # stored on `quantiles_`, surfaced as a documented public fitted attribute
    # (classification.py:1141), and never read by the branch that builds the set:
    # perturbing it by a factor of 1000 leaves the returned sets bit-identical,
    # which probes/lac_crossval_dead_value.py executes and commits rather than
    # asserting from a reading. The expression that DOES resolve the
    # level on this path is `_alpha * (n - 1)` inside get_prediction_sets, and it
    # is not the same rule -- it carries no (n+1) correction at all. APS's
    # identical-looking line at aps.py:201 is a genuine site by contrast: its set
    # does move when quantiles_ moves. Anchoring the criterion's own words --
    # "determines a returned set" -- to the wrong line is exactly the failure this
    # census exists to prevent, so the anchor moved rather than the criterion.
    S("mapie 1.4.1", "mapie/conformity_scores/sets/lac.py", 214,
      "LACConformityScore.get_prediction_sets [cv/crossval]",
      "np.greater_equal(y_pred_included - _alpha * (n - 1), -EPSILON)",
      "the crossval branch compares an INCLUSION COUNT against alpha*(n-1), a "
      "count-scale threshold carrying no (n+1) correction and not the (n+1)"
      "(1-alpha) computed at line 158. That line's value is exposed as the public "
      "`quantiles_` attribute and never read here -- scaling it by 1000 leaves the "
      "sets identical. No branch letter applies; conformance_suite.py labels the "
      "count-returning helper `count`",
      "count", False, True, ["CrossConformalClassifier.predict_set"]),
    S("mapie 1.4.1", "mapie/conformity_scores/regression.py", 211,
      "BaseRegressionScore._beta_optimize",
      "_alpha = float(_alpha)",
      "beta grid alpha/(n+1)..alpha, then nanquantile(upper, 1-alpha+betas, "
      "'higher') and nanquantile(lower, betas, 'lower'). It returns a BETA, not a "
      "threshold: it selects the level get_quantile then resolves, so it is one "
      "step removed. Composed with get_quantile and run, the pair is branch (b) "
      "-- corrected level, clamped at the boundary. Reached via "
      "predict_interval(minimize_interval_width=True)",
      "b", False, True, ["CrossConformalRegressor.predict_interval",
                         "SplitConformalRegressor.predict_interval"]),
    S("mapie 1.4.1", "mapie/regression/quantile_regression.py", 1037,
      "_MapieQuantileRegressor.predict",
      "q = (1 - (alpha)) * (1 + (1 / n))",
      "corrected level, method='higher', no clip -- raises above 1",
      "a", False, True, ["ConformalizedQuantileRegressor.predict_interval"]),
    S("mapie 1.4.1", "mapie/utils.py", 547,
      "_check_alpha_and_n_samples",
      "def _check_alpha_and_n_samples(",
      "GUARD, not a resolver: raises when 1/alpha exceeds n",
      "a", False, False, []),

    # --------------------------------------------------------------- crepes --
    S("crepes 0.9.1", "crepes/base.py", 340,
      "ConformalClassifier.predict_set",
      "prediction_sets = (p_values >= 1-confidence).astype(int)",
      "set membership by thresholding p-values at 1 - confidence",
      "f/e", True, True, ["ConformalClassifier.predict_set"]),
    S("crepes 0.9.1", "crepes/base.py", 2909,
      "p_values_batch",
      "(np.sum(alphas_cal >= alphas_test[i])+1)/(q+1)",
      "exact p-value j/(n+1); smoothed variant one branch above",
      "f/e", True, True, ["ConformalClassifier.predict_p",
                          "ConformalRegressor.predict_p",
                          "ConformalPredictiveSystem.predict_p"]),
    S("crepes 0.9.1", "crepes/base.py", 900,
      "ConformalRegressor.predict_int",
      "alpha_index = int((1-confidence)*(len(self.alphas)+1))-1",
      "(n+1) index with an explicit `>= 0` guard, else +/-inf",
      "c", False, True, ["ConformalRegressor.predict_int"]),
    S("crepes 0.9.1", "crepes/base.py", 1037,
      "ConformalRegressor.predict_int_online",
      "alpha_index = int((1-confidence)*(len(alphas_cal)+1))-1",
      "same rule, recomputed per step as the calibration set grows",
      "c", True, True, ["ConformalRegressor.predict_int_online"]),
    S("crepes 0.9.1", "crepes/base.py", 1525,
      "ConformalPredictiveSystem.predict_int",
      "lower_percentile = (1-confidence)/2*100",
      "a TWO-SIDED percentile pair handed to predict_percentiles, so each rail "
      "resolves at 1-(1-conf)/2 and not at conf. Run: returns +inf below the "
      "boundary and WARNS, first delivers 0.90 at n=19. Branch (c). A candidate "
      "set with only one-sided rails misread this as (g), which is why "
      "conformance_suite.py fits both rail conventions",
      "c", False, True, ["ConformalPredictiveSystem.predict_int",
                         "ConformalPredictiveSystem.predict_percentiles"]),
    S("crepes 0.9.1", "crepes/base.py", 1613,
      "ConformalPredictiveSystem.predict_int_online",
      "index_low = int((1-confidence)/2*(len(alphas_cal)+1))-1",
      "two-sided (n+1) index pair, symmetric by construction",
      "c", True, True, ["ConformalPredictiveSystem.predict_int_online"]),

    # ---------------------------------------------------------------- puncc --
    S("puncc 0.9.3", "deel/puncc/api/calibration.py", 268,
      "BaseCalibrator.compute_quantile",
      "lemma_residuals = np.concatenate((residuals, infty_array), axis=0)",
      "appends +inf to the scores, then inverted_cdf at 1 - alpha -- EXACT by "
      "construction rather than by correcting a level, and it sits on the base "
      "class, so EVERY puncc calibrator inherits it",
      "c-exact", False, True, ["BaseCalibrator.calibrate",
                               "SplitCP.predict", "CvPlus.predict"]),
    S("puncc 0.9.3", "deel/puncc/api/calibration.py", 374,
      "ClasswiseCalibrator.compute_quantile",
      "class_scores_with_inf, 1 - alpha, method=\"inverted_cdf\"",
      "the same augmentation per class, with an explicit +inf and a warning "
      "when a class has no calibration samples",
      "c-exact", False, True, ["ClasswiseCalibrator.calibrate"]),
    S("puncc 0.9.3", "deel/puncc/api/utils.py", 423,
      "quantile",
      "return np.quantile(a, q, axis=axis, method=\"inverted_cdf\")",
      "shared helper. method='inverted_cdf' lands on rank ceil(level*n), so "
      "given a RAW level it delivers the uncorrected rank -- branch (d), verified "
      "by running it. puncc is correct not because of this function but because "
      "every caller appends +inf to the scores before calling it",
      "d", False, True, ["BaseCalibrator.calibrate"]),
    S("puncc 0.9.3", "deel/puncc/api/calibration.py", 254,
      "BaseCalibrator.compute_quantile (Bonferroni)",
      "alpha = correction(alpha)",
      "alpha is Bonferroni-divided across output dimensions BEFORE the "
      "feasibility check, so multivariate targets tighten the check rather "
      "than loosen it",
      "?", False, False, []),
    S("puncc 0.9.3", "deel/puncc/api/utils.py", 247,
      "alpha_calib_check",
      "def alpha_calib_check(",
      "GUARD, not a resolver: raises with 1/(n+1) <= alpha < 1 derived in the "
      "docstring, plus the two-sided n/(n+1) bound for jackknife+/CV+",
      "a", True, False, []),
    S("puncc 0.9.3", "deel/puncc/regression.py", 847,
      "EnbPI.predict",
      "res_quantile = np.quantile(self.residuals, (1 - alpha), method=\"linear\")",
      "UNCORRECTED level with explicit method='linear'; the source comment "
      "beside it reads 'TODO: go back to EnbPI-v1 paper and double check'",
      "d", False, True, ["EnbPI.predict"]),
    S("puncc 0.9.3", "deel/puncc/regression.py", 896,
      "EnbPI.predict (online update)",
      "res_quantile = np.quantile(updated_residuals, (1 - alpha))",
      "uncorrected level, numpy default interpolation, recomputed per batch",
      "d", False, True, ["EnbPI.predict"]),

    # --------------------------------------------------------------- torchcp --
    S("torchcp 1.2.1", "torchcp/utils/common.py", 74,
      "calculate_conformal_value",
      "torch.kthvalue(scores, math.ceil((N + 1) * (1 - alpha)), dim=0)",
      "integer order statistic ceil((N+1)(1-alpha)); tests > 1 first, warns "
      "and returns torch.inf -- the reference implementation",
      "c", True, True, ["SplitPredictor.calibrate",
                        "ClassConditionalPredictor.calibrate",
                        "ClusterPredictor.calibrate",
                        "RC3P.calibrate",
                        "GraphSplitPredictor.calibrate"]),
    S("torchcp 1.2.1", "torchcp/classification/utils/metrics.py", 425,
      "compute_p_values",
      "p_values = (greater + (equal + 1)) / (n_cal + 1)",
      "exact and smoothed p-values, but this module is metrics -- no predictor "
      "routes a returned set through it",
      "e/f", True, False, []),
    S("torchcp 1.2.1", "torchcp/classification/predictor/cluster.py", 278,
      "ClusterPredictor.__embed_all_classes",
      "torch.kthvalue(class_i_scores, int(math.ceil(cts[i] * q[k])), dim=0)",
      "UNCORRECTED ceil(n*q) rank, but it builds clustering embeddings; the "
      "returned qhats come from calculate_conformal_value",
      "d", False, False, []),

    # --------------------------------------------------------- nonconformist --
    S("nonconformist 2.1.0", "nonconformist/icp.py", 241,
      "IcpClassifier.predict",
      "p[j, i] = n_gt / (n_cal + 1)",
      "n_gt/(n_cal+1), smoothed term added at :244, non-smoothed at :246",
      "e/f", True, True, ["IcpClassifier.predict"]),
    S("nonconformist 2.1.0", "nonconformist/nc.py", 162,
      "AbsErrorErrFunc.apply_inverse",
      "border = int(np.floor(significance * (nc.size + 1))) - 1",
      "floor((n+1)*significance) - 1 into scores sorted DESCENDING, which is "
      "exactly the required rank -- then `min(max(border, 0), nc.size - 1)` "
      "CLAMPS the index, so below the feasibility bound it silently returns "
      "max(scores). The line above the clamp reads `# TODO: should probably warn "
      "against too few calibration examples`: the only clamp in this census that "
      "its own authors have already flagged. Classified by running, not reading "
      "-- see conformance_suite.py",
      "b", False, True, ["IcpRegressor.predict"]),
    S("nonconformist 2.1.0", "nonconformist/nc.py", 191,
      "SignErrorErrFunc.apply_inverse",
      "upper = int(np.floor((significance / 2) * (nc.size + 1)))",
      "the two-rail variant of the same rule, with the same pair of clamps",
      "b", False, True, ["IcpRegressor.predict"]),

    # --------------------------------------------------------------- sktime --
    S("sktime 1.1.0", "sktime/forecasting/conformal.py", 319,
      "ConformalIntervals._predict_interval [method='empirical']",
      "quantiles = 0.5 + np.tile([-0.5, 0.5], len(coverage)) * coverage2",
      "two uncorrected levels 0.5 -/+ coverage/2 on SIGNED residuals",
      "d", True, True, ["ConformalIntervals.predict_interval"]),
    S("sktime 1.1.0", "sktime/forecasting/conformal.py", 322,
      "ConformalIntervals._predict_interval [method='empirical_residual']",
      "quantiles = 0.5 - 0.5 * coverage2",
      "0.5 - coverage/2 on ABSOLUTE residuals; at coverage 0.90 both rails "
      "resolve to the 0.05 quantile of |resid|",
      "d", True, True, ["ConformalIntervals.predict_interval"]),
    S("sktime 1.1.0", "sktime/forecasting/conformal.py", 326,
      "ConformalIntervals._predict_interval [method='conformal_bonferroni']",
      "quantiles = 1 - alphas / len(fh)",
      "uncorrected level with a Bonferroni split across the horizon",
      "d", True, True, ["ConformalIntervals.predict_interval"]),
    S("sktime 1.1.0", "sktime/forecasting/conformal.py", 329,
      "ConformalIntervals._predict_interval [method='conformal']",
      "quantiles = coverage2",
      "the requested coverage used directly as the level on |resid|",
      "d", True, True, ["ConformalIntervals.predict_interval"]),
    S("sktime 1.1.0", "sktime/libs/_aws_fortuna_enbpi/enbpi.py", 128,
      "EnbPI.conformal_interval",
      "residuals_quantile = np.quantile(train_residuals, q=1 - error, axis=0)",
      "uncorrected 1 - error on bootstrap residuals, numpy default "
      "interpolation; reached from sktime's own EnbPIForecaster",
      "d", False, True, ["EnbPIForecaster.predict_interval"]),

    # ---------------------------------------------------------------- river --
    S("river 0.25.0", "river/conf/jackknife.py", 97,
      "RegressionJackknife.predict_one",
      "alpha = (1 - confidence_level) / 2",
      "two uncorrected levels fed to stats.Quantile / RollingQuantile, which "
      "is the P-squared APPROXIMATION, not an empirical order statistic",
      "g", True, True, ["RegressionJackknife.predict_one"]),

    # -------------------------------------------------------- statsforecast --
    S("statsforecast 2.1.1", "statsforecast/models.py", 127,
      "_add_conformal_distribution_intervals",
      "cuts = [alpha / 200 for alpha in reversed(alphas)]",
      "uncorrected alpha/200 and 1 - alpha/200 over vstack([mean-cs, mean+cs])",
      "d", True, True, ["StatsForecast.forecast(prediction_intervals=ConformalIntervals())"]),
    S("statsforecast 2.1.1", "statsforecast/models.py", 156,
      "_add_conformal_error_intervals",
      "quantiles = {lv: np.quantile(cs, lv / 100, axis=0) for lv in level}",
      "uncorrected lv/100 on absolute conformity scores",
      "d", True, True, ["StatsForecast.forecast(prediction_intervals=ConformalIntervals())"]),
    S("statsforecast 2.1.1", "statsforecast/models.py", 4302,
      "ConformalSeasonalPool._oriented_index",
      "return min(1.0, float(np.ceil((n + 1.0) * q)) / n)",
      "corrected level, clamped: min(1.0, ceil((n+1)q)/n) upper, "
      "max(0.0, floor((n+1)q)/n) lower -- the lower max() is dead code",
      "b", True, True, ["ConformalSeasonalPool.forecast"]),
    S("statsforecast 2.1.1", "statsforecast/models.py", 4311,
      "ConformalSeasonalPool._intervals_from_samples",
      "quantiles = np.quantile(samples, oriented, axis=0)",
      "consumes _oriented_index over SAMPLE PATHS; a second call site with the "
      "same rule but a different score set",
      "b", False, True, ["ConformalSeasonalPool.forecast"]),

    # ---------------------------------------------------------------- darts --
    S("darts 0.46.1", "darts/models/forecasting/conformal_models.py", 1688,
      "ConformalNaiveModel._calibrate_interval",
      "q=self.interval_range_sym,",
      "ONE uncorrected level (q_high - q_low) with method='higher' over "
      "absolute errors; interval is centre +/- one threshold",
      "d", True, True, ["ConformalNaiveModel.predict"]),
    S("darts 0.46.1", "darts/models/forecasting/conformal_models.py", 1838,
      "ConformalQRModel._calibrate_interval",
      "residuals_, q=self.interval_range_sym, method=\"higher\", axis=2",
      "same rule over incs_qr non-conformity scores",
      "d", True, True, ["ConformalQRModel.predict"]),

    # -------------------------------------------------------- neuralforecast --
    S("neuralforecast 3.2.1", "neuralforecast/utils.py", 559,
      "add_conformal_distribution_intervals",
      "scores_quantiles = np.quantile(",
      "uncorrected cuts over vstack([mean-scores, mean+scores]); default "
      "n_windows=2 makes the stock calibration set four rows",
      "d", True, True, ["NeuralForecast.predict(prediction_intervals=PredictionIntervals())"]),
    S("neuralforecast 3.2.1", "neuralforecast/utils.py", 620,
      "add_conformal_error_intervals",
      "scores_quantiles = np.quantile(",
      "uncorrected cuts over n_windows rows, applied as mean +/-",
      "d", True, True, ["NeuralForecast.predict(prediction_intervals=PredictionIntervals())"]),
]

# Master-only mapie sites: documented by this audit, absent from the pinned version.
MASTER_ONLY = [
    ("mapie master", "_compute_classification_quantile", "(a) + off-by-one, PR #973"),
    ("mapie master", "QuantileRegressionScore", "(b) via cancelled halving, PR #978"),
]

# ---------------------------------------------------------------------------
# How each P2 site reaches the conformance table, declared per site.
#
# The census counts 35 resolution sites and Table "conformance" carries 14 rows.
# Those are counts of different populations -- a row is a call path the suite can
# construct, and one row can stand for several sites that call the same
# expression -- and until this block existed nothing in either paper said so. A
# reader who put "8 of 14 call paths" beside a 35-site census had no way to close
# the gap.
#
#   driven    the suite builds this library's helper and calls it; the row is
#             named here so the join is checkable against the printed table
#   shared    the site hands its level straight to numpy, uncorrected, and the
#             table lists that one expression once instead of repeating it under
#             every caller that reaches it. Asserted to be branch (d) below: the
#             claim is only that the site resolves through the expression the row
#             executes, which is what branch (d) means
#   pvalue    a p-value denominator. Nothing here is an order statistic, so the
#             suite has no rank to extract and no coverage figure to print.
#             Asserted to be branch e/f
#   absent    not in the conformance table. Says what this suite reaches, nothing
#             about the site itself -- no per-site explanation is offered, since
#             an unmeasured explanation is what got retracted here twice
# ---------------------------------------------------------------------------
# The rows of the conformance table, read off its committed outputs rather than
# retyped. A 'driven' site must name one of these, so the join counts rows and the
# 35-sites-to-14-rows arithmetic closes instead of stopping at a four-way split.
def _conformance_rows():
    here = os.path.dirname(os.path.abspath(__file__))
    rows = set()
    for env in ("forecasting", "tabular"):
        path = os.path.join(here, "..", "outputs",
                            f"probe_output_conformance_{env}.txt")
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                m = re.match(r"^(\S.*?)\s{2,}(?:a|b|c|d|e/f|f/e|g|count)\s+"
                             r"(?:raises|\+inf|[\d.]+)\s", ln)
                if m and not m.group(1).startswith("reference"):
                    rows.add(m.group(1).strip())
    assert len(rows) >= 12, (
        f"parsed {len(rows)} conformance rows; the table has more than that, so "
        f"the row parser is dropping rows and every join below is understated")
    return rows


CONFORMANCE_ROWS = _conformance_rows()

DISPOSITION_LABELS = [
    ("driven", "the suite constructs this helper and calls it"),
    ("shared", "resolves through a numpy quantile expression the table carries once"),
    ("pvalue", "a p-value denominator: no threshold, so no rank to report"),
    ("absent", "not in the conformance table; the suite does not reach it"),
]

SUITE_DISPOSITION = {
    # ---- mapie -----------------------------------------------------------
    "BaseConformityScore.get_quantile": ("driven", "mapie get_quantile"),
    # Same row as the LAC prefit/mean site below: that row delegates here, so one
    # row executes two sites. Naming the row rather than the delegation is what
    # lets the join count ROWS and reconcile 35 sites against 14 table rows.
    "_compute_quantiles":
        ("driven", "mapie LAC quantiles [prefit/mean -> delegates]"),
    "LACConformityScore.get_conformity_score_quantiles [prefit/mean]":
        ("driven", "mapie LAC quantiles [prefit/mean -> delegates]"),
    "LACConformityScore.get_prediction_sets [cv/crossval]":
        ("driven", "mapie LAC quantiles [cv=5/crossval -> count scale]"),
    "BaseRegressionScore._beta_optimize":
        ("driven", "mapie _beta_optimize + get_quantile (composed)"),
    "_MapieQuantileRegressor.predict": ("absent", ""),
    # ---- crepes ----------------------------------------------------------
    "ConformalClassifier.predict_set": ("pvalue", ""),
    "p_values_batch": ("pvalue", ""),
    "ConformalRegressor.predict_int": ("driven", "crepes ConformalRegressor.predict_int"),
    "ConformalRegressor.predict_int_online": ("absent", ""),
    "ConformalPredictiveSystem.predict_int":
        ("driven", "crepes ConformalPredictiveSystem.predict_int"),
    "ConformalPredictiveSystem.predict_int_online": ("absent", ""),
    # ---- puncc -----------------------------------------------------------
    "BaseCalibrator.compute_quantile": ("driven", "puncc BaseCalibrator.compute_quantile"),
    "ClasswiseCalibrator.compute_quantile": ("absent", ""),
    "quantile": ("driven", "puncc api/utils.py quantile (shared utility)"),
    "EnbPI.predict": ("absent", ""),
    "EnbPI.predict (online update)": ("absent", ""),
    # ---- torchcp ---------------------------------------------------------
    "calculate_conformal_value": ("driven", "torchcp calculate_conformal_value"),
    # ---- nonconformist ---------------------------------------------------
    "IcpClassifier.predict": ("pvalue", ""),
    "AbsErrorErrFunc.apply_inverse": ("driven", "nonconformist AbsErrorErrFunc.apply_inverse"),
    "SignErrorErrFunc.apply_inverse": ("absent", ""),
    # ---- sktime ----------------------------------------------------------
    "ConformalIntervals._predict_interval [method='empirical']":
        ("shared", "numpy method='linear'"),
    "ConformalIntervals._predict_interval [method='empirical_residual']":
        ("shared", "numpy method='linear'"),
    "ConformalIntervals._predict_interval [method='conformal_bonferroni']":
        ("shared", "numpy method='linear'"),
    "ConformalIntervals._predict_interval [method='conformal']":
        ("shared", "numpy method='linear'"),
    "EnbPI.conformal_interval": ("shared", "numpy method='linear'"),
    # ---- river -----------------------------------------------------------
    "RegressionJackknife.predict_one": ("absent", ""),
    # ---- statsforecast ---------------------------------------------------
    "_add_conformal_distribution_intervals": ("shared", "numpy method='linear'"),
    "_add_conformal_error_intervals": ("shared", "numpy method='linear'"),
    "ConformalSeasonalPool._oriented_index":
        ("driven", "statsforecast _oriented_index"),
    "ConformalSeasonalPool._intervals_from_samples": ("absent", ""),
    # ---- darts -----------------------------------------------------------
    "ConformalNaiveModel._calibrate_interval": ("shared", "numpy method='higher'"),
    "ConformalQRModel._calibrate_interval": ("shared", "numpy method='higher'"),
    # ---- neuralforecast --------------------------------------------------
    "add_conformal_distribution_intervals": ("shared", "numpy method='linear'"),
    "add_conformal_error_intervals": ("shared", "numpy method='linear'"),
}

DIST_DIRS = {  # cp-src directory name -> path prefix inside it, when they differ
    "puncc": "",
    "nonconformist": "",
}

TOP_MODULE = {
    "mapie": "mapie", "crepes": "crepes", "puncc": "deel.puncc",
    "torchcp": "torchcp", "nonconformist": "nonconformist", "sktime": "sktime",
    "river": "river", "statsforecast": "statsforecast", "darts": "darts",
    "neuralforecast": "neuralforecast",
}


def self_check():
    """Assertions on the manifest itself, before anything touches the disk."""
    keys = [(s["path"], s["line"]) for s in MANIFEST]
    assert len(keys) == len(set(keys)), "duplicate site in the manifest"
    for s in MANIFEST:
        assert s["branch"] in {"a", "b", "c", "d", "e/f", "f/e", "g", "c-exact",
                               "count", "?"}, s
        assert s["anchor"], s
        # a site that determines output must name at least one entry point,
        # and one that does not must name none -- that is what makes P3 a count
        # of the user-facing surface rather than a restatement of P2
        assert bool(s["entries"]) == s["output"], s["symbol"]

    # ---- the join to the conformance table ---------------------------------
    # Every P2 site is placed and nothing else is. A site added to the manifest
    # without a disposition fails here rather than quietly shrinking a
    # denominator the manuscript quotes, which is how a 35-versus-14 gap went
    # unstated through three revisions.
    p2_syms = [s["symbol"] for s in MANIFEST if s["output"]]
    assert len(p2_syms) == len(set(p2_syms)), "two P2 sites share a symbol"
    missing = [x for x in p2_syms if x not in SUITE_DISPOSITION]
    extra = [x for x in SUITE_DISPOSITION if x not in p2_syms]
    assert not missing, f"P2 sites with no declared disposition: {missing}"
    assert not extra, f"SUITE_DISPOSITION names non-P2 sites: {extra}"
    known = {k for k, _ in DISPOSITION_LABELS}
    for s in MANIFEST:
        if not s["output"]:
            continue
        disp, detail = SUITE_DISPOSITION[s["symbol"]]
        assert disp in known, (s["symbol"], disp)
        # The two substantive dispositions are claims about the site, so each is
        # tied to the branch letter the census already recorded. 'shared' says the
        # site resolves through the bare quantile expression, which is what branch
        # (d) means; 'pvalue' says there is no threshold, which is branch e/f.
        # Relabel a site without changing its branch and this fires.
        if disp == "shared":
            assert s["branch"] == "d", (
                f"{s['symbol']} is declared to resolve through the shared numpy "
                f"expression but its branch is ({s['branch']}), not (d) -- a site "
                f"that corrects its level does not share that row")
            assert detail, f"{s['symbol']}: 'shared' must name the row"
        if disp == "pvalue":
            assert s["branch"] in {"e/f", "f/e"}, (
                f"{s['symbol']} is declared a p-value path but its branch is "
                f"({s['branch']})")
        if disp == "driven":
            assert detail, f"{s['symbol']}: 'driven' must name its table row"
            assert detail in CONFORMANCE_ROWS, (
                f"{s['symbol']} names the row {detail!r}, which is not a row of "
                f"the conformance table. A 'driven' detail that is a sentence "
                f"rather than a row name satisfies a non-empty check and still "
                f"leaves the site/row join uncountable")
    libs = {s["lib"].split()[0] for s in MANIFEST}
    assert libs == set(TOP_MODULE), libs ^ set(TOP_MODULE)


self_check()


def resolve(site, root):
    """Locate the file for a site, by source tree first and import second."""
    lib = site["lib"].split()[0]
    rel = site["path"]
    if root:
        hits = glob.glob(os.path.join(root, lib, "**", rel), recursive=True)
        hits += glob.glob(os.path.join(root, lib, rel))
        if hits:
            return sorted(hits, key=len)[0], "source tree"
    spec = importlib.util.find_spec(TOP_MODULE[lib].split(".")[0])
    if spec and spec.submodule_search_locations:
        base = os.path.dirname(list(spec.submodule_search_locations)[0])
        cand = os.path.join(base, rel)
        if os.path.exists(cand):
            return cand, "installed"
    return None, "NOT FOUND"


def check(site, root):
    """Verify the anchor is where the manifest says. Returns (state, detail)."""
    path, how = resolve(site, root)
    if not path:
        return "MISSING", "no source found for this package"
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    want, at = site["anchor"], site["line"]
    if 1 <= at <= len(lines) and want in lines[at - 1]:
        return "OK", how
    for off in range(1, WINDOW + 1):
        for cand in (at - 1 - off, at - 1 + off):
            if 0 <= cand < len(lines) and want in lines[cand]:
                return "MOVED", f"{how}, now at line {cand + 1}"
    found = [i + 1 for i, ln in enumerate(lines) if want in ln]
    return "GONE", f"{how}, anchor not within +/-{WINDOW} lines" + (
        f" (present at {found})" if found else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../cp-src" if os.path.isdir("../cp-src") else None,
                    help="directory of unpacked package sources, one dir per distribution")
    ap.add_argument("--out", default="outputs/probe_output_helper_census.txt")
    args = ap.parse_args()

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 108)
    say("PER-HELPER CENSUS -- the unit stated first, every site verified on disk")
    say("=" * 108)
    say("self_check() passed at import (manifest is internally consistent)")
    say(f"source root: {args.root or '<none: using installed packages>'}")
    say("")
    say("UNIT -- a resolution site: a single expression converting a nominated level")
    say("into a position within the calibration score array, or into a p-value")
    say("denominator, and whose output controls a delivered bound, region, set or p-value.")
    say("")

    states = {}
    for s in MANIFEST:
        states[id(s)] = check(s, args.root)

    say(f"{'library':<21} {'site':<52} {'br':<8} {'map':<4} {'out':<4} verify")
    say("-" * 108)
    for s in MANIFEST:
        state, detail = states[id(s)]
        say(f"{s['lib']:<21} {s['symbol'][:52]:<52} {s['branch']:<8} "
            f"{'yes' if s['in_map'] else '-':<4} {'yes' if s['output'] else '-':<4} "
            f"{state}" + (f" ({detail})" if state != "OK" else ""))
    say("")

    for s in MANIFEST:
        say(f"  {s['lib']}  {s['symbol']}")
        say(f"      {s['path']}:{s['line']}")
        say(f"      rule: {s['rule']}")
    say("")

    out_sites = [s for s in MANIFEST if s["output"]]
    p1 = [s for s in MANIFEST if s["in_map"]]
    p1_out = [s for s in p1 if s["output"]]
    p2 = out_sites
    p3 = sorted({e for s in p2 for e in s["entries"]})
    excluded = [s for s in MANIFEST if not s["output"]]

    say("=" * 108)
    say("COUNTS")
    say("=" * 108)
    say(f"  P1  sites documented before this census                       "
        f"{len(p1):>3}   ({len(p1_out)} of them determine a returned bound)")
    say(f"  P2  sites determining a returned bound, pinned versions       {len(p2):>3}"
        f"   <-- USE THIS ONE, and say so")
    say(f"  P3  public API entry points routing through a P2 site         {len(p3):>3}")
    say("")
    say(f"  new in this census, absent from the map: "
        f"{len([s for s in p2 if not s['in_map']])} of the {len(p2)} P2 sites")
    say(f"  unclassified branch ('?') and therefore open audit work: "
        f"{len([s for s in p2 if s['branch'] == '?'])}")
    say("")
    say("  P2 by library:")
    for lib in sorted({s["lib"] for s in p2}):
        n = len([s for s in p2 if s["lib"] == lib])
        new = len([s for s in p2 if s["lib"] == lib and not s["in_map"]])
        say(f"      {lib:<24} {n:>2}" + (f"   ({new} new)" if new else ""))
    say("")
    say("  " + "-" * 104)
    say("  HOW THE P2 SITES REACH THE CONFORMANCE TABLE")
    say("  " + "-" * 104)
    say("  Sites and table rows are two different tallies, so the mapping between")
    say("  them is printed instead of left to a reader. Each row is a call path")
    say("  the suite could build. A site gets to one by three routes below, or by")
    say("  none.")
    say("")
    for key, label in DISPOSITION_LABELS:
        members = [s for s in p2 if SUITE_DISPOSITION[s["symbol"]][0] == key]
        say(f"  {key:<9} {len(members):>2}   {label}")
    say("")
    for key, _ in DISPOSITION_LABELS:
        for s in p2:
            disp, detail = SUITE_DISPOSITION[s["symbol"]]
            if disp == key:
                say(f"      {key:<9} {s['lib'].split()[0]:<14} "
                    f"{s['symbol'][:44]:<44} {detail}")
    say("")
    driven_rows = sorted({SUITE_DISPOSITION[s["symbol"]][1] for s in p2
                          if SUITE_DISPOSITION[s["symbol"]][0] == "driven"})
    shared_rows = sorted({SUITE_DISPOSITION[s["symbol"]][1] for s in p2
                          if SUITE_DISPOSITION[s["symbol"]][0] == "shared"})
    n_driven = len([s for s in p2 if SUITE_DISPOSITION[s["symbol"]][0] == "driven"])
    n_shared = len([s for s in p2 if SUITE_DISPOSITION[s["symbol"]][0] == "shared"])
    say("  and the arithmetic closes, which a four-way split of the sites does not:")
    say(f"      {n_driven:>2} driven sites  ->  {len(driven_rows):>2} table row(s)")
    say(f"      {n_shared:>2} shared sites  ->  {len(shared_rows):>2} table row(s)")
    say(f"                        ->  {len(driven_rows) + len(shared_rows):>2} "
        f"distinct resolution sites in the table")
    extra = len(CONFORMANCE_ROWS) - (len(driven_rows) + len(shared_rows))
    say(f"                        +   {extra:>1} second path on one of them "
        f"(a public keyword)")
    say(f"                        ->  {len(CONFORMANCE_ROWS):>2} call paths, "
        f"which is what the table's rows count")
    say("")
    say(f"  So a proportion over {len(CONFORMANCE_ROWS)} call paths is not a proportion over")
    say(f"  {len(p2)} sites and not a proportion of the ecosystem. The manuscript must say")
    say("  which, and the two rows below are the ones that stand for more than one site.")
    for row in shared_rows:
        n_here = len([s for s in p2
                      if SUITE_DISPOSITION[s["symbol"]] == ("shared", row)])
        say(f"      {row:<58} {n_here:>2} site(s)")
    say("")
    say(f"  excluded from P2, listed so the exclusion is auditable ({len(excluded)}):")
    for s in excluded:
        say(f"      {s['lib']:<21} {s['symbol']:<48} {s['rule'].split(';')[0][:60]}")
    say("")
    say(f"  master-only, documented but absent from the pinned versions "
        f"({len(MASTER_ONLY)}):")
    for lib, sym, note in MASTER_ONLY:
        say(f"      {lib:<21} {sym:<48} {note}")
    say("  and the exclusion is verified, not assumed -- grepped over the pinned mapie:")
    for _, sym, _ in MASTER_ONLY:
        hits = []
        for path in glob.glob(os.path.join(args.root or "", "mapie", "**", "*.py"),
                              recursive=True) if args.root else []:
            with open(path, encoding="utf-8", errors="replace") as fh:
                if sym in fh.read():
                    hits.append(path)
        state = "absent" if not hits else f"PRESENT in {hits[:3]} -- reclassify it"
        say(f"      {sym:<48} {state}")
    say("")
    say("  P3 entry points:")
    for e in p3:
        say(f"      {e}")

    bad = [(s, st, d) for s in MANIFEST for st, d in [states[id(s)]] if st != "OK"]
    say("")
    if bad:
        say(f"!! {len(bad)} site(s) did not verify at the recorded line:")
        for s, st, d in bad:
            say(f"     {st:<8} {s['lib']} {s['symbol']} -- {s['path']}:{s['line']} ({d})")
        say("   A MOVED site is a line-number fix. A GONE or MISSING site means the")
        say("   count above is stale and must be re-derived before it is quoted.")
    else:
        say("all sites verified at the recorded line")

    say("")
    say("What replaces \"seven have a distinct quantile/p-value helper\"")
    say("-" * 108)
    say(f"  \"Across the ten packages that derive a coverage guarantee from a calibration")
    say(f"  array, we count {len(p2)} distinct resolution sites -- expressions mapping a")
    say(f"  requested level to an index into the calibration set -- reachable through {len(p3)} public")
    say(f"  API entry points.\"")
    say("")
    say(f"  {len(p1_out)} of the {len(p2)} were already documented by this audit; the other")
    say(f"  {len(p2) - len(p1_out)} this census located. \"Seven\" understated the surface by "
        f"{len(p2) / 7:.1f}x.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {args.out}")
    return 1 if any(st in ("GONE", "MISSING") for st, _ in states.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
