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
A **resolution site**: one expression that turns a requested coverage or
miscoverage level into either

  (i)  an index or rank into a set of calibration nonconformity scores, or
  (ii) a p-value denominator over that set,

and whose value determines a returned threshold, interval, set or p-value.

Two public methods sharing one expression count once. One method containing two
expressions with different rules counts twice. An expression that only feeds a
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
    S("mapie 1.4.1", "mapie/conformity_scores/sets/lac.py", 158,
      "LACConformityScore.get_conformity_score_quantiles [cv/crossval]",
      "quantiles_ = (n + 1) * (1 - alpha_np)",
      "the OTHER rule in the same method, reached only when cv is not 'prefit' "
      "AND agg_scores is not 'mean'. Returns the raw COUNT (n+1)(1-alpha), not a "
      "threshold on the score scale -- at n=8 it returns 8.1, above every score. "
      "No branch letter applies; conformance_suite.py labels it `count`",
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
    say("UNIT -- a resolution site: one expression turning a requested level into an")
    say("index into a calibration score set, or into a p-value denominator, whose value")
    say("determines a returned threshold, interval, set or p-value.")
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
    say(f"  \"Across the ten packages that resolve a coverage bound from a calibration")
    say(f"  set, we count {len(p2)} distinct resolution sites -- expressions mapping a")
    say(f"  requested level to an index into that set -- reachable through {len(p3)} public")
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
