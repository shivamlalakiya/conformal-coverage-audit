#!/usr/bin/env python3
"""Does a CALLER have a public route to the below-floor behaviour, per row?

Why this probe exists
---------------------
`conformance_suite.py` classifies fourteen shipped call paths and reports how many
return a finite number in silence below the feasibility floor. Every one of those
rows is a helper INVOCATION: the suite reaches the expression the way the library
reaches it internally. Tallying invocations answers a different question from
tallying the entry points a user is able to call, and a maintainer refereeing this
work will ask the second question first. The distinction decides whether a row is a
defect a user can hit or an internal expression that a guard above it protects.

So this probe asks a different question of the same fourteen rows: starting from
the shallowest entry point the library documents, at a calibration size below the
floor, what does the caller get? Four answers are possible and all four are
reported as found:

  raises   the entry point refuses. There is no public route to the behaviour and
           the guard sits above the helper.
  +inf     the entry point returns an infinite bound, which is the honest answer
           at an infeasible size.
  finite   the entry point returns a finite number where no finite bound is valid.
  none     no documented entry point reaches the helper at all.

Each of the first three is recorded with whether a warning was emitted, because a
finite number WITH a warning is a different defect from a finite number in silence.

What this probe claims, and what it does not
--------------------------------------------
It claims that the named entry point, called as written here, produced the recorded
outcome at the pinned version. It does NOT claim the entry point is the only public
route, and a `raises` verdict is therefore evidence about the routes tried and not a
proof of unreachability -- which is why the routes tried are printed per row rather
than summarised. Where a library offers several high-level predictors, more than one
is called and all of them are printed.

Two rows are CITED rather than executed, and named as citations. `numpy` is not a
conformal library and its two rows carry their consumers in their labels; the
consumers' public routes were executed in the paired real-data arms, below the floor,
and those outputs are committed. Re-executing them here would measure the same call a
second time and let the two answers drift.

    python route_reachability.py --arm tabular       # crepes, puncc, torchcp,
                                                     # nonconformist, mapie
    python route_reachability.py --arm forecasting   # statsforecast
"""

import argparse
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LEVEL = 0.90
RAISED = "raises"
NCAL = 8            # the floor at 0.90 is n >= 9, so 8 is infeasible
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def feasibility_floor(level):
    """Smallest n at which ceil((n+1)*level) <= n, by search rather than formula."""
    n = 1
    while math.ceil((n + 1) * level) > n:
        n += 1
        assert n < 10000, level
    return n


def outcome(fn):
    """(verdict, warned, value) for one public call at an infeasible size."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            v = fn()
            arr = np.asarray(v, dtype=float)
            finite = bool(np.all(np.isfinite(arr)))
            return ("finite" if finite else "+inf", bool(caught),
                    float(arr.ravel()[0]) if arr.size else float("nan"))
        except Exception as exc:
            # A warning emitted before the raise still counts as a warning: the
            # caller saw it. torchcp warns that it is setting the threshold to
            # +inf and then fails constructing an interval out of it.
            msg = " ".join(str(exc).split())
            return (RAISED, bool(caught), f"{type(exc).__name__}: {msg[:70]}")


def self_check():
    """The classifier must tell the three outcomes apart, and warnings from silence."""
    assert feasibility_floor(0.90) == 9, feasibility_floor(0.90)
    assert feasibility_floor(0.95) == 19, feasibility_floor(0.95)
    v, w, _ = outcome(lambda: 1.0)
    assert (v, w) == ("finite", False), (v, w)
    v, w, _ = outcome(lambda: float("inf"))
    assert (v, w) == ("+inf", False), (v, w)

    def raiser():
        raise ValueError("no")
    v, w, _ = outcome(raiser)
    assert (v, w) == (RAISED, False), (v, w)

    def warner():
        warnings.warn("noisy")
        return 1.0
    v, w, _ = outcome(warner)
    assert (v, w) == ("finite", True), (v, w)


def _tabular_fixture():
    from sklearn.linear_model import LinearRegression, LogisticRegression
    rng = np.random.default_rng(7)
    coef = np.array([1.0, -2.0, 0.5])
    Xtr = rng.normal(size=(200, 3))
    ytr = Xtr @ coef + rng.normal(size=200)
    Xc = rng.normal(size=(NCAL, 3))
    yc = Xc @ coef + rng.normal(size=NCAL)
    Xt = rng.normal(size=(3, 3))
    ctr = (Xtr[:, 0] > 0).astype(int)
    cc = (Xc[:, 0] > 0).astype(int)
    return LinearRegression, LogisticRegression, Xtr, ytr, Xc, yc, Xt, ctr, cc


def tabular_routes():
    """(helper row, entry point, callable) for every route importable here."""
    LR, LogR, Xtr, ytr, Xc, yc, Xt, ctr, cc = _tabular_fixture()
    out, skipped = [], []

    try:
        from crepes import WrapRegressor

        def crepes_wrap():
            w = WrapRegressor(LR())
            w.fit(Xtr, ytr)
            w.calibrate(Xc, yc)
            return w.predict_int(Xt, confidence=LEVEL)

        def crepes_wrap_cps():
            w = WrapRegressor(LR())
            w.fit(Xtr, ytr)
            w.calibrate(Xc, yc, cps=True)
            return w.predict_int(Xt, confidence=LEVEL)

        out.append(("crepes ConformalRegressor.predict_int",
                    "crepes.WrapRegressor.predict_int(confidence=0.90)",
                    crepes_wrap))
        out.append(("crepes ConformalPredictiveSystem.predict_int",
                    "crepes.WrapRegressor.calibrate(cps=True).predict_int",
                    crepes_wrap_cps))
    except Exception as exc:
        skipped.append(("crepes WrapRegressor", type(exc).__name__))

    try:
        from nonconformist.icp import IcpRegressor
        from nonconformist.nc import NcFactory

        def nc_icp():
            m = LR().fit(Xtr, ytr)
            icp = IcpRegressor(NcFactory.create_nc(m))
            icp.fit(Xtr, ytr)
            icp.calibrate(Xc, yc)
            return icp.predict(Xt, significance=1 - LEVEL)

        out.append(("nonconformist AbsErrorErrFunc.apply_inverse",
                    "nonconformist.icp.IcpRegressor.predict(significance=0.10)",
                    nc_icp))
    except Exception as exc:
        skipped.append(("nonconformist IcpRegressor", type(exc).__name__))

    try:
        from deel.puncc.api.prediction import BasePredictor, MeanVarPredictor
        from deel.puncc.regression import CVPlus, LocallyAdaptiveCP, SplitCP

        def puncc_split():
            m = LR().fit(Xtr, ytr)
            cp = SplitCP(BasePredictor(m, is_trained=True), train=False)
            cp.fit(X_calib=Xc, y_calib=yc)
            return cp.predict(Xt, alpha=1 - LEVEL)[1]

        def puncc_lacp():
            mv = MeanVarPredictor([LR(), LR()])
            cp = LocallyAdaptiveCP(mv, train=True)
            cp.fit(X_fit=Xtr, y_fit=ytr, X_calib=Xc, y_calib=yc)
            return cp.predict(Xt, alpha=1 - LEVEL)[1]

        def puncc_cvplus():
            cp = CVPlus(BasePredictor(LR()), K=4, random_state=0)
            cp.fit(X=Xc, y=yc)
            return cp.predict(Xt, alpha=1 - LEVEL)[1]

        # Both puncc rows sit under the same three high-level predictors: the
        # api/utils.py quantile is what BaseCalibrator calls, and BaseCalibrator
        # is what each of these three calls. One route each, three tried.
        for row in ("puncc BaseCalibrator.compute_quantile",
                    "puncc api/utils.py quantile (shared utility)"):
            out.append((row, "deel.puncc.regression.SplitCP.predict(alpha=0.10)",
                        puncc_split))
            out.append((row, "deel.puncc.regression.LocallyAdaptiveCP.predict",
                        puncc_lacp))
            out.append((row, "deel.puncc.regression.CVPlus.predict", puncc_cvplus))
    except Exception as exc:
        skipped.append(("puncc high-level predictors", type(exc).__name__))

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from torchcp.regression.predictor import SplitPredictor
        from torchcp.regression.score import ABS

        def torchcp_split():
            model = torch.nn.Linear(3, 1)
            cal = DataLoader(
                TensorDataset(torch.tensor(Xc, dtype=torch.float32),
                              torch.tensor(yc, dtype=torch.float32)),
                batch_size=NCAL)
            p = SplitPredictor(score_function=ABS(), model=model)
            p.calibrate(cal, alpha=1 - LEVEL)
            # q_hat is read before predict(), because predict() is where an
            # infinite q_hat fails and the threshold is the quantity in question.
            q = float(p.q_hat)
            if not math.isfinite(q):
                return float("inf")
            return p.predict(torch.tensor(Xt, dtype=torch.float32)).numpy()

        out.append(("torchcp calculate_conformal_value",
                    "torchcp.regression.SplitPredictor.calibrate(alpha=0.10)",
                    torchcp_split))
    except Exception as exc:
        skipped.append(("torchcp SplitPredictor", type(exc).__name__))

    try:
        from mapie.classification import (CrossConformalClassifier,
                                          SplitConformalClassifier)
        from mapie.regression import SplitConformalRegressor

        def mapie_default():
            m = LR().fit(Xtr, ytr)
            r = SplitConformalRegressor(estimator=m, confidence_level=LEVEL,
                                        prefit=True)
            r.conformalize(Xc, yc)
            return r.predict_interval(Xt)[1]

        def mapie_unbounded():
            m = LR().fit(Xtr, ytr)
            r = SplitConformalRegressor(estimator=m, confidence_level=LEVEL,
                                        prefit=True)
            r.conformalize(Xc, yc)
            # the flag is a keyword on predict_interval, not on the constructor
            return r.predict_interval(Xt, allow_infinite_bounds=True)[1]

        def mapie_lac_split():
            m = LogR().fit(Xtr, ctr)
            c = SplitConformalClassifier(estimator=m, confidence_level=LEVEL,
                                         conformity_score="lac", prefit=True)
            c.conformalize(Xc, cc)
            return c.predict_set(Xt).astype(float)

        def mapie_lac_cross():
            c = CrossConformalClassifier(estimator=LogR(),
                                         confidence_level=LEVEL,
                                         conformity_score="lac", cv=5)
            c.fit_conformalize(Xc, cc)
            return c.predict_set(Xt, agg_scores="crossval").astype(float)

        def mapie_min_width():
            m = LR().fit(Xtr, ytr)
            r = SplitConformalRegressor(estimator=m, confidence_level=LEVEL,
                                        prefit=True)
            r.conformalize(Xc, yc)
            return r.predict_interval(Xt, minimize_interval_width=True)[1]

        out.append(("mapie get_quantile",
                    "mapie.regression.SplitConformalRegressor.predict_interval",
                    mapie_default))
        out.append(("mapie get_quantile [allow_infinite_bounds=True]",
                    "SplitConformalRegressor.predict_interval"
                    "(allow_infinite_bounds=True)", mapie_unbounded))
        out.append(("mapie LAC quantiles [prefit/mean -> delegates]",
                    "mapie.classification.SplitConformalClassifier.predict_set",
                    mapie_lac_split))
        out.append(("mapie LAC quantiles [cv=5/crossval -> count scale]",
                    "CrossConformalClassifier.predict_set(agg_scores='crossval')",
                    mapie_lac_cross))
        out.append(("mapie _beta_optimize + get_quantile (composed)",
                    "SplitConformalRegressor.predict_interval"
                    "(minimize_interval_width=True)", mapie_min_width))
    except Exception as exc:
        skipped.append(("mapie public predictors", type(exc).__name__))

    return out, skipped


def forecasting_routes():
    out, skipped = [], []
    try:
        from statsforecast.models import ConformalSeasonalPool as CSP

        def sf_csp():
            rng = np.random.default_rng(3)
            y = 10 + np.arange(120) * 0.1 + rng.normal(size=120)
            res = CSP(season_length=12, n_samples=NCAL).forecast(
                y=y, h=1, level=[int(LEVEL * 100)])
            return float(res[f"lo-{int(LEVEL * 100)}"][0])

        out.append(("statsforecast _oriented_index",
                    "statsforecast.models.ConformalSeasonalPool(n_samples=8)"
                    ".forecast(level=[90])", sf_csp))
    except Exception as exc:
        skipped.append(("statsforecast ConformalSeasonalPool", type(exc).__name__))
    return out, skipped


# Rows whose public route was executed in another committed output. Named, with
# the output and the cell, so a reader checks the claim where it was measured.
CITED = {
    "numpy method='linear' (sktime, statsforecast, neuralforecast)": (
        "statsforecast ConformalIntervals(n_windows=2) via "
        "StatsForecast.forecast(level=[90])",
        "finite", False,
        "probe_output_real_data_statsforecast.txt: the shipped default at two "
        "calibration windows, where the floor is nine, returns a finite interval "
        "and delivers 0.5840 against 0.90"),
    "numpy method='higher' (darts)": (
        "darts ConformalNaiveModel(cal_length=10).predict"
        "(predict_likelihood_parameters=True)",
        "finite", False,
        "probe_output_real_data_darts.txt: nominal 0.95, cal_length=10, floor 19 "
        "-- the required rank exceeds n in 250 of 250 cells and the shipped route "
        "returns a finite interval in every one, delivering 0.8960"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("tabular", "forecasting"), default="tabular")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(
        ROOT, "outputs", f"probe_output_routes_{args.arm}.txt")

    self_check()
    floor = feasibility_floor(LEVEL)
    say("=" * 100)
    say("PUBLIC ROUTES TO THE BELOW-FLOOR BEHAVIOUR, ONE ROW AT A TIME")
    say("=" * 100)
    say("self_check() passed at import: the feasibility floor is found by search,")
    say("and the outcome classifier separates finite, +inf and raises, and")
    say("separates a warning from silence.")
    say("")
    say(f"arm: {args.arm}")
    say(f"requested coverage: {LEVEL:.2f}")
    say(f"feasibility floor:  n >= {floor}; every route below is called at "
        f"n={NCAL}")
    say("")
    say("A verdict is about the routes tried, printed below it. `raises` says the")
    say("guard sits above the helper on every route tried here, not that no route")
    say("exists anywhere.")
    say("")

    routes, skipped = (tabular_routes() if args.arm == "tabular"
                       else forecasting_routes())
    seen = {}
    order = []
    for row, entry, fn in routes:
        verdict, warned, value = outcome(fn)
        seen.setdefault(row, []).append((entry, verdict, warned, value))
        if row not in order:
            order.append(row)

    if args.arm == "tabular":
        for row, (entry, verdict, warned, why) in CITED.items():
            seen.setdefault(row, []).append((entry, verdict, warned, "cited"))
            if row not in order:
                order.append(row)

    say(f"{'helper row':<52} {'route':>8} {'warns':>6}")
    say("-" * 100)
    silent_finite = []
    for row in order:
        tried = seen[row]
        # The row's verdict is the WORST outcome a caller can reach: a finite
        # number in silence is worse than a warning, which is worse than +inf,
        # which is worse than a refusal. A single reachable silent finite makes
        # the row reachable however many other entry points refuse.
        rank = {"finite": 0, "+inf": 1, RAISED: 2}
        best = sorted(tried, key=lambda t: (rank[t[1]], t[2]))[0]
        entry, verdict, warned, value = best
        say(f"{row:<52} {verdict:>8} {('yes' if warned else '-'):>6}")
        if verdict == "finite" and not warned:
            silent_finite.append(row)
    say("")
    say(f"rows with a public route returning a finite number in silence: "
        f"{len(silent_finite)} of {len(order)} on this arm")
    for row in silent_finite:
        say(f"      {row}")
    say("")
    say("Routes tried, per row")
    say("-" * 100)
    for row in order:
        say(f"  {row}")
        for entry, verdict, warned, value in seen[row]:
            note = "" if value != "cited" else "  [CITED, not executed here]"
            say(f"      {verdict:<7} {'warns' if warned else 'silent':<7} "
                f"{entry}{note}")
            if isinstance(value, str) and value not in ("cited",):
                say(f"              {value}")
        if row in CITED:
            say(f"              {CITED[row][3]}")
    if skipped:
        say("")
        say(f"routes not loaded on this arm ({len(skipped)}) -- reported, not dropped:")
        for name, why in skipped:
            say(f"      {name:<44} {why}")
        say("      run this file again with the other --arm to cover them")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
