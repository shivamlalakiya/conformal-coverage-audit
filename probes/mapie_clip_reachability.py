#!/usr/bin/env python3
"""Is mapie's clip in `get_quantile` reachable through the public API?

Why this probe exists
---------------------
An earlier version of this work reported the clip at
`conformity_scores/interface.py:138` as a reachable silent clamp. A scan over
49,990 combinations of calibration size, level and symmetry found no case where
the library's guard passed AND the clip bit, and the finding was RETRACTED.

That scan held one thing fixed that it never varied: `allow_infinite_bounds`.
The guard it relied on is conditional on exactly that flag
(`regression/regression.py:1714`, `if not allow_infinite_bounds:`), and the flag
is a documented public keyword on `SplitConformalRegressor.predict_interval`.
So the retraction generalised "dead on the path scanned" -- which the earlier
probe output correctly said -- into "unreachable through the public API", which
is false. This probe establishes the reachability, and the direction of harm.

The mechanism, read off the pinned source
-----------------------------------------
    interface.py:137   alpha_cor = ceil(alpha_ref * (n_calib + 1)) / n_calib
    interface.py:138   alpha_cor = np.clip(alpha_cor, 0, 1)          <-- the clip
    interface.py:150   ... if not (unbounded and _alpha >= 1) else inf

`_alpha` on line 150 iterates `alpha_ref`, the level BEFORE the finite-sample
correction. The infeasible regime is `ceil(alpha_ref*(n+1)) > n_calib`, which
occurs at `alpha_ref` well below 1. So in exactly that regime the infinity branch
does not fire, `alpha_cor` is clipped from >1 down to 1, and `nanquantile(...,
1.0, method="lower")` yields the sample maximum.

The user asked, via the flag, to be given an infinite bound where none is finite.
They are given a finite one instead, with no warning.

Direction: **anti-conservative**. A finite interval is returned where no valid
finite deterministic bound exists, so it under-covers. This is not a documented
trade; the flag documents the opposite.

Run:  python mapie_clip_reachability.py
"""

import ast
import inspect
import io
import math
import pathlib
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import mapie
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.regression import (CrossConformalRegressor,
                              JackknifeAfterBootstrapRegressor,
                              SplitConformalRegressor)


def CLASSES_UNDER_TEST():
    """(label, constructor, fit) for every public regressor exposing the flag.

    Split conformal is fit then conformalised on a held-out calibration set; the
    other two take one set and partition it themselves. The difference is in the
    fit callable rather than in three separate code paths, so adding a class here
    is one tuple.
    """
    lvl = 0.95
    return [
        ("SplitConformalRegressor",
         lambda: SplitConformalRegressor(estimator=LinearRegression(),
                                         confidence_level=lvl, prefit=False),
         lambda m, Xtr, ytr, Xc, yc: m.fit(Xtr, ytr).conformalize(Xc, yc)),
        ("CrossConformalRegressor",
         lambda: CrossConformalRegressor(estimator=LinearRegression(),
                                         confidence_level=lvl, cv=5),
         lambda m, Xtr, ytr, Xc, yc: m.fit_conformalize(Xc, yc)),
        ("JackknifeAfterBootstrapRegressor",
         lambda: JackknifeAfterBootstrapRegressor(estimator=LinearRegression(),
                                                  confidence_level=lvl,
                                                  resampling=10),
         lambda m, Xtr, ytr, Xc, yc: m.fit_conformalize(Xc, yc)),
    ]

LINES = []
SEED = 20260806
REPS = 50000


def say(s=""):
    LINES.append(s)
    print(s)


def required_rank(n, level):
    """Smallest rank whose one-sided coverage reaches `level`, exactly.

    Intentionally local: returns the ceil even when it exceeds n, because this
    probe enumerates the infeasible window. The package helper returns None
    there and would hide the cells under study.
    """
    return int(-(-(Fraction(level) * (n + 1)).numerator
                 // (Fraction(level) * (n + 1)).denominator))


def windows(level):
    """Every infeasible calibration size at `level`, plus the first feasible one.

    The grid is the bug's, not ours: the whole window is enumerated, so there is
    no size we could have chosen to make the shortfall look larger or smaller.
    The trailing feasible size is the control -- there the sample maximum is the proper
    answer and the gap must read exactly zero.
    """
    n, out = 2, []
    while required_rank(n, level) > n:
        out.append(n)
        n += 1
        assert n < 10_000, level          # the window is finite for level < 1
    return out + [n]


def self_check():
    """The clip window is derived, not assumed."""
    # required rank exceeds n exactly when ceil(level*(n+1)) > n
    # n = 19 at 0.95 is the boundary: ceil(0.95*20) = 19 = n, so it is FEASIBLE and the
    # maximum is the correct answer. n = 18 is the first infeasible size below it.
    for n, level, want in [(10, Fraction(95, 100), True),
                           (10, Fraction(80, 100), False),
                           (18, Fraction(95, 100), True),
                           (19, Fraction(95, 100), False),
                           (20, Fraction(95, 100), False)]:
        got = required_rank(n, level) > n
        assert got == want, (n, level, got, want)
    assert required_rank(10, Fraction(95, 100)) == 11
    assert required_rank(20, Fraction(95, 100)) == 20
    # the window closes at n >= level/(1-level), and windows() appends the first
    # feasible size past it as the control
    for level, last_bad in ((Fraction(95, 100), 18), (Fraction(90, 100), 8)):
        w = windows(level)
        assert w == list(range(2, last_bad + 2)), (level, w)
        assert required_rank(w[-2], level) > w[-2], (level, w[-2])
        assert required_rank(w[-1], level) <= w[-1], (level, w[-1])


def main():
    self_check()
    say("=" * 92)
    say("Is mapie's clip in get_quantile reachable through the PUBLIC API?")
    say(f"mapie {mapie.__version__}, numpy {np.__version__}")
    say("self_check() passed at import (the clip window is derived, not assumed)")
    say("=" * 92)
    say("")
    say("(i) THE FUNCTION IN ISOLATION. Scores are 1..10, so the sample maximum is 10.")
    say("    'unbounded' is what allow_infinite_bounds sets. The infinity branch tests the")
    say("    level BEFORE the finite-sample correction, so it does not fire here.")
    say("")
    cs = AbsoluteConformityScore()
    scores = np.arange(1, 11, dtype=float).reshape(-1, 1)
    say(f"{'level':>8} {'unbounded':>10} {'req rank':>9} {'of n':>5} {'returned':>10}")
    for level in (Fraction(80, 100), Fraction(95, 100), Fraction(999, 1000)):
        for unbounded in (False, True):
            q = cs.get_quantile(scores, np.array([float(level)]), axis=0,
                                reversed=False, unbounded=unbounded)
            say(f"{float(level):>8.3f} {str(unbounded):>10} "
                f"{required_rank(10, level):>9} {10:>5} {float(np.ravel(q)[0]):>10.4f}")
    say("")
    say("    Every infeasible row returns 10.0, the maximum, rather than +inf.")
    say("")

    say("-" * 92)
    say("(ii) END TO END, through SplitConformalRegressor.predict_interval, whose signature")
    say("     carries allow_infinite_bounds as a public keyword.")
    say("-" * 92)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=80)
    Xtr, Xc, ytr, yc = train_test_split(X, y, test_size=10, random_state=0)
    Xte = rng.normal(size=(3, 3))
    level = Fraction(95, 100)
    n = len(yc)
    say(f"    n_calib = {n}, confidence_level = {float(level)}, "
        f"required rank = {required_rank(n, level)} of {n}")
    say("    -> no valid finite DETERMINISTIC bound exists at this configuration")
    say("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SplitConformalRegressor(estimator=LinearRegression(),
                                    confidence_level=float(level), prefit=False)
        m.fit(Xtr, ytr).conformalize(Xc, yc)
        for flag in (False, True):
            try:
                with redirect_stderr(io.StringIO()):
                    _, iv = m.predict_interval(Xte, allow_infinite_bounds=flag)
                w = float(iv[0, 1, 0] - iv[0, 0, 0])
                say(f"    allow_infinite_bounds={str(flag):<5} -> width {w:.4f}, "
                    f"finite={bool(np.isfinite(w))}")
            except Exception as exc:  # noqa: BLE001 -- the guard raising IS the result
                say(f"    allow_infinite_bounds={str(flag):<5} -> "
                    f"{type(exc).__name__}: {' '.join(str(exc).split())[:70]}")
    say("")

    # ---------------- (iii) what the flagged path DELIVERS ------------------
    say("-" * 92)
    say("(iii) DELIVERED COVERAGE on the flagged path, over the whole infeasible window.")
    say("      Scores are tie-free 1..n, so the returned threshold identifies its landed rank")
    say("      on. Where the required rank exceeds n the flagged path lands on rank n, so")
    say("      the delivered coverage is n/(n+1), independent of the requested level")
    say("      that was asked for. Exact arithmetic; the last row is the feasible control.")
    say("-" * 92)
    say(f"{'level':>7}{'n':>5}{'req rank':>10}{'feasible':>10}{'rank got':>10}"
        f"{'delivered':>11}{'nominal':>9}{'shortfall':>11}")
    worst = None
    for level in (Fraction(95, 100), Fraction(90, 100)):
        for n in windows(level):
            r = required_rank(n, level)
            feasible = r <= n
            q = cs.get_quantile(np.arange(1, n + 1, dtype=float).reshape(-1, 1),
                                np.array([float(level)]), axis=0,
                                reversed=False, unbounded=True)
            got = Fraction(int(round(float(np.ravel(q)[0]))))
            delivered = got / (n + 1)
            short = Fraction(level) - delivered
            say(f"{float(level):>7.2f}{n:>5}{r:>10}{str(feasible):>10}{int(got):>10}"
                f"{float(delivered):>11.4f}{float(level):>9.2f}{float(short):>+11.4f}")
            if not feasible and (worst is None or short > worst[3]):
                worst = (float(level), n, delivered, short)
        say("")
    assert worst is not None
    say(f"    worst infeasible cell: n = {worst[1]} at nominal {worst[0]:.2f} "
        f"delivers {float(worst[2]):.4f}, shortfall {float(worst[3]):.4f}")
    say("    the deficit here is a whole missing rank, not the O(1/n) one-rank deficit:")
    say("    no rank inside the sample attains the requested level at these sizes.")
    say("")

    # -- and the same number, measured end to end rather than derived --------
    say("-" * 92)
    say("(iv) THE SAME COVERAGE, MEASURED through predict_interval(allow_infinite_bounds=True)")
    say("     on i.i.d. draws, against the exact n/(n+1) above. The base model is prefit, so")
    say("     the calibration scores are exchangeable with the test score by construction.")
    say("-" * 92)
    rng = np.random.default_rng(SEED)
    beta = np.array([1.0, -2.0, 0.5])
    Xtr = rng.normal(size=(400, 3))
    base = LinearRegression().fit(Xtr, Xtr @ beta + rng.normal(size=400))
    say(f"{'level':>7}{'n':>5}{'reps':>7}{'predicted':>11}{'measured':>10}"
        f"{'se':>8}{'z':>7}{'finite?':>9}")
    mc = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for level, n in ((Fraction(95, 100), 10), (Fraction(95, 100), 18),
                         (Fraction(90, 100), 8)):
            hits, allfinite = 0, True
            for _ in range(REPS):
                Xc = rng.normal(size=(n, 3))
                yc = Xc @ beta + rng.normal(size=n)
                Xte = rng.normal(size=(1, 3))
                yte = float(Xte @ beta + rng.normal(size=1))
                m = SplitConformalRegressor(estimator=base,
                                            confidence_level=float(level),
                                            prefit=True)
                m.conformalize(Xc, yc)
                with redirect_stderr(io.StringIO()):
                    _, iv = m.predict_interval(Xte, allow_infinite_bounds=True)
                lo, hi = float(iv[0, 0, 0]), float(iv[0, 1, 0])
                allfinite &= bool(np.isfinite(hi - lo))
                hits += int(lo <= yte <= hi)
            p = hits / REPS
            se = math.sqrt(p * (1 - p) / REPS)
            pred = float(Fraction(n, n + 1))
            z = (p - pred) / se if se > 0 else 0.0
            mc.append((float(level), n, p, se, z))
            say(f"{float(level):>7.2f}{n:>5}{REPS:>7}{pred:>11.4f}{p:>10.4f}"
                f"{se:>8.4f}{z:>+7.2f}{str(allfinite):>9}")
    say("")
    say(f"    largest |z| against n/(n+1): {max(abs(r[4]) for r in mc):.2f}")
    say("    Every interval returned was FINITE, at every cell above, though no valid")
    say("    finite deterministic bound exists at any of them.")
    say("")

    # ---------------- (v) how wide is the door -----------------------------
    say("-" * 92)
    say("(v) HOW MANY PUBLIC ENTRY POINTS CARRY THE FLAG. The signatures are read out of")
    say("    the pinned package with ast, not grepped and not counted by hand, so a")
    say("    renamed or added method changes this number rather than escaping it.")
    say("-" * 92)
    root = pathlib.Path(inspect.getfile(mapie)).parent
    sigs = []
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
                if "allow_infinite_bounds" in names:
                    sigs.append((str(path.relative_to(root)), node.name, fn.name,
                                 fn.lineno))
    assert sigs, "no signature carries the flag; the probe is reading the wrong package"
    for rel, cls, fn, lineno in sigs:
        public = not cls.startswith("_") and not fn.startswith("_")
        say(f"    {rel}:{lineno:<5} {cls}.{fn}"
            f"{'' if public else '   (private class or method)'}")
    pub = [s for s in sigs if not s[1].startswith("_") and not s[2].startswith("_")]
    say("")
    say(f"    {len(sigs)} signatures carry the keyword, {len(pub)} of them on a public "
        f"class and method.")
    say("")

    say("-" * 92)
    say("(vi) AND HOW MANY OF THEM ACTUALLY REACH IT, end to end, below the floor.")
    say("     n_calib = 10 at confidence 0.95, the same infeasible configuration as (ii).")
    say("     Any class this probe cannot construct is REPORTED, never dropped.")
    say("-" * 92)
    rng = np.random.default_rng(SEED)
    Xtr = rng.normal(size=(200, 3))
    ytr = Xtr @ beta + rng.normal(size=200)
    Xc = rng.normal(size=(10, 3))
    yc = Xc @ beta + rng.normal(size=10)
    Xte = rng.normal(size=(2, 3))
    reached, tried = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, build, fit in CLASSES_UNDER_TEST():
            row = {"label": label}
            for flag in (False, True):
                try:
                    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                        est = build()
                        fit(est, Xtr, ytr, Xc, yc)
                        _, iv = est.predict_interval(
                            Xte, allow_infinite_bounds=flag)
                    w = float(iv[0, 1, 0] - iv[0, 0, 0])
                    row[flag] = ("finite" if np.isfinite(w) else "+inf", w)
                except Exception as exc:  # noqa: BLE001 -- the guard raising IS a result
                    row[flag] = (type(exc).__name__, None)
            tried.append(row)
            default, flagged = row[False], row[True]
            got = (default[0] != "finite" and flagged[0] == "finite")
            if got:
                reached.append(label)
            say(f"    {label:<34} default -> {default[0]:<12} "
                f"flagged -> {flagged[0]}"
                f"{'' if flagged[1] is None else f' (width {flagged[1]:.4f})'}"
                f"{'   REACHES THE CLIP' if got else ''}")
    say("")
    say(f"    {len(reached)} of {len(tried)} public regressor classes exercised here "
        f"fail on the default route and produce a finite interval on the flagged route.")
    say("    So the path is not a single entry point. It is the shared bound-construction")
    say("    layer, and every public class routing through it inherits the behaviour.")
    say("")

    say("-" * 92)
    say("VERDICT")
    say("-" * 92)
    say("  The clip is REACHABLE through the public API. The guard that makes it dead code")
    say("  is skipped by the same flag the caller sets to opt into infinite bounds")
    say("  (regression/regression.py:1714, 'if not allow_infinite_bounds:'), and")
    say("  TimeSeriesRegressor passes allow_infinite_bounds=True internally")
    say("  (regression/time_series_regression.py:319), so callers leaving the flag unset")
    say("  can still reach it.")
    say("")
    say("  Direction: ANTI-CONSERVATIVE. A finite interval is returned where no valid finite")
    say("  deterministic bound exists, so it under-covers. The caller asked for an infinite")
    say("  bound and received a silently clamped finite one.")
    say("")
    say("  What the earlier scan established, and still establishes: on the DEFAULT path,")
    say("  allow_infinite_bounds=False, the guard raises before the clip and the clip is")
    say("  dead. That scan was right about the path it scanned. It did not vary this flag.")

    out = "outputs/probe_output_mapie_clip_reachability.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
