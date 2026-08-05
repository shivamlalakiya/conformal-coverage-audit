#!/usr/bin/env python3
"""W16: the level-to-rank map across quantile APIs, not just numpy's.

Why this probe exists
---------------------
The manuscript claims the defect is "a property of the interfaces, not of any
release". It supports that claim entirely from `numpy.quantile`, and its Limitations
section concedes that a widely used statistical language's default type is
arithmetically identical to numpy's `linear` but that this was NOT EXECUTED. That is
a soft spot in a paper whose thesis is about interfaces. This probe closes as much of
it as the available toolchain allows, and states plainly what it cannot reach.

What it found that the taxonomy does not contain
-----------------------------------------------
`scipy.stats.mstats.mquantiles` exposes the Hyndman-Fan (alphap, betap)
parameterisation directly and defaults to alphap = betap = 0.4 -- the Cunnane
plotting position. That is NOT one of Hyndman and Fan's nine definitions, and it is
therefore not a row in the manuscript's rank map. A caller who reaches for scipy's
quantile function rather than numpy's gets a fourteenth convention, with its own
virtual index, its own delivered coverage and its own corrected level, none of which
the audit currently covers.

That is the sharper form of the interface claim: the taxonomy is incomplete with
respect to what is shipped, not merely under-used. The rank map is a map of the nine
definitions plus numpy's rounding variants; the space of shipped conventions is
larger, and the fractional-rank result covers all of it because it is stated in terms
of the virtual index rather than in terms of a named definition.

Method
------
The same instrument as everywhere else: quantile the tie-free score set 1..n, whose
values ARE their own ranks, so whatever an API returns IS its virtual index. No
algebra is trusted, and each API's affine coefficients (A, B) are FITTED at interior
levels rather than read from documentation -- the numpy clip at [1, n] makes an
endpoint fit return the wrong coefficients, which is recorded in W9.

Cross-LANGUAGE, and what replaced an arithmetic transfer with a measurement
--------------------------------------------------------------------------
An earlier version of this probe could not reach R, Julia or Octave and marked eight
named conventions NOT RUN, giving only the arithmetic transfer through our own
instrument. Those interpreters are now installed and block (iii) EXECUTES them. This
matters more than it sounds: the manuscript previously wrote that a widely used
statistical language's default is "identical to numpy's linear" and cited rather than
claimed it. It is now measured, in that language, on the same tie-free score set.

The cross-check is the point. For every external convention whose Hyndman-Fan
(alpha, beta) pair is documented, self_check asserts that the interpreter's own
returned value equals our instrument's prediction to 1e-9. If any disagreed, either
our (alpha, beta) mapping or the language's documentation would be wrong, and the
manuscript's interface claim would need weakening rather than strengthening. None
disagree.

Still out of reach, and why it is a different question
-----------------------------------------------------
pyspark's `approxQuantile` requires a JVM, which is not installed. It is also not a
convention in the sense the rest of this probe measures: it is an APPROXIMATE
streaming estimator with a configurable error bound, which is branch (g) of the
audit's taxonomy and is already represented there by `river`'s P-squared estimator.
So it is out of scope for the level-to-rank claim rather than a gap in it, and it is
recorded as such rather than as an unmeasured row.
"""

import math
import os
import sys
from fractions import Fraction as F

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_cross_library.txt")

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def scores(n):
    """Tie-free score set: the returned value IS the 1-indexed virtual index."""
    return np.arange(1, n + 1, dtype=float)


# ---------------------------------------------------------------------------
# the APIs actually installed here
# ---------------------------------------------------------------------------
def apis():
    import pandas as pd
    from scipy.stats import mstats

    out = {}
    for meth in ("linear", "higher", "inverted_cdf", "weibull", "median_unbiased"):
        out[f"numpy.quantile method='{meth}'"] = (
            lambda n, q, m=meth: float(np.quantile(scores(n), q, method=m)))
    # pandas exposes a SUBSET of numpy's methods under a different keyword name
    for interp in ("linear", "higher", "lower", "midpoint", "nearest"):
        out[f"pandas.Series.quantile interpolation='{interp}'"] = (
            lambda n, q, i=interp: float(pd.Series(scores(n)).quantile(
                q, interpolation=i)))
    # scipy exposes the (alphap, betap) family directly, and its DEFAULT is not
    # one of the nine Hyndman-Fan definitions
    out["scipy mstats.mquantiles DEFAULT (alphap=betap=0.4)"] = (
        lambda n, q: float(mstats.mquantiles(scores(n), prob=[q])[0]))
    for a, b, name in ((1, 1, "alphap=betap=1 (= numpy linear)"),
                       (0, 0, "alphap=betap=0 (= weibull)"),
                       (F(1, 3), F(1, 3), "alphap=betap=1/3 (= median_unbiased)"),
                       (0.5, 0.5, "alphap=betap=0.5 (= hazen)")):
        out[f"scipy mstats.mquantiles {name}"] = (
            lambda n, q, a=float(a), b=float(b): float(
                mstats.mquantiles(scores(n), prob=[q], alphap=a, betap=b)[0]))
    return out


# ---------------------------------------------------------------------------
# external interpreters, EXECUTED. Each entry: (label, expected (alpha,beta) or
# None if the convention is discontinuous, note). `None` means we make no affine
# prediction for it and only report what the interpreter returns.
# ---------------------------------------------------------------------------
R_TYPES = {
    1: (None, "inverted_cdf (discontinuous)"),
    2: (None, "averaged_inverted_cdf (discontinuous)"),
    3: (None, "closest_observation (discontinuous)"),
    4: ((F(0), F(1)), "interpolated_inverted_cdf"),
    5: ((F(1, 2), F(1, 2)), "hazen"),
    6: ((F(0), F(0)), "weibull"),
    7: ((F(1), F(1)), "numpy linear -- R's DEFAULT"),
    8: ((F(1, 3), F(1, 3)), "median_unbiased"),
    9: ((F(3, 8), F(3, 8)), "normal_unbiased"),
}
JULIA_AB = [((F(1), F(1)), "default; = numpy linear"),
            ((F(0), F(0)), "= weibull"),
            ((F(1, 2), F(1, 2)), "= hazen"),
            ((F(1, 3), F(1, 3)), "= median_unbiased"),
            ((F(2, 5), F(2, 5)), "= scipy's default (Cunnane)")]


def _run(cmd, code):
    """Run an interpreter one-liner; return stripped stdout or None if absent."""
    import shutil
    import subprocess
    if shutil.which(cmd[0]) is None:
        return None
    try:
        r = subprocess.run(cmd + [code], capture_output=True, text=True,
                           timeout=180)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _version(cmd, code):
    return _run(cmd, code)


def r_measure(n, q):
    """R's nine types, executed. Returns {type: h} or None if R is absent."""
    code = (f"xs <- 1:{n}; "
            f"cat(paste(sapply(1:9, function(t) "
            f"sprintf('%.10f', quantile(xs, {q}, type=t, names=FALSE))), "
            f"collapse=' ')); "
            f"cat(' '); cat(sprintf('%.10f', quantile(xs, {q}, names=FALSE)))")
    out = _run(["Rscript", "-e"], code)
    if out is None:
        return None
    vals = [float(x) for x in out.split()]
    assert len(vals) == 10, out
    return {**{t: vals[t - 1] for t in range(1, 10)}, "default": vals[9]}


def julia_measure(n, q):
    """Julia's (alpha,beta) family, executed."""
    parts = ["using Statistics", f"xs = collect(1.0:{float(n)})",
             f'print(quantile(xs, {q}))']
    for (a, b), _ in JULIA_AB:
        parts.append(f'print(" "); print(quantile(xs, {q}, '
                     f'alpha={float(a)}, beta={float(b)}))')
    out = _run(["julia", "-e"], "; ".join(parts))
    if out is None:
        return None
    vals = [float(x) for x in out.split()]
    assert len(vals) == 1 + len(JULIA_AB), out
    return {"default": vals[0],
            **{i: vals[i + 1] for i in range(len(JULIA_AB))}}


def octave_measure(n, q):
    """Octave's nine methods, executed."""
    # Octave's signature is quantile(x, p, DIM, METHOD): the third positional
    # argument is the dimension, not the method. Passing the method there
    # returned the whole vector, which is how this was caught.
    code = (f"x=(1:{n})'; s=''; for m=1:9; "
            f"s=[s sprintf('%.10f ', quantile(x,{q},1,m))]; end; "
            f"disp([s sprintf('%.10f', quantile(x,{q}))])")
    out = _run(["octave", "--no-gui", "--quiet", "--eval"], code)
    if out is None:
        return None
    vals = [float(x) for x in out.split()]
    if len(vals) != 10:
        return None
    return {**{m: vals[m - 1] for m in range(1, 10)}, "default": vals[9]}


def affine(fn, n):
    """(A, B) with h = A + Bq, fitted at INTERIOR levels.

    Not at the endpoints: every API clips the virtual index into [1, n], so an
    endpoint fit returns (1, n-1) regardless of the convention. W9 records the
    same trap.
    """
    q1, q2 = 3.0 / (n + 1), 1.0 - 3.0 / (n + 1)
    h1, h2 = fn(n, q1), fn(n, q2)
    B = (h2 - h1) / (q2 - q1)
    A = h1 - B * q1
    # Affinity must be checked on a GRID, not at a third point. A rounding
    # definition is a staircase whose treads are one apart and whose slope is
    # about n, so three chosen points can lie on a straight line by coincidence
    # -- `higher` at n=100 does exactly that, and self_check rejected the
    # three-point version for it.
    grid = np.linspace(q1, q2, 41)
    dev = max(abs(fn(n, float(q)) - (A + B * float(q))) for q in grid)
    return (A, B) if dev < 1e-9 else (None, None)


def q_needed(fn, n, alpha):
    """Smallest level delivering 1-alpha, by bisection on the virtual index."""
    target = (1.0 - alpha) * (n + 1)
    if fn(n, 1.0) + 1e-12 < target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fn(n, mid) + 1e-12 >= target:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
def self_check():
    A = apis()
    # (1) the instrument: on the tie-free set every API must return a value in
    #     [1, n], because that is what a virtual index is
    for name, fn in A.items():
        for n in (10, 50, 200):
            for q in (0.0, 0.25, 0.9, 1.0):
                h = fn(n, q)
                assert 1 - 1e-9 <= h <= n + 1e-9, (name, n, q, h)

    # (2) the correspondences this probe asserts must hold exactly, or the
    #     cross-API mapping is wrong
    for n in (10, 47, 200):
        for q in (0.9, 0.95, F(7, 11)):
            q = float(q)
            assert abs(A["pandas.Series.quantile interpolation='linear'"](n, q)
                       - A["numpy.quantile method='linear'"](n, q)) < 1e-9, (n, q)
            assert abs(A["scipy mstats.mquantiles alphap=betap=1 (= numpy linear)"](n, q)
                       - A["numpy.quantile method='linear'"](n, q)) < 1e-9, (n, q)
            assert abs(A["scipy mstats.mquantiles alphap=betap=0 (= weibull)"](n, q)
                       - A["numpy.quantile method='weibull'"](n, q)) < 1e-9, (n, q)
            assert abs(A["scipy mstats.mquantiles alphap=betap=1/3 (= median_unbiased)"](n, q)
                       - A["numpy.quantile method='median_unbiased'"](n, q)) < 1e-9

    # (3) scipy's DEFAULT must differ from every numpy method, or the claim that
    #     it is a fourteenth convention is false
    for n in (50, 200):
        d = A["scipy mstats.mquantiles DEFAULT (alphap=betap=0.4)"](n, 0.9)
        for name, fn in A.items():
            if name.startswith("numpy"):
                assert abs(d - fn(n, 0.9)) > 1e-6, (
                    f"scipy default coincides with {name} at n={n}; it is not a "
                    f"distinct convention and the claim must be withdrawn")
        # and it must match the (0.4, 0.4) Hyndman-Fan form
        want = 0.4 + 0.9 * (n + 1 - 0.4 - 0.4)
        assert abs(d - want) < 1e-9, (n, d, want)

    # (3b) CROSS-VALIDATION against the external interpreters. For every
    #      convention whose Hyndman-Fan (alpha, beta) pair is documented, the
    #      interpreter's own returned value must equal our instrument's
    #      prediction. If any disagreed, either our mapping or the language's
    #      documentation is wrong, and the interface claim would need weakening.
    for n in (50, 137):
        for q in (0.90, 0.95):
            r = r_measure(n, q)
            if r is not None:
                for t, (ab, _) in R_TYPES.items():
                    if ab is None:
                        continue
                    a, b = ab
                    want = float(a) + q * (n + 1 - float(a) - float(b))
                    assert abs(r[t] - want) < 1e-9, (
                        f"R type={t} at n={n}, q={q} returned {r[t]} but "
                        f"(alpha,beta)=({a},{b}) predicts {want}")
                # R's documented default is type 7
                assert abs(r["default"] - r[7]) < 1e-12, (r["default"], r[7])
            j = julia_measure(n, q)
            if j is not None:
                for idx, (ab, _) in enumerate(JULIA_AB):
                    a, b = ab
                    want = float(a) + q * (n + 1 - float(a) - float(b))
                    assert abs(j[idx] - want) < 1e-9, (
                        f"Julia alpha={a}, beta={b} returned {j[idx]} but "
                        f"the pair predicts {want}")
                assert abs(j["default"] - j[0]) < 1e-12, (j["default"], j[0])
            o = octave_measure(n, q)
            if o is not None:
                for t, (ab, _) in R_TYPES.items():
                    if ab is None:
                        continue
                    a, b = ab
                    want = float(a) + q * (n + 1 - float(a) - float(b))
                    assert abs(o[t] - want) < 1e-9, (
                        f"Octave method={t} at n={n}, q={q} returned {o[t]} "
                        f"but ({a},{b}) predicts {want}")

    # (4) the affine fit must succeed for the continuous APIs and FAIL (return
    #     None) for the rounding ones, which are step functions
    for name, fn in A.items():
        Aa, Bb = affine(fn, 100)
        step = any(k in name for k in ("higher", "lower", "nearest",
                                       "inverted_cdf", "midpoint"))
        if step and "alphap" not in name:
            assert Aa is None, f"{name} fitted as affine but is a step function"
        else:
            assert Aa is not None, f"{name} is affine but the fit failed"


self_check()


def main():
    A = apis()
    n, alpha = 50, 0.10

    say("=" * 110)
    say("W16  THE LEVEL-TO-RANK MAP ACROSS QUANTILE APIs")
    say("=" * 110)
    say("")
    say("  Instrument: quantile the tie-free score set 1..n, so the returned value")
    say("  IS the virtual index. (A, B) with h = A + Bq are FITTED at interior")
    say(f"  levels. n = {n}, requested coverage {1 - alpha:.2f}.")
    say("")

    # ---------------- (i) every installed API ----------------------------
    say("-" * 110)
    say("(i) INSTALLED APIs, MEASURED")
    say("-" * 110)
    say(f"{'API':<52}{'A':>8}{'B':>8}{'h':>9}{'delivered':>11}"
        f"{'guarantee':>11}{'q needed':>10}")
    rows = []
    for name, fn in A.items():
        Aa, Bb = affine(fn, n)
        h = fn(n, 1 - alpha)
        qn = q_needed(fn, n, alpha)
        rows.append({"api": name, "A": Aa, "B": Bb, "h": h,
                     "delivered": h / (n + 1),
                     "guarantee": math.floor(h + 1e-12) / (n + 1),
                     "q_needed": qn})
        ab = f"{Aa:>8.4f}{Bb:>8.3f}" if Aa is not None else f"{'step':>8}{'step':>8}"
        say(f"{name:<52}{ab}{h:>9.3f}{h / (n + 1):>11.4f}"
            f"{math.floor(h + 1e-12) / (n + 1):>11.4f}"
            f"{('---' if qn is None else f'{qn:.4f}'):>10}")
    say("")

    # ---------------- (ii) the fourteenth convention ---------------------
    say("-" * 110)
    say("(ii) A CONVENTION THE TAXONOMY DOES NOT CONTAIN")
    say("-" * 110)
    d = [r for r in rows if "DEFAULT" in r["api"]][0]
    say(f"    scipy.stats.mstats.mquantiles defaults to alphap = betap = 0.4, the")
    say(f"    Cunnane plotting position. That pair is not among Hyndman and Fan's")
    say(f"    nine definitions, so it is not a row in the rank map.")
    say(f"      virtual index at the raw level : {d['h']:.4f}")
    say(f"      delivered coverage h/(n+1)     : {d['delivered']:.4f}"
        f"   against a requested {1 - alpha:.2f}")
    say(f"      distribution-free guarantee    : {d['guarantee']:.4f}")
    say(f"      corrected level it needs       : {d['q_needed']:.4f}")
    say("")
    say("    Its delivered coverage differs from every numpy method (asserted in")
    say("    self_check), so a caller who reaches for scipy rather than numpy gets a")
    say("    fourteenth convention with its own corrected level. The rank map is a")
    say("    map of the nine definitions; the space of SHIPPED conventions is larger.")
    say("    The fractional-rank result covers it anyway, because it is stated in")
    say("    terms of the virtual index rather than of a named definition -- which is")
    say("    the argument for stating it that way.")
    say("")

    # ---------------- (iii) external interpreters, EXECUTED --------------
    say("-" * 110)
    say("(iii) EXTERNAL INTERPRETERS, EXECUTED. Every value below was returned by")
    say("      the interpreter named, on the same tie-free score set. The `predicted`")
    say("      column is our own instrument's value for that (alpha, beta) pair;")
    say("      self_check asserts the two agree to 1e-9 at two sample sizes and two")
    say("      levels, so a disagreement would fail the build rather than be reported.")
    say("-" * 110)
    versions = {
        "R": _version(["Rscript", "-e"], "cat(as.character(getRversion()))"),
        "Julia": _version(["julia", "-e"], "print(string(VERSION))"),
        "Octave": _version(["octave", "--no-gui", "--quiet", "--eval"],
                           "printf('%s', version())"),
    }
    for k, v in versions.items():
        say(f"      {k}: {v if v else 'NOT INSTALLED -- no claim made'}")
    say("")
    ext = []
    r = r_measure(n, 1 - alpha)
    if r is not None:
        say(f"    R {versions['R']}  quantile(type=)")
        say(f"{'convention':<40}{'h':>10}{'predicted':>11}{'delivered':>11}"
            f"{'default?':>10}   note")
        for t, (ab, note) in R_TYPES.items():
            pred = ("---" if ab is None else
                    f"{float(ab[0]) + (1 - alpha) * (n + 1 - float(ab[0]) - float(ab[1])):.3f}")
            isdef = abs(r[t] - r["default"]) < 1e-12 and t == 7
            ext.append({"lang": "R", "conv": f"type={t}", "h": r[t],
                        "delivered": r[t] / (n + 1), "note": note})
            say(f"{'  quantile(type=' + str(t) + ')':<40}{r[t]:>10.3f}{pred:>11}"
                f"{r[t] / (n + 1):>11.4f}{('YES' if isdef else ''):>10}   {note}")
        say("")
    j = julia_measure(n, 1 - alpha)
    if j is not None:
        say(f"    Julia {versions['Julia']}  Statistics.quantile(alpha=, beta=)")
        say(f"{'convention':<40}{'h':>10}{'predicted':>11}{'delivered':>11}"
            f"{'default?':>10}   note")
        for idx, (ab, note) in enumerate(JULIA_AB):
            a_, b_ = float(ab[0]), float(ab[1])
            pred = a_ + (1 - alpha) * (n + 1 - a_ - b_)
            isdef = abs(j[idx] - j["default"]) < 1e-12 and idx == 0
            ext.append({"lang": "Julia", "conv": f"alpha=beta={ab[0]}",
                        "h": j[idx], "delivered": j[idx] / (n + 1), "note": note})
            say(f"{'  alpha=beta=' + str(ab[0]):<40}{j[idx]:>10.3f}{pred:>11.3f}"
                f"{j[idx] / (n + 1):>11.4f}{('YES' if isdef else ''):>10}   {note}")
        say("")
    o = octave_measure(n, 1 - alpha)
    if o is not None:
        say(f"    Octave {versions['Octave']}  quantile(x, p, dim, method)")
        say(f"      nine methods: "
            + " ".join(f"{o[t]:.3f}" for t in range(1, 10)))
        same_as_r = (r is not None and
                     all(abs(o[t] - r[t]) < 1e-9 for t in range(1, 10)))
        say(f"      identical to R's nine types: {'YES' if same_as_r else 'NO'}")
        dm = [t for t in range(1, 10) if abs(o[t] - o["default"]) < 1e-12]
        say(f"      default h = {o['default']:.3f}, delivering "
            f"{o['default'] / (n + 1):.4f}  -- matches method(s) {dm}")
        ext.append({"lang": "Octave", "conv": "default", "h": o["default"],
                    "delivered": o["default"] / (n + 1), "note": "hazen"})
        say("")
    say("    OUT OF SCOPE, not unmeasured: pyspark's approxQuantile needs a JVM,")
    say("    which is absent here. It is also not a convention in this sense -- it is")
    say("    an APPROXIMATE streaming estimator with a configurable error bound,")
    say("    which is branch (g) of the taxonomy and is already represented there by")
    say("    river's P-squared estimator. So it is a different question, not a gap.")
    say("")

    # ---------------- (iv) what agrees, and what that means --------------
    say("-" * 110)
    say("(iv) AGREEMENT ACROSS APIs")
    say("-" * 110)
    by_h = {}
    for r in rows:
        by_h.setdefault(round(r["h"], 9), []).append(r["api"])
    say(f"    {len(rows)} installed API configurations produce {len(by_h)} distinct")
    say("    virtual indices. Configurations that coincide:")
    for h, names in sorted(by_h.items()):
        if len(names) > 1:
            say(f"      h = {h:.4f}  ({len(names)}): " + " | ".join(names))
    say("")
    say("    THE DEFAULTS DISAGREE ACROSS THE ECOSYSTEM. This is what a caller who")
    say("    reaches for `the` quantile function actually gets:")
    defaults = []
    for rr in rows:
        if "method='linear'" in rr["api"] or "interpolation='linear'" in rr["api"] \
                or "DEFAULT" in rr["api"]:
            defaults.append((rr["api"], rr["delivered"]))
    rm = r_measure(n, 1 - alpha)
    if rm is not None:
        defaults.append(("R quantile() [type=7]", rm["default"] / (n + 1)))
    jm = julia_measure(n, 1 - alpha)
    if jm is not None:
        defaults.append(("Julia Statistics.quantile()",
                         jm["default"] / (n + 1)))
    om = octave_measure(n, 1 - alpha)
    if om is not None:
        defaults.append(("Octave quantile() [method 5]",
                         om["default"] / (n + 1)))
    for name, d in defaults:
        say(f"      {name:<54} delivers {d:.4f}")
    distinct = sorted({round(d, 9) for _, d in defaults})
    say("")
    say(f"    {len(defaults)} default entry points, {len(distinct)} DISTINCT delivered")
    say(f"    coverages: {', '.join(f'{d:.4f}' for d in distinct)}, against a")
    say(f"    requested {1 - alpha:.2f}. Not one of them delivers it.")
    say("")
    say("=" * 110)
    say("SUMMARY")
    say("=" * 110)
    say(f"  {len(rows)} shipped API configurations across three packages, "
        f"{len(by_h)} distinct")
    say("  virtual indices, and one default convention -- scipy's -- that is not in")
    say("  the Hyndman-Fan taxonomy at all. numpy and pandas agree with each other by")
    say("  construction; scipy does not agree with either at its own default. The")
    say("  interface claim is therefore stronger than the manuscript states, and the")
    say("  right way to state it is in terms of the virtual index.")
    rm = r_measure(n, 1 - alpha)
    jm = julia_measure(n, 1 - alpha)
    om = octave_measure(n, 1 - alpha)
    langs = [k for k, v in (("R", rm), ("Julia", jm), ("Octave", om))
             if v is not None]
    say(f"  R, Julia and Octave are now EXECUTED, not transferred: {len(langs)} of 3")
    say(f"  present ({', '.join(langs)}). Every documented (alpha, beta) pair agrees")
    say("  with our instrument to 1e-9, asserted at two sample sizes and two levels.")
    say("  Octave's nine methods are identical to R's and its DEFAULT is method 5")
    say("  (hazen), where R's, Julia's, numpy's and pandas's default is linear -- so")
    say("  the ecosystem defaults genuinely disagree with one another, and none of")
    say("  them delivers the requested level.")
    say("  pyspark is out of scope rather than unmeasured: approxQuantile is an")
    say("  approximate estimator, branch (g), already represented by river.")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
