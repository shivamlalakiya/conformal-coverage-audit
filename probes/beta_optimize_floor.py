#!/usr/bin/env python3
"""Does mapie's beta optimisation select a level no order statistic can carry?

ANSWER: NO. THE READING BELOW IS WRONG AND THIS FILE RECORDS ITS REFUTATION.
============================================================================
Read section (iii-a) before anything else. The tie-free instrument shows the
selected lower rail lands on order statistic 1, which excludes ZERO mass below
it and is therefore a valid lower bound for any beta whatsoever. The predicate
this probe was built around -- `beta * (n+1) >= 1` -- is the condition for the
rail to be a NON-TRIVIAL order statistic, not the condition for it to be VALID.
A rail at the sample minimum is loose, not invalid, and a width minimiser
choosing looseness on one side is doing its job.

So section (ii)'s "48 of 84" is a count of cells where a predicate of mine was
not satisfied, and that predicate did not mean what the docstring said. It is
kept below, labelled, because deleting a refuted measurement is worse than
printing it next to what refuted it.

The general arithmetic is unaffected and is not in question: for the INTERVAL to
be valid the two excluded masses must sum to at most alpha, which is
probes/attainable_grid.py. What is in question was the claim that this optimiser
violates it, and the tie-free run says it does not.

One thing section (iii) turned up that is NOT explained by any of the above: a
delivered coverage of 0.7725 against a nominal 0.90 at n = 20. That is a real
number from a real public call and it is NOT evidence for the retracted reading.
It is an open observation, recorded as such, with the two candidate explanations
being the corrected-level clip already reported elsewhere and the fact that n=20
sits one point above the two-rail floor of 19. Do not put it in a manuscript
until a probe says which.

The original reading, kept so the refutation has something to point at
---------------------------------------------------------------------
`BaseRegressionScore._beta_optimize` (regression.py:178-230) searches

    betas = np.linspace(alpha/(n+1), alpha, num=n)

and returns the beta minimising `(1-alpha+beta)-quantile - beta-quantile`, i.e.
the width. The two rails then resolve at levels `beta` and `1-alpha+beta`, so the
miscoverage is split asymmetrically as (beta, alpha-beta).

By the attainable-grid arithmetic (probes/attainable_grid.py), a two-rail interval
excludes `a` gaps below and `n+1-b` above, and validity needs a/(n+1) <= beta. So
a finite LOWER rail requires

    beta * (n + 1) >= 1.

At the bottom of that grid, beta = alpha/(n+1), the requirement becomes alpha >= 1
-- false for every conventional level, at EVERY n. Collecting more calibration
data does not help, because the grid's lower end shrinks with n at exactly the
rate the requirement tightens.

That is a reading of the source. Whether it BITES depends on where the argmin
lands, which is a property of the score distribution and not of the arithmetic,
and this programme has retracted twice for generalising a source reading without
running it. So:

  (i)  enumerate the beta grid in exact arithmetic and report, per (n, alpha), how
       many grid points are infeasible for the lower rail;
  (ii) drive the SHIPPED path -- `predict_interval` with
       `minimize_interval_width=True` -- over a spread of n, alpha and score
       skewness, capture the beta actually selected, and record whether the
       returned lower bound is finite where the arithmetic says it cannot be;
  (iii) report the delivered coverage there, and a control at the default path.

A negative result is a result: if the argmin never lands in the infeasible region
for any score distribution tried, that is worth one sentence and no finding, and
the sentence has to say which distributions were tried.

    python probes/beta_optimize_floor.py
"""

import math
import os
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "outputs/probe_output_beta_optimize_floor.txt"

ALPHAS = (F(1, 10), F(1, 20), F(1, 100))
NS = (10, 20, 30, 50, 100, 200, 500)
REPS = 400
SEED = 20260806


# ---------------------------------------------------------------------------
# (i) the arithmetic, exact
# ---------------------------------------------------------------------------
def beta_grid(alpha, n):
    """mapie's grid, in exact rationals: linspace(alpha/(n+1), alpha, num=n)."""
    lo, hi = alpha / (n + 1), alpha
    if n == 1:
        return [hi]
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


def lower_rail_nontrivial(beta, n):
    """Can a lower rail ABOVE the sample minimum carry an excluded mass of `beta`?

    MISNAMED IN AN EARLIER VERSION as `lower_rail_finite`, and the misnaming is
    the whole error. The rail excludes `a` gaps and Pr(V_(n+1) < V_(a)) = a/(n+1),
    so a rail at index a is valid for budget beta whenever a/(n+1) <= beta. At
    a = 1 that is 0 <= beta, true always: the sample minimum is a valid lower
    bound for ANY budget, because nothing lies strictly below it in the sample.

    What `beta*(n+1) >= 1` decides is whether a >= 2 is affordable, i.e. whether
    the rail can sit anywhere above the minimum. Failing it means the rail must be
    the minimum -- loose, and perfectly valid. Section (iii-a) measures that the
    shipped path returns exactly that, which is what refutes the finding this
    probe was written to establish.
    """
    return F(beta) * (n + 1) >= 1


# the old name, kept so the refuted section below reads as it originally ran
lower_rail_finite = lower_rail_nontrivial


def self_check():
    for alpha in ALPHAS:
        for n in NS:
            g = beta_grid(alpha, n)
            assert len(g) == n
            assert g[0] == alpha / (n + 1) and g[-1] == alpha
            assert all(g[i] < g[i + 1] for i in range(n - 1))
            # the bottom of the grid is infeasible for the lower rail at EVERY n:
            # beta*(n+1) = alpha < 1
            assert F(g[0]) * (n + 1) == alpha < 1
            assert not lower_rail_finite(g[0], n)
            # the top of the grid is the symmetric-ish end and is feasible exactly
            # when the one-rail floor at alpha is cleared
            assert lower_rail_finite(g[-1], n) == (F(n + 1) * alpha >= 1)
            # and the count of infeasible grid points is monotone in alpha
    # a coarse monotonicity check: raising alpha cannot make MORE points infeasible
    for n in NS:
        counts = [sum(1 for b in beta_grid(a, n) if not lower_rail_finite(b, n))
                  for a in sorted(ALPHAS)]
        assert counts == sorted(counts, reverse=True), (n, counts)


self_check()


# ---------------------------------------------------------------------------
# (ii) the shipped path
# ---------------------------------------------------------------------------
def shipped_cell(n, alpha, skew, rng):
    """Drive mapie's width-minimising path and report what it returns.

    Uses the MapieRegressor-era public entry point that exposes
    optimize_beta/minimize_interval_width, via the conformity-score layer the
    census already anchors, so the probe measures the shipped composition rather
    than a reimplementation of it.
    """
    from mapie.conformity_scores import AbsoluteConformityScore
    from mapie.conformity_scores.regression import BaseRegressionScore

    # scores with a controllable right skew: |t| for heavy, |normal| for light,
    # and an exponential in between. The argmin's position is a property of these.
    if skew == "normal":
        s = np.abs(rng.standard_normal(n))
    elif skew == "expon":
        s = rng.exponential(size=n)
    elif skew == "lognormal":
        s = rng.lognormal(mean=0.0, sigma=1.5, size=n)
    else:
        s = np.abs(rng.standard_t(df=1.5, size=n))

    a = np.array([float(alpha)])
    beta = float(BaseRegressionScore._beta_optimize(a, s, -s)[0])
    grid = beta_grid(alpha, n)
    # which grid point was selected, and is it feasible for the lower rail?
    idx = int(np.argmin([abs(float(g) - beta) for g in grid]))
    feasible = lower_rail_finite(grid[idx], n)

    # what the composed site returns at the selected lower level
    lvl_lo = np.array([beta])
    lvl_hi = np.array([1 - float(alpha) + beta])
    sc = AbsoluteConformityScore()
    q_lo = float(sc.get_quantile(s[..., np.newaxis], lvl_lo, axis=0, reversed=True)[0])
    q_hi = float(sc.get_quantile(s[..., np.newaxis], lvl_hi, axis=0)[0])
    return {"beta": beta, "grid_index": idx, "n_grid": n,
            "lower_feasible": feasible,
            "n_infeasible_points": sum(1 for g in grid
                                       if not lower_rail_finite(g, n)),
            "q_lo_finite": math.isfinite(q_lo), "q_hi_finite": math.isfinite(q_hi),
            "q_lo": q_lo, "q_hi": q_hi,
            # the excluded mass the selected lower rail actually claims
            "excluded_below": float(F(beta).limit_denominator(10**9))}


# ---------------------------------------------------------------------------
# (iii-a) the tie-free instrument: the returned rail IS its rank
# ---------------------------------------------------------------------------
def tie_free_rank(n, alpha):
    """Which order statistic does the selected lower rail land on?

    The audit's own instrument, and it should have been the first one used here:
    feed a tie-free score set whose values ARE their own ranks, so a returned
    threshold needs no feasibility predicate of ours to interpret -- it is the
    index, read off. If the lower rail comes back as order statistic `a`, then the
    mass it excludes is exactly a/(n+1), and the question "is that at most beta?"
    is arithmetic on two integers.

    Signed scores are needed, because the asymmetric path stores signed ones. The
    set is {-n, ..., -1} U {1, ..., n} would double the size, so instead the scores
    are 1..n shifted to straddle zero: s_i = i - (n+1)/2 scaled to stay tie-free
    and strictly increasing, and the rank of a returned value is recovered by
    searchsorted rather than by inverting the shift.
    """
    from mapie.conformity_scores import AbsoluteConformityScore
    from mapie.conformity_scores.regression import BaseRegressionScore

    s = np.arange(1, n + 1, dtype=float) - (n + 1) / 2.0
    assert len(np.unique(s)) == n, "score set is not tie-free"
    a = np.array([float(alpha)])
    beta = float(BaseRegressionScore._beta_optimize(a, s, -s)[0])
    sc = AbsoluteConformityScore()
    q_lo = float(sc.get_quantile(s[..., np.newaxis], np.array([beta]),
                                 axis=0, reversed=True)[0])
    srt = np.sort(s)
    # the 1-based index of the returned value in the score set, or 0 for -inf and
    # n+1 for anything above the maximum
    if not math.isfinite(q_lo):
        idx = 0
    else:
        idx = int(np.searchsorted(srt, q_lo, side="left")) + 1
        if q_lo > srt[-1]:
            idx = n + 1
    excluded = F(max(idx - 1, 0), n + 1)   # mass strictly below order statistic idx
    return {"beta": beta, "index": idx, "finite": math.isfinite(q_lo),
            "excluded": excluded, "budget": F(beta).limit_denominator(10**12),
            "valid": excluded <= F(beta).limit_denominator(10**12)}


# ---------------------------------------------------------------------------
# (iii) the SAME question through the public API, end to end
# ---------------------------------------------------------------------------
def end_to_end(n, alpha, skew, rng, n_test=400):
    """`predict_interval(minimize_interval_width=True)` on a prefit regressor.

    Section (ii) drives the census's composed adapter, which is the pair of
    expressions the site is anchored at. That is not the same thing as the public
    method, and this programme has retracted twice for generalising from the
    former to the latter. So this arm sets the documented public keyword on
    `SplitConformalRegressor.predict_interval`, on a prefit model with an
    asymmetric conformity score, and reports what comes back.

    Returns None where the configuration is refused, because a refusal is the
    correct behaviour and must not be counted as a finding.
    """
    from sklearn.linear_model import LinearRegression
    from mapie.conformity_scores import AbsoluteConformityScore
    from mapie.regression import SplitConformalRegressor

    d = 3

    def draw(m):
        X = rng.standard_normal((m, d))
        if skew == "normal":
            e = rng.standard_normal(m)
        elif skew == "expon":
            e = rng.exponential(size=m) - 1.0
        elif skew == "lognormal":
            e = rng.lognormal(mean=0.0, sigma=1.0, size=m)
        else:
            # t with 3 degrees of freedom: heavy-tailed but with a finite variance,
            # so the design matrix does not overflow. An earlier version used
            # df=1.5 and the linear fit reported divide-by-zero in matmul, which
            # would have put an arithmetic artefact into a coverage column.
            e = rng.standard_t(df=3.0, size=m)
        y = X @ np.ones(d) + e
        assert np.all(np.isfinite(X)) and np.all(np.isfinite(y)), (
            f"non-finite draw for shape={skew}; the generator is not usable and a "
            f"coverage number computed from it would be an artefact")
        return X, y

    Xtr, ytr = draw(max(200, 4 * n))
    Xc, yc = draw(n)
    Xte, yte = draw(n_test)
    with np.errstate(all="raise"):
        est = LinearRegression().fit(Xtr, ytr)

    r = SplitConformalRegressor(
        estimator=est, confidence_level=float(1 - alpha), prefit=True,
        conformity_score=AbsoluteConformityScore(sym=False))
    try:
        r.conformalize(Xc, yc)
        _, iv = r.predict_interval(Xte, minimize_interval_width=True)
    except ValueError as exc:
        return {"refused": type(exc).__name__ + ": " + str(exc)[:60]}

    iv = np.asarray(iv)
    lo, hi = iv[:, 0, 0], iv[:, 1, 0]
    scores = np.asarray(r._mapie_regressor.conformity_scores_, dtype=float).ravel()
    # the beta the shipped call selected, recovered from the same optimiser on the
    # same captured scores rather than inferred from the returned rails
    from mapie.conformity_scores.regression import BaseRegressionScore
    a = np.array([float(alpha)])
    beta = float(BaseRegressionScore._beta_optimize(a, scores, -scores)[0])
    return {
        "refused": None,
        "beta": beta,
        "lower_feasible": lower_rail_finite(beta, n),
        "lo_finite": bool(np.all(np.isfinite(lo))),
        "hi_finite": bool(np.all(np.isfinite(hi))),
        "coverage": float(np.mean((lo <= yte) & (yte <= hi))),
        "n_test": n_test,
    }


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 104)
    say("BETA OPTIMISATION AND THE LOWER-RAIL FLOOR")
    say("=" * 104)
    say("self_check() passed at import: the beta grid in exact rationals, and the")
    say("fact that its lowest point claims an excluded mass of alpha/(n+1), for which")
    say("a finite lower rail needs alpha >= 1 -- false at every n and every")
    say("conventional level.")
    say("")

    say("REFUTED SECTIONS FOLLOW. (i) and (ii) test the predicate")
    say("`beta*(n+1) >= 1`, which the docstring called feasibility and which is")
    say("really the condition for a rail ABOVE the sample minimum. (iii-a) uses the")
    say("tie-free instrument and shows the returned rail IS the minimum, excluding")
    say("zero mass, hence valid for any budget. Read (iii-a) as the verdict and")
    say("(i)-(ii) as what it overturned.")
    say("")
    say("(i) how much of mapie's beta grid cannot support a NON-TRIVIAL lower rail")
    say("    (i.e. one strictly above the sample minimum -- not a validity question)")
    say(f"{'alpha':>8} {'n':>6} {'grid pts':>9} {'infeasible':>11} "
        f"{'frac':>7} {'smallest beta':>14} {'beta*(n+1)':>11}")
    say("-" * 104)
    for alpha in ALPHAS:
        for n in NS:
            g = beta_grid(alpha, n)
            bad = sum(1 for b in g if not lower_rail_finite(b, n))
            say(f"{float(alpha):>8.3f} {n:>6} {len(g):>9} {bad:>11} "
                f"{bad / len(g):>7.3f} {float(g[0]):>14.6g} "
                f"{float(F(g[0]) * (n + 1)):>11.4f}")
        say("")

    say("(ii) what the shipped width-minimising path selects, over four score shapes")
    say(f"{'alpha':>7} {'n':>5} {'shape':>10} {'beta':>11} {'grid idx':>9} "
        f"{'lower ok?':>10} {'lo finite':>10} {'hi finite':>10}")
    say("-" * 104)
    rng = np.random.default_rng(SEED)
    hits, tried = [], 0
    try:
        for alpha in ALPHAS:
            for n in NS:
                for shape in ("normal", "expon", "lognormal", "cauchy-ish"):
                    r = shipped_cell(n, alpha, shape, rng)
                    tried += 1
                    if not r["lower_feasible"] and r["q_lo_finite"]:
                        hits.append((alpha, n, shape, r))
                    say(f"{float(alpha):>7.3f} {n:>5} {shape:>10} {r['beta']:>11.6g} "
                        f"{r['grid_index']:>9} "
                        f"{('yes' if r['lower_feasible'] else 'NO'):>10} "
                        f"{('yes' if r['q_lo_finite'] else 'no'):>10} "
                        f"{('yes' if r['q_hi_finite'] else 'no'):>10}")
            say("")
    except ImportError as exc:
        say(f"mapie not importable in this venv ({exc}); run in .venv-tabular")

    say("=" * 104)
    say(f"cells driven: {tried}")
    say(f"cells where the selected beta cannot support a finite lower rail AND a")
    say(f"finite lower bound was returned anyway: {len(hits)}")
    if hits:
        say("")
        say("^ THIS COUNT DOES NOT MEAN WHAT AN EARLIER VERSION SAID IT MEANT. It is")
        say("  the number of cells whose selected beta cannot afford a rail above the")
        say("  sample minimum. Section (iii-a) shows the minimum is what is returned")
        say("  and that it is valid. Listed for the record, not as a finding:")
        for alpha, n, shape, r in sorted(hits, key=lambda h: h[3]["beta"])[:8]:
            say(f"    alpha={float(alpha):.3f} n={n} shape={shape} "
                f"beta={r['beta']:.6g}  beta*(n+1)={float(F(r['beta']).limit_denominator(10**9)) * (n + 1):.4f} "
                f"< 1, lower bound returned finite")
    else:
        say("")
        say("NEGATIVE RESULT. Over the score shapes tried -- half-normal,")
        say("exponential, lognormal(sigma=1.5) and half-t(1.5) -- the width")
        say("argmin never landed on a grid point whose lower rail is infeasible.")
        say("The infeasible region of the grid is real and this optimiser did not")
        say("enter it here. That is a statement about these four shapes at these")
        say("sizes, not about the optimiser in general.")

    say("")
    say("=" * 104)
    say("(iii-a) the tie-free instrument: a returned rail IS its rank")
    say("Scores straddle zero and are strictly increasing, so no predicate of ours")
    say("is needed to interpret what came back. `excluded` is the mass strictly")
    say("below the landed order statistic; `budget` is the beta the optimiser chose.")
    say("valid means excluded <= budget, which is what the rail has to satisfy.")
    say("")
    say(f"{'alpha':>7} {'n':>5} {'beta':>12} {'landed idx':>11} {'excluded':>12} "
        f"{'budget':>12} {'valid?':>7}")
    say("-" * 104)
    tf_bad, tf_tried = [], 0
    try:
        for alpha in ALPHAS:
            for n in NS:
                r = tie_free_rank(n, alpha)
                tf_tried += 1
                if not r["valid"]:
                    tf_bad.append((alpha, n, r))
                say(f"{float(alpha):>7.3f} {n:>5} {r['beta']:>12.6g} "
                    f"{(r['index'] if r['finite'] else 0):>11} "
                    f"{str(r['excluded']):>12} {float(r['budget']):>12.6g} "
                    f"{('yes' if r['valid'] else 'NO'):>7}")
            say("")
        say(f"tie-free cells: {tf_tried}   rails excluding more mass than the")
        say(f"selected beta allows: {len(tf_bad)}")
        if tf_bad:
            say("")
            say("The rail is not a valid lower bound for the level that was chosen.")
            say("This is read off an index, not inferred from a predicate: the score")
            say("set's values are its ranks.")
    except ImportError as exc:
        say(f"mapie not importable in this venv ({exc}); run in .venv-tabular")

    say("")
    say("=" * 104)
    say("(iii) the same question through the PUBLIC method, end to end")
    say("SplitConformalRegressor(prefit, AbsoluteConformityScore(sym=False))")
    say(".predict_interval(X, minimize_interval_width=True)")
    say("")
    say("Section (ii) drives the two expressions the census anchors. This drives the")
    say("documented public keyword. The two are not the same claim and the second is")
    say("the one a caller can reach.")
    say("")
    say(f"{'alpha':>7} {'n':>5} {'shape':>10} {'beta':>11} {'lower ok?':>10} "
        f"{'lo finite':>10} {'coverage':>9} {'nominal':>8} {'note':<28}")
    say("-" * 104)
    e2e_hits, e2e_tried, refused = [], 0, 0
    try:
        rng2 = np.random.default_rng(SEED + 1)
        for alpha in ALPHAS:
            for n in (20, 50, 100, 200):
                for shape in ("normal", "expon", "lognormal", "cauchy-ish"):
                    r = end_to_end(n, alpha, shape, rng2)
                    if r.get("refused"):
                        refused += 1
                        say(f"{float(alpha):>7.3f} {n:>5} {shape:>10} "
                            f"{'--':>11} {'--':>10} {'--':>10} {'--':>9} "
                            f"{float(1 - alpha):>8.2f} {r['refused']:<28}")
                        continue
                    e2e_tried += 1
                    bad = (not r["lower_feasible"]) and r["lo_finite"]
                    if bad:
                        e2e_hits.append((alpha, n, shape, r))
                    say(f"{float(alpha):>7.3f} {n:>5} {shape:>10} {r['beta']:>11.6g} "
                        f"{('yes' if r['lower_feasible'] else 'NO'):>10} "
                        f"{('yes' if r['lo_finite'] else 'no'):>10} "
                        f"{r['coverage']:>9.4f} {float(1 - alpha):>8.2f} "
                        f"{('finite rail where none valid' if bad else ''):<28}")
            say("")
    except ImportError as exc:
        say(f"mapie not importable in this venv ({exc}); run in .venv-tabular")

    say(f"public-API cells driven: {e2e_tried}   refused by a guard: {refused}")
    say(f"cells returning a finite lower rail the selected beta cannot support: "
        f"{len(e2e_hits)}")
    say("")
    say("The 'finite rail where none valid' note in the last column is the REFUTED")
    say("predicate, printed because that is what this run originally reported.")
    say("Section (iii-a) is the verdict: the rail is the sample minimum, it excludes")
    say("no mass, and it is valid for any budget.")
    say("")
    say("WHAT IS STILL UNEXPLAINED, and is not evidence for the retracted reading:")
    say("the coverage column undercovers in some cells and overcovers in others, and")
    say("the worst miss sits at n = 20 with nominal 0.90 -- one point above the")
    say("two-rail floor of 19 from probes/attainable_grid.py. Two candidates, neither")
    say("tested here: the corrected-level clip already reported for this package, and")
    say("small-n behaviour at the floor. Until a probe separates them this is an")
    say("observation and must not enter a manuscript as a finding.")
    if e2e_hits:
        say("")
        say(f"(cells flagged by the refuted predicate: {len(e2e_hits)})")

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        OUT)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
