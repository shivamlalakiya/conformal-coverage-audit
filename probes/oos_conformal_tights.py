#!/usr/bin/env python3
"""Out-of-census: the resolution site in conformal-tights, and what the clip does to it.

Why this is not an adapter in `conformance_suite.py`
---------------------------------------------------
The suite classifies a callable of the shape (scores, level) -> threshold. The site
here is not that shape. `_conformal_coherent_quantile_regressor.py:255` resolves a
level into a position in a held-out array and returns a per-quantile *bias* on top of
a fitted quantile regressor's prediction, and the array is that regressor's residuals
on the level-2 calibration split, which no caller supplies. So the level-to-rank
question is asked here by running the library and reading its own fitted state back,
not by handing it a score set.

What is measured
----------------
For each cell: the virtual index numpy's linear default lands on for the requested
level at the live level-2 size, the rank a one-sided guarantee at that level requires,
the quantile the library computed, the coherence clip interval, and the bias it
actually stored. The reconstruction is checked against the library's own value first
(`self_check`), because a rank read off an array the library did not use would be
arithmetic about a lookalike.

Two facts decide how the result reads:

  1. The block is skipped entirely below 128 level-2 examples (line 242), and the bias
     stays exactly zero. A dataset of 1000 rows lands there under the default
     calibration sizes, so the site does not fire at all at ordinary sizes.
  2. Where it does fire, the coherence clip can override the resolved value. Where it
     does, the rank the quantile call landed on decides nothing, and no coverage
     consequence follows from it. Reported per cell rather than averaged.

Environment: [4] out-of-census in probe-requirements.txt, plus conformal-tights.
    python probes/oos_conformal_tights.py
"""

import math
import sys

import numpy as np

QUANTILES = (0.025, 0.5, 0.975)
# 1000 lands under the 128-example floor; 1500 is the first size that clears it; 3000
# and 4500 are past twice that, because a sweep that stops at the first boundary it
# finds cannot tell a boundary from a plateau.
SIZES = (1000, 1500, 3000, 4500)
SEED = 20260831


def virtual_index(q, n):
    """The 1-based position numpy's linear default lands on."""
    return 1.0 + q * (n - 1)


def requirement(q, n):
    """(position, direction) a one-sided guarantee at level q needs, or None.

    The upper rule is the audit's: the least rank from the top whose exact coverage
    reaches q, ceil((n+1)q), and a helper landing BELOW it undercovers. A lower
    quantile is the mirror of that rule on the reflected sample, floor((n+1)q)
    counted from the bottom, and there a helper landing ABOVE it raises the lower
    bound, narrows the interval, and undercovers in the same direction. The median
    carries no one-sided coverage claim, so no comparison is made at it. Applying the
    upper rule to a lower quantile would report the lower cells as conservative when
    they are not, which is why the direction is returned rather than assumed.
    """
    if q == 0.5:
        return None, "median, no one-sided claim"
    if q > 0.5:
        return math.ceil((n + 1) * q), "upper: needs virtual >= required"
    return math.floor((n + 1) * q), "lower: needs virtual <= required"


def generators():
    """(label, callable) pairs. The second is misspecified on purpose."""
    def linear(rng, n):
        X = rng.normal(size=(n, 3))
        y = X @ np.array([1.5, -2.0, 0.5]) + rng.normal(scale=0.7, size=n)
        return X, y

    def heteroscedastic(rng, n):
        X = rng.normal(size=(n, 3))
        scale = 0.2 + np.abs(X[:, 0])
        y = np.sin(2.0 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(scale=scale, size=n)
        return X, y

    return (("well-specified linear", linear),
            ("heteroscedastic, misspecified mean", heteroscedastic))


def fit(gen, n_rows, seed=SEED):
    from conformal_tights import ConformalCoherentQuantileRegressor
    rng = np.random.default_rng(seed)
    X, y = gen(rng, n_rows)
    clf = ConformalCoherentQuantileRegressor(random_state=42).fit(X, y)
    qs = np.asarray(QUANTILES)
    clf.predict_quantiles(X[:5], quantiles=qs)   # the conformal fit is lazy
    return clf, qs


def internals(clf, qs, target_type="Δŷ"):
    """The exact arrays the library's own quantile call ran on."""
    qt = tuple(qs)
    cqr_l1 = clf.conformal_l1_[target_type][qt]
    bias = np.asarray(clf.conformal_l2_[target_type][qt], dtype=float)
    rel = "/ŷ" in target_type
    eps = np.finfo(clf.ŷ_calib_l1_.dtype).eps

    def cqr_y(residuals, yhat):
        return -residuals / (np.maximum(np.abs(yhat), eps) if rel else 1)

    y_l1 = cqr_y(clf.residuals_calib_l1_, clf.ŷ_calib_l1_)
    y_l2 = cqr_y(clf.residuals_calib_l2_, clf.ŷ_calib_l2_)
    X_l1 = clf.ŷ_calib_l1_nonconformity_
    X_l2 = clf.ŷ_calib_l2_nonconformity_
    return cqr_l1, bias, X_l1, X_l2, y_l1, y_l2


def cells(clf, qs):
    """One row per (target type, quantile), or a single skipped row below 128."""
    rows = []
    for target_type in ("Δŷ", "Δŷ/ŷ"):
        cqr_l1, bias, X_l1, X_l2, y_l1, y_l2 = internals(clf, qs, target_type)
        n = len(y_l2)
        if n < 128:
            rows.append(dict(target=target_type, n=n, q=None,
                             skipped=True, bias_all_zero=bool(np.all(bias == 0))))
            continue
        pred = cqr_l1.predict(X_l2)
        clip = cqr_l1.intercept_clip(np.vstack([X_l1, X_l2]),
                                     np.hstack([y_l1, y_l2]))
        for j, q in enumerate(qs):
            arr = np.sort(y_l2 - pred[:, j])
            resolved = float(np.quantile(arr, q))
            reclipped = float(np.clip(resolved, clip[0, j], clip[1, j]))
            rr, direction = requirement(q, n)
            vi = virtual_index(q, n)
            if rr is None:
                undercovers = None
            elif q > 0.5:
                undercovers = vi < rr
            else:
                undercovers = vi > rr
            rows.append(dict(
                target=target_type, n=n, q=float(q), skipped=False,
                virtual=vi, required=rr, direction=direction,
                resolved=resolved, clip_lo=float(clip[0, j]),
                clip_hi=float(clip[1, j]), stored=float(bias[j]),
                at_required=float(arr[rr - 1]) if rr else None,
                undercovers=undercovers,
                # The clip is read off the interval, not off a difference of two
                # floats: `stored` comes back through a float32 prediction path and
                # differs from the recomputation by ~1e-10, which a threshold on that
                # difference reads as clipping on every row.
                clipped=bool(resolved < clip[0, j] or resolved > clip[1, j]),
                recon_residual=reclipped - float(bias[j]),
            ))
    return rows


def self_check():
    """The reconstruction must reproduce the library's own bias, and must be able not to.

    Failing input: the same computation over the level-2 array with one element
    dropped. If that still matched, the check would be reading a coincidence.
    """
    clf, qs = fit(generators()[0][1], 1500)
    rows = [r for r in cells(clf, qs) if not r["skipped"]]
    assert rows, "no cell cleared the 128-example floor at 1500 rows"
    worst = max(abs(r["recon_residual"]) for r in rows)
    assert worst < 1e-6, f"reconstruction disagrees with the library by {worst:.3e}"

    cqr_l1, bias, X_l1, X_l2, y_l1, y_l2 = internals(clf, qs)
    pred = cqr_l1.predict(X_l2)
    clip = cqr_l1.intercept_clip(np.vstack([X_l1, X_l2]), np.hstack([y_l1, y_l2]))
    j = len(qs) - 1
    wrong = float(np.clip(np.quantile((y_l2 - pred[:, j])[:-1], qs[j]),
                          clip[0, j], clip[1, j]))
    if math.isclose(wrong, float(bias[j]), rel_tol=0, abs_tol=1e-9):
        raise AssertionError("dropping a level-2 example changed nothing; the check "
                             "cannot distinguish the library's array from a subset")
    return worst


def main():
    worst = self_check()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    import conformal_tights
    from importlib.metadata import version
    say("=" * 104)
    say("OUT OF CENSUS: conformal-tights, the level-2 conformal bias")
    say("=" * 104)
    say(f"conformal-tights {version('conformal-tights')}, "
        f"numpy {np.__version__}")
    say("")
    say("self_check() passed: the reconstruction reproduces the library's own stored")
    say(f"bias to {worst:.2e} (float32 prediction path), and dropping one level-2")
    say("example makes it disagree, so it is that array and not a lookalike.")
    say("")
    say("The site is `np.quantile(y_cqr_l2 - dy_calib_l2_quantiles[:, j], quantile)` at")
    say("_conformal_coherent_quantile_regressor.py:255. The requested level goes in")
    say("unchanged: no (n+1) correction, numpy's linear default, and the position it")
    say("lands on is the virtual index below. Line 256 then clips the result to the")
    say("coherence interval, and line 242 skips the whole block below 128 examples.")
    say("")

    clip_bound = fired = short = comparable = decisive = 0
    skipped_n, head = None, None
    for gen_label, gen in generators():
        say("-" * 104)
        say(f"generator: {gen_label}")
        say("-" * 104)
        say(f"{'rows':>5} {'target':>8} {'n_l2':>5} {'level':>6} {'virtual':>9} "
            f"{'req':>4} {'side':>6} {'under?':>7} {'resolved':>11} {'clip lo':>11} "
            f"{'clip hi':>11} {'stored':>11} {'clip?':>6}")
        for n_rows in SIZES:
            clf, qs = fit(gen, n_rows)
            for r in cells(clf, qs):
                if r["skipped"]:
                    if skipped_n is None:
                        skipped_n = r["n"]
                    say(f"{n_rows:>5} {r['target']:>8} {r['n']:>5}   "
                        f"below the 128-example floor: the block is skipped and the "
                        f"bias stays zero ({r['bias_all_zero']})")
                    continue
                fired += 1
                clip_bound += r["clipped"]
                short += bool(r["undercovers"])
                comparable += r["undercovers"] is not None
                decisive += bool(r["undercovers"]) and not r["clipped"]
                if head is None and r["q"] > 0.5:
                    head = r
                side = ("up" if r["q"] > 0.5 else
                        "low" if r["q"] < 0.5 else "med")
                under = ("-" if r["undercovers"] is None
                         else ("YES" if r["undercovers"] else "no"))
                say(f"{n_rows:>5} {r['target']:>8} {r['n']:>5} {r['q']:>6.3f} "
                    f"{r['virtual']:>9.3f} {str(r['required'] or '-'):>4} {side:>6} "
                    f"{under:>7} {r['resolved']:>+11.6f} "
                    f"{r['clip_lo']:>+11.6f} {r['clip_hi']:>+11.6f} "
                    f"{r['stored']:>+11.6f} {'YES' if r['clipped'] else 'no':>6}")
        say("")

    say("=" * 104)
    say("WHAT THIS SETTLES AND WHAT IT DOES NOT")
    say("=" * 104)
    # One key per line, fixed prefixes: a macro is read off these, and a summary
    # spread over two lines cannot be parsed without guessing.
    for key, value in (("cells fired", fired),
                       ("cells carrying a one-sided claim", comparable),
                       ("cells undercovering in their own direction", short),
                       ("cells where the clip overrode the resolved value", clip_bound),
                       ("cells undercovering with no clip override", decisive),
                       ("level-2 examples the block requires", 128),
                       ("level-2 examples at 1000 rows", skipped_n),
                       ("headline cell level-2 size", head["n"]),
                       ("headline cell level", head["q"]),
                       ("headline cell virtual index", f"{head['virtual']:.3f}"),
                       ("headline cell required rank", head["required"]),
                       ("reconstruction residual, worst", f"{worst:.3e}")):
        say(f"  {key:<50}: {value}")
    say("")
    say("The headline cell is the upper quantile at the smallest level-2 size that")
    say("clears the floor. It is one cell of the table above and is named here so a")
    say("reader quoting it does not have to pick one.")
    say("")
    say("The site resolves a level the way the audit describes: the level goes to")
    say("numpy unchanged and lands between order statistics, short of the rank a")
    say("one-sided guarantee at that level requires. What does NOT follow is a")
    say("coverage consequence. Where the clip binds, the value the library keeps is a")
    say("coherence bound and not the quantile, so the rank decided nothing, and a")
    say("probe that reported only realised coverage here would read as 'no defect'")
    say("while the resolution is still wrong. Both columns are printed for that")
    say("reason. Below 128 level-2 examples the block never runs, and the default")
    say("calibration sizes put a 1000-row dataset there.")
    say("")
    say("Not measured here: delivered interval coverage on real data, and whether the")
    say("clip binds under a caller-supplied estimator other than the default XGBoost")
    say("pair. Two generators and four dataset sizes are a sweep, not a census.")

    out = "outputs/probe_output_oos_conformal_tights.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    sys.exit(main())
