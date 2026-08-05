"""
convention_probe.py  --  run this INSTEAD OF the crepes section of library_probe_v4.py

    pip install crepes mapie scikit-learn
    python convention_probe.py 2>&1 | tee convention_output.txt

WHY v4's CREPES SECTION MUST NOT BE RUN AS WRITTEN
--------------------------------------------------
v4 identifies a branch by nearest-curve distance over the grid
(8, 18, 40, 98, 99, 200) at alpha=0.01, with n_y=99 as the sole cell
separating "correct" (quantile) from "pval_ns".

That cell does not separate them in crepes. Verified verbatim from the
installed crepes 0.9.1, src/crepes/base.py:345:

    prediction_sets = (p_values >= 1-confidence).astype(int)

and base.py:2905 (p_values_batch, smoothing=False, unbinned branch):

    p_values = np.array([[(np.sum(alphas_cal >= alphas_test[i,c])+1)/(q+1) ...

so the non-smoothed p-value is exactly j/(n_y+1). At confidence=0.99,
`1-confidence` evaluates to 0.010000000000000009, strictly greater than
1/100 = 0.01, so the extreme test point is EXCLUDED and the observed
curve collapses onto the quantile curve.

Consequence: v4's diagnose() will label crepes smoothing=False as
"(c) CORRECT quantile / returns +inf at the boundary" with
mean|obs-correct| ~ 0.0007 -- comfortably inside the 0.005 confidence
gate. crepes computes no quantile and never returns +inf. The harness
would print a confident false label into a file marked "paste verbatim".

The float sign flips with alpha:
    alpha   1-(1-alpha)              boundary shifts?
    0.20    0.19999999999999996      no
    0.10    0.09999999999999998      no
    0.05    0.050000000000000044     yes, down one
    0.01    0.010000000000000009     yes, down one
    0.005   0.0050000000000000044    yes, down one

So the two conventions ARE separable in crepes at alpha=0.10 and are
NOT separable at alpha=0.01. v4 sweeps at 0.01 and only fingerprints
at 0.10 (at n_y=40, which is not a disagreement cell).

WHERE THE CONVENTIONS ACTUALLY DISAGREE
---------------------------------------
Verified in exact rational arithmetic, not float. Two DIFFERENT claims,
only one of which is about a unique cell:

  CONVENTION disagreement -- the two rules return different coverage.
  This is the whole residue class alpha*(n_y+1) in Z, i.e.
      alpha=0.01 -> n_y in {99, 199, 299, ...}
      alpha=0.05 -> n_y in {19, 39, 59, ...}
      alpha=0.10 -> n_y in { 9, 19, 29, ...}
  "n_y=99 is the only cell that separates them" is FALSE as stated.

  VACUITY disagreement -- one rule returns everything and the other does
  not. Checked for all n_y <= 500 at five alphas: this happens at exactly
  ONE cell, n_y = 1/alpha - 1, and nowhere else. Past that cell both are
  finite and differ by 1/(n_y+1) on the residue class above.

So keep n_y=99 in the grid; it is still the only cell with the vacuity
property. Just do not claim it is the only cell where the rules differ.

WHAT THIS SCRIPT DOES
---------------------
Section A identifies the crepes branch STRUCTURALLY, from one predict_p
call, with no curve fitting and no distance threshold. Non-smoothed
conformal p-values live on the lattice {1/(n+1), 2/(n+1), ...}. A quantile
implementation cannot produce that lattice. This is decisive at every
n_y and does not depend on alpha, on the score distribution, or on any
float comparison.

Section B sweeps coverage at both alphas over grids containing each
alpha's own disagreement cells, reports predictions per branch, and
prints a PER-CELL PAIRED S.E. so an underpowered cell cannot be read as
a match. Starred (disagreement) cells get 10x the draws.

Section C separates the quantile branches by coverage at two cells: an
INTERIOR cell where (b) clip is provably identical to (c) correct, and a
VACUITY cell where it is not. Section C in the previous revision labelled
an "always take the max" arm as "clip"; at alpha=0.10, n_y=40 the
clipping level is 0.9*41/40 = 0.9225 < 1, so no clipping occurs and
branch (b) returns exactly the correct answer there. That arm measured
nothing a library can exhibit.

Section D is the MAPIE arm. It extracts the library's ACTUAL threshold
and compares it to the reference candidates including
max(minority calibration scores) -- the one comparison that separates
(b) clip from (d) no-correction, which coverage alone cannot do. It also
settles the _check_alpha_and_n_samples guard empirically.

FIXED IN THIS REVISION
----------------------
1. predicted_coverage("clip") returned n_y/(n_y+1) at every cell. Branch
   (b) clips the quantile LEVEL to 1.0, which only bites when the level
   exceeds 1, i.e. inside the vacuity region. Outside it, clip is
   byte-identical to correct. The closed form is min(k, n_y)/(n_y+1).
   As coded it was wrong on 1840 of 2500 checked cells, e.g. at
   alpha=0.01, n_y=200 it said 0.9950 where the truth is 0.9900. A
   genuine branch-(b) library probed there would have been scored as a
   mismatch against clip and possibly labelled correct.
2. Section B's bare `except Exception: obs = "--"` now prints a
   traceback on first failure instead of returning a table of dashes.
3. Section C's middle arm is named always_max, not clip, and a cell
   where clip is actually distinguishable has been added.
4. A self-check against exact rational arithmetic runs at import, so
   this class of predicted-curve error fails loudly instead of printing.
"""

import inspect
import traceback
import warnings
from fractions import Fraction
from math import ceil

import numpy as np
from sklearn.linear_model import LogisticRegression

SEED = 0
N_TEST_MINORITY = 20_000
N_CAL_DRAWS = 20
N_CAL_DRAWS_STARRED = 200        # disagreement cells carry the whole table
N_FINGERPRINT_DRAWS = 200

# each alpha gets a grid containing its own disagreement cells (n_y = -1 mod 1/alpha)
GRIDS = {
    0.01: (8, 18, 40, 98, 99, 100, 199, 200),
    0.10: (8, 9, 18, 19, 40, 98, 99, 200),
}


# ----------------------------------------------------------------- reference
def k_index(n_y, alpha):
    return int(np.ceil((n_y + 1) * (1 - alpha)))


def boundary_quantile(alpha):
    n = 1
    while k_index(n, alpha) > n:
        n += 1
    return n - 1


def reference_threshold(scores, alpha):
    """(c) correct: s_(k), or +inf when k > n_y."""
    s = np.sort(np.asarray(scores))
    k = k_index(len(s), alpha)
    return np.inf if k > len(s) else s[k - 1]


def clip_threshold(scores, alpha):
    """(b) clip: quantile LEVEL capped at 1.0 -> order-stat index min(k, n)."""
    s = np.sort(np.asarray(scores))
    return s[min(k_index(len(s), alpha), len(s)) - 1]


def disagrees(n_y, alpha):
    """True iff the quantile and non-smoothed p-value conventions differ here."""
    x = alpha * (n_y + 1)
    return abs(x - round(x)) < 1e-9


def predicted_coverage(n_y, alpha, branch):
    k = k_index(n_y, alpha)
    if branch == "correct":
        return 1.0 if k > n_y else k / (n_y + 1)
    if branch == "clip":
        # level = min(1, (1-alpha)(n_y+1)/n_y) -> order-stat index min(k, n_y).
        # Equals "correct" everywhere OUTSIDE the vacuity region k > n_y.
        return min(k, n_y) / (n_y + 1)
    if branch == "pval_ns":
        # include iff (#{cal >= s_test} + 1)/(n_y+1) >= alpha
        rank = max(int(np.ceil(alpha * (n_y + 1) - 1 - 1e-12)), 0)
        return 1.0 if rank <= 0 else 1.0 - rank / (n_y + 1)
    if branch == "pval_s":
        return 1.0 - alpha
    return None


BRANCHES = ("correct", "clip", "pval_ns", "pval_s")


# ------------------------------------------------- self-check, exact rationals
def _self_check():
    """Assert every closed form against ground truth derived in Fractions.

    Cheap, runs at import. The clip bug this replaces printed wrong numbers
    into a file whose last line says 'paste the whole output'.
    """
    for a_f in (Fraction(1, 100), Fraction(1, 20), Fraction(1, 10),
                Fraction(1, 5), Fraction(1, 200)):
        a = float(a_f)
        for n in range(1, 501):
            k = ceil((1 - a_f) * (n + 1))
            want_correct = Fraction(1) if k > n else Fraction(k, n + 1)
            want_clip = Fraction(min(k, n), n + 1)
            m = max(ceil(a_f * (n + 1)) - 1, 0)
            want_pval = Fraction(n + 1 - m, n + 1)
            assert k == k_index(n, a), (a, n, "k_index")
            for br, want in (("correct", want_correct), ("clip", want_clip),
                             ("pval_ns", want_pval)):
                got = predicted_coverage(n, a, br)
                assert abs(got - float(want)) < 1e-12, (a, n, br, got, float(want))
            assert disagrees(n, a) == (want_correct != want_pval), (a, n, "disagrees")
        # vacuity disagreement happens at exactly one cell, n_y = 1/alpha - 1
        vac = [n for n in range(1, 501)
               if (predicted_coverage(n, a, "correct") == 1.0)
               != (predicted_coverage(n, a, "pval_ns") == 1.0)]
        assert vac == [int(1 / a_f) - 1], (a, vac)


_self_check()


# ------------------------------------------------------------------ fixtures
def gen_minority(rng, n, sep=2.0, d=5):
    X = rng.normal(0, 1, size=(n, d))
    X[:, 0] += sep
    return X


def build():
    rng = np.random.default_rng(SEED)
    y = (rng.random(40_000) < 0.01).astype(int)
    X = rng.normal(0, 1, size=(40_000, 5))
    X[y == 1, 0] += 2.0
    clf = LogisticRegression(max_iter=5000).fit(X, y)
    return clf, gen_minority(rng, N_TEST_MINORITY)


def draw_calibration(n_y, draw_id, n_majority=2000):
    rng = np.random.default_rng((SEED, n_y, draw_id))
    Xmin = gen_minority(rng, n_y)
    Xmaj = rng.normal(0, 1, size=(n_majority, 5))
    X = np.vstack([Xmin, Xmaj])
    y = np.concatenate([np.ones(n_y, int), np.zeros(n_majority, int)])
    p = rng.permutation(len(y))
    return X[p], y[p]


clf, Xte_min = build()
s_te = 1.0 - clf.predict_proba(Xte_min)[:, 1]


# --------------------------------------------------------------- crepes glue
_WRAP_FALLBACK_REPORTED = False


def make_calibrated(Xcal, ycal):
    """WrapClassifier if it accepts a pre-fitted learner, else the decoupled API.

    v4 calls WrapClassifier(clf).calibrate(...) without ever calling .fit() on
    the wrapper. Verified against crepes 0.9.1: that IS accepted. The fallback
    stays for other versions, but it now announces itself the first time it
    fires instead of silently changing which code path is under test.
    """
    global _WRAP_FALLBACK_REPORTED
    from crepes import WrapClassifier
    try:
        w = WrapClassifier(clf)
        w.calibrate(Xcal, ycal, class_cond=True)
        return ("wrap", w)
    except Exception:
        if not _WRAP_FALLBACK_REPORTED:
            _WRAP_FALLBACK_REPORTED = True
            print("\n  !! WrapClassifier path failed; falling back to the decoupled")
            print("     ConformalClassifier API. This changes which code path is")
            print("     under test -- the traceback is part of the finding:")
            traceback.print_exc()
        from crepes import ConformalClassifier
        from crepes.extras import hinge
        cc = ConformalClassifier()
        cc.fit(hinge(clf.predict_proba(Xcal), clf.classes_, ycal), bins=ycal)
        return ("decoupled", cc)


def p_values(obj_kind, obj, X, smoothing):
    kind, o = obj_kind, obj
    if kind == "wrap":
        return np.asarray(o.predict_p(X, smoothing=smoothing, seed=0))
    from crepes.extras import hinge
    bins = np.ones(len(X), dtype=int)
    return np.asarray(o.predict_p(hinge(clf.predict_proba(X)), bins=bins,
                                  smoothing=smoothing, seed=0))


def minority_coverage(obj_kind, obj, alpha, smoothing):
    if obj_kind == "wrap":
        S = np.asarray(obj.predict_set(Xte_min, labels=False,
                                       confidence=1 - alpha,
                                       smoothing=smoothing, seed=0))
    else:
        from crepes.extras import hinge
        S = np.asarray(obj.predict_set(hinge(clf.predict_proba(Xte_min)),
                                       bins=np.ones(len(Xte_min), dtype=int),
                                       confidence=1 - alpha,
                                       smoothing=smoothing, seed=0))
    if S.ndim == 3:
        S = S[:, :, 0]
    return float(np.asarray(S).astype(bool)[:, 1].mean())


# ============================================================== A. structural
print("=" * 78)
print("A. STRUCTURAL BRANCH ID -- one predict_p call, no curve fitting")
print("=" * 78)
print("  Non-smoothed conformal p-values lie on the lattice {j/(n_y+1)}.")
print("  A quantile implementation cannot produce that lattice. Decisive.")

try:
    import crepes
    print(f"\n  crepes version: {getattr(crepes, '__version__', '?')}")
    from crepes.base import ConformalClassifier as _CC, p_values_batch as _pvb
    src = inspect.getsource(_CC.predict_set)
    line = next((l.strip() for l in src.splitlines() if "prediction_sets = (" in l),
                "<not found>")
    print(f"  ConformalClassifier.predict_set thresholding line, verbatim:")
    print(f"    {line}")
    ns_line = next((l.strip() for l in inspect.getsource(_pvb).splitlines()
                    if "alphas_cal >= alphas_test[i]" in l), "<not found>")
    print(f"  p_values_batch non-smoothed numerator, verbatim:")
    print(f"    {ns_line}")
    print("    -> exactly j/(n+1). The float narrative below is no longer")
    print("       conditional on an unread function body.")

    for n_y in (18, 99):
        Xcal, ycal = draw_calibration(n_y, 0)
        kind, obj = make_calibrated(Xcal, ycal)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p_ns = p_values(kind, obj, Xte_min, smoothing=False)[:, 1]
            p_sm = p_values(kind, obj, Xte_min, smoothing=True)[:, 1]
        u = np.unique(p_ns)
        on_lattice = np.allclose(u * (n_y + 1), np.round(u * (n_y + 1)), atol=1e-9)
        print(f"\n  n_y={n_y}  (api={kind})")
        print(f"    smoothing=False: {len(u):>5} distinct p-values, "
              f"min={u.min():.10f}, 1/(n_y+1)={1/(n_y+1):.10f}")
        print(f"                     on the j/(n_y+1) lattice? {on_lattice}"
              "   <-- True proves the p-value branch (e)")
        print(f"    smoothing=True : {len(np.unique(p_sm)):>5} distinct p-values"
              "   <-- continuous proves branch (f)")
        print(f"    warnings emitted during both calls: {len(w)}")
        for x in w:
            print(f"      [{x.category.__name__}] {x.message}")
        for a in (0.01, 0.10):
            thr = 1 - (1 - a)
            print(f"    alpha={a}: 1-confidence={thr!r}  min_p>=thr? "
                  f"{u.min() >= thr}   -> minority set is "
                  f"{'trivially complete' if u.min() >= thr else 'NOT complete'}")
except ImportError as e:
    print(f"\n  crepes NOT INSTALLED ({e}). Not a finding -- pip install and rerun.")
except Exception:
    traceback.print_exc()
    print("\n  -> crash. The traceback above is the issue body.")


# ================================================================== B. sweeps
print("\n" + "=" * 78)
print("B. COVERAGE SWEEPS -- each alpha on a grid containing its own")
print("   disagreement cells (n_y = -1 mod 1/alpha, marked *)")
print("=" * 78)
print(f"  draws: {N_CAL_DRAWS} per cell, {N_CAL_DRAWS_STARRED} at starred cells.")
print("  +- is the PAIRED s.e. across calibration draws. A cell whose s.e. is")
print("  not small against the correct-vs-pval_ns gap decides nothing; the")
print("  previous revision printed 20-draw means at cells needing ~150.")

_SWEEP_FAIL_REPORTED = False

for alpha in sorted(GRIDS):
    grid = GRIDS[alpha]
    print(f"\n  alpha={alpha}   quantile boundary n_y<={boundary_quantile(alpha)}   "
          f"1-confidence={1-(1-alpha)!r}")
    hdr = (f"    {'n_y':>5} | " + " | ".join(f"{b:>9}" for b in BRANCHES)
           + " | " + " | ".join(f"{s:>17}" for s in ("crepes sm=T", "crepes sm=F")))
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for n_y in grid:
        preds = " | ".join(f"{predicted_coverage(n_y, alpha, b):>9.4f}"
                           for b in BRANCHES)
        star = disagrees(n_y, alpha)
        ndraw = N_CAL_DRAWS_STARRED if star else N_CAL_DRAWS
        obs = {}
        for sm in (True, False):
            try:
                vals = []
                for d in range(ndraw):
                    kind, obj = make_calibrated(*draw_calibration(n_y, d))
                    vals.append(minority_coverage(kind, obj, alpha, sm))
                v = np.asarray(vals)
                se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
                obs[sm] = f"{v.mean():.4f}+-{se:.4f}"
            except Exception:
                if not _SWEEP_FAIL_REPORTED:
                    _SWEEP_FAIL_REPORTED = True
                    print(f"\n    !! FIRST FAILURE at n_y={n_y}, alpha={alpha}, "
                          f"smoothing={sm}. Traceback, not a dash:")
                    traceback.print_exc()
                    print()
                obs[sm] = "FAILED"
        print(f"    {n_y:>5}{' *' if star else '  '}| {preds} | "
              f"{obs[True]:>17} | {obs[False]:>17}")
    print("    * = cells where 'correct' and 'pval_ns' predict different numbers")
    print(f"        (the whole residue class alpha*(n_y+1) in Z, not just "
          f"n_y={int(round(1/alpha))-1}).")
    print("      If crepes sm=F matches 'correct' at the starred cells, that is "
          "the\n      1-confidence float artifact, NOT a quantile implementation. "
          "Section A\n      already settled which branch it is; this table only "
          "measures the size.")


# ============================================================ C. clip / nocorr
print("\n" + "=" * 78)
print("C. THRESHOLD FINGERPRINT -- (b) clip vs (c) correct vs (d) no-correction")
print("=" * 78)
print("  Coverage alone cannot separate clip from nocorr in the interior,")
print("  because clip IS correct in the interior. Two cells are needed.")
print("  The 'always_max' arm is a NULL CONTROL: no branch takes the max at an")
print("  interior cell. It is reported to show what the previous revision was")
print("  measuring when it called that arm 'clip' and quoted 25.3 s.e.")


def fingerprint(alpha, n_y, ndraws):
    rows = {"correct": [], "clip": [], "nocorr": [], "always_max": []}
    for d in range(ndraws):
        Xcal, ycal = draw_calibration(n_y, 10_000 + d)
        s = np.sort(1.0 - clf.predict_proba(Xcal[ycal == 1])[:, 1])
        rows["correct"].append((s_te <= reference_threshold(s, alpha)).mean())
        rows["clip"].append((s_te <= clip_threshold(s, alpha)).mean())
        rows["nocorr"].append((s_te <= np.quantile(s, 1 - alpha)).mean())
        rows["always_max"].append((s_te <= s.max()).mean())
    return {k: np.array(v) for k, v in rows.items()}


for alpha, n_y in ((0.10, 40), (0.10, 8)):
    k = k_index(n_y, alpha)
    region = ("INTERIOR -- k=%d <= n_y, clipping level %.4f < 1, clip==correct"
              % (k, (1 - alpha) * (n_y + 1) / n_y)) if k <= n_y else (
        "VACUITY -- k=%d > n_y, correct is +inf and clip is NOT correct" % k)
    print(f"\n  alpha={alpha}, n_y={n_y}, {N_FINGERPRINT_DRAWS} paired draws")
    print(f"    {region}")
    rows = fingerprint(alpha, n_y, N_FINGERPRINT_DRAWS)
    for kk, v in rows.items():
        pred = predicted_coverage(n_y, alpha, kk) if kk in BRANCHES else None
        tail = f"   (predicted {pred:.4f})" if pred is not None else ""
        print(f"    {kk:<11} -> minority coverage {v.mean():.4f}{tail}")
    print("    pairwise, paired s.e. across the same draws:")
    for a, b in (("correct", "clip"), ("correct", "nocorr"), ("clip", "nocorr"),
                 ("correct", "always_max")):
        diff = rows[a] - rows[b]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        z = abs(diff.mean()) / se if se > 0 else float("inf")
        note = ""
        if (a, b) == ("correct", "clip") and k <= n_y:
            note = "   <-- identical by construction here, not a fingerprint"
        if b == "always_max":
            note = "   <-- NULL CONTROL, no branch does this"
        print(f"      {a:>7} - {b:<11} gap={diff.mean():+.4f}  s.e.={se:.4f}"
              f"  -> {z:5.1f} s.e.{note}")
    if k <= n_y:
        d = rows["correct"] - rows["nocorr"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"    LOAD-BEARING NUMBER at this cell: correct-nocorr = "
              f"{abs(d.mean())/se:.1f} s.e. Quote this, not the null control.")

print(f"\n  Test-set MC s.e. is only {np.sqrt(0.9*0.1/N_TEST_MINORITY):.4f}; quoting")
print("  that as the denominator overstates resolution ~5x. Use the paired s.e.")


# ================================================================== D. MAPIE
print("\n" + "=" * 78)
print("D. MAPIE -- threshold extraction, the only test that separates (b)/(d)")
print("=" * 78)
print("  MAPIE v1 exposes no first-class class-conditional classifier;")
print("  MondrianCP re-integration is open. Conformalizing on class-1 rows only")
print("  reproduces the classwise score exactly, so this probes the SHARED")
print("  QUANTILE HELPER that MondrianCP would inherit. Say that in the issue.")

ALPHA_MAPIE = 0.10


def threshold_candidates(s, alpha):
    """Every threshold a plausible implementation could return, on score scale."""
    s = np.sort(np.asarray(s))
    return {
        "correct   (c)": reference_threshold(s, alpha),
        "clip      (b)": clip_threshold(s, alpha),
        "nocorr-lin(d)": float(np.quantile(s, 1 - alpha)),
        "nocorr-hi (d)": float(np.quantile(s, 1 - alpha, method="higher")),
        "always_max   ": float(s.max()),
    }


def find_thresholds(obj, depth=4, _seen=None):
    """Walk a fitted object for scalars/small arrays that could be the quantile."""
    out = {}
    if _seen is None:
        _seen = set()
    if depth < 0 or id(obj) in _seen:
        return out
    _seen.add(id(obj))
    for name, val in list(getattr(obj, "__dict__", {}).items()):
        if name.startswith("__"):
            continue
        if isinstance(val, (float, np.floating)):
            out[name] = float(val)
        elif isinstance(val, np.ndarray) and val.dtype.kind == "f" and val.size <= 8:
            out[name] = val.ravel().tolist()
        elif hasattr(val, "__dict__"):
            for k, v in find_thresholds(val, depth - 1, _seen).items():
                out[f"{name}.{k}"] = v
    return out


def probe_mapie():
    import mapie
    from mapie.classification import SplitConformalClassifier
    from mapie.utils import _check_alpha_and_n_samples, _compute_quantiles
    print(f"\n  version: {getattr(mapie, '__version__', '?')}")

    print("\n  --- the guard, verbatim from mapie/utils.py ---")
    guard = inspect.getsource(_check_alpha_and_n_samples)
    for l in guard.splitlines():
        if "if n <" in l or "raise ValueError" in l:
            print(f"    {l.strip()}")
    print("    -> refuses n < max(1/alpha, 1/(1-alpha)). At alpha=0.01 that is")
    print("       n < 100. The quantile rule needs k=ceil((1-a)(n+1)) <= n, i.e.")
    print("       n >= 1/a - 1 = 99. n_y=99 admits a FINITE threshold and is")
    print("       refused: the guard is exactly one sample too strict.")

    print("\n  --- guard probe, alpha=0.01, conformalize on class-1 rows only ---")
    for n_y in (98, 99, 100):
        Xcal, ycal = draw_calibration(n_y, 0)
        m = ycal == 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scc = SplitConformalClassifier(estimator=clf, confidence_level=0.99,
                                               prefit=True)
                scc.conformalize(Xcal[m], ycal[m])
                scc.predict_set(Xte_min)
            k = k_index(n_y, 0.01)
            print(f"    n_y={n_y:<4} ACCEPTED   (k={k}, finite threshold: {k <= n_y})")
        except Exception as e:
            k = k_index(n_y, 0.01)
            print(f"    n_y={n_y:<4} {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:70]}")
            print(f"           k={k} <= n_y={n_y}? {k <= n_y}"
                  f"{'   <-- REFUSED A COMPUTABLE CELL' if k <= n_y else ''}")

    print(f"\n  --- coverage + THRESHOLD at alpha={ALPHA_MAPIE} ---")
    print(f"    coverage averaged over {N_CAL_DRAWS} draws +- paired s.e.; the")
    print("    threshold match is exact and comes from draw 0.")
    print(f"    {'n_y':>5} | {'observed':>17} | " +
          " | ".join(f"{b:>9}" for b in BRANCHES) + " | threshold matches")
    print("    " + "-" * 100)

    def mapie_fit(Xcal, ycal):
        m = ycal == 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scc = SplitConformalClassifier(
                estimator=clf, confidence_level=1 - ALPHA_MAPIE, prefit=True)
            scc.conformalize(Xcal[m], ycal[m])
            out = scc.predict_set(Xte_min)
        S = np.asarray(out[1] if isinstance(out, tuple) else out)
        if S.ndim == 3:
            S = S[:, :, 0]
        return scc, float(S.astype(bool)[:, 1].mean())

    for n_y in (18, 19, 40, 98, 99, 200):
        try:
            covs = []
            for d in range(N_CAL_DRAWS):
                scc, c = mapie_fit(*draw_calibration(n_y, d))
                covs.append(c)
                if d == 0:
                    scc0 = scc
            v = np.asarray(covs)
            se = v.std(ddof=1) / np.sqrt(len(v))
            Xcal, ycal = draw_calibration(n_y, 0)
            s = np.sort(1.0 - clf.predict_proba(Xcal[ycal == 1])[:, 1])
            cands = threshold_candidates(s, ALPHA_MAPIE)
            found = find_thresholds(scc0)
            lab = set()
            for nm, val in found.items():
                for cname, c in cands.items():
                    if np.isfinite(c) and any(np.isclose(x, c, atol=1e-12)
                                              for x in np.ravel(val)):
                        lab.add(cname.strip())
            preds = " | ".join(f"{predicted_coverage(n_y, ALPHA_MAPIE, b):>9.4f}"
                               for b in BRANCHES)
            print(f"    {n_y:>5} | {v.mean():.4f}+-{se:.4f} | {preds} | "
                  f"{', '.join(sorted(lab)) or 'NONE -- see off-by-one sweep'}")
            if n_y in (40, 99):
                tag = "(b)/(d) separator" if n_y == 40 else "the residue-class cell"
                print(f"\n      threshold detail at n_y={n_y}  [{tag}]:")
                for cname, c in cands.items():
                    print(f"        candidate {cname} = {c:.12f}")
                for nm, val in sorted(found.items()):
                    if "quantiles_" in nm:
                        print(f"        fitted    {nm:<42} = {val}")
                print()
        except Exception as e:
            print(f"    {n_y:>5} | {type(e).__name__}: {str(e).splitlines()[0][:60]}")

    print("  --- the shared helper, called directly ---")
    print(f"    {inspect.signature(_compute_quantiles)}")
    src_q = next((l.strip() for l in inspect.getsource(_compute_quantiles).splitlines()
                  if "((n + 1)" in l), "<not found>")
    print(f"    level, verbatim: np.quantile(vector, {src_q} method='higher')")
    print("    np.quantile's virtual index is q*(n-1); the conformal index is")
    print("    k=ceil((1-a)(n+1)) on n+1 slots. Those agree only when (1-a)(n+1)")
    print("    is NOT an integer. Swept below.")
    for n_y in (18, 40, 99):
        Xcal, ycal = draw_calibration(n_y, 0)
        s = np.sort(1.0 - clf.predict_proba(Xcal[ycal == 1])[:, 1])
        try:
            q = float(np.ravel(_compute_quantiles(s, np.array([ALPHA_MAPIE])))[0])
            cands = threshold_candidates(s, ALPHA_MAPIE)
            match = [c for c, v in cands.items()
                     if np.isfinite(v) and np.isclose(q, v, atol=1e-12)]
            idx = int(np.searchsorted(s, q)) + 1
            print(f"    n_y={n_y:<4} _compute_quantiles={q:.12f}  order stat "
                  f"{idx} of {n_y} (correct k={k_index(n_y, ALPHA_MAPIE)})  == "
                  f"{', '.join(match) or 'NONE of the candidates'}")
        except Exception as e:
            print(f"    n_y={n_y:<4} {type(e).__name__}: {str(e).splitlines()[0][:60]}")

    print("\n  --- OFF-BY-ONE SWEEP: _compute_quantiles vs the conformal k-th ---")
    print("    synthetic scores, so this is a property of the helper, not the fixture")
    rng = np.random.default_rng(0)
    mismatch, agree = [], 0
    for a_f in (Fraction(1, 10), Fraction(1, 20), Fraction(1, 100)):
        a = float(a_f)
        for n in range(2, 401):
            k = ceil((1 - a_f) * (n + 1))
            if k > n:
                continue                      # vacuity region; the guard refuses it
            s = np.sort(rng.random(n))
            q = float(np.ravel(_compute_quantiles(s, np.array([a])))[0])
            if abs(q - s[k - 1]) < 1e-15:
                agree += 1
            else:
                mismatch.append((a_f, n, k, int(np.searchsorted(s, q)) + 1))
    on_class = all((m[0] * (m[1] + 1)).denominator == 1 for m in mismatch)
    plus_one = all(m[3] == m[2] + 1 for m in mismatch)
    resid = {(m[0], m[1]) for m in mismatch}
    allres = {(a, n) for a in (Fraction(1, 10), Fraction(1, 20), Fraction(1, 100))
              for n in range(2, 401)
              if ceil((1 - a) * (n + 1)) <= n and (a * (n + 1)).denominator == 1}
    print(f"    agree at {agree} cells, mismatch at {len(mismatch)}")
    print(f"    every mismatch is exactly +1 order statistic?      {plus_one}")
    print(f"    every mismatch lies on the class alpha*(n+1) in Z? {on_class}")
    print(f"    every such cell is a mismatch?                     {resid == allres}")
    print(f"      exceptions: {sorted((str(a), n) for a, n in (allres - resid))}")
    print("      -> those are exactly n = 1/alpha - 1, where (1-a)(n+1) = n and the")
    print("         two index conventions coincide. That cell is ALSO the one the")
    print("         guard above refuses, so it is unreachable in practice anyway.")
    print("\n    FINDING: MAPIE returns the (k+1)-th order statistic instead of the")
    print("    k-th at exactly the cells alpha*(n+1) in Z with n > 1/alpha - 1.")
    print("    Direction is CONSERVATIVE: coverage (k+1)/(n+1), not k/(n+1). This is")
    print("    over-coverage by 1/(n+1), NOT a validity bug -- say that in sentence one.")
    print("    Same residue class that governs the crepes convention disagreement.")


try:
    probe_mapie()
except ImportError as e:
    print(f"\n  MAPIE NOT INSTALLED ({e}). Not a finding -- pip install and rerun.")
except Exception:
    traceback.print_exc()
    print("\n  -> crash. The traceback above is the issue body.")

print("\n" + "=" * 78)
print("Paste the whole output, including Section A.")
print("=" * 78)
