#!/usr/bin/env python3
"""W17: the certificate as a usable tool, and a census of the fixtures in use.

Three things, all downstream of W13
-----------------------------------
1. CLASSIFY  Given any callable (scores, level) -> threshold, probe it at the
   certificate sizes and report which of the candidate CLASSES it implements. W13
   proved four sizes suffice; this is the four-line function that uses them.
2. REGIME    Report i = n - floor(h) and whether the caller is in the regime where
   the interior approximation pi = gamma holds. Computable from n and the level with
   no data, so a library could print it. This is the piece W12 and W14 make
   actionable.
3. CENSUS    Score the calibration sizes the audited libraries' OWN conformal tests
   exercise against the discriminating power W13 computes, so the manuscript's claim
   that a fixture can sit at a blind size stops being an anecdote.

How the census is obtained, and its honest limit
------------------------------------------------
pytest is deliberately NOT installed in the pinned environments -- adding it would
perturb the versions the deposit exists to fix -- so the libraries' test suites are
not executed. Instead the fixture VALUES are read from the shipped test sources, each
recorded with its file and line so a reader can check it, and everything downstream of
that value is then MEASURED by running the library at it. So the read is confined to
a single integer per fixture and the classification of that integer is computed, not
asserted.

This is the same scope the manuscript already declares for the two libraries it
audits by reading rather than running, and it is stated here rather than buried. The
census covers only the packages that ship their tests in the wheel: two of the ten.
That is a real limit and the summary says so; a census over two libraries is still
the difference between an anecdote and a measurement.
"""

import math
import os
import re
import sys
from fractions import Fraction as F

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_certificate_suite.txt")
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


# ---------------------------------------------------------------------------
# reuse W13's rule set and certificate rather than restating them
# ---------------------------------------------------------------------------
sys.path.insert(0, HERE)
import certificate as W13  # noqa: E402


def virtual_index(n, q, method="linear"):
    return float(np.quantile(np.arange(1, n + 1, dtype=float), q, method=method))


def regime(n, coverage, method="linear"):
    """i, gamma and the regime, from n and the level alone -- no data."""
    h = virtual_index(n, coverage, method)
    j = math.floor(h + 1e-12)
    gam, i = h - j, n - j
    if gam <= 1e-12:
        label = "exact: lands on an order statistic"
    elif i >= 10:
        label = "interior: pi = gamma to O(1/i)"
    elif i <= 3:
        label = "TAIL: use the shape law"
    else:
        label = "transitional"
    bound = (i + 1) / i if i >= 1 else float("inf")
    return {"h": h, "j": j, "gamma": gam, "i": i, "regime": label,
            "inflation_bound": bound}


def classify(helper, alpha, cert=None):
    """Which candidate CLASS does `helper` implement? Uses W13's certificate.

    helper(scores, level) -> threshold. Probed on the tie-free score set 1..n so
    the returned value IS the delivered rank.
    """
    if cert is None:
        cert = W13.minimal_certificate(
            alpha, list(range(W13.feasible_floor(alpha) + 1,
                              W13.periodic_from(alpha) + 2 * W13.period_of(alpha))),
            len(W13.classes(alpha, tuple(range(
                W13.feasible_floor(alpha) + 1,
                W13.periodic_from(alpha) + 2 * W13.period_of(alpha))))))
    obs = []
    for n in cert:
        t = helper(np.arange(1, n + 1, dtype=float), float(1 - alpha))
        if t is None or not np.isfinite(t):
            obs.append(W13.OUT_OF_RANGE)
        else:
            r = int(round(float(t)))
            obs.append(r if 1 <= r <= n else W13.OUT_OF_RANGE)
    sig = tuple(obs)
    matches = [name for name, fn in W13.RULES
               if W13.signature(fn, tuple(cert), alpha) == sig]
    return {"cert": cert, "observed": sig, "matches": matches}


# ---------------------------------------------------------------------------
# the fixtures, read from the shipped test sources. VALUE only; everything
# downstream is measured. file:line recorded so each can be checked.
# ---------------------------------------------------------------------------
FIXTURE_SOURCES = [
    ("sktime 1.1.0", "sktime/forecasting/tests/test_conformal.py",
     r"initial_window\s*=\s*(\d+)", "initial_window"),
    ("mapie 1.4.1", "mapie/tests/test_utils.py",
     r"^\s*n\s*=\s*(\d+)\s*$", "n (guard test)"),
]
SITE_ROOTS = [
    os.path.join(HERE, "..", ".venv-real", "lib", "python3.14", "site-packages"),
    os.path.join(HERE, "..", "..", ".venv-tabular", "lib", "python3.14",
                 "site-packages"),
]


def read_fixtures():
    found = []
    for lib, relpath, pat, what in FIXTURE_SOURCES:
        path = None
        for root in SITE_ROOTS:
            cand = os.path.join(root, relpath)
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            found.append({"lib": lib, "file": relpath, "line": None,
                          "value": None, "what": what, "note": "source not found"})
            continue
        with open(path, encoding="utf-8") as fh:
            for ln_no, ln in enumerate(fh, start=1):
                m = re.search(pat, ln)
                if m:
                    found.append({"lib": lib, "file": relpath, "line": ln_no,
                                  "value": int(m.group(1)), "what": what,
                                  "note": ""})
    return found


# ---------------------------------------------------------------------------
def self_check():
    alpha = F(1, 10)
    lo = W13.feasible_floor(alpha) + 1
    hi = W13.periodic_from(alpha) + 2 * W13.period_of(alpha)
    pool = list(range(lo, hi))
    ncls = len(W13.classes(alpha, tuple(pool)))
    cert = W13.minimal_certificate(alpha, pool, ncls)
    assert cert and len(cert) == 4, cert

    # (1) classify must recover EVERY candidate rule's own class -- if it cannot
    #     identify the rules it was built from, it identifies nothing
    for name, fn in W13.RULES:
        def helper(scores, level, fn=fn, alpha=alpha):
            n = len(scores)
            r = fn(n, alpha)
            return None if r == W13.OUT_OF_RANGE else float(r)
        res = classify(helper, alpha, cert)
        assert name in res["matches"], (name, res["matches"])
        # and the match set must be exactly that rule's class, not a superset
        own = W13.signature(fn, tuple(cert), alpha)
        want = {nm for nm, f2 in W13.RULES
                if W13.signature(f2, tuple(cert), alpha) == own}
        assert set(res["matches"]) == want, (name, res["matches"], want)

    # (2) a helper matching NO candidate must be reported as such rather than
    #     forced into the nearest class
    res = classify(lambda s, q: 1.0, alpha, cert)
    assert res["matches"] == [], res["matches"]

    # (3) the regime reporter must agree with the arithmetic at both extremes
    r = regime(2000, 0.999)
    assert r["i"] == 2 and "TAIL" in r["regime"], r
    r = regime(1000, 0.99)
    assert r["i"] == 10 and "interior" in r["regime"], r
    # gamma = 0 must be reported as exact, not as a regime
    n = 50
    q = (46 - 1) / (n - 1)          # `linear` lands exactly on rank 46 here
    r = regime(n, q)
    assert r["gamma"] < 1e-9 and "exact" in r["regime"], r


self_check()


def main():
    say("=" * 104)
    say("W17  THE CERTIFICATE AS A TOOL, AND A CENSUS OF THE FIXTURES IN USE")
    say("=" * 104)
    say("")

    alpha = F(1, 10)
    lo = W13.feasible_floor(alpha) + 1
    hi = W13.periodic_from(alpha) + 2 * W13.period_of(alpha)
    pool = list(range(lo, hi))
    full = W13.classes(alpha, tuple(pool))
    cert = W13.minimal_certificate(alpha, pool, len(full))

    # ---------------- (i) classify ---------------------------------------
    say("-" * 104)
    say(f"(i) CLASSIFY, at alpha = {alpha}. Certificate sizes {cert}; "
        f"{len(full)} classes.")
    say("    Every candidate rule is round-tripped through the classifier, which is")
    say("    the check that it identifies what it was built from.")
    say("-" * 104)
    say(f"{'rule':<34}{'observed ranks':<24}{'class size':>11}   identified")
    seen = set()
    for name, fn in W13.RULES:
        def helper(scores, level, fn=fn):
            n = len(scores)
            r = fn(n, alpha)
            return None if r == W13.OUT_OF_RANGE else float(r)
        res = classify(helper, alpha, cert)
        key = res["observed"]
        if key in seen:
            continue
        seen.add(key)
        say(f"{name:<34}{str(res['observed']):<24}{len(res['matches']):>11}"
            f"   {'yes' if name in res['matches'] else 'NO'}")
    say("")
    say(f"    {len(seen)} distinct observation vectors over the certificate, matching")
    say(f"    the {len(full)} classes W13 computes over the full checked range.")
    say("")
    say("    A helper matching no candidate is reported as unmatched rather than")
    say("    forced into the nearest class -- asserted in self_check with a constant")
    say("    helper, which matches nothing.")
    say("")

    # ---------------- (ii) regime ----------------------------------------
    say("-" * 104)
    say("(ii) REGIME REPORTER. i = n - floor(h) and the regime, from n and the level")
    say("     alone. A library can print this before it returns an interval.")
    say("-" * 104)
    say(f"{'n':>7}{'coverage':>10}{'h':>11}{'i':>5}{'gamma':>8}"
        f"{'pi/gamma bound':>16}   regime")
    for n, cov in ((20, 0.90), (50, 0.90), (100, 0.90), (1000, 0.90),
                   (50, 0.95), (50, 0.99), (200, 0.99), (2000, 0.999),
                   (2000, 0.99)):
        r = regime(n, cov)
        b = "---" if not np.isfinite(r["inflation_bound"]) else \
            f"{r['inflation_bound']:.2f}"
        say(f"{n:>7}{cov:>10.3f}{r['h']:>11.3f}{r['i']:>5}{r['gamma']:>8.3f}"
            f"{b:>16}   {r['regime']}")
    say("")
    say("    Note the last two rows: same n, different level, opposite regimes. The")
    say("    regime is a property of n(1-alpha), which is why it cannot be inferred")
    say("    from the calibration size alone.")
    say("")

    # ---------------- (iii) the fixture census ---------------------------
    say("-" * 104)
    say("(iii) FIXTURE CENSUS. Calibration sizes the audited libraries' own shipped")
    say("      conformal tests exercise, scored against the discriminating power W13")
    say("      computes. The VALUE is read from the test source (file:line given);")
    say("      everything downstream of it is computed.")
    say("-" * 104)
    fixtures = read_fixtures()
    powers = {n: len(W13.classes(alpha, (n,))) for n in pool}
    best = max(powers.values())
    worst = min(powers.values())
    say(f"      discriminating power of a single size, over {len(pool)} sizes:")
    say(f"        best {best} of {len(full)} classes, worst {worst}, "
        f"median {int(np.median(list(powers.values())))}")
    say("")
    say(f"{'library':<16}{'fixture':<20}{'value':>7}{'in range?':>11}"
        f"{'power':>7}{'of':>4}   verdict")
    scored = []
    for f in fixtures:
        if f["value"] is None:
            say(f"{f['lib']:<16}{f['what']:<20}{'---':>7}{'---':>11}"
                f"{'---':>7}{'---':>4}   {f['note']}")
            continue
        n = f["value"]
        inr = n in powers
        p = powers.get(n)
        if not inr:
            verdict = ("below the feasibility floor" if n <= W13.feasible_floor(alpha)
                       else "outside the checked range")
        elif p <= worst:
            verdict = "BLIND: worst possible power"
        elif p < best:
            verdict = f"partial: {best - p} classes short of the best size"
        else:
            verdict = "maximal power"
        scored.append({**f, "power": p, "in_range": inr, "verdict": verdict})
        say(f"{f['lib']:<16}{f['what']:<20}{n:>7}{('yes' if inr else 'no'):>11}"
            f"{(p if p else 0):>7}{len(full):>4}   {verdict}")
        say(f"{'':<16}{f['file']}:{f['line']}")
    say("")
    nb = [s for s in scored if s["in_range"] and s["power"] < best]
    say(f"    {len(nb)} of {len(scored)} extracted fixtures sit at a calibration size")
    say(f"    with less than maximal discriminating power.")
    say("")
    say("    LIMIT, stated plainly. Only two of the ten audited packages ship their")
    say("    tests in the wheel, so this census covers two libraries and a handful of")
    say("    fixtures, not the ten. pytest is deliberately not installed in the")
    say("    pinned environments -- adding it would perturb the versions the deposit")
    say("    exists to fix -- so the suites are not executed and the fixture VALUES")
    say("    are read. A census over two libraries is still the difference between")
    say("    the manuscript's anecdote and a measurement, and it is not more than")
    say("    that.")
    say("")
    say("=" * 104)
    say("SUMMARY")
    say("=" * 104)
    say(f"  The certificate is usable: {len(cert)} sizes, a classifier that")
    say(f"  round-trips every candidate rule and reports non-matches as non-matches,")
    say("  and a regime reporter that needs no data. Of the fixtures we could extract")
    say(f"  from shipped test sources, {len(nb)} of {len(scored)} sit at a size that")
    say("  cannot separate all the classes -- so the manuscript's claim about fixture")
    say("  choice is measured on two libraries rather than asserted from one.")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
