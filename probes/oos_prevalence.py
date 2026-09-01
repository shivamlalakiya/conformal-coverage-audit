#!/usr/bin/env python3
"""Can the site count be automated? Calibrate the shortcut before spending it.

WHAT WAS MISSING
----------------
The out-of-census read files a verdict per package and prints no site count, on
the stated ground that the census counts one site per distinct expression and
that ruling was never made outside the ten. A referee reads that as work not
done. It is worth knowing what the work would cost, and whether the obvious
shortcut, a machine tally of the expressions, can be substituted for it.

WHAT THIS PROBE DOES
--------------------
It builds the shortcut, then measures it against a known-positive set before
letting it near an unaudited package. The census manifest carries every audited
site with a file and a line, so the enumerator can be pointed at the same ten
packages and scored on how many of those lines it recovers. That fraction is its
recall, and it decides whether any count it produces elsewhere means anything.

WHAT IT CONCLUDES
-----------------
It does not produce a prevalence, and the reason is a number rather than a
preference. The enumerator recovers only part of the census, and the sites it
loses are lost through four different mechanisms, each anchored below. A rate
built on it would be wrong by the size of that gap, in a direction that varies
by package, and would then sit in a table beside the hand-made count as though
the two were the same kind of object.

What it does produce is the candidate surface of each unaudited package, which
bounds nothing on its own but does show where a hand read would have to go, and
one finding that is not about the instrument at all: a package already filed as
carrying one site has several more expressions of the same shape, none of them
ruled on.

    python probes/oos_prevalence.py [--root DIR]
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import helper_census as HC  # noqa: E402

OUT = os.path.join(HERE, "..", "outputs", "probe_output_oos_prevalence.txt")

# An expression that could put a level onto a position. Every ordering call the
# audit has met, plus the index-building shapes it has met.
CALL = re.compile(
    r"(np\.quantile|numpy\.quantile|np\.percentile|numpy\.percentile|jnp\.quantile"
    r"|torch\.quantile|\.quantile\(|np\.sort\(|np\.argsort\(|np\.partition\("
    r"|\.argmax\(|\.argmin\(|searchsorted|np\.ceil|np\.floor|math\.ceil|math\.floor"
    r"|\bceil\s*\(|\bfloor\s*\(|int\s*\(|round\s*\()")
LEVEL = re.compile(
    r"(alpha|confidence|coverage|significance|miscoverage|nominal|error_rate"
    r"|\berror\b|\blevel\b|quantile|percentile)", re.I)
SKIPDIR = ("/test", "/tests", "/.git", "/docs", "/doc/", "/examples", "/example",
           "/benchmark", "/.github", "/build/", "/dist/")

# How far above and below a call the level may be bound before the enumerator
# stops seeing it. Widening this does not rescue the misses below -- they fail
# for reasons a window cannot fix -- and it does raise the noise.
UP, DOWN = 7, 3

# The unaudited side of the download denominator, and where each unpacks.
OOS = ["fortuna", "uncertainty-toolbox", "quantile-forest", "conformal-tights",
       "crepes-weighted", "venn-abers", "nixtla", "pytorch-forecasting",
       "gluonts", "prophet", "skforecast", "orbit-ml"]

# Directory names under the source root, where they differ from the distribution.
DIRNAME = {"conformal-tights": "conformal_tights", "crepes-weighted": "crepes_weighted",
           "venn-abers": "venn_abers", "uncertainty-toolbox": "uncertainty_toolbox",
           "quantile-forest": "quantile_forest", "pytorch-forecasting": "pytorch_forecasting",
           "orbit-ml": "orbit_ml"}

# The four ways a real site escapes a line-and-window match, each named at the
# census anchor that demonstrates it. These are read out of the recall run
# rather than asserted: if a mechanism stops applying, the run stops reporting it.
MECHANISM = {
    ("crepes", 340): ("comparison", "the level meets the score array through `>=`, "
                      "not through an index or an ordering call"),
    ("lac.py", 156): ("delegation", "the level is passed onward and the position "
                      "is chosen inside the callee"),
    ("jackknife.py", 97): ("transformation", "the level is halved into a local name "
                           "and the ordering happens against that name later"),
    ("models.py", 127): ("plumbing", "the level becomes a list of cut points, which "
                         "then reaches the ordering call as an argument"),
}

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def enumerate_pkg(root):
    """Every (relpath, line, text) whose statement could resolve a level."""
    out = []
    if not os.path.isdir(root):
        return None
    for dp, _, fns in os.walk(root):
        if any(x in dp + "/" for x in SKIPDIR):
            continue
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            try:
                lines = open(p, errors="replace").read().splitlines()
            except Exception:
                continue
            for i, ln in enumerate(lines, 1):
                if ln.strip().startswith("#") or not CALL.search(ln):
                    continue
                lo, hi = max(0, i - UP), min(len(lines), i + DOWN)
                if LEVEL.search("\n".join(lines[lo:hi])):
                    out.append((os.path.relpath(p, root), i, ln.strip()))
    return out


def self_check():
    """Failing inputs for the enumerator, before its recall means anything.

    A recall figure is only interesting if the instrument works at all. These
    are the two directions it could be broken in: silent on a plain site, or
    loud on a line that resolves nothing.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "a.py")
        open(good, "w").write("alpha = 0.1\nq = np.quantile(scores, 1 - alpha)\n")
        assert enumerate_pkg(d), "the enumerator misses a plain resolution site"
        os.remove(good)
        open(os.path.join(d, "b.py"), "w").write("x = 1\ny = x + 2\n")
        assert not enumerate_pkg(d), "the enumerator fires on arithmetic"
        os.remove(os.path.join(d, "b.py"))
        # an ordering call with no level anywhere near it is not a site
        open(os.path.join(d, "c.py"), "w").write("z = np.sort(values)\n")
        assert not enumerate_pkg(d), (
            "the enumerator counts an ordering call carrying no level; its "
            "candidate counts would then be a count of every sort in the tree")
    assert [s for s in HC.MANIFEST if s["output"]], "the census manifest carries no sites"
    return True


self_check()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "..", "..", "cp-src"))
    args = ap.parse_args()
    root = args.root

    say("=" * 104)
    say("OUT-OF-CENSUS PREVALENCE -- what the shortcut recovers, measured first")
    say("=" * 104)
    say("self_check() passed at import: the enumerator finds a plain site, stays")
    say("quiet on arithmetic, and does not count an ordering call that carries no")
    say("level.")
    say("")
    say("A prevalence needs a numerator and a denominator built the same way. The")
    say("census's numerator is a hand read, one expression at a time. What follows")
    say("asks whether a mechanical read can stand in for it.")
    say("")

    # ---- calibration ----------------------------------------------------
    say("-" * 104)
    say("CALIBRATION -- the enumerator against the census, which is known-positive")
    say("-" * 104)
    sites = [s for s in HC.MANIFEST if s["output"]]
    per_lib = {}
    for s in sites:
        per_lib.setdefault(s["lib"].split()[0], []).append(s)
    say(f"{'package':<20}{'candidates':>12}{'census sites':>14}{'recovered':>11}")
    found, misses, absent = 0, [], []
    for lib in sorted(per_lib):
        cand = enumerate_pkg(os.path.join(root, lib))
        if cand is None:
            absent.append(lib)
            say(f"{lib:<20}{'NOT HELD':>12}{len(per_lib[lib]):>14}{'-':>11}")
            continue
        idx = {}
        for rel, i, _ in cand:
            idx.setdefault(os.path.basename(rel), set()).add(i)
        got = 0
        for s in per_lib[lib]:
            want_f, want_l = os.path.basename(s["path"]), s["line"]
            if any(abs(i - want_l) <= 3 for i in idx.get(want_f, ())):
                got += 1
            else:
                misses.append((lib, s["path"], s["line"], s["symbol"]))
        found += got
        say(f"{lib:<20}{len(cand):>12}{len(per_lib[lib]):>14}{got:>11}")
    assert not absent, (
        f"these packages are not under {root}: {absent}. A recall figure computed "
        f"over the ones that happen to be unpacked is not a recall figure.")
    total = len(sites)
    say("")
    say(f"  census sites                 {total}")
    say(f"  recovered by the enumerator  {found}")
    say(f"  lost                         {total - found}")
    assert found < total, (
        "the enumerator now recovers every census site. That is the good outcome "
        "and it invalidates the reasoning below; re-read this probe before "
        "quoting it.")
    say("")

    # ---- why ------------------------------------------------------------
    say("-" * 104)
    say("HOW A REAL SITE ESCAPES A MECHANICAL READ")
    say("-" * 104)
    say("Each mechanism below is named at a census anchor the enumerator lost. A")
    say("wider window does not reach any of them; the level and the ordering are")
    say("not in one statement in the first place.")
    say("")
    seen = set()
    for lib, path, line, symbol in misses:
        for (key, want_l), (name, why) in MECHANISM.items():
            if want_l != line:
                continue
            if not (key in path or key == lib):
                continue
            if name in seen:
                continue
            seen.add(name)
            say(f"  {name.upper()}")
            say(f"    {lib} {path}:{line}")
            say(f"    {symbol}")
            say(f"    {why}")
            say("")
    assert len(seen) >= 4, (
        f"only {len(seen)} of the four documented mechanisms were reproduced by "
        f"this run; the paragraph naming four is stale")
    say(f"  every site the enumerator lost, in full ({len(misses)}):")
    for lib, path, line, _ in misses:
        say(f"    {lib:<16} {path}:{line}")
    say("")

    # ---- the unaudited side ---------------------------------------------
    say("-" * 104)
    say("THE UNAUDITED SIDE -- candidate surface, which is not a site count")
    say("-" * 104)
    say("Read with the instrument calibrated above. These are lines a hand read")
    say("would have to visit, not sites, and the recall figure says the list is")
    say("incomplete as well as noisy.")
    say("")
    say(f"{'package':<24}{'candidates':>12}   {'held under cp-src'}")
    oos_counts = {}
    for pkg in OOS:
        d = DIRNAME.get(pkg, pkg)
        cand = enumerate_pkg(os.path.join(root, d))
        oos_counts[pkg] = cand
        say(f"{pkg:<24}{(len(cand) if cand is not None else 'not held'):>12}")
    say("")

    # ---- the finding that is not about the instrument --------------------
    say("-" * 104)
    say("ONE PACKAGE FILED WITH A SINGLE ANCHOR HAS MORE OF THE SAME SHAPE")
    say("-" * 104)
    cand = oos_counts.get("fortuna")
    if cand:
        conf = [(f, i, t) for f, i, t in cand
                if "/conformal/" in "/" + f.replace(os.sep, "/")
                and re.search(r"quantile\(|percentile\(", t)]
        say("The out-of-census read files fortuna at one anchor, quantile.py:90,")
        say("with a byte-identical twin named beside it. Its conformal package")
        say("holds these expressions of the same shape, each reachable from a")
        say("public entry point and none of them ruled on here:")
        say("")
        for f, i, t in sorted(conf):
            say(f"    {f}:{i}")
            say(f"        {t[:96]}")
        say("")
        say(f"  expressions of the shape in fortuna/conformal   {len(conf)}")
        say("  ruled on by this probe                              0")
        assert len(conf) > 2, (
            "fortuna's conformal package no longer holds more than the filed "
            "anchor and its twin; the paragraph above is stale")
    say("")

    # ---- counts ---------------------------------------------------------
    say("=" * 104)
    say("THE COUNTS")
    say("=" * 104)
    say(f"  census sites offered              {total}")
    say(f"  enumerator recovered              {found}")
    say(f"  enumerator lost                   {total - found}")
    say(f"  loss mechanisms reproduced        {len(seen)}")
    say(f"  unaudited packages enumerated     {sum(1 for v in oos_counts.values() if v is not None)}")
    say(f"  fortuna conformal expressions     {len(conf) if cand else 0}")
    say(f"  sites ruled on outside the census 0")
    say("")
    say("=" * 104)
    say("WHAT THIS DOES NOT SETTLE, AND WHY NO RATE IS PRINTED")
    say("=" * 104)
    say("No prevalence appears anywhere above. The instrument that would have")
    say("produced one loses part of the census it was calibrated against, and the")
    say("loss is not a constant: it depends on how each package happens to move a")
    say("level from its public signature to its ordering call. A rate carrying")
    say("that error would be printed next to a count made by hand and would invite")
    say("precisely the comparison the error rules out.")
    say("")
    say("The candidate counts are not an upper bound either. They over-generate,")
    say("because most lines matching an ordering call resolve nothing, and they")
    say("under-generate, for the four reasons anchored above.")
    say("")
    say("What would produce a prevalence is the same work the census did: one")
    say("expression at a time, each ruled against the criterion and anchored, with")
    say("a second reader asked to break the ruling. This probe measures the size")
    say("of that job and does not do it.")

    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
