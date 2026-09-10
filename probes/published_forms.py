#!/usr/bin/env python3
"""W16: how the primary sources spell the correction, read from the sources.

Why this probe exists
---------------------
Both manuscripts say the correction has more than one published form, and that
different sources state it as a rank, as an un-ceiled level and as a ceiled one.
Until this probe, nothing in the artifact had read those sources. The sentence
rested on one review pass's summary of them, and a clause of that summary was
wrong: it reported one author's released code as passing a ceiled level to a
rounding definition, when the code selects an index directly and calls no
quantile function at all. A claim about a third party's code needs the file and
the line in a committed output, so this fetches each source, records its digest,
and prints the line each claim rests on with its page or line number.

What is measured, and what is only quoted
-----------------------------------------
Block (ii) QUOTES. It fetches four sources, extracts their text and prints the
matched line verbatim with its location. Nothing is interpreted there.

Block (iii) EXECUTES. The arithmetic in those lines is transcribed into five
expressions and each is evaluated over every calibration size to n = 400 at
alpha = 1/10, then handed to numpy on scores 1..n so the rank that comes back is
read rather than derived. That is what separates the forms: two of them name a
rank, one names a level below the rank, one names the level that reaches it, and
one -- a ceiling taken over the whole ratio rather than over the numerator --
resolves to the level 1.0 wherever the requirement is reachable at all, which
hands back the largest calibration score.

Reproducing it
--------------
Needs the network and `pypdf`; no audited library. The sources are cached under
`probes/.sources/`, which is gitignored: they are third-party material, read and
never redistributed. A source that has been reposted changes its digest and the
committed output stops matching, which is the intended failure -- the quotes are
pinned to the bytes they were read from.

    ../.venv-real/bin/python probes/published_forms.py
"""

import hashlib
import math
import os
import re
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".sources")
OUT = os.path.join(HERE, "..", "outputs", "probe_output_published_forms.txt")

ALPHA = 1 / 10
NMAX = 400

# Each source: where it came from, and the lines the manuscripts rest on. The
# patterns are deliberately short and structural -- a longer one would go stale on
# a whitespace change in the extractor rather than on a change in the source.
SOURCES = (
    {
        "key": "lei2018",
        "kind": "pdf",
        "file": "lei2018.pdf",
        "url": "https://arxiv.org/pdf/1604.04173",
        "what": "Lei, G'Sell, Rinaldo, Tibshirani, Wasserman, arXiv:1604.04173",
        "quotes": (("Algorithm 2, split conformal: stated as a RANK",
                    r"kth smallest value"),),
    },
    {
        "key": "romano2019",
        "kind": "pdf",
        "file": "romano2019.pdf",
        "url": "https://arxiv.org/pdf/1905.03222",
        "what": "Romano, Patterson, Candes, arXiv:1905.03222",
        "quotes": (("equation (11): stated as an un-ceiled LEVEL",
                    r"-th empirical quantile of\s*\{Ei"),),
    },
    {
        "key": "romano2019-code",
        "kind": "text",
        "file": "cqr-nonconformist-nc.py",
        "url": "https://raw.githubusercontent.com/yromano/cqr/master/"
               "nonconformist/nc.py",
        "what": "yromano/cqr, nonconformist/nc.py, the released code",
        "quotes": (("class QuantileRegErrFunc", r"^class QuantileRegErrFunc"),
                   ("its apply_inverse selects an INDEX, no quantile call",
                    r"index = int\(np\.ceil\(\(1 - significance\) \* ")),
    },
    {
        "key": "angelopoulos2023",
        "kind": "pdf",
        "file": "angelopoulos2023.pdf",
        "url": "https://arxiv.org/pdf/2107.07511",
        "what": "Angelopoulos and Bates, arXiv:2107.07511",
        "quotes": (("prose: stated as a CEILED level",
                    r"empirical quantile of s1,\s*\.\.\.,\s*s\s*n,\s*where"),
                   ("body code: the ceiling over the NUMERATOR",
                    r"np\.quantile\(cal_scores, np\.ceil"),
                   ("appendix code: the ceiling over the WHOLE RATIO",
                    r"np\.quantile\(calib_scores, np\.ceil")),
    },
)

# The arithmetic of those lines, transcribed. Each returns a LEVEL or None where
# the line names a rank directly; the rank column is then read off numpy.
FORMS = (
    ("rank, Lei Algorithm 2 at |I2| = n",
     "ceil((n + 1)(1 - alpha))", None),
    ("rank, Romano's code, its index made 1-based",
     "int(ceil((1 - alpha)(n + 1))) - 1, plus 1", None),
    ("level, Romano's text",
     "(1 - alpha)(1 + 1/n)", lambda n: (1 - ALPHA) * (1 + 1 / n)),
    ("level, Angelopoulos and Bates body",
     "ceil((n + 1)(1 - alpha)) / n", lambda n: math.ceil((n + 1) * (1 - ALPHA)) / n),
    ("level, Angelopoulos and Bates appendix",
     "ceil((n + 1)(1 - alpha) / n)", lambda n: math.ceil((n + 1) * (1 - ALPHA) / n)),
)

LINES = []
FORM_COUNTS = []


def say(s=""):
    print(s)
    LINES.append(s)


def fetch(src):
    """The source bytes, cached. Returns (bytes, sha256, from_cache)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, src["file"])
    cached = os.path.exists(path)
    if not cached:
        req = urllib.request.Request(
            src["url"], headers={"User-Agent": "conformal-coverage-audit "
                                               "(citation check)"})
        with urllib.request.urlopen(req, timeout=120) as fh:
            blob = fh.read()
        with open(path, "wb") as fh:
            fh.write(blob)
    with open(path, "rb") as fh:
        blob = fh.read()
    return blob, hashlib.sha256(blob).hexdigest(), cached


def located_lines(src, path):
    """(location label, line) for every line of the source, in order."""
    if src["kind"] == "text":
        with open(path, encoding="utf-8", errors="replace") as fh:
            return [(f"line {i}", ln.rstrip())
                    for i, ln in enumerate(fh.read().split("\n"), 1)]
    import pypdf
    out = []
    for pno, page in enumerate(pypdf.PdfReader(path).pages, 1):
        for ln in (page.extract_text() or "").split("\n"):
            out.append((f"page {pno}", ln.rstrip()))
    return out


def rank_from_numpy(n, level):
    """The rank numpy's `higher` returns at `level` on tie-free scores 1..n.

    Scores are 1..n, so the value that comes back IS its own rank and nothing is
    derived. Levels above 1 are what a mis-parenthesised ceiling produces, and
    numpy raises on them; the caller wants that reported, not caught.
    """
    return int(np.quantile(np.arange(1, n + 1, dtype=float), level,
                           method="higher"))


def self_check():
    """The two facts block (iii) rests on, each with an input that breaks it."""
    # (1) The appendix form's ceiling is taken over a ratio that is at most 1 at
    #     every feasible size, so it resolves to the level 1.0 there and to more
    #     than 1.0 below the floor. Checked at the boundary, not at round sizes:
    #     n = 1/alpha - 1 is the first feasible size and the ratio is exactly 1
    #     there, which is where an off-by-one would sit.
    first = int(round(1 / ALPHA - 1))
    assert math.ceil((first + 1) * (1 - ALPHA) / first) == 1, first
    assert math.ceil((first - 1 + 1) * (1 - ALPHA) / (first - 1)) == 2, first
    # (2) The body form and the appendix form are different levels wherever the
    #     first is not 1. If a later edit loses a bracket this fires.
    n = 50
    assert (math.ceil((n + 1) * (1 - ALPHA)) / n
            != math.ceil((n + 1) * (1 - ALPHA) / n)), n
    # (3) The instrument. Scores 1..n equal their own ranks, so what `higher`
    #     returns IS the rank. numpy's virtual index for the rounding definitions
    #     is q(n-1) and not qn, so a level of k/n does NOT come back as k: an
    #     earlier version of this check assumed it did and numpy said otherwise.
    #     Nothing below converts a level into a rank by hand.
    assert rank_from_numpy(50, 1.0) == 50
    assert rank_from_numpy(50, 0.0) == 1
    assert rank_from_numpy(50, 46 / 50) == 46 + 1


self_check()


def main():
    say("=" * 104)
    say("W16  THE PUBLISHED CORRECTION, READ FROM THE SOURCES THAT PUBLISH IT")
    say("=" * 104)
    say("")
    say("CLAIM  The correction has more than one published form. Some sources")
    say("       state it as a RANK, one states it as an un-ceiled LEVEL, and one")
    say("       states it as a ceiled level -- twice, with the ceiling in two")
    say("       different places. Nothing here is read off a secondary summary.")
    say(f"alpha {ALPHA}   sizes 2..{NMAX}   numpy {np.__version__}")
    say("")

    say("-" * 104)
    say("(i) SOURCES, AS FETCHED. The digest pins the quotes below to the bytes")
    say("    they were read from; a reposted source changes it.")
    say("-" * 104)
    say(f"{'key':<20}{'bytes':>10}  {'sha256':<64}")
    texts = {}
    for src in SOURCES:
        blob, digest, cached = fetch(src)
        texts[src["key"]] = located_lines(src, os.path.join(CACHE, src["file"]))
        say(f"{src['key']:<20}{len(blob):>10}  {digest}")
        say(f"{'':<20}{'':>10}  {src['what']}")
        say(f"{'':<20}{'':>10}  {src['url']}")
    say("")

    say("-" * 104)
    say("(ii) THE LINES. Printed verbatim from the extracted text, with the page")
    say("     or line they sit on. No interpretation in this block.")
    say("-" * 104)
    found = 0
    for src in SOURCES:
        for label, pat in src["quotes"]:
            hits = [(loc, ln) for loc, ln in texts[src["key"]]
                    if re.search(pat, ln)]
            assert hits, (
                f"{src['key']}: /{pat}/ matched nothing. The source has moved or "
                f"the extractor has changed; do not soften the pattern until the "
                f"digest above has been compared with the recorded one")
            loc, ln = hits[0]
            found += 1
            say(f"{src['key']:<20}{loc:<10}{label}")
            say(f"    {ln.strip()}")
            if len(hits) > 1:
                say(f"    (also at {', '.join(l for l, _ in hits[1:5])})")
            say("")

    say("-" * 104)
    say("(iii) THE SAME ARITHMETIC, EXECUTED. Each spelling evaluated at every")
    say("      size, and the rank read back from numpy on scores 1..n rather than")
    say("      derived. k* is the required rank ceil((n+1)(1-alpha)).")
    say("-" * 104)
    feasible = [n for n in range(2, NMAX + 1)
                if math.ceil((n + 1) * (1 - ALPHA)) <= n]
    first = feasible[0]
    say(f"    feasible sizes (k* <= n): {len(feasible)}, from n = {first} to "
        f"n = {feasible[-1]}")
    say("")
    say("      A rank spelling is compared with k* directly. A level spelling is")
    say("      passed to numpy's `higher`, which is what both code lines pass it")
    say("      to, and the rank that comes back is scored against k*.")
    say("")
    say(f"{'spelling':<46}{'n=9':>9}{'n=50':>9}{'n=400':>9}"
        f"{'exact':>8}{'over':>7}{'short':>7}")
    show = (9, 50, 400)
    for label, expr, level in FORMS:
        kstar = [math.ceil((n + 1) * (1 - ALPHA)) for n in show]
        if level is None:
            vals = [f"{k:d}" for k in kstar]
            got = [math.ceil((n + 1) * (1 - ALPHA)) for n in feasible]
        else:
            vals = [f"{level(n):.4f}" for n in show]
            got = [rank_from_numpy(n, level(n)) for n in feasible]
        want = [math.ceil((n + 1) * (1 - ALPHA)) for n in feasible]
        ex = sum(1 for g, w in zip(got, want) if g == w)
        ov = sum(1 for g, w in zip(got, want) if g > w)
        sh = sum(1 for g, w in zip(got, want) if g < w)
        say(f"{label:<46}" + "".join(f"{v:>9}" for v in vals)
            + f"{ex:>8}{ov:>7}{sh:>7}")
        FORM_COUNTS.append((label, ex, ov, sh))
    say("")
    say(f"    counts are over the {len(feasible)} feasible sizes. The transcribed")
    say("    arithmetic, spelling by spelling:")
    for label, expr, _ in FORMS:
        say(f"      {label:<48}{expr}")
    say("")
    say("    Romano's index is 0-based; the row adds the one back, which is what")
    say("    subscripting a sorted array does. It is listed to show that the")
    say("    released code selects an index and calls no quantile function, which")
    say("    is the clause a review pass had backwards.")
    say("")
    say("    The last two rows share their counts and are not the same call. The")
    say("    body form is exact wherever k* = n and over by a rank elsewhere; the")
    say("    appendix form is the level 1.0 everywhere, so it returns n, which")
    say("    equals k* at the same sizes and for a different reason. The level")
    say("    columns are what tell them apart.")
    say("")

    # the appendix form, on its own, because it is the one that surprises
    ap = FORMS[-1][2]
    sat = [n for n in feasible if ap(n) == 1.0]
    over = [n for n in range(2, NMAX + 1) if ap(n) > 1.0]
    ranks = {rank_from_numpy(n, 1.0) - n for n in sat}
    say("-" * 104)
    say("(iv) THE APPENDIX SPELLING, ALONE. The ceiling wraps the ratio, so what")
    say("     is rounded up is a number at most 1 rather than a rank.")
    say("-" * 104)
    say(f"    resolves to the level 1.0 at {len(sat)} of the {len(feasible)} "
        f"feasible sizes, that is at all of them")
    say(f"    exceeds 1.0, where numpy raises, at {len(over)} sizes, all of them "
        f"below the feasibility floor n = {first}")
    say(f"    at level 1.0 the rank returned is n at every one of those sizes: "
        f"offsets from n seen = {sorted(ranks)}")
    say("    A threshold at rank n covers n/(n+1) whatever level was requested,")
    say("    so nothing here is invalid; it is the widest interval this many")
    say("    scores can produce. It is also what a helper returns when it has")
    say("    clamped a level it could not honour, a different event with the same")
    say("    output.")
    say("")

    say("=" * 104)
    say(f"MACHINE pf_sources={len(SOURCES)} pf_quotes={found} "
        f"pf_forms={len(FORMS)} pf_feasible={len(feasible)} "
        f"pf_appendix_saturates={len(sat)}")
    for label, ex, ov, sh in FORM_COUNTS:
        say(f"MACHINE-FORM {label} | exact={ex} over={ov} short={sh}")
    say("=" * 104)

    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwritten -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
