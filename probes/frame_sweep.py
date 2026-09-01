#!/usr/bin/env python3
"""Was the candidate list a query, or a memory? Sweep the index and find out.

WHAT WAS MISSING
----------------
The audit picked its packages by name. Nobody queried the package index, so a
distribution that no one working in this area would think of stayed outside the
study before any criterion was applied to it, and the manuscript could only say
so and move on. A stated limitation with no number attached is the weakest thing
a threats section can carry.

WHAT THIS PROBE DOES
--------------------
Two sweeps over the Python Package Index, chosen because they fail in opposite
directions.

  ARM A -- every distribution name on the index is matched against a pattern of
  conformal-prediction word stems. Complete over the index by construction: the
  full name list is downloaded and every entry is tested. A name is cheap and
  says almost nothing, so each hit is then opened through the JSON API and kept
  only where its published summary, README, declared keywords or trove
  classifiers state the statistical object in so many words. The arm is exhaustive on names and blind to a
  package whose name is a word like `river`.

  ARM B -- the head of the index ordered by downloads, every one of them opened
  and read the same way. Blind below its own download floor, and sighted where
  arm A is not: a general-purpose library announcing a conformal feature in its
  README is caught here and nowhere else.

WHAT MAKES THE RESULT READABLE
------------------------------
Both arms are turned on the audit's own candidate list first. That list is a set
of packages already known to belong, so the fraction of it each arm recovers is
the arm's recall against a known-positive set, measured rather than asserted. An
arm that cannot find what is already known to be there cannot be trusted to
report an absence.

WHAT IT DOES NOT CLAIM
----------------------
No resolution site is ruled on. The census counts one site per distinct
expression reachable from a public entry point, and that ruling is a read of the
source, one package at a time. What runs here instead is a SCREEN: a file
holding both an ordering or rounding call and a level-shaped name. It bounds the
set from above and it decides nothing. A package the screen passes may hold no
site at all.

Downloads throughout come from one source in one run, so the ratio printed at
the end is internally consistent. They are NOT comparable with the pypistats
figures in the install-weight output, which are a different API on a different
date.

    python probes/frame_sweep.py
"""

import datetime
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_frame_sweep.txt")

UA = {"User-Agent": "conformal-coverage-audit/frame-sweep"}
SIMPLE = "https://pypi.org/simple/"
PYPI = "https://pypi.org/pypi/{}/json"
ECO = ("https://packages.ecosyste.ms/api/v1/registries/pypi.org/packages"
       "?sort=downloads&order=desc&per_page=100&page={}")
ECO_ONE = "https://packages.ecosyste.ms/api/v1/packages/lookup?ecosystem=pypi&name={}"

# How deep arm B reads. Every distribution above the download floor this implies
# is opened; the floor itself is printed, because a population is only a
# population once its edge is stated.
ARM_B_PAGES = 50

# The audit's own candidate list: the ten audited and the twelve the download
# denominator carries on its unaudited side. Both arms are scored against it.
AUDITED = ["crepes", "darts", "mapie", "neuralforecast", "nonconformist",
           "puncc", "river", "sktime", "statsforecast", "torchcp"]
UNAUDITED = ["fortuna", "uncertainty-toolbox", "quantile-forest",
             "conformal-tights", "crepes-weighted", "venn-abers", "nixtla",
             "pytorch-forecasting", "gluonts", "prophet", "skforecast",
             "orbit-ml"]
FRAME = AUDITED + UNAUDITED

# Arm A's name pattern. Word stems a conformal-prediction distribution plausibly
# puts in its own name. Deliberately wider than the phrases below, because a
# name is only the first filter and a cheap false positive costs one HTTP call.
NAME_PAT = re.compile(
    r"(conform|venn.?abers|\bcqr\b|cqr[-_]|[-_]cqr|\bicp\b|icp[-_]|[-_]icp)", re.I)

# The evidence a hit has to carry to survive. Phrases, not words: `conformal`
# alone reaches molecular conformers and protocol conformance suites, and bare
# `nonconformity` is a quality-management term that pulled in forty ERP modules
# on the first run. Every phrase here has to mean the statistical object.
EVIDENCE = re.compile(
    r"conformal predict|conformal infer|conformal regress|conformal classif|"
    r"conformal anomaly|conformal risk|conformal calibrat|conformalized|"
    r"conformalised|inductive conformal|split conformal|venn[-\s]abers|"
    r"nonconformity (?:score|measure|function)|"
    r"non-conformity (?:score|measure|function)|"
    r"distribution[-\s]free (?:prediction|uncertainty|coverage|interval)", re.I)

# The screen. One file containing both a rounding or ordering call and a name
# shaped like a level. This is not the census criterion and is not offered as one.
SHAPE = re.compile(rb"(np\.quantile|numpy\.quantile|np\.percentile|"
                   rb"numpy\.percentile|torch\.quantile|jnp\.quantile|"
                   rb"\.quantile\(|np\.sort|numpy\.sort|np\.argsort|"
                   rb"ceil\s*\(|floor\s*\(|np\.partition)")
LEVEL = re.compile(rb"(1\s*-\s*alpha|alpha|confidence|coverage|significance|"
                   rb"miscoverage)", re.I)

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def get_json(url, timeout=60, tries=4):
    for att in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            time.sleep(1.5 * (att + 1))
        except Exception:
            time.sleep(1.5 * (att + 1))
    return "unreachable"


def metadata_blob(name):
    """Summary, README, keywords and classifiers as one string, or None."""
    d = get_json(PYPI.format(name), timeout=30)
    if d is None:
        return ""
    if d == "unreachable":
        return None
    info = d["info"]
    return " ".join([info.get("summary") or "", info.get("description") or "",
                     info.get("keywords") or "",
                     " ".join(info.get("classifiers") or [])])


def evidence(name):
    """The conformal-prediction phrases a distribution's own metadata carries."""
    blob = metadata_blob(name)
    if blob is None:
        return None
    return sorted({m.group(0).lower() for m in EVIDENCE.finditer(blob)})


def self_check():
    """Failing inputs for both halves of the instrument, before it is trusted.

    The evidence pattern has to fire on a statement that means the statistical
    object and stay silent on one that does not; a screen that cannot tell a
    conformal predictor from an ERP nonconformity register would report the
    index as full of conformal libraries. And the name pattern has to admit that
    it cannot see `river`, because the whole reading of arm A's recall rests on
    that blindness being real.
    """
    fires = "Split conformal prediction with nonconformity scores"
    quiet = "Management System - Nonconformity register for audits"
    assert EVIDENCE.search(fires), "the evidence pattern misses a plain statement"
    assert not EVIDENCE.search(quiet), (
        "the evidence pattern fires on quality-management nonconformity; it "
        "would count ERP modules as conformal-prediction libraries")
    assert NAME_PAT.search("nonconformist"), "the name pattern misses a known hit"
    assert not NAME_PAT.search("river"), (
        "the name pattern claims to reach a package named after a word; arm A's "
        "recall figure is then measuring something else")
    assert not NAME_PAT.search("sktime"), "the name pattern reaches sktime"
    # The screen has to separate its two halves. A file that orders an array but
    # never mentions a level is not a candidate resolution site.
    assert SHAPE.search(b"np.quantile(scores, 0.9)"), "the screen misses an ordering call"
    assert not SHAPE.search(b"x = y + 1"), "the screen fires on arithmetic"
    assert LEVEL.search(b"alpha = 0.1"), "the level pattern misses alpha"
    assert not LEVEL.search(b"threshold = 3"), "the level pattern fires on a bare name"
    # The two sides of the candidate list must not overlap, or a package would be
    # scored twice and the recall denominator would be wrong.
    assert not (set(AUDITED) & set(UNAUDITED)), "a package is on both sides of the frame"
    assert len(FRAME) == len(set(FRAME)), "the frame repeats a package"
    return True


self_check()


def shape_screen(name):
    """(files read, files where a rounding-or-ordering call meets a level-ish name)."""
    d = get_json(PYPI.format(name), timeout=40)
    if d is None or d == "unreachable":
        return None, None
    urls = d.get("urls") or []
    pick = (next((u for u in urls if u["packagetype"] == "sdist"), None)
            or next((u for u in urls if u["packagetype"] == "bdist_wheel"), None))
    if not pick:
        return None, None
    try:
        req = urllib.request.Request(pick["url"], headers=UA)
        with urllib.request.urlopen(req, timeout=180) as fh:
            raw = fh.read()
    except Exception:
        return None, None
    members = []
    try:
        if pick["filename"].endswith((".tar.gz", ".tgz")):
            with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                for m in tf.getmembers():
                    if m.isfile() and m.name.endswith(".py") and m.size < 4_000_000:
                        members.append(tf.extractfile(m).read())
        else:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for m in zf.namelist():
                    if m.endswith(".py"):
                        members.append(zf.read(m))
    except Exception:
        return None, None
    hits = sum(1 for b in members if SHAPE.search(b) and LEVEL.search(b))
    return len(members), hits


def downloads(name):
    d = get_json(ECO_ONE.format(name), timeout=40)
    if not d or d == "unreachable" or not isinstance(d, list) or not d:
        return None
    v = d[0].get("downloads")
    return v if isinstance(v, int) else None


def execute_quantconformal():
    """Run the package rather than read it. Reports what happened, or why not.

    A deposit output that describes behaviour it never triggered is the failure
    this repository keeps finding in other people's libraries. So the rows below
    are executed, and if the package is absent from the environment the probe
    says that and claims nothing.
    """
    try:
        import quantconformal as qc  # noqa: F401
    except Exception as exc:
        return [f"NOT EXECUTED: {type(exc).__name__} importing quantconformal.",
                "No behavioural claim is made about it from this run. Pin it in",
                "probe-requirements.txt and re-run to replace this block."]
    import math
    from fractions import Fraction

    import numpy as np

    out = [f"executed against quantconformal {getattr(qc, '__version__', 'unknown')}"]
    cells = 0
    for a in ("0.1", "0.05", "0.2", "0.01", "1/3"):
        for n in (1, 2, 3, 9, 10, 19, 20, 39, 40, 99, 100):
            want = math.ceil(Fraction(n + 1) * (1 - Fraction(a)))
            got = qc.conformal_rank(n, a)
            assert got == want, (
                f"quantconformal returns rank {got} at n={n}, alpha={a}; the "
                f"order-statistic identity requires {want}")
            cells += 1
    out.append(f"rank equals ceil((n+1)(1-alpha)) in all {cells} cells tried")

    n_bad, a_bad = 10, 0.05
    try:
        qc.calibrate(np.arange(float(n_bad)), alpha=a_bad)
        out.append(f"below the floor at n={n_bad}, alpha={a_bad}: RETURNED SILENTLY")
    except Exception as exc:
        out.append(f"below the floor at n={n_bad}, alpha={a_bad}: raises "
                   f"{type(exc).__name__}")
    r = qc.calibrate(np.arange(float(n_bad)), alpha=a_bad, unavailable="infinite")
    out.append(f"opted in below the floor: threshold {r.threshold}, "
               f"eligible {r.eligible}, rank {r.rank} of {n_bad}")
    try:
        qc.calibrate(np.array([1.0, 2.0, float("inf")]), 0.1)
        out.append("a non-finite calibration score is ACCEPTED")
    except ValueError:
        out.append("a non-finite calibration score is rejected")
    return out


def main():
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    say("=" * 104)
    say("FRAME SWEEP -- the candidate list against a query over the index")
    say("=" * 104)
    say("self_check() passed at import: the evidence pattern fires on a")
    say("conformal-prediction statement and stays silent on an ERP nonconformity")
    say("register, the name pattern admits it cannot see a package called river,")
    say("and the screen separates an ordering call from bare arithmetic.")
    say("")
    say(f"queried {today} UTC. Names come from PyPI's own index, metadata from the")
    say("PyPI JSON API, download counts from ecosyste.ms in this same run. The")
    say("counts here are a dated snapshot. They are NOT the pypistats figures the")
    say("install-weight output carries: different API, different date, and the two")
    say("must not be set against each other.")
    say("")

    # ---- the index ------------------------------------------------------
    req = urllib.request.Request(
        SIMPLE, headers={**UA, "Accept": "application/vnd.pypi.simple.v1+json"})
    with urllib.request.urlopen(req, timeout=180) as fh:
        index = json.load(fh)
    names = [p["name"] for p in index["projects"]]
    assert len(names) > 100_000, (
        f"the index returned {len(names)} names; that is not the whole index and "
        f"a sweep over it would report a false absence")
    say("-" * 104)
    say("ARM A -- every name on the index, then the metadata behind each hit")
    say("-" * 104)
    say(f"  distributions on the index          {len(names):>8,}")

    cands = sorted({n for n in names if NAME_PAT.search(n)})
    say(f"  names matching the stem pattern     {len(cands):>8,}")

    with ThreadPoolExecutor(max_workers=6) as ex:
        ev = list(ex.map(lambda n: (n, evidence(n)), cands))
    in_scope = sorted(n for n, m in ev if m)
    rejected = [n for n, m in ev if m == []]
    unreach = [n for n, m in ev if m is None]
    say(f"  carrying conformal evidence         {len(in_scope):>8,}")
    say(f"  name matched, metadata did not      {len(rejected):>8,}")
    say(f"  metadata unreachable                {len(unreach):>8,}")
    say("")

    # ---- arm B ----------------------------------------------------------
    say("-" * 104)
    say("ARM B -- the download-ranked head, read the same way")
    say("-" * 104)
    head = []
    for page in range(1, ARM_B_PAGES + 1):
        d = get_json(ECO.format(page), timeout=60)
        if not d or d == "unreachable":
            break
        head += [(x["name"], x.get("downloads") or 0) for x in d]
        time.sleep(0.3)
    assert len(head) >= 1000, (
        f"the ranked head returned {len(head)} rows; an absence measured over a "
        f"population that small says nothing")
    floor = min(dl for _, dl in head)
    say(f"  distributions read                  {len(head):>8,}")
    say(f"  monthly-download floor at the edge  {floor:>8,}")

    with ThreadPoolExecutor(max_workers=6) as ex:
        evb = list(ex.map(lambda r: (r[0], r[1], evidence(r[0])), head))
    b_hit = sorted([(n, dl, m) for n, dl, m in evb if m], key=lambda r: -r[1])
    b_unreach = [n for n, _, m in evb if m is None]
    say(f"  carrying conformal evidence         {len(b_hit):>8,}")
    say(f"  metadata unreachable                {len(b_unreach):>8,}")
    inside = sorted(n for n in FRAME if n in dict(head))
    say(f"  candidate-list packages inside it   {len(inside):>8,}   {', '.join(inside)}")
    say("")
    if not b_hit:
        say("  Nothing above that floor says the word. The conformal-prediction")
        say("  ecosystem sits entirely below the head of the index, which is why")
        say("  a download-ordered read cannot assemble the candidate list either.")
        say("")

    # ---- recall ---------------------------------------------------------
    say("-" * 104)
    say("RECALL -- both arms turned on the candidate list, which is known-positive")
    say("-" * 104)
    say("Each arm is asked to find packages already known to belong. What it")
    say("misses here is what it would miss anywhere, and the miss is the reason")
    say("the manuscript keeps a named list rather than replacing it with a query.")
    say("")
    with ThreadPoolExecutor(max_workers=6) as ex:
        frame_ev = dict(ex.map(lambda n: (n, evidence(n)), FRAME))
    by_name = [n for n in FRAME if NAME_PAT.search(n)]
    by_meta = [n for n in FRAME if frame_ev.get(n)]
    invisible = [n for n in FRAME if not frame_ev.get(n)]
    aud_invisible = [n for n in invisible if n in AUDITED]
    say(f"  candidate list                      {len(FRAME):>8}")
    say(f"  reachable by the name pattern       {len(by_name):>8}   {', '.join(by_name)}")
    say(f"  declaring it in their own metadata  {len(by_meta):>8}")
    say(f"  declaring nothing anywhere          {len(invisible):>8}   {', '.join(invisible)}")
    say(f"  of those, audited by this paper     {len(aud_invisible):>8}   {', '.join(aud_invisible)}")
    say("")
    above_floor = [n for n in aud_invisible if n in dict(head)]
    say("  A query reaches a package only where the package announces itself. Of")
    say(f"  the ten under audit, {len(aud_invisible)} announce nothing of the kind anywhere in what")
    say(f"  they publish; {len(above_floor)} of them clear arm B's floor, were opened in full, and")
    say(f"  came back empty all the same: {', '.join(above_floor) if above_floor else 'none'}.")
    say("")

    # ---- what the list missed -------------------------------------------
    outside = [n for n in in_scope
               if n.lower().replace("_", "-") not in {f.lower() for f in FRAME}]
    say("-" * 104)
    say(f"WHAT THE CANDIDATE LIST MISSED -- conformal-first, outside it   ({len(outside)})")
    say("-" * 104)
    with ThreadPoolExecutor(max_workers=5) as ex:
        dls = dict(ex.map(lambda n: (n, downloads(n)), outside))
    with ThreadPoolExecutor(max_workers=5) as ex:
        scr = dict(ex.map(lambda n: (n, shape_screen(n)), outside))
    say(f"{'package':<36}{'monthly':>10}  {'py files':>9}{'screened in':>13}")
    missed_dl = 0
    screened = 0
    for n in sorted(outside, key=lambda x: -(dls.get(x) or 0)):
        dl = dls.get(n)
        nf, hits = scr.get(n, (None, None))
        missed_dl += dl or 0
        if hits:
            screened += 1
        say(f"{n:<36}{(f'{dl:,}' if dl is not None else '?'):>10}  "
            f"{(nf if nf is not None else '?'):>9}{(hits if hits is not None else '?'):>13}")
    say("")
    say(f"  monthly downloads over the {len(outside)}      {missed_dl:>10,}")
    say(f"  holding a file the screen passes    {screened:>10}")
    say("")

    aud_dl = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        aud_dl = dict(ex.map(lambda n: (n, downloads(n)), AUDITED))
    aud_tot = sum(v or 0 for v in aud_dl.values())
    say(f"  the audited ten, same source, same run  {aud_tot:>10,}")
    assert aud_tot > 0, "the audited ten returned no downloads; the ratio is not reportable"
    say(f"  what the candidate list missed, as a share of the audited ten   "
        f"{100.0 * missed_dl / aud_tot:.2f}%")
    say("")

    # ---- the one that changes something ---------------------------------
    say("-" * 104)
    say("THE ONE ROW THAT IS NOT A LONG TAIL")
    say("-" * 104)
    qc = "quantconformal"
    if qc in outside:
        d = get_json(PYPI.format(qc), timeout=40)
        info = d["info"]
        sd = next((u for u in d["urls"] if u["packagetype"] == "sdist"), None)
        rel = sorted((v, f[0]["upload_time_iso_8601"][:10])
                     for v, f in d["releases"].items() if f)
        say(f"  {qc} {info['version']}")
        say(f"    summary: {info['summary']}")
        say(f"    first release on the index: {rel[0][1]}" if rel else "")
        if sd:
            say(f"    sdist sha256: {sd['digests']['sha256']}")
        say("")
        say("    It ships the behaviour this paper's checklist asks for, and the")
        say("    lines below are what it did when run rather than what it says.")
        say("")
        qc_lines = execute_quantconformal()
        for line in qc_lines:
            say(f"    {line}")
        qc_cells = next((int(m.group(1)) for m in
                         (re.search(r"in all (\d+) cells tried", x) for x in qc_lines)
                         if m), 0)
        qc_date = rel[0][1] if rel else ""
    else:
        qc_cells, qc_date = 0, ""
        say(f"  {qc} is no longer in the swept set; the paragraph naming it is stale")
    say("")

    # ---- the counts a parser reads --------------------------------------
    say("=" * 104)
    say("THE COUNTS")
    say("=" * 104)
    say(f"  distributions on the index         {len(names)}")
    say(f"  arm A name candidates              {len(cands)}")
    say(f"  arm A carrying evidence            {len(in_scope)}")
    say(f"  arm B distributions read           {len(head)}")
    say(f"  arm B download floor               {floor}")
    say(f"  arm B carrying evidence            {len(b_hit)}")
    say(f"  candidate list size                {len(FRAME)}")
    say(f"  candidate list found by name       {len(by_name)}")
    say(f"  candidate list declaring evidence  {len(by_meta)}")
    say(f"  candidate list declaring nothing   {len(invisible)}")
    say(f"  audited packages declaring nothing {len(aud_invisible)}")
    say(f"  conformal-first outside the list   {len(outside)}")
    say(f"  of those the screen passes         {screened}")
    say(f"  their monthly downloads            {missed_dl}")
    say(f"  audited ten monthly downloads      {aud_tot}")
    say(f"  missed share of audited percent    {100.0 * missed_dl / aud_tot:.2f}")
    say(f"  quantconformal rank cells          {qc_cells}")
    say(f"  quantconformal first release       {qc_date}")
    say("")
    say("=" * 104)
    say("WHAT THIS DOES NOT SETTLE")
    say("=" * 104)
    say("Nothing here rules on a site. The screen puts a ceiling on the missed")
    say("set and settles no package inside it; ruling means reading one expression")
    say("at a time against the criterion, and this sweep read none that way.")
    say("")
    say("Arm A cannot see a conformal library named after an ordinary word, and")
    say("the recall block measures exactly how badly. Arm B cannot see anything")
    say("below its floor, which is where every conformal-first package lives.")
    say("Between them they bound the gap; neither closes it, and a package that")
    say("publishes no metadata at all is outside both.")
    say("")
    say("The index moves. Names are added daily and download ranks turn over, so")
    say("a re-run will not reproduce these counts exactly. The date above is the")
    say("claim's scope.")

    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
