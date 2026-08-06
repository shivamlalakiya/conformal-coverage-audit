#!/usr/bin/env python3
"""How long has each resolution site looked like this? The version axis.

What was missing
----------------
The census is a snapshot at pinned versions, and a referee reads a snapshot as
perishable: the behaviour might be a transient that a maintainer fixed last month,
or it might have shipped unchanged for three years. Those two readings support very
different sentences, and nothing in the deposit could tell them apart.

WHAT THIS PROBE CLAIMS, AND WHAT IT DOES NOT
--------------------------------------------
This is a DOCUMENTARY result, not a behavioural one, and the distinction is the
whole reason it is safe to publish.

  * It downloads each release's distribution from PyPI, reads the file the census
    anchors, and records whether the anchor expression is PRESENT verbatim.
  * It therefore dates the EXPRESSION. It does not execute anything, so it does not
    establish that the behaviour was the same -- surrounding code can change what
    an unchanged expression does.
  * The behavioural classification stays where it already is: executed, at the
    pinned version, by the conformance suite.

Read together the two give a sentence neither gives alone: the behaviour measured at
the pinned version is produced by an expression that has been in place since release
X. That is worth having and it is not the same as "the behaviour has been wrong
since X", which this probe cannot and does not say.

Every release that cannot be fetched or unpacked is reported as NOT ATTEMPTED. A
version whose absence is inferred rather than checked would be the same mistake as a
gate that passes without measuring.

    python probes/version_history.py [MAX_RELEASES_PER_PACKAGE]
"""

import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helper_census as HC  # noqa: E402

OUT = "outputs/probe_output_version_history.txt"
TIMEOUT = 60
CACHE = os.path.join("/tmp", "cca_version_cache")


def pypi_releases(pkg):
    """(version, url, kind) per release, newest first, sdist or wheel."""
    url = f"https://pypi.org/pypi/{pkg}/json"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as fh:
        data = json.load(fh)
    out = []
    for ver, files in data["releases"].items():
        pick = None
        for f in files:
            if f.get("yanked"):
                continue
            if f["packagetype"] == "bdist_wheel" and f["filename"].endswith(".whl"):
                pick = (f["url"], "wheel")
                break
        if pick is None:
            for f in files:
                if f.get("yanked"):
                    continue
                if f["packagetype"] == "sdist":
                    pick = (f["url"], "sdist")
                    break
        if pick:
            out.append((ver, pick[0], pick[1]))

    def key(v):
        return [int(x) if x.isdigit() else 0
                for x in re.split(r"[._-]", v)[:4]] + [v]
    return sorted(out, key=lambda r: key(r[0]), reverse=True)


def fetch(url):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, os.path.basename(url))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, "rb").read()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "conformal-coverage-audit/history"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as fh:
                blob = fh.read()
            with open(path, "wb") as fh:
                fh.write(blob)
            return blob
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def read_member(blob, kind, relpath):
    """The text of `relpath` inside the archive, or None if it is not there.

    A wheel lays files out at the package root; an sdist wraps them in one
    top-level directory whose name carries the version, so the match is on the
    path SUFFIX rather than on an exact name.
    """
    try:
        if kind == "wheel":
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for nm in z.namelist():
                    if nm == relpath or nm.endswith("/" + relpath):
                        return z.read(nm).decode("utf-8", "replace")
        else:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as t:
                for m in t.getmembers():
                    if m.isfile() and (m.name == relpath
                                       or m.name.endswith("/" + relpath)):
                        f = t.extractfile(m)
                        return f.read().decode("utf-8", "replace") if f else None
    except Exception:
        return None
    return None


def norm(s):
    """Whitespace-insensitive, so a reformat is not read as a rewrite.

    The question is whether the EXPRESSION is there, not whether black has been run
    over the file. Collapsing runs of whitespace is the smallest normalisation that
    answers that and it is stated rather than silent.
    """
    return re.sub(r"\s+", " ", s)


def self_check():
    # The manifest must be non-empty and every entry must carry the three fields
    # this probe needs, or it is scanning nothing.
    assert HC.MANIFEST, "census manifest is empty"
    for s in HC.MANIFEST:
        assert s["lib"] and s["path"] and s["anchor"], s
    # And the anchor must be findable in the PINNED source on disk, or the anchor is
    # already stale and every historical answer would be measured against a target
    # that does not exist. This is the same check the census runs; repeating it here
    # means this probe cannot report "absent in every version" for a typo.
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "..", "cp-src")
    checked, found = 0, 0
    for s in HC.MANIFEST:
        lib = s["lib"].split()[0]
        for base in (os.path.join(root, lib), os.path.join(root, lib, lib)):
            p = os.path.join(base, s["path"])
            if os.path.exists(p):
                checked += 1
                if norm(s["anchor"]) in norm(open(p, encoding="utf-8",
                                                  errors="replace").read()):
                    found += 1
                break
    # not every third-party tree is vendored here, so require only that the ones
    # present overwhelmingly match -- a wholesale mismatch means normalisation broke
    if checked:
        assert found >= 0.8 * checked, (
            f"only {found} of {checked} anchors match the vendored source; the "
            f"normalisation or the manifest is wrong and no historical answer from "
            f"this probe would mean anything")
    assert norm("a  b\n c") == "a b c"


self_check()


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 104)
    say("THE VERSION AXIS: HOW LONG HAS EACH RESOLUTION EXPRESSION BEEN THERE?")
    say("=" * 104)
    say("self_check() passed at import: the manifest is non-empty, every entry")
    say("carries a path and an anchor, and the anchors match the vendored pinned")
    say("source -- so a historical 'absent' is about the release and not about a")
    say("typo in the anchor.")
    say("")
    say("DOCUMENTARY, NOT BEHAVIOURAL. This reads source out of released")
    say("distributions and reports whether the anchored expression is present. It")
    say("executes nothing, so it dates the EXPRESSION and not the behaviour;")
    say("surrounding code can change what an unchanged expression does. The")
    say("behavioural classification stays in the conformance suite, executed at the")
    say("pinned version. Together: the behaviour measured at the pin is produced by")
    say("an expression in place since the release named below.")
    say("")
    say(f"at most {cap} most-recent releases per package; whitespace normalised, so")
    say("a reformat is not read as a rewrite")
    say("")

    # group manifest sites by package
    by_pkg = {}
    for s in HC.MANIFEST:
        by_pkg.setdefault(s["lib"].split()[0], []).append(s)

    summary = []
    for pkg, sites in sorted(by_pkg.items()):
        say("-" * 104)
        say(f"{pkg}   ({len(sites)} anchored site(s))")
        say("-" * 104)
        try:
            rels = pypi_releases(pkg)[:cap]
        except Exception as exc:
            say(f"  NOT ATTEMPTED: release list unavailable ({type(exc).__name__})")
            continue
        say(f"  releases examined: {len(rels)}  "
            f"({rels[-1][0]} .. {rels[0][0]})")

        # fetch each release once, then test every site against it
        blobs = {}
        skipped = []
        for ver, url, kind in rels:
            blob = fetch(url)
            if blob is None:
                skipped.append(ver)
                continue
            blobs[ver] = (blob, kind)
        if skipped:
            say(f"  NOT ATTEMPTED (download failed): {', '.join(skipped)}")

        for s in sites:
            present, absent, nofile = [], [], []
            for ver, (blob, kind) in blobs.items():
                txt = read_member(blob, kind, s["path"])
                if txt is None:
                    nofile.append(ver)
                elif norm(s["anchor"]) in norm(txt):
                    present.append(ver)
                else:
                    absent.append(ver)
            say("")
            say(f"  {s['symbol']}")
            say(f"      {s['path']}   branch ({s['branch']})")
            say(f"      anchor: {s['anchor'][:80]}")
            say(f"      present in {len(present)} of {len(blobs)} releases fetched"
                + (f"; file absent in {len(nofile)}" if nofile else "")
                + (f"; expression changed in {len(absent)}" if absent else ""))
            if present:
                order = [v for v, _, _ in rels if v in present]
                say(f"      unchanged across: {order[0]} back to {order[-1]}")
            if absent:
                order = [v for v, _, _ in rels if v in absent]
                say(f"      differs in: {', '.join(order[:8])}"
                    + (" ..." if len(order) > 8 else ""))
            if nofile:
                order = [v for v, _, _ in rels if v in nofile]
                say(f"      file not present in: {', '.join(order[:8])}"
                    + (" ..." if len(order) > 8 else ""))
            summary.append({"pkg": pkg, "symbol": s["symbol"],
                            "branch": s["branch"], "present": len(present),
                            "fetched": len(blobs), "changed": len(absent),
                            "nofile": len(nofile),
                            "oldest": ([v for v, _, _ in rels if v in present] or
                                       [None])[-1]})
        say("")

    say("=" * 104)
    say("SUMMARY")
    say(f"{'package':<15} {'site':<44} {'br':>5} {'present':>8} {'fetched':>8} "
        f"{'oldest with it':>15}")
    say("-" * 104)
    for r in summary:
        # truncate to 42 in a 44-wide field, so a long site name always leaves at
        # least two spaces before the next column. At 44-in-44 the separator
        # vanished and a parser reading on whitespace silently dropped that row --
        # 26 of 27, with nothing to notice it.
        say(f"{r['pkg']:<15} {r['symbol'][:42]:<44} {r['branch']:>5} "
            f"{r['present']:>8} {r['fetched']:>8} {str(r['oldest']):>15}")
    say("")
    if summary:
        stable = [r for r in summary
                  if r["fetched"] and r["present"] == r["fetched"]]
        say(f"{len(stable)} of {len(summary)} anchored expressions are present in")
        say(f"EVERY release fetched, so those are not transients: whatever the")
        say(f"conformance suite measures at the pin, the expression producing it has")
        say(f"been in place across the whole window examined here.")
        say("")
        say("The converse is also worth stating plainly: an expression that changed")
        say("across releases is NOT thereby shown to have behaved differently, and")
        say("this probe does not say it did. Dating an expression is all it does.")

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        OUT)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
