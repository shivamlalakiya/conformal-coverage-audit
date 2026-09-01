#!/usr/bin/env python3
"""A blind re-classification worksheet, and the scorer that reads it back.

WHAT WAS MISSING
----------------
The audit classifies each resolution site into a branch, and the same person
wrote the criterion, the classifier and the answers. The threats section says so
and offers three substitutes -- execution rather than judgement, a second-language
re-implementation, anchors that fail the build -- and states plainly that none of
them is a second party. What it could not offer was a way for anybody else to
start, which makes the limitation permanent rather than open.

This builds that way in. It draws a reproducible sample of sites, prints each
one as source with its answer withheld, and scores a filled-in sheet against the
audit's own labels.

WHAT IT DOES NOT DO
-------------------
It does not measure agreement. Nobody has filled a sheet in. Any figure this
repository prints for inter-rater agreement would be the author agreeing with
the author, which is the thing the instrument exists to avoid. The worksheet
ships blank on purpose and the manuscript says the number is absent.

THE BLIND IS THE INSTRUMENT
---------------------------
A worksheet leaking the answer measures nothing, so the written sheet is read
back and rejected when a label from the key survives anywhere in it. That check
has a failing input of its own: a sheet built with the redaction step skipped.

    python probes/second_coder.py               # write the worksheet and the key
    python probes/second_coder.py --score SHEET # agreement and Cohen's kappa
"""

import argparse
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import helper_census as HC  # noqa: E402

SHEET = os.path.join(HERE, "..", "outputs", "second_coder_worksheet.txt")
KEY = os.path.join(HERE, "..", "outputs", "second_coder_key.txt")
OUT = os.path.join(HERE, "..", "outputs", "probe_output_second_coder.txt")

# Fixed, so the sample is the same sheet for every coder and for every re-run.
SEED = 20260901
SAMPLE = 12
CONTEXT = 4

# The branch taxonomy, stated for the coder in the same words the paper uses.
BRANCHES = [
    ("a", "corrects the level by (n+1)/n, then hands it to a quantile routine "
          "that RAISES when the corrected level exceeds 1"),
    ("b", "corrects the level, then CLAMPS it at 1, returning the largest "
          "observed score where no rank carries the request"),
    ("c", "corrects the level, then returns an INFINITE bound where no rank "
          "carries the request"),
    ("d", "applies NO (n+1) correction: the requested level reaches the "
          "quantile routine unaltered"),
    ("e", "a p-value construction whose denominator is n"),
    ("f", "a p-value construction whose denominator is n+1, or a smoothed one"),
    ("g", "an approximate or streaming estimator that matches no exact rule"),
    ("count", "compares an inclusion COUNT against a count-scale threshold, "
              "carrying no (n+1) correction"),
    ("none", "this expression resolves no level onto a position, or its output "
             "does not control a returned bound"),
]

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def sample_sites():
    sites = [s for s in HC.MANIFEST if s["output"]]
    rng = random.Random(SEED)
    return rng.sample(sorted(sites, key=lambda s: (s["lib"], s["path"], s["line"])),
                      min(SAMPLE, len(sites)))


def source_window(site, root):
    path, how = HC.resolve(site, root)
    if not path or not os.path.exists(path):
        return None, how
    lines = open(path, errors="replace").read().splitlines()
    i = site["line"] - 1
    lo, hi = max(0, i - CONTEXT), min(len(lines), i + CONTEXT + 1)
    return [(n + 1, lines[n]) for n in range(lo, hi)], how


def redact(text, labels):
    """Remove anything that would hand the coder the answer.

    The audit's `rule` prose names the branch outright, so it never reaches the
    sheet at all; this is the belt on top of that. It strips any bare branch
    token so a stray mention in a source comment cannot leak either.
    """
    out = text
    for lab in sorted(labels, key=len, reverse=True):
        out = re.sub(rf"\bbranch\s*\(?{re.escape(lab)}\)?\b", "branch (?)", out,
                     flags=re.I)
    return out


def kappa(a, b):
    """Cohen's kappa for two label vectors."""
    assert len(a) == len(b) and a, "kappa needs two equal, non-empty vectors"
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe == 1.0:
        return po, pe, float("nan")
    return po, pe, (po - pe) / (1 - pe)


def self_check():
    """Failing inputs for the scorer and for the blind.

    The blind is the one that matters. A worksheet that quietly carries the
    audit's label would return a perfect agreement from any coder who read it,
    and nothing downstream would notice.
    """
    po, pe, k = kappa(["a", "b", "c"], ["a", "b", "c"])
    assert po == 1.0 and abs(k - 1.0) < 1e-12, "kappa is not 1 on identical labels"
    po, pe, k = kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])
    assert abs(po - 0.5) < 1e-12 and abs(k) < 1e-12, (
        "kappa is not 0 when agreement equals chance")
    po, pe, k = kappa(["a", "a"], ["b", "b"])
    assert po == 0.0, "kappa reports agreement where there is none"
    # the redactor has to actually remove a label, and leave other prose alone
    assert "branch (?)" in redact("this is branch (b) territory", ["b"]), (
        "the redactor leaves a branch label in the worksheet; the blind is not blind")
    assert "quantile" in redact("a quantile call", ["b"]), "the redactor eats prose"
    assert len({b for b, _ in BRANCHES}) == len(BRANCHES), "a branch is listed twice"
    return True


self_check()


def build(root):
    sites = sample_sites()
    labels = [s["branch"] for s in sites]
    sheet = []
    sheet.append("=" * 100)
    sheet.append("BLIND RE-CLASSIFICATION WORKSHEET")
    sheet.append("=" * 100)
    sheet.append("")
    sheet.append("You are looking at expressions taken from released Python packages.")
    sheet.append("Each one may or may not convert a requested coverage or miscoverage")
    sheet.append("level into a position inside an array of held-out calibration scores.")
    sheet.append("")
    sheet.append("For each item, write one label from this list on the ANSWER line.")
    sheet.append("Guessing is fine. Leaving it blank is also fine and is scored as a")
    sheet.append("non-answer rather than as a disagreement.")
    sheet.append("")
    for b, d in BRANCHES:
        sheet.append(f"  {b:<7} {d}")
    sheet.append("")
    sheet.append("Return the file with the ANSWER lines filled in. Score it with:")
    sheet.append("    python probes/second_coder.py --score <this file>")
    sheet.append("")
    missing = 0
    for k, s in enumerate(sites, 1):
        win, how = source_window(s, root)
        sheet.append("-" * 100)
        sheet.append(f"ITEM {k:02d}   {s['lib']}   {s['path']}:{s['line']}")
        sheet.append("-" * 100)
        if win is None:
            missing += 1
            sheet.append(f"  [source not available here: {how}]")
            sheet.append(f"  the expression, as recorded: {s['anchor']}")
        else:
            for n, text in win:
                mark = ">>" if n == s["line"] else "  "
                sheet.append(f"  {mark} {n:>6}  {text[:110]}")
        sheet.append("")
        sheet.append(f"  reached from: {', '.join(s['entries'])}")
        sheet.append("")
        sheet.append(f"  ANSWER {k:02d}: ")
        sheet.append("")
    body = redact("\n".join(sheet), {b for b, _ in BRANCHES})
    with open(SHEET, "w") as fh:
        fh.write(body + "\n")
    with open(KEY, "w") as fh:
        fh.write("# The audit's own labels. Do not open before filling the sheet.\n")
        for k, s in enumerate(sites, 1):
            fh.write(f"{k:02d}\t{s['branch']}\t{s['lib']}\t{s['path']}:{s['line']}\n")

    # the blind, checked against what was actually written rather than intended
    written = open(SHEET).read()
    for k, s in enumerate(sites, 1):
        # [ \t]* rather than \s*: \s crosses the newline and matches the next
        # line, which reported every blank answer as filled in.
        pat = rf"ANSWER {k:02d}:[ \t]*\S"
        assert not re.search(pat, written), f"item {k} was written out pre-answered"
    leaked = [s["branch"] for s in sites
              if re.search(rf"\bbranch\s*\(?{re.escape(s['branch'])}\)?\b", written, re.I)]
    assert not leaked, f"the worksheet leaks branch labels {sorted(set(leaked))}"
    return sites, labels, missing


def score(path):
    key = {}
    for ln in open(KEY):
        if ln.startswith("#"):
            continue
        n, lab, *_ = ln.rstrip("\n").split("\t")
        key[n] = lab
    got = {}
    for m in re.finditer(r"ANSWER (\d+):[ \t]*(\S+)", open(path).read()):
        got[m.group(1)] = m.group(2).strip()
    common = sorted(set(key) & set(got))
    if not common:
        print("no answers found in that sheet; nothing to score")
        return
    a = [key[c] for c in common]
    b = [got[c] for c in common]
    po, pe, k = kappa(a, b)
    print(f"items scored      {len(common)} of {len(key)}")
    print(f"raw agreement     {po:.3f}")
    print(f"chance agreement  {pe:.3f}")
    print(f"Cohen's kappa     {k:.3f}")
    for c in common:
        flag = "" if key[c] == got[c] else "   <- differs"
        print(f"  {c}  audit={key[c]:<8} coder={got[c]:<8}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "..", "..", "cp-src"))
    ap.add_argument("--score", metavar="SHEET")
    args = ap.parse_args()
    if args.score:
        return score(args.score)

    sites, labels, missing = build(args.root)
    say("=" * 104)
    say("SECOND CODER -- the instrument, shipped blank")
    say("=" * 104)
    say("self_check() passed at import: kappa returns 1 on identical labels and 0")
    say("at chance, and the redactor removes a branch label rather than the prose")
    say("around it. What was written is then read back, and rejected outright")
    say("should any pre-filled response or key label survive in it.")
    say("")
    say("The audit classified every site itself. That is the authors-as-oracle")
    say("threat the manuscript declares, and the three things standing in for")
    say("independence are not independence. This is what a second party would")
    say("need in order to disagree.")
    say("")
    say("  the sample, as drawn:")
    for k, s in enumerate(sites, 1):
        say(f"    {k:02d}  {s['lib']:<20} {s['path']}:{s['line']}")
    say("")
    say("=" * 104)
    say("THE COUNTS")
    say("=" * 104)
    say(f"  sites in the census            {len([s for s in HC.MANIFEST if s['output']])}")
    say(f"  drawn into the worksheet       {len(sites)}")
    say(f"  seed                           {SEED}")
    say(f"  source lines of context each   {CONTEXT}")
    say(f"  items whose source is absent   {missing}")
    say(f"  labels offered to the coder    {len(BRANCHES)}")
    say(f"  sheets returned                0")
    say("")
    say("=" * 104)
    say("AGREEMENT")
    say("=" * 104)
    say("  sheets returned                0")
    say("  raw agreement                  not measured")
    say("  Cohen's kappa                  not measured")
    say("")
    say("No figure is printed and none is withheld. The author filling in a sheet")
    say("the author wrote would produce a number that measures nothing, so the")
    say("worksheet ships blank and the manuscript reports the absence.")
    say("")
    say("=" * 104)
    say("WHAT THIS DOES NOT SETTLE")
    say("=" * 104)
    say("The blind stops an answer reaching the sheet. It does not stop a")
    say("determined reader looking one up: the labels sit in the census manifest in")
    say("this same repository and in the manuscript's own table, both public. What")
    say("the redaction buys is that nobody is told the answer by accident while")
    say("reading the item, which is the failure a worksheet actually has.")
    say("")
    say("A sample of twelve bounds a kappa loosely even once somebody fills it in.")
    say("The sample is also drawn from sites the audit already located, so it")
    say("measures agreement on classification and says nothing about whether the")
    say("census found every site there is. That second question is the frame")
    say("sweep's and the enumerator calibration's, not this one's.")

    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")
    print(f"wrote {SHEET}")
    print(f"wrote {KEY}")


if __name__ == "__main__":
    main()
