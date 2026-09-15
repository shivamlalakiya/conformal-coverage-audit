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

OFFERED = {b for b, _ in BRANCHES}

# The census vocabulary is wider than the label list a coder is handed, and the
# gap is not cosmetic: a key label outside OFFERED cannot be matched by any
# coder, so the item scores as a disagreement however well the site was read.
# Every such label is either given a rule naming the offered labels that count
# as agreement, or declared unscoreable. Silence is the one option that lets the
# instrument report a ceiling it does not have.
#
# 'e/f' and 'f/e' are one class in two spellings, and the ambiguity is on the
# SAME axis the coder is choosing along: (e) and (f) both name a p-value
# denominator, both are offered, and a site carrying the exact form in one
# branch and the smoothed form in the next admits either letter. Membership.
SCORES_AS = {
    "e/f": {"e", "f"},
    "f/e": {"e", "f"},
}

# 🛑 Unscoreable, and the reason matters more than the list.
#
# '?' carries no executed label, so there is nothing to score against.
#
# 'c-exact' was given a rule mapping it to (c) and that rule MANUFACTURED
# AGREEMENT. It is a statement about the MECHANISM -- the site appends +inf to
# the scores and inverts, so it is exact without correcting a level -- while
# (a), (b) and (c) are statements about BEHAVIOUR WHERE NO RANK CARRIES THE
# REQUEST. Those are two axes and no offered label spans them. For
# BaseCalibrator.compute_quantile the default path raises from an explicit
# pre-check eleven lines above the drawn line (api/calibration.py:256-257,
# alpha_calib_check, which raises ValueError when alpha < 1/(n+1)), and the
# +inf append that (c) describes is reached only when weights are passed:
# probe_output_routes_tabular.txt:21 records the site as `raises`. So (a) is
# wrong (the raise is not from the quantile routine and the level is not
# corrected), (c) is wrong (that is the weighted path), and (d) is literally
# true of the level and false of the outcome. An item with no correct answer is
# not scored; it is declared.
UNSCOREABLE = {"?", "c-exact"}

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
    # Every declared rule has to name offered labels, or it moves the ceiling
    # instead of removing it.
    for lab, ok in SCORES_AS.items():
        assert ok and ok <= OFFERED, (
            f"SCORES_AS[{lab!r}] names {sorted(ok - OFFERED)}, which the coder "
            f"was never offered")
        assert lab not in OFFERED, (
            f"{lab!r} is offered to the coder and needs no membership rule")
    assert not (SCORES_AS.keys() & UNSCOREABLE), (
        "a label cannot be both unscoreable and given a scoring rule")
    assert not (UNSCOREABLE & OFFERED), (
        "an offered label cannot be unscoreable")
    # 🛑 The rule this replaced said SCORES_AS["c-exact"] = {"c"}, which scored
    # two readers' (c) as agreement on a site the audit's own routes output
    # records as `raises`. A mapping onto a different axis is how an instrument
    # reports agreement it did not measure.
    assert "c-exact" not in SCORES_AS, (
        "'c-exact' is a mechanism label and (a)/(b)/(c) are behaviour labels; "
        "mapping between the axes manufactures agreement")
    assert SCORES_AS["e/f"] == SCORES_AS["f/e"], (
        "the two spellings of the p-value class score differently")
    return True


self_check()


def build(root):
    sites = sample_sites()
    labels = [s["branch"] for s in sites]
    # The key is only scoreable against the labels the coder was offered. A
    # census value that is neither offered nor given a rule in SCORES_AS makes
    # the item unmatchable, which is a silent ceiling on agreement rather than
    # a measurement; a '?' site has no executed label at all.
    undeclared = sorted({b for b in labels if b not in OFFERED
                         and b not in SCORES_AS and b not in UNSCOREABLE})
    assert not undeclared, (
        f"drawn sites carry census labels {undeclared} that are neither in the "
        f"{len(OFFERED)} offered to the coder nor declared in SCORES_AS; every "
        f"such item scores as a disagreement however well it was read. A '?' "
        f"site reaches here too, and it must not be given a rule: it carries "
        f"no executed label to score against")
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
    answered = sorted(set(key) & set(got))
    # An item whose key has no correct answer in the offered labels is excluded
    # rather than counted. Counting it either penalises every coder or, if it is
    # mapped onto a neighbouring label, credits one who was wrong.
    excluded = [c for c in answered if key[c] in UNSCOREABLE]
    common = [c for c in answered if key[c] not in UNSCOREABLE]
    if not common:
        print("no scoreable answers found in that sheet; nothing to score")
        return
    # An item whose key sits outside the offered labels is scored on membership
    # (SCORES_AS), not equality. Without this a correct reading of a 'c-exact'
    # or 'e/f' site is counted as a disagreement and the ceiling is below 1.0.
    def agrees(k_lab, c_lab):
        return c_lab in SCORES_AS.get(k_lab, {k_lab})

    # kappa needs one label per rater, so a membership hit is folded onto the
    # coder's own letter; that is what agreement means for these items.
    a = [got[c] if agrees(key[c], got[c]) else key[c] for c in common]
    b = [got[c] for c in common]
    po, pe, k = kappa(a, b)
    by_membership = [c for c in common if key[c] in SCORES_AS]
    blank = sorted(set(key) - set(got))
    print(f"items scored      {len(common)} of {len(key)}")
    print(f"  answered         {len(answered)}")
    print(f"  left blank       {len(blank)}"
          + (f"  ({', '.join(blank)})" if blank else ""))
    print(f"  excluded         {len(excluded)}"
          + (f"  ({', '.join(excluded)}: no offered label is correct)"
             if excluded else ""))
    print(f"  correct          {sum(1 for c in common if agrees(key[c], got[c]))}"
          f" of {len(common)} scoreable")
    print(f"raw agreement     {po:.3f}")
    print(f"chance agreement  {pe:.3f}")
    print(f"Cohen's kappa     {k:.3f}")
    if by_membership:
        print(f"scored on membership rather than equality: "
              f"{', '.join(by_membership)}")
        print("  (the key label is outside the labels the coder was offered; "
              "SCORES_AS names what agrees)")
    for c in excluded:
        print(f"  {c}  audit={key[c]:<8} coder={got[c]:<8}   <- EXCLUDED, "
              f"no offered label is correct for this site")
    for c in common:
        hit = agrees(key[c], got[c])
        flag = "" if hit else "   <- differs"
        if hit and key[c] != got[c]:
            flag = f"   <- agrees, key ({key[c]}) is not an offered label"
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
