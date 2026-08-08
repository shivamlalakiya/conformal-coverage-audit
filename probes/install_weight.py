#!/usr/bin/env python3
"""What fraction of installs does the census cover? The denominator.

WHAT WAS MISSING
----------------
"Ten packages" invites "which ten, and do they matter". The audit reported a count and
no denominator, so a referee could not tell whether it covered the ecosystem or a
corner of it. A count without a denominator is the same rhetorical move the audit
criticises elsewhere.

WHAT THIS PROBE CLAIMS, AND WHAT IT DOES NOT
--------------------------------------------
It reads recent download counts from the public pypistats.org API for the audited
packages and for conformal-prediction packages deliberately NOT audited, and reports
the audited share.

  * Downloads are not installs and not users. A CI job pulling a wheel a thousand
    times a day counts a thousand times; a package vendored into a container image
    counts once. The number bounds attention, not adoption, and it is reported as a
    share of downloads rather than as a share of anything else.
  * The window is whatever the API's `recent` endpoint returns, which is a rolling
    period ending at the query date. The date is recorded in the output so the figure
    is a dated snapshot rather than a standing claim.
  * A package the API cannot answer for is reported as NOT ATTEMPTED and excluded from
    BOTH numerator and denominator. Inferring a zero would understate the denominator
    and flatter the audit.

The unaudited set is chosen adversarially: it is the conformal-prediction and
quantile-interval packages we know of and did not measure, so the share is a lower
bound on how much of the field the audit missed rather than an upper bound on how much
it covered.

    python probes/install_weight.py
"""

import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import helper_census as HC  # noqa: E402

OUT = os.path.join(HERE, "..", "outputs", "probe_output_install_weight.txt")
API = "https://pypistats.org/api/packages/{}/recent"
TIMEOUT = 30

# Conformal-prediction, quantile-regression and prediction-interval packages that
# expose a coverage target and are NOT in the census. Listed so the denominator is
# adversarial: every name here counts against the audited share.
UNAUDITED = [
    "mapie-learn",          # a distinct distribution name, kept in case it resolves
    "fortuna",
    "uncertainty-toolbox",
    "quantile-forest",
    "conformal-tights",
    "crepes-weighted",
    "deel-puncc",
    "venn-abers",
    "nixtla",
    "pytorch-forecasting",
    "gluonts",
    "prophet",
    "skforecast",
    "orbit-ml",
]

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


# pypistats rate-limits a burst. A first run queried 24 packages back to back and got
# three NOT ATTEMPTED -- mapie, crepes and river, all of which answer fine when asked
# with a pause. A throttled query reported as a missing package would have understated
# the numerator by the flagship conformal library, which is the one direction of error
# that flatters the audit.
PACE = 2.0


def recent(pkg):
    """(last_day, last_week, last_month) or None if the API cannot answer."""
    url = API.format(pkg)
    time.sleep(PACE)
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "conformal-coverage-audit/weight"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as fh:
                data = json.load(fh)
            d = data["data"]
            return (d["last_day"], d["last_week"], d["last_month"])
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # No such distribution. Distinct from a transport failure, and the
                # distinction is reported: calling a name that does not exist
                # "not attempted" implies missing data where there is none.
                return "absent"
            time.sleep(3 * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def self_check():
    # the census must be non-empty and must name packages, or the numerator is empty
    libs = sorted({s["lib"].split()[0] for s in HC.MANIFEST})
    assert libs, "the census manifest names no packages"
    # the audited and unaudited sets must be disjoint, or a package would be counted
    # on both sides of the ratio and the share would be meaningless
    overlap = set(libs) & set(UNAUDITED)
    assert not overlap, f"a package is on both sides of the ratio: {overlap}"
    # and the unaudited set must be non-trivial: a share computed against two
    # abandoned packages would be true and worthless
    assert len(UNAUDITED) >= len(libs), (
        f"only {len(UNAUDITED)} unaudited packages against {len(libs)} audited; the "
        f"denominator is not adversarial enough to be worth reporting")
    return libs


AUDITED = self_check()


def main():
    say("=" * 100)
    say("INSTALL WEIGHT: WHAT SHARE OF DOWNLOADS DOES THE CENSUS COVER?")
    say("=" * 100)
    say("self_check() passed at import: the census names packages, the audited and")
    say("unaudited sets are disjoint, and the unaudited set is at least as large as")
    say("the audited one so the share is not flattered by a thin denominator.")
    say("")
    say(f"queried {datetime.datetime.now(datetime.timezone.utc).date().isoformat()} "
        f"UTC from pypistats.org; the API's `recent` window is rolling and ends at")
    say("that date, so every figure below is a dated snapshot and not a standing")
    say("claim. DOWNLOADS ARE NOT INSTALLS AND NOT USERS: a CI job pulling a wheel a")
    say("thousand times counts a thousand times, a vendored package counts once. The")
    say("number bounds attention.")
    say("")

    rows = []
    for pkg, audited in ([(p, True) for p in AUDITED]
                         + [(p, False) for p in UNAUDITED]):
        got = recent(pkg)
        rows.append({"pkg": pkg, "audited": audited, "counts": got})

    def block(label, want):
        say("-" * 100)
        say(f"{label}")
        say("-" * 100)
        say(f"{'package':<24} {'last day':>12} {'last week':>12} {'last month':>12}")
        tot = 0
        miss = []
        for r in rows:
            if r["audited"] != want:
                continue
            if r["counts"] == "absent":
                say(f"{r['pkg']:<24} "
                    f"{'no such distribution on PyPI -- not a gap':>38}")
                continue
            if r["counts"] is None:
                miss.append(r["pkg"])
                say(f"{r['pkg']:<24} {'NOT ATTEMPTED -- excluded from both sides':>38}")
                continue
            day, week, month = r["counts"]
            tot += month
            say(f"{r['pkg']:<24} {day:>12,} {week:>12,} {month:>12,}")
        say("")
        say(f"  {label.lower()} monthly total: {tot:,}")
        if miss:
            say(f"  not attempted ({len(miss)}): {', '.join(miss)}")
        say("")
        return tot, miss

    aud, aud_miss = block("AUDITED -- the census", True)
    una, una_miss = block("NOT AUDITED -- deliberately adversarial", False)
    # The share is only worth printing if the large audited packages answered. A run
    # that lost mapie to a rate limit would report 28.9% where the figure is 31.4%,
    # and would report the narrow share as 11.6% where it is above 80% -- an error
    # that runs against the audit and would still be wrong.
    assert "mapie" not in aud_miss, (
        "mapie did not answer, so the audited total is missing the flagship conformal "
        "library. Re-run rather than publish the share.")
    assert len(aud_miss) <= 1, (
        f"{len(aud_miss)} audited packages did not answer ({aud_miss}); the share is "
        f"not reportable from this run")

    say("=" * 100)
    say("THE SHARE")
    say("=" * 100)
    denom = aud + una
    if denom == 0:
        say("NOT ATTEMPTED: the API answered for nothing, so no share is reported.")
        say("A share computed from an empty denominator would be a fabrication.")
    else:
        share = 100.0 * aud / denom
        say(f"audited monthly downloads      {aud:>14,}")
        say(f"unaudited monthly downloads    {una:>14,}")
        say(f"combined                       {denom:>14,}")
        say(f"audited share                  {share:>13.1f}%")
        say("")
        ok = [r for r in rows if isinstance(r["counts"], tuple)]
        n_aud = sum(1 for r in ok if r["audited"])
        n_una = sum(1 for r in ok if not r["audited"])
        say(f"over {n_aud} audited and {n_una} unaudited packages the API answered for.")
        say("")
        if share >= 50:
            say("So the census covers the majority of downloads in the set considered,")
            say("against a denominator chosen to work against it.")
        else:
            say("So the census covers a MINORITY of downloads in the set considered.")
            say("That is the honest figure and it is the one reported. The unaudited")
            say("side is dominated by general forecasting libraries whose conformal")
            say("path is one feature among many, which is why the share is low; a")
            say("denominator restricted to conformal-first packages would be higher")
            say("and would also be chosen to flatter.")
        say("")
        # the same ratio over conformal-FIRST packages only, stated as the narrower
        # question rather than substituted for the broad one
        cf_first = {"mapie", "crepes", "nonconformist", "puncc", "torchcp"}
        cf_una = {"fortuna", "uncertainty-toolbox", "conformal-tights",
                  "crepes-weighted", "deel-puncc", "venn-abers", "mapie-learn"}
        a2 = sum(r["counts"][2] for r in rows
                 if isinstance(r["counts"], tuple) and r["pkg"] in cf_first)
        u2 = sum(r["counts"][2] for r in rows
                 if isinstance(r["counts"], tuple) and r["pkg"] in cf_una)
        if a2 + u2:
            say("THE NARROWER QUESTION, asked separately rather than instead. Restricted")
            say("to packages centred on conformal prediction:")
            say(f"  audited   {a2:>12,}    unaudited {u2:>12,}    "
                f"share {100.0 * a2 / (a2 + u2):>5.1f}%")
            say("")
            narrow = 100.0 * a2 / (a2 + u2)
            assert narrow > share, (
                f"the narrow share {narrow:.1f}% is not above the broad {share:.1f}%, "
                f"so the sentence explaining why the broad figure is low is backwards")
            say("Both figures are reported because each answers a different objection.")
            say("The broad one answers 'do these packages matter'; the narrow one")
            say("answers 'did you audit the conformal ecosystem or a sample of it'.")

    say("")
    say("=" * 100)
    say("WHAT THIS DOES NOT SETTLE")
    say("=" * 100)
    say("A share of PyPI downloads says excludes R, Julia, SQL, and spreadsheet-style workflows")
    say("users, and the interface census covers those separately without a")
    say("denominator of any kind. It says nothing about which code paths inside a")
    say("downloaded package are executed. And a package absent from PyPI entirely --")
    say("vendored, internal, or installed from source -- is invisible here.")

    with open(os.path.abspath(OUT), "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwritten -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
