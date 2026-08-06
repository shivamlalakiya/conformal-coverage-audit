#!/usr/bin/env python3
"""The disclosure record as data: what happened to each filing, queried not asserted.

WHAT WAS MISSING
----------------
The disclosure table listed fifteen filings and a Status column written by hand. A
paper reporting its own disclosure outcomes is more credible than one reporting only
that it filed -- but only if the outcomes are read off the tracker rather than typed,
because a hand-written status is exactly the kind of claim that goes stale silently and
in the flattering direction.

WHAT THIS PROBE CLAIMS, AND WHAT IT DOES NOT
--------------------------------------------
For every URL in DISCLOSURE.md it queries the public GitHub API and records:

  * state (open / closed), and for a pull request whether it was MERGED;
  * whether a maintainer -- anyone other than the filer -- has commented, which is the
    difference between acknowledged and unanswered;
  * the number of comments, and the date of the last one.

  It does NOT classify a finding as confirmed. A closed issue may be closed as
  wontfix, as a duplicate, or by a stale-bot, and a merged pull request is the
  strongest available signal but is still the maintainer's judgement of the patch and
  not an endorsement of the paper. Every state is reported as the tracker's own word.

  It does NOT report silence as rejection. An unanswered filing four days old is an
  unanswered filing four days old, and the age is printed so a reader can weigh it.

  A URL the API cannot answer for is NOT ATTEMPTED. Inferring a state would be the
  same error as a gate that passes without measuring.

The filer's login is read from the repository's own git config rather than typed, so
"a maintainer replied" means "somebody who is not the person who filed it replied".

    python probes/disclosure_status.py
"""

import datetime
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs", "probe_output_disclosure_status.txt")
DISCLOSURE = os.path.join(ROOT, "DISCLOSURE.md")

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def gh(path):
    """A GitHub API call through the gh CLI, or None if it cannot be answered."""
    try:
        r = subprocess.run(["gh", "api", path], capture_output=True, text=True,
                           timeout=60)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except Exception:
        return None


def filings():
    """(date, package, owner, repo, kind, number, url) per row of DISCLOSURE.md.

    Parsed out of the committed table rather than listed here, so a filing added to
    the disclosure and not to this probe cannot go unqueried.
    """
    out = []
    with open(DISCLOSURE, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.startswith("| 20"):
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) < 4:
                continue
            # EVERY url in the Report cell, not the first. Row 31 lists an issue and
            # the pull request fixing it; querying only the first dropped a merged PR
            # from the tally, in the direction that understates the record.
            found = re.findall(
                r"https://github\.com/([^/]+)/([^/)]+)/(issues|pull)/(\d+)", cells[3])
            for j, (owner, repo, kind, num) in enumerate(found):
                out.append({"date": cells[0], "pkg": cells[1].strip("`"),
                            "site": cells[2], "owner": owner, "repo": repo,
                            "kind": "pr" if kind == "pull" else "issue",
                            "num": int(num), "primary": j == 0,
                            "url": f"https://github.com/{owner}/{repo}/{kind}/{num}"})
    return out


def filer_login():
    """The filing identity, from git config -- not typed into this file."""
    try:
        r = subprocess.run(["gh", "api", "user", "--jq", ".login"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return None


def self_check():
    rows = filings()
    # the table must parse, and to the count the papers state
    assert rows, "no filings parsed out of DISCLOSURE.md"
    assert len(rows) >= 15, (
        f"only {len(rows)} filings parsed; the disclosure table lists more and a "
        f"probe reporting on a subset would understate the record")
    # every row must carry a resolvable target, or a status would be about nothing
    for r in rows:
        assert r["owner"] and r["repo"] and r["num"] > 0, r
        assert r["kind"] in ("issue", "pr"), r
        datetime.date.fromisoformat(r["date"])
    # both kinds must be present: a probe that only handled issues would silently
    # never report a merge, which is the strongest signal in the table
    kinds = {r["kind"] for r in rows}
    assert kinds == {"issue", "pr"}, f"only {kinds} in the disclosure record"
    # and the URLs must be distinct, or one filing is counted twice
    urls = [r["url"] for r in rows]
    assert len(set(urls)) == len(urls), "a URL appears twice in the disclosure table"
    # every tracker URL in the file must be one this probe queries. A row listing an
    # issue and its fix PR had its PR dropped by a regex taking only the first match,
    # and nothing said so -- the tally simply read one merge low.
    with open(DISCLOSURE, encoding="utf-8") as fh:
        in_file = set(re.findall(
            r"https://github\.com/[^/]+/[^/)]+/(?:issues|pull)/\d+", fh.read()))
    queried = set(urls)
    assert in_file <= queried, (
        f"tracker URLs in DISCLOSURE.md that this probe does not query: "
        f"{sorted(in_file - queried)}")
    return rows


FILINGS = self_check()


def main():
    today = datetime.datetime.now(datetime.timezone.utc).date()
    say("=" * 108)
    say("THE DISCLOSURE RECORD AS DATA: QUERIED FROM THE TRACKERS, NOT ASSERTED")
    say("=" * 108)
    say("self_check() passed at import: every row of DISCLOSURE.md parses to a")
    say("resolvable target, both issues and pull requests are present, no URL appears")
    say("twice, and every date is a date.")
    say("")
    say(f"queried {today.isoformat()} UTC through the gh CLI against the public API.")
    say("")
    say("WHAT A STATE MEANS AND DOES NOT. `merged` is the maintainer accepting a patch")
    say("and is the strongest signal available; it is not an endorsement of this paper.")
    say("`closed` may be a fix, a wontfix, a duplicate or a bot -- the tracker's own")
    say("word is printed and no further reading is imposed. Silence is reported as")
    say("silence with an age beside it, never as rejection. A target the API cannot")
    say("answer for is NOT ATTEMPTED.")
    say("")
    me = filer_login()
    say(f"filing identity from the local gh credential: "
        f"{me if me else 'UNKNOWN -- reply attribution not attempted'}")
    say("")

    rows = []
    for f in FILINGS:
        base = f"repos/{f['owner']}/{f['repo']}"
        num = f["num"]
        data = gh(f"{base}/issues/{num}")
        if data is None:
            rows.append({**f, "state": None})
            continue
        merged = None
        if f["kind"] == "pr":
            pr = gh(f"{base}/pulls/{num}")
            merged = bool(pr.get("merged_at")) if pr else None
        comments = gh(f"{base}/issues/{num}/comments") or []
        others = [c for c in comments
                  if me and c.get("user", {}).get("login") != me]
        last = None
        if comments:
            last = max(c["created_at"][:10] for c in comments)
        age = (today - datetime.date.fromisoformat(f["date"])).days
        rows.append({**f, "state": data.get("state"), "merged": merged,
                     "n_comments": len(comments), "n_others": len(others),
                     "last": last, "age": age,
                     "closed_at": (data.get("closed_at") or "")[:10]})

    say("-" * 108)
    say(f"{'date':>10} {'package':<15} {'target':<34} {'state':>8} {'merged':>7} "
        f"{'replies':>8} {'not mine':>9} {'age d':>6}")
    say("-" * 108)
    for r in rows:
        tgt = f"{r['owner']}/{r['repo']}#{r['num']}"
        if r["state"] is None:
            say(f"{r['date']:>10} {r['pkg']:<15} {tgt:<34} "
                f"{'NOT ATTEMPTED -- the API did not answer':>41}")
            continue
        say(f"{r['date']:>10} {r['pkg']:<15} {tgt:<34} {r['state']:>8} "
            f"{('yes' if r['merged'] else ('no' if r['merged'] is False else '-')):>7} "
            f"{r['n_comments']:>8} {r['n_others']:>9} {r['age']:>6}")
    say("")

    ok = [r for r in rows if r["state"] is not None]
    miss = [r for r in rows if r["state"] is None]
    say("=" * 108)
    say("THE TALLY")
    say("=" * 108)
    merged = [r for r in ok if r["merged"]]
    closed = [r for r in ok if r["state"] == "closed" and not r["merged"]]
    openr = [r for r in ok if r["state"] == "open"]
    engaged = [r for r in ok if r["n_others"] > 0]
    silent = [r for r in ok if r["n_comments"] == 0]
    prim = [r for r in ok if r["primary"]]
    say(f"targets queried            {len(ok)} of {len(rows)}")
    say(f"  distinct filings         {len(prim)}")
    say(f"  linked fix PRs           {len(ok) - len(prim)}")
    if miss:
        say(f"  not attempted            {len(miss)}: "
            f"{', '.join(f'{r['owner']}/{r['repo']}#{r['num']}' for r in miss)}")
    say(f"merged                     {len(merged)}")
    say(f"closed, not merged         {len(closed)}")
    say(f"open                       {len(openr)}")
    say(f"drew a reply from someone")
    say(f"  other than the filer     {len(engaged)}")
    say(f"no comment at all          {len(silent)}")
    if ok:
        say(f"oldest filing              {min(r['age'] for r in ok)} to "
            f"{max(r['age'] for r in ok)} days")
    say("")
    assert len(ok) + len(miss) == len(rows)
    assert len(merged) + len(closed) + len(openr) == len(ok), (
        "the three states do not partition the answered filings")
    say("READ THIS AS A SHORT WINDOW, WHICH IS WHAT IT IS. The filings run from")
    say(f"{min(r['date'] for r in rows)} to {max(r['date'] for r in rows)} and the")
    say(f"oldest is {max(r['age'] for r in ok) if ok else 0} days old. Maintainer")
    say("response times in these projects are measured in weeks, so the open count is")
    say("a statement about elapsed time and not about the findings. The table is")
    say("regenerated at submission rather than frozen here, and a filing that turns")
    say("into a dispute will appear as one.")
    say("")
    say("WHAT WOULD FALSIFY A FINDING is a maintainer explaining that the probe is")
    say("wrong, and that has a place in this table: it appears as a closed issue with")
    say("replies and no merge. We do not have the standing to grade our own filings,")
    say("which is why the columns are states and counts rather than verdicts.")

    with open(os.path.abspath(OUT), "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwritten -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
