#!/usr/bin/env python3
"""The out-of-census read: which packages outside the census carry a site, and which do not.

WHAT WAS MISSING
----------------
Four packages the census does not cover were classified by the suite, and the write-up
said what each of them does. It never said where those four came from. Four names that
turn out to carry the shape, with no stated population behind them, is a numerator with
nothing under it -- and that is the move this work objects to when a library makes it.

WHAT THIS PROBE CLAIMS, AND WHAT IT DOES NOT
--------------------------------------------
It files a read. Each package on the unaudited side of the download denominator was
opened and looked through for a site: one expression taking a nominated miscoverage
target to a position inside a held-out array of scores, arrived at from a public entry
point. The finding for each is recorded below beside the file and the line somebody
else can open.

  * These are READINGS. Four were afterwards put through the suite, and those runs
    are the conformance arm's own three outputs; the remaining eight were read and
    are filed as read. Reading can turn up an absence. It cannot measure one, and no
    figure below pretends otherwise.
  * NO SITE COUNT IS PRINTED. Counting sites needs one expression-level ruling per
    package -- the rule the census applies -- and this read never made those rulings.
    A loosely built numerator set beside the census's own would not be the same
    quantity twice.
  * Every anchor was taken from a released sdist at the version named. Two of the
    packages sit under ../cp-src, and for those the line is opened again while this
    runs and what came back is printed. The rest carry the sdist digest instead and
    say plainly that no local copy backs them. That is thinner evidence and it is
    marked, not buried.
  * The population is not a sweep of the index. It is whichever names the download
    denominator already lists as unaudited, and that list was written by hand. A
    package nobody in this area would think to name is outside it before any rule
    applies.

    python probes/oos_frame.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_oos_frame.txt")
SRC = os.path.join(HERE, "..", "..", "cp-src")
WEIGHT = os.path.join(HERE, "..", "outputs", "probe_output_install_weight.txt")
EXECUTED = ("probe_output_conformance_oos.txt",
            "probe_output_conformance_oos_fortuna.txt",
            "probe_output_oos_conformal_tights.txt")

# verdicts. `site` was afterwards executed by the suite; `refuted` claimed a site on
# the first read and lost it on the second; `none` never claimed one.
SITE, REFUTED, NONE = "site", "refuted", "none"

# One entry per package on the unaudited side of the download denominator. `anchor` is
# the path inside the released sdist; `held` is where this repository keeps a copy, and
# is None where it keeps none. `digest` is the sdist's own sha256 where the read
# recorded it, and None where it did not -- an absent digest is an absent digest.
MANIFEST = [
    dict(pkg="conformal-tights", version="0.5.0", verdict=SITE,
         anchor="src/conformal_tights/_conformal_coherent_quantile_regressor.py:255",
         expr="intercept_l2 = np.quantile(y_cqr_l2 - Δŷ_calib_l2_quantiles[:, j], quantile)",
         why="the asked-for coverage reaches numpy unaltered over the level-2 CQR scores; "
             "no (n+1) step, and a coherence clip then decides what is kept",
         digest="346043c85416028bb8af90da239a3ade830fd2343606aff55dc18492558740dd",
         held="conformal_tights/conformal_tights/_conformal_coherent_quantile_regressor.py",
         held_version="0.5.0"),
    dict(pkg="crepes-weighted", version="0.1.3", verdict=SITE,
         anchor="src/crepes_weighted/base.py:565",
         expr="alpha_index = int((1 - confidence) * (len(self.alphas) + 1)) - 1",
         why="a whole-number index into held-out absolute residuals sorted downward; the "
             "branch the fork adds is a second expression in the same method, at :576",
         digest="f6691660e255d23d5ccde2fc27298756a6cc2386b2021a437fbd60ce05f0eb47",
         held=None, held_version=None),
    dict(pkg="fortuna", version="0.2.0", verdict=SITE,
         anchor="fortuna/conformal/regression/quantile.py:90",
         expr="return jnp.quantile(scores, jnp.ceil((n + 1) * (1 - error)) / n)",
         why="the rank is worked out and then divided by n and passed on as a level, so it "
             "goes above 1 wherever no rank reaches the request; the same line sits in "
             "onedim_uncertainty.py:90",
         digest="8e21d65958c71fb4e95842a233d5700b5ad7885b0a8f83d92e25f8325c392811",
         held=None, held_version=None),
    dict(pkg="skforecast", version="0.24.0", verdict=SITE,
         anchor="skforecast/preprocessing/_preprocessing.py:2694",
         expr="self.correction_factor_[k] = float(np.quantile(conformity_scores, self.nominal_coverage))",
         why="the asked-for coverage becomes the numpy level as it stands, over held-out "
             "conformity scores, with no (n+1) step and nothing watching n",
         digest="46122c7cf14a487501e71f1d27980871d3f8d032d3fa9d3791df8715a6448de1",
         held=None, held_version=None),

    dict(pkg="quantile-forest", version="1.4.2", verdict=REFUTED,
         anchor="examples/plot_qrf_conformalized_intervals.py:118",
         expr="s = np.quantile(conf_scores, np.clip((1 - alpha) * (1 + (1 / (len(y_calib)))), 0, 1))",
         why="the expression is genuine and it is in examples/, which the census rules out "
             "for not being public API; the estimator's own predict(quantiles=) weighs "
             "training responses pooled from leaves, so no held-out array is ranked",
         digest=None, held=None, held_version=None),
    dict(pkg="venn-abers", version="1.5.4", verdict=REFUTED,
         anchor="src/venn_abers.py:708",
         expr="self.m_parameter = int(np.floor(self.epsilon * (len(y_cal) + 1) / 2))",
         why="a level does become a whole-number position on a genuinely held-out split, but "
             "what gets sorted is the calibration response labels; those order statistics "
             "only clip what feeds the isotonic step and are not what comes back out",
         digest="a6bf5383e618ce42d0ecb2335bba0d6b104ecbe979041f10019b57572364bc69",
         held="venn_abers/venn_abers-1.5.3/src/venn_abers.py", held_version="1.5.3"),

    dict(pkg="pytorch-forecasting", version="1.8.0", verdict=NONE,
         anchor="pytorch_forecasting/metrics/base_metrics/_base_metrics.py:133",
         expr="y_pred = torch.quantile(",
         why="taken over the network's own predicted sample axis; elsewhere the quantiles are "
             "pinball-trained heads or a fitted icdf, and nowhere is there a held-out score set",
         digest=None, held=None, held_version=None),
    dict(pkg="uncertainty-toolbox", version="0.1.1", verdict=NONE,
         anchor="uncertainty_toolbox/metrics_calibration.py:626",
         expr="    quantile_prediction = norm.ppf(quantile).flatten()",
         why="every bound comes off a predicted mean and spread; isotonic recalibration bends "
             "the level first, but its observed proportions are tallies on a fixed grid rather "
             "than sample ranks, and the residuals are never put in order",
         digest=None, held=None, held_version=None),
    dict(pkg="gluonts", version="0.17.0", verdict=NONE,
         anchor="src/gluonts/model/forecast.py:494",
         expr="        sample_idx = int(np.round((self.num_samples - 1) * q))",
         why="a position on sorted Monte Carlo draws from the model; 'conformal', "
             "'nonconformity' and 'calibrat' occur nowhere in the sdist",
         digest=None, held=None, held_version=None),
    dict(pkg="nixtla", version="0.8.0", verdict=NONE,
         anchor="nixtla/nixtla_client.py:1928",
         expr='        out = _maybe_add_intervals(out, resp["intervals"])',
         why="a client against a hosted model: the level travels in the request body and the "
             "bounds arrive already built, so no arithmetic over scores happens on this side",
         digest=None, held=None, held_version=None),
    dict(pkg="prophet", version="1.4.0", verdict=NONE,
         anchor="prophet/forecaster.py:1693",
         expr="        lower_p = 100 * (1.0 - self.interval_width) / 2",
         why="a percentile down simulated posterior-predictive trajectories, which are drawn "
             "from the fitted model and are not held-out scores",
         digest="cf54064e9ad1be56cf4c6fb70f95c7b0fccc095fdb90a53b72407262d423a70d",
         held=None, held_version=None),
    dict(pkg="orbit-ml", version="1.1.5.1", verdict=NONE,
         anchor="orbit/utils/predictions.py:67",
         expr="        computed_array = np.percentile(v, percentiles, axis=0)",
         why="a percentile across posterior draws; there is no calibration split, and neither "
             "'conformal' nor 'nonconformity' appears in the source",
         digest="f5895d4a999eff7d7dc536c84b62bbf73712e6e6d89ed0ecacaa0eedcdb0cb2a",
         held=None, held_version=None),
]

# ---------------------------------------------------------------------------
# The census side of the same question. The reading corpus under ../cp-src is
# what "we downloaded and read N packages" refers to, and until now no output
# carried N, which package each verdict belongs to, or where to look.
#
# `audited` is the census's ten. `excluded` is a package the criterion turns
# away and the turning-away still stands. `corrected` is the one the criterion
# turned away on a reading that a later pass overturned by running it -- kept
# as its own verdict rather than quietly moved, because a corpus that hides
# its corrections is the thing this work objects to.
AUDITED, EXCLUDED, CORRECTED = "audited", "excluded", "corrected"

CORPUS = [
    dict(pkg="crepes", verdict=AUDITED), dict(pkg="darts", verdict=AUDITED),
    dict(pkg="mapie", verdict=AUDITED), dict(pkg="neuralforecast", verdict=AUDITED),
    dict(pkg="nonconformist", verdict=AUDITED), dict(pkg="puncc", verdict=AUDITED),
    dict(pkg="river", verdict=AUDITED), dict(pkg="sktime", verdict=AUDITED),
    dict(pkg="statsforecast", verdict=AUDITED), dict(pkg="torchcp", verdict=AUDITED),

    dict(pkg="uq360", verdict=EXCLUDED, version="0.2",
         held="uq360/uq360/algorithms/infinitesimal_jackknife/infinitesimal_jackknife.py",
         line=80,
         expr="y_lower = np.quantile(y, q=0.5 * self.config['alpha'], axis=0)",
         why="the array it takes a percentile of is built at :76-79 by setting the model's "
             "parameters to each perturbed draw and predicting again, so it is a percentile "
             "across a predictive distribution and no rank or finite-sample coverage attaches "
             "to it. The same reasoning turns away the CV+ and jackknife+ aggregation "
             "elsewhere, which quantile cross-fit prediction arrays rather than scores"),
    dict(pkg="venn_abers", verdict=EXCLUDED, version="1.5.3",
         held="venn_abers/venn_abers-1.5.3/src/venn_abers.py", line=706,
         expr="self.m_parameter = int(np.round(self.epsilon * (len(y_cal) + 1) / 2))",
         why="epsilon does reach an integer index, and the split really is withheld. What "
             "gets ordered at :710 is y_cal itself. Those positions serve only to clip what "
             "feeds the isotonic step, and what comes back out are multiprobability bounds. "
             "There is no miscoverage target anywhere in the path"),

    dict(pkg="conformal_tights", verdict=CORRECTED, version="0.5.0",
         held="conformal_tights/conformal_tights/_xgboost_weighted_quantile.py", line=13,
         expr="def _weighted_quantile(",
         why="set aside on this helper, where XGBoost emits the quantiles and nothing "
             "withheld gets ordered. Right about the file, wrong about the distribution. "
             "Line 255 of the coherent-quantile module puts the caller's own level onto "
             "level-2 scores from a third partition, and the criterion admits that. Reading "
             "missed it; execution did not, and the out-of-census arm carries the numbers"),
]

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def _read(name):
    with open(os.path.join(HERE, "..", "outputs", name)) as fh:
        return fh.read()


def frame_from_download_denominator():
    """The unaudited names the download probe's own output carries.

    Parsed out of the committed output rather than imported from install_weight.py,
    because the output is what the paper's denominator macro is read from. A package
    added to one file and not the other has to break something, and this is it.
    """
    names, side = [], None
    for ln in _read("probe_output_install_weight.txt").splitlines():
        if ln.startswith("AUDITED -- "):
            side = "audited"
        elif ln.startswith("NOT AUDITED -- "):
            side = "unaudited"
        row = re.match(r"^([a-z0-9._-]+)\s+[\d,]+\s+[\d,]+\s+[\d,]+\s*$", ln)
        if row and side == "unaudited":
            names.append(row.group(1))
    return names


def executed_packages():
    """Which packages the three out-of-census runs actually name in their own output."""
    blob = "\n".join(_read(n) for n in EXECUTED).lower()
    return {e["pkg"] for e in MANIFEST if e["pkg"] in blob}


def audited_from_census():
    """The libraries the census output itself names under 'P2 by library'.

    Parsed rather than restated. Audit one more package and this list grows while
    CORPUS does not, which is the drift that has to be loud.
    """
    names, inside = [], False
    for ln in _read("probe_output_helper_census.txt").splitlines():
        if ln.strip() == "P2 by library:":
            inside = True
            continue
        if inside:
            m = re.match(r"^\s{6}([a-z0-9._-]+) \S+\s+\d+", ln)
            if m:
                names.append(m.group(1))
            elif ln.strip() == "":
                break
    return names


def check_corpus_anchor(e):
    """Open the line. Every non-audited row is held under ../cp-src, so all verify."""
    path = os.path.join(SRC, e["held"])
    if not os.path.exists(path):
        return "MISSING from ../cp-src"
    with open(path, encoding="utf-8") as fh:
        held = fh.read().splitlines()
    if e["line"] > len(held):
        return f"line {e['line']} past end of file"
    return ("re-read, matches" if held[e["line"] - 1].strip() == e["expr"].strip()
            else "RE-READ DIFFERS")


def self_check():
    """Every way this file can go quietly wrong, made loud.

    Each assertion below has an input that fires it: drop a row and the frame check
    goes; mislabel a verdict and the executed check goes; add a package to the
    download probe and the first one goes. An assertion with no such input is a
    decoration and does not belong here.
    """
    pkgs = [e["pkg"] for e in MANIFEST]
    assert len(pkgs) == len(set(pkgs)), "a package is listed twice"

    frame = frame_from_download_denominator()
    assert frame, "the download output gave up no unaudited rows; the parser has drifted"
    assert set(pkgs) == set(frame), (
        "this read and the download denominator disagree about the population: "
        f"only here {sorted(set(pkgs) - set(frame))}, only there "
        f"{sorted(set(frame) - set(pkgs))}")

    for e in MANIFEST:
        assert e["verdict"] in (SITE, REFUTED, NONE), f"{e['pkg']}: unknown verdict"
        assert re.match(r"^\S+:\d+$", e["anchor"]), (
            f"{e['pkg']}: no file and line, so the finding rests on nothing a reader "
            f"can open")
        assert e["why"], f"{e['pkg']}: a verdict with no reason attached"

    # ---- the census side ----------------------------------------------------
    corpus = [e["pkg"] for e in CORPUS]
    assert len(corpus) == len(set(corpus)), "a package is listed twice in CORPUS"
    on_disk = sorted(d for d in os.listdir(SRC)
                     if os.path.isdir(os.path.join(SRC, d)) and d != "literature")
    assert sorted(corpus) == on_disk, (
        "CORPUS and the reading corpus under ../cp-src disagree: only in CORPUS "
        f"{sorted(set(corpus) - set(on_disk))}, only on disk "
        f"{sorted(set(on_disk) - set(corpus))}")
    audited = [e["pkg"] for e in CORPUS if e["verdict"] == AUDITED]
    census = audited_from_census()
    assert census, "the census output gave up no 'P2 by library' rows; the parser drifted"
    assert sorted(audited) == sorted(census), (
        "CORPUS's audited set is not the set the census output names: only here "
        f"{sorted(set(audited) - set(census))}, only there "
        f"{sorted(set(census) - set(audited))}")
    for e in CORPUS:
        if e["verdict"] == AUDITED:
            continue
        for need in ("version", "held", "line", "expr", "why"):
            assert e.get(need), f"{e['pkg']}: a non-audited row with no {need}"
    # The corrected row exists because a package the census turned away was later
    # measured. If it is not in the out-of-census site set, the correction is a
    # story with no measurement under it.
    corrected = {e["pkg"].replace("_", "-") for e in CORPUS if e["verdict"] == CORRECTED}
    assert corrected <= {e["pkg"] for e in MANIFEST if e["verdict"] == SITE}, (
        f"{sorted(corrected)} is filed as a corrected exclusion but does not appear "
        f"among the packages the out-of-census runs measured")

    claimed = {e["pkg"] for e in MANIFEST if e["verdict"] == SITE}
    ran = executed_packages()
    assert claimed == ran, (
        "the packages filed as carrying a site are not the packages the out-of-census "
        f"runs name: filed {sorted(claimed)}, run {sorted(ran)}")


def check_anchor(e):
    """Re-open the line, where this repository holds the source. Otherwise say so."""
    if not e["held"]:
        return "no local copy" if not e["digest"] else "no local copy, digest filed"
    path = os.path.join(SRC, e["held"])
    if not os.path.exists(path):
        return "MISSING from ../cp-src"
    lineno = int(e["anchor"].rsplit(":", 1)[1])
    with open(path, encoding="utf-8") as fh:
        held = fh.read().splitlines()
    if lineno > len(held):
        return f"line {lineno} past end of held copy"
    got = held[lineno - 1]
    if e["held_version"] != e["version"]:
        same = "same" if got.strip() == e["expr"].strip() else "differs"
        return f"held {e['held_version']}, read {e['version']}: {same} at :{lineno}"
    return "re-read, matches" if got.strip() == e["expr"].strip() else "RE-READ DIFFERS"


def main():
    self_check()
    say("=" * 104)
    say("OUT-OF-CENSUS READ -- what was looked at outside the ten, and what came back")
    say("=" * 104)
    say("self_check() passed at import: the population here is the download")
    say("denominator's own unaudited side, every row carries a file and a line, and the")
    say("rows filed as carrying a site are exactly the packages the three out-of-census")
    say("runs name in their outputs.")
    say()
    say("A SITE is one expression taking a nominated miscoverage target to a position")
    say("inside a held-out array of scores, arrived at from a public entry point. That is")
    say("the census's own wording and it is applied unchanged, which is the only reason")
    say("the two populations can be spoken about together.")
    say()

    say("=" * 104)
    say("THE READING CORPUS -- what was downloaded and opened, and what became of it")
    say("=" * 104)
    say("The census reports ten libraries. This is the set those ten came out of, which")
    say("no output carried before: the directories under ../cp-src, minus the literature")
    say("folder, with every non-audited row opened again while this runs.")
    say()
    counts_c = {}
    for verdict, title in ((AUDITED, "AUDITED -- the census's ten"),
                           (EXCLUDED, "TURNED AWAY BY THE CRITERION, AND IT STANDS"),
                           (CORRECTED, "TURNED AWAY ON A READING THAT RUNNING IT OVERTURNED")):
        rows = [e for e in CORPUS if e["verdict"] == verdict]
        counts_c[verdict] = len(rows)
        say("-" * 104)
        say(f"{title}   ({len(rows)})")
        say("-" * 104)
        if verdict == AUDITED:
            say("  " + ", ".join(e["pkg"] for e in rows))
            say("  (the same ten the census output names under 'P2 by library', checked here)")
            say()
            continue
        for e in rows:
            say(f"{e['pkg']} {e['version']}")
            say(f"    {e['held']}:{e['line']}")
            say(f"        {e['expr']}")
            for chunk in _wrap(e["why"], 96):
                say(f"    {chunk}")
            say(f"    anchor now: {check_corpus_anchor(e)}")
            say()
    say("  packages downloaded and read   %d" % len(CORPUS))
    say("  audited                        %d" % counts_c[AUDITED])
    say("  turned away, standing          %d" % counts_c[EXCLUDED])
    say("  turned away, overturned        %d" % counts_c[CORRECTED])
    say()
    say("🛑 The overturned row is the one to read. conformal_tights was set aside on")
    say("_xgboost_weighted_quantile.py, which is a gradient-boosted quantile helper and")
    say("forms no order statistic of anything held out. That was true of the file and")
    say("false of the distribution. Its conformal path puts the caller's own level onto")
    say("scores from a third partition, and it took execution to see that -- which is the")
    say("same failure of reading the audit reports in ten other packages, occurring here")
    say("in the audit's own frame.")
    say()

    frame = frame_from_download_denominator()
    say("=" * 104)
    say("THE UNAUDITED SIDE -- a different population, and not a subset of the one above")
    say("=" * 104)
    say(f"population: {len(frame)} packages, read off probe_output_install_weight.txt")
    say("            (the two names PyPI could not answer for are not in it)")
    say()

    for verdict, title, note in (
        (SITE, "CARRIES A SITE -- afterwards put through the suite",
         "each of these was executed; the rows are in the conformance arm's outputs"),
        (REFUTED, "CLAIMED A SITE ON THE FIRST READ, LOST IT ON THE SECOND",
         "a second reader was told to break the claim, and did"),
        (NONE, "NO SITE",
         "reached by reading alone; nothing here measures it"),
    ):
        rows = [e for e in MANIFEST if e["verdict"] == verdict]
        say("-" * 104)
        say(f"{title}   ({len(rows)})")
        say(f"  {note}")
        say("-" * 104)
        for e in rows:
            say(f"{e['pkg']} {e['version']}")
            say(f"    {e['anchor']}")
            say(f"        {e['expr']}")
            for chunk in _wrap(e["why"], 96):
                say(f"    {chunk}")
            say(f"    anchor now: {check_anchor(e)}")
            if e["digest"]:
                say(f"    sdist sha256: {e['digest']}")
            say()

    say("=" * 104)
    say("THE COUNTS")
    say("=" * 104)
    counts = {v: len([e for e in MANIFEST if e["verdict"] == v])
              for v in (SITE, REFUTED, NONE)}
    say(f"  packages read                 {len(MANIFEST)}")
    say(f"  carrying a site               {counts[SITE]}")
    say(f"  refuted on the second read    {counts[REFUTED]}")
    say(f"  no site                       {counts[NONE]}")
    say()
    say("no site count is printed here, on purpose: ruling on how many distinct")
    say("expressions each package exposes is work this read did not do, and a number")
    say("built without it would not stand beside the census's own.")
    say()
    say("=" * 104)
    say("WHAT THIS DOES NOT SETTLE")
    say("=" * 104)
    say("The population was written by name and not swept out of the package index, so")
    say("a library nobody here would think of is outside it before any rule is applied.")
    say("Eight of the twelve rest on reading alone, at one version apiece, and a package")
    say("with nothing at the version opened may well have something at another.")

    with open(os.path.abspath(OUT), "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwritten -> {os.path.abspath(OUT)}")


def _wrap(text, width):
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


self_check()

if __name__ == "__main__":
    main()
