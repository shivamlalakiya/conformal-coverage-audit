#!/usr/bin/env python3
"""What the four-size certificate cannot see: rules outside its sixteen candidates.

probes/certificate.py proves that four calibration sizes separate the sixteen
candidate rules it enumerates (rail x correction x family). The proof is about
those sixteen. A helper that resolves its level some other way can agree with one
of the sixteen at all four certificate sizes and still differ from it elsewhere,
and then the four sizes classify it as that candidate. The criterion is the full
checked range (the irregular prefix plus two periods), which certificate.py
establishes is complete for the sixteen; the four sizes are a screen.

This probe runs that case on numpy's thirteen np.quantile methods, each at the
raw level L and at the corrected level L(n+1)/n (clipped at one), on the score
set 1..n, so the delivered rank is the number of scores at or below the returned
threshold and the required rank k* = ceil((n+1)L) is known.

One float standard throughout: the exact rational level is rounded once to a
double and handed to numpy, the threshold numpy returns is used as it is, and the
delivered rank is counted against it. The sixteen candidates are run on the same
standard (they are numpy methods too), so a candidate and a numpy rule are
compared on the same footing; the probe also reports where that standard and the
exact rational rule of certificate.py disagree, since a disagreement there would
move the certificate itself.

    python probes/certificate_blind_spot.py
"""

import importlib.util
import math
import os
from fractions import Fraction as F

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_certificate_blind_spot.txt")

# certificate.py by path, so a module of the same name elsewhere cannot stand in
_spec = importlib.util.spec_from_file_location(
    "deposit_certificate", os.path.join(HERE, "certificate.py"))
cert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cert)

METHODS = ("inverted_cdf", "averaged_inverted_cdf", "closest_observation",
           "interpolated_inverted_cdf", "hazen", "weibull", "linear",
           "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint",
           "nearest")
ALPHAS = (F(1, 10), F(1, 20))


def level(alpha, rail, corr, n):
    """The exact level, rounded once to a double."""
    L = cert.RAILS[rail](alpha)
    if corr == "corrected":
        L = min(F(1), L * (n + 1) / n)
    return float(L)


def delivered(method, q, n):
    """Scores of 1..n at or below numpy's threshold: the rank a caller gets."""
    scores = np.arange(1, n + 1, dtype=np.float64)
    v = float(np.quantile(scores, q, method=method))
    return int(np.sum(scores <= v))


def float_candidate(name, n, alpha):
    """A certificate candidate under the float standard (the 'direct' family
    reads an index, so it is its exact rule)."""
    rail, corr, fam = name.split("/")
    if fam == "direct":
        return dict(cert.RULES)[name](n, alpha)
    r = delivered(fam, level(alpha, rail, corr, n), n)
    return r if 1 <= r <= n else cert.OUT_OF_RANGE


def required(alpha, rail, n):
    return math.ceil((n + 1) * cert.RAILS[rail](alpha))


def self_check():
    # The rank counter can fail: a threshold of 45.5 on 1..50 delivers 45, not 46.
    s = np.arange(1, 51, dtype=np.float64)
    assert int(np.sum(s <= 45.5)) == 45
    # numpy 'higher' at q=0.9, n=50 is the 46th score, which is k* there.
    assert delivered("higher", 0.9, 50) == 46 == required(F(1, 10), "one-sided", 50)
    # hazen at q=0.9, n=10 has virtual index 9.5: delivered rank 9, one short of 10.
    assert delivered("hazen", 0.9, 10) == 9


self_check()


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say("WHAT THE FOUR-SIZE CERTIFICATE CANNOT SEE: NUMPY RULES OUTSIDE THE SIXTEEN")
    say("=" * 100)
    say(f"numpy {np.__version__}; score set 1..n; level = exact rational rounded once to a double")
    say("A numpy rule is SCREENED AS a candidate when it delivers that candidate's rank at")
    say("every certificate size; it is BLIND when it is screened as one yet differs from it")
    say("somewhere in the complete checked range. short = sizes in the range where the")
    say("numpy rule delivers fewer than k* ranks.")
    say("")
    totals = {}
    for alpha in ALPHAS:
        d = cert.period_of(alpha)
        lo = cert.feasible_floor(alpha) + 1
        n0 = cert.periodic_from(alpha)
        wide = list(range(lo, n0 + 2 * d))
        groups = cert.classes(alpha, tuple(wide))
        S = cert.minimal_certificate(alpha, wide, len(groups))
        say("-" * 100)
        say(f"alpha = {alpha}   certificate S = {S}   complete range {lo}..{wide[-1]} "
            f"({len(wide)} sizes)")
        say("-" * 100)
        # where the float standard and the exact rule disagree for a candidate
        drift = 0
        for name, fn in cert.RULES:
            for n in wide:
                if float_candidate(name, n, alpha) != fn(n, alpha):
                    drift += 1
                    say(f"DRIFT alpha={alpha} rule={name} n={n} "
                        f"exact={fn(n, alpha)} float={float_candidate(name, n, alpha)}")
        say(f"FLOATDRIFT alpha={alpha} cells={drift}")
        cand = {name: [float_candidate(name, n, alpha) for n in wide]
                for name, _ in cert.RULES}
        idx = [wide.index(n) for n in S]
        screened = blind = 0
        for rail in cert.RAILS:
            for corr in cert.CORRECTIONS:
                for m in METHODS:
                    obs = [delivered(m, level(alpha, rail, corr, n), n) for n in wide]
                    k = [required(alpha, rail, n) for n in wide]
                    short = sum(o < kk for o, kk in zip(obs, k))
                    hits = [name for name, c in cand.items()
                            if name.startswith(rail + "/")
                            and all(c[i] == obs[i] for i in idx)]
                    if not hits:
                        say(f"RULE alpha={alpha} rail={rail} corr={corr} method={m} "
                            f"screened_as=none short={short}")
                        continue
                    screened += 1
                    name = hits[0]
                    diff = [n for n, o, c in zip(wide, obs, cand[name]) if o != c]
                    diff_short = sum(1 for n, o, kk in zip(wide, obs, k)
                                     if n in diff and o < kk)
                    if diff:
                        blind += 1
                    say(f"RULE alpha={alpha} rail={rail} corr={corr} method={m} "
                        f"screened_as={name} differs={len(diff)} "
                        f"differs_short={diff_short} short={short} "
                        f"first_diff={diff[0] if diff else 'none'}")
        say(f"TOTAL alpha={alpha} rules={len(cert.RAILS) * len(cert.CORRECTIONS) * len(METHODS)} "
            f"screened={screened} blind={blind}")
        totals[alpha] = (screened, blind)
        say("")
    say("Read: a BLIND rule passes the four-size screen as a candidate it is not. The")
    say("full checked range is the criterion; the certificate is a screen for the")
    say("sixteen candidates; a rule that is none of them is outside its reach.")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
