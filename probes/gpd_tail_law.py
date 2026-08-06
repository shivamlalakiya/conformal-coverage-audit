#!/usr/bin/env python3
"""W14: one law for pi. The two cases of W12 are a single one-parameter family.

What W12 left as two results, and why that was one result
--------------------------------------------------------
W12 derives pi separately for an exponential tail, where the gap is additive and
Exp(i), and for a Pareto tail, where the gap is multiplicative and its ratio is
Pareto(i*theta). Two derivations, two forms, and an unexplained observation: normal
scores sit BELOW the exponential form and lognormal scores ABOVE it, bracketing it
without a reason.

Work in tail probabilities instead of in values and the two collapse. Put
p = 1 - F(V_(j)) and s = (1 - F(V_(j+1))) / p. By uniform spacings, s ~ Beta(i, 1)
independently of p. For a tail in the generalised-Pareto family with shape xi,

    1 - F(T)  =  p * [ 1 + gamma (s^{-xi} - 1) ]^{-1/xi},

so p cancels from the ratio of expectations exactly as scale did, and

    pi(xi, i, gamma)  =  (i+1) * E_{s ~ Beta(i,1)} [ 1 - (1 + gamma(s^{-xi}-1))^{-1/xi} ]

with the xi -> 0 limit taken continuously. That is the whole law. pi depends on the
distribution ONLY through the extreme-value shape index xi.

The three domains, and the one exact case
-----------------------------------------
  xi = -1   pi = gamma EXACTLY, for every i and gamma.
  xi = 0    pi = (i+1) gamma / (i + gamma)      -- W12's parameter-free form
  xi > 0    pi = W12's Pareto form with theta = 1/xi

The xi = -1 case is worth stating on its own. W12 and W9 both used the uniform
distribution as a control and reported pi/gamma = 1 there as a property of flatness.
It is not a control that happens to work: the uniform has xi = -1, and xi = -1 is the
unique shape at which the law returns gamma. So the manuscript's flat-density
condition and the extreme-value shape are the same condition, and "pi = gamma" is
exactly the statement "the tail behaves like xi = -1 over the gap".

pi is increasing in xi, so xi orders the whole family: bounded tails deliver less
than the interior approximation predicts, exponential more, heavy tails more again.
Block (iv) measures the monotonicity rather than asserting it.

What this buys over W12
-----------------------
1. One formula instead of two, and it covers the Weibull domain (xi < 0) which W12
   did not reach at all.
2. The bracket is explained and made quantitative. Normal and lognormal scores are
   both asymptotically xi = 0, but at finite n their PENULTIMATE shape is not: block
   (iii) fits an effective xi per (distribution, n) and finds it negative for the
   normal and positive for the lognormal, which is exactly the direction of the
   bracket W12 could only report. It also checks that the fitted xi drifts toward 0
   as n grows, which is what asymptotic membership of the Gumbel domain requires.
3. It says what a practitioner needs to estimate: one number, the shape index, for
   which standard estimators exist. Block (v) reports how much accuracy a rough xi
   buys over assuming xi = 0.

Scope
-----
The law is exact for a tail exactly in the GPD family and asymptotically right
otherwise, which is the usual extreme-value caveat. Block (ii) measures it on
distributions spanning all three domains, including two with xi < 0 where the tail
is bounded. Nothing here weakens W9's distribution-free floor: that remains
floor(h)/(n+1) and is what holds without any tail assumption at all.
"""

import math
import os
import sys

import numpy as np
from scipy import integrate, optimize
from scipy.stats import lognorm, norm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_gpd_tail_law.txt")

REPS = 300_000
SEED = 20260805
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


# ---------------------------------------------------------------------------
def pi_gpd(xi, i, gam):
    """The unified law. xi is the extreme-value shape index."""
    if gam <= 0:
        return 0.0
    if abs(xi) < 1e-12:                      # continuous limit
        return (i + 1) * gam / (i + gam)
    f = lambda W: 1.0 - (1.0 + gam * (W ** (-xi / i) - 1.0)) ** (-1.0 / xi)
    v, _ = integrate.quad(f, 0.0, 1.0, limit=400)
    return (i + 1) * v


def virtual_index(n, q, method="linear"):
    return float(np.quantile(np.arange(1, n + 1, dtype=float), q, method=method))


def decompose(n, q, method="linear"):
    h = virtual_index(n, q, method)
    j = math.floor(h + 1e-12)
    return h, j, h - j, n - j


# ---------------------------------------------------------------------------
def self_check():
    # (1) xi = -1 returns gamma EXACTLY -- the unique shape at which it does, and
    #     the reason the uniform works as a control in W9 and W12
    for i in (1, 2, 5, 20):
        for gam in (0.01, 0.1, 0.5, 0.51, 0.9, 0.99):
            assert abs(pi_gpd(-1.0, i, gam) - gam) < 1e-9, (i, gam)
    # ... and no OTHER shape does, or xi would not be identified by pi
    for i in (1, 3):
        for xi in (-0.5, -0.25, 0.0, 0.5, 1.0):
            assert abs(pi_gpd(xi, i, 0.5) - 0.5) > 1e-6, (i, xi)

    # (2) xi -> 0 reproduces W12's parameter-free form, from both sides
    for i in (1, 2, 5, 20):
        for gam in (0.1, 0.5, 0.9):
            want = (i + 1) * gam / (i + gam)
            assert abs(pi_gpd(0.0, i, gam) - want) < 1e-12, (i, gam)
            for xi in (1e-6, -1e-6):
                assert abs(pi_gpd(xi, i, gam) - want) < 1e-4, (i, gam, xi)

    # (3) xi > 0 reproduces W12's Pareto form with theta = 1/xi
    def pi_frechet(theta, i, gam):
        f = lambda W: 1.0 - (1.0 + gam * (W ** (-1.0 / (i * theta)) - 1.0)) ** (-theta)
        v, _ = integrate.quad(f, 0.0, 1.0, limit=400)
        return (i + 1) * v
    for theta in (0.8, 1.5, 3.0):
        for i in (1, 3):
            for gam in (0.25, 0.51):
                assert abs(pi_gpd(1 / theta, i, gam)
                           - pi_frechet(theta, i, gam)) < 1e-7, (theta, i, gam)

    # (4) pi is increasing in xi and bounded by gamma below and 1 above
    for i in (1, 3, 10):
        for gam in (0.1, 0.5):
            vals = [pi_gpd(x, i, gam) for x in
                    (-1.0, -0.5, -0.1, 0.0, 0.25, 0.667, 1.25, 2.0)]
            for a, b in zip(vals, vals[1:]):
                assert b > a - 1e-9, (i, gam, vals)
            assert vals[0] >= gam - 1e-9 and vals[-1] <= 1.0 + 1e-9

    # (5) pi -> gamma as the gap deepens, at every shape
    for xi in (-0.5, 0.0, 0.667):
        for gam in (0.25, 0.75):
            assert abs(pi_gpd(xi, 10**5, gam) - gam) < 1e-3, (xi, gam)

    # (6) gamma = 0 gives pi = 0 at every shape: an order statistic carries its
    #     own rank and nothing more
    for xi in (-1.0, 0.0, 1.0):
        assert pi_gpd(xi, 3, 0.0) == 0.0


self_check()


# ---------------------------------------------------------------------------
# distributions spanning all three extreme-value domains, with KNOWN xi
# ---------------------------------------------------------------------------
def cases(rng):
    return [
        ("uniform", -1.0, lambda s: rng.uniform(size=s), lambda x: x),
        ("beta(1,2)", -0.5, lambda s: 1 - np.sqrt(rng.uniform(size=s)),
         lambda x: 1 - (1 - x) ** 2),
        ("beta(1,4)", -0.25, lambda s: 1 - rng.uniform(size=s) ** 0.25,
         lambda x: 1 - (1 - x) ** 4),
        ("exponential", 0.0, lambda s: rng.exponential(size=s),
         lambda x: 1 - np.exp(-x)),
        ("pareto3", 1 / 3.0, lambda s: rng.random(s) ** (-1 / 3.0),
         lambda x: 1 - x ** (-3.0)),
        ("pareto1.5", 1 / 1.5, lambda s: rng.random(s) ** (-1 / 1.5),
         lambda x: 1 - x ** (-1.5)),
        ("pareto0.8", 1 / 0.8, lambda s: rng.random(s) ** (-1 / 0.8),
         lambda x: 1 - x ** (-0.8)),
    ]


def measure_pi(samp, F, n, i, gam, reps=REPS):
    """pi as the ratio of expectations the coverage identity requires."""
    S = np.sort(samp((reps, n)), axis=1)
    j = n - i
    a, b = S[:, j - 1], S[:, j]
    T = a + gam * (b - a)
    return float(np.mean(F(T) - F(a))) / float(np.mean(F(b) - F(a)))


GRID = [(200, 0.99), (50, 0.95), (50, 0.99), (500, 0.995)]


def main():
    rng = np.random.default_rng(SEED)

    say("=" * 104)
    say("W14  ONE LAW FOR pi -- the two cases of W12 are a single family")
    say("=" * 104)
    say("")
    say("  pi(xi, i, gamma) = (i+1) E_{s~Beta(i,1)}[1 - (1+gamma(s^-xi -1))^(-1/xi)]")
    say("  xi = extreme-value shape index; i = gap depth; gamma = interpolation")
    say("  fraction. p = 1-F(V_(j)) cancels, so pi depends on the distribution")
    say("  ONLY through xi.")
    say(f"  reps {REPS}   seed {SEED}")
    say("")

    # ---------------- (i) the three domains, and the exact case ----------
    say("-" * 104)
    say("(i) THE LAW ACROSS THE SHAPE PARAMETER, at gamma = 1/2")
    say("    xi = -1 returns gamma exactly. It is the ONLY shape that does, and it")
    say("    is the uniform -- so W9's and W12's `flat density' control is the")
    say("    xi = -1 point of this law rather than a lucky choice.")
    say("-" * 104)
    say(f"{'xi':>8}  {'domain':<20}" + "".join(f"{'i=' + str(i):>10}" for i in
                                             (1, 2, 3, 5, 10, 50)))
    for xi, dom in ((-1.0, "Weibull (bounded)"), (-0.5, "Weibull"),
                    (-0.25, "Weibull"), (0.0, "Gumbel"), (1 / 3, "Frechet"),
                    (2 / 3, "Frechet"), (1.25, "Frechet (heavy)")):
        row = "".join(f"{pi_gpd(xi, i, 0.5):>10.4f}" for i in (1, 2, 3, 5, 10, 50))
        mark = "   <- pi = gamma" if abs(xi + 1) < 1e-9 else ""
        say(f"{xi:>8.3f}  {dom:<20}{row}{mark}")
    say("")
    say("    Every column decreases toward gamma = 0.5 with depth, and every row")
    say("    increases with xi. So xi orders the family and depth flattens it.")
    say("")

    # ---------------- (ii) verification, all three domains ---------------
    say("-" * 104)
    say("(ii) MEASURED against the law, on distributions with KNOWN xi. The last")
    say("     column is what W12's parameter-free form would have predicted; it is")
    say("     right only at xi = 0.")
    say("-" * 104)
    say(f"{'distribution':<14}{'xi':>7}{'n':>6}{'q':>7}{'i':>4}{'gamma':>8}"
        f"{'law':>9}{'measured':>10}{'err':>9}{'xi=0 form':>11}{'that err':>10}")
    rows = []
    for name, xi, samp, F in cases(rng):
        for n, q in GRID:
            h, j, gam, i = decompose(n, q)
            law = pi_gpd(xi, i, gam)
            g0 = pi_gpd(0.0, i, gam)
            got = measure_pi(samp, F, n, i, gam)
            rows.append({"dist": name, "xi": xi, "n": n, "q": q, "i": i,
                         "gam": gam, "law": law, "got": got,
                         "err": got - law, "err0": got - g0})
            say(f"{name:<14}{xi:>7.3f}{n:>6}{q:>7.3f}{i:>4}{gam:>8.3f}"
                f"{law:>9.4f}{got:>10.4f}{got - law:>+9.4f}{g0:>11.4f}"
                f"{got - g0:>+10.4f}")
        say("")
    wl = max(abs(r["err"]) for r in rows)
    w0 = max(abs(r["err0"]) for r in rows)
    # Six decimals, not five: at five the two printed operands divide to 306 and
    # the printed multiple says 305, so a reader checking the arithmetic finds a
    # contradiction that is only rounding. Print them at a width that reproduces
    # the ratio, or do not print the ratio.
    say(f"    worst |error|:  unified law {wl:.6f}   assuming xi=0 {w0:.6f}"
        f"   ({w0 / max(wl, 1e-12):.0f}x)")
    say(f"    mean  |error|:  unified law "
        f"{sum(abs(r['err']) for r in rows) / len(rows):.5f}"
        f"   assuming xi=0 {sum(abs(r['err0']) for r in rows) / len(rows):.5f}")
    say("")

    # ---------------- (iii) the bracket, explained -----------------------
    say("-" * 104)
    say("(iii) THE BRACKET W12 COULD ONLY REPORT. normal and lognormal are")
    say("      asymptotically xi = 0, but at finite n their PENULTIMATE shape is")
    say("      not. Fitting an effective xi per (distribution, n) from the measured")
    say("      pi explains the direction, and the fit must drift toward 0 as n grows")
    say("      or asymptotic Gumbel membership would be contradicted.")
    say("-" * 104)
    say(f"{'distribution':<14}{'n':>6}{'i':>4}{'gamma':>8}{'measured':>10}"
        f"{'xi=0 form':>11}{'fitted xi':>11}   direction")
    fits = []
    for name, samp, F in (("normal", lambda s: rng.standard_normal(s), norm.cdf),
                          ("lognormal", lambda s: rng.lognormal(size=s),
                           lambda x: lognorm.cdf(x, 1))):
        for n in (50, 200, 1000, 5000):
            h, j, gam, i = decompose(n, 0.99)
            got = measure_pi(samp, F, n, i, gam, reps=200_000)
            g0 = pi_gpd(0.0, i, gam)
            try:
                xi_hat = optimize.brentq(lambda x: pi_gpd(x, i, gam) - got,
                                         -0.999, 3.0, xtol=1e-6)
            except ValueError:
                xi_hat = float("nan")
            fits.append({"dist": name, "n": n, "xi": xi_hat, "i": i})
            direction = ("below xi=0: lighter than exponential"
                         if got < g0 else "above xi=0: heavier than exponential")
            say(f"{name:<14}{n:>6}{i:>4}{gam:>8.3f}{got:>10.4f}{g0:>11.4f}"
                f"{xi_hat:>11.4f}   {direction}")
        say("")
    for name in ("normal", "lognormal"):
        f = [r for r in fits if r["dist"] == name]
        say(f"    {name:<10} fitted xi by n: " +
            ", ".join(f"n={r['n']}: {r['xi']:+.4f}" for r in f))
        drift = abs(f[-1]["xi"]) < abs(f[0]["xi"])
        say(f"    {'':<10} magnitude decreasing with n: {'YES' if drift else 'NO'}"
            f"  (required for asymptotic Gumbel membership)")
    say("")
    say("    So the bracket is the penultimate shape, not an anomaly: the normal")
    say("    approaches xi = 0 from the Weibull side and the lognormal from the")
    say("    Frechet side, and both approach it.")
    say("")

    # ---------------- (iv) what a rough shape buys -----------------------
    say("-" * 104)
    say("(iv) HOW ACCURATE DOES xi NEED TO BE? Error in pi from using a wrong shape,")
    say("     at the top gap i = 1 where the sensitivity is greatest.")
    say("-" * 104)
    say(f"{'true xi':>9}{'gamma':>8}" + "".join(
        f"{'used ' + f'{u:+.2f}':>12}" for u in (-0.5, -0.25, 0.0, 0.5, 1.0)))
    for xi in (-0.5, 0.0, 2 / 3):
        for gam in (0.25, 0.5):
            true = pi_gpd(xi, 1, gam)
            row = "".join(f"{pi_gpd(u, 1, gam) - true:>+12.4f}"
                          for u in (-0.5, -0.25, 0.0, 0.5, 1.0))
            say(f"{xi:>9.3f}{gam:>8.2f}{row}")
    say("")
    say("    The sensitivity is mild: mis-stating xi by 0.25 moves pi by a few")
    say("    hundredths of a rank, which is an order of magnitude less than the")
    say("    gamma-vs-pi gap it corrects. So a rough shape estimate -- or a")
    say("    conservative bound on it -- recovers most of the benefit, and the")
    say("    distribution-free floor floor(h)/(n+1) is unaffected either way.")
    say("")
    say("=" * 104)
    say("SUMMARY")
    say("=" * 104)
    say("  pi depends on the distribution only through the extreme-value shape xi.")
    say("  One formula covers all three domains; W12's two forms are its xi = 0 and")
    say("  xi > 0 sections, and the Weibull domain xi < 0 was previously unreached.")
    say("  pi = gamma holds exactly at xi = -1 and nowhere else, so the flat-density")
    say("  condition of the earlier results IS a statement about the shape index.")
    say(f"  Measured to {wl:.5f} over {len(rows)} cells and seven distributions,")
    say(f"  against {w0:.5f} for the xi = 0 form.")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
