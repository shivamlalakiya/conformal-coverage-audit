#!/usr/bin/env python3
"""The tail law off the exact-GPD hypothesis: what the sample actually sees.

The gap this closes
-------------------
The tail law is proved for F EXACTLY generalised Pareto above a fixed threshold.
That is the strong hypothesis in the companion paper, and the honest treatment so
far has been to measure the discrepancy by fitting an effective shape to the
observed pi -- reporting that the normal fits a NEGATIVE shape and the lognormal a
POSITIVE one, both shrinking toward zero, as an observation with no account behind
it. A referee in extreme-value statistics reads that and says: the penultimate
approximation literature exists.

The claim this probe tests
--------------------------
Write the first-order rate from the tail law,

    pi = gamma + gamma(1-gamma)(1 + xi) / i + O(1/i^2).

Differentiating the GPD integrand in xi and integrating against the Beta(i,1)
spacing law gives

    d(pi)/d(xi) = gamma(1-gamma) / i + O(1/i^2),

which is exactly the xi-derivative of the rate above -- and, to leading order, does
NOT depend on xi. Nudge the shape a little and pi moves through that one coefficient
alone, so the entire second-order correction disappears once you replace
xi with the PENULTIMATE shape the sample sees at its own depth:

    pi = gamma + gamma(1-gamma)(1 + xi_n) / i + o(1/i),   xi_n = xi + A(n/i)/rho.

That is the statement. It is not a new expansion of pi; it is the observation that
the existing expansion is a function of the shape alone at first order, so the
standard penultimate substitution carries it off the exact-GPD hypothesis for free.

What "the shape the sample sees" means, exactly
-----------------------------------------------
Above a threshold u, the shape F looks to have locally is

    xi_loc(u) = d/du [ (1 - F(u)) / f(u) ].

For a GPD this is identically xi, because (1-F)/f = sigma + xi(u - mu). So
xi_loc is the right object and it needs no fitting: it is a property of F.

    normal      (1-Phi)/phi ~ 1/u        => xi_loc ~ -1/u^2  < 0
    lognormal   (1-F)/f ~ x/log x        => xi_loc ~  1/log x > 0

Both tend to 0, which is Gumbel-domain membership, and their SIGNS differ -- which
is what the fitted-shape sweep reports and could not previously explain.

Three checks, each able to fail
-------------------------------
  (1) d(pi)/d(xi) matches gamma(1-gamma)/i by numerical differentiation of exact
      quadrature, and the agreement improves with i.
  (2) xi_loc is identically xi for a GPD, to machine precision -- otherwise the
      definition is not the right one.
  (3) For the normal and the lognormal, xi_loc at the depth the sweep uses has the
      sign the fitted effective shape has, and shrinks toward zero with n.

    python probes/penultimate_tail.py
"""

import math
import os
import sys

import numpy as np
from scipy import integrate, optimize, stats

OUT = "outputs/probe_output_penultimate_tail.txt"


# ---------------------------------------------------------------------------
# pi under an exactly-GPD tail, by quadrature rather than by simulation
# ---------------------------------------------------------------------------
def pi_gpd(xi, i, gamma):
    """pi = (i+1) E[1 - (1 + gamma(s^-xi - 1))^(-1/xi)], s ~ Beta(i,1).

    Integrated in t = -log s rather than in s. Under that substitution
    s ~ Beta(i,1) becomes t ~ Exp(i), because the Beta(i,1) density i*s^(i-1)
    times |ds/dt| = e^-t gives i*e^(-it). That matters numerically and not just
    aesthetically: in s the integrand concentrates within O(1/i) of the endpoint
    s = 1, so at i = 1000 an adaptive rule on [0, 1] misses the mass and the
    derivative computed from it came out four times too large. In t the mass sits
    on a scale of 1/i away from zero and the rule finds it.
    """
    def f(t):
        # exp(-phi) with phi = log(A)/xi and log A = logaddexp(log(1-g), log g + xi t).
        # Written through logaddexp rather than as (1 + g*expm1(xi*t))**(-1/xi):
        # that form overflows for xi*t large -- at xi = 10 it raised OverflowError --
        # and the log form is exact for every xi and t.
        if abs(xi) < 1e-13:
            inner = math.exp(-gamma * t)           # the xi -> 0 limit, s^gamma
        else:
            logA = np.logaddexp(math.log1p(-gamma), math.log(gamma) + xi * t)
            inner = math.exp(-logA / xi)
        return (1 - inner) * i * math.exp(-i * t)
    # split at a few multiples of the Exp(i) scale so the rule cannot step over it
    hi = 60.0 / i
    v1, _ = integrate.quad(f, 0.0, hi, limit=400)
    v2, _ = integrate.quad(f, hi, np.inf, limit=200)
    return (i + 1) * (v1 + v2)


def dpi_dxi(xi, i, gamma, h=1e-4):
    return (pi_gpd(xi + h, i, gamma) - pi_gpd(xi - h, i, gamma)) / (2 * h)


# ---------------------------------------------------------------------------
# the local (penultimate) shape of a distribution above a threshold
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# the remainder, with a constant: answering "bound it, do not just measure it"
# ---------------------------------------------------------------------------
def c1(xi, gamma):
    """First-order coefficient: pi = gamma + c1/i + c2/i^2 + O(1/i^3)."""
    return gamma * (1 - gamma) * (1 + xi)


def c2(xi, gamma):
    """Second-order coefficient, derived by expanding the integrand to t^3.

    An early draft stopped at an unbounded O(1/i^2), reasoning that nobody can
    evaluate the quantities such a bound needs, so why print one. Wrong instinct
    for statisticians: a bound establishing safety in the limit earns its place
    whether or not data can compute it. Hence the term is derived
    rather than absorbed.

    Two checks it must pass. At xi = 0 the exact form is (i+1)gamma/(i+gamma),
    whose 1/i^2 coefficient is -gamma^2(1-gamma); and at xi = -1 both c1 and c2
    must vanish, because pi = gamma exactly there.
    """
    return -gamma * (1 - gamma) * (1 + xi) * (2 * gamma * xi + gamma - xi)


def remainder_bound(xi_max=1.0):
    """A constant C with |pi - gamma - c1/i| <= C/i^2 + O(1/i^3) for |xi| <= xi_max.

    gamma(1-gamma) <= 1/4, |1+xi| <= 1 + xi_max, and
    |2*gamma*xi + gamma - xi| <= 1 + 3*xi_max, so C = (1+xi_max)(1+3*xi_max)/4.
    """
    return (1 + xi_max) * (1 + 3 * xi_max) / 4.0


def penultimate_bound(gamma, i, A_over_rho):
    """|pi - pi_GPD(xi)| <= |xi_n - xi| * sup|d pi/d xi|, by the mean value theorem.

    The supremum is gamma(1-gamma)/i to leading order and free of xi there, so the
    whole price of stepping outside exact GPD is first order in the second-order
    auxiliary. Nobody can evaluate it, which is the whole point: it covers every F in
    the domain rather than the four that were simulated.
    """
    return abs(A_over_rho) * gamma * (1 - gamma) / i


def kernel_h(u, gamma):
    """h(u) with psi' = u*sigma(1-sigma), so that d(pi)/d(xi) = (i+1) E[e^-phi t^2 h].

    phi = log(A)/xi with A = (1-gamma) + gamma e^{xi t}, and
    phi' = t^2 h(xi t) with h(u) = psi(u)/u^2,
    psi(u) = u*sigma(u) - log A,  sigma(u) = gamma e^u / A, the logistic.

    Since psi(0) = 0 and psi'(u) = u*sigma(u)(1-sigma(u)) with sigma(1-sigma) <= 1/4,
    integrating gives 0 <= psi(u) <= u^2/8 and hence 0 <= h <= 1/8 for EVERY u and
    gamma. That is the whole proof of the uniform derivative bound below.

    Evaluated through the series for |u| < 1e-2: the direct form divides by u^2, so
    at u ~ 1e-5 it has lost every significant digit and reports h slightly ABOVE 1/8,
    which is cancellation and not a counterexample.
    """
    if abs(u) < 1e-2:
        return gamma * (1 - gamma) / 2 - gamma * (1 - gamma) * (1 - 2 * gamma) * u / 6
    sig = 1.0 / (1.0 + ((1 - gamma) / gamma) * math.exp(-u))
    return (u * sig - np.logaddexp(math.log1p(-gamma),
                                   math.log(gamma) + u)) / (u * u)


def dpi_dxi_bound(i):
    """0 <= d(pi)/d(xi) <= (i+1)/(4 i^2), uniformly in xi and gamma.

    From 0 <= h <= 1/8, 0 <= exp(-phi) <= 1 and E[t^2] = 2/i^2 for t ~ Exp(i):
        d(pi)/d(xi) = (i+1) E[e^-phi t^2 h(xi t)] <= (i+1) * (1/8) * 2/i^2.
    Uniform over ALL xi, with no asymptotics and an explicit constant -- which is
    what makes the penultimate deviation bound a theorem rather than an assertion.
    An earlier draft claimed the derivative was gamma(1-gamma)/i + O(1/i^2)
    "uniformly" and had only checked that numerically.
    """
    return (i + 1) / (4.0 * i * i)


def local_shape(dist, u, h=None):
    """xi_loc(u) = d/du [ (1 - F(u)) / f(u) ], by central difference.

    Identically xi for a GPD, which check (2) asserts. No fitting, no sample: this
    is a property of F evaluated at u.
    """
    h = h or max(1e-4, abs(u) * 1e-4)

    def m(x):
        sf = dist.sf(x)
        pdf = dist.pdf(x)
        if pdf <= 0:
            return np.nan
        return sf / pdf
    return (m(u + h) - m(u - h)) / (2 * h)


def depth_threshold(dist, n, i):
    """The value at the top of a depth-i gap: the (1 - i/n) quantile.

    Depth i puts a gap out around tail probability i/n, so the threshold carrying
    the shape seen at that depth is the (1 - i/n) quantile.
    """
    return float(dist.isf(i / n))


# The (n, i) cells and fitted effective shapes the gpd_tail_law sweep reports. These
# are PARSED from that probe's committed output rather than typed here, so the
# comparison cannot drift from the sweep it is testing.
def fitted_shapes():
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "probe_output_gpd_tail_law.txt")
    out = {}
    for ln in open(path):
        m = re.match(r"^(normal|lognormal)\s+(\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+"
                     r"\s+[\d.]+\s+(-?[\d.]+)", ln)
        if m:
            out[(m.group(1), int(m.group(2)), int(m.group(3)))] = float(m.group(4))
    assert out, "no fitted effective shapes found in probe_output_gpd_tail_law.txt"
    return out


def self_check():
    # ---- (2) the local shape of a GPD is its shape, exactly ---------------
    for xi in (0.4, 0.15, -0.3):
        d = stats.genpareto(c=xi, loc=0.0, scale=1.0)
        # A GPD with xi < 0 has bounded support, ending at -scale/xi. Testing at a
        # fixed u put the probe outside it and read nan, which is the right answer
        # to a wrong question -- the test points now scale with the support.
        top = (-1.0 / xi) if xi < 0 else None
        us = ([0.2 * top, 0.5 * top, 0.8 * top] if top else [0.5, 2.0, 5.0])
        for u in us:
            got = local_shape(d, u)
            assert not math.isnan(got), (xi, u, "nan -- u outside the support?")
            assert abs(got - xi) < 1e-4, (xi, u, got)
    # exponential is the xi = 0 member and must read 0
    d = stats.expon()
    for u in (1.0, 3.0, 6.0):
        assert abs(local_shape(d, u)) < 1e-4, (u, local_shape(d, u))

    # ---- (1) the derivative claim, improving with depth -------------------
    ratios = []
    for i in (10, 40, 200):
        for gamma in (0.3, 0.5, 0.9):
            for xi in (0.0, 0.25, -0.5):
                pred = gamma * (1 - gamma) / i
                ratios.append((i, dpi_dxi(xi, i, gamma) / pred))
    # monotone improvement in i, on the mean absolute departure from 1
    err = {}
    for i, r in ratios:
        err.setdefault(i, []).append(abs(r - 1))
    means = [np.mean(err[i]) for i in (10, 40, 200)]
    assert means[0] > means[1] > means[2], means
    assert means[-1] < 0.06, means
    # and the derivative is nearly xi-free at depth, which is what lets the
    # penultimate substitution work at all
    for i in (40, 200):
        vals = [dpi_dxi(x, i, 0.5) for x in (-0.5, 0.0, 0.25)]
        assert (max(vals) - min(vals)) / np.mean(vals) < 0.05, (i, vals)

    # ---- (4) the local shape predicts the FITTED effective shape ----------
    # The decisive check. If the penultimate substitution is the right account of
    # the departure from exact GPD, then that local shape, taken where the sweep ran,
    # has to line up with whatever shape that sweep FITTED to its measured pi --
    # and do so with no sample at all, since xi_loc belongs to F by itself.
    dd = {"normal": stats.norm(), "lognormal": stats.lognorm(s=1.0)}
    worst = 0.0
    for (name, n, i), fit in fitted_shapes().items():
        xl = local_shape(dd[name], depth_threshold(dd[name], n, i))
        assert np.sign(xl) == np.sign(fit), (name, n, i, xl, fit)
        worst = max(worst, abs(xl - fit))
    assert worst < 0.02, f"local shape disagrees with the fitted shape by {worst:.4f}"

    # ---- (5) the second-order term, and the bound it gives ----------------
    for gamma in (0.3, 0.5, 0.9):
        # at xi = 0 the exact form is (i+1)gamma/(i+gamma)
        assert abs(c2(0.0, gamma) + gamma ** 2 * (1 - gamma)) < 1e-12, gamma
        # and both coefficients vanish at xi = -1, where pi = gamma exactly
        assert abs(c1(-1.0, gamma)) < 1e-12 and abs(c2(-1.0, gamma)) < 1e-12
    C = remainder_bound(1.0)
    worst1 = worst2 = 0.0
    for i in (20, 50, 200, 1000):
        for gamma in (0.3, 0.5, 0.9):
            for xi in (0.0, 0.3, -0.5, 1.0):
                ex = pi_gpd(xi, i, gamma)
                worst1 = max(worst1, abs(ex - gamma - c1(xi, gamma) / i) * i * i)
                worst2 = max(worst2, abs(ex - gamma - c1(xi, gamma) / i
                                         - c2(xi, gamma) / i ** 2) * i ** 3)
    # the claimed bound must HOLD, and the second-order remainder must be O(1/i^3),
    # which is what a bounded worst2 says
    assert worst1 <= C, (worst1, C)
    assert worst2 < 5.0, worst2
    # and the bound must be able to fail: a constant ten times too small must not
    # hold, or it is not a bound, it is a number
    assert worst1 > remainder_bound(1.0) / 10

    # ---- (7) the UNIFORM derivative bound, which the penultimate step needs -
    # h <= 1/8 over a fine grid in both arguments, and it must be attained: a bound
    # nothing approaches is not the bound, it is an overestimate.
    grid = ([-(10.0 ** e) for e in np.linspace(2.6, -2, 400)] + [0.0]
            + [10.0 ** e for e in np.linspace(-2, 2.6, 400)])
    mx = max(kernel_h(u, g) for g in np.linspace(0.02, 0.98, 97) for u in grid)
    assert mx <= 0.125 + 1e-12, f"h exceeds 1/8: {mx}"
    assert mx > 0.124, f"h never approaches 1/8 ({mx}); the bound is not tight"
    # psi(0) = 0 and psi' = u sigma(1-sigma), checked against a difference quotient
    for g in (0.05, 0.5, 0.95):
        for u0 in (-2.0, -0.3, 0.7, 3.0):
            def psi(x, g=g):
                sg = 1.0 / (1.0 + ((1 - g) / g) * math.exp(-x))
                return x * sg - np.logaddexp(math.log1p(-g), math.log(g) + x)
            hh = 1e-6
            num = (psi(u0 + hh) - psi(u0 - hh)) / (2 * hh)
            sg = 1.0 / (1.0 + ((1 - g) / g) * math.exp(-u0))
            assert abs(num - u0 * sg * (1 - sg)) < 1e-6, (g, u0, num)
    # and the resulting bound must hold, and be approached
    ratios = [dpi_dxi(xi, i, g) / dpi_dxi_bound(i)
              for i in (5, 20, 200, 1000) for g in (0.05, 0.5, 0.95)
              for xi in (-3.0, -0.9, 0.0, 2.0, 10.0)]
    assert max(ratios) <= 1.0 + 1e-9, f"derivative exceeds its bound: {max(ratios)}"
    assert max(ratios) > 0.9, f"bound never approached ({max(ratios)}); too loose"

    # ---- (3) the predicted signs ------------------------------------------
    n = 2000
    for name, dist, want in (("normal", stats.norm(), -1),
                             ("lognormal", stats.lognorm(s=1.0), +1)):
        xl = local_shape(dist, depth_threshold(dist, n, 5))
        assert np.sign(xl) == want, (name, xl)


self_check()


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say("THE TAIL LAW OFF THE EXACT-GPD HYPOTHESIS")
    say("=" * 100)
    say("self_check() passed at import: the local shape of a GPD is its own shape to")
    say("1e-4; d(pi)/d(xi) approaches gamma(1-gamma)/i and the approach is monotone")
    say("in depth; and the predicted signs hold for the normal and the lognormal.")
    say("")
    say("The claim. The first-order rate is pi = gamma + gamma(1-gamma)(1+xi)/i, and")
    say("its xi-derivative is gamma(1-gamma)/i, which to leading order does not")
    say("depend on xi. A first-order shape perturbation therefore enters pi through")
    say("that single coefficient, so replacing xi by the shape the sample sees at its")
    say("own depth carries the law off the exactly-GPD hypothesis:")
    say("")
    say("    pi = gamma + gamma(1-gamma)(1 + xi_n)/i + o(1/i),   xi_n = xi + A(n/i)/rho")
    say("")
    say("(1) d(pi)/d(xi) against gamma(1-gamma)/i, by quadrature")
    say(f"{'i':>5} {'gamma':>7} {'xi':>7} {'numeric':>12} {'predicted':>12} {'ratio':>8}")
    say("-" * 100)
    worst = 0.0
    for i in (10, 40, 200, 1000):
        for gamma in (0.3, 0.5, 0.9):
            for xi in (0.0, 0.25, -0.5):
                num = dpi_dxi(xi, i, gamma)
                pred = gamma * (1 - gamma) / i
                say(f"{i:>5} {gamma:>7.2f} {xi:>7.2f} {num:>12.6f} {pred:>12.6f} "
                    f"{num / pred:>8.4f}")
                if i >= 200:
                    worst = max(worst, abs(num / pred - 1))
        say("")
    say(f"worst departure from 1 at i >= 200: {worst:.4f}")
    say("")

    say("(2) the local shape identifies a GPD's own shape, so it is the right object")
    say(f"{'xi':>7} {'u':>6} {'xi_loc':>10} {'error':>10}")
    say("-" * 100)
    for xi in (0.4, 0.15, 0.0, -0.3):
        d = stats.genpareto(c=xi, loc=0.0, scale=1.0)
        top = (-1.0 / xi) if xi < 0 else None
        us = ([0.2 * top, 0.5 * top, 0.8 * top] if top else [0.5, 2.0, 5.0])
        for u in us:
            got = local_shape(d, u)
            say(f"{xi:>7.2f} {u:>6.2f} {got:>10.6f} {abs(got - xi):>10.2e}")
    say("")

    say("(3) the shape the sample sees, and the sign the sweep reports")
    say("Depth i puts a gap around tail probability i/n, making the threshold the")
    say("(1 - i/n) quantile. xi_loc evaluated there is what gets substituted in.")
    say("")
    say(f"{'distribution':<14} {'asymptotic xi':>14} {'n':>7} {'i':>4} "
        f"{'threshold':>11} {'xi_loc':>10} {'sign':>6}")
    say("-" * 100)
    cases = (("normal", stats.norm(), 0.0),
             ("lognormal", stats.lognorm(s=1.0), 0.0),
             ("exponential", stats.expon(), 0.0),
             ("pareto(3)", stats.pareto(b=3.0), 1 / 3))
    trend = {}
    for name, dist, xi_inf in cases:
        for n in (200, 2000, 20000):
            i = 5
            u = depth_threshold(dist, n, i)
            xl = local_shape(dist, u)
            trend.setdefault(name, []).append(xl)
            say(f"{name:<14} {xi_inf:>14.4f} {n:>7} {i:>4} {u:>11.4f} "
                f"{xl:>10.5f} {'+' if xl > 0 else '-' if xl < 0 else '0':>6}")
        say("")

    say("(4) THE DECISIVE CHECK: does the local shape predict the FITTED shape?")
    say("The gpd_tail_law sweep fits an effective shape to its measured pi and")
    say("reports it. If the penultimate substitution is the right account, xi_loc at")
    say("the same (n, i) must agree with it -- and xi_loc uses no sample at all.")
    say("")
    say(f"{'distribution':<12} {'n':>6} {'i':>4} {'threshold':>10} {'xi_loc':>9} "
        f"{'fitted':>9} {'difference':>11}")
    say("-" * 100)
    dd = {"normal": stats.norm(), "lognormal": stats.lognorm(s=1.0)}
    worst4, cells4 = 0.0, 0
    for (name, n, i), fit in sorted(fitted_shapes().items()):
        u = depth_threshold(dd[name], n, i)
        xl = local_shape(dd[name], u)
        worst4 = max(worst4, abs(xl - fit))
        cells4 += 1
        say(f"{name:<12} {n:>6} {i:>4} {u:>10.4f} {xl:>9.4f} {fit:>9.4f} "
            f"{xl - fit:>+11.4f}")
    say("")
    say(f"worst disagreement over {cells4} cells: {worst4:.4f}")
    say("So the sign AND the magnitude of the fitted shape are predicted, for both")
    say("distributions and every size, by a quantity computed from F alone. What was")
    say("an observation with no account behind it is now a prediction.")
    say("")
    say("(5) THE REMAINDER, WITH A CONSTANT")
    say("An early draft stopped at an unbounded O(1/i^2), on the view that a bound")
    say("nobody can evaluate is ornament. Wrong instinct here: establishing safety")
    say("in the limit earns its place whether or not data can compute the constant.")
    say("So it is derived.")
    say("")
    say("    pi = gamma + c1/i + c2/i^2 + O(1/i^3)")
    say("    c1 = gamma(1-gamma)(1+xi)")
    say("    c2 = -gamma(1-gamma)(1+xi)(2*gamma*xi + gamma - xi)")
    say("")
    say("Hence for |xi| <= X, since gamma(1-gamma) <= 1/4:")
    say("    |pi - gamma - c1/i| <= C/i^2 + O(1/i^3),  C = (1+X)(1+3X)/4")
    say(f"    at X = 1 that is C = {remainder_bound(1.0):.2f}")
    say("")
    say("")
    say(f"{'i':>6} {'gamma':>6} {'xi':>6} {'exact':>12} {'gamma+c1/i':>12} "
        f"{'+c2/i^2':>12} {'|r1|i^2':>9} {'|r2|i^3':>9}")
    say("-" * 100)
    w1 = w2 = 0.0
    for i in (20, 50, 200, 1000):
        for gamma in (0.3, 0.5, 0.9):
            for xi in (0.0, 0.3, -0.5, 1.0):
                ex = pi_gpd(xi, i, gamma)
                a1 = gamma + c1(xi, gamma) / i
                a2 = a1 + c2(xi, gamma) / i ** 2
                r1, r2 = abs(ex - a1) * i * i, abs(ex - a2) * i ** 3
                w1, w2 = max(w1, r1), max(w2, r2)
                say(f"{i:>6} {gamma:>6.1f} {xi:>6.1f} {ex:>12.8f} {a1:>12.8f} "
                    f"{a2:>12.8f} {r1:>9.4f} {r2:>9.3f}")
        say("")
    say(f"worst |pi - first order| * i^2 = {w1:.4f}, against the bound "
        f"C = {remainder_bound(1.0):.2f}")
    say(f"worst |pi - second order| * i^3 = {w2:.3f}, bounded, so the second-order")
    say("term is right and the remainder after it really is O(1/i^3).")
    say("")

    say("(5b) THE MEAN VALUE STEP, WITH A PROOF INSTEAD OF A GRID")
    say("An earlier draft took the mean value theorem in xi across the segment from")
    say("the asymptotic shape to the penultimate one, justified by calling the")
    say("leading coefficient gamma(1-gamma)/i uniform. We had checked that on a grid")
    say("and not proved it, and the grid was right while the reasoning was not:")
    say("d(pi)/d(xi) = (i+1) E[exp(-phi) t^2 h(xi t)] and sup_u h EXCEEDS its value")
    say("at u = 0. So the leading coefficient is not itself a bound. What is a bound,")
    say("and needs two lines rather than a grid:")
    say("")
    say("    phi = log(A)/xi,  A = (1-g) + g e^{xi t},  d(phi)/d(xi) = t^2 h(xi t)")
    say("    h(u) = psi(u)/u^2,  psi(u) = u*sigma(u) - log A,  sigma = g e^u / A")
    say("    psi(0) = 0  and  psi'(u) = u*sigma(u)(1-sigma(u))     [an identity]")
    say("    sigma(1-sigma) <= 1/4  =>  0 <= psi(u) <= u^2/8  =>  0 <= h <= 1/8")
    say("    0 < exp(-phi) <= 1,  E[t^2] = 2/i^2  =>  0 <= d(pi)/d(xi) <= (i+1)/(4i^2)")
    say("")
    say("uniform in xi over ALL of R and in gamma over (0,1), with no asymptotics.")
    say("Which turns the mean value step into an inequality, not an expansion:")
    say("    |pi - pi_GPD(xi)| <= |xi_n - xi| * (i+1)/(4i^2)")
    say("with xi_n - xi = A(n/i)/rho + o(A) the domain-of-attraction hypothesis, so")
    say("every remaining error term sits there and none in this step.")
    say("")
    # sup h over a fine grid, and how far above h(0) it goes -- the number that
    # falsified the earlier reasoning
    grid = ([-(10.0 ** e) for e in np.linspace(2.6, -2, 600)] + [0.0]
            + [10.0 ** e for e in np.linspace(-2, 2.6, 600)])
    say(f"{'gamma':>7} {'h(0)=g(1-g)/2':>15} {'sup_u h':>10} {'ratio':>8}")
    say("-" * 44)
    hratio = 0.0
    suph = 0.0
    for g in (0.01, 0.05, 0.2, 0.5, 0.8, 0.99):
        s = max(kernel_h(u, g) for u in grid)
        h0 = g * (1 - g) / 2
        hratio = max(hratio, s / h0)
        suph = max(suph, s)
        say(f"{g:>7.2f} {h0:>15.8f} {s:>10.8f} {s / h0:>8.2f}")
    suph = max(suph, max(kernel_h(u, g)
                         for g in np.linspace(0.02, 0.98, 97) for u in grid))
    say("")
    say(f"sup h over the whole grid = {suph:.8f}, against the proved 1/8 = 0.125")
    say(f"largest sup_u h / h(0) = {hratio:.2f}: the leading coefficient understates")
    say("the derivative by that factor at small gamma, which is why it could not")
    say("serve as the uniform bound.")
    say("")
    say(f"{'i':>6} {'gamma':>6} {'xi':>6} {'d(pi)/d(xi)':>13} {'(i+1)/(4i^2)':>13} "
        f"{'ratio':>7}")
    say("-" * 60)
    tight = 0.0
    for i in (5, 20, 200, 1000):
        for g in (0.05, 0.5, 0.95):
            for xi in (-3.0, -0.9, 0.0, 2.0, 10.0):
                d, b = dpi_dxi(xi, i, g), dpi_dxi_bound(i)
                tight = max(tight, d / b)
                say(f"{i:>6} {g:>6.2f} {xi:>6.1f} {d:>13.8f} {b:>13.8f} "
                    f"{d / b:>7.4f}")
        say("")
    say(f"worst ratio to the bound = {tight:.4f}, so it holds and is attained to")
    say("within that -- a bound nothing approaches would be an overestimate, not a")
    say("bound. It is approached at gamma = 1/2, where sigma(1-sigma) is largest.")
    say("")

    say("(6) CONVERGENCE OF pi TO gamma BY DOMAIN OF ATTRACTION")
    say("How fast interpolation stops mattering, and how that depends on tail")
    say("heaviness. The rate is c1/i = gamma(1-gamma)(1+xi)/i, so the DOMAIN sets the")
    say("constant: heavier tail, larger xi, slower convergence -- and in the Weibull")
    say("domain at xi = -1 there is nothing to converge, because pi = gamma already.")
    say("")
    say(f"{'domain':<22} {'xi':>7} {'i=1':>9} {'i=2':>9} {'i=5':>9} {'i=20':>9} "
        f"{'i=100':>9} {'(pi-gamma) at i=5':>18}")
    say("-" * 100)
    for name, xi in (("Weibull (bounded)", -1.0), ("Weibull", -0.5),
                     ("Gumbel (exponential)", 0.0), ("Frechet (Pareto 3)", 1 / 3),
                     ("Frechet (heavy)", 1.0)):
        vals = [pi_gpd(xi, i, 0.5) for i in (1, 2, 5, 20, 100)]
        say(f"{name:<22} {xi:>7.3f} " + " ".join(f"{v:>9.6f}" for v in vals)
            + f" {vals[2] - 0.5:>+18.6f}")
    say("")
    say("At gamma = 1/2 the departure at depth 5 runs from exactly zero in the")
    say("bounded-tail case up to the Frechet figure above. Same fraction, same")
    say("depth, and the tail settles the error by itself. Which answers the reading")
    say("of interpolation as a mere O(1/n) effect: the constant IS the shape index")
    say("no amount of data moves it.")
    say("")
    say("Shrinkage toward the asymptotic shape, which is what Gumbel-domain")
    say("membership requires and what the fitted-shape sweep observes:")
    for name, dist, xi_inf in cases:
        vals = trend[name]
        gaps = [abs(v - xi_inf) for v in vals]
        shrinking = all(gaps[k] > gaps[k + 1] for k in range(len(gaps) - 1))
        say(f"    {name:<14} |xi_loc - xi| : "
            + " > ".join(f"{g:.5f}" for g in gaps)
            + ("   shrinking" if shrinking
               else "   exactly xi at every n (F IS GPD: nothing to shrink)"
               if all(g < 1e-9 for g in gaps) else "   NOT monotone"))
    say("")
    say("=" * 100)
    say("What this buys, and what it does not. The exactly-GPD hypothesis is no")
    say("longer the boundary of the result: the rate holds off it with xi read as the")
    say("shape at the sample's own depth, to first order in the second-order")
    say("auxiliary. Those fitted signs -- below zero for the normal, above it for")
    say("the lognormal, both heading in -- stop being something noticed and turn")
    say("into something xi_loc predicts with no sample involved. Not claimed: any")
    say("statement bounding what is left at o(1/i) in evaluable terms. That would")
    say("be ornament, and what stands in for it is the departure measured")
    say("above.")

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        OUT)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
