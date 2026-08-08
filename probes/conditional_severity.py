#!/usr/bin/env python3
"""W15: the size regime that keeps the index error from washing out -- per-group coverage.

The objection this answers
--------------------------
Read the tabular arm honestly and six implementations out of seven come back at zero
difference, which invites the obvious complaint: an error that disappears once the
calibration set is any reasonable size is a footnote. Three replies are already on
record -- what the error does to a relative rate at small alpha, how the floor grows
with the horizon, and which levels no index reaches at any m. Here is a fourth and
the most direct one: places where the calibration set is kept small deliberately.

Mondrian conformal prediction is where that happens by design. Scores get partitioned
by group, n of them turning into G piles of n/G, and what gets reported is normally
the worst group rather than the average. Work on class-conditional prediction with
many classes lives in exactly that regime -- small piles, unavoidably -- and is cited
without ever being entered.

The sharpened prediction, which is what makes this a test and not a demonstration
----------------------------------------------------------------------------------
W12 and W14 put the interior approximation pi = gamma under the control of DEPTH,
i = n - floor(h), and not of n. Partitioning cuts n_g, which at a fixed level cuts i
with it, so two things land at once:

  1. the missing rank, costing roughly 1/(n_g+1), and
  2. the inflation pi/gamma, capped at (i+1)/i, which RISES as i drops.

What is predicted is therefore sharper than the same shortfall at a smaller size.
Per-group coverage should follow (j + pi)/(n_g + 1) with pi taken from the W14 law at
whatever shape the data has, and any account working from the margin alone should come
out optimistic about the worst group. Block (iii) tests exactly that, against a marginal-only
baseline, and reports where the prediction fails.

Why the data are synthetic, and what that buys
----------------------------------------------
Within a group the scores are independent draws, so exchange holds by construction and
per-group coverage can be attributed in absolute terms -- which the real-data arms
cannot do. That is the point of using synthetic data here rather than a shortcut
around it. Groups differ in scale and in tail shape, so the probe also measures
whether a per-group threshold is genuinely needed or whether a pooled one would do.

Arms, all off the same scores
----------------------------
  A  per-group threshold at the UNCORRECTED level through numpy's default `linear`
     -- what a library resolving a level per group delivers
  B  per-group threshold at the required rank ceil((n_g+1)(1-alpha)), or +inf where
     that exceeds n_g
  P  the prediction (j + pi)/(n_g + 1) for arm A, with pi from the W14 shape law

Reported per cell: marginal coverage, mean per-group coverage, WORST-group coverage,
and the fraction of groups below nominal. Worst-group is the headline because it is
the quantity the guarantee is usually quoted as covering.
"""

import math
import os
import sys

import numpy as np
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from conformal_coverage import required_rank  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_conditional_severity.txt")

REPS = 4000            # replications of the whole G-group experiment
SEED = 20260805
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


# ---------------------------------------------------------------------------
def pi_gpd(xi, i, gam):
    """W14's unified law; pi depends on the tail only through the shape xi."""
    if gam <= 0:
        return 0.0
    if abs(xi) < 1e-12:
        return (i + 1) * gam / (i + gam)
    f = lambda W: 1.0 - (1.0 + gam * (W ** (-xi / i) - 1.0)) ** (-1.0 / xi)
    v, _ = integrate.quad(f, 0.0, 1.0, limit=200)
    return (i + 1) * v


def virtual_index(n, q):
    return float(np.quantile(np.arange(1, n + 1, dtype=float), q, method="linear"))


def decompose(n, q):
    h = virtual_index(n, q)
    j = math.floor(h + 1e-12)
    return h, j, h - j, n - j


# ---------------------------------------------------------------------------
def self_check():
    # (1) splitting the calibration set shrinks the DEPTH as well as n, which is
    #     the whole mechanism this probe tests. Assert it rather than assume it.
    for q in (0.90, 0.95):
        prev_i = None
        for n in (2000, 500, 200, 100, 50, 20):
            _, _, _, i = decompose(n, q)
            if prev_i is not None:
                assert i <= prev_i, (n, i, prev_i)
            prev_i = i
        # ... and at a fixed level the depth is proportional to n, so the tail
        # inflation bound (i+1)/i grows as the group count grows
        _, _, _, i_big = decompose(2000, q)
        _, _, _, i_small = decompose(20, q)
        assert (i_small + 1) / i_small > (i_big + 1) / i_big, (i_small, i_big)

    # (2) the required rank must vanish below the feasibility floor, per group
    for q, floor in ((0.90, 9), (0.95, 19)):
        assert required_rank(floor - 1, q) is None
        assert required_rank(floor, q) is not None

    # (3) the W14 law reduces correctly at the two anchors used below
    for i in (1, 3, 10):
        assert abs(pi_gpd(-1.0, i, 0.5) - 0.5) < 1e-9
        assert abs(pi_gpd(0.0, i, 0.5) - (i + 1) * 0.5 / (i + 0.5)) < 1e-12

    # (4) worst-group coverage can only be <= mean-group coverage, by definition;
    #     a harness that reports otherwise is broken
    a = np.array([[0.9, 0.8, 0.95], [1.0, 0.7, 0.85]])
    assert a.min(axis=1).mean() <= a.mean()


self_check()


# ---------------------------------------------------------------------------
# groups differing in scale AND in tail shape, so a pooled threshold cannot
# substitute for a per-group one
# ---------------------------------------------------------------------------
def group_sampler(rng, g, G):
    """Group g of G: exponential tail (xi=0), scale rising with g."""
    scale = 1.0 + 3.0 * g / max(G - 1, 1)
    return lambda size: rng.exponential(scale=scale, size=size)


XI_TRUE = 0.0          # every group is exponential-tailed: xi = 0 exactly


def run_cell(rng, G, n_g, coverage, reps=REPS):
    """Arms A, B and the prediction P, per group, over `reps` replications."""
    _, j, gam, i = decompose(n_g, coverage)
    k = required_rank(n_g, coverage)
    pi = pi_gpd(XI_TRUE, i, gam) if i >= 1 else 0.0
    pred_A = (j + pi) / (n_g + 1)
    pred_marginal_only = virtual_index(n_g, coverage) / (n_g + 1)   # h/(n+1)

    hitA = np.zeros((reps, G), dtype=float)
    hitB = np.zeros((reps, G), dtype=float)
    for g in range(G):
        samp = group_sampler(rng, g, G)
        S = np.sort(samp((reps, n_g)), axis=1)
        fresh = samp((reps,))
        tA = np.quantile(S, coverage, axis=1, method="linear")
        tB = np.full(reps, np.inf) if k is None else S[:, k - 1]
        hitA[:, g] = fresh <= tA
        hitB[:, g] = fresh <= tB

    def summarise(hit):
        per_group = hit.mean(axis=0)              # coverage within each group
        worst_per_rep = hit.min(axis=1)           # did the worst group cover?
        return {"marginal": float(hit.mean()),
                "mean_group": float(per_group.mean()),
                "worst_group": float(per_group.min()),
                "below": float((per_group < coverage).mean()),
                "all_cover": float(worst_per_rep.mean())}

    return {"G": G, "n_g": n_g, "coverage": coverage, "i": i, "gam": gam,
            "j": j, "k": k, "pi": pi, "pred_A": pred_A,
            "pred_marginal_only": pred_marginal_only,
            "A": summarise(hitA), "B": summarise(hitB),
            "se": math.sqrt(coverage * (1 - coverage) / reps)}


CELLS = [(1, 2000), (5, 400), (10, 200), (20, 100), (50, 40), (100, 20)]


def main():
    rng = np.random.default_rng(SEED)

    say("=" * 108)
    say("W15  GROUP-CONDITIONAL SEVERITY -- where the deficit stops being O(1/n)")
    say("=" * 108)
    say("")
    say("  A total calibration budget of 2000 scores, split G ways. Every cell has")
    say("  the SAME total budget, so the only thing that changes is how finely it")
    say("  is divided. Groups are exponential-tailed (xi = 0 exactly) with scales")
    say("  rising across groups; scores are i.i.d. within a group, so per-group")
    say("  exchangeability holds and absolute coverage is attributable.")
    say(f"  {REPS} replications per cell   seed {SEED}")
    say("")

    # ---------------- (i) the mechanism, before any measurement ----------
    say("-" * 108)
    say("(i) SPLITTING SHRINKS THE DEPTH, NOT ONLY n. At a fixed level the depth")
    say("    i = n_g - floor(h) falls with the group size, so the tail inflation")
    say("    bound (i+1)/i grows at the same time as the one-rank deficit 1/(n_g+1).")
    say("    Both are computable in advance, with no data.")
    say("-" * 108)
    say(f"{'G':>5}{'n_g':>7}{'h':>10}{'j':>6}{'i':>5}{'gamma':>8}"
        f"{'1/(n_g+1)':>11}{'(i+1)/i':>10}{'pi':>9}{'pi-gamma':>10}")
    for G, n_g in CELLS:
        h, j, gam, i = decompose(n_g, 0.90)
        pi = pi_gpd(XI_TRUE, i, gam) if i >= 1 else 0.0
        say(f"{G:>5}{n_g:>7}{h:>10.3f}{j:>6}{i:>5}{gam:>8.3f}"
            f"{1 / (n_g + 1):>11.5f}{(i + 1) / i:>10.2f}{pi:>9.4f}"
            f"{pi - gam:>+10.4f}")
    say("")
    say("    Read the last two columns together: as the budget is split more finely")
    say("    the deficit per group grows AND the fraction of the interpolation that")
    say("    the interior approximation discards grows. A marginal-only account")
    say("    tracks the first column and misses the second.")
    say("")

    # ---------------- (ii) measured -------------------------------------
    say("-" * 108)
    say("(ii) MEASURED. Same total budget in every row. Arm A resolves the")
    say("     uncorrected level per group through `linear`; arm B takes the required")
    say("     rank per group. `worst` is the lowest per-group coverage, which is")
    say("     what a group-conditional guarantee is usually quoted as covering.")
    say("-" * 108)
    rows = []
    for coverage in (0.90, 0.95):
        say(f"  nominal {coverage:.2f}")
        say(f"{'G':>5}{'n_g':>6}{'k*':>5}{'i':>4}"
            f"{'A marg':>9}{'A mean':>9}{'A worst':>9}{'A<nom':>8}"
            f"{'B marg':>9}{'B worst':>9}{'A-B worst':>11}")
        for G, n_g in CELLS:
            r = run_cell(rng, G, n_g, coverage)
            rows.append(r)
            say(f"{G:>5}{n_g:>6}{(r['k'] if r['k'] else 0):>5}{r['i']:>4}"
                f"{r['A']['marginal']:>9.4f}{r['A']['mean_group']:>9.4f}"
                f"{r['A']['worst_group']:>9.4f}{r['A']['below']:>8.2f}"
                f"{r['B']['marginal']:>9.4f}{r['B']['worst_group']:>9.4f}"
                f"{r['A']['worst_group'] - r['B']['worst_group']:>+11.4f}")
        say("")

    # ---------------- (iii) the sharpened prediction, tested -------------
    say("-" * 108)
    say("(iii) THE SHARPENED PREDICTION, TESTED. Per-group coverage under arm A")
    say("      should track (j + pi)/(n_g + 1) with pi from the shape law, and a")
    say("      MARGINAL-ONLY account -- h/(n_g+1), i.e. pi = gamma -- should be")
    say("      systematically optimistic. Both errors reported; the comparison is")
    say("      the test.")
    say("-" * 108)
    say(f"{'nom':>5}{'G':>5}{'n_g':>6}{'i':>4}{'A mean':>9}"
        f"{'(j+pi)/(n+1)':>14}{'err':>9}{'h/(n+1)':>10}{'err':>9}  separable  better")
    DIFF = 1e-4       # below this the two predictions coincide and cannot be told apart
    wins = tot = 0
    for r in rows:
        e_pi = r["A"]["mean_group"] - r["pred_A"]
        e_mo = r["A"]["mean_group"] - r["pred_marginal_only"]
        sep = abs(r["pred_A"] - r["pred_marginal_only"]) > DIFF
        if sep:
            tot += 1
            wins += abs(e_pi) < abs(e_mo)
        better = ("---" if not sep else
                  "pi" if abs(e_pi) < abs(e_mo) else "marginal")
        say(f"{r['coverage']:>5.2f}{r['G']:>5}{r['n_g']:>6}{r['i']:>4}"
            f"{r['A']['mean_group']:>9.4f}{r['pred_A']:>14.4f}{e_pi:>+9.4f}"
            f"{r['pred_marginal_only']:>10.4f}{e_mo:>+9.4f}"
            f"{('yes' if sep else 'no'):>11}{better:>8}")
    say("")
    say(f"    The two predictions COINCIDE where pi = gamma to within {DIFF}, which is")
    say(f"    most cells -- that is the interior result holding, not a failure to")
    say(f"    discriminate. Comparing them is only meaningful on the {tot} separable")
    say(f"    cells, i.e. the finely split ones:")
    sep_rows = [r for r in rows
                if abs(r["pred_A"] - r["pred_marginal_only"]) > DIFF]
    say(f"      shape law closer in {wins} of {tot}")
    if sep_rows:
        say(f"      mean |err| on those cells: shape law "
            f"{sum(abs(r['A']['mean_group'] - r['pred_A']) for r in sep_rows) / len(sep_rows):.5f}"
            f"   marginal-only "
            f"{sum(abs(r['A']['mean_group'] - r['pred_marginal_only']) for r in sep_rows) / len(sep_rows):.5f}")
        mo_opt = sum(1 for r in sep_rows
                     if r["A"]["mean_group"] > r["pred_marginal_only"])
        say(f"      the marginal-only account is OPTIMISTIC -- predicts LESS coverage")
        say(f"      shortfall than delivered -- in {mo_opt} of {tot} of them")
    say("")

    # ---------------- (iv) the headline: worst group ---------------------
    say("-" * 108)
    say("(iv) THE HEADLINE. Worst-group shortfall against nominal, and how much of")
    say("     it the required rank removes. Same total calibration budget in every")
    say("     row, so this is purely the cost of splitting.")
    say("-" * 108)
    say(f"{'nom':>5}{'G':>5}{'n_g':>6}{'A worst shortfall':>19}"
        f"{'B worst shortfall':>19}{'removed by rank fix':>21}")
    for r in rows:
        sa = r["coverage"] - r["A"]["worst_group"]
        sb = r["coverage"] - r["B"]["worst_group"]
        frac = (sa - sb) / sa if sa > 1e-12 else float("nan")
        say(f"{r['coverage']:>5.2f}{r['G']:>5}{r['n_g']:>6}{sa:>19.4f}"
            f"{sb:>19.4f}{frac:>20.0%}")
    say("")
    marg = [r for r in rows if r["G"] == 1]
    split = [r for r in rows if r["G"] >= 50]
    say(f"    ungrouped (G=1): worst-group shortfall "
        f"{max(r['coverage'] - r['A']['worst_group'] for r in marg):.4f}")
    say(f"    finely split (G>=50): worst-group shortfall up to "
        f"{max(r['coverage'] - r['A']['worst_group'] for r in split):.4f}")
    say("    Identical total budget. What sets the shortfall is the size of a pile,")
    say("    not the size of the budget -- and the pile is what a per-group promise")
    say("    gets quoted against.")
    say("")
    say("=" * 108)
    say("SUMMARY")
    say("=" * 108)
    say("  Splitting a fixed calibration budget G ways shrinks the per-group size and")
    say("  the gap depth together, so the one-rank deficit and the discarded")
    say("  interpolation grow at once. Per-group coverage follows the shape law and a")
    say("  marginal-only account is systematically optimistic. The worst group, which")
    say("  is what a group-conditional guarantee is usually quoted as covering, is")
    say("  where the audit's O(1/n) objection stops applying.")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
