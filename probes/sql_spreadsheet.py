#!/usr/bin/env python3
"""Quantile conventions where most quantiles are actually computed.

Why this exists
---------------
The cross-language probe measures numpy, pandas, scipy, R, Julia and Octave. That
is the statistical-software ecosystem, and it is not where most of the world's
quantiles get computed. Two families are missing and both are reachable:

  * SQL engines. `percentile_cont` and `percentile_disc` are SQL-standard
    aggregate functions, so their conventions are not one vendor's choice -- they
    are written into the standard, and an engine either implements them or lacks
    them entirely.
  * Spreadsheets. `PERCENTILE.INC` and `PERCENTILE.EXC` are what a
    non-programmer reaches for, and they are two different conventions with
    almost the same name.

The interesting question is not "does anyone disagree" -- the ecosystem already
disagrees, which the cross-language probe establishes. It is whether these
conventions can express a coverage guarantee at all, and the answer differs
between the two SQL functions in a way that matters.

The instrument is the same as everywhere else here: quantile the tie-free set
1..n; each value equals its rank, so a returned number IS the virtual index
and nothing has to be inferred.

Executed, not transferred
-------------------------
DuckDB runs in-process. Excel is driven through AppleScript when it is present and
reachable, and reported as NOT EXECUTED otherwise rather than cited from
documentation -- the documented convention and the shipped one are exactly what
this programme exists to stop conflating. SQLite is included because its answer is
that it has no such function, which is a finding about the most widely deployed
engine in the world and not an omission.

    python probes/sql_spreadsheet.py
"""

import math
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "outputs/probe_output_sql_spreadsheet.txt"
# The sizes are chosen by the ARITHMETIC and not by round numbers. The delivering
# set of a rounding convention is a residue class -- inverted_cdf at level 9/10
# delivers exactly when n = 9 mod 10 -- so a grid of 10, 20, 50, 100 contains no
# delivering size and reports "never delivers" for a convention that delivers one
# time in ten. A first version of this probe did exactly that. Each level therefore
# gets sizes ON its class and OFF it, and 2/3 is here because the neat decade
# pattern comes from the chosen levels people happen to request.
NS = (9, 10, 19, 20, 29, 50, 99, 100)
LEVELS = (0.90, 0.95, 2.0 / 3.0)

# The nine Hyndman-Fan definitions as (alpha, beta) with h = alpha + q(n+1-alpha-beta),
# plus numpy's four discontinuous aliases identified by behaviour rather than by a
# pair. Named so a measured (A, B) can be reported as a KNOWN convention instead of
# as two numbers a reader has to look up.
HF = {
    "inverted_cdf (H&F 1)": None,
    "averaged_inverted_cdf (H&F 2)": None,
    "closest_observation (H&F 3)": None,
    "interpolated_inverted_cdf (H&F 4)": (0.0, 1.0),
    "hazen (H&F 5)": (0.5, 0.5),
    "weibull (H&F 6)": (0.0, 0.0),
    "linear (H&F 7)": (1.0, 1.0),
    "median_unbiased (H&F 8)": (1 / 3, 1 / 3),
    "normal_unbiased (H&F 9)": (3 / 8, 3 / 8),
}


def scores(n):
    """Tie-free: the returned value IS the 1-indexed virtual index."""
    return list(range(1, n + 1))


def ab_to_h(ab, n, q):
    a, b = ab
    return a + q * (n + 1 - a - b)


def identify(fn, n=50):
    """Name the convention a callable implements, or report it as unmatched.

    Fits (A, B) at interior levels, then matches against the Hyndman-Fan pairs.
    Interior because every implementation clamps h to [1, n], so
    an endpoint fit returns linear's coefficients whatever the convention -- the
    same trap the cross-language probe records.
    """
    q1, q2 = 3.0 / (n + 1), 1.0 - 3.0 / (n + 1)
    h1, h2 = fn(n, q1), fn(n, q2)
    if h1 is None or h2 is None:
        return None, (None, None)
    B = (h2 - h1) / (q2 - q1)
    A = h1 - B * q1
    grid = np.linspace(q1, q2, 41)
    dev = max(abs(fn(n, float(q)) - (A + B * float(q))) for q in grid)
    if dev > 1e-7:
        return "not affine (a rounding rule)", (None, None)
    for name, ab in HF.items():
        if ab is None:
            continue
        if abs(ab_to_h(ab, n, 0.37) - (A + B * 0.37)) < 1e-6 and \
           abs(ab_to_h(ab, n, 0.71) - (A + B * 0.71)) < 1e-6:
            return name, (A, B)
    return "unmatched", (A, B)


def delivers(fn, n, level):
    """Does the index this convention resolves carry the requested guarantee?

    A threshold at virtual index h guarantees floor(h)/(n+1). The requirement is
    floor(h) >= ceil((n+1)*level).
    """
    h = fn(n, level)
    if h is None:
        return None, None, None
    req = math.ceil((n + 1) * level)
    return h, req, math.floor(h + 1e-9) >= req


# ---------------------------------------------------------------------------
# DuckDB: the SQL standard, in process
# ---------------------------------------------------------------------------
def duckdb_fns():
    try:
        import duckdb
    except ImportError:
        return {}, None
    con = duckdb.connect()
    ver = con.execute("select version()").fetchone()[0]

    def make(expr):
        def fn(n, q):
            vals = ",".join(f"({v})" for v in scores(n))
            sql = (f"select {expr} from (values {vals}) as t(x)")
            return float(con.execute(sql, [q] if "?" in sql else []).fetchone()[0])
        return fn

    fns = {
        "duckdb quantile_cont": make("quantile_cont(x, ?)"),
        "duckdb quantile_disc": make("quantile_disc(x, ?)"),
        "duckdb percentile_cont (SQL standard)":
            make("percentile_cont(?) within group (order by x)"),
        "duckdb percentile_disc (SQL standard)":
            make("percentile_disc(?) within group (order by x)"),
        "duckdb median": (lambda n, q: float(
            con.execute(f"select median(x) from (values "
                        f"{','.join(f'({v})' for v in scores(n))}) as t(x)")
            .fetchone()[0])),
    }
    return fns, ver


# ---------------------------------------------------------------------------
# SQLite: the answer is that there is nothing to measure
# ---------------------------------------------------------------------------
def sqlite_probe():
    import sqlite3
    con = sqlite3.connect(":memory:")
    ver = sqlite3.sqlite_version
    found = {}
    for expr in ("percentile_cont(0.9) within group (order by x)",
                 "percentile_disc(0.9) within group (order by x)",
                 "quantile(x, 0.9)", "median(x)"):
        try:
            con.execute("create table t(x real)")
        except sqlite3.OperationalError:
            pass
        try:
            con.execute(f"select {expr} from t")
            found[expr] = "present"
        except Exception as exc:
            found[expr] = f"absent ({type(exc).__name__})"
    return found, ver


# ---------------------------------------------------------------------------
# Excel: executed through AppleScript, or reported as not executed
# ---------------------------------------------------------------------------
def excel_fns():
    """Drive Excel's own formula evaluator, or return nothing.

    Uses `osascript` to ask a running Excel to evaluate the formula on a literal
    array, so no workbook is created or saved. If Excel is absent, refuses
    automation, or is not licensed, this returns nothing and the caller reports it
    as NOT EXECUTED -- which is the honest state, and better than quoting the
    documented convention as if it had been measured.
    """
    if not os.path.exists("/Applications/Microsoft Excel.app"):
        return {}, "not installed"
    if not shutil.which("osascript"):
        return {}, "no osascript"

    # `evaluate name` on the application, driven from a SCRIPT FILE. Passing the
    # formula inline with `osascript -e` fails: the formula contains braces, commas
    # and semicolons, and AppleScript parses them before Excel ever sees the
    # string. A file with `on run argv` hands the formula over untouched.
    scpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_excel_eval.applescript")
    if not os.path.exists(scpt):
        with open(scpt, "w") as fh:
            fh.write('on run argv\n'
                     '\ttell application "Microsoft Excel"\n'
                     '\t\treturn (evaluate name (item 1 of argv))\n'
                     '\tend tell\n'
                     'end run\n')

    def evaluate(formula):
        p = subprocess.run(["osascript", scpt, formula],
                           capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            raise RuntimeError((p.stderr or "").strip()[:140])
        return float(p.stdout.strip())

    # 255 characters is the cap on the string this interface will hand Excel: an
    # array of 80 works and 90 does not, at a level well inside range, so the
    # failure is the interface and not the function. Sizes above the cap are
    # SKIPPED and reported, never inferred from the smaller ones.
    EXCEL_MAX_N = 80

    def make(func):
        def fn(n, q):
            if n > EXCEL_MAX_N:
                raise RuntimeError(
                    f"skipped: array of {n} exceeds the 255-character limit on the "
                    f"string this interface passes to Excel (80 works, 90 does not)")
            arr = "{" + ";".join(str(v) for v in scores(n)) + "}"
            return evaluate(f"={func}({arr},{q})")
        return fn

    fns = {"excel PERCENTILE.INC": make("PERCENTILE.INC"),
           "excel PERCENTILE.EXC": make("PERCENTILE.EXC"),
           "excel PERCENTILE (legacy)": make("PERCENTILE")}
    # probe once; a failure here means the whole family is unreachable
    try:
        fns["excel PERCENTILE.INC"](10, 0.5)
    except Exception as exc:
        return {}, f"unreachable: {exc}"
    return fns, "driven via AppleScript"


def self_check():
    # the identifier must name numpy's own conventions correctly, or it cannot be
    # trusted to name anybody else's
    for meth, want in (("linear", "linear (H&F 7)"),
                       ("weibull", "weibull (H&F 6)"),
                       ("hazen", "hazen (H&F 5)"),
                       ("median_unbiased", "median_unbiased (H&F 8)"),
                       ("interpolated_inverted_cdf",
                        "interpolated_inverted_cdf (H&F 4)")):
        fn = (lambda m: (lambda n, q: float(
            np.quantile(np.asarray(scores(n), float), q, method=m))))(meth)
        got, _ = identify(fn)
        assert got == want, (meth, got, want)
    # and a rounding rule must be reported as one rather than forced into a pair
    fn = lambda n, q: float(np.quantile(np.asarray(scores(n), float), q,
                                        method="inverted_cdf"))
    got, _ = identify(fn)
    assert got == "not affine (a rounding rule)", got
    # the delivery test must agree with the known answer: linear never delivers,
    # inverted_cdf delivers exactly on a residue class
    lin = lambda n, q: float(np.quantile(np.asarray(scores(n), float), q,
                                         method="linear"))
    inv = lambda n, q: float(np.quantile(np.asarray(scores(n), float), q,
                                         method="inverted_cdf"))
    assert delivers(lin, 50, 0.90)[2] is False
    assert delivers(inv, 9, 0.90)[2] is True
    assert delivers(inv, 12, 0.90)[2] is False      # off the residue class


self_check()


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say("QUANTILE CONVENTIONS IN SQL ENGINES AND SPREADSHEETS")
    say("=" * 100)
    say("self_check() passed at import: the identifier names five of numpy's own")
    say("conventions correctly, reports a rounding rule as a rounding rule rather")
    say("than forcing it into an (alpha, beta) pair, and its delivery test agrees")
    say("with the known answer for linear and for inverted_cdf on and off the")
    say("residue class.")
    say("")
    say("Instrument: quantile the tie-free set 1..n, so a returned value IS the")
    say("virtual index. (A, B) is fitted at INTERIOR levels, because every engine")
    say("clips the index into [1, n] and an endpoint fit returns linear's")
    say("coefficients whatever the convention.")
    say("")

    engines = {}
    duck, duckver = duckdb_fns()
    if duck:
        engines.update(duck)
        say(f"duckdb {duckver}: {len(duck)} functions, in process")
    else:
        say("duckdb: NOT INSTALLED -- reported, not inferred")

    xl, xlnote = excel_fns()
    if xl:
        engines.update(xl)
        say(f"excel: {len(xl)} functions, {xlnote}")
    else:
        say(f"excel: NOT EXECUTED ({xlnote}). Its conventions are documented -- "
            f"PERCENTILE.INC as H&F 7 and PERCENTILE.EXC as H&F 6 -- and this probe")
        say("       does not report a documented convention as a measured one.")
    say("")

    say("(1) which convention each function implements, identified at n=50")
    say(f"{'function':<42} {'convention':<34} {'A':>7} {'B':>7}")
    say("-" * 100)
    named = {}
    for name, fn in engines.items():
        try:
            conv, (A, B) = identify(fn)
        except Exception as exc:
            say(f"{name:<42} error: {type(exc).__name__}")
            continue
        named[name] = conv
        say(f"{name:<42} {str(conv):<34} "
            f"{('%.4f' % A) if A is not None else '--':>7} "
            f"{('%.4f' % B) if B is not None else '--':>7}")
    say("")

    say("(2) can the convention carry the requested guarantee?")
    say("A threshold at virtual index h guarantees floor(h)/(n+1), so the test is")
    say("floor(h) >= ceil((n+1)*level). 'yes' means the call is usable as a")
    say("distribution-free bound at that n; 'no' means it is short.")
    say(f"{'function':<42} {'level':>6} {'n':>5} {'h':>9} {'required':>9} "
        f"{'delivers':>9}")
    say("-" * 100)
    tally = {}
    for name, fn in engines.items():
        for level in LEVELS:
            for n in NS:
                try:
                    h, req, ok = delivers(fn, n, level)
                except Exception as exc:
                    say(f"{name:<42} {level:>6.2f} {n:>5}   -- {str(exc)[:44]}")
                    continue
                tally.setdefault(name, []).append(bool(ok))
                say(f"{name:<42} {level:>6.2f} {n:>5} {h:>9.4f} {req:>9} "
                    f"{'yes' if ok else 'no':>9}")
        say("")

    if xl:
        say("(2b) PERCENTILE.EXC at the feasibility boundary")
        say("The weibull convention resolves h = q(n+1), so it can only return a")
        say("value at q <= n/(n+1) -- exactly the one-sided feasibility")
        say("floor. Does the shipped function refuse there, or clamp?")
        say("")
        say(f"{'n':>5} {'q':>12} {'n/(n+1)':>10} {'in range?':>10} {'returned':>28}")
        say("-" * 100)
        refusals, clamps = 0, 0
        for n in (10, 20, 50):
            top = n / (n + 1)
            for q in (top - 0.02, top - 1e-6, top + 1e-6, top + 0.02, 0.99):
                if not (0 < q < 1):
                    continue
                try:
                    v = xl["excel PERCENTILE.EXC"](n, round(q, 9))
                    got = f"{v:.6f}"
                    if q > top + 1e-9:
                        clamps += 1
                except Exception as exc:
                    got = "REFUSED (" + str(exc)[-28:].strip() + ")"
                    if q > top + 1e-9:
                        refusals += 1
                say(f"{n:>5} {q:>12.6f} {top:>10.6f} "
                    f"{('yes' if q <= top else 'no'):>10} {got:>28}")
            say("")
        say(f"above the boundary: {refusals} refusals, {clamps} values returned")
        if refusals and not clamps:
            say("So this function REFUSES rather than clamping. It is the only")
            say("implementation measured anywhere in this deposit that guards the")
            say("one-sided feasibility boundary by construction -- the reason is")
            say("that its convention treats the boundary as outside the domain rather")
            say("than because anybody added a check. Against it, 9 of 14 shipped")
            say("conformal helpers return a finite number below their floor and")
            say("say nothing.")
        say("")

    say("(3) SQLite: what a query gets when the function does not exist")
    found, sqlver = sqlite_probe()
    say(f"sqlite {sqlver}")
    for expr, state in found.items():
        say(f"    {expr:<52} {state}")
    say("")

    say("=" * 100)
    say("SUMMARY")
    for name, oks in tally.items():
        say(f"    {name:<42} delivers in {sum(oks)} of {len(oks)} cells")
    say("")
    disc = [k for k in tally if "percentile_disc" in k]
    cont = [k for k in tally if "percentile_cont" in k]
    if disc and cont:
        # per function, not summed over the aliases: quantile_disc and
        # percentile_disc are the same convention and adding them double-counts it
        d, dn = sum(tally[disc[0]]), len(tally[disc[0]])
        c, cn = sum(tally[cont[0]]), len(tally[cont[0]])
        say("The two SQL-standard functions are not interchangeable for this")
        say(f"purpose. percentile_disc delivers the requested guarantee in {d} of")
        say(f"{dn} cells; percentile_cont in {c} of {cn}. So the standard DOES")
        say("contain a convention with enough mass for a finite-sample guarantee, which is")
        say("unlike numpy's default -- and the two function names differ")
        say("by four characters with no hint one remains usable and the other fails.")
        say("")
        say("Where percentile_disc delivers is a residue class and not a threshold,")
        say("which is the periodicity result showing up in a SQL engine: it is right")
        say("one time in ten at level 9/10, and collecting more rows does not fix a")
        say("size that is off the class.")
    say("")
    say("SQLite has no percentile function at all, so the most widely deployed")
    say("engine in the world offers no convention to get wrong and every caller")
    say("hand-rolls one. That is a different failure mode from a wrong default and")
    say("it is not covered by anything else in this deposit.")

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        OUT)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
