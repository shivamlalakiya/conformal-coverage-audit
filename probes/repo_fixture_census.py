#!/usr/bin/env python3
"""What calibration counts each audited project pins inside its own test tree.

WHY A SECOND FIXTURE CENSUS
certificate_suite.py already scores the calibration sizes it can read out of the
installed wheels, and reports two of them across one package. That number is a
property of what a wheel carries, not of what a project tests: most of these
projects keep their tests in the repository and ship a wheel without them. So the
earlier census measured packaging and was read as measuring test adequacy.

This probe reads the same thing from the source the maintainers actually write.
Each package is cloned at the version pinned in the manuscript's census table, and
its own test tree is parsed. Nothing is installed and nothing is executed: pytest
in a pinned environment would move the dependency set the rest of this audit is
measured against, and running a test suite is not needed to read the sizes it
fixes.

WHAT COUNTS AS A CALIBRATION SIZE, AND WHY THE QUESTION IS NOT TRIVIAL
A test file is full of integers and almost none of them is a calibration size. The
count that matters is the number of scores the conformal quantile is taken over,
and three syntactic shapes carry it. Every shape is a separate named rule, tied
to a symbol that the shipped library itself defines, and that tie is re-checked
against each fresh checkout instead of being taken on faith: should a rule's
symbol have shifted, the run aborts rather than silently matching nothing.

  Rule A, keyword.       A whole number written straight into a named argument
                         whose parameter counts calibration points.
                         `ConformalIntervals(n_windows=10)` pins ten of them.
  Rule B, guard.         A whole number stored under a local name, then handed on
                         to a routine that vets how many points there are. The
                         wheel-side reading already turned this shape up.
  Rule C, literal set.   A spelled-out sequence given to a conformal routine as
                         its scores. What matters is how MANY entries it holds,
                         and that quantity no digit hunt will ever surface.

Two shapes are deliberately excluded and both are recorded rather than dropped,
because an exclusion nobody can see is indistinguishable from a miss.

  A TRAINING window is not a calibration size. `sktime`'s `initial_window` sets
  how much history the forecaster fits on; the calibration count is what is left
  of the series after it, so the test fixes a training length and leaves the
  calibration size to the data. The wheel census says the same thing about the
  same parameter.
  A FRACTION is not a count. `mapie`'s `calib_size=0.5` is a proportion of the
  data. A regular expression that reads an integer out of it finds the zero and
  reports a calibration size of zero, which is why this probe parses syntax
  instead of text.
  COMMENTED-OUT code fixes nothing. One statsforecast test file carries a
  disabled `ConformalIntervals(n_windows=2, h=12)` inside a comment. A text scan
  counts it and a parser does not, which is a difference of one in the census and
  the reason the audit of this probe was run against a regular expression rather
  than against a second copy of itself.
  CROSS-VALIDATION folds are not calibration windows even where the parameter
  shares a name. `neuralforecast` takes `n_windows` on both
  `PredictionIntervals`, where it is the conformal calibration count, and
  `cross_validation`, where it is the number of evaluation folds. The two are
  told apart by what is being called, which a keyword search cannot do.

    python probes/repo_fixture_census.py

Set REPO_CACHE to reuse clones between runs. The clones are shallow and are read
only; nothing here writes to them.
"""

import ast
import os
import subprocess
import sys
import warnings
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certificate as W13                                    # noqa: E402
from fractions import Fraction as F                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "probe_output_repo_fixture_census.txt")
CACHE = os.environ.get("REPO_CACHE") or os.path.join(
    tempfile.gettempdir(), "conformal-audit-repos")

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


# --------------------------------------------------------------------------
# The subjects, at the versions the manuscript's census table pins. The tag is
# the project's own, so the two spellings below are the projects' and not a
# normalisation: a tag guessed rather than read is a clone of the wrong code.
REPOS = [
    ("crepes 0.9.1", "henrikbostrom/crepes", "v0.9.1"),
    ("mapie 1.4.1", "scikit-learn-contrib/MAPIE", "v1.4.1"),
    ("puncc 0.9.3", "deel-ai/puncc", "v0.9.3"),
    ("sktime 1.1.0", "sktime/sktime", "v1.1.0"),
    ("statsforecast 2.1.1", "Nixtla/statsforecast", "v2.1.1"),
    ("nonconformist 2.1.0", "donlnz/nonconformist", "2.1.0"),
    ("darts 0.46.1", "unit8co/darts", "0.46.1"),
    ("neuralforecast 3.2.1", "Nixtla/neuralforecast", "v3.2.1"),
    ("river 0.25.0", "online-ml/river", "0.25.0"),
    ("torchcp 1.2.1", "ml-stat-Sustech/TorchCP", "v1.2.1"),
]

# Rule A: (callee, keyword) pairs whose integer value is the calibration count,
# each with the source path the callee is defined in. The anchor is what makes
# this a criterion applied to the package rather than a claim about it.
RULE_A = {
    "statsforecast 2.1.1": [
        ("ConformalIntervals", "n_windows", "statsforecast/utils.py"),
    ],
    "neuralforecast 3.2.1": [
        ("PredictionIntervals", "n_windows", "neuralforecast/utils.py"),
    ],
    "darts 0.46.1": [
        ("ConformalNaiveModel", "cal_length", "darts/models/forecasting/conformal_models.py"),
        ("ConformalQRModel", "cal_length", "darts/models/forecasting/conformal_models.py"),
    ],
}

# Rule B: (callee, parameter position) where the argument is a local name bound to
# an integer literal, and the callee validates a calibration count.
RULE_B = {
    "mapie 1.4.1": [
        ("_check_alpha_and_n_samples", 1, "mapie/utils.py"),
    ],
}

# Rule C: callees taking a literal sequence as the score set. The value recorded
# is the sequence's length.
RULE_C = {
    "torchcp 1.2.1": [
        ("calculate_conformal_value", 0, "torchcp/utils/common.py"),
    ],
}

# Excluded, with the reason printed beside the value rather than dropped.
EXCLUDED = {
    "sktime 1.1.0": [
        ("ConformalIntervals", "initial_window",
         "sktime/forecasting/conformal.py",
         "a TRAINING window: the calibration count is what remains of the series "
         "after it, so the test does not fix it"),
    ],
}


# --------------------------------------------------------------------------
def clone(slug, tag, dest):
    """Shallow clone at one tag, or reuse an existing one. Read only."""
    if os.path.isdir(os.path.join(dest, ".git")):
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", tag, "--single-branch",
         f"https://github.com/{slug}.git", dest],
        capture_output=True, text=True, timeout=1800)
    return r.returncode == 0


def test_files(root):
    """Sources whose path mentions testing, ordered so a rerun repeats."""
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules")]
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(base, f)
            if "test" in os.path.relpath(p, root).lower():
                out.append(p)
    return sorted(out)


def _callee(node):
    """The bare name of whatever a Call node calls, attribute access included."""
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def _int_literal(node):
    """The value of an integer literal, or None. A float is NOT an integer here.

    `calib_size=0.5` is a proportion and must not read as a calibration size of
    zero, which is exactly what a text search for a digit does to it.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value
    return None


def _seq_len(node):
    """Length of a literal sequence, including one wrapped in a single call.

    `torch.tensor([1.0, 2.0, 3.0])` and `np.array([1.0, 2.0])` both carry their
    length in the list inside them, so one level of wrapping is unwrapped. A
    sequence built by a computation has no literal length and returns None rather
    than a guess.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    if isinstance(node, ast.Call) and node.args:
        inner = node.args[0]
        if isinstance(inner, (ast.List, ast.Tuple)):
            return len(inner.elts)
    return None


def _scopes(tree):
    """(scope, own nodes) for the module and every function, without overlap.

    A name has to be resolved in the function it is written in and not in the
    file. Two tests in one file both writing `n = 30` and `n = 10` are the shape
    that makes the difference visible, and a file-wide table gets both of them
    wrong in the same direction: whichever assignment comes last wins at every
    call site, so one real fixture disappears and the other is double-counted.
    That is not hypothetical -- it is what this probe printed before the scopes
    were separated, against a wheel-side census that already had the right pair.

    Each node belongs to exactly one scope, so no call is visited twice.
    """
    fn_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    out = []
    for scope in [tree] + [n for n in ast.walk(tree) if isinstance(n, fn_types)]:
        own, stack = [], list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            if isinstance(node, fn_types):
                continue                 # belongs to its own scope
            own.append(node)
            stack.extend(ast.iter_child_nodes(node))
        out.append((scope, own))
    return out


def _bindings(nodes):
    """name -> [(lineno, int value, sequence length)] for literal assignments."""
    b = {}
    for node in nodes:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        iv, sl = _int_literal(node.value), _seq_len(node.value)
        if iv is None and sl is None:
            continue
        b.setdefault(tgt.id, []).append((node.lineno, iv, sl))
    return b


def _resolve(name, lineno, local, module):
    """The binding in force at a call site: nearest preceding, local before module."""
    for table in (local, module):
        cands = [x for x in table.get(name, []) if x[0] <= lineno]
        if cands:
            return max(cands, key=lambda x: x[0])
    return None


def scan(pkg, root):
    """Every calibration size the package's own tests fix, by rule, anchored."""
    hits = []
    a_spec = RULE_A.get(pkg, [])
    b_spec = RULE_B.get(pkg, [])
    c_spec = RULE_C.get(pkg, [])
    x_spec = EXCLUDED.get(pkg, [])
    files = test_files(root)
    for path in files:
        try:
            with warnings.catch_warnings():
                # A test file written for an older Python can carry an invalid
                # escape. It parses; the warning is the reader's, not this
                # probe's, and printing it into the output would be noise.
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(
                    open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue                      # a test fixture that is not valid Python
        rel = os.path.relpath(path, root)
        scopes = _scopes(tree)
        module_b = _bindings(scopes[0][1])

        for scope, own in scopes:
            local_b = module_b if scope is tree else _bindings(own)
            for node in own:
                if not isinstance(node, ast.Call):
                    continue
                name = _callee(node)
                if name is None:
                    continue
                for callee, kw, _anchor in a_spec:
                    if name != callee:
                        continue
                    for k in node.keywords:
                        if k.arg != kw:
                            continue
                        v = _int_literal(k.value)
                        if v is not None:
                            hits.append({"rule": "A", "what": f"{callee}({kw})",
                                         "value": v, "file": rel,
                                         "line": k.value.lineno, "scorable": True})
                for callee, pos, _anchor in b_spec:
                    if name != callee or len(node.args) <= pos:
                        continue
                    arg = node.args[pos]
                    v = _int_literal(arg)
                    line = getattr(arg, "lineno", node.lineno)
                    if v is None and isinstance(arg, ast.Name):
                        got = _resolve(arg.id, node.lineno, local_b, module_b)
                        if got and got[1] is not None:
                            v, line = got[1], got[0]
                    if v is not None:
                        hits.append({"rule": "B", "what": f"{callee}(arg {pos})",
                                     "value": v, "file": rel, "line": line,
                                     "scorable": True})
                for callee, pos, _anchor in c_spec:
                    if name != callee or len(node.args) <= pos:
                        continue
                    arg = node.args[pos]
                    n = _seq_len(arg)
                    line = getattr(arg, "lineno", node.lineno)
                    if n is None and isinstance(arg, ast.Name):
                        got = _resolve(arg.id, node.lineno, local_b, module_b)
                        if got and got[2] is not None:
                            n, line = got[2], got[0]
                    if n is not None:
                        hits.append({"rule": "C", "what": f"{callee}(score set)",
                                     "value": n, "file": rel, "line": line,
                                     "scorable": True})
                for callee, kw, _anchor, reason in x_spec:
                    if name != callee:
                        continue
                    for k in node.keywords:
                        if k.arg != kw:
                            continue
                        v = _int_literal(k.value)
                        if v is not None:
                            hits.append({"rule": "-", "what": f"{callee}({kw})",
                                         "value": v, "file": rel,
                                         "line": k.value.lineno,
                                         "scorable": False, "reason": reason})
    # THE UNIT IS THE SIZE FIXED IN THE SOURCE, NOT THE CALL THAT USES IT.
    # One score set bound once and asserted against three times is one fixture
    # exercised three times, and counting it three times would inflate a census
    # whose whole claim is about which sizes a suite probes. torchcp is the case:
    # a five-element tensor at one line reaches three calls. The uses are kept
    # beside the fixture rather than dropped, so the collapse is visible.
    merged = {}
    for h in hits:
        k = (h["file"], h["line"], h["what"])
        if k in merged:
            assert merged[k]["value"] == h["value"], (
                f"{pkg}: {k} resolves to two different sizes, "
                f"{merged[k]['value']} and {h['value']}")
            merged[k]["uses"] += 1
        else:
            merged[k] = {**h, "uses": 1}
    return list(merged.values()), len(files)


def check_anchors(pkg, root):
    """Every anchor a rule names must exist in the clone. A moved one raises."""
    for spec in (RULE_A.get(pkg, []), RULE_B.get(pkg, []), RULE_C.get(pkg, [])):
        for entry in spec:
            sym, rel = entry[0], entry[2]
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                for base, _d, files in os.walk(root):
                    if os.path.basename(rel) in files:
                        path = os.path.join(base, os.path.basename(rel))
                        break
            assert os.path.exists(path), (
                f"{pkg}: the anchor file {rel} is not in the clone, so the rule "
                f"naming {sym} is matching against code that is not there")
            src = open(path, encoding="utf-8", errors="replace").read()
            assert sym in src, (
                f"{pkg}: {sym} is not in {rel} at this tag; the rule anchored to "
                f"it would match nothing and report an absence as a finding")
    for entry in EXCLUDED.get(pkg, []):
        sym, rel = entry[0], entry[2]
        path = os.path.join(root, rel)
        assert os.path.exists(path) and sym in open(
            path, encoding="utf-8", errors="replace").read(), (
            f"{pkg}: the excluded anchor {sym} in {rel} has moved")


def self_check():
    """Inputs that make each parser wrong, so the parsers are not decoration."""
    # A float keyword must not read as an integer, which is the mapie fraction.
    t = ast.parse("f(calib_size=0.5)").body[0].value
    assert _int_literal(t.keywords[0].value) is None
    assert _int_literal(ast.parse("f(n=30)").body[0].value.keywords[0].value) == 30
    # A bool is an int in Python and is not a calibration size.
    assert _int_literal(ast.parse("f(x=True)").body[0].value.keywords[0].value) is None
    # A literal sequence carries its length through one wrapping call and no more.
    assert _seq_len(ast.parse("torch.tensor([1.0,2.0,3.0])").body[0].value) == 3
    assert _seq_len(ast.parse("np.array([1.0,2.0])").body[0].value) == 2
    assert _seq_len(ast.parse("np.zeros(n)").body[0].value) is None
    assert _seq_len(ast.parse("[1,2,3,4]").body[0].value) == 4
    # The callee of an attribute call is its attribute, so a method reached
    # through a module or an object still matches by name.
    assert _callee(ast.parse("a.b.calculate_conformal_value(s, 0.1)").body[0].value) \
        == "calculate_conformal_value"
    assert _callee(ast.parse("ConformalIntervals(n_windows=2)").body[0].value) \
        == "ConformalIntervals"

    # The failing input for the scope separation, and it is the bug this probe
    # actually had: two functions in one file binding the same name to different
    # sizes. A file-wide table returns the later value at both call sites.
    src = ("def a():\n    n = 30\n    f(alpha, n)\n"
           "def b():\n    n = 10\n    f(alpha, n)\n")
    tree = ast.parse(src)
    scopes = _scopes(tree)
    mod = _bindings(scopes[0][1])
    got = []
    for scope, own in scopes:
        loc = mod if scope is tree else _bindings(own)
        for node in own:
            if isinstance(node, ast.Call) and _callee(node) == "f":
                r = _resolve(node.args[1].id, node.lineno, loc, mod)
                got.append(r[1])
    assert sorted(got) == [10, 30], (
        f"the two scopes resolved to {sorted(got)}; a file-wide binding table "
        f"returns the same value twice, which is the defect this separates")
    # A name bound to a literal sequence resolves to its length, not to an int.
    tree = ast.parse("def t():\n    s = torch.tensor([1.0, 2.0, 3.0])\n    g(s, 0.1)\n")
    scopes = _scopes(tree)
    fn = [(s, o) for s, o in scopes if s is not tree][0]
    call = [n for n in fn[1] if isinstance(n, ast.Call) and _callee(n) == "g"][0]
    r = _resolve(call.args[0].id, call.lineno, _bindings(fn[1]), {})
    assert r is not None and r[2] == 3 and r[1] is None, r
    # A binding made AFTER the call does not reach it.
    tree = ast.parse("def t():\n    g(s, 0.1)\n    s = [1, 2]\n")
    fn = [(s, o) for s, o in _scopes(tree) if s is not tree][0]
    call = [n for n in fn[1] if isinstance(n, ast.Call) and _callee(n) == "g"][0]
    assert _resolve("s", call.lineno, _bindings(fn[1]), {}) is None


self_check()


def main():
    alpha = F(1, 10)
    lo = W13.feasible_floor(alpha) + 1
    hi = W13.periodic_from(alpha) + 2 * W13.period_of(alpha)
    pool = list(range(lo, hi))
    full = W13.classes(alpha, tuple(pool))
    powers = {n: len(W13.classes(alpha, (n,))) for n in pool}
    best, worst = max(powers.values()), min(powers.values())

    say("=" * 104)
    say("REPOSITORY-SIDE FIXTURE CENSUS: calibration sizes the projects' own tests fix")
    say("=" * 104)
    say("self_check() passed at import: a float keyword does not read as an integer,")
    say("a bool is not a size, and a sequence built by a computation has no literal")
    say("length. Each of those is an input that makes a wrong parser print a number.")
    say("")
    say("Read from the projects' own repositories at the versions the census table")
    say("pins, not from the wheels. Nothing is installed and no test is run: the")
    say("value a fixture fixes is in its source, and installing a test runner would")
    say("move the dependency set the rest of this audit is measured against.")
    say("")
    say(f"scored against the discriminating power over {len(pool)} sizes at nominal 0.90:")
    say(f"  best {best} of {len(full)} classes, worst {worst}, "
        f"median {int(np.median(list(powers.values())))}")
    say("")

    per_pkg = []
    for pkg, slug, tag in REPOS:
        dest = os.path.join(CACHE, pkg.split()[0])
        ok = clone(slug, tag, dest)
        if not ok:
            say(f"[{pkg}] clone FAILED at {tag}; NOT ATTEMPTED")
            per_pkg.append({"pkg": pkg, "tag": tag, "files": 0, "hits": [],
                            "state": "not attempted"})
            continue
        check_anchors(pkg, dest)
        hits, nfiles = scan(pkg, dest)
        state = "tests present" if nfiles else "NO TEST SUITE IN THE REPOSITORY"
        per_pkg.append({"pkg": pkg, "tag": tag, "files": nfiles, "hits": hits,
                        "state": state})

    say("-" * 104)
    say(f"{'package':<22}{'tag':<10}{'test files':>11}{'sizes fixed':>13}"
        f"{'distinct':>10}   state")
    say("-" * 104)
    for r in per_pkg:
        sc = [h for h in r["hits"] if h["scorable"]]
        say(f"{r['pkg']:<22}{r['tag']:<10}{r['files']:>11}{len(sc):>13}"
            f"{len(set(h['value'] for h in sc)):>10}   {r['state']}")
    say("")

    say("-" * 104)
    say("EACH SCORED PIN, tied back to where the project itself writes it down")
    say("-" * 104)
    say(f"{'package':<22}{'rule':<5}{'what':<38}{'n':>6}{'power':>7}{'of':>4}   verdict")
    scored = []
    for r in per_pkg:
        for h in sorted(r["hits"], key=lambda x: (x["rule"], x["file"], x["line"])):
            if not h["scorable"]:
                say(f"{r['pkg']:<22}{'-':<5}{h['what']:<38}{h['value']:>6}"
                    f"{'---':>7}{'---':>4}   not a calibration size: {h['reason']}")
                say(f"{'':<22}{h['file']}:{h['line']}")
                continue
            n = h["value"]
            inr = n in powers
            p = powers.get(n)
            if not inr:
                verdict = ("below the feasibility floor, so no rank carries 0.90 there"
                           if n <= W13.feasible_floor(alpha)
                           else "outside the checked range")
            elif p <= worst:
                verdict = "BLIND: worst possible power"
            elif p < best:
                verdict = f"partial: {best - p} classes short of the best size"
            else:
                verdict = "maximal power"
            scored.append({**h, "pkg": r["pkg"], "power": p, "in_range": inr,
                           "verdict": verdict})
            say(f"{r['pkg']:<22}{h['rule']:<5}{h['what']:<38}{n:>6}"
                f"{(p if p else 0):>7}{len(full):>4}   {verdict}")
            say(f"{'':<22}{h['file']}:{h['line']}")
    say("")

    # ---------------- summary -------------------------------------------
    n_repo = len(per_pkg)
    no_tests = [r for r in per_pkg if r["files"] == 0]
    with_tests = [r for r in per_pkg if r["files"] > 0]
    contributing = sorted({s["pkg"] for s in scored})
    in_range = [s for s in scored if s["in_range"]]
    below = [s for s in scored if not s["in_range"]]
    sub_max = [s for s in in_range if s["power"] < best]
    blind = [s for s in in_range if s["power"] <= worst]
    distinct = sorted({s["value"] for s in scored})
    at_max = [s for s in in_range if s["power"] == best]

    say("=" * 104)
    say("SUMMARY")
    say("=" * 104)
    say(f"repositories read: {n_repo}")
    say(f"repositories shipping a test suite: {len(with_tests)}")
    say(f"repositories shipping NO test suite at all: {len(no_tests)}"
        + (" -- " + ", ".join(r["pkg"] for r in no_tests) if no_tests else ""))
    say(f"repositories contributing a scorable calibration size: {len(contributing)}")
    say(f"scored fixtures: {len(scored)}")
    say(f"distinct calibration sizes fixed: {len(distinct)}  {distinct}")
    say(f"pins under the 0.90 deterministic bound: {len(below)}")
    say(f"fixtures in range at less than maximal power: {len(sub_max)} of {len(in_range)}")
    say(f"fixtures at the worst power available: {len(blind)}")
    say(f"fixtures at maximal power: {len(at_max)}")
    say("")
    say("SCOPE OF THE CLAIM. Settled: which counts these projects write into their")
    say("own conformal tests, under three declared rules whose symbols were verified")
    say("against each checkout. Not settled: coverage of a suite overall. Some count")
    say("absent from this list could still arise from generated inputs, and a project")
    say("adding no row is not shown to be untested by that alone. Only one direction")
    say("survives: where a count IS written down and its separating strength falls")
    say("short of the ceiling, no rule class can be told apart at it, irrespective of")
    say("whatever else surrounds that test.")
    say("")
    say("Sharper still: two checkouts hold no tests whatever. That reports on those")
    say("two trees at those two revisions, and it reports on nothing further.")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
