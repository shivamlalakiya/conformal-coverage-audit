#!/usr/bin/env python3
"""Run every reproducer in this directory and report which findings still reproduce.

Each script here targets ONE upstream filing, imports only its own library, and exits
0 when the finding reproduces and 1 when it does not. That polarity is deliberate: a
maintainer who fixes a defect should see this runner turn red on that row, and a red
row here is good news rather than a broken test.

Scripts are matched to the environment by the library named in the filename, because
the audited libraries pin incompatible numpy versions and cannot share one virtual
environment. A script whose library is not importable here is reported as NOT RUN, not
as passing.

    python repro/run_all.py
"""

import importlib.util
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_OF = {"mapie": "mapie", "crepes": "crepes", "sktime": "sktime",
          "statsforecast": "statsforecast", "torchcp": "torchcp"}


def scripts():
    out = []
    for name in sorted(os.listdir(HERE)):
        if name.endswith(".py") and name != os.path.basename(__file__):
            lib = next((v for k, v in LIB_OF.items() if name.startswith(k)), None)
            out.append((name, lib))
    return out


def main():
    rows = scripts()
    if not rows:
        print("no reproducers found; nothing to run")
        return 1
    print(f"{'script':<44}{'library':<15}{'result':<16}{'secs':>6}")
    print("-" * 82)
    tally = {"REPRODUCES": 0, "does not": 0, "NOT RUN": 0, "ERROR": 0}
    for name, lib in rows:
        if lib and importlib.util.find_spec(lib) is None:
            print(f"{name:<44}{lib or '-':<15}{'NOT RUN':<16}{'-':>6}")
            tally["NOT RUN"] += 1
            continue
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, name)],
                           capture_output=True, text=True, timeout=300)
        dt = time.time() - t
        if r.returncode == 0:
            verdict = "REPRODUCES"
        elif r.returncode == 1 and "does not reproduce" in r.stdout:
            verdict = "does not"
        else:
            verdict = "ERROR"
        tally[verdict] += 1
        print(f"{name:<44}{lib or '-':<15}{verdict:<16}{dt:>6.1f}")
        if verdict == "ERROR":
            print(f"    {(r.stderr.strip().splitlines() or ['no stderr'])[-1][:110]}")

    print()
    print(f"{tally['REPRODUCES']} still reproduce, {tally['does not']} no longer do, "
          f"{tally['NOT RUN']} not runnable in this environment, "
          f"{tally['ERROR']} errored")
    print()
    print("A row that stops reproducing is the outcome this project is filed to")
    print("produce. Check it against the upstream thread before assuming the probe")
    print("broke: the library may simply have been fixed.")
    return 0 if not tally["ERROR"] else 1


if __name__ == "__main__":
    sys.exit(main())
