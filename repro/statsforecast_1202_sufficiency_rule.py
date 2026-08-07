#!/usr/bin/env python3
"""statsforecast #1202: CSP's documented sufficiency rule names one rail, and its
worked example is off by one.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
`ConformalSeasonalPool`'s docstring states a sample-count requirement for a level-L
interval and illustrates it with a number. Two things are wrong with the sentence.

  1. Evaluating the stated formula at the level in its own example gives one less than
     the number it gives. The formula is right; the parenthetical is off by one.
  2. The requirement is written for "the orientation-corrected lower bound" -- one rail
     -- while what the user asks for is an interval. A rule stated for a rail reads as a
     rule for the interval.

The formula and the example are both parsed out of the installed docstring, so this
compares the library against itself. Nothing here is paraphrased.
"""

import math
import re
import sys


def main():
    import inspect
    import statsforecast
    from statsforecast.models import ConformalSeasonalPool

    print(f"statsforecast {statsforecast.__version__}")
    print()
    src = inspect.getsource(ConformalSeasonalPool)
    doc = re.search(r'"""(.*?)"""', src, re.S)
    assert doc, "no class docstring found; the filing is about its text"
    text = " ".join(doc.group(1).split())

    # NOT [^.]*\. -- the sentence contains "e.g.", so a period-terminated match stops
    # inside the parenthetical and drops the worked example this filing is about.
    sent = re.search(r"For a level-L interval.*?interval\)", text)
    assert sent, ("the sufficiency sentence is no longer in the docstring; it may have "
                  "been rewritten, in which case check the upstream thread")
    sentence = sent.group(0)
    print("--- the sentence, verbatim from the installed docstring ---")
    for i in range(0, len(sentence), 92):
        print(f"  {sentence[i:i + 92]}")
    print("--- end ---")
    print()

    # the formula, as written
    formula = re.search(r"ceil\(2/\(1\s*-\s*L/100\)\)\s*-\s*1", sentence)
    example = re.search(r"[≥>=]\s*(\d+)\s*for a (\d+)%", sentence)
    assert formula, f"the formula is not in the sentence as expected: {sentence}"
    assert example, f"no worked example found in the sentence: {sentence}"
    claimed, level = int(example.group(1)), int(example.group(2))

    def rule(L):
        return math.ceil(2 / (1 - L / 100)) - 1

    computed = rule(level)
    print(f"{'quantity':<46}{'value':>8}")
    print("-" * 56)
    print(f"{'the formula the docstring states':<46}"
          f"{'ceil(2/(1-L/100)) - 1':>8}")
    print(f"{f'evaluated at its own example level, L = {level}':<46}{computed:>8}")
    print(f"{'the number the docstring gives':<46}{claimed:>8}")
    print(f"{'difference':<46}{claimed - computed:>8}")
    print()

    # and the independent check: the two-rail floor is n >= 2/alpha - 1
    alpha = 1 - level / 100
    floor = 2 / alpha - 1
    print(f"Independently, a two-rail interval needs n >= 2/alpha - 1; at "
          f"alpha = {alpha:.2f} that is {floor:g}.")
    print(f"So the FORMULA agrees with the arithmetic ({computed}) and the "
          f"PARENTHETICAL does not ({claimed}).")
    print()

    rails = re.search(r"lower bound|lower rail", sentence)
    print(f"does the sentence name one rail or the interval? "
          f"{'one rail: ' + rails.group(0) if rails else 'the interval'}")
    print()

    off_by_one = claimed != computed
    one_rail = bool(rails)
    if off_by_one and one_rail:
        print("REPRODUCES. The worked example is one above what the stated formula "
              "gives, and the requirement is written for a single rail while the object "
              "the caller receives is an interval.")
        return 0
    print(f"does not reproduce: off-by-one {off_by_one}, single-rail wording {one_rail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
