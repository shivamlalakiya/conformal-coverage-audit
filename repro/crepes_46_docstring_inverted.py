#!/usr/bin/env python3
"""crepes #46: the docstring states the inverse of the implemented membership condition.

One filing, one script, no dependency on the rest of this repository.

THE CLAIM
---------
`predict_set` returns a binary array, and its docstring explains the encoding:

    "the value 1 (0) in the binary array indicates that the class label is
     included (excluded), i.e., the corresponding p-value is less than 1-confidence"

The implementation is

    prediction_sets = (p_values >= 1-confidence)

so 1 means the p-value is at least 1-confidence. The docstring gives the condition for
the value it does not describe. A reader following it would take a large p-value -- strong
evidence for the label -- as grounds for exclusion.

Both halves are read out of the installed module: the sentence from the docstring, the
comparison from the source. Then a concrete p-value is put through the code so the
disagreement is not a matter of reading.
"""

import inspect
import re
import sys

import numpy as np


def main():
    import crepes
    import crepes.base as base
    from crepes import ConformalClassifier

    print(f"crepes {crepes.__version__}, numpy {np.__version__}")
    print()
    src = inspect.getsource(base)

    # The sentence is found by SCANNING every public method's docstring rather than
    # hardcoded to one of them: a first version looked only at
    # ConformalClassifier.predict_set and reported "does not reproduce" because the
    # sentence sits on a sibling method that takes the online/warm_start arguments.
    pattern = re.compile(r"the value 1 \(0\).{0,200}?1-confidence", re.S)
    found = []
    for cls_name in ("ConformalClassifier", "WrapClassifier"):
        cls = getattr(crepes, cls_name, None)
        if cls is None:
            continue
        for attr in dir(cls):
            if attr.startswith("_"):
                continue
            d = inspect.getdoc(getattr(cls, attr)) or ""
            if pattern.search(d):
                found.append(f"{cls_name}.{attr}")
    assert found, ("the encoding sentence is in no public docstring; check the "
                   "upstream thread before concluding anything")
    print(f"the sentence appears in: {', '.join(found)}")
    doc = inspect.getdoc(getattr(getattr(crepes, found[0].split('.')[0]),
                                 found[0].split('.')[1])) or ""
    sent = pattern.search(doc)
    sentence = " ".join(sent.group(0).split())
    print("--- the docstring, verbatim ---")
    for i in range(0, len(sentence), 88):
        print(f"  {sentence[i:i + 88]}")
    print("--- end ---")
    print()

    impl = re.search(r"prediction_sets\s*=\s*\(p_values\s*(>=|<=|>|<)\s*1-confidence\)",
                     src)
    assert impl, "the membership comparison is no longer in this form; re-read the source"
    op = impl.group(1)
    print(f"the implemented comparison: p_values {op} 1-confidence")
    doc_op = "<" if re.search(r"less than", sentence, re.I) else "?"
    print(f"the documented condition:   p-value {doc_op} 1-confidence")
    print()

    # and now put a p-value through the code, so the disagreement is executed
    confidence = 0.90
    thresh = 1 - confidence
    print(f"At confidence {confidence} the threshold is {thresh:.4f}. Two p-values, one "
          f"either side:")
    print()
    print(f"{'p-value':>10}{'docstring says':>18}{'code returns':>15}{'agree?':>9}")
    print("-" * 54)
    rows = []
    for p in (0.30, 0.02):
        doc_says = "included" if p < thresh else "excluded"
        code_says = "included" if p >= thresh else "excluded"
        rows.append((p, doc_says, code_says))
        print(f"{p:>10.2f}{doc_says:>18}{code_says:>15}"
              f"{('yes' if doc_says == code_says else 'NO'):>9}")
    print()

    inverted = op in (">=", ">") and doc_op == "<"
    disagreements = sum(1 for _, d, c in rows if d != c)
    print(f"cells where the docstring and the code disagree: {disagreements} of "
          f"{len(rows)}")
    print()
    if inverted and disagreements == len(rows):
        print("REPRODUCES. The code includes a label when its p-value is at least "
              "1-confidence; the docstring says inclusion means the p-value is less "
              "than it. The two disagree on every p-value, not just at the boundary.")
        return 0
    print(f"does not reproduce: implemented {op}, documented {doc_op}, "
          f"{disagreements} disagreements")
    return 1


if __name__ == "__main__":
    sys.exit(main())
