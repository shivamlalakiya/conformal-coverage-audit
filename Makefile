# The ten-minute path. Full reproduction needs three pinned environments because the
# audited libraries disagree on numpy and pandas; `make verify` needs one and numpy,
# and re-derives the quantities in the audit's abstract from first principles.
#
#   make verify    the headline numbers, minutes
#   make ledger    the macro ledgers both manuscripts point at
#   make all       both

PY ?= ./.venv-real/bin/python

.PHONY: verify ledger all
verify:
	$(PY) verify_headline.py

ledger:
	$(PY) ../paperlib/ledger.py paper1
	$(PY) ../paperlib/ledger.py paper2

all: verify ledger
