# The ten-minute path. Three pinned environments cover full reproduction, since the
# audited libraries pin conflicting numpy and pandas releases; this target needs
# only one of them plus numpy, and rebuilds the headline numbers from scratch.
#
#   make verify    the headline numbers, minutes

PY ?= python3

.PHONY: verify
verify:
	$(PY) verify_headline.py
