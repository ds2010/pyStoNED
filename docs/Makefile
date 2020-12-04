# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build/

default: html
	@true  # this is needed, or the % rule below is done for default.
	@echo
	@echo "Build successful.  Run 'make check' to check for important warnings"

check:
	make html SPHINXOPTS="-w warnings.txt"
	python meta/check-warnings.py --fail warnings.txt
	@echo
	@echo "OK: no errors"

# Put it first so that "make" without argument is like "make help".
help:
	$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: default help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
