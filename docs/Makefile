# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
YAML2RST      = yaml2rst

../docs/source/%.rst:
	$(YAML2RST) ../src/locpix/templates/$*.yaml $@

ITEMS := $(wildcard ../src/locpix/templates/*.yaml)
TEMPLATES = $(patsubst ../src/locpix/templates/%.yaml, ../docs/source/%.rst, $(ITEMS))

templates: $(TEMPLATES)

clean_templates:
ifneq($(TEMPLATES),)
	rm -f $(TEMPLATES)
endif

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
