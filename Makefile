.PHONY: semantic-conflicts test audit install

PYTHON ?= python3
export PYTHONPATH := semantic_conflicts/src:$(PYTHONPATH)

install:
	$(PYTHON) -m pip install -e ".[dev]"

semantic-conflicts:
	$(PYTHON) -m semantic_conflicts.pipeline deterministic

check:
	$(PYTHON) -m semantic_conflicts.pipeline check

audit:
	$(PYTHON) -m semantic_conflicts.audit --strict

test:
	$(PYTHON) -m pytest semantic_conflicts/tests -q
