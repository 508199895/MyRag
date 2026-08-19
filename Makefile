PYTHON ?= ./ENV/RAG_2026/python.exe
PYTEST_BASETEMP ?= .pytest_tmp

.PHONY: ci-local test test-cov test-prompts-cov

ci-local: test-cov test-prompts-cov
	@echo "ci-local passed"

test-cov:
	$(PYTHON) -m pytest tests/unit/test_config.py -v --tb=short --cov=src.config --cov-report=term-missing --cov-fail-under=100 --basetemp=$(PYTEST_BASETEMP)

test-prompts-cov:
	$(PYTHON) -m pytest tests/unit/test_prompts.py -v --cov=src.generation.prompts --cov-report=term-missing --cov-fail-under=100
