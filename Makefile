PYTHON ?= ./ENV/RAG_2026/python.exe
PYTEST_BASETEMP ?= .pytest_tmp

.PHONY: ci-local test test-cov

ci-local: test-cov
	@echo "ci-local passed"

test-cov:
	$(PYTHON) -m pytest tests -v --tb=short --cov=src --cov-report=term-missing --basetemp=$(PYTEST_BASETEMP)
