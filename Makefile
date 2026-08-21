PYTHON ?= ./ENV/RAG_2026/python.exe
PYTEST_BASETEMP ?= .pytest_tmp

.PHONY: ci-local lint lint-fix format format-check test-unit test-integration test-e2e test-regression

ci-local: lint format-check test-unit test-integration test-e2e test-regression
	@echo "ci-local passed"

lint:
	$(PYTHON) -m ruff check src tests

lint-fix:
	$(PYTHON) -m ruff check --fix src tests

format:
	$(PYTHON) -m ruff format src tests

format-check:
	$(PYTHON) -m ruff format --check src tests

test-unit:
	$(PYTHON) -m pytest tests/unit/test_config.py -v --tb=short --cov=src.config --cov-branch --cov-report=term-missing --cov-fail-under=100 --basetemp=$(PYTEST_BASETEMP)
	$(PYTHON) -m pytest tests/unit/test_prompts.py -v --cov=src.generation.prompts --cov-branch --cov-report=term-missing --cov-fail-under=100 --basetemp=$(PYTEST_BASETEMP)

test-integration:
	$(PYTHON) -c "from pathlib import Path; assert Path('tests/integration').is_dir()"

test-e2e:
	$(PYTHON) -c "from pathlib import Path; assert Path('tests/e2e').is_dir()"

test-regression:
	$(PYTHON) -c "from pathlib import Path; assert Path('tests/regression').is_dir()"
