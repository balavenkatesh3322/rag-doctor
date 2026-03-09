# rag-doctor — Makefile
# Usage: make <target>

.PHONY: help install install-full test test-pytest test-samples lint format \
        build clean publish-test publish github docs-check

help:
	@echo ""
	@echo "  🩺  rag-doctor — development commands"
	@echo ""
	@echo "  make install        Install in editable mode (core only)"
	@echo "  make install-full   Install with sentence-transformers + dev deps"
	@echo "  make test           Run 62 offline unit tests"
	@echo "  make test-pytest    Run with pytest + coverage report"
	@echo "  make test-samples   Run all 6 samples"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Auto-format with ruff"
	@echo "  make build          Build wheel + sdist"
	@echo "  make clean          Remove build artifacts"
	@echo "  make publish-test   Upload to TestPyPI"
	@echo "  make publish        Upload to PyPI (live!)"
	@echo "  make github URL=https://github.com/ORG/REPO.git  Push to GitHub"
	@echo ""

install:
	pip install -e "."

install-full:
	pip install -e ".[dev]"
	pip install sentence-transformers
	pre-commit install

test:
	python run_tests.py

test-pytest:
	pytest tests/ -v --cov=rag_doctor --cov-report=term-missing

test-samples:
	@echo "Running all samples..."
	@for f in samples/0*.py; do \
		echo -n "  $$f: "; \
		python $$f > /dev/null 2>&1 && echo "OK" || echo "FAIL (check manually)"; \
	done

lint:
	ruff check .

format:
	ruff check . --fix
	ruff format .

build: clean
	python -m build
	twine check dist/*

clean:
	rm -rf dist/ build/ *.egg-info/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

publish-test: build
	twine upload --repository testpypi dist/*

publish: build
	twine upload dist/*

github:
	@if [ -z "$(URL)" ]; then \
		echo "Usage: make github URL=https://github.com/ORG/rag-doctor.git"; \
	else \
		./scripts/push_github.sh $(URL); \
	fi
