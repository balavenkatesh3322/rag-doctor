# rag-doctor Makefile
# Usage: make <target>

.PHONY: help install test lint format build clean publish-test publish

help:
	@echo "rag-doctor development commands"
	@echo ""
	@echo "  make install       Install in editable mode with dev deps"
	@echo "  make test          Run all tests"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Auto-format with ruff"
	@echo "  make build         Build distribution packages"
	@echo "  make clean         Remove build artifacts"
	@echo "  make publish-test  Upload to TestPyPI"
	@echo "  make publish       Upload to PyPI"

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	python run_tests.py

test-pytest:
	pytest tests/ -v --cov=rag_doctor --cov-report=term-missing

lint:
	ruff check .

format:
	ruff check . --fix
	ruff format .

build: clean
	python -m build
	twine check dist/*

clean:
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

publish-test: build
	twine upload --repository testpypi dist/*

publish: build
	twine upload dist/*
