.PHONY: help install test clean lint format quickstart examples benchmark docs

help:
	@echo "i-LAVA Voice-to-Voice Pipeline - Available Commands"
	@echo "=================================================="
	@echo "  make install       - Install dependencies"
	@echo "  make quickstart    - Run quick start demo"
	@echo "  make test          - Run test suite"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make examples      - Run all examples"
	@echo "  make benchmark     - Run benchmarks"
	@echo "  make clean         - Clean temporary files"
	@echo "  make docs          - Show documentation files"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install -e ".[dev]"

quickstart:
	@echo "Running quick start demo..."
	python quickstart.py

test:
	@echo "Running test suite..."
	pytest -v

test-cov:
	@echo "Running tests with coverage..."
	pytest --cov=ilava --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-fast:
	@echo "Running fast tests (skipping integration)..."
	pytest -v -m "not integration"

lint:
	@echo "Running linters..."
	flake8 ilava tests examples --max-line-length=100 --ignore=E203,W503
	mypy ilava --ignore-missing-imports

format:
	@echo "Formatting code..."
	black ilava tests examples
	isort ilava tests examples

examples:
	@echo "Running basic conversation example..."
	python examples/basic_conversation.py
	@echo ""
	@echo "Running streaming demo..."
	python examples/streaming_demo.py

benchmark:
	@echo "Running benchmarks..."
	python examples/benchmark.py

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/
	rm -f output_*.wav synthetic_input.wav streaming_output.wav
	@echo "Cleanup complete"

docs:
	@echo "Documentation files:"
	@echo "===================="
	@echo "  README.md                - Main documentation"
	@echo "  USAGE.md                 - Detailed usage guide"
	@echo "  IMPLEMENTATION_NOTES.md  - Technical details"
	@echo "  PROJECT_SUMMARY.md       - Project overview"
	@echo ""
	@echo "Use 'cat <filename>' to view any file"

check-env:
	@echo "Checking environment..."
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@python -c "import os; print(f'OpenAI API Key: {\"Set\" if os.getenv(\"OPENAI_API_KEY\") else \"Not Set\"}')"

info:
	@echo "i-LAVA Voice-to-Voice Pipeline"
	@echo "=============================="
	@echo "Paper: https://arxiv.org/html/2509.20971v1"
	@echo ""
	@echo "Project Structure:"
	@echo "  ilava/           - Main package"
	@echo "  examples/        - Usage examples"
	@echo "  tests/           - Test suite"
	@echo ""
	@echo "Quick Start: make quickstart"
	@echo "Run Tests:   make test"
	@echo "View Docs:   make docs"

