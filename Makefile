# Makefile for Face and Body Swap Pipeline

.PHONY: help install test clean run-api run-cli docker-build docker-run setup

help:
	@echo "Face and Body Swap Pipeline - Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run setup verification"
	@echo "  make setup        - Complete setup (install + test)"
	@echo "  make run-api      - Start API server"
	@echo "  make run-cli      - Show CLI usage"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run with Docker Compose"
	@echo "  make clean        - Clean temporary files"

install:
	pip install -r requirements.txt

test:
	python scripts/test_setup.py

setup: install test
	@echo "Setup complete!"

run-api:
	python -m src.api.main

run-cli:
	@echo "CLI usage:"
	@echo "  python -m src.api.cli swap --customer-photos <photo> --template <template> --output <output>"
	@python scripts/run_example.py

docker-build:
	docker build -t face-body-swap .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "Clean complete!"

