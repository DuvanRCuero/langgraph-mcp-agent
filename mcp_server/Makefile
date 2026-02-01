.PHONY: help install dev-install test lint format clean run docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  dev-install  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linters"
	@echo "  format       Format code"
	@echo "  clean        Clean up temporary files"
	@echo "  run          Run development server"
	@echo "  docker-build Build Docker images"
	@echo "  docker-up    Start Docker services"
	@echo "  docker-down  Stop Docker services"

install:
	@echo "Installing production dependencies..."
	poetry install --no-dev

dev-install:
	@echo "Installing development dependencies..."
	poetry install

test:
	@echo "Running tests..."
	poetry run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	@echo "Running linters..."
	poetry run flake8 src/ tests/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/
	poetry run mypy src/

format:
	@echo "Formatting code..."
	poetry run black src/ tests/
	poetry run isort src/ tests/

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name "*.pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".hypothesis" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage .pytest_cache build/ dist/

run:
	@echo "Starting development server..."
	poetry run python -m src.server

docker-build:
	@echo "Building Docker images..."
	docker-compose -f docker/docker-compose.yml build

docker-up:
	@echo "Starting Docker services..."
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	@echo "Showing Docker logs..."
	docker-compose -f docker/docker-compose.yml logs -f

setup-pre-commit:
	@echo "Setting up pre-commit hooks..."
	poetry run pre-commit install

update-dependencies:
	@echo "Updating dependencies..."
	poetry update

check-security:
	@echo "Checking for security vulnerabilities..."
	poetry export --without-hashes --format=requirements.txt | safety check --stdin

generate-docs:
	@echo "Generating API documentation..."
	poetry run python -m src.generate_docs

benchmark:
	@echo "Running benchmarks..."
	poetry run python -m src.benchmark