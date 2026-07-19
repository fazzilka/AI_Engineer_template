.DEFAULT_GOAL := help

.PHONY: help install run dev format lint typecheck test eval check build docker-build up down logs

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target>\n\n"} /^[a-zA-Z_-]+:.*?##/ {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install locked development dependencies
	uv sync --locked --dev

run: ## Run the API
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

dev: ## Run the API with auto-reload
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

format: ## Format code and fix safe lint violations
	uv run ruff format .
	uv run ruff check . --fix

lint: ## Check formatting and lint rules
	uv run ruff format . --check
	uv run ruff check .

typecheck: ## Run strict static type checks
	uv run mypy

test: ## Run tests with branch coverage
	uv run pytest --cov=app --cov-report=term-missing --cov-report=xml

eval: ## Run the offline-safe AI evaluation set
	uv run python -m evals.run

check: lint typecheck test eval ## Run every local quality gate

build: ## Build Python source and wheel distributions
	uv build

docker-build: ## Build the production container
	docker build --tag ai-engineer-template:local .

up: ## Start the service with Docker Compose
	docker compose up --build --detach

down: ## Stop the Docker Compose stack
	docker compose down

logs: ## Follow application logs
	docker compose logs --follow app
