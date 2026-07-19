.DEFAULT_GOAL := help

.PHONY: help install install-all run dev format lint typecheck test test-unit test-integration test-model eval check security build docker-build up down logs qdrant-up qdrant-down model-download model-smoke

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target>\n\n"} /^[a-zA-Z_-]+:.*?##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install locked runtime and development dependencies
	uv sync --locked --all-groups

install-all: ## Install locked dependencies including optional hybrid retrieval
	uv sync --locked --all-groups --all-extras

run: ## Run the API with one model-owning worker
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

dev: ## Run the API with auto-reload and one worker
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

format: ## Format code and fix safe lint violations
	uv run ruff format .
	uv run ruff check . --fix

lint: ## Check formatting and lint rules
	uv run ruff format . --check
	uv run ruff check .

typecheck: ## Run strict static type checks
	uv run mypy

test: ## Run offline tests with branch coverage
	uv run pytest -m "not model and not network and not slow" --cov=app --cov-report=term-missing --cov-report=xml

test-unit: ## Run deterministic unit tests
	uv run pytest -m unit

test-integration: ## Run in-process integration tests
	uv run pytest -m integration

test-model: ## Run opt-in tests against pre-downloaded local model files
	uv run pytest -m model --no-cov

eval: ## Run offline chat, retrieval, RAG, and security evaluations
	uv run python -m evals.run

check: lint typecheck test eval ## Run every offline local quality gate

security: ## Audit locked Python dependencies for known vulnerabilities
	uv run pip-audit

build: ## Build Python source and wheel distributions
	uv build

docker-build: ## Build the production container without model weights
	docker build --tag ai-engineer-template:local .

up: ## Start the hardened application service
	docker compose up --build --detach app

down: ## Stop the Compose stack
	docker compose --profile qdrant-server down

logs: ## Follow application logs
	docker compose logs --follow app

qdrant-up: ## Start the optional Qdrant server profile
	docker compose --profile qdrant-server up --detach qdrant

qdrant-down: ## Stop the optional Qdrant server
	docker compose --profile qdrant-server stop qdrant

model-download: ## Download pinned generator and embedding snapshots explicitly
	@test -n "$(GENERATOR_ID)" -a -n "$(GENERATOR_REVISION)" -a -n "$(EMBEDDING_ID)" -a -n "$(EMBEDDING_REVISION)" || (echo "Set GENERATOR_ID, GENERATOR_REVISION, EMBEDDING_ID, and EMBEDDING_REVISION"; exit 2)
	uv run ai-template-download-models --generator-id "$(GENERATOR_ID)" --generator-revision "$(GENERATOR_REVISION)" --embedding-id "$(EMBEDDING_ID)" --embedding-revision "$(EMBEDDING_REVISION)"

model-smoke: ## Load and query a pre-downloaded local model
	uv run ai-template-model-smoke
