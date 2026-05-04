SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---------- Help ----------
help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------- Stack lifecycle ----------
build: ## Build all docker images.
	docker compose build

up: ## Start full stack (postgres + backend + frontend) detached.
	docker compose up -d

up-fg: ## Start full stack in the foreground.
	docker compose up

down: ## Stop and remove containers (keeps volumes).
	docker compose down

clean: ## Stop and remove containers AND volumes (DESTRUCTIVE).
	docker compose down -v

logs: ## Tail logs for all services.
	docker compose logs -f

ps: ## List running containers.
	docker compose ps

# ---------- Testing ----------
test: test-backend test-frontend ## Run unit + integration tests (BE pytest + FE vitest).

test-backend: ## Run backend pytest inside a freshly-built backend image.
	docker compose run --rm --build backend pytest -q

test-backend-local: ## Run backend pytest using a local uv venv (faster iteration).
	cd backend && uv sync --quiet --extra dev && uv run pytest -q

test-frontend: ## Run frontend Vitest unit tests.
	cd frontend && yarn install --frozen-lockfile --silent && yarn vitest run

e2e: ## Run Playwright e2e tests (requires `make up` first).
	cd frontend && yarn playwright test

# ---------- Dev ergonomics ----------
shell-backend: ## Open a bash shell inside the backend container.
	docker compose exec backend bash

shell-postgres: ## Open psql inside the postgres container.
	docker compose exec postgres psql -U $${POSTGRES_USER:-app} -d $${POSTGRES_DB:-cbir}

migrate: ## Run alembic migrations (added in Phase 1).
	docker compose exec backend alembic upgrade head

seed: ## Populate corpus with AFHQ subset (added in Phase 6).
	docker compose exec backend python -m scripts.seed_afhq

.PHONY: help build up up-fg down clean logs ps test test-backend test-backend-local test-frontend e2e shell-backend shell-postgres migrate seed
