# Animal Face CBIR

A content-based image retrieval system for animal faces. Built with handcrafted CV features (HSV / Color Moments / LBP / GLCM / HOG / Hu Moments) — no deep learning. The UI is the centrepiece: every preprocessing step, every feature, every similarity score is inspectable in the browser.

> **Status: Phase 0 (Bootstrap)** — docker-compose, FastAPI hello-world, Next.js hello-world, test runners wired. See [PLAN.md](./PLAN.md) for the full roadmap.

---

## Stack

| Layer    | Tech                                                       |
|----------|------------------------------------------------------------|
| DB       | PostgreSQL 16 (no extensions)                              |
| Backend  | FastAPI · Python 3.12 · SQLAlchemy 2.0 (async) · Alembic   |
| CV / ML  | opencv-python-headless · scikit-image · numpy · matplotlib |
| Frontend | Next.js 15 (App Router) · Tailwind · shadcn/ui · Recharts  |
| Testing  | pytest + httpx (BE), Vitest + Playwright (FE)              |
| Infra    | Docker Compose                                             |

---

## Quick start

```bash
cp .env.example .env
make build       # build all images
make up          # start postgres + backend + frontend
make test        # run BE pytest + FE vitest
```

Open:
- Frontend: <http://localhost:3000>
- Backend OpenAPI: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

```bash
make help        # full target listing
make logs        # tail all services
make down        # stop (keep volumes)
make clean       # stop and wipe volumes (destructive)
```

---

## Repository layout

```
.
├── backend/          # FastAPI service
├── frontend/         # Next.js app
├── docs/BASE.md      # original CV/ML methodology spec
├── PLAN.md           # active implementation plan
├── PLAN.v1.md        # earlier plan (kept for reference)
├── docker-compose.yml
└── Makefile
```

See [PLAN.md §4](./PLAN.md) for the full layout and [PLAN.md §11](./PLAN.md) for the phase-by-phase roadmap.
