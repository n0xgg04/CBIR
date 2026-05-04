# Animal Face CBIR — Implementation Plan v2

> **Reset rationale.** v1 was correct but backend-heavy. v2 puts the **UI query inspector** at the centre: every preprocessing step, every feature, every similarity score is visible and inspectable in the browser. Storage is deliberately boring (PostgreSQL + JSONB, no extensions); complexity lives in the Python service layer where it belongs. v1 is preserved at `PLAN.v1.md` for reference.

---

## 1. Design Principles

1. **The UI is a microscope, not a search box.** The `/search` page must let a user *see* what the system is doing — not just see the answer. Preprocessing intermediates, per-feature visualisations, similarity computations and rankings are all rendered live as the query runs.
2. **Comparison is the default lens.** Query vs. result side-by-side, feature-by-feature. The user should never wonder *why* an image ranked where it did.
3. **Boring storage, expressive services.** PostgreSQL stores rows with `JSONB` for vectors — no `pgvector`, no `.npy`, no extensions. 500 images × ~9 K floats fit easily in a Python in-memory cache; brute-force cosine on numpy arrays is < 50 ms.
4. **TDD per phase.** Each phase ships with its own test plan. Coverage target ≥ 80 % on backend services; Playwright smoke test for the critical search-pipeline E2E.
5. **Small files, small phases.** Each backend module ≤ 400 lines; each phase ≤ 3 days; each phase produces an end-to-end vertical slice that can be demonstrated.

---

## 2. Tech Stack

| Layer       | Choice                                                | Why                                                                                       |
|-------------|-------------------------------------------------------|-------------------------------------------------------------------------------------------|
| DB          | PostgreSQL 16 (no extensions)                         | Works on every managed PG; JSONB is enough for 500-row corpus                             |
| Backend     | FastAPI (Python 3.12), SQLAlchemy 2.0 (async), Alembic | Async, OpenAPI auto-docs, native WebSocket support for streaming pipeline progress        |
| CV / ML     | opencv-python-headless, scikit-image, numpy, Pillow    | Identical handcrafted features to BASE.md (HSV/CM/LBP/GLCM/HOG/Hu); zero deep learning    |
| Plotting    | matplotlib (server-side PNG)                          | Heavy plotting off the browser; cacheable static PNGs                                     |
| Frontend    | Next.js 15 (App Router, RSC), TypeScript              | Server components for stats; client components for interactive inspector                  |
| UI kit      | Tailwind + shadcn/ui                                  | Accessible primitives, no CSS bikeshedding                                                |
| Charts      | Recharts                                              | Composable React charts for histograms, bars, heatmaps                                    |
| Data        | TanStack Query                                        | Cache, dedupe, background refetch                                                         |
| Streaming   | FastAPI WebSocket → frontend EventSource-like hook    | Live pipeline events: `preprocess.done`, `feature.done:lbp`, `rank.done`                  |
| Testing     | pytest + httpx (BE), Vitest + Playwright (FE)         | Unit + integration + E2E                                                                  |
| Container   | Docker Compose                                        | One-command local: `make up`                                                              |

No `pgvector`. No `.npy`. No FAISS unless / until the corpus exceeds 5 K images.

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         Next.js 15 (App Router)                    │
│  /            /library           /upload         /search ⭐        │
│  Dashboard    Browse/Detail      Drop+Tag        Pipeline Inspector │
│  /compare     /evaluate                                            │
└────────────────┬────────────────────────────────────┬──────────────┘
                 │ REST (JSON)                        │ WebSocket
                 │                                    │ (live query stages)
┌────────────────▼────────────────────────────────────▼──────────────┐
│                       FastAPI (Python 3.12)                        │
│ ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│ │ images     │  │ search     │  │ visualize  │  │ evaluate   │    │
│ │ router     │  │ router+ws  │  │ router     │  │ router     │    │
│ └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    │
│       │               │               │               │            │
│ ┌─────▼─────────────────────────────────────────────────────────┐  │
│ │   services: preprocess · features · search_engine · plot      │  │
│ │              evaluation · feature_cache (in-memory)           │  │
│ └─────┬─────────────────────────────────────┬───────────────────┘  │
└───────┼─────────────────────────────────────┼──────────────────────┘
        │                                     │
   ┌────▼────────┐                       ┌────▼─────────┐
   │ PostgreSQL  │                       │ Filesystem   │
   │ images      │                       │ ./storage/   │
   │ feature_sets│                       │   originals/ │
   │ search_runs │                       │   plots/     │
   │ eval_runs   │                       └──────────────┘
   └─────────────┘
```

---

## 4. Repository Layout

```
csdldptv2/
├── docker-compose.yml
├── Makefile                       # up / down / seed / test / lint
├── README.md
├── PLAN.md                        # this file
├── PLAN.v1.md                     # earlier plan (kept for reference)
├── docs/
│   └── BASE.md                    # original CV/ML spec (unchanged)
│
├── backend/
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── alembic/                   # migrations
│   ├── app/
│   │   ├── main.py                # FastAPI app + lifespan
│   │   ├── config.py              # pydantic-settings
│   │   ├── db.py                  # async engine, session factory
│   │   ├── models.py              # SQLAlchemy ORM
│   │   ├── schemas.py             # Pydantic DTOs
│   │   ├── routers/
│   │   │   ├── images.py
│   │   │   ├── search.py
│   │   │   ├── search_ws.py
│   │   │   ├── visualize.py
│   │   │   ├── evaluate.py
│   │   │   └── stats.py
│   │   ├── services/
│   │   │   ├── preprocess.py      # resize / blur / CLAHE
│   │   │   ├── features/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── hsv.py
│   │   │   │   ├── color_moments.py
│   │   │   │   ├── lbp.py
│   │   │   │   ├── glcm.py
│   │   │   │   ├── hog.py
│   │   │   │   └── hu.py
│   │   │   ├── search_engine.py
│   │   │   ├── feature_cache.py   # in-memory matrix per feature
│   │   │   ├── plot.py            # matplotlib PNG renderers
│   │   │   ├── evaluation.py
│   │   │   └── pipeline_emitter.py # broadcasts WS events
│   │   ├── storage/
│   │   │   └── local.py           # filesystem adapter (S3-swappable)
│   │   └── utils.py
│   ├── scripts/
│   │   ├── seed_afhq.py           # download + ingest AFHQ subset
│   │   └── reset_db.py
│   └── tests/
│       ├── unit/                  # one file per service
│       ├── integration/           # API round-trips
│       └── conftest.py
│
├── frontend/
│   ├── package.json
│   ├── Dockerfile
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── playwright.config.ts
│   ├── public/
│   ├── tests/                     # Playwright e2e
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx           # Dashboard
│       │   ├── library/
│       │   │   ├── page.tsx
│       │   │   └── [id]/page.tsx
│       │   ├── upload/page.tsx
│       │   ├── search/page.tsx    # ⭐ pipeline inspector
│       │   ├── compare/page.tsx
│       │   └── evaluate/page.tsx
│       ├── components/
│       │   ├── pipeline/          # PipelineTimeline, StageCard, FeatureCard
│       │   ├── upload/            # Dropzone, BulkTagger
│       │   ├── visualize/         # HSVChart, LBPImage, GLCMHeatmap, HOGOverlay, HuBars, MomentsBars
│       │   ├── compare/           # ImagePair, FeatureDiffStrip
│       │   ├── evaluate/          # MetricsCards, AblationTable, AblationChart, ConfusionHeatmap
│       │   └── ui/                # shadcn primitives
│       ├── hooks/
│       │   ├── usePipelineStream.ts  # WS hook
│       │   ├── useSearch.ts
│       │   ├── useImages.ts
│       │   └── useEvaluation.ts
│       └── lib/
│           ├── api.ts
│           ├── ws.ts
│           └── format.ts
│
├── storage/                       # gitignored, mounted into containers
│   ├── originals/
│   │   ├── cat/
│   │   └── ...
│   └── plots/                     # cached matplotlib PNGs
│
└── data/                          # dataset staging
    └── afhq_subset/
```

---

## 5. Data Model

```sql
-- Migration 0001 — initial schema

CREATE TABLE images (
    id              BIGSERIAL PRIMARY KEY,
    sha256          CHAR(64)    NOT NULL UNIQUE,         -- dedupe by content
    filename        TEXT        NOT NULL,
    storage_path    TEXT        NOT NULL,                -- relative to STORAGE_ROOT
    animal_type     TEXT        NOT NULL,                -- 'cat' | 'dog' | 'fox' | …
    width           INTEGER     NOT NULL,
    height          INTEGER     NOT NULL,
    size_bytes      INTEGER     NOT NULL,
    role            TEXT        NOT NULL DEFAULT 'corpus'  -- 'corpus' | 'query'
                    CHECK (role IN ('corpus', 'query')),
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_images_animal_type ON images(animal_type);
CREATE INDEX idx_images_role        ON images(role);

CREATE TABLE feature_sets (
    image_id        BIGINT PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    vectors         JSONB       NOT NULL,                -- { hsv:[...], cm:[...], lbp:[...], glcm:[...], hog:[...], hu:[...] }
    dims            JSONB       NOT NULL,                -- { hsv:768, cm:9, lbp:18, glcm:40, hog:8100, hu:7 }
    extracted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    extractor_ver   TEXT        NOT NULL                 -- e.g. 'v1.0' — invalidate cache when feature code changes
);

CREATE TABLE search_runs (
    id              BIGSERIAL PRIMARY KEY,
    query_image_id  BIGINT      REFERENCES images(id) ON DELETE SET NULL,
    weights         JSONB       NOT NULL,
    results         JSONB       NOT NULL,                -- [{rank, image_id, similarity, detail{...}}, …]
    pipeline_trace  JSONB       NOT NULL,                -- ordered list of stage events
    elapsed_ms      INTEGER     NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE evaluation_runs (
    id              BIGSERIAL PRIMARY KEY,
    method          TEXT        NOT NULL,                -- 'hsv_only' | 'combined_equal' | 'combined_tuned' | …
    precision_at_5  NUMERIC(5,4),
    map_at_10       NUMERIC(5,4),
    per_class       JSONB,                                -- { cat: {p5:.., map:..}, dog: {…} }
    ablation        JSONB,                                -- per-feature MAP for the ablation chart
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ
);
```

**Why `JSONB`, not `pgvector`?**

- 500 images × 6 features × ≤ 8 100 floats × 4 B ≈ 78 MB on disk; loaded once into a Python `dict[str, np.ndarray]` matrix at startup.
- Brute-force cosine on a `(N, D)` matrix is `np.dot(M, q)` — < 50 ms for 500 × 8 100.
- pgvector buys ANN search at scale; we are not at scale. Drop the dependency.
- Cache invalidates on any insert/update/delete via a tiny `LISTEN/NOTIFY` channel (or simply on first read after a write event).

---

## 6. Feature Pipeline (unchanged from BASE.md)

| Feature       | Dim    | File                              | Group   |
|---------------|--------|-----------------------------------|---------|
| HSV histogram | 768    | `services/features/hsv.py`        | colour  |
| Color moments | 9      | `services/features/color_moments.py` | colour |
| LBP           | 18     | `services/features/lbp.py`        | texture |
| GLCM          | 40     | `services/features/glcm.py`       | texture |
| HOG           | ~8 100 | `services/features/hog.py`        | shape   |
| Hu moments    | 7      | `services/features/hu.py`         | shape   |

Each module exports a single pure function `extract(img: np.ndarray) -> np.ndarray` returning an L2-normalised 1-D vector. Tests pin shape, dtype, normalisation and determinism.

Default fusion weights (tunable at query time):

```python
WEIGHTS = {
    "hog":           0.25,  # shape — most discriminative for animal faces
    "hsv":           0.20,  # colour — fur tone
    "lbp":           0.15,  # texture — local
    "glcm":          0.15,  # texture — spatial
    "hu":            0.15,  # shape — global, rotation-invariant
    "color_moments": 0.10,  # colour — compact summary
}
```

Cosine similarity reduces to `np.dot(a, b)` because vectors are pre-normalised.

---

## 7. API Surface

### REST (`/api/v1`)

```
# Images
POST   /images                        multipart upload  →  ImageDTO + FeatureSetDTO
POST   /images/batch                  zip upload (kept simple: synchronous for v1)
GET    /images?type=&page=&limit=
GET    /images/{id}
GET    /images/{id}/features          full vectors
DELETE /images/{id}

# Search
POST   /search                        multipart query   →  { run_id, ws_url }
GET    /search/{run_id}               final result document
PATCH  /search/{run_id}/weights       re-rank using cached vectors (no re-extract)

# Visualize  (PNG, cacheable)
GET    /visualize/{image_id}/preprocess          # 4-panel: orig, resized, blurred, CLAHE
GET    /visualize/{image_id}/feature/{name}      # name ∈ {hsv, color_moments, lbp, glcm, hog, hu}
GET    /visualize/compare?a={id}&b={id}&feature={name}

# Evaluate
POST   /evaluate                       run ablation with current corpus + split
GET    /evaluate                       latest results
GET    /evaluate/{id}                  specific run

# Stats
GET    /stats                          dashboard payload
```

### WebSocket

```
WS /ws/search/{run_id}
```

Server-pushed events (`type` is the discriminator):

```json
{"type":"stage.start","stage":"preprocess"}
{"type":"stage.done", "stage":"preprocess",          "data":{"plot_url":"/api/v1/visualize/.../preprocess.png","ms":12}}
{"type":"feature.start","name":"hsv"}
{"type":"feature.done", "name":"hsv","data":{"dim":768,"plot_url":"...","ms":4,"preview":[0.012,0.043,...]}}
... (six feature events)
{"type":"rank.start"}
{"type":"rank.tick", "data":{"processed":120,"total":500}}
{"type":"rank.done", "data":{"results":[ ... ],"elapsed_ms":340}}
{"type":"done"}
```

The frontend's `usePipelineStream(run_id)` hook pushes each event into a Zustand store; the timeline animates as events arrive.

---

## 8. Frontend Pages

### `/` Dashboard
- Stat cards: image count, species count, last 24 h queries, median query latency.
- Species distribution donut.
- Recent queries table (link to each `search_runs` row).
- CTA buttons: Upload, Search, Evaluate.

### `/library` Image gallery
- Server-rendered grid (RSC), 50 per page.
- Filter by species; sort by uploaded_at / filename.
- Hover → quick metadata; click → detail.

### `/library/[id]`
- Big image header with metadata.
- Tabs: **Preprocess · HSV · Color Moments · LBP · GLCM · HOG · Hu** — each renders the visualize PNG and a short paragraph explaining what the feature measures.
- "Use as query" button → `/search?from=:id`.

### `/upload`
- Multi-file dropzone with progress bars.
- After a file uploads: collapsible card showing all 6 feature visualisations.
- Bulk-tag editor: select rows, set animal_type for many at once.

### ⭐ `/search` — Pipeline Inspector

This is the centrepiece. Layout:

```
┌─────────────────────────────────────────────────────────────────┐
│  Drop a query image  ──────  [or pick from library]             │
└─────────────────────────────────────────────────────────────────┘
┌──────────── Pipeline Timeline (animated) ───────────────────────┐
│  ① Original          ░░░░░░░░░░░░░░░░░░░░  [thumbnail]          │
│  ② Preprocess        ▓▓▓▓▓░░░░░░░░░░░░░░░  [orig | CLAHE]       │
│  ③ HSV histogram     ▓▓▓▓▓▓▓▓░░░░░░░░░░░  [chart, dim=768, 4ms] │
│  ④ Color moments     ▓▓▓▓▓▓▓▓▓░░░░░░░░░░  [bars, 9, 1ms]        │
│  ⑤ LBP               ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░  [LBP image, 18, 6ms]  │
│  ⑥ GLCM              ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░  [heatmap, 40, 9ms]    │
│  ⑦ HOG               ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░  [overlay, 8100, 22ms] │
│  ⑧ Hu moments        ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░  [bars, 7, 1ms]        │
│  ⑨ Similarity scan   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░  [42/500 …]            │
│  ⑩ Ranking           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░  [scores chart]        │
└─────────────────────────────────────────────────────────────────┘
┌──────────────── Top 5 results (grid) ──────┐  ┌── Weights ────┐
│ #1  fox_023.jpg   sim 0.892   [card]       │  │ HOG   ▭▭▬─ .25 │
│ #2  fox_011.jpg   sim 0.865   [card]       │  │ HSV   ▭▭▭── .20 │
│ #3  cat_045.jpg   sim 0.634   [card]       │  │ LBP   ▭▭▬── .15 │
│ #4  …                                       │  │ GLCM  ▭▭▬── .15 │
│ #5  …                                       │  │ Hu    ▭▭▬── .15 │
└────────────────────────────────────────────┘  │ CM    ▭▭─── .10 │
                                                │ [Re-rank]       │
                                                └─────────────────┘
┌────── Side-by-side comparison (click any result) ───────────────┐
│  Query     │  Result #1                                         │
│  [image]   │  [image]                                           │
│  HSV bars  │  HSV bars  (overlay diff)                          │
│  HOG over. │  HOG over. (overlay diff)                          │
│  …                                                              │
└─────────────────────────────────────────────────────────────────┘
```

Behaviour:
- Each timeline row is a `StageCard` that reveals its visualisation when WS event arrives.
- Weight sliders call `PATCH /search/{run_id}/weights` — re-rank only, no recompute. Returns new ordering in < 100 ms.
- Click a result → side-by-side comparison panel slides in.

### `/compare`
- Pick two images from library (autocomplete).
- All 6 features rendered side by side, plus a row of per-feature cosine similarities and the weighted total.
- Diff overlays where it makes sense (HSV histograms overlaid; LBP histograms overlaid; HOG visualisations side-by-side).

### `/evaluate`
- Train/test split slider (default 80/20, deterministic via seed).
- "Run evaluation" button → progress bar (also WebSocket-streamed if the run is long).
- Big numbers: Precision@5, MAP@10.
- Ablation table + bar chart (HSV-only, HOG-only, …, combined-equal, combined-tuned).
- Per-class confusion heatmap.

---

## 9. Search Engine Internals

```python
# services/feature_cache.py — module-level singleton
_matrices: dict[str, np.ndarray]      # feature_name -> (N, D) matrix, L2-normalised
_image_ids: list[int]                  # row index → image_id

# services/search_engine.py
async def run_search(query_path: str, weights: dict[str, float], emit) -> SearchRun:
    img = preprocess(query_path)               # await emit("stage.done","preprocess",…)
    qf  = extract_all(img, emit=emit)          # emits feature.start/done per feature
    M   = feature_cache.matrices()             # cached; refreshed on dirty flag

    sims = np.zeros(len(M["hog"]))
    for name, w in weights.items():
        sims += w * (M[name] @ qf[name])       # cosine because pre-normalised
        await emit("rank.tick", {"after": name})

    top_idx = np.argsort(-sims)[:5]
    results = [build_result(idx, sims) for idx in top_idx]
    await emit("rank.done", {"results": results})
    return persist(query_path, weights, results, emit.trace)
```

**Cache freshness**: feature_cache observes a `dirty` flag toggled by image insert/update/delete; the next access reloads from `feature_sets`. For 500 rows the reload is < 200 ms — fine.

**Re-rank without re-extract**: `PATCH /search/{run_id}/weights` reuses the stored per-feature sub-scores from the previous run.

---

## 10. Visualisation Service

All visualisations are matplotlib PNGs rendered server-side and cached on disk under `storage/plots/{image_id}/{feature}.png`. Cache key includes `extractor_ver` so a feature-code change invalidates plots automatically.

| Endpoint                                    | Renders                                                      |
|---------------------------------------------|--------------------------------------------------------------|
| `/visualize/{id}/preprocess`                | 4-panel: original, resized, blurred, CLAHE                  |
| `/visualize/{id}/feature/hsv`               | three overlaid histograms (H, S, V)                          |
| `/visualize/{id}/feature/color_moments`     | bar chart 3 channels × {mean, std, skew}                     |
| `/visualize/{id}/feature/lbp`               | LBP-encoded image + 18-bin histogram                         |
| `/visualize/{id}/feature/glcm`              | heatmap of normalised co-occurrence matrix                  |
| `/visualize/{id}/feature/hog`               | gradient-orientation overlay on grayscale                    |
| `/visualize/{id}/feature/hu`                | log-transformed bar chart of 7 moments                       |
| `/visualize/compare?a=&b=&feature=`         | dual-panel comparison + diff strip                           |

Returned content-type: `image/png`; `Cache-Control: public, max-age=31536000, immutable` (URL contains `extractor_ver`).

---

## 11. Implementation Phases

Each phase ends with a runnable demo and tests. Roughly 2–3 days each, but order matters more than duration.

### Phase 0 — Bootstrap
- `docker-compose.yml` with postgres / backend / frontend.
- FastAPI hello-world; Next.js hello-world; both reachable on `localhost:8000` / `:3000`.
- Pytest + Vitest + Playwright wired up; CI script `make test`.
- **Demo**: `make up && make test` is green.

### Phase 1 — Data layer + ingestion
- Migrations 0001 (schema above).
- SQLAlchemy models, Pydantic DTOs.
- `services/preprocess.py` and the six feature extractors with **unit tests pinning shape, dtype, L2-norm, determinism**.
- `POST /images` endpoint: upload → preprocess → extract → store (image row + feature_set row).
- **Demo**: `curl -F file=@cat.jpg /api/v1/images` returns the full feature set; PG row exists.

### Phase 2 — Search core (REST)
- `services/feature_cache.py`.
- `services/search_engine.py` — synchronous, no streaming yet.
- `POST /search` returns final results; persists `search_runs` row.
- Frontend `/search` MVP: dropzone → call API → show top 5 grid (no timeline yet).
- **Demo**: upload a query image, see five reasonable results.

### Phase 3 — Pipeline inspector (WebSocket)
- `services/pipeline_emitter.py` (broadcaster).
- `WS /ws/search/{id}`.
- Refactor `run_search` to call `await emit(...)` between stages.
- `services/plot.py` — all six feature renderers + preprocess panel.
- Frontend: `usePipelineStream` hook, `PipelineTimeline` + `StageCard` components.
- **Demo**: each query animates through the timeline; every stage's PNG appears as it completes.

### Phase 4 — Comparison + interactive weights
- `PATCH /search/{run_id}/weights` (re-rank only).
- `/compare` page (pick 2 → side-by-side).
- "Compare with query" panel on `/search`.
- **Demo**: drag a weight slider; ranking re-orders in < 100 ms.

### Phase 5 — Evaluation
- `services/evaluation.py` (Precision@K, MAP, ablation runner).
- Train/test split persisted to `images.role`.
- `POST /evaluate`, `GET /evaluate`.
- Frontend `/evaluate` dashboard with ablation table + chart + per-class heatmap.
- **Demo**: click "Run evaluation"; see numbers and ablation chart.

### Phase 6 — Dataset bootstrap
- `scripts/seed_afhq.py`: download AFHQ subset (cat / dog / wild) + Oxford Pet (extra species), normalise to project's animal_type taxonomy, ingest.
- Verify ≥ 500 images, ≥ 15 species, balanced enough for evaluation.
- **Demo**: `make seed` populates DB; `/library` shows hundreds of images.

### Phase 7 — Polish
- Empty / loading / error states everywhere.
- Accessibility pass (alt text, keyboard nav, focus rings, contrast).
- README with screenshots; deployment notes.
- Final E2E (Playwright): upload → search → see all 10 timeline stages → result grid → comparison panel.

**Cumulative estimate**: ~14–16 days solo, considerably less with parallel frontend/backend work.

---

## 12. Testing Strategy

| Layer                      | Tools                | What it pins                                           |
|----------------------------|----------------------|--------------------------------------------------------|
| Feature extractors (unit)  | pytest + numpy.testing | shape, dtype, L2-norm, determinism on fixture image     |
| Preprocess (unit)          | pytest               | output size, dtype, value range                         |
| Search engine (unit)       | pytest               | identity-vector → similarity 1.0; weight sum invariance |
| API (integration)          | pytest + httpx       | upload → search round trip; 4xx error paths             |
| WebSocket events (integ)   | pytest + websockets  | event order, all six features fire, terminal `done`     |
| Visualization (snapshot)   | pytest-mpl or hash   | PNG byte-hash stable on fixture image                   |
| Frontend components (unit) | Vitest + RTL         | StageCard renders states; weight sliders emit changes  |
| E2E (smoke)                | Playwright           | upload → search → see all stages → see top 5            |

Coverage gate: backend services ≥ 80 %.

---

## 13. Risks & Open Decisions

| Risk / question                                                | Mitigation / answer                                                                                  |
|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| HOG vector size in `JSONB` (~32 KB / row × 500 = 16 MB)        | Acceptable; PG handles JSONB up to 1 GB. If it bites, swap to `BYTEA` of float32 little-endian.       |
| WebSocket complexity for v1                                    | Keep REST as primary; WS is an enhancement. Phase 2 ships without it.                                 |
| Cache invalidation when feature code changes                   | `extractor_ver` column; on bump, re-extract all (one-shot script).                                    |
| Auth                                                           | Not in scope for v1. Endpoints are open. Auth middleware skeleton exists but disabled.                |
| 500-image MAP target                                           | Aim for P@5 ≥ 70 %, MAP@10 ≥ 65 % (achievable with handcrafted features per the BASE.md citations).   |
| Frontend bundle size with charts                               | Recharts is tree-shakeable; lazy-load `/evaluate` and `/visualize` chart bundles.                     |
| Search latency under live re-rank                              | Stored sub-scores allow O(N × |features|) re-rank — < 5 ms for 500 images.                            |
| Image dedupe                                                   | `images.sha256 UNIQUE` — re-uploads return the existing record.                                       |

---

## 14. Success Criteria

1. `make up` brings the stack online with `make seed` populating ≥ 500 images, ≥ 15 species.
2. Uploading any animal-face image yields top-5 results with **all 10 pipeline stages visibly rendered** in the timeline and corresponding visualisations.
3. Side-by-side comparison renders for every (query, result) pair.
4. Adjusting weights re-ranks results in < 100 ms.
5. Evaluation page reports Precision@5 ≥ 0.70 and MAP@10 ≥ 0.65 on the 80/20 split.
6. Backend service coverage ≥ 80 %; one Playwright E2E smoke green.
7. No `pgvector`, no `.npy` files, no FAISS — verified by `grep` against the repo.

---

## 15. Out of Scope (v1)

- Authentication / user accounts.
- Multi-tenant separation.
- ANN index (FAISS / HNSW) — revisit at > 5 K images.
- GPU acceleration — none of the features need it.
- Mobile native apps.
- Real-time webcam input.
