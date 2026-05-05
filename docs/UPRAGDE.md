# CBIR — pgvector Refactor: Technical Requirements

> **Scope**: Migrate `feature_sets.vectors` (JSONB) → six typed `vector(D)` columns with HNSW
> indexes, replacing the in-memory RAM cache with database-side ANN search.  
> **Trigger**: RAM usage at 50 K images exceeds 4.5 GB; startup corpus-load time exceeds
> acceptable bounds.  
> **Baseline system**: PostgreSQL 16 + JSONB, process-level NumPy cache, brute-force dot product.  
> **Target system**: PostgreSQL 16 + `pgvector` extension, HNSW per-feature index, optional
> in-memory hot cache for corpora ≤ 5 K images.

---

## 1. Problem Statement

### 1.1 Current architecture

```
PostgreSQL  →  feature_cache.py (startup load)  →  FeatureMatrices (NumPy, RAM)
                                                         ↓
                                               search_engine.py  M @ q  →  top-K
```

`feature_cache.py` loads **every row** of `feature_sets` at startup and holds 6 per-feature
`(N, D)` matrices in process memory.

### 1.2 RAM projection

| Images | HOG (8 100 D) | All 6 features (8 954 D total) | Notes                     |
| ------ | ------------- | ------------------------------ | ------------------------- |
| 500    | 16 MB         | 18 MB                          | current — fine            |
| 10 K   | 324 MB        | 358 MB                         | borderline                |
| 50 K   | 1.6 GB        | 1.8 GB                         | tight on 2 GB containers  |
| 100 K  | 3.2 GB        | **4.5 GB**                     | exceeds typical pod limit |

### 1.3 Startup latency

At 50 K images, `SELECT * FROM feature_sets` transfers ~350 MB of JSON over the local socket and
deserialises it into NumPy arrays. Measured: **8–15 s cold-start** on a 2-core dev machine.

### 1.4 Key constraint to preserve

`search_engine.py` supports **runtime weight adjustment** (`PATCH /search/{id}/weights`) that
re-fuses 6 per-feature cosine scores without re-querying the database. This must survive the
refactor.

---

## 2. Goals and Non-Goals

### Goals

- G1: Reduce RSS memory for the search process to < 200 MB at 100 K images.
- G2: Reduce cold-start time to < 1 s regardless of corpus size.
- G3: Keep end-to-end search latency ≤ 20 ms (p95) at 100 K images.
- G4: Preserve the full API contract — zero breaking changes to REST or WebSocket.
- G5: Preserve dynamic weight re-rank (`PATCH /weights`) with identical semantics.
- G6: Feature extractors (`services/features/`) are untouched.
- G7: Preprocessing pipeline (`services/preprocess.py`) is untouched.
- G8: Zero downtime migration path for existing data.

### Non-Goals

- Switching to a standalone vector database (Qdrant, Milvus, Weaviate).
- Deep-learning embeddings or CLIP vectors.
- GPU-accelerated index build.
- Auth or multi-tenancy.

---

## 3. Chosen Architecture: Multi-Column pgvector (Option B)

Three options were evaluated:

| Option | Description                                                     | Dynamic weights         | Index benefit        | Complexity |
| ------ | --------------------------------------------------------------- | ----------------------- | -------------------- | ---------- |
| A      | Single concatenated `vector(8954)` with baked-in weights        | ❌ Loses dynamic weight | ✅ One HNSW index    | Low        |
| **B**  | Six separate `vector(D)` columns, 6 HNSW indexes, Python fusion | ✅ Preserved            | ✅ Per-feature ANN   | Medium     |
| C      | Keep JSONB, add Redis for in-process cache offload              | ✅ Preserved            | ❌ Still brute-force | High ops   |

**Option B selected.** It is the only option that satisfies G4 + G5 together while gaining ANN
speedup at scale.

### 3.1 Search flow (post-refactor)

```
POST /search
    ↓
preprocess + extract 6 query vectors (unchanged)
    ↓
For each feature f ∈ {hog, hsv, lbp, glcm, hu, cm}:
    SELECT image_id, 1 - (vec_{f} <=> $query_f) AS cos_sim
    FROM feature_sets
    ORDER BY vec_{f} <=> $query_f
    LIMIT candidate_k            ← HNSW ANN, default candidate_k = 50
    ↓
Merge 6 result sets by image_id (union of candidates)
    ↓
Weighted fusion: score = Σ w_f * cos_f   (Python, NumPy)
    ↓
Top-K by fused score
    ↓
Persist search_run (per-feature sub-scores unchanged)
```

`candidate_k` is a tunable parameter. Default 50 ensures the true top-5 is almost certainly
captured even with per-feature ANN approximation (see §7 for recall analysis).

---

## 4. Database Changes

### 4.1 Extension

```sql
-- Migration 0002, step 1
CREATE EXTENSION IF NOT EXISTS vector;
```

Must be executed by a superuser or a role with `CREATE EXTENSION` privilege. Add to
`docker-compose.yml` init SQL or as the first statement in the Alembic migration.

### 4.2 Schema migration (`feature_sets`)

```sql
-- Migration 0002, step 2: add typed vector columns
ALTER TABLE feature_sets
    ADD COLUMN vec_hog  vector(8100),
    ADD COLUMN vec_hsv  vector(768),
    ADD COLUMN vec_lbp  vector(18),
    ADD COLUMN vec_glcm vector(40),
    ADD COLUMN vec_hu   vector(7),
    ADD COLUMN vec_cm   vector(9);
```

Columns are initially nullable. Backfill from JSONB (§4.4), then add NOT NULL constraint.

### 4.3 HNSW indexes

```sql
-- Migration 0002, step 3: build indexes after backfill
CREATE INDEX CONCURRENTLY idx_hnsw_hog
    ON feature_sets USING hnsw (vec_hog vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY idx_hnsw_hsv
    ON feature_sets USING hnsw (vec_hsv vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY idx_hnsw_lbp
    ON feature_sets USING hnsw (vec_lbp vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY idx_hnsw_glcm
    ON feature_sets USING hnsw (vec_glcm vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY idx_hnsw_hu
    ON feature_sets USING hnsw (vec_hu vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY idx_hnsw_cm
    ON feature_sets USING hnsw (vec_cm vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

**Parameter rationale:**

| Parameter         | Value               | Rationale                                                              |
| ----------------- | ------------------- | ---------------------------------------------------------------------- |
| `m`               | 16                  | Standard default; edges per node. Higher → better recall, more memory. |
| `ef_construction` | 64                  | Build-time beam width. 64 is safe for recall ≥ 95 % at this scale.     |
| `ef_search`       | 40                  | Set at query time via `SET hnsw.ef_search = 40`.                       |
| metric            | `vector_cosine_ops` | Vectors are L2-normalised; cosine = inner product here.                |

Note: LBP (18-D), GLCM (40-D), Hu (7-D) and CM (9-D) are low-dimensional — HNSW may fall back
to exact search automatically (pgvector ≥ 0.6 behaviour). This is correct and expected.

### 4.4 Backfill script

```python
# scripts/backfill_pgvector.py
"""
One-shot migration: read feature_sets.vectors (JSONB) → write vec_* columns.
Safe to run multiple times (idempotent: skips rows where vec_hog IS NOT NULL).
"""
import asyncio
from sqlalchemy import text, select
from app.db import async_session
from app.models import FeatureSet

BATCH = 500  # rows per commit

async def backfill() -> None:
    async with async_session() as session:
        result = await session.execute(
            select(FeatureSet).where(FeatureSet.vec_hog.is_(None))
        )
        rows = result.scalars().all()
        for i, row in enumerate(rows):
            for col in ("hog", "hsv", "lbp", "glcm", "hu", "cm"):
                key = "cm" if col == "cm" else col  # internal key alias
                setattr(row, f"vec_{col}", row.vectors[key])
            if i % BATCH == BATCH - 1:
                await session.commit()
        await session.commit()
    print(f"Backfilled {len(rows)} rows.")

asyncio.run(backfill())
```

### 4.5 NOT NULL constraint (after backfill)

```sql
-- Migration 0002, step 4 (run after backfill completes)
ALTER TABLE feature_sets
    ALTER COLUMN vec_hog  SET NOT NULL,
    ALTER COLUMN vec_hsv  SET NOT NULL,
    ALTER COLUMN vec_lbp  SET NOT NULL,
    ALTER COLUMN vec_glcm SET NOT NULL,
    ALTER COLUMN vec_hu   SET NOT NULL,
    ALTER COLUMN vec_cm   SET NOT NULL;
```

### 4.6 JSONB retention policy

`vectors` (JSONB) is **kept** in this release. It serves:

- Re-extraction fallback if a vector column is corrupt.
- The existing `GET /images/{id}/features` endpoint.
- SQLite-backed unit tests (pgvector unavailable in SQLite).

Scheduled for removal in a future migration once `vec_*` columns are proven stable.

---

## 5. Backend Code Changes

### 5.1 Dependencies (`pyproject.toml`)

```toml
[tool.poetry.dependencies]
pgvector = ">=0.3"        # SQLAlchemy integration for vector type
```

### 5.2 `app/models.py` — FeatureSet additions

```python
from pgvector.sqlalchemy import Vector

class FeatureSet(Base):
    __tablename__ = "feature_sets"

    # ... existing columns unchanged ...

    # New typed vector columns (nullable=True until backfill complete)
    vec_hog:  Mapped[list[float] | None] = mapped_column(Vector(8100), nullable=True)
    vec_hsv:  Mapped[list[float] | None] = mapped_column(Vector(768),  nullable=True)
    vec_lbp:  Mapped[list[float] | None] = mapped_column(Vector(18),   nullable=True)
    vec_glcm: Mapped[list[float] | None] = mapped_column(Vector(40),   nullable=True)
    vec_hu:   Mapped[list[float] | None] = mapped_column(Vector(7),    nullable=True)
    vec_cm:   Mapped[list[float] | None] = mapped_column(Vector(9),    nullable=True)

    __table_args__ = (
        # HNSW indexes are created via SQL in the Alembic migration;
        # they are not declared here to avoid DDL conflicts.
    )
```

`Vector(D)` is provided by the `pgvector` package and maps to the `vector(D)` PostgreSQL type.
On SQLite (test environment), the column degrades to `BLOB` and is ignored in search queries
(tests use the brute-force path; see §6.2).

### 5.3 Ingestion — `routers/images.py`

On `POST /images` (and batch), after feature extraction, populate both JSONB and `vec_*`:

```python
# services/ingestion.py (new helper, called from router)
def build_feature_set(image_id: int, vectors: dict[str, np.ndarray], ver: str) -> FeatureSet:
    return FeatureSet(
        image_id=image_id,
        vectors={k: v.tolist() for k, v in vectors.items()},   # JSONB (legacy)
        dims={k: int(v.size) for k, v in vectors.items()},
        extractor_ver=ver,
        vec_hog=vectors["hog"].tolist(),
        vec_hsv=vectors["hsv"].tolist(),
        vec_lbp=vectors["lbp"].tolist(),
        vec_glcm=vectors["glcm"].tolist(),
        vec_hu=vectors["hu"].tolist(),
        vec_cm=vectors["cm"].tolist(),
    )
```

### 5.4 `services/vector_search.py` (new file, replaces hot path in `feature_cache.py`)

```python
"""
Database-backed ANN search via pgvector HNSW.

Returns per-feature candidate sets that search_engine.py fuses using the same
weighted cosine logic as before — API contract is unchanged.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Columns to search and their internal feature key
FEATURE_COLS: dict[str, str] = {
    "hog":  "vec_hog",
    "hsv":  "vec_hsv",
    "lbp":  "vec_lbp",
    "glcm": "vec_glcm",
    "hu":   "vec_hu",
    "cm":   "vec_cm",
}

DEFAULT_CANDIDATE_K: int = 50   # ANN over-fetch factor; tune upward if recall drops
DEFAULT_EF_SEARCH:   int = 40   # HNSW beam width at query time


async def ann_candidates(
    session: AsyncSession,
    query_vectors: dict[str, np.ndarray],
    candidate_k: int = DEFAULT_CANDIDATE_K,
    ef_search: int = DEFAULT_EF_SEARCH,
) -> dict[str, dict[int, float]]:
    """
    Run one HNSW query per feature. Returns:
        {feature_name: {image_id: cosine_similarity}}
    """
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))

    results: dict[str, dict[int, float]] = {}
    for name, col in FEATURE_COLS.items():
        q = query_vectors[name].tolist()
        rows = await session.execute(
            text(f"""
                SELECT image_id,
                       1.0 - ({col} <=> CAST(:q AS vector)) AS cosine_sim
                FROM feature_sets
                ORDER BY {col} <=> CAST(:q AS vector)
                LIMIT :k
            """),
            {"q": str(q), "k": candidate_k},
        )
        results[name] = {row.image_id: float(row.cosine_sim) for row in rows}
    return results


def fuse_candidates(
    per_feature: dict[str, dict[int, float]],
    weights: dict[str, float],
) -> list[tuple[int, float, dict[str, float]]]:
    """
    Weighted fusion over the union of ANN candidate sets.

    Returns list of (image_id, fused_score, per_feature_sims) sorted by
    fused_score descending. Identical contract to search_engine._fuse().
    Missing scores (image not in a feature's candidate set) default to 0.0.
    """
    all_ids: set[int] = set()
    for candidates in per_feature.values():
        all_ids.update(candidates.keys())

    fused: list[tuple[int, float, dict[str, float]]] = []
    for image_id in all_ids:
        per_f: dict[str, float] = {}
        score = 0.0
        for name, candidates in per_feature.items():
            sim = candidates.get(image_id, 0.0)
            per_f[name] = sim
            score += weights.get(name, 0.0) * sim
        fused.append((image_id, score, per_f))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
```

### 5.5 `services/feature_cache.py` — hybrid mode

`feature_cache.py` is **retained but made conditional** on corpus size:

```python
BRUTE_FORCE_THRESHOLD: int = 5_000  # rows; below this, use in-memory cache

async def get_matrices(session: AsyncSession) -> FeatureMatrices | None:
    """
    Returns FeatureMatrices if corpus ≤ BRUTE_FORCE_THRESHOLD, else None.
    Callers that receive None must use vector_search.ann_candidates() instead.
    """
    corpus_size = await _count_corpus(session)
    if corpus_size > BRUTE_FORCE_THRESHOLD:
        return None
    # existing cache logic unchanged
    ...
```

`search_engine.run_search()` dispatches based on the return value:

```python
snapshot = await feature_cache.get_matrices(session)
if snapshot is not None:
    # Existing brute-force path (corpus ≤ 5 K)
    per_feature_sims = _cosine_per_feature(snapshot, query_vectors)
else:
    # pgvector ANN path (corpus > 5 K)
    candidates = await vector_search.ann_candidates(session, query_vectors)
    per_feature_sims = _sims_from_candidates(candidates, query_vectors)
```

This dual-path approach keeps the existing test suite green without modification.

### 5.6 `_sims_from_candidates` adapter

Converts the `{image_id: cosine_sim}` dict format from `vector_search` back into the
`{feature: np.ndarray}` format that `_fuse` and `assemble_results` already expect:

```python
def _sims_from_candidates(
    candidates: dict[str, dict[int, float]],
    query_vectors: dict[str, np.ndarray],
) -> tuple[tuple[int, ...], dict[str, np.ndarray]]:
    """
    Build ordered image_ids and per-feature similarity arrays from ANN candidates.
    Returns (image_ids_tuple, {feature: (N,) float32 array}).
    """
    all_ids = sorted(
        {iid for cands in candidates.values() for iid in cands}
    )
    image_ids = tuple(all_ids)
    idx = {iid: i for i, iid in enumerate(image_ids)}
    N = len(image_ids)

    sims: dict[str, np.ndarray] = {}
    for name, cands in candidates.items():
        arr = np.zeros(N, dtype=np.float32)
        for iid, sim in cands.items():
            arr[idx[iid]] = sim
        sims[name] = arr
    return image_ids, sims
```

`assemble_results` is then called with a lightweight `CandidateSnapshot` namedtuple instead of
`FeatureMatrices` — only `image_ids` is needed at that point.

### 5.7 Re-rank path (`PATCH /search/{id}/weights`)

No changes required. `rerank_from_persisted()` in `search_engine.py` already operates purely on
the per-feature sub-scores stored in `search_runs.pipeline_trace`. It never touches the database
or the cache. G5 is satisfied without modification.

---

## 6. Testing Requirements

### 6.1 New unit tests

| Test file                                 | What it covers                                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `tests/unit/test_vector_search.py`        | `ann_candidates` returns correct shape; `fuse_candidates` weighted sum is correct; missing feature defaults to 0.0 |
| `tests/unit/test_feature_cache_hybrid.py` | Returns `FeatureMatrices` when N ≤ 5 K; returns `None` when N > 5 K                                                |
| `tests/unit/test_sims_adapter.py`         | `_sims_from_candidates` produces arrays with correct values and ordering                                           |

### 6.2 SQLite compatibility

Feature extractors and re-rank tests run against SQLite. `vector_search.py` is **not exercised**
in SQLite tests — tests that require ANN search are marked `@pytest.mark.pg` and skipped when
`DATABASE_URL` is not a PostgreSQL URL.

```python
# conftest.py
import pytest
pg_only = pytest.mark.skipif(
    "sqlite" in str(settings.DATABASE_URL),
    reason="pgvector requires PostgreSQL"
)
```

### 6.3 Integration tests (PostgreSQL)

```
tests/integration/test_pgvector_search.py
```

- Seed 20 fixture images (5 cat, 5 dog, 5 wild, 5 mixed) with known feature vectors.
- Assert `ann_candidates` returns `candidate_k` rows per feature.
- Assert `fuse_candidates` top-1 is the identity image (query = corpus image → score ≈ 1.0).
- Assert end-to-end `POST /search` returns HTTP 200 with 5 results.
- Assert `PATCH /weights` returns re-ordered results within 100 ms.

### 6.4 Recall regression test

```
tests/integration/test_recall_regression.py
```

Using a fixed 500-image corpus with ground truth (same-species = relevant):

- Brute-force path: Precision@5 must be ≥ 0.70 (existing gate).
- pgvector ANN path (triggered by setting `BRUTE_FORCE_THRESHOLD = 0`):
  Precision@5 must be ≥ 0.67 (−3 pp tolerance for approximation).

### 6.5 Existing tests

All tests in `tests/unit/` and `tests/integration/` must remain green with zero modifications.
Coverage gate remains ≥ 80 % on `app/services/`.

---

## 7. Performance Targets

| Metric                  | Current (500 imgs) | Target (100 K imgs)  | Measurement method                            |
| ----------------------- | ------------------ | -------------------- | --------------------------------------------- |
| Cold-start time         | < 0.5 s            | < 1 s                | Time from process start to first `/stats` 200 |
| RSS memory (idle)       | ~100 MB            | < 200 MB             | `docker stats` after startup                  |
| `POST /search` p50      | < 50 ms            | < 20 ms              | `wrk -t4 -c10 -d30s`                          |
| `POST /search` p95      | < 100 ms           | < 50 ms              | same                                          |
| `PATCH /weights` p95    | < 10 ms            | < 10 ms              | unchanged (no DB call)                        |
| Index build time        | —                  | < 5 min (100 K rows) | `CREATE INDEX CONCURRENTLY` wall time         |
| Recall@5 vs brute-force | 100 %              | ≥ 95 %               | Recall regression test (§6.4)                 |

---

## 8. Migration Plan

### 8.1 Prerequisites

- `pgvector` extension available on the PostgreSQL 16 instance (`CREATE EXTENSION vector`).
- `pgvector` Python package added to `pyproject.toml` and locked.
- Docker image updated: `ankane/pgvector:pg16` or official `postgres:16` + manual extension.

### 8.2 Step sequence

```
Step 1  Install pgvector Python package + update docker-compose image
Step 2  Write Alembic migration 0002:
          a. CREATE EXTENSION vector
          b. ALTER TABLE feature_sets ADD COLUMN vec_* vector(D) [nullable]
Step 3  Deploy migration (alembic upgrade head) — zero downtime, columns nullable
Step 4  Run backfill script (scripts/backfill_pgvector.py)
          — reads JSONB, writes vec_* columns in batches of 500
          — idempotent, safe to re-run
Step 5  Verify: SELECT count(*) FROM feature_sets WHERE vec_hog IS NULL → 0
Step 6  Alembic migration 0003:
          a. ALTER COLUMN vec_* SET NOT NULL
          b. CREATE INDEX CONCURRENTLY idx_hnsw_* (6 indexes)
Step 7  Deploy code: vector_search.py, updated models.py, updated feature_cache.py,
          updated search_engine.py
Step 8  Update POST /images ingestion to write vec_* on insert
Step 9  Run integration + recall regression tests
Step 10 Monitor: RSS memory, p95 latency, error rate for 24 h
Step 11 (Optional) Schedule removal of feature_sets.vectors (JSONB) for next release
```

### 8.3 Rollback

Steps 1–7 are additive (no columns removed). Rollback is:

```
1. Revert to previous code deployment (JSONB path still works; vec_* columns ignored)
2. alembic downgrade -1   (drops vec_* columns and indexes)
3. feature_cache.py resumes serving from JSONB
```

Data integrity is preserved because `vectors` (JSONB) is never modified during the migration.

---

## 9. docker-compose Changes

```yaml
# docker-compose.yml — db service
services:
  db:
    image: pgvector/pgvector:pg16   # replaces postgres:16
    environment:
      POSTGRES_DB: cbir
      POSTGRES_USER: cbir
      POSTGRES_PASSWORD: cbir
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/00_extensions.sql

# scripts/init.sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Alternatively, keep `postgres:16` and add to `Dockerfile` or migration:

```sql
-- Alembic migration 0002, step 0
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 10. Open Decisions

| Question                                 | Recommendation                                                                      | Owner   |
| ---------------------------------------- | ----------------------------------------------------------------------------------- | ------- |
| `candidate_k` default value              | Start at 50; tune based on recall regression test results.                          | Backend |
| `ef_search` per-query override           | Expose as optional query param `?ef_search=N` for power users.                      | Backend |
| When to remove JSONB `vectors` column    | After 30 days of stable `vec_*` operation and confirmed recall ≥ G3.                | Backend |
| Low-D feature indexes (LBP 18-D, Hu 7-D) | Benchmark: if HNSW gives no speedup over seq scan, drop those specific indexes.     | Backend |
| pgvector version                         | Pin to `pgvector >= 0.6.0` which introduced HNSW (vs only IVF in earlier versions). | Infra   |

---

## 11. Files Changed Summary

| File                                                | Change type | Notes                                                                          |
| --------------------------------------------------- | ----------- | ------------------------------------------------------------------------------ |
| `docker-compose.yml`                                | Modified    | Use `pgvector/pgvector:pg16` image                                             |
| `pyproject.toml`                                    | Modified    | Add `pgvector >= 0.3` dependency                                               |
| `alembic/versions/0002_pgvector_columns.py`         | New         | ADD COLUMN vec\_\*                                                             |
| `alembic/versions/0003_pgvector_notnull_indexes.py` | New         | NOT NULL + HNSW indexes                                                        |
| `app/models.py`                                     | Modified    | Add `vec_*` mapped columns                                                     |
| `app/services/vector_search.py`                     | New         | ANN query + fusion logic                                                       |
| `app/services/feature_cache.py`                     | Modified    | Hybrid mode, threshold guard                                                   |
| `app/services/search_engine.py`                     | Modified    | Dispatch brute-force vs ANN                                                    |
| `scripts/backfill_pgvector.py`                      | New         | One-shot JSONB → vec\_\* migration                                             |
| `tests/unit/test_vector_search.py`                  | New         | Unit tests for vector_search                                                   |
| `tests/integration/test_pgvector_search.py`         | New         | E2E ANN search tests                                                           |
| `tests/integration/test_recall_regression.py`       | New         | Recall gate                                                                    |
| **Untouched**                                       | —           | `services/features/`, `services/preprocess.py`, all routers, all frontend code |

---

## 12. Acceptance Criteria

- [ ] `make up` starts the stack with `pgvector` extension active.
- [ ] `alembic upgrade head` completes without error on a clean DB.
- [ ] `python scripts/backfill_pgvector.py` completes and all `vec_*` columns are populated.
- [ ] `pytest -m pg` passes all pgvector integration tests.
- [ ] Recall regression: ANN Precision@5 ≥ 0.67 (−3 pp vs brute-force baseline).
- [ ] `docker stats` shows RSS < 200 MB at 100 K images.
- [ ] `PATCH /search/{id}/weights` returns 200 in < 100 ms (no regression).
- [ ] `grep -r "feature_cache" app/routers/` — routers never import `feature_cache` directly.
- [ ] All existing tests remain green (`pytest tests/unit/ tests/integration/`).
- [ ] API contract verified: responses from `POST /search` and `GET /search/{id}` are
      schema-identical before and after refactor (validated by existing integration test suite).
