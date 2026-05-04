# Animal Face CBIR — Implementation Plan

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Database | PostgreSQL 15 + pgvector | Replaces SQLite; pgvector enables efficient vector similarity search for feature vectors |
| Backend | FastAPI (Python 3.11+) | Async-native, OpenAPI auto-docs, ideal for CPU-bound image processing with background tasks |
| Frontend | Next.js 14 (App Router) | SSR/SSG for dashboards, excellent image handling, React Server Components for stats |
| CV/ML | OpenCV, scikit-image, numpy | Same handcrafted features from BASE.md (no deep learning) |
| Deployment | Docker Compose | Single-command local deployment; services: postgres, backend, frontend |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Next.js Frontend                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Dashboard│ │  Upload  │ │  Search  │ │Visualize │ │ Evaluate │  │
│  │  (/)     │ │ (/upload)│ │(/search) │ │(/viz/:id)│ │(/eval)   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP/JSON
┌──────────────────────────────▼──────────────────────────────────────┐
│                        FastAPI Backend                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Images     │  │   Search     │  │  Evaluation  │              │
│  │   Router     │  │   Router     │  │   Router     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Preprocess   │  │  Features    │  │ Visualization│              │
│  │  Service     │  │  Service     │  │   Service    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└──────────────┬──────────────────────────────┬───────────────────────┘
               │                              │
        ┌──────▼──────┐              ┌────────▼────────┐
        │ PostgreSQL  │              │  File Storage   │
        │  + pgvector │              │  (./storage/)   │
        └─────────────┘              └─────────────────┘
```

---

## Directory Structure

```
animal-face-cbir/
├── docker-compose.yml
├── Makefile
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── alembic/                    # DB migrations
│   ├── src/
│   │   ├── main.py                 # FastAPI app entry
│   │   ├── config.py               # Settings (pydantic-settings)
│   │   ├── database.py             # SQLAlchemy + pgvector setup
│   │   ├── models.py               # SQLAlchemy ORM models
│   │   ├── schemas.py              # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── images.py           # CRUD + upload
│   │   │   ├── search.py           # Similarity search
│   │   │   ├── visualize.py        # Feature visualizations
│   │   │   └── evaluate.py         # Metrics + ablation study
│   │   ├── services/
│   │   │   ├── preprocess.py       # Resize, Gaussian, CLAHE
│   │   │   ├── features.py         # 6 feature extractors
│   │   │   ├── search_engine.py    # Weighted cosine similarity
│   │   │   ├── visualization.py    # Matplotlib chart generation
│   │   │   └── evaluation.py       # Precision@K, MAP
│   │   └── utils.py
│   └── tests/
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx            # Dashboard
│   │   │   ├── layout.tsx          # Root layout
│   │   │   ├── upload/page.tsx     # Upload + preprocessing preview
│   │   │   ├── search/page.tsx     # Search results
│   │   │   ├── visualize/[id]/page.tsx  # Feature detail view
│   │   │   └── evaluate/page.tsx   # Metrics dashboard
│   │   ├── components/
│   │   │   ├── upload/
│   │   │   │   ├── ImageDropzone.tsx
│   │   │   │   ├── PreprocessPreview.tsx
│   │   │   │   └── FeatureGallery.tsx
│   │   │   ├── search/
│   │   │   │   ├── QueryCard.tsx
│   │   │   │   ├── ResultGrid.tsx
│   │   │   │   ├── SimilarityBars.tsx
│   │   │   │   └── FeatureBreakdown.tsx
│   │   │   ├── visualize/
│   │   │   │   ├── HOGVis.tsx
│   │   │   │   ├── LBPVis.tsx
│   │   │   │   ├── HSVHistogram.tsx
│   │   │   │   ├── GLCMHeatmap.tsx
│   │   │   │   └── MomentsChart.tsx
│   │   │   ├── evaluate/
│   │   │   │   ├── MetricsTable.tsx
│   │   │   │   ├── AblationChart.tsx
│   │   │   │   └── ConfusionMatrix.tsx
│   │   │   └── ui/                 # shadcn/ui primitives
│   │   ├── lib/
│   │   │   ├── api.ts              # Typed API client (fetch wrapper)
│   │   │   └── utils.ts
│   │   └── hooks/
│   │       ├── useSearch.ts
│   │       ├── useUpload.ts
│   │       └── useStats.ts
│   └── public/
│
├── storage/
│   ├── images/                     # Original uploaded images
│   │   ├── cat/
│   │   ├── dog/
│   │   └── ...
│   └── features/                   # Cached .npy files (optional)
│
└── data/                           # Dataset seed (500+ images)
    └── raw/
```

---

## Database Schema

### Why pgvector?

The original spec stores feature vectors in `.npy` files. With PostgreSQL + pgvector, we store vectors natively in the database and use vector similarity operators for search. This eliminates file I/O, enables concurrent access, and scales beyond 500 images.

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Images metadata table
CREATE TABLE images (
    id              BIGSERIAL PRIMARY KEY,
    filename        TEXT NOT NULL,
    filepath        TEXT NOT NULL,           -- relative path in storage/
    animal_type     TEXT NOT NULL,           -- e.g., 'cat', 'dog', 'tiger'
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    is_query        BOOLEAN DEFAULT FALSE,   -- for evaluation split
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_images_animal_type ON images(animal_type);
CREATE INDEX idx_images_is_query ON images(is_query);

-- Feature vectors table (one row per feature type per image)
-- Using pgvector's vector type with appropriate dimensions
CREATE TABLE feature_vectors (
    id              BIGSERIAL PRIMARY KEY,
    image_id        BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    feature_type    TEXT NOT NULL CHECK (feature_type IN (
                        'hsv_histogram', 'color_moments', 'lbp',
                        'glcm', 'hog', 'hu_moments'
                    )),
    vector          VECTOR(8192),            -- max dimension (HOG ~8100)
    dimension       INTEGER NOT NULL,        -- actual dimension
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(image_id, feature_type)
);

CREATE INDEX idx_feature_vectors_image ON feature_vectors(image_id);
CREATE INDEX idx_feature_vectors_type ON feature_vectors(feature_type);

-- Search logs (for analytics)
CREATE TABLE search_logs (
    id              BIGSERIAL PRIMARY KEY,
    query_image_id  BIGINT REFERENCES images(id),
    results         JSONB NOT NULL,          -- [{image_id, similarity, detail}]
    elapsed_ms      INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Ablation study results cache
CREATE TABLE evaluation_results (
    id              BIGSERIAL PRIMARY KEY,
    method          TEXT NOT NULL,           -- 'hsv_only', 'hog_only', 'combined', etc.
    precision_at_5  NUMERIC(5,4),
    map_at_10       NUMERIC(5,4),
    details         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### Vector Storage Strategy

| Feature | Dimensions | pgvector type |
|---------|-----------|---------------|
| HSV Histogram | 768 | `vector(768)` |
| Color Moments | 9 | `vector(9)` |
| LBP | 18 | `vector(18)` |
| GLCM | 40 | `vector(40)` |
| HOG | ~8100 | `vector(8192)` padded |
| Hu Moments | 7 | `vector(7)` |

All vectors are L2-normalized before storage (as per BASE.md).

---

## API Design

### Base URL: `/api/v1`

#### Images
```
POST   /images                    # Upload image + auto-extract features
GET    /images                   # List images (paginated, filter by animal_type)
GET    /images/{id}              # Get image metadata
DELETE /images/{id}              # Delete image + features
GET    /images/{id}/features     # Get all feature vectors for image
GET    /images/stats             # DB stats (total, per species, etc.)
```

#### Search
```
POST   /search                   # Upload query image → top-5 similar
Request:
{
  "image": <multipart file>,
  "top_k": 5,
  "weights": {                   # optional override
    "hog": 0.25,
    "hsv_histogram": 0.20,
    ...
  }
}

Response:
{
  "data": {
    "query_id": "temp_abc123",
    "elapsed_ms": 340,
    "results": [
      {
        "rank": 1,
        "image": { "id": 42, "filename": "fox_023.jpg", "animal_type": "fox", ... },
        "similarity": 0.892,
        "detail": {
          "hog": 0.91,
          "hsv_histogram": 0.95,
          "lbp": 0.88,
          "glcm": 0.82,
          "hu_moments": 0.87,
          "color_moments": 0.90
        }
      },
      ...
    ]
  }
}
```

#### Visualization
```
GET    /visualize/{image_id}/preprocess     # Before/after preprocessing
GET    /visualize/{image_id}/hsv            # HSV histogram bar chart
GET    /visualize/{image_id}/lbp            # LBP image + histogram
GET    /visualize/{image_id}/hog            # HOG visualization
GET    /visualize/{image_id}/glcm           # GLCM heatmap
GET    /visualize/{image_id}/moments        # Color moments + Hu moments charts
GET    /visualize/compare?a={id}&b={id}     # Side-by-side feature comparison
```

All visualization endpoints return `image/png`.

#### Evaluation
```
POST   /evaluate/run              # Run full evaluation (Precision@5, MAP@10, ablation)
GET    /evaluate/results          # Get cached results
GET    /evaluate/ablation         # Ablation study table
```

---

## Frontend Page Design

### 1. Dashboard (`/`)
- **Stats Cards**: Total images, species count, avg search time, last upload
- **Species Distribution**: Pie/bar chart of animal types in DB
- **Recent Searches**: Table of latest queries
- **Quick Actions**: Upload button, Search button, Evaluate button

### 2. Upload (`/upload`)
- **Dropzone**: Drag & drop or click to select image
- **Preprocessing Pipeline** (step-by-step visualization):
  1. Original image
  2. Resized 128×128
  3. Gaussian blur
  4. CLAHE result (before/after slider)
- **Feature Extraction Gallery**: Grid of 6 feature visualizations
- **Metadata Form**: Animal type (auto-detected or manual), filename
- **Save Button**: Store to DB

### 3. Search (`/search`)
- **Query Section**: Upload area + query image preview
- **Top 5 Results**: Horizontal scroll/grid of result cards
  - Each card: image, rank, similarity %, species badge
  - Hover: tooltip with per-feature similarity scores
- **Feature Breakdown**: For selected result, show horizontal bar chart of 6 feature similarities
- **Comparison Toggle**: Click to compare query vs result side-by-side

### 4. Visualize (`/visualize/{id}`)
- **Image Header**: Full image + metadata
- **Tabs**: Preprocess | HSV | LBP | GLCM | HOG | Moments
- **Preprocess Tab**: Original vs processed with CLAHE comparison
- **HSV Tab**: 3D histogram or 3 separate channel histograms
- **LBP Tab**: Grayscale LBP image + 18-bin histogram
- **GLCM Tab**: Heatmap of co-occurrence matrix
- **HOG Tab**: Gradient orientation visualization overlaid on image
- **Moments Tab**: Color moments (mean/std/skew per channel) + Hu moments bar chart

### 5. Evaluate (`/evaluate`)
- **Metrics Overview**: Precision@5, MAP@10 big numbers
- **Ablation Study Table**:
  | Method | Precision@5 | MAP@10 |
  |--------|------------|--------|
  | HSV only | xx% | xx% |
  | HOG only | xx% | xx% |
  | ... | ... | ... |
  | Combined (tuned) | xx% | xx% |
- **Bar Chart**: Visual comparison of methods
- **Per-Species Breakdown**: Which species search best/worst
- **Run Evaluation Button**: Recompute with progress indicator

---

## Feature Extraction Pipeline

Identical to BASE.md but wrapped as async FastAPI background tasks:

```python
# services/preprocess.py
def preprocess(image_path: str, target_size=(128, 128)) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img

# services/features.py
def extract_all_features(img: np.ndarray) -> dict[str, np.ndarray]:
    return {
        'hsv_histogram': extract_hsv_histogram(img),    # 768-d
        'color_moments': extract_color_moments(img),    # 9-d
        'lbp': extract_lbp(img),                        # 18-d
        'glcm': extract_glcm(img),                      # 40-d
        'hog': extract_hog(img),                        # ~8100-d
        'hu_moments': extract_hu_moments(img),          # 7-d
    }
```

### Weights (Late Fusion)

```python
WEIGHTS = {
    'hog': 0.25,
    'hsv_histogram': 0.20,
    'lbp': 0.15,
    'glcm': 0.15,
    'hu_moments': 0.15,
    'color_moments': 0.10,
}
```

### Similarity: Cosine (optimized for normalized vectors)

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # vectors already L2-normalized
```

---

## Search Algorithm

```python
# services/search_engine.py
async def search(
    query_image_path: str,
    top_k: int = 5,
    weights: dict[str, float] | None = None
) -> list[SearchResult]:
    # 1. Preprocess + extract features
    img = preprocess(query_image_path)
    query_features = extract_all_features(img)

    # 2. Load all DB features from PostgreSQL
    #    (with 500 images this is fast; for >10K consider FAISS)
    db_features = await load_all_features()

    # 3. Compute weighted similarity
    scores = []
    for img_id, db_feat in db_features.items():
        sim_detail = {
            name: cosine_similarity(query_features[name], db_feat[name])
            for name in WEIGHTS
        }
        final_sim = sum(WEIGHTS[n] * sim_detail[n] for n in WEIGHTS)
        scores.append({
            'id': img_id,
            'similarity': final_sim,
            'detail': sim_detail
        })

    # 4. Sort + return top_k
    scores.sort(key=lambda x: x['similarity'], reverse=True)
    return scores[:top_k]
```

**Performance**: ~340ms for 500 images (brute force). If scaling beyond 5K images, add FAISS IndexFlatIP.

---

## Intermediate Result Visualization

The original requirement (item 4b) demands showing intermediate results. Each visualization is generated by matplotlib on the backend and served as PNG:

| Step | Endpoint | Visual Output |
|------|----------|---------------|
| Preprocessing | `/visualize/{id}/preprocess` | Side-by-side original vs CLAHE |
| HSV | `/visualize/{id}/hsv` | 3 overlaid histograms (H, S, V) |
| Color Moments | `/visualize/{id}/moments` | Bar chart: 3 channels × 3 moments |
| LBP | `/visualize/{id}/lbp` | LBP-encoded image + 18-bin histogram |
| GLCM | `/visualize/{id}/glcm` | Heatmap of normalized GLCM |
| HOG | `/visualize/{id}/hog` | HOG cells overlay on grayscale |
| Hu Moments | `/visualize/{id}/moments` | Log-transformed bar chart |
| Comparison | `/visualize/compare` | Dual-panel with per-feature diff |

---

## Evaluation Metrics

```python
# services/evaluation.py

def precision_at_k(query_type: str, results: list, metadata: dict, k=5) -> float:
    hits = sum(1 for r in results[:k] if metadata[r['id']] == query_type)
    return hits / k

def average_precision(query_type: str, results: list, metadata: dict, depth=10) -> float:
    hits = 0
    sum_prec = 0.0
    for i, r in enumerate(results[:depth]):
        if metadata[r['id']] == query_type:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / hits if hits > 0 else 0.0

def mean_average_precision(queries: list, db_features: dict, metadata: dict) -> float:
    aps = [average_precision(qt, search(qp, db_features), metadata) for qp, qt in queries]
    return np.mean(aps)
```

### Ablation Study

Run search with each feature individually + combined, store results in `evaluation_results` table.

---

## Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: ankane/pgvector:latest
    environment:
      POSTGRES_USER: cbir
      POSTGRES_PASSWORD: cbir_pass
      POSTGRES_DB: animal_cbir
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./backend/alembic:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"

  backend:
    build: ./backend
    environment:
      DATABASE_URL: postgresql+asyncpg://cbir:cbir_pass@postgres:5432/animal_cbir
      STORAGE_PATH: /storage
    volumes:
      - ./storage:/storage
      - ./data:/data
    ports:
      - "8000:8000"
    depends_on:
      - postgres

  frontend:
    build: ./frontend
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000/api/v1
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  pgdata:
```

---

## Implementation Phases

### Phase 1: Foundation (Days 1-2)
- [ ] Initialize backend: FastAPI project structure, SQLAlchemy models, Alembic migrations
- [ ] Initialize frontend: Next.js + Tailwind + shadcn/ui
- [ ] Docker Compose: postgres, backend, frontend communicating
- [ ] Health check endpoints

### Phase 2: Core Pipeline (Days 3-4)
- [ ] Implement `preprocess.py` (resize, Gaussian, CLAHE)
- [ ] Implement all 6 feature extractors in `features.py`
- [ ] Build DB pipeline: `build_db.py` adapted for PostgreSQL
- [ ] Seed database with 500+ images
- [ ] API: `POST /images` (upload + extract + store)

### Phase 3: Search (Days 5-6)
- [ ] Implement cosine similarity search engine
- [ ] API: `POST /search` (query → top-5)
- [ ] Frontend: Upload page with dropzone
- [ ] Frontend: Search results page with similarity bars

### Phase 4: Visualization (Days 7-8)
- [ ] Backend: Matplotlib visualization service (6 feature types)
- [ ] API: All `/visualize/{id}/*` endpoints
- [ ] Frontend: `/visualize/[id]` page with tabs
- [ ] Frontend: Preprocessing comparison slider

### Phase 5: Evaluation (Days 9-10)
- [ ] Implement Precision@K, MAP
- [ ] Ablation study automation
- [ ] API: `/evaluate/*` endpoints
- [ ] Frontend: `/evaluate` dashboard with charts

### Phase 6: Polish (Days 11-12)
- [ ] Dashboard stats page
- [ ] Comparison mode (side-by-side)
- [ ] Responsive design
- [ ] Loading states & error handling
- [ ] README + documentation

---

## Dataset Acquisition Plan

If no dataset exists:
1. **Primary**: Animal Faces-HQ (AFHQ) — 15K images, 3 categories, download via [StarGANv2 repo](https://github.com/clovaai/stargan-v2)
2. **Secondary**: Oxford-IIIT Pet — 37 breeds, ~7.4K images
3. **Tertiary**: Custom crawl with `duckduckgo-images` or similar

**Requirement check**:
- ≥ 500 images ✓
- Front-facing animal faces ✓
- Same size (resize during preprocess to 128×128) ✓
- Same aspect ratio (crop/pad during preprocess) ✓
- ≥ 15 species ✓ (AFHQ has 3, supplement with Oxford or crawl)

---

## Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector storage | pgvector in PostgreSQL | Native similarity ops, no file I/O, transactional |
| Search method | Brute-force cosine | 500 images = <500ms; FAISS only if scaling |
| Feature normalization | L2 per-feature separately | Prevents HOG (8000d) from dominating Hu (7d) |
| Image storage | Filesystem + path in DB | BLOBs in PG are fine but files are easier to serve |
| Visualization | Backend matplotlib → PNG | CPU-heavy plotting offloaded from browser |
| Frontend state | React Query (TanStack) | Caching, deduping, background refetch for stats |

---

## Success Criteria

1. Upload any animal face image → see preprocessing + 6 feature visualizations
2. Search → get top 5 similar images in <1s with per-feature similarity breakdown
3. Evaluation page shows ablation study proving combined features outperform individual
4. Precision@5 > 70% and MAP@10 > 65% on test set
5. System handles 500+ images with PostgreSQL, no SQLite/.npy files in production

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| pgvector not available in managed PG | Use `vector` as `jsonb` fallback, or self-host via Docker |
| HOG vector too large (8100 dims) | pgvector supports up to 16K dims; pad to 8192 |
| 500 images insufficient for good MAP | Augment with flips/rotations, or use AFHQ full set |
| Slow search with brute force | Cache feature matrix in RAM; FAISS if >5K images |
| Frontend build size with charts | Dynamic import `recharts` or `chart.js`; code split |
