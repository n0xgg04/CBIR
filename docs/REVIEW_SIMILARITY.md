# Đánh giá cách tính tương đồng — Animal Face CBIR

**Ngày:** 2026-06-11

---

## 1. Tổng quan pipeline

```
Ảnh query
  → decode BGR
  → resize 128×128 + GaussianBlur(3,3) + CLAHE trên kênh L (LAB)
  → trích 6 đặc trưng handcrafted (từng vector được L2-normalize riêng)
  → ANN HNSW per-feature (top-50 candidate/feature) → union candidates
  → weighted cosine similarity fusion (late fusion)
  → top-K
```

### 6 đặc trưng

| Feature    | Dim   | Nhóm   | Trọng số mặc định |
|-----------|-------|--------|--------------------|
| HOG       | 8100 | Shape  | 0.25               |
| HSV       | 768  | Color  | 0.20               |
| LBP       | 18   | Texture| 0.15               |
| GLCM      | 40   | Texture| 0.15               |
| Hu Moments| 7    | Shape  | 0.15               |
| Color Moments | 9 | Color  | 0.10               |

### Phép đo tương đồng

- Tất cả vector được **L2-normalize riêng** sau khi trích xuất
- Cosine similarity = **dot product** giữa 2 vector đã L2-normalize
- **Late fusion**: weighted sum các cosine similarity per-feature → fused score

---

## 2. Điểm mạnh

### 2.1. L2-normalize từng đặc trưng riêng (⭐⭐⭐⭐⭐)

Đây là quyết định thiết kế **quan trọng nhất** của pipeline. HOG có 8100 chiều, Hu chỉ có 7 chiều. Nếu gộp chung rồi mới normalize, HOG sẽ "lấn át" Hu hoàn toàn. Normalize riêng từng feature đảm bảo mỗi đặc trưng đều có norm = 1 → đóng góp công bằng khi fusion.

### 2.2. Cosine similarity qua dot product (⭐⭐⭐⭐)

Vì vector đã L2-normalize, `cosine(a,b)` = `dot(a,b)`. Cách này vừa chính xác về mặt toán học, vừa hiệu quả tính toán (1 phép nhân ma trận thay vì phải tính norm mỗi lần so sánh).

Pgvector HNSW dùng toán tử `<=>` (cosine distance) và chuyển đổi đúng: `1.0 - (col <=> query) = cosine_similarity`.

### 2.3. Late fusion — kết hợp ở mức similarity score (⭐⭐⭐⭐)

Kết hợp similarity scores (thay vì nối vector rồi tính 1 lần) là lựa chọn đúng cho heterogeneous features:
- Các feature khác nhau về số chiều (7 → 8100)
- Khác nhau về ngữ nghĩa (màu sắc, kết cấu, hình dạng)
- Cho phép xem per-feature sub-scores → **tính giải thích được (explainability)** cao
- Hỗ trợ re-rank với trọng số mới mà không cần trích xuất lại (`PATCH /search/{id}/weights`)

### 2.4. Kiến trúc lưu sub-scores để re-rank (⭐⭐⭐⭐)

```python
# search_engine.py: per_feature_sims được persist vào pipeline_trace
# PATCH /weights chỉ cần re-fuse, không cần extract lại
def rerank_from_persisted(image_ids, per_feature_sims, weights, top_k):
    fused = _fuse(per_feature_sims, weights)
    ...
```

Thiết kế này cho phép người dùng điều chỉnh trọng số và xem kết quả mới ngay lập tức — rất hữu ích cho việc phân tích và demo.

### 2.5. Chuẩn hóa trọng số an toàn (⭐⭐⭐⭐)

```python
def normalise_weights(weights):
    # Drop unknown keys, clamp negatives, renormalise to sum=1
    # Empty/all-zero → fallback DEFAULT_WEIGHTS
```

Không bao giờ trả về zero ranking do lỗi weights. Fallback về default khi input không hợp lệ. Thiết kế phòng thủ tốt.

### 2.6. Xử lý edge case trong GLCM (⭐⭐⭐)

```python
values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
```

Xử lý đúng các ảnh có vùng hằng số (constant patches) gây NaN trong GLCM.

### 2.7. Đa dạng đặc trưng — 3 nhóm, mỗi nhóm 2 feature (⭐⭐⭐⭐)

Color (HSV + CM), Texture (LBP + GLCM), Shape (HOG + Hu) — mỗi nhóm có 2 feature bổ trợ:
- HSV histogram (chi tiết phân bố) + Color Moments (tóm tắt thống kê)
- LBP (cục bộ) + GLCM (quan hệ không gian)
- HOG (chi tiết cạnh) + Hu (tổng quát, bất biến)

---

## 3. Điểm yếu & Đề xuất cải thiện

### 3.1. 🔴 CRITICAL — ANN candidate pruning làm mất kết quả tiềm năng

**Vấn đề:**

```python
# vector_search.py
DEFAULT_CANDIDATE_K: int = 50  # mỗi feature chỉ lấy top-50
```

Pipeline hiện tại:
1. Mỗi feature tự tìm top-50 candidate riêng (HNSW)
2. Union các candidate sets → tập ứng viên
3. Tính fused score trong tập union → rank

**Một ảnh bị loại khỏi top-50 của TẤT CẢ 6 feature sẽ KHÔNG BAO GIỜ xuất hiện trong kết quả**, ngay cả khi fused score của nó lẽ ra nằm trong top-5.

Ví dụ: ảnh X rank #60 ở HOG, #55 ở HSV, #70 ở LBP — không vào được top-50 của bất kỳ feature nào — nhưng weighted fusion có thể đưa nó lên top-3. Ảnh này bị "mất tích" vĩnh viễn.

**Mức độ:** CRITICAL — đây là lỗi cấu trúc, không phải lỗi tham số.

**Đề xuất sửa:**

```python
# Tăng candidate_k lên 200-500
DEFAULT_CANDIDATE_K: int = 200  # an toàn hơn, vẫn nhanh với HNSW

# HOẶC: dùng multi-vector search nếu pgvector hỗ trợ
# HOẶC: kết hợp thêm 1 bước brute-force re-rank trên top candidates sau fusion
```

**Trade-off:** Candidate K lớn hơn → chậm hơn nhưng recall cao hơn. Với HNSW, query 200 thay vì 50 chỉ chậm hơn ~20-30%, nhưng giảm rủi ro mất kết quả đáng kể.

---

### 3.2. 🟠 HIGH — Phân phối cosine similarity không đồng đều giữa các feature

**Vấn đề:**

Các vector L2-normalized ở không gian chiều khác nhau cho phân phối cosine similarity **rất khác nhau**:

- **HOG (8100-D):** Vector trên siêu cầu 8100 chiều → cosine similarity giữa 2 vector ngẫu nhiên tập trung gần 0 (curse of dimensionality). Một cosine = 0.05 đã là CÓ Ý NGHĨA (xa vùng ngẫu nhiên).
- **Hu (7-D):** Vector trên siêu cầu 7 chiều → 2 vector ngẫu nhiên có thể có cosine ±0.3-0.5 dễ dàng. Cosine = 0.5 có thể chỉ là may rủi.

Hiện tại, `_fuse()` cộng trực tiếp các cosine similarity mà không calibration:

```python
def _fuse(per_feature_sims, weights):
    for name, sims in per_feature_sims.items():
        weighted = sims * float(weights.get(name, 0.0))  # raw cosine × weight
        fused = weighted if fused is None else fused + weighted
```

**Hệ quả:** Hu (7-D) có thể đóng góp noise lớn hơn tín hiệu thực. Trọng số 0.15 không giải quyết được vấn đề gốc — đây là vấn đề về **phân phối**, không phải về **độ lớn**.

**Mức độ:** HIGH.

**Đề xuất sửa:**

```python
# Phương án A: Z-score normalize per-feature similarity TRƯỚC khi fusion
# (cần precompute mean/std của cosine trên toàn corpus)
def _fuse_zscore(per_feature_sims, weights, stats):
    fused = np.zeros_like(list(per_feature_sims.values())[0])
    for name, sims in per_feature_sims.items():
        z = (sims - stats[name]['mean']) / (stats[name]['std'] + 1e-8)
        fused += z * weights[name]
    return fused

# Phương án B: Reciprocal Rank Fusion (RRF) — scale-free
# RRF_score = Σ 1/(k + rank_feature) với k=60
```

---

### 3.3. 🟠 HIGH — Color Moments trên kênh Hue bị sai do tính tròn (circular)

**Vấn đề:**

```python
# color_moments.py
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
for ch in range(3):  # H, S, V
    channel = hsv[..., ch].astype(np.float64)
    mean = float(channel.mean())  # SAI với H — H là circular!
```

Kênh **H (Hue)** là đại lượng **tròn** (0° = 360° = 179 trong OpenCV). Giá trị 179 và 0 là cạnh nhau trên vòng tròn màu, nhưng arithmetic mean sẽ cho kết quả ~90 — hoàn toàn sai.

Ví dụ: ảnh toàn màu đỏ (H ≈ 0° và H ≈ 179°) → mean ≈ 90° (cyan) — vô nghĩa.

**Mức độ:** HIGH. Ảnh hưởng trực tiếp đến 3/9 giá trị của Color Moments (mean, std, skew của H).

**Đề xuất sửa:**

```python
# Circular mean cho H
h_rad = channel * (np.pi / 90.0)  # map [0, 180) → [0, 2π)
sin_sum = np.sin(h_rad).mean()
cos_sum = np.cos(h_rad).mean()
mean_h = np.arctan2(sin_sum, cos_sum) * 90.0 / np.pi  # map về [0, 180)
if mean_h < 0:
    mean_h += 180.0

# Circular std cho H
# std = sqrt(-2 * ln(R)), với R = sqrt(sin_sum² + cos_sum²)
R = np.sqrt(sin_sum**2 + cos_sum**2)
R = min(R, 1.0)  # tránh floating point error
std_h = np.sqrt(-2 * np.log(R)) * 90.0 / np.pi

# S và V giữ nguyên (linear)
```

---

### 3.4. 🟡 MEDIUM — Trọng số mặc định không có cơ sở thực nghiệm

**Vấn đề:**

```python
DEFAULT_WEIGHTS = {
    "hog": 0.25, "hsv": 0.20, "lbp": 0.15,
    "glcm": 0.15, "hu": 0.15, "cm": 0.10,
}
```

Các trọng số được gán dựa trên trực giác, không qua tối ưu hóa. Dự án đã có sẵn evaluator (`evaluator.py`) và ablation framework để đo lường chính xác contribution của từng feature nhưng chưa tận dụng để **học trọng số tối ưu**.

**Mức độ:** MEDIUM.

**Đề xuất sửa:**

```python
# Grid search trên validation set
from itertools import product

best_weights = None
best_map = 0.0
for w_hog in [0.15, 0.20, 0.25, 0.30, 0.35]:
    for w_hsv in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for w_lbp in [0.10, 0.15, 0.20]:
            # ... normalize → evaluate → track best
```

Hoặc dùng Bayesian optimization (Optuna) để tìm trọng số tối ưu tự động, dựa trên MAP@10 từ tập query đã tách.

---

### 3.5. 🟡 MEDIUM — Các feature tương quan (correlated) bị "double-count"

**Vấn đề:**

HSV histogram và Color Moments đều hoạt động trên không gian HSV → tương quan. LBP và GLCM đều là texture — cũng tương quan. HOG và Hu đều từ grayscale shape. Các cặp tương quan này được tính trọng số **độc lập**, dẫn đến thông tin bị đếm 2 lần:

- Nhóm Color: HSV (0.20) + CM (0.10) = **0.30** — nhưng 2 feature này overlap đáng kể
- Nhóm Texture: LBP (0.15) + GLCM (0.15) = **0.30**

**Mức độ:** MEDIUM.

**Đề xuất sửa:**

```python
# Điều chỉnh trọng số nhóm thay vì từng feature
# Color:  0.25 (HSV 0.17, CM 0.08)  — giảm CM vì redundant với HSV
# Texture: 0.30 (LBP 0.15, GLCM 0.15)
# Shape:  0.45 (HOG 0.30, Hu 0.15)
```

Hoặc dùng **feature decorrelation** trước khi fusion (whitening transform trên ma trận tương quan giữa các feature).

---

### 3.6. 🟡 MEDIUM — Thiếu khoảng cách thay thế (chỉ có cosine)

**Vấn đề:**

Mọi thứ đều dùng cosine similarity. Trong một số trường hợp, các khoảng cách khác có thể phù hợp hơn:

| Feature | Không gian | Metric phù hợp |
|---------|------------|----------------|
| HSV histogram | Histogram | **Histogram intersection**, **Chi-squared**, **Earth Mover's Distance** |
| LBP histogram | Histogram | **Chi-squared**, **Histogram intersection** |
| HOG | Gradient | **Cosine** (đang dùng, OK) |
| Hu | Shape | **Euclidean** (sau log-transform, cosine cũng tốt) |

**Đặc biệt:** Histogram intersection và Chi-squared thường tốt hơn cosine cho histogram features vì chúng tính đến bản chất phân phối (tổng = 1) của histogram.

**Mức độ:** MEDIUM.

**Đề xuất sửa:**

```python
# Cho HSV và LBP: dùng histogram intersection thay vì cosine
def histogram_intersection(h1, h2):
    """Cả h1, h2 đều là histogram đã L1-normalize (tổng = 1)."""
    return np.sum(np.minimum(h1, h2))

# HOẶC: Chi-squared distance → chuyển thành similarity
# sim = 1.0 / (1.0 + chi2_distance)
```

---

### 3.7. 🟡 MEDIUM — Double L2-normalization trên HOG

**Vấn đề:**

HOG gốc của Dalal-Triggs đã bao gồm block-level L2 normalization bên trong OpenCV. Sau đó code lại L2-normalize toàn bộ vector HOG một lần nữa:

```python
# hog.py
feature = _hog().compute(gray)         # ĐÃ có L2 norm per block bên trong
return l2_normalize(feature.astype(np.float32))  # L2 normalize LẦN 2
```

L2(L2 qua từng block) ≠ L2 toàn cục. Việc normalize lần 2 làm **thay đổi trọng số tương đối giữa các block**. Các block có norm nhỏ (vùng phẳng, ít cạnh) sẽ được "khuếch đại" tương đối so với block có norm lớn.

**Mức độ:** MEDIUM (không hẳn sai, nhưng đáng cân nhắc).

**Đề xuất:** Giữ nguyên nếu có chủ đích (để tránh block có nhiều cạnh dominate). Ghi chú rõ lý do trong code.

---

### 3.8. 🟢 LOW — Thiếu trọng số động dựa trên chất lượng ảnh

**Vấn đề:** Cùng một bộ trọng số được áp dụng cho mọi ảnh query, bất kể đặc điểm riêng. Một ảnh thiếu sáng sẽ có HSV kém tin cậy — lúc đó nên giảm trọng số Color, tăng Shape/Texture.

**Mức độ:** LOW — improvement nâng cao.

**Đề xuất:** Dựa trên thống kê ảnh query (độ sáng trung bình, contrast, noise level) để điều chỉnh trọng số động:
```python
if avg_brightness < 50:  # ảnh tối
    weights["hsv"] *= 0.5
    weights["hog"] *= 1.3
    weights = normalise_weights(weights)
```

---

### 3.9. 🟢 LOW — Hu Moments bị outlier-sensitive

**Vấn đề:**

Hu Moments tính từ `cv2.moments(gray)` — spatial moments của toàn bộ ảnh grayscale. Nếu ảnh có vùng sáng/tối cực đoan (background trắng, foreground tối), Hu moments sẽ bị ảnh hưởng mạnh.

**Đề xuất:** Có thể cân nhắc tính Hu moments trên ảnh đã được threshold (Otsu) hoặc trên edge map (Canny) thay vì ảnh grayscale thô.

---

### 3.10. 🟢 LOW — Thiếu data augmentation trong quá trình indexing

**Vấn đề:** Hiện tại mỗi ảnh corpus được index với 1 bộ feature từ ảnh gốc. Với dataset nhỏ (500 ảnh), có thể augment (xoay nhẹ ±5°, flip ngang) và lưu thêm feature vectors để tăng độ phủ.

**Mức độ:** LOW — thuộc về data strategy hơn là thuật toán similarity.

---

## 4. Đánh giá tổng quan

### Chất lượng kiến trúc: ⭐⭐⭐⭐ (4/5)

| Tiêu chí | Điểm | Ghi chú |
|----------|------|---------|
| Thiết kế pipeline | ⭐⭐⭐⭐⭐ | Late fusion + per-feature L2 norm là chuẩn mực |
| Tính đúng đắn toán học | ⭐⭐⭐⭐ | -1 cho circular mean của Hue |
| Tính giải thích được | ⭐⭐⭐⭐⭐ | Per-feature sub-scores + re-rank rất tốt |
| Hiệu năng (scalability) | ⭐⭐⭐⭐ | HNSW tốt, nhưng candidate pruning có rủi ro |
| Khả năng mở rộng | ⭐⭐⭐⭐ | Thêm feature mới dễ dàng nhờ late fusion |
| Tối ưu hóa tham số | ⭐⭐⭐ | Trọng số chưa được học từ dữ liệu |
| Xử lý edge case | ⭐⭐⭐⭐ | NaN handling, fallback weights tốt |

### Mức độ ưu tiên sửa chữa

| # | Vấn đề | Mức độ | Impact | Effort |
|---|--------|--------|--------|--------|
| 3.1 | ANN candidate pruning | 🔴 CRITICAL | Mất kết quả đúng | Thấp (đổi hằng số) |
| 3.2 | Phân phối cosine không đều | 🟠 HIGH | Noise trong ranking | Trung bình |
| 3.3 | Circular mean của Hue | 🟠 HIGH | Sai 3/9 giá trị CM | Thấp |
| 3.4 | Trọng số chưa tối ưu | 🟡 MEDIUM | MAP có thể cao hơn | Thấp (có sẵn evaluator) |
| 3.5 | Feature tương quan double-count | 🟡 MEDIUM | Thiên vị nhóm feature | Thấp (điều chỉnh weights) |
| 3.6 | Thiếu metric histogram | 🟡 MEDIUM | HSV/LBP discrimination | Trung bình |
| 3.7 | HOG double L2 | 🟡 MEDIUM | Thay đổi relative weight | Thấp |
| 3.8 | Trọng số động theo ảnh | 🟢 LOW | Cải thiện biên | Cao |
| 3.9 | Hu outlier-sensitive | 🟢 LOW | Ảnh hưởng nhỏ | Thấp |

### Kết luận

Pipeline similarity của dự án có **kiến trúc tốt về mặt tổng thể** — late fusion + per-feature L2 normalization + sub-score persistence là những quyết định thiết kế chuẩn mực. Các vấn đề tìm thấy chủ yếu là **tinh chỉnh tham số** và **xử lý edge case**, không phải lỗi cấu trúc.

**3 việc nên làm ngay (effort thấp, impact cao):**
1. Tăng `DEFAULT_CANDIDATE_K` lên 200-300 hoặc thêm cơ chế đảm bảo recall
2. Sửa circular mean cho kênh Hue trong Color Moments
3. Chạy grid search trọng số dùng evaluator có sẵn

---

## 5. Điều tra case thực tế: dog query → top-2 là fox (hu=1.0, cm=0.995)

### 5.1. Hiện tượng

Query: `flickr_dog_000250.jpg` (dog, có trong corpus, id=919)
Top-2: `flickr_wild_000166.jpg` (fox, id=1304) với per-feature scores:

| Feature | Score  | Ghi chú          |
|---------|--------|-------------------|
| hog     | 0.0000 | 🔴 Bị ANN prune  |
| hsv     | 0.7829 | ✅ Trong top-50  |
| lbp     | 0.0000 | 🔴 Bị ANN prune  |
| glcm    | 0.0000 | 🔴 Bị ANN prune  |
| hu      | 1.0000 | ✅ Rank #8       |
| cm      | 0.9951 | ✅ Rank #31      |

### 5.2. Điều tra

#### Bước 1: Trích xuất toàn bộ feature vectors từ DB

Lấy vector gốc của cả 2 ảnh từ PostgreSQL:

```
Hu (dog): [0.1226, 0.3599, 0.4126, 0.4133, -0.4141, -0.4141, 0.4141]
Hu (fox): [0.1213, 0.3586, 0.4136, 0.4134, -0.4142, -0.4142, 0.4142]

→ Cosine similarity: 0.9999978 ≈ 1.0000 ✅ Khớp với kết quả user
```

```
CM (dog): [0.2916, 0.2550, 0.1972, 0.2659, 0.2046, 0.2095, 0.7371, 0.3146, -0.1111]
CM (fox): [0.2974, 0.2931, 0.2127, 0.2674, 0.2169, 0.2280, 0.7081, 0.2738, -0.1828]

→ Cosine similarity: 0.9951 ✅ Khớp
```

```
LBP (dog) vs LBP (fox): Cosine similarity = 0.9497
GLCM (dog) vs GLCM (fox): Cosine similarity = 0.9533
HOG (dog) vs HOG (fox): Cosine similarity = 0.5966
```

**Phát hiện then chốt:** LBP (0.95) và GLCM (0.95) thực tế rất cao nhưng user thấy 0.0000 → **xác nhận lỗi ANN candidate pruning**. Fox không nằm trong top-50 candidate của LBP, HOG, GLCM.

#### Bước 2: Rank thực tế của fox cho từng feature (corpus = 3089 ảnh)

| Feature | Rank của fox | Cosine  | Trong top-50? |
|---------|-------------|---------|---------------|
| **Hu**  | **#8**      | 1.0000  | ✅ CÓ         |
| **HSV** | **#14**     | 0.7829  | ✅ CÓ         |
| **CM**  | **#31**     | 0.9951  | ✅ CÓ         |
| **LBP** | **#1798**   | 0.9497  | ❌ KHÔNG      |
| **HOG** | **#2943**   | 0.5966  | ❌ KHÔNG      |
| **GLCM**| **#2955**   | 0.9533  | ❌ KHÔNG      |

→ 3/6 feature bị prune hoàn toàn vì `candidate_k=50`. Điểm fused chỉ đến từ 3 feature còn lại.

#### Bước 3: Hu moments — HOÀN TOÀN không phân biệt được loài

**404/3089 ảnh (13% corpus) có Hu cosine > 0.999** với query dog này.

Nguyên nhân: Sau preprocessing (CLAHE + resize 128×128 + GaussianBlur), tất cả ảnh mặt động vật đều có global shape gần như giống hệt nhau (hình oval, trung tâm sáng, viền tối). Hu moments — vốn chỉ đo global shape — trở nên vô dụng cho discrimination.

**Hệ quả:** Hu đóng góp flat ~0.15 điểm vào fused score cho 13% corpus, bất kể đó là dog, cat, fox, tiger hay lion. Nó làm **loãng** sự khác biệt thực sự giữa các ảnh.

#### Bước 4: True fused score (nếu có đủ 6 features) — còn tệ hơn!

```
Query: flickr_dog_000250.jpg (dog, id=919)
Corpus: 3089 ảnh — tính toán đầy đủ 6 features

Rank  TrueFused  Animal   HOG    HSV    LBP    GLCM   HU      CM
#1    0.883      TIGER   0.678  0.865  0.956  0.986  1.000  0.991
#2    0.863      CAT     0.636  0.793  0.987  0.993  1.000  0.984
#3    0.841      FOX     0.597  0.783  0.950  0.953  1.000  0.995  ← case user
#4    0.837      DOG     0.658  0.646  0.995  0.993  1.000  0.949  ← dog đầu tiên!
#5    0.836      FOX     0.671  0.636  0.983  0.970  1.000  0.981
#6    0.834      FOX     0.639  0.661  0.974  0.982  1.000  0.979
#7    0.832      FOX     0.639  0.682  0.986  0.956  1.000  0.949
#8    0.829      DOG     0.685  0.556  0.998  0.989  1.000  0.990  ← dog thứ 2
```

> **🔴 SHOCK: Ngay cả với ĐẦY ĐỦ 6 features, dog thật đầu tiên chỉ xếp thứ 4** — sau 1 tiger, 1 cat, và 1 fox!

#### Bước 5: Phân tích từng feature riêng lẻ

**HOG only top-10:** 5 dogs + 5 lions. HOG không phân biệt được dog vs lion vì hình dạng khuôn mặt tương tự (mõm dài, tai cụp). HOG similarity giữa dog query và dog khác ≈ 0.75-0.79 — không đủ cao để tách biệt rõ ràng.

**HSV only top-10:** tiger, cat, fox, dog, tiger, dog, dog, cat, panther, tiger. Màu lông (vàng/nâu/cam) xuất hiện ở hầu hết các loài — không phân biệt được loài.

**LBP/GLCM only top-10:** Hầu như toàn dog và cat (kết cấu lông tương tự). Nhưng score rất khít nhau (0.997-0.999) → ít discrimination margin.

### 5.3. Root cause: 3 vấn đề chồng lớp

```
┌──────────────────────────────────────────────────────────────┐
│  Vấn đề 1: Hu moments HOÀN TOÀN không discrimination         │
│  → 404/3089 ảnh (13%) có Hu cosine > 0.999                  │
│  → Đóng góp flat 0.15 điểm cho 13% corpus                   │
│  → Làm "loãng" sự khác biệt giữa các ảnh                    │
│  → Nguyên nhân gốc: CLAHE + resize → mọi mặt đều là oval    │
├──────────────────────────────────────────────────────────────┤
│  Vấn đề 2: ANN candidate pruning (candidate_k=50)            │
│  → 3/6 feature bị loại cho fox (rank #1798, #2943, #2955)   │
│  → Fused score chỉ dựa trên color + Hu                      │
│  → HOG — feature shape tốt nhất — bị prune mất              │
├──────────────────────────────────────────────────────────────┤
│  Vấn đề 3: Toàn bộ feature set thiếu discrimination power    │
│  → Ngay cả full 6 features, dog thật xếp thứ 4              │
│  → Features "toàn cục" (global) không bắt được CHI TIẾT     │
│    phân biệt loài: khoảng cách mắt, hình dáng mũi, tai...   │
│  → Preprocessing làm ảnh quá giống nhau ở mức global        │
└──────────────────────────────────────────────────────────────┘
```

### 5.4. Đề xuất sửa cụ thể

#### 🔴 Cấp thiết (fix ngay, effort thấp)

**1. Giảm trọng số Hu xuống gần 0**
```python
DEFAULT_WEIGHTS = {
    "hog":  0.35,   # tăng từ 0.25 — shape detail
    "hsv":  0.20,
    "lbp":  0.15,
    "glcm": 0.15,
    "hu":   0.02,   # giảm từ 0.15 → gần như vô dụng cho face
    "cm":   0.13,
}
```
Lý do: Hu đã được chứng minh là không phân biệt được loài nào trên dataset này. 0.15 × 1.0 = flat bonus cho 13% corpus.

**2. Tăng candidate_k lên 200-300**
```python
DEFAULT_CANDIDATE_K: int = 200  # tăng từ 50
```
Lý do: Fox rank #1798 cho LBP. Với candidate_k=200, ít nhất LBP và GLCM (có tính discrimination) sẽ được include. Giảm thiểu "mất feature" trong fusion.

#### 🟠 Nên làm (effort trung bình)

**3. Thêm spatial pyramid features**
Toàn bộ feature hiện tại là global (toàn ảnh). Với ảnh mặt, cần thêm:
- **Spatial pyramid (2×2, 4×4 grid)**: tính HSV, LBP trên từng cell → bắt được spatial layout (tai trái/phải, mắt, mũi ở đâu)
- **Center-weighted features**: vùng trung tâm (mắt-mũi-miệng) quan trọng hơn viền

**4. Cân nhắc thêm local feature matching**
- **SIFT/SURF keypoints + Bag of Visual Words**: bắt được local patterns đặc trưng từng loài (mắt mèo khác mắt chó, tai cáo khác tai hổ)
- Hoặc dùng MobileNetV2 pretrained (ImageNet) để extract embedding 1280-D, kết hợp với handcrafted features qua late fusion

**5. Tối ưu trọng số từ dữ liệu**
```bash
# Chạy ablation study có sẵn để đo contribution thực của từng feature
curl -X POST http://localhost:8000/api/v1/evaluate/ablation
```
Dùng kết quả để xác định feature nào thực sự discrimination (HOG, LBP), feature nào gây nhiễu (Hu).

#### 🟢 Cân nhắc dài hạn

**6. Thay đổi preprocessing để giữ discriminative information**
- CLAHE hiện tại làm ảnh quá "sạch" → mất texture detail
- Có thể giảm CLAHE clip limit (2.0 → 1.0) hoặc bỏ hẳn CLAHE
- Resize lớn hơn (128→256) để giữ nhiều detail hơn

**7. Thêm face alignment**
- Dùng face detection + alignment (dlib, mediapipe) trước khi extract feature
- Đảm bảo mắt-mũi-miệng thẳng hàng → HOG, LBP sẽ bắt được pattern nhất quán hơn

---

## 6. Kết quả đánh giá thực nghiệm sau khi fix

### 6.1. Thay đổi đã áp dụng

| File | Thay đổi | Before | After |
|------|----------|--------|-------|
| `search_engine.py` | DEFAULT_WEIGHTS | Hu=0.15, HOG=0.25 | Hu=**0.02**, HOG=**0.35** |
| `vector_search.py` | DEFAULT_CANDIDATE_K | 50 | **200** |

### 6.2. Kết quả tổng quan (brute-force evaluation, 3090 ảnh)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| **Precision@5** | 0.3318 | **0.4679** | **+41.0%** |
| **MAP@10** | 0.2195 | **0.3593** | **+63.7%** |

### 6.3. Kết quả per-class MAP@10

| Class | Before | After | Delta |
|-------|--------|-------|-------|
| lion | 0.3922 | **0.5739** | +46.3% |
| tiger | 0.3336 | **0.5514** | +65.3% |
| panther | 0.1661 | **0.3412** | **+105.4%** |
| cat | 0.2180 | **0.3390** | +55.5% |
| fox | 0.1174 | **0.2797** | **+138.3%** |
| wolf | 0.1629 | **0.2794** | +71.5% |
| dog | 0.0726 | **0.0984** | +35.5% |

### 6.4. Case study: `flickr_dog_000250.jpg` (dog query)

**Before (old weights + k=50):**
```
Rank 1: dog (self) — score 1.000 ✅
Rank 2: FOX          — score 0.406 ❌ ← case user báo
Rank 3: dog          — ...
```

**After (new weights + k=200):**
```
Rank 1: dog (self)  — score 1.000 ✅
Rank 2: DOG         — score 0.546 ✅
Rank 3: panther     — score 0.499
Rank 4: DOG         — score 0.413 ✅
Rank 5: DOG         — score 0.408 ✅
Rank 6: DOG         — score 0.406 ✅
Rank 7: DOG         — score 0.402 ✅
Rank 8: wolf        — score 0.402
Rank 9: lion        — score 0.401
Rank 10: tiger      — score 0.389
```
→ **6/10 là dog** (trước đây ~3/10). Fox đã biến mất khỏi top-10.

### 6.5. Phân tích

**Thành công:**
- Giảm Hu weight gần như loại bỏ được "nhiễu" từ feature không discrimination
- Tăng HOG weight giúp shape descriptor — feature tốt nhất — được ưu tiên
- Cải thiện đồng đều trên TẤT CẢ các lớp, không chỉ dog

**Hạn chế còn tồn tại:**
- **Dog vẫn là lớp tệ nhất** (MAP=0.098) — chó có quá nhiều giống với ngoại hình rất khác nhau
- **ANN search thực tế vẫn tệ hơn evaluation** này vì candidate pruning (k=200 vẫn bỏ sót)
- **Global features không đủ** để phân biệt loài — cần local features (face landmarks, SIFT)
