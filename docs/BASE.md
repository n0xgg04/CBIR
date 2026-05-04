# ANIMAL FACE CBIR — AI AGENT SPECIFICATION

## Mục tiêu

Xây dựng hệ CSDL lưu trữ và tìm kiếm ảnh mặt động vật dựa trên nội dung (Content-Based Image Retrieval). Đầu vào: 1 ảnh mặt động vật. Đầu ra: 5 ảnh giống nhất từ CSDL, xếp theo độ tương đồng giảm dần.

---

## Yêu cầu đề bài

1. Bộ dữ liệu: ≥ 500 ảnh mặt động vật nhìn thẳng, cùng kích thước, cùng tỉ lệ khung hình.
2. Xây dựng bộ thuộc tính (đặc trưng) để nhận diện + phân biệt mặt động vật.
3. Xây dựng CSDL quản trị siêu dữ liệu (metadata) phục vụ tìm kiếm.
4. Hệ thống tìm kiếm: ảnh mới → top 5 ảnh giống nhất (kể cả con vật chưa có trong DB).
5. Demo + đánh giá kết quả.

---

## Kiến trúc tổng quan

```
GIAI ĐOẠN OFFLINE (build DB):

  500+ ảnh gốc
       │
       ▼
  [Tiền xử lý] → Resize 128×128, GaussianBlur, CLAHE
       │
       ├──→ [HSV Histogram]   → vector 768 chiều
       ├──→ [Color Moments]   → vector 9 chiều
       ├──→ [LBP]             → vector 26 chiều
       ├──→ [GLCM]            → vector 20 chiều
       ├──→ [HOG]             → vector ~8000 chiều
       └──→ [Hu Moments]      → vector 7 chiều
                │
                ▼
       [Chuẩn hóa L2 từng đặc trưng riêng]
                │
                ▼
       [Lưu vào SQLite (metadata) + .npy (features)]


GIAI ĐOẠN ONLINE (search):

  Ảnh query
       │
       ▼
  [Cùng pipeline tiền xử lý + trích 6 đặc trưng]
       │
       ▼
  [Tính weighted similarity với từng ảnh trong DB]
       │
       ▼
  [Sắp xếp giảm dần → trả về Top 5]
```

---

## Cấu trúc thư mục dự án

```
animal-face-cbir/
├── data/
│   └── images/                  # 500+ ảnh gốc (.jpg), tổ chức theo thư mục con (tên loài)
│       ├── cat/
│       │   ├── cat_001.jpg
│       │   └── ...
│       ├── dog/
│       ├── tiger/
│       └── ...
├── database/
│   ├── metadata.db              # SQLite database
│   └── features.npy             # Numpy dict: {image_id: {feat_name: vector}}
├── src/
│   ├── preprocess.py            # Tiền xử lý ảnh
│   ├── features.py              # 6 hàm trích xuất đặc trưng
│   ├── build_db.py              # Xây dựng CSDL
│   ├── search.py                # Tìm kiếm top-K
│   ├── evaluate.py              # Đánh giá Precision@K, MAP
│   └── utils.py                 # Tiện ích chung
├── app.py                       # Giao diện Streamlit
├── requirements.txt
└── README.md
```

---

## Dependencies

```
requirements.txt:

opencv-python>=4.5
numpy>=1.21
scikit-image>=0.19
scipy>=1.7
streamlit>=1.20
matplotlib>=3.5
Pillow>=9.0
```

Cài đặt: `pip install -r requirements.txt`

Lưu ý: KHÔNG dùng deep learning, KHÔNG cần PyTorch/TensorFlow. Toàn bộ đặc trưng là handcrafted.

---

## Module 1: Tiền xử lý (`preprocess.py`)

### Hàm `preprocess(image_path, target_size=(128, 128))`

Đầu vào: đường dẫn ảnh gốc (bất kỳ kích thước).
Đầu ra: numpy array BGR shape (128, 128, 3), đã xử lý.

Các bước:

1. Đọc ảnh bằng `cv2.imread(image_path)`.
2. Resize về `target_size` (128×128) bằng `cv2.resize` với `INTER_AREA` (shrink) hoặc `INTER_LINEAR` (enlarge).
3. Lọc nhiễu Gaussian nhẹ: `cv2.GaussianBlur(img, (3, 3), 0)`.
4. Cân bằng sáng CLAHE trên kênh L của không gian LAB:
   - Chuyển BGR → LAB: `cv2.cvtColor(img, cv2.COLOR_BGR2LAB)`
   - Tạo CLAHE: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`
   - Áp dụng CLAHE lên kênh L (channel 0)
   - Chuyển LAB → BGR
5. Return ảnh đã xử lý.

### Tại sao dùng CLAHE mà không dùng histogram equalization thường?

CLAHE (Contrast Limited Adaptive Histogram Equalization) cân bằng sáng CỤC BỘ (từng vùng nhỏ 8×8), không làm mất chi tiết như equalization toàn cục. Giúp ảnh chụp tối hoặc ngược sáng vẫn nhìn rõ đặc trưng mặt.

---

## Module 2: Trích xuất đặc trưng (`features.py`)

### Nguyên tắc thiết kế

- Mỗi đặc trưng là 1 hàm riêng biệt, nhận ảnh đã tiền xử lý (numpy BGR 128×128×3), trả về numpy 1D vector.
- Tất cả vector đều chuẩn hóa L2 sau khi trích xuất.
- 6 đặc trưng chia thành 3 nhóm: Color (2), Texture (2), Shape (2).

---

### Đặc trưng 1: HSV Histogram

**Hàm:** `extract_hsv_histogram(img, h_bins=12, s_bins=8, v_bins=8)`

**Cơ sở lý thuyết:**

- Chuyển BGR → HSV. Không gian HSV tách riêng sắc độ (H), bão hòa (S), giá trị sáng (V).
- H biểu diễn màu sắc thuần túy, không phụ thuộc ánh sáng → bất biến với điều kiện chụp.
- Chia mỗi kênh thành bins, đếm số pixel rơi vào mỗi ô → histogram 3D → flatten.

**Implementation:**

```python
def extract_hsv_histogram(img, h_bins=12, s_bins=8, v_bins=8):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [h_bins, s_bins, v_bins],
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist  # shape: (768,)
```

**Ý nghĩa:** vector 768 chiều (12×8×8), mỗi phần tử = tỷ lệ pixel có màu/bão hòa/sáng tương ứng.

**Vai trò:** Phân biệt động vật theo màu lông (hổ cam vs gấu đen vs thỏ trắng). MAP cao nhất trong nhóm handcrafted theo repo pochih/CBIR (0.614).

---

### Đặc trưng 2: Color Moments

**Hàm:** `extract_color_moments(img)`

**Cơ sở lý thuyết:**

- Phân bố màu có thể tóm tắt bằng 3 moment thống kê cho mỗi kênh:
  - Mean (moment bậc 1): màu trung bình
  - Standard deviation (moment bậc 2): độ biến đổi màu
  - Skewness (moment bậc 3): độ lệch phân bố
- Tính trên 3 kênh HSV → 3 × 3 = 9 giá trị.
- Ưu điểm: cực kỳ compact (9 số), bổ sung cho histogram.

**Implementation:**

```python
def extract_color_moments(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    moments = []
    for i in range(3):  # H, S, V channels
        channel = hsv[:, :, i].astype(np.float64)
        mean = np.mean(channel)
        std = np.std(channel)
        # Skewness: moment bậc 3
        skewness = np.cbrt(np.mean((channel - mean) ** 3))
        moments.extend([mean, std, skewness])
    return np.array(moments, dtype=np.float32)  # shape: (9,)
```

**Vai trò:** Bổ sung cho histogram — histogram cho biết phân bố chi tiết, moments cho biết đặc tính tổng quát (trung bình, đa dạng, lệch). Paper IJASEIT chứng minh Color Moments + GLCM + Hu Moments đưa MAP từ 67% lên 89%.

---

### Đặc trưng 3: LBP (Local Binary Pattern)

**Hàm:** `extract_lbp(img, radius=2, n_points=16)`

**Cơ sở lý thuyết:**

- Tại mỗi pixel, so sánh giá trị với n_points láng giềng trên vòng tròn bán kính radius.
- Nếu láng giềng ≥ trung tâm → 1, ngược lại → 0.
- Được chuỗi nhị phân → chuyển thành số thập phân → mã LBP của pixel đó.
- Dùng phương pháp "uniform": chỉ giữ pattern có ≤ 2 chuyển 0→1 hoặc 1→0 (chiếm >90% pattern tự nhiên).
- Tính histogram các mã LBP → vector đặc trưng kết cấu.

**Implementation:**

```python
from skimage.feature import local_binary_pattern

def extract_lbp(img, radius=2, n_points=16):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = n_points + 2  # uniform LBP có n_points+2 pattern
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist  # shape: (18,)
```

**Tham số:**

- `radius=2`: vòng tròn bán kính 2 pixel, đủ để bắt kết cấu lông mà không quá nhạy với nhiễu.
- `n_points=16`: 16 điểm lấy mẫu trên vòng tròn, cân bằng giữa chi tiết và tốc độ.

**Vai trò:** Phân biệt kết cấu bề mặt — lông mượt (mèo nhà), lông xù (chó poodle), da trơn (ếch), lông có pattern (ngựa vằn). Paper Nature 2025 liệt kê LBP trong bộ handcrafted features hiệu quả nhất.

---

### Đặc trưng 4: GLCM (Gray-Level Co-occurrence Matrix)

**Hàm:** `extract_glcm(img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])`

**Cơ sở lý thuyết:**

- GLCM đếm tần suất cặp pixel (giá trị i, giá trị j) xuất hiện ở khoảng cách d theo hướng θ.
- Từ ma trận GLCM, rút 5 chỉ số Haralick:
  - **Contrast**: độ chênh lệch cường độ giữa pixel và láng giềng. Cao = kết cấu mạnh (sọc rõ).
  - **Dissimilarity**: tương tự contrast nhưng tuyến tính thay vì bình phương.
  - **Homogeneity**: độ đồng nhất cục bộ. Cao = kết cấu mịn.
  - **Energy**: tổng bình phương các phần tử GLCM. Cao = ít pattern (đồng nhất).
  - **Correlation**: tương quan tuyến tính giữa pixel và láng giềng. Cao = pattern có quy luật.
- Tính trên nhiều distances × angles → nối lại.

**Implementation:**

```python
from skimage.feature import graycomatrix, graycoprops

def extract_glcm(img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Quantize về 32 mức xám (giảm noise, tăng tốc)
    gray = (gray / 8).astype(np.uint8)

    glcm = graycomatrix(gray, distances=distances, angles=angles,
                        levels=32, symmetric=True, normed=True)

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = []
    for prop in properties:
        values = graycoprops(glcm, prop)  # shape: (len(distances), len(angles))
        features.extend(values.flatten())

    return np.array(features, dtype=np.float32)  # shape: (40,) = 5 props × 2 dist × 4 angles
```

**Tham số:**

- `distances=[1, 3]`: khoảng cách 1 pixel (chi tiết mịn) và 3 pixel (chi tiết thô).
- `angles`: 4 hướng (0°, 45°, 90°, 135°) bao phủ đủ các hướng kết cấu.
- Quantize 32 mức: giảm ma trận GLCM từ 256×256 xuống 32×32, giảm nhiễu.

**Vai trò:** Bổ sung cho LBP — LBP bắt kết cấu CỤC BỘ (1 pixel vs láng giềng), GLCM bắt QUAN HỆ KHÔNG GIAN giữa các pixel ở khoảng cách xa hơn. Paper Nature 2025 + IJASEIT đều dùng GLCM kết hợp LBP.

**Khác biệt LBP vs GLCM:**

- LBP: "pixel này sáng hơn hay tối hơn xung quanh?" → pattern cục bộ
- GLCM: "pixel sáng có hay nằm cạnh pixel tối không?" → quan hệ không gian

---

### Đặc trưng 5: HOG (Histogram of Oriented Gradients)

**Hàm:** `extract_hog(img)`

**Cơ sở lý thuyết:**

- Tính gradient (Gx, Gy) tại mỗi pixel bằng bộ lọc Sobel.
- Từ Gx, Gy → tính magnitude G = √(Gx² + Gy²) và direction θ = arctan(Gy/Gx).
- Chia ảnh thành cells nhỏ, mỗi cell tính histogram 9 bins (0°-180°, mỗi bin 20°).
- Nhóm cells thành blocks, chuẩn hóa L2 trong mỗi block.
- Nối tất cả blocks → vector HOG.

**Implementation:**

```python
def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_descriptor = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    feature = hog_descriptor.compute(gray)
    return feature.flatten()  # shape: (~8100,)
```

**Tham số:**

- `winSize=(128, 128)`: kích thước ảnh đầu vào.
- `blockSize=(16, 16)`: mỗi block gồm 2×2 cells.
- `blockStride=(8, 8)`: block trượt 8 pixel (chồng 50%).
- `cellSize=(8, 8)`: mỗi cell 8×8 pixel.
- `nbins=9`: 9 hướng gradient (0°-180°).

**Vai trò:** Đặc trưng hình dạng QUAN TRỌNG NHẤT cho mặt động vật. Bắt được cấu trúc tai (nhọn vs cụp), mõm (dài vs ngắn), mắt (tròn to vs nhỏ). MAP = 0.450 trong repo pochih/CBIR, cao nhất nhóm shape.

---

### Đặc trưng 6: Hu Moments

**Hàm:** `extract_hu_moments(img)`

**Cơ sở lý thuyết:**

- 7 moment bất biến do Ming-Kuei Hu đề xuất năm 1962.
- Bất biến với: translation (dịch), scale (co giãn), rotation (xoay).
- Tính từ central moments và normalized moments của ảnh nhị phân hoặc xám.
- Thường lấy log transform để giảm dynamic range: sign(h) × log10(|h|).

**Implementation:**

```python
def extract_hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()  # 7 giá trị
    # Log transform để giảm dynamic range
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.astype(np.float32)  # shape: (7,)
```

**Vai trò:** Bổ sung cho HOG — HOG mô tả hình dạng CHI TIẾT cục bộ (8000 số), Hu mô tả hình dạng TỔNG QUÁT toàn cục (7 số). Hu bất biến xoay/scale nên ảnh to/nhỏ/nghiêng nhẹ đều cho vector gần giống. Paper IJASEIT: kết hợp Hu + GLCM + Color Moments → MAP 89.71%.

---

### Hàm tổng hợp: `extract_all_features(img)`

```python
def extract_all_features(img):
    """
    Trích xuất 6 đặc trưng từ ảnh đã tiền xử lý.
    Return: dict {feature_name: normalized_vector}
    """
    features = {
        'hsv_histogram': extract_hsv_histogram(img),
        'color_moments': extract_color_moments(img),
        'lbp': extract_lbp(img),
        'glcm': extract_glcm(img),
        'hog': extract_hog(img),
        'hu_moments': extract_hu_moments(img),
    }

    # Chuẩn hóa L2 từng đặc trưng riêng
    for key in features:
        norm = np.linalg.norm(features[key])
        if norm > 0:
            features[key] = features[key] / norm

    return features
```

**Tại sao chuẩn hóa L2 RIÊNG từng đặc trưng?**

- HOG có ~8000 chiều, Hu chỉ có 7 chiều.
- Nếu nối thẳng rồi chuẩn hóa → HOG "lấn át" Hu hoàn toàn.
- Chuẩn hóa riêng → mỗi đặc trưng đều có norm = 1 → công bằng khi so sánh.

---

## Module 3: Xây dựng CSDL (`build_db.py`)

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    filepath        TEXT NOT NULL,
    animal_type     TEXT NOT NULL,       -- tên loài: "cat", "dog", "tiger"
    width           INTEGER,
    height          INTEGER,
    file_size_bytes INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_animal_type ON images(animal_type);
```

**Metadata (siêu dữ liệu)** gồm: tên file, đường dẫn, loài động vật, kích thước ảnh, dung lượng, thời gian tạo. Đây là thông tin MÔ TẢ ảnh, không phải nội dung ảnh.

### Features Storage

Lưu features dưới dạng numpy dict vào file `.npy`:

```python
# Structure:
# {
#     image_id_1: {
#         'hsv_histogram': np.array([...]),   # shape (768,)
#         'color_moments': np.array([...]),   # shape (9,)
#         'lbp':           np.array([...]),   # shape (18,)
#         'glcm':          np.array([...]),   # shape (40,)
#         'hog':           np.array([...]),   # shape (~8100,)
#         'hu_moments':    np.array([...]),   # shape (7,)
#     },
#     image_id_2: { ... },
#     ...
# }
```

**Tại sao không lưu features trong SQLite?**

- Vector HOG ~8000 float32 = ~32KB mỗi ảnh. SQLite xử lý BLOB lớn kém hiệu quả.
- Numpy `.npy` load toàn bộ vào RAM nhanh, tính toán vector trực tiếp.
- SQLite chỉ lưu metadata nhẹ, `.npy` lưu features nặng → tách biệt hợp lý.

### Build pipeline

```python
def build_database(image_folder, db_path, features_path):
    """
    Quét toàn bộ ảnh trong image_folder, trích đặc trưng, lưu vào DB.

    image_folder structure:
        images/
        ├── cat/
        │   ├── cat_001.jpg
        │   └── ...
        ├── dog/
        └── ...

    animal_type = tên thư mục con (cat, dog, tiger, ...)
    """
    conn = sqlite3.connect(db_path)
    all_features = {}

    for animal_dir in sorted(os.listdir(image_folder)):
        animal_path = os.path.join(image_folder, animal_dir)
        if not os.path.isdir(animal_path):
            continue

        animal_type = animal_dir  # "cat", "dog", ...

        for filename in sorted(os.listdir(animal_path)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            filepath = os.path.join(animal_path, filename)
            img_raw = cv2.imread(filepath)
            h, w = img_raw.shape[:2]
            file_size = os.path.getsize(filepath)

            # Insert metadata
            cursor = conn.execute(
                'INSERT INTO images (filename, filepath, animal_type, width, height, file_size_bytes) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (filename, filepath, animal_type, w, h, file_size)
            )
            img_id = cursor.lastrowid

            # Preprocess + extract features
            img = preprocess(filepath)
            features = extract_all_features(img)
            all_features[img_id] = features

            print(f"[{img_id}] {filepath} → {animal_type}")

    conn.commit()
    conn.close()
    np.save(features_path, all_features, allow_pickle=True)
```

---

## Module 4: Tìm kiếm (`search.py`)

### Trọng số kết hợp đặc trưng (Late Fusion)

```python
WEIGHTS = {
    'hog':            0.25,   # Shape — quan trọng nhất cho mặt động vật
    'hsv_histogram':  0.20,   # Color — phân biệt màu lông hiệu quả
    'lbp':            0.15,   # Texture — kết cấu cục bộ
    'glcm':           0.15,   # Texture — quan hệ không gian
    'hu_moments':     0.15,   # Shape — bất biến xoay/scale
    'color_moments':  0.10,   # Color — bổ sung compact
}
# Tổng = 1.0
# Shape: 40%, Color: 30%, Texture: 30%
```

**Lý do trọng số:**

- HOG (0.25): Mặt động vật khác nhau NHIỀU NHẤT ở hình dạng tai, mõm, mắt. HOG nắm bắt tốt nhất.
- HSV (0.20): Màu lông là đặc trưng trực quan nhất (hổ cam, gấu đen). MAP cao nhất nhóm handcrafted.
- LBP + GLCM (0.15 mỗi cái): Kết cấu lông (mượt/xù/sọc/đốm) giúp phân biệt trong cùng loài.
- Hu (0.15): Hình dạng tổng quát, bất biến. Hữu ích khi ảnh query khác góc nhẹ.
- Color Moments (0.10): Compact, bổ sung histogram. Thấp nhất vì histogram đã bao phủ tốt.

### Hàm đo similarity

```python
def cosine_similarity(a, b):
    """Cosine similarity giữa 2 vector đã chuẩn hóa L2."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return dot / norm
```

**Tại sao dùng Cosine mà không dùng Euclidean?**

- Cosine đo GÓC giữa 2 vector → không phụ thuộc magnitude → phù hợp khi vector đã chuẩn hóa.
- Euclidean bị ảnh hưởng bởi chiều dài vector → vector dài (HOG) sẽ dominate.

### Hàm tìm kiếm

```python
def search(query_image_path, db_features, top_k=5):
    """
    Tìm top_k ảnh giống nhất với query.

    Args:
        query_image_path: đường dẫn ảnh query
        db_features: dict {image_id: {feat_name: vector}}
        top_k: số kết quả trả về

    Returns:
        list of dict: [{
            'id': int,
            'similarity': float,         # weighted similarity tổng hợp
            'detail': {feat_name: float}  # similarity từng đặc trưng
        }, ...]
    """
    # 1. Trích đặc trưng ảnh query
    img = preprocess(query_image_path)
    query_features = extract_all_features(img)

    # 2. Tính similarity với từng ảnh trong DB
    scores = []
    for img_id, db_feat in db_features.items():
        # Tính similarity RIÊNG từng đặc trưng
        sim_detail = {}
        for feat_name in WEIGHTS:
            sim_detail[feat_name] = cosine_similarity(
                query_features[feat_name],
                db_feat[feat_name]
            )

        # Kết hợp trọng số
        final_sim = sum(
            WEIGHTS[feat_name] * sim_detail[feat_name]
            for feat_name in WEIGHTS
        )

        scores.append({
            'id': img_id,
            'similarity': final_sim,
            'detail': sim_detail
        })

    # 3. Sắp xếp giảm dần
    scores.sort(key=lambda x: x['similarity'], reverse=True)
    return scores[:top_k]
```

### Hiển thị kết quả trung gian

Khi trả về kết quả, CẦN hiển thị cả similarity chi tiết từng đặc trưng (trong `detail`), không chỉ score tổng. Đề bài yêu cầu "trình bày các kết quả trung gian trong quá trình tìm kiếm".

Ví dụ output:

```
Query: fox_new.jpg

#1: fox_023.jpg (similarity: 0.892)
    HOG: 0.91 | HSV: 0.95 | LBP: 0.88 | GLCM: 0.82 | Hu: 0.87 | CM: 0.90

#2: fox_011.jpg (similarity: 0.865)
    HOG: 0.88 | HSV: 0.92 | LBP: 0.85 | GLCM: 0.79 | Hu: 0.84 | CM: 0.88

#3: cat_045.jpg (similarity: 0.634)
    HOG: 0.72 | HSV: 0.45 | LBP: 0.65 | GLCM: 0.58 | Hu: 0.70 | CM: 0.52
```

---

## Module 5: Đánh giá (`evaluate.py`)

### Precision@K

```python
def precision_at_k(query_animal_type, results, db_metadata, k=5):
    """
    Precision@K = số ảnh đúng loài trong top K / K

    Args:
        query_animal_type: loài của ảnh query (ground truth)
        results: list of search results (từ search())
        db_metadata: dict {image_id: animal_type}
        k: top K
    """
    hits = 0
    for result in results[:k]:
        if db_metadata[result['id']] == query_animal_type:
            hits += 1
    return hits / k
```

### Mean Average Precision (MAP)

```python
def average_precision(query_animal_type, results, db_metadata, depth=10):
    """AP cho 1 query."""
    hits = 0
    sum_precision = 0.0
    for i, result in enumerate(results[:depth]):
        if db_metadata[result['id']] == query_animal_type:
            hits += 1
            sum_precision += hits / (i + 1)
    if hits == 0:
        return 0.0
    return sum_precision / hits

def mean_average_precision(all_queries, db_features, db_metadata, depth=10):
    """MAP trên toàn bộ tập query."""
    aps = []
    for query_path, query_animal_type in all_queries:
        results = search(query_path, db_features, top_k=depth)
        ap = average_precision(query_animal_type, results, db_metadata, depth)
        aps.append(ap)
    return np.mean(aps)
```

### So sánh đặc trưng đơn lẻ vs kết hợp

Cần thực hiện ablation study — chạy tìm kiếm với:

1. Chỉ HSV → tính MAP
2. Chỉ HOG → tính MAP
3. Chỉ LBP → tính MAP
4. ... (từng đặc trưng)
5. Kết hợp 6 đặc trưng → tính MAP
6. (Bonus) Kết hợp + trọng số tối ưu → tính MAP

Xuất kết quả dạng bảng:

```
| Phương pháp              | Precision@5 | MAP@10 |
|--------------------------|-------------|--------|
| HSV Histogram only       | xx%         | xx%    |
| Color Moments only       | xx%         | xx%    |
| LBP only                 | xx%         | xx%    |
| GLCM only                | xx%         | xx%    |
| HOG only                 | xx%         | xx%    |
| Hu Moments only          | xx%         | xx%    |
| Combined (equal weights) | xx%         | xx%    |
| Combined (tuned weights) | xx%         | xx%    |
```

---

## Module 6: Giao diện Demo (`app.py`)

Dùng **Streamlit** — tạo web app đơn giản.

### Layout

```
┌──────────────────────────────────────────────────┐
│  🐾 Hệ thống tìm kiếm ảnh mặt động vật          │
├──────────────────────────────────────────────────┤
│                                                  │
│  [Upload ảnh]  hoặc  [Chọn ảnh mẫu ▼]           │
│                                                  │
│  ┌──────────┐                                    │
│  │  Query   │  Loài dự đoán: Fox                 │
│  │  Image   │  Thời gian xử lý: 0.3s            │
│  └──────────┘                                    │
│                                                  │
│  ═══ Top 5 kết quả ═══                           │
│                                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │ #1  │ │ #2  │ │ #3  │ │ #4  │ │ #5  │       │
│  │ 89% │ │ 86% │ │ 78% │ │ 72% │ │ 65% │       │
│  │ fox │ │ fox │ │ cat │ │ fox │ │ dog │       │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
│                                                  │
│  ═══ Chi tiết đặc trưng (ảnh #1) ═══            │
│  HOG: 0.91 ████████████████████░░                │
│  HSV: 0.95 ████████████████████░                 │
│  LBP: 0.88 ██████████████████░░░                 │
│  GLCM: 0.82 █████████████████░░░░                │
│  Hu:  0.87 ██████████████████░░░                 │
│  CM:  0.90 ████████████████████░                 │
│                                                  │
│  ═══ Thống kê CSDL ═══                           │
│  Tổng ảnh: 500 | Số loài: 15 | ...              │
└──────────────────────────────────────────────────┘
```

### Implementation sketch

```python
import streamlit as st

st.title("🐾 Hệ thống tìm kiếm ảnh mặt động vật")

# Load DB
db_features = np.load('database/features.npy', allow_pickle=True).item()
conn = sqlite3.connect('database/metadata.db')

# Upload
uploaded = st.file_uploader("Upload ảnh mặt động vật", type=['jpg', 'png', 'jpeg'])

if uploaded:
    # Save temp
    temp_path = f"/tmp/{uploaded.name}"
    with open(temp_path, 'wb') as f:
        f.write(uploaded.read())

    # Display query
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(temp_path, caption="Ảnh đầu vào", width=200)

    # Search
    import time
    start = time.time()
    results = search(temp_path, db_features, top_k=5)
    elapsed = time.time() - start

    with col2:
        st.write(f"Thời gian xử lý: {elapsed:.3f}s")

    # Display results
    st.subheader("Top 5 kết quả giống nhất")
    cols = st.columns(5)
    for i, result in enumerate(results):
        img_info = conn.execute(
            'SELECT filepath, animal_type FROM images WHERE id=?',
            (result['id'],)
        ).fetchone()
        with cols[i]:
            st.image(img_info[0], caption=f"#{i+1} | {img_info[1]}")
            st.write(f"Similarity: {result['similarity']:.2f}")

    # Detail
    st.subheader("Chi tiết đặc trưng (ảnh #1)")
    detail = results[0]['detail']
    for feat_name, sim_value in sorted(detail.items(), key=lambda x: -x[1]):
        st.progress(sim_value, text=f"{feat_name}: {sim_value:.3f}")
```

---

## Lưu ý quan trọng khi implement

### 1. Dataset

Nếu không có sẵn 500 ảnh, có thể dùng:

- **Animal Faces HQ (AFHQ)**: ~15,000 ảnh, 3 loại (cat, dog, wildlife), 512×512.
  Download: search "AFHQ dataset" hoặc dùng subset.
- **Oxford-IIIT Pet**: 37 loại chó/mèo, ~7400 ảnh.
- Hoặc tự crawl từ Google Images + crop mặt.

Cần ít nhất 15 loài, mỗi loài ≥ 30 ảnh. Tổng ≥ 500.

### 2. Tách train/test cho đánh giá

- Giữ lại ~20% ảnh mỗi loài làm query (không nằm trong DB).
- DB chứa 80% còn lại.
- Dùng query set để tính Precision@5 và MAP@10.

### 3. Kết quả trung gian cần trình bày

Đề bài yêu cầu "trình bày các kết quả trung gian". Cần show:

- Ảnh sau tiền xử lý (trước/sau CLAHE).
- Visualization HOG (dùng `skimage.feature.hog` với `visualize=True`).
- LBP image (ảnh LBP trước khi tính histogram).
- GLCM matrix heatmap.
- Histogram HSV (bar chart).
- Biểu đồ so sánh similarity từng đặc trưng.

### 4. Xử lý ảnh chưa có trong DB

Đề bài nói "ảnh của con vật đã có và CHƯA CÓ trong dữ liệu". Hệ thống tìm kiếm similarity-based tự động xử lý: nếu con vật chưa có, nó vẫn trả về 5 ảnh "gần giống nhất" (có thể là loài khác nhưng có đặc điểm tương đồng). KHÔNG cần logic đặc biệt cho trường hợp này.

### 5. Performance

- 500 ảnh × 6 đặc trưng: build DB mất ~2-5 phút (không cần GPU).
- Search 1 query trong 500 ảnh: < 1 giây.
- Nếu cần nhanh hơn: dùng FAISS IndexFlatIP thay vì brute-force loop.
