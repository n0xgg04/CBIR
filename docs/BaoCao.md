# BÁO CÁO BÀI TẬP LỚN

## HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN

**Đề tài:** Xây dựng hệ cơ sở dữ liệu lưu trữ và tìm kiếm ảnh mặt động vật

---

## MỤC LỤC

1. [Câu 1: Đặc điểm của kho ảnh](#câu-1-đặc-điểm-củakho-ảnh)
2. [Câu 2: Kỹ thuật xử lý và tìm kiếm ảnh mặt động vật](#câu-2-kỹ-thuật-xử-lý-và-tìm-kiếm-ảnh-mặt-động-vật)
   1. [Quá trình xử lý ảnh](#quá-trình-xử-lý-ảnh)
   2. [Đặc trưng của ảnh và biểu diễn ảnh](#đặc-trưng-củảảnh-và-biểu-diễn-ảnh)
   3. [Các phương pháp xử lý ảnh hiện nay](#các-phương-pháp-xử-lý-ảnh-hiện-nay)
   4. [Kỹ thuật tìm kiếm ảnh theo nội dung (CBIR)](#kỹ-thuật-tìm-kiếm-ảnh-theo-nội-dung-cbir)
   5. [Các phương pháp rút trích đặc trưng ảnh](#các-phương-pháp-rút-trích-đặc-trưng-ảnh)
3. [Câu 3: Xây dựng hệ thống (phần lý thuyết)](#câu-3-xây-dựng-hệ-thống)
   1. [Sơ đồ khối hệ thống](#sơ-đồ-khối-hệ-thống)
   2. [Quy trình thực hiện](#quy-trình-thực-hiện)
   3. [Kết quả trung gian trong quá trình tìm kiếm](#kết-quả-trung-gian-trong-quá-trình-tìm-kiếm)
4. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Câu 1: Đặc điểm của kho ảnh

### 1.1. Nguồn gốc và quy mô dữ liệu

Hệ thống sử dụng bộ dữ liệu **AFHQ (Animal Faces-HQ)** và **Oxford-IIIT Pet Dataset** — các bộ dữ liệu công khai chuyên dụng cho nghiên cứu xử lý ảnh mặt động vật. Bộ dữ liệu bao gồm:

- **Tổng số ảnh:** ít nhất 500 files ảnh
- **Số loài động vật:** ít nhất 15 loài khác nhau (mèo, chó, cáo, sói, gấu, v.v.)
- **Phân loại:** Ảnh được chia thành các thư mục theo loài động vật

### 1.2. Đặc điểm hình ảnh

| Thuộc tính | Giá trị / Mô tả |
|-----------|-----------------|
| Kích thước | Chuẩn hóa về 128×128 pixel |
| Tỉ lệ khung hình | 1:1 (vuông), mặt động vật nằm ở trung tâm |
| Góc chụp | Nhìn thẳng (frontal view) |
| Số đối tượng | Mỗi ảnh chỉ có 1 mặt động vật duy nhất |
| Định dạng | JPG / PNG |
| Đa dạng | Mỗi loài có nhiều tư thế, màu lông, góc nghiêng khác nhau |

### 1.3. Tại sao chọn kích thước 128×128?

- **Đủ lớn để giữ chi tiết:** Khuôn mặt động vật có đặc trưng hình dạng phức tạp (mũi, tai, râu) cần độ phân giải đủ cao để HOG và các đặc trưng hình dạng hoạt động hiệu quả.
- **Đủ nhỏ để tính toán nhanh:** 128×128 = 16.384 pixel, gấp 4 lần 64×64, nhưng vẫn cho phép trích xuất đặc trưng HOG 8.100 chiều trong thờigian thực (< 50ms).
- **Tương thích bộ lọc:** Kích thước 128 chia hết cho 2, 4, 8, 16, 32 — phù hợp với các tham số HOG (blockSize=16, cellSize=8) và LBP (radius=2).

---

## Câu 2: Kỹ thuật xử lý và tìm kiếm ảnh mặt động vật

### Quá trình xử lý ảnh

Quá trình xử lý ảnh trong hệ thống được chia thành 3 giai đoạn chính: **thu nhận ảnh → tiền xử lý → trích xuất đặc trưng**.

#### 2.1.1. Thu nhận ảnh

Ảnh được thu thập từ các nguồn dữ liệu công khai (AFHQ, Oxford Pet) và được lưu trữ vào hệ thống qua API upload. Mỗi ảnh khi upload sẽ được:

- Tính giá trị băm SHA-256 để kiểm tra trùng lặp
- Gán nhãn loài động vật (`animal_type`)
- Phân loại vai trò: `corpus` (ảnh trong kho) hoặc `query` (ảnh truy vấn)

#### 2.1.2. Tiền xử lý ảnh (Preprocessing)

Tiền xử lý là bước quan trọng để chuẩn hóa đầu vào trước khi trích xuất đặc trưng. Pipeline tiền xử lý của hệ thống gồm 3 bước:

**Bước 1: Resize (Chỉnh kích thước)**

Ảnh đầu vào có kích thước bất kỳ được resize về 128×128 pixel. Thuật toán sử dụng nội suy:
- `INTER_AREA` nếu ảnh gốc lớn hơn 128×128 (lấy mẫu tốt, giảm răng cưa)
- `INTER_LINEAR` nếu ảnh gốc nhỏ hơn (nội suy tuyến tính)

**Bước 2: Gaussian Blur (Làm mờ nhẹ)**

Áp dụng bộ lọc Gaussian với kernel 3×3, sigma=0 để:
- Loại bỏ nhiễu cường độ cao tần số (salt & pepper noise)
- Làm mịn biên ảnh mà không mất chi tiết cấu trúc quan trọng
- Giúp các đặc trưng như HOG và LBP ổn định hơn

**Bước 3: CLAHE (Cân bằng histogram thích ứng)**

CLAHE (Contrast Limited Adaptive Histogram Equalization) được áp dụng trên kênh L của không gian màu LAB:

1. Chuyển ảnh BGR → LAB
2. Tách kênh L (độ sáng), A, B
3. Áp dụng CLAHE với `clipLimit=2.0`, `tileGridSize=(8,8)`
4. Gộp kênh L đã cân bằng với A, B
5. Chuyển về BGR

*Tác dụng của CLAHE:*
- Cải thiện độ tương phản cục bộ trên các vùng ảnh (ví dụ: vùng lông sáng/tối)
- Giới hạn khuếch đại nhiễu bằng `clipLimit`
- Phù hợp với ảnh mặt động vật có vùng sáng/tối không đều

---

### Đặc trưng của ảnh và biểu diễn ảnh

#### 2.2.1. Ảnh số là gì?

Ảnh số là ma trận 2 chiều (hoặc 3 chiều với ảnh màu) trong đó mỗi phần tử gọi là **pixel** (picture element). Với ảnh màu BGR 8-bit:

```
I(x, y) = [B(x,y), G(x,y), R(x,y)]
```

Trong đó `x ∈ [0, W-1]`, `y ∈ [0, H-1]`, và `B, G, R ∈ [0, 255]`.

#### 2.2.2. Biểu diễn ảnh trong không gian đặc trưng

Thay vì so sánh ảnh trực tiếp theo pixel (nhạy cảm với dịch chuyển, xoay, thay đổi sáng), hệ thống biểu diễn mỗi ảnh bằng một **vector đặc trưng** (feature vector) trong không gian chiều cao:

```
f(I) = [f₁, f₂, ..., f_D] ∈ ℝ^D
```

Hệ thống sử dụng 6 loại đặc trưng, tổng cộng **8.942 chiều**:

| Đặc trưng | Ký hiệu | Số chiều | Nhóm |
|-----------|---------|----------|------|
| HSV Histogram | HSV | 768 | Màu sắc |
| Color Moments | CM | 9 | Màu sắc |
| Local Binary Pattern | LBP | 18 | Kết cấu |
| Gray Level Co-occurrence Matrix | GLCM | 40 | Kết cấu |
| Histogram of Oriented Gradients | HOG | 8.100 | Hình dạng |
| Hu Moments | Hu | 7 | Hình dạng |

Mỗi vector đặc trưng được **chuẩn hóa L2** (L2-normalized) trước khi lưu trữ và so sánh:

```
v̂ = v / ||v||₂,   với ||v||₂ = √(v₁² + v₂² + ... + v_D²)
```

*Lý do chuẩn hóa L2:*
- Đảm bảo các đặc trưng có độ dài đơn vị, không phụ thuộc vào kích thước ảnh gốc
- Biến độ tương đồng cosine thành tích vô hướng đơn giản: `cos(θ) = a · b`
- Cân bằng đóng góp giữa các đặc trưng có số chiều khác nhau

---

### Các phương pháp xử lý ảnh hiện nay

Ngành xử lý ảnh hiện đại chia thành hai hướng tiếp cận chính:

#### 2.3.1. Phương pháp truyền thống (Hand-crafted features)

Sử dụng các đặc trưng được thiết kế thủ công bởi chuyên gia:
- **Ưu điểm:** Không cần dữ liệu huấn luyện lớn, giải thích được (interpretable), nhẹ, chạy nhanh trên CPU
- **Nhược điểm:** Hiệu suất thấp hơn deep learning trên dữ liệu phức tạp, cần hiểu biết chuyên môn để thiết kế

Hệ thống của chúng ta thuộc nhóm này, sử dụng 6 đặc trưng thủ công.

#### 2.3.2. Phương pháp học sâu (Deep Learning)

Sử dụng mạng nơ-ron tích chập (CNN) để tự động học đặc trưng:
- **Ưu điểm:** Hiệu suất cao, tự động học đặc trưng phức tạp
- **Nhược điểm:** Cần GPU, dữ liệu huấn luyện lớn, khó giải thích, đòi hỏi kiến thức chuyên sâu

Với yêu cầu đề bài (kho ảnh 500+ files) và mục tiêu học tập, phương pháp hand-crafted features là lựa chọn phù hợp vì:
1. Đủ chính xác cho kho dữ liệu vừa và nhỏ
2. Dễ dàng trực quan hóa và giải thích từng đặc trưng
3. Không yêu cầu phần cứng chuyên dụng (GPU)
4. Phù hợp với mục tiêu tìm kiếm (retrieval) hơn phân loại (classification)

---

### Kỹ thuật tìm kiếm ảnh theo nội dung (CBIR)

#### 2.4.1. Content-Based Image Retrieval (CBIR) là gì?

CBIR là kỹ thuật tìm kiếm ảnh dựa trên **nội dung trực quan** của ảnh (màu sắc, kết cấu, hình dạng) thay vì dựa trên metadata (tên file, tag, mô tả văn bản).

**Quy trình CBIR trong hệ thống:**

```
Ảnh truy vấn (Query) → Tiền xử lý → Trích xuất đặc trưng
                                            ↓
Ảnh kết quả ← Sắp xếp theo độ tương đồng ← So sánh với kho ảnh
```

#### 2.4.2. Độ đo tương đồng Cosine

Hệ thống sử dụng **Cosine Similarity** để đo mức độ giống nhau giữa hai ảnh:

```
cos(θ) = (A · B) / (||A||₂ × ||B||₂)
```

Vì tất cả vector đặc trưng đã được chuẩn hóa L2 (`||A||₂ = ||B||₂ = 1`), công thức đơn giản hóa thành:

```
cos(θ) = A · B   (tích vô hướng)
```

*Ý nghĩa:*
- `cos(θ) = 1`: Hai ảnh giống nhau hoàn toàn (theo đặc trưng đó)
- `cos(θ) = 0`: Hai ảnh hoàn toàn khác nhau
- `cos(θ) < 0`: Hai ảnh có đặc trưng đối lập (hiếm gặp sau chuẩn hóa)

#### 2.4.3. Kết hợp đặc trưng (Feature Fusion)

Vì mỗi loại đặc trưng nhìn nhận ảnh theo một góc độ khác nhau, hệ thống kết hợp 6 đặc trưng bằng **trung bình có trọng số** (weighted sum):

```
S_total = w_hsv × S_hsv + w_cm × S_cm + w_lbp × S_lbp
        + w_glcm × S_glcm + w_hog × S_hog + w_hu × S_hu
```

**Trọng số mặc định:**

| Đặc trưng | Trọng số | Lý do |
|-----------|----------|-------|
| HOG | 0.25 | Hình dạng khuôn mặt là đặc trưng phân biệt mạnh nhất |
| HSV | 0.20 | Màu lông quan trọng để phân biệt loài |
| LBP | 0.15 | Kết cấu lông cục bộ |
| GLCM | 0.15 | Kết cấu lông không gian |
| Hu | 0.15 | Hình dạng tổng thể, bất biến xoay |
| CM | 0.10 | Thống kê màu sắc tóm tắt |

Trọng số được chuẩn hóa để tổng bằng 1. Người dùng có thể điều chỉnh trọng số để ưu tiên đặc trưng phù hợp với truy vấn.

#### 2.4.4. Tìm kiếm top-K

Sau khi tính điểm tương đồng tổng hợp với toàn bộ kho ảnh (N ảnh), hệ thống trả về **K ảnh có điểm cao nhất** (K=5 theo yêu cầu đề bài). Thuật toán sắp xếp sử dụng `argsort` ổn định để xử lý trường hợp điểm bằng nhau.

---

### Các phương pháp rút trích đặc trưng ảnh

#### 2.5.1. HSV Histogram (768 chiều)

**Ý tưởng:** Biểu diễn phân bố màu sắc của ảnh trong không gian HSV (Hue-Saturation-Value).

**Không gian màu HSV:**
- **H (Hue):** Màu sắc chính (0-180° trong OpenCV)
- **S (Saturation):** Độ bão hòa màu (0-255)
- **V (Value):** Độ sáng (0-255)

**Cách tính:**
1. Chuyển ảnh BGR → HSV
2. Tính histogram 3D với kích thước lưới `12 × 8 × 8` (H × S × V)
3. Tổng số bin = 12 × 8 × 8 = **768**
4. Chuẩn hóa L2

*Lý do chọn HSV thay vì RGB:*
- HSV tách biệt màu sắc (H) khỏi độ sáng (V), giúp hệ thống bền vững hơn trước thay đổi ánh sáng
- Phù hợp để phân biệt màu lông giữa các loài (vàng, nâu, đen, trắng, v.v.)

*Thông tin đặc trưng:*
- Độ tương đồng: hai ảnh có cùng màu lông sẽ có HSV histogram gần nhau
- Độ khác biệt: các loài có màu lông khác nhau (ví dụ: cáo đỏ vs sói xám) sẽ có histogram khác biệt rõ rệt

#### 2.5.2. Color Moments (9 chiều)

**Ý tưởng:** Thay vì lưu toàn bộ histogram, chỉ lưu 3 thống kê cơ bản (moment) của mỗi kênh màu.

**Công thức:** Trên mỗi kênh H, S, V:
- **Mean (μ):** Giá trị trung bình → đại diện cho màu chủ đạo
- **Std (σ):** Độ lệch chuẩn → đại diện cho độ đa dạng màu
- **Skewness:** Độ xiên → đại diện cho sự phân bố lệch của màu

Tổng: 3 kênh × 3 moment = **9 chiều**.

*Ưu điểm:* Rất nhỏ gọn (9 số), so sánh nhanh, không phụ thuộc kích thước ảnh.

#### 2.5.3. Local Binary Pattern (LBP) — 18 chiều

**Ý tưởng:** Mô tả kết cấu cục bộ (texture) bằng cách so sánh mỗi pixel với các láng giềng xung quanh.

**Thuật toán:**
1. Với mỗi pixel, xét vòng tròn láng giềng bán kính R=2, P=16 điểm
2. So sánh giá trị của 16 láng giềng với pixel trung tâm: lớn hơn → 1, nhỏ hơn → 0
3. Ghép 16 bit thành một số nhị phân → giá trị LBP của pixel đó
4. Tính histogram 18 bin từ ảnh LBP (uniform LBP: 16+2 patterns)
5. Chuẩn hóa L2

*Lý do chọn LBP:*
- Bất biến với thay đổi độ sáng đơn giản (monotonic illumination)
- Phát hiện tốt các kết cấu lông, da, vân đặc trưng của từng loài
- Tính toán nhanh, hiệu quả trên ảnh xám

#### 2.5.4. GLCM Haralick Features (40 chiều)

**Ý tưởng:** Mô tả kết cấu không gian bằng ma trận đồng xuất hiện mức xám (Gray-Level Co-occurrence Matrix).

**Cách tính:**
1. Chuyển ảnh xám về 32 mức xám (quantization)
2. Xây dựng ma trận đồng xuất hiện GLCM với:
   - 2 khoảng cách: d=1, d=3
   - 4 hướng: 0°, 45°, 90°, 135°
3. Tính 5 đặc trưng Haralick trên mỗi (khoảng cách, hướng):
   - **Contrast:** Độ tương phản cục bộ
   - **Dissimilarity:** Độ khác biệt trung bình
   - **Homogeneity:** Tính đồng nhất (ngược với contrast)
   - **Energy:** Năng lượng / độ đều của phân bố
   - **Correlation:** Tương quan tuyến tính
4. Tổng: 5 × 2 × 4 = **40 chiều**

*Lý do chọn GLCM:*
- Bắt được quan hệ không gian giữa các pixel (khác với LBP chỉ nhìn cục bộ)
- Hiệu quả trong việc phân biệt kết cấu lông dài/ngắn, mịn/thô

#### 2.5.5. HOG — Histogram of Oriented Gradients (8.100 chiều)

**Ý tưởng:** Mô tả hình dạng (shape) bằng cách thống kê hướng gradient trong các vùng cục bộ.

**Cách tính:**
1. Chuyển ảnh về xám
2. Chia ảnh thành các ô (cell) 8×8 pixel
3. Tính gradient (độ lớn và hướng) tại mỗi pixel
4. Trong mỗi cell, tạo histogram 9 bin của hướng gradient
5. Nhóm 2×2 cell thành 1 block, chuẩn hóa block
6. Trượt block với bước 8 pixel (stride)

**Tham số:**
- `winSize = (128, 128)`
- `blockSize = (16, 16)`
- `blockStride = (8, 8)`
- `cellSize = (8, 8)`
- `nbins = 9`
- Số block: 15 × 15 = 225
- Mỗi block: 4 cell × 9 bin = 36 giá trị
- Tổng: 225 × 36 = **8.100 chiều**

*Lý do chọn HOG:*
- Đặc trưng hình dạng mạnh nhất trong hệ thống (trọng số 0.25)
- Phát hiện tốt các cạnh và đường viền khuôn mặt động vật (tai, mũi, hàm)
- Đã được chứng minh hiệu quả trong nhận dạng người (Dalal & Triggs, 2005)

#### 2.5.6. Hu Moments (7 chiều)

**Ý tưởng:** Sử dụng mô-men hình học (geometric moments) để mô tả hình dạng tổng thể, bất biến với phép tịnh tiến, tỷ lệ và xoay.

**Cách tính:**
1. Tính mô-men hình học bậc 3 của ảnh xám
2. Tính 7 moment bất biến Hu từ các mô-men trung tâm
3. Log-transform: `h = -sign(hu) × log₁₀(|hu| + 10⁻¹⁰)` để nén dải giá trị
4. Chuẩn hóa L2

*Lý do chọn Hu Moments:*
- Bất biến xoay (rotation invariant): phù hợp vì ảnh mặt động vật có thể có góc nghiêng nhẹ
- Chỉ 7 chiều, rất nhỏ gọn nhưng mang thông tin hình dạng toàn cục
- Bổ sung cho HOG (HOG nhìn cục bộ, Hu nhìn toàn cục)

#### 2.5.7. Tổng hợp 6 đặc trưng

| Đặc trưng | Chiều | Nhóm | Điểm mạnh | Điểm yếu |
|-----------|-------|------|-----------|----------|
| HSV | 768 | Màu | Phân biệt màu lông | Nhạy cảm ánh sáng |
| CM | 9 | Màu | Nhỏ gọn, nhanh | Mất thông tin phân bố |
| LBP | 18 | Kết cấu | Bất biến sáng, tính lông | Chỉ nhìn cục bộ |
| GLCM | 40 | Kết cấu | Quan hệ không gian | Chậm hơn LBP |
| HOG | 8.100 | Hình dạng | Cạnh, viền khuôn mặt | Chiều cao, cần ảnh rõ |
| Hu | 7 | Hình dạng | Bất biến xoay, toàn cục | Không nhạy với chi tiết |

**Lý do kết hợp 6 đặc trưng:**
- Mỗi đặc trưng bắt được một khía cạnh khác nhau của ảnh (màu, kết cấu, hình dạng)
- Kết hợp giúp hệ thống vững vàng hơn khi một đặc trưng thất bại (ví dụ: hai loài cùng màu nâu thì HSV không phân biệt được, nhưng HOG vẫn phân biệt được hình dáng)
- Trọng số có thể điều chỉnh để ưu tiên đặc trưng phù hợp

---

## Câu 3: Xây dựng hệ thống

### Sơ đồ khối hệ thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Next.js 15 (Frontend)                       │
│  /dashboard    /library    /upload    /search    /evaluate          │
│  Thống kê     Duyệt ảnh   Tải lên   Tìm kiếm    Đánh giá           │
└────────────────┬────────────────────────────────────┬───────────────┘
                 │ REST (JSON)                        │ WebSocket
                 │                                    │ (truy vấn live)
┌────────────────▼────────────────────────────────────▼───────────────┐
│                       FastAPI (Python 3.12)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ /images  │  │ /search  │  │/visualize│  │/evaluate │           │
│  │  Router  │  │  Router  │  │  Router  │  │  Router  │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │             │             │             │                  │
│  ┌────▼─────────────▼─────────────▼─────────────▼─────────────┐   │
│  │  Services: preprocess · features · search_engine · plot    │   │
│  │            evaluator · feature_cache (in-memory)           │   │
│  └────┬──────────────────────────┬─────────────────────────────┘   │
└───────┼──────────────────────────┼─────────────────────────────────┘
        │                          │
   ┌────▼─────┐              ┌─────▼──────┐
   │PostgreSQL │              │ Filesystem │
   │  images   │              │ ./storage/ │
   │feature_sets│             │  originals/│
   │search_runs │             │   plots/   │
   │eval_runs   │             └────────────┘
   └───────────┘
```

**Giải thích các khối:**

1. **Frontend (Next.js 15):** Giao diện người dùng, hiển thị kết quả tìm kiếm, timeline pipeline, biểu đồ đặc trưng.
2. **Backend (FastAPI):** Xử lý API REST, WebSocket streaming, tiền xử lý ảnh, trích xuất đặc trưng, tính toán độ tương đồng.
3. **PostgreSQL:** Lưu trữ metadata ảnh, vector đặc trưng (JSONB), lịch sử tìm kiếm, kết quả đánh giá.
4. **Filesystem:** Lưu ảnh gốc và ảnh trực quan hóa đặc trưng (PNG).

### Quy trình thực hiện

#### 3.2.1. Quy trình thu thập và lưu trữ ảnh (Ingestion)

```
Ảnh đầu vào → Decode BGR → Preprocess (128×128, blur, CLAHE)
                                    ↓
                              Trích xuất 6 đặc trưng
                                    ↓
              ┌─────────────────────┼─────────────────────┐
              ↓                     ↓                     ↓
         Lưu ảnh gốc         Lưu metadata          Lưu feature_set
         (filesystem)         (images table)        (JSONB in PG)
```

**Chi tiết:**
1. **Decode:** Ảnh được giải mã từ bytes thành ma trận BGR (OpenCV)
2. **Preprocess:** Resize → Gaussian Blur → CLAHE, ra ảnh 128×128 BGR
3. **Extract:** Chạy 6 extractor, thu được 6 vector đã L2-normalized
4. **Persist:**
   - Ảnh gốc lưu vào `storage/originals/`
   - Metadata (filename, animal_type, kích thước, SHA-256) lưu vào bảng `images`
   - Vector đặc trưng lưu vào bảng `feature_sets` (dạng JSONB)
5. **Cache invalidation:** Đánh dấu cache ma trận đặc trưng "dirty" để rebuild lần tìm kiếm sau

#### 3.2.2. Quy trình tìm kiếm ảnh (Search)

```
Ảnh truy vấn → Decode → Preprocess → Extract 6 features
                                              ↓
                                       Load corpus matrices
                                       (from cache / PG)
                                              ↓
                                    Cosine similarity per feature
                                              ↓
                                       Weighted fusion
                                              ↓
                                       Top-K ranking (K=5)
                                              ↓
                                       Persist search_run
```

**Các bước chi tiết:**

| Bước | Tên | Mô tả | Đầu ra |
|------|-----|-------|--------|
| 1 | Decode | Giải mã ảnh upload | BGR uint8 array |
| 2 | Preprocess | Resize + Blur + CLAHE | 128×128 BGR |
| 3 | Extract | Trích xuất 6 đặc trưng | 6 vector L2-normalized |
| 4 | Load Corpus | Tải ma trận đặc trưng toàn bộ kho | `dict[str, (N, D) ndarray]` |
| 5 | Cosine | Tính cosine từng đặc trưng: `M @ q` | 6 vector điểm `(N,)` |
| 6 | Fuse | Trung bình có trọng số | Vector điểm tổng `(N,)` |
| 7 | Rank | Sắp xếp giảm dần, lấy top-K | K kết quả có rank, score |
| 8 | Persist | Lưu kết quả vào `search_runs` | `run_id` cho re-rank sau |

#### 3.2.3. Quy trình Re-rank (điều chỉnh trọng số)

Khi người dùng điều chỉnh thanh trượt trọng số trên giao diện:
1. Hệ thống KHÔNG chạy lại extract (tiết kiệm thờigian)
2. Lấy lại các `per_feature_sims` đã tính từ lần truy vấn trước
3. Tính lại `S_total` với trọng số mới
4. Sắp xếp và trả về top-K mới (< 100ms)

### Kết quả trung gian trong quá trình tìm kiếm

Trong quá trình xử lý một truy vấn, hệ thống sinh ra các kết quả trung gian quan trọng:

#### 3.3.1. Pipeline Trace

Mỗi lần tìm kiếm được ghi lại chi tiết các stage và thờigian thực thi:

```json
[
  {"name": "decode", "elapsed_ms": 5, "detail": {}},
  {"name": "preprocess", "elapsed_ms": 12, "detail": {}},
  {"name": "extract", "elapsed_ms": 45, "detail": {"dims": {"hsv":768, "cm":9, ...}}},
  {"name": "feature.hsv", "elapsed_ms": 4, "detail": {"dim": 768}},
  {"name": "feature.cm", "elapsed_ms": 1, "detail": {"dim": 9}},
  ...
  {"name": "load_corpus", "elapsed_ms": 8, "detail": {"corpus_size": 500}},
  {"name": "cosine", "elapsed_ms": 15, "detail": {"hsv": 500, "cm": 500, ...}},
  {"name": "rank", "elapsed_ms": 2, "detail": {"top_k": 5, "weights": {"hog":0.25, ...}}}
]
```

#### 3.3.2. Per-Feature Sub-scores

Hệ thống lưu điểm tương đồng riêng của từng đặc trưng để phục vụ re-rank và phân tích:

```json
{
  "image_ids": [1, 2, 3, ..., 500],
  "scores": {
    "hsv": [0.82, 0.34, 0.91, ...],
    "hog": [0.75, 0.88, 0.12, ...],
    ...
  }
}
```

#### 3.3.3. Visualisations

Tại mỗi bước, hệ thống có thể sinh ảnh trực quan:
- **Preprocess:** 4-panel (original → resized → blurred → CLAHE)
- **HSV:** Biểu đồ histogram 3D
- **LBP:** Ảnh mã hóa LBP + histogram 18 bin
- **GLCM:** Heatmap ma trận đồng xuất hiện
- **HOG:** Ảnh overlay hướng gradient lên ảnh xám
- **Hu:** Biểu đồ cột 7 moment

---

### Cấu trúc cơ sở dữ liệu

Hệ thống quản lý 4 bảng chính trong PostgreSQL:

**Bảng `images`:**
- `id`: Khóa chính
- `sha256`: Giá trị băm nội dung (duy nhất, chống trùng lặp)
- `filename`, `storage_path`: Tên file và đường dẫn lưu trữ
- `animal_type`: Loài động vật (cat, dog, fox, ...)
- `width`, `height`, `size_bytes`: Kích thước ảnh
- `role`: `corpus` (ảnh kho) hoặc `query` (ảnh truy vấn)
- `uploaded_at`: Thờigian tải lên

**Bảng `feature_sets`:**
- `image_id`: Khóa chính, tham chiếu `images.id`
- `vectors`: JSONB chứa 6 vector đặc trưng `{ "hsv": [...], "hog": [...], ... }`
- `dims`: JSONB chứa số chiều mỗi vector `{ "hsv": 768, ... }`
- `extractor_ver`: Phiên bản extractor (để invalidate khi cập nhật code)

**Bảng `search_runs`:**
- `id`: Khóa chính
- `query_image_id`: Ảnh truy vấn (có thể null)
- `weights`: Trọng số sử dụng (JSONB)
- `results`: Top-K kết quả (JSONB)
- `pipeline_trace`: Chi tiết từng bước xử lý (JSONB)
- `elapsed_ms`: Tổng thờigian xử lý (ms)

**Bảng `evaluation_runs`:**
- Lưu kết quả đánh giá hiệu suất hệ thống (Precision@K, MAP, ablation)

---

## Tài liệu tham khảo

1. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*.
2. Haralick, R. M., et al. (1973). Textural features for image classification. *IEEE SMC*.
3. Ojala, T., Pietikäinen, M., & Harwood, D. (1996). A comparative study of texture measures with classification based on featured distributions. *Pattern Recognition*.
4. Hu, M. K. (1962). Visual pattern recognition by moment invariants. *IRE Trans. Info Theory*.
5. Smeulders, A. W., et al. (2000). Content-based image retrieval at the end of the early years. *IEEE TPAMI*.
6. OpenCV Documentation: https://docs.opencv.org/
7. Scikit-image Documentation: https://scikit-image.org/
8. SQLAlchemy 2.0 Documentation: https://docs.sqlalchemy.org/
9. FastAPI Documentation: https://fastapi.tiangolo.com/

---

> **Ghi chú:** Phần xây dựng ứng dụng chi tiết (cài đặt thư viện, viết code frontend/backend, triển khai Docker, demo hệ thống) sẽ được trình bày trong phần tiếp theo.
