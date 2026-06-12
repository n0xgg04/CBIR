# Bộ lọc ảnh mặt động vật đầu vào

## 1. Mục tiêu

Bộ lọc chạy trước bước ingest vào vector database. Nó không tách nền và không
thay đổi ảnh gốc. Mỗi ảnh được phân vào một trong ba trạng thái:

- `ACCEPT`: đủ điều kiện tự động ingest.
- `REVIEW`: model chưa đủ chắc chắn, cần kiểm tra thủ công.
- `REJECT`: vi phạm ít nhất một điều kiện cứng.

Pipeline:

```text
Ảnh đầu vào
  -> kiểm tra file và SHA-256
  -> Grounding DINO phát hiện mặt/đầu động vật
  -> gộp các bounding box trùng hoặc bao chứa nhau
  -> CLIP xác minh crop bằng prompt dương/âm
  -> đo kích thước, độ nét, ánh sáng, tương phản và vị trí box
  -> ACCEPT / REVIEW / REJECT
  -> xuất manifest và preview
```

## 2. Model sử dụng

- Detector: `IDEA-Research/grounding-dino-tiny`.
- Semantic verifier: `openai/clip-vit-base-patch32`.
- Quality metrics: OpenCV Laplacian variance, Tenengrad, brightness,
  contrast, dark ratio và bright ratio.

Grounding DINO nhận prompt theo metadata:

```text
<animal_type> face
<animal_type> head
animal face
animal head
```

CLIP so sánh crop với hai nhóm mô tả. Giá trị lưu trong `clip_margin` là:

$$
M_{\mathrm{clip}}=
\max(S_{\mathrm{positive}})
-
\max(S_{\mathrm{negative}})
$$

Margin càng lớn thì crop càng giống một mặt động vật rõ ràng hơn các trường
hợp toàn thân, quay lưng, mờ hoặc không có mặt.

## 3. Cài đặt

```bash
cd backend
uv sync --extra dev --extra filter
```

Model weights được tải từ Hugging Face trong lần chạy đầu tiên. Hai model dùng
khoảng hơn 1 GB cache.

## 4. Chạy thử

Một ảnh mỗi lớp:

```bash
cd backend
uv run python -m scripts.filter_animal_faces \
  --source ../data/animals \
  --output ../data/face_filter_trial \
  --max-per-class 1 \
  --device mps \
  --save-previews
```

Chạy toàn bộ dataset:

```bash
cd backend
uv run python -m scripts.filter_animal_faces \
  --source ../data/animals \
  --output ../data/face_filter_full \
  --device mps \
  --save-previews
```

Thêm `--materialize` nếu cần tạo cây thư mục `accepted`, `review` và
`rejected`. Script dùng hard link nếu filesystem hỗ trợ, nếu không sẽ copy.

## 5. Đầu ra

```text
data/face_filter_full/
  manifest.csv
  summary.json
  previews/
    accept/
    review/
    reject/
```

`manifest.csv` chứa:

- Đường dẫn, nhãn metadata và SHA-256.
- Bounding box và confidence.
- Kích thước và tỷ lệ diện tích mặt.
- CLIP margin.
- Laplacian variance và Tenengrad.
- Brightness, contrast, dark ratio, bright ratio.
- Trạng thái và lý do quyết định.

Các lý do thường gặp:

```text
NO_ANIMAL_FACE
DUPLICATE
FACE_TOO_SMALL
FACE_AREA_TOO_SMALL
LOW_DETECTION_CONFIDENCE
CLIP_REJECTED_FACE
TOO_BLURRY
TOO_DARK
OVEREXPOSED
LOW_CONTRAST
MULTIPLE_FACES
FACE_TOUCHES_BORDER
```

## 6. Kết quả thử nghiệm ban đầu

Mẫu 55 ảnh, gồm 5 ảnh cho mỗi lớp:

| Trạng thái | Số ảnh |
|---|---:|
| ACCEPT | 29 |
| REVIEW | 18 |
| REJECT | 8 |

Thời gian trung bình sau khi model đã được cache là khoảng `0.70 giây/ảnh`
trên Apple M3 Pro dùng MPS.

Kết quả cho thấy CLIP có thiên lệch theo tên lớp. Năm ảnh `panther` trong mẫu
đều rơi vào `REVIEW` do `CLIP_UNCERTAIN`, trong khi các lớp `cat`, `lion`,
`tiger` và `wolf` được nhận ổn định hơn. Vì vậy không nên tự động biến
`CLIP_UNCERTAIN` thành `REJECT`.

## 7. Hiệu chỉnh trước khi dùng chính thức

Các ngưỡng hiện tại là baseline kỹ thuật, chưa phải ngưỡng khoa học cuối cùng.
Cần chấm tay ít nhất 200-500 ảnh và chọn ngưỡng trên tập calibration.

Mục tiêu khuyến nghị:

$$
\operatorname{Precision}_{ACCEPT}\ge 0.98
$$

Nghĩa là ít nhất 98% ảnh được tự động nhận phải thực sự đạt tiêu chuẩn. Các
ảnh không chắc chắn được đưa vào `REVIEW` thay vì ép nhận hoặc loại.

Tập test cuối phải độc lập với tập dùng chọn ngưỡng. Báo cáo:

- Precision và recall của `ACCEPT`.
- False Acceptance Rate.
- False Rejection Rate.
- Tỷ lệ `REVIEW`.
- MAP@10 của hệ thống CBIR trước và sau lọc.
