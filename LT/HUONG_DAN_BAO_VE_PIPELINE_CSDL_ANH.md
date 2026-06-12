# Hướng dẫn trình bày và bảo vệ pipeline tìm kiếm ảnh động vật

> Tài liệu này được viết theo **mã nguồn hiện tại của project** trong thư mục
> `csdldptv2`, không lấy nguyên câu trả lời của đề tài hoa hoặc hình học.
>
> Hệ thống là CBIR (Content-Based Image Retrieval): tìm ảnh khuôn mặt động vật
> theo nội dung thị giác bằng sáu đặc trưng thủ công, không dùng nhãn loài để
> tính trực tiếp độ giống nhau.

---

## 1. Câu trả lời mở đầu ngắn gọn

Nếu giảng viên yêu cầu trình bày toàn bộ đề tài trong khoảng một phút, có thể
trả lời:

> Hệ thống của em là hệ tìm kiếm ảnh dựa trên nội dung cho ảnh khuôn mặt động
> vật. Kho file hiện có 7.826 ảnh thuộc 11 loại: bear, cat, chicken, dog, fox,
> horse, lion, panther, peacock, tiger và wolf. Mỗi ảnh được giải mã về BGR,
> resize về 128 x 128, lọc Gaussian 3 x 3 và tăng tương phản bằng CLAHE trên
> kênh L của LAB. Sau đó hệ thống trích sáu vector gồm HSV Histogram và Color
> Moments cho màu sắc, LBP và GLCM cho kết cấu, HOG và Hu Moments cho hình dạng.
> Mỗi vector được chuẩn hóa L2 và lưu trong PostgreSQL bằng sáu cột pgvector.
> Khi tìm kiếm, hệ thống trích cùng bộ đặc trưng cho ảnh truy vấn, tính cosine
> similarity từng đặc trưng, nhân trọng số, cộng điểm và xếp hạng Top-K.

Ba ý phải nhớ:

1. Ảnh chuẩn của pipeline là **128 x 128**, không phải 256 x 256.
2. `animal_type` là metadata/ground truth, không phải đầu vào của công thức
   cosine trong chế độ tìm kiếm content-only.
3. Pipeline hiện tại tính đặc trưng trên **toàn ảnh**, chưa có mask tách vật thể
   và nền. Không được khẳng định hệ thống biết chính xác pixel nào là động vật.

---

## 2. Bộ dữ liệu

### 2.1. Bộ dữ liệu gồm bao nhiêu loại?

Kho ảnh tại `storage/originals` có **11 thư mục nhãn**:

| STT | Nhãn `animal_type` | Số file ảnh trên storage |
|---:|---|---:|
| 1 | `bear` | 146 |
| 2 | `cat` | 1.590 |
| 3 | `chicken` | 150 |
| 4 | `dog` | 1.144 |
| 5 | `fox` | 730 |
| 6 | `horse` | 150 |
| 7 | `lion` | 1.139 |
| 8 | `panther` | 888 |
| 9 | `peacock` | 149 |
| 10 | `tiger` | 1.074 |
| 11 | `wolf` | 666 |
| | **Tổng** | **7.826** |

Lưu ý:

- Đây là số file ảnh hợp lệ trên filesystem tại thời điểm kiểm tra.
- `storage/originals/.DS_Store` không phải ảnh và không được tính.
- PostgreSQL không chạy tại thời điểm soạn tài liệu, vì vậy phải dùng SQL ở
  phần dưới để xác nhận số bản ghi thật khi demo.
- Thư mục nguồn `data/animals` có cấu hình cân bằng 150 ảnh cho mỗi lớp, tổng
  1.650 ảnh. `storage/originals` lớn hơn vì đã chứa thêm dữ liệu từ các lần
  nhập khác.

### 2.2. Vì sao số file và số bản ghi CSDL có thể khác?

Một file có trên ổ đĩa chưa chắc đã có bản ghi đầy đủ trong CSDL. Ngược lại,
một bản ghi có thể trỏ đến file bị di chuyển hoặc xóa. Ngoài ra hệ thống chống
trùng bằng SHA-256 nên hai file khác tên nhưng cùng nội dung chỉ tạo một ảnh.

Khi bảo vệ nên nói:

> Em phân biệt ba khái niệm: dữ liệu nguồn, file đã lưu trên storage và dữ liệu
> đã index trong PostgreSQL. Corpus thật dùng để tìm kiếm là các ảnh có cả bản
> ghi `images` và vector tương ứng trong `feature_sets`.

### 2.3. Cách show số lượng dữ liệu

Đếm file trên storage:

```bash
find storage/originals -type f \( \
  -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o \
  -iname '*.webp' -o -iname '*.bmp' -o -iname '*.tif' -o \
  -iname '*.tiff' \
\) | wc -l
```

Đếm theo lớp trong PostgreSQL:

```sql
SELECT animal_type, COUNT(*) AS image_count
FROM images
WHERE role = 'corpus'
GROUP BY animal_type
ORDER BY animal_type;
```

Đếm số ảnh đã có đủ đặc trưng:

```sql
SELECT
    COUNT(*) AS total_images,
    COUNT(fs.image_id) AS indexed_images,
    COUNT(*) - COUNT(fs.image_id) AS missing_feature_sets
FROM images AS i
LEFT JOIN feature_sets AS fs ON fs.image_id = i.id
WHERE i.role = 'corpus';
```

### 2.4. Dữ liệu có cân bằng không?

Kho `storage/originals` hiện **không cân bằng**. Cat có 1.590 ảnh trong khi bear
chỉ có 146 ảnh. Điều này có hai hệ quả:

- Chỉ số micro có thể bị các lớp đông chi phối.
- Một lớp đông có nhiều cơ hội xuất hiện trong Top-K hơn.

Vì vậy khi đánh giá nên báo cáo:

- Precision@K và MAP@K tổng thể.
- Chỉ số theo từng lớp.
- Macro average để mỗi loài có trọng số ngang nhau.
- Số lượng ảnh mỗi lớp đi kèm kết quả.

---

## 3. Pipeline tổng thể

```text
Ảnh query hoặc ảnh corpus
        |
        v
Đọc bytes / file bằng OpenCV
        |
        v
Ảnh BGR uint8, kích thước gốc H x W x 3
        |
        v
Resize 128 x 128
        |
        v
Gaussian blur 3 x 3
        |
        v
CLAHE trên kênh L của LAB
        |
        v
Trích xuất 6 đặc trưng
  + HSV Histogram: 768 chiều
  + Color Moments: 9 chiều
  + LBP: 18 chiều
  + GLCM: 40 chiều
  + HOG: 8.100 chiều
  + Hu Moments: 7 chiều
        |
        v
Chuẩn hóa L2 riêng từng vector
        |
        v
Lưu PostgreSQL/pgvector hoặc dùng làm query vector
        |
        v
Cosine similarity theo từng đặc trưng
        |
        v
Weighted late fusion
        |
        v
Sắp xếp giảm dần và trả Top-K
```

Tổng số phần tử của sáu vector là:

$$
768+9+18+40+8100+7=8942
$$

Hệ thống **không nối 8.942 số thành một vector duy nhất để so sánh**. Mỗi nhóm
được chuẩn hóa và so sánh riêng, sau đó mới kết hợp điểm. Cách này gọi là
`late fusion`.

---

## 4. Tiền xử lý và câu hỏi về 256 x 256

### 4.1. Pipeline dùng kích thước nào?

Backend hiện tại quy định:

```python
TARGET_SIZE = (128, 128)
```

Ảnh sau tiền xử lý có dạng:

```text
(height, width, channels) = (128, 128, 3)
```

Tức là có:

$$
128 \times 128 = 16.384\ \text{pixel}
$$

và mỗi pixel có ba kênh BGR kiểu `uint8`.

### 4.2. Vậy 256 biểu diễn cái gì?

Trong project hiện tại, số `256` có ba ngữ cảnh dễ nhầm:

1. Pixel `uint8` có 256 mức nguyên từ `0` đến `255`.
2. OpenCV dùng khoảng histogram S và V `[0, 256)`, cận trên 256 không phải một
   giá trị pixel hợp lệ.
3. Biểu đồ HOG chỉ lấy mẫu tối đa 256 phần tử để dễ vẽ; vector HOG thật vẫn có
   8.100 chiều.

Không nên nói pipeline resize ảnh thành `256 x 256`, vì điều đó sai với code
hiện tại.

Ngoài ra giao diện `frontend/src/app/search/page.tsx` còn một dòng mô tả cũ
`Resize 224×224`, nhưng phần hiển thị chi tiết và backend đều dùng 128 x 128.
Khi giải thích phải lấy backend làm nguồn sự thật.

### 4.3. Resize làm gì?

Resize giúp:

- Mọi ảnh có cùng kích thước đầu vào.
- HOG luôn sinh đúng 8.100 phần tử.
- Giảm chi phí tính toán.
- Cho phép so sánh vector cùng số chiều.

Nhược điểm:

- Resize thẳng về hình vuông có thể làm méo tỷ lệ.
- Ảnh quá lớn bị mất chi tiết.
- Ảnh quá nhỏ phải nội suy thêm pixel.

Code chọn nội suy:

- `INTER_AREA` khi thu nhỏ.
- `INTER_LINEAR` khi phóng lớn.

### 4.4. Gaussian blur 3 x 3 làm gì?

Gaussian blur làm giảm nhiễu cao tần và artefact nén trước khi trích đặc trưng.
Kernel nhỏ 3 x 3 được chọn để không làm mất quá nhiều cạnh và texture.

Nếu blur quá mạnh:

- HOG mất cạnh.
- LBP mất pattern lông nhỏ.
- GLCM trở nên đồng nhất quá mức.

### 4.5. CLAHE làm gì?

Ảnh được đổi từ BGR sang LAB. CLAHE chỉ tác động lên kênh `L`, tức độ sáng:

```text
BGR -> LAB -> tách L, a, b
    -> CLAHE(L)
    -> ghép L mới với a, b cũ
    -> LAB -> BGR
```

Thông số:

```text
clipLimit = 2.0
tileGridSize = 8 x 8
```

Mục tiêu:

- Tăng tương phản cục bộ.
- Làm rõ vùng mặt tối hoặc thiếu sáng.
- Hạn chế khuếch đại nhiễu hơn histogram equalization toàn cục.

---

## 5. Bộ thuộc tính của một ảnh

Có hai lớp thông tin:

### 5.1. Metadata

Metadata mô tả file và quản lý dữ liệu:

| Thuộc tính | Ý nghĩa |
|---|---|
| `id` | Khóa chính |
| `sha256` | Mã băm nội dung để chống trùng |
| `filename` | Tên file gốc |
| `storage_path` | Đường dẫn tương đối trên storage |
| `animal_type` | Nhãn loài |
| `width`, `height` | Kích thước ảnh gốc |
| `size_bytes` | Dung lượng file |
| `role` | `corpus` hoặc `query` |
| `uploaded_at` | Thời điểm nhập |

### 5.2. Đặc trưng nội dung

| Đặc trưng | Nhóm thông tin | Số chiều |
|---|---|---:|
| HSV Histogram | Màu sắc | 768 |
| Color Moments | Màu sắc thống kê | 9 |
| LBP | Kết cấu cục bộ | 18 |
| GLCM | Quan hệ mức xám | 40 |
| HOG | Cạnh và hình dạng cục bộ | 8.100 |
| Hu Moments | Hình dạng toàn cục | 7 |

Metadata giúp quản lý, lọc và đánh giá. Vector giúp tính độ tương đồng nội dung.

---

## 6. HSV Histogram: từng phần tử biểu diễn gì?

### 6.1. Vì sao HSV biểu diễn được màu?

OpenCV chuyển mỗi pixel BGR thành ba đại lượng:

- `H`: loại màu/sắc độ.
- `S`: độ bão hòa, màu đậm hay nhạt.
- `V`: độ sáng.

Hệ thống chia miền giá trị thành:

```text
H: 12 khoảng trên [0, 180)
S:  8 khoảng trên [0, 256)
V:  8 khoảng trên [0, 256)
```

Mỗi pixel rơi vào đúng một ô `(h_bin, s_bin, v_bin)`. Hệ thống đếm số pixel
trong từng ô.

### 6.2. Vì sao vector có 768 phần tử?

$$
12 \times 8 \times 8 = 768
$$

Histogram ba chiều được flatten thành vector một chiều.

Với chỉ số vector `i`:

```text
h_bin = i // 64
remainder = i % 64
s_bin = remainder // 8
v_bin = remainder % 8
```

Ngược lại:

$$
i=h\_bin \times 64+s\_bin \times 8+v\_bin
$$

### 6.3. Một phần tử cụ thể có ý nghĩa gì?

Ví dụ `vector[75]`:

```text
h_bin = 75 // 64 = 1
s_bin = (75 % 64) // 8 = 1
v_bin = 75 % 8 = 3
```

Nó biểu diễn lượng pixel thuộc tổ hợp:

- Hue bin số 1.
- Saturation bin số 1.
- Value bin số 3.

Trước chuẩn hóa, giá trị là số pixel. Sau chuẩn hóa L2, giá trị là số đếm đã
chia cho độ dài Euclidean của toàn histogram:

$$
\hat{h}_i=\frac{h_i}{\sqrt{\sum_j h_j^2}}
$$

Vì vậy không nên gọi chính xác nó là “tỷ lệ pixel” trừ khi histogram được
chuẩn hóa L1. Trong code hiện tại đó là **magnitude đã chuẩn hóa L2**.

### 6.4. Vector HSV mẫu thực tế

Ảnh mẫu:

```text
storage/originals/cat/flickr_cat_000143.jpg
```

Hai mươi phần tử đầu:

```text
[0.000000, 0.010409, 0.014871, 0.010781, 0.026023,
 0.015986, 0.026023, 0.037548, 0.000372, 0.006692,
 0.007435, 0.013755, 0.012640, 0.015614, 0.017473,
 0.005205, 0.000744, 0.008551, 0.011153, 0.008551]
```

Thông tin kiểm tra:

```text
số chiều       = 768
L2 norm        = 1.0
số bin khác 0  = 187
giá trị lớn nhất = 0.371392
```

Các bin mạnh nhất gồm:

| Index | `(H bin, S bin, V bin)` | Giá trị L2 |
|---:|---|---:|
| 75 | `(1, 1, 3)` | 0.371392 |
| 222 | `(3, 3, 6)` | 0.349086 |
| 229 | `(3, 4, 5)` | 0.294065 |
| 68 | `(1, 0, 4)` | 0.281053 |
| 69 | `(1, 0, 5)` | 0.276220 |

### 6.5. Tại sao dùng HSV mà không dùng RGB?

RGB/BGR trộn thông tin màu và độ sáng trong cả ba kênh. Khi ánh sáng thay đổi,
cả ba giá trị có thể đổi đồng thời. HSV tách tương đối rõ:

- Hue: loại màu.
- Saturation: độ đậm màu.
- Value: độ sáng.

Điều này làm mô tả màu dễ giải thích hơn và bền hơn phần nào với chiếu sáng.

Tuy nhiên không nên trả lời “HSV hoàn toàn không bị ảnh hưởng bởi ánh sáng”.
Điều đó không đúng. Hue có thể không ổn định ở vùng saturation thấp, camera và
white balance vẫn làm màu thay đổi.

### 6.6. Làm sao biết histogram của vật thể và histogram của nền?

**Trong code hiện tại không biết.**

Lệnh:

```python
cv2.calcHist([hsv], [0, 1, 2], None, ...)
```

truyền mask là `None`, nghĩa là mọi pixel đều được đếm:

$$
Histogram_{toàn\ ảnh}
=Histogram_{động\ vật}+Histogram_{nền}
$$

Do đó:

- Nền chiếm diện tích lớn có thể chi phối HSV.
- Hệ thống không thể chỉ nhìn vector hiện tại và đánh dấu chắc chắn bin nào
  thuộc vật thể.
- Dữ liệu cận mặt chỉ **giảm** ảnh hưởng nền, không loại bỏ nó.

Muốn tách phải có mask:

```python
hist_foreground = cv2.calcHist(
    [hsv],
    [0, 1, 2],
    foreground_mask,
    [12, 8, 8],
    [0, 180, 0, 256, 0, 256],
)
```

Mask có thể lấy từ:

- Segmentation model.
- Face/head detector rồi crop.
- GrabCut với bounding box.
- Mask được gán nhãn thủ công.

---

## 7. Color Moments: con số biểu diễn gì và vì sao có dấu âm?

### 7.1. Cấu trúc vector

Vector có thứ tự:

```text
[
  H_mean, H_std, H_third,
  S_mean, S_std, S_third,
  V_mean, V_std, V_third
]
```

Mỗi kênh có ba đại lượng:

Trung bình:

$$
\mu=\frac{1}{N}\sum_i x_i
$$

Độ lệch chuẩn:

$$
\sigma=\sqrt{\frac{1}{N}\sum_i(x_i-\mu)^2}
$$

Moment trung tâm bậc ba có lấy căn bậc ba:

$$
t=\sqrt[3]{\frac{1}{N}\sum_i(x_i-\mu)^3}
$$

Code gọi biến này là `skew`, nhưng về toán học nó không phải skewness chuẩn

$$
\gamma_1=\frac{E[(X-\mu)^3]}{\sigma^3}
$$

Nên khi bị hỏi kỹ, hãy gọi đúng là:

> signed cube-root of the third central moment.

### 7.2. Ý nghĩa từng số

- `mean`: mức trung bình của kênh.
- `std`: mức phân tán quanh trung bình.
- `third`: phân bố lệch về phía giá trị thấp hay cao.

`mean` và `std` không âm. Thành phần bậc ba có thể âm.

### 7.3. Vì sao có số âm?

Nếu phần lớn giá trị nằm phía trên trung bình nhưng có đuôi dài về phía thấp,
moment bậc ba có thể âm:

$$
E[(X-\mu)^3] < 0
$$

Hàm `np.cbrt` bảo toàn dấu nên vector có số âm. Chuẩn hóa L2 chỉ chia cho một
số dương, do đó không làm mất dấu.

### 7.4. Vector mẫu

Giá trị thô của ảnh mèo mẫu:

```text
[ 36.621643, 24.213877, 31.793571,
  56.146729, 48.279136, 46.968666,
 160.680786, 54.086982,-38.496547]
```

Sau chuẩn hóa L2:

```text
[ 0.181186, 0.119798, 0.157299,
  0.277786, 0.238861, 0.232378,
  0.794969, 0.267595,-0.190462]
```

Phần tử cuối âm vì phân bố kênh V có moment trung tâm bậc ba âm.

### 7.5. Hạn chế quan trọng

Hue là đại lượng tuần hoàn. Trong OpenCV, Hue gần 0 và gần 179 có thể đều là
vùng đỏ, nhưng trung bình số học coi chúng cách xa nhau. Color Moments hiện
tại chưa dùng circular statistics cho Hue. Đây là một hạn chế nên chủ động
nêu nếu bị hỏi sâu.

---

## 8. LBP: kết cấu cục bộ được tạo ra như thế nào?

### 8.1. Nguyên lý

Ảnh được chuyển sang grayscale. Với mỗi pixel trung tâm, lấy 16 điểm lân cận
trên bán kính 2:

```text
P = 16
R = 2
method = uniform
```

Mỗi điểm lân cận được so với pixel trung tâm:

$$
s(g_p-g_c)=
\begin{cases}
1,&g_p\ge g_c\\
0,&g_p<g_c
\end{cases}
$$

Chuỗi 16 bit mô tả pattern sáng-tối quanh pixel.

### 8.2. Từng phần tử vector biểu diễn gì?

Hệ thống dùng Uniform LBP. Các pattern ít chuyển trạng thái `0 <-> 1` được gom
theo loại. Histogram có:

$$
P+2=18\ \text{bin}
$$

Mỗi phần tử vector là số pixel toàn ảnh có mã LBP thuộc pattern tương ứng, sau
đó toàn histogram được chuẩn hóa L2.

### 8.3. LBP nhận biết điều gì?

- Vùng phẳng.
- Cạnh.
- Góc.
- Đốm sáng/tối.
- Lông mịn hay thô.
- Pattern sọc hoặc nhiều chi tiết.

LBP không biết đó là “lông mèo” theo nghĩa ngữ nghĩa. Nó chỉ đo pattern
sáng-tối cục bộ; ý nghĩa loài xuất hiện khi các pattern thống kê lặp lại trên
nhiều ảnh cùng loại.

### 8.4. Hạn chế

Histogram LBP toàn ảnh bỏ mất vị trí. Hai ảnh có cùng tỷ lệ pattern nhưng
pattern nằm ở vị trí khác nhau vẫn có thể cho vector gần giống.

---

## 9. GLCM: quan hệ không gian giữa các mức xám

### 9.1. Tạo ma trận GLCM

Ảnh grayscale 0..255 được lượng tử hóa về 32 mức:

```python
quantised = gray // 8
```

GLCM đếm số lần một pixel mức `i` xuất hiện cạnh một pixel mức `j` tại:

```text
Khoảng cách: d = 1, 3
Góc: 0°, 45°, 90°, 135°
```

Ma trận được đặt:

```text
symmetric = True
normed = True
```

### 9.2. Vector 40 chiều từ đâu?

Năm đại lượng Haralick:

1. Contrast.
2. Dissimilarity.
3. Homogeneity.
4. Energy.
5. Correlation.

Mỗi đại lượng được tính tại:

$$
2\ khoảng\ cách \times 4\ góc=8\ cấu\ hình
$$

Tổng số chiều:

$$
5 \times 2 \times 4=40
$$

Vector được sắp theo từng property, bên trong là các tổ hợp khoảng cách và góc.

### 9.3. Từng đại lượng nói gì?

- Contrast cao: mức xám lân cận khác nhau mạnh.
- Dissimilarity cao: cặp pixel thường chênh lệch.
- Homogeneity cao: giá trị tập trung gần đường chéo GLCM.
- Energy cao: texture có ít trạng thái nổi bật, lặp lại đều.
- Correlation: mức phụ thuộc tuyến tính giữa các pixel lân cận.

### 9.4. Khác LBP ở đâu?

LBP hỏi:

> Pixel xung quanh sáng hơn hay tối hơn pixel trung tâm?

GLCM hỏi:

> Cặp mức xám nào thường xuất hiện cùng nhau ở khoảng cách và hướng xác định?

LBP rất cục bộ. GLCM mô tả quan hệ thống kê theo hướng và khoảng cách.

---

## 10. HOG: đặc trưng hình dạng 8.100 chiều

### 10.1. HOG lấy thông tin hình dạng bằng cách nào?

HOG không tìm contour động vật trực tiếp. Nó đo thay đổi độ sáng:

$$
G_x=I(x+1,y)-I(x-1,y)
$$

$$
G_y=I(x,y+1)-I(x,y-1)
$$

Độ lớn và hướng gradient:

$$
m=\sqrt{G_x^2+G_y^2}
$$

$$
\theta=\operatorname{atan2}(G_y,G_x)
$$

Nơi có cạnh tai, mắt, mũi, mõm hoặc ranh giới đầu-nền sẽ có gradient mạnh.

### 10.2. Cấu hình HOG

```text
Ảnh grayscale: 128 x 128
Window:         128 x 128
Block:           16 x 16
Block stride:     8 x 8
Cell:             8 x 8
Số bin hướng:     9
```

Một block có:

```text
2 x 2 cell = 4 cell
4 cell x 9 bin = 36 giá trị
```

Số vị trí block theo một chiều:

$$
\frac{128-16}{8}+1=15
$$

Tổng số block:

$$
15 \times 15=225
$$

Tổng số phần tử:

$$
225 \times 4 \times 9=8100
$$

### 10.3. Từng phần tử HOG biểu diễn gì?

Một phần tử thuộc:

1. Một vị trí block cụ thể.
2. Một cell trong block đó.
3. Một bin hướng gradient cụ thể.

Giá trị là tổng có trọng số theo magnitude của các gradient có hướng rơi vào
bin đó, sau các bước chuẩn hóa nội bộ của HOG và chuẩn hóa L2 toàn vector.

Nói ngắn gọn:

> Một số HOG không phải tọa độ của một điểm ảnh. Nó là cường độ tương đối của
> một hướng cạnh trong một vùng nhỏ của ảnh.

### 10.4. Vì sao HOG mô tả được hình dạng?

Hình dạng được tạo bởi bố cục cạnh:

- Tai nhọn tạo cạnh chéo.
- Mõm dài tạo các dải gradient theo hướng khác mõm ngắn.
- Mắt và mũi tạo các cấu trúc gradient tập trung.
- Bờ đầu tạo contour tổng quát.

HOG giữ thông tin vùng vì histogram được tính trên từng cell/block, không gom
toàn ảnh thành một histogram duy nhất.

### 10.5. Dấu âm trong HOG?

Vector HOG của OpenCV trong pipeline này thường không âm vì nó là histogram
magnitude gradient. Nếu thấy số âm trong một payload tổng hợp, phải kiểm tra
đó có phải HOG thật hay là Hu/Color Moments/similarity.

### 10.6. Visualize HOG hiện tại có điểm gì cần lưu ý?

Backend hiện chỉ vẽ **256 giá trị lấy mẫu** từ vector 8.100 chiều:

```python
stride = vec.size // 256
sampled = vec[::stride][:256]
```

Biểu đồ này cho thấy cấu trúc vector, nhưng chưa phải hình các đoạn thẳng HOG
overlay trên ảnh. Nếu giảng viên hỏi “hướng cạnh nằm ở đâu”, cách trực quan tốt
hơn là vẽ glyph hướng gradient trên từng cell.

---

## 11. Hu Moments: hình dạng toàn cục

### 11.1. Cách lấy

Pipeline:

```text
BGR -> grayscale
-> cv2.moments(gray)
-> cv2.HuMoments(...)
-> 7 giá trị
-> log transform
-> L2 normalize
```

Hu Moments được xây dựng từ moment không gian, central moments và normalized
central moments. Chúng tóm tắt phân bố khối lượng mức xám của toàn ảnh.

### 11.2. Vì sao log transform?

Hu Moments gốc có thể chênh nhau nhiều bậc độ lớn. Code dùng:

$$
h'_i=-\operatorname{sign}(h_i)
\log_{10}(|h_i|+10^{-10})
$$

Log nén miền giá trị để các phần tử dễ so sánh hơn.

### 11.3. Vì sao Hu có dấu âm?

Dấu đến từ:

- Dấu của Hu moment gốc.
- Công thức `-sign * log10`.

Sau đó L2-normalize không xóa dấu.

Vector Hu mẫu:

```text
[ 0.122164, 0.352153, 0.414599, 0.414865,
 -0.415154,-0.415154, 0.415154]
```

### 11.4. Hu có thực sự là silhouette của động vật không?

Không hoàn toàn. Code tính moments trên ảnh grayscale toàn bộ, không threshold
và không có mask silhouette. Vì vậy Hu hiện tại chứa cả:

- Hình dạng và mức xám của khuôn mặt.
- Nền.
- Ánh sáng.
- Bố cục toàn ảnh.

Muốn Hu mô tả hình dạng vật thể rõ hơn cần segment foreground, lấy binary mask
hoặc contour rồi mới tính moments.

---

## 12. Làm sao phân biệt hình tròn, xoắn ốc hoặc hai hình dạng khác?

Đây là câu hỏi từ đề tài khác nhưng có thể dùng để giải thích nguyên lý.

### 12.1. Histogram màu không đủ

Một hình tròn đỏ và một xoắn ốc đỏ có thể có HSV histogram gần giống nếu diện
tích màu giống nhau. Histogram màu không giữ vị trí pixel.

### 12.2. HOG phân biệt bằng bố cục hướng cạnh

- Hình tròn: hướng gradient thay đổi đều quanh một contour khép kín.
- Xoắn ốc: có nhiều contour cong bên trong, hướng gradient lặp theo nhiều vòng
  và xuất hiện ở nhiều cell trung tâm.

Vì HOG giữ histogram theo vùng, vector hai hình khác nhau.

### 12.3. Hu phân biệt bằng moment toàn cục

Moment của một khối tròn đặc khác moment của đường xoắn ốc. Tuy nhiên hiệu quả
phụ thuộc ảnh nhị phân/mask tốt. Tính Hu trên grayscale có nền sẽ kém rõ hơn.

### 12.4. Visualize để chứng minh

Nên show bốn hình:

1. Ảnh gốc.
2. Grayscale.
3. Edge map hoặc gradient magnitude.
4. HOG glyph/contour.

Sau đó show:

- Vector HOG của hai ảnh.
- Vector Hu của hai ảnh.
- Cosine similarity từng vector.

Đối với đề tài động vật, thay ví dụ bằng:

- Mèo tai nhọn và chó tai cụp.
- Cáo mõm dài và mèo mõm ngắn.
- Hổ có sọc và sư tử có texture đồng đều hơn.

---

## 13. Làm sao biết đâu là động vật và đâu là nền?

### 13.1. Câu trả lời đúng với code hiện tại

> Pipeline hiện tại không thực hiện semantic segmentation. Nó giả định ảnh đầu
> vào là ảnh cận mặt động vật nên vùng quan tâm chiếm phần lớn khung hình. Sáu
> đặc trưng đều được tính trên toàn ảnh, vì vậy nền vẫn đóng góp vào vector.

### 13.2. Điều gì trong project giúp giảm vấn đề?

- Dữ liệu hướng tới khuôn mặt/cận đầu động vật.
- Có script lọc ảnh đầu vào.
- Resize đưa bố cục về kích thước chung.
- CLAHE giảm khác biệt chiếu sáng phần nào.

Nhưng các bước trên không tạo mask.

### 13.3. Muốn xác định cánh hoa/mặt động vật/vật thể thì làm gì?

Quy trình đúng:

```text
Detect hoặc segment vật thể
-> lấy bounding box/mask
-> crop và alignment
-> tính color/texture/shape trên foreground
-> có thể tính thêm background feature riêng
```

Có thể lưu hai bộ đặc trưng:

```text
foreground_hsv
background_hsv
foreground_hog
...
```

hoặc chỉ lưu foreground nếu bài toán không cần nền.

---

## 14. So sánh hai vector và hai ảnh

### 14.1. Chuẩn hóa L2

Mỗi vector được biến đổi:

$$
\hat{x}=\frac{x}{||x||_2}
$$

Sau chuẩn hóa:

$$
||\hat{x}||_2=1
$$

Điều này đã được kiểm tra trên ảnh mẫu: norm các vector xấp xỉ 1.

### 14.2. Cosine similarity

$$
\cos(\theta)=
\frac{x\cdot y}{||x||_2||y||_2}
$$

Vì cả hai vector đã chuẩn hóa:

$$
\cos(\theta)=\hat{x}\cdot\hat{y}
$$

Trong Python:

```python
similarity = float(left_vector @ right_vector)
```

Trong pgvector:

```sql
1.0 - (vec_hsv <=> CAST(:query AS vector))
```

`<=>` trả cosine distance, nên lấy `1 - distance` để ra similarity.

### 14.3. Hai ảnh được so sánh như thế nào?

Tính sáu điểm:

```text
s_hsv, s_cm, s_lbp, s_glcm, s_hog, s_hu
```

Sau đó:

$$
S(q,d)=
\sum_f w_f s_f(q,d)
$$

Backend hiện tại dùng:

```python
{
    "hog":  0.60,
    "lbp":  0.25,
    "glcm": 0.10,
    "cm":   0.05,
    "hu":   0.00,
    "hsv":  0.00,
}
```

Ví dụ:

```text
HOG  = 0.80 -> đóng góp 0.60 x 0.80 = 0.480
LBP  = 0.72 -> đóng góp 0.25 x 0.72 = 0.180
GLCM = 0.65 -> đóng góp 0.10 x 0.65 = 0.065
CM   = 0.70 -> đóng góp 0.05 x 0.70 = 0.035

Fused score = 0.760
```

### 14.4. Vì sao không nối tất cả vector?

HOG có 8.100 chiều, Hu chỉ có 7 chiều. Nếu nối thẳng, HOG có thể chi phối vì
quá nhiều phần tử. Late fusion:

- Chuẩn hóa từng feature riêng.
- Cho trọng số theo hiệu quả thực nghiệm.
- Show được điểm riêng để giải thích.
- Cho phép đổi trọng số và rerank mà không trích xuất lại.

### 14.5. Cosine âm có nghĩa gì?

HSV, LBP và HOG thường không âm nên cosine thường không âm. Color Moments và
Hu có thành phần âm nên cosine có thể âm. Cosine âm nghĩa là hai vector hướng
ngược nhau trong không gian đặc trưng; không có nghĩa “ảnh âm” hay pixel âm.

---

## 15. Làm sao biết đặc trưng nào quan trọng?

Không thể kết luận chỉ vì vector dài hoặc nhìn biểu đồ đẹp. Có bốn cách.

### 15.1. Ablation study

Đánh giá hệ thống đầy đủ, sau đó lần lượt bỏ từng đặc trưng:

```text
All features
Minus HOG
Minus HSV
Minus LBP
Minus GLCM
Minus Hu
Minus CM
```

Nếu bỏ HOG làm MAP giảm mạnh, HOG quan trọng. Nếu bỏ một feature làm MAP tăng,
feature đó đang gây nhiễu trong cấu hình hiện tại.

API:

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"method":"ablation","top_k":10}'
```

### 15.2. Chỉ số đánh giá

Precision@K:

$$
P@K=\frac{\text{số ảnh relevant trong Top-K}}{K}
$$

Average Precision@K:

$$
AP@K=
\frac{\sum_{k=1}^{K}P@k\cdot rel(k)}
{\min(K,\text{tổng số relevant})}
$$

MAP@K là trung bình AP trên nhiều query.

### 15.3. Học trọng số

Project có script `backend/scripts/learn_search_weights.py` dùng:

- Z-score similarity theo feature.
- Pairwise RankNet objective.
- Ràng buộc trọng số không âm và tổng bằng 1.
- Nested stratified cross-validation.

Kết quả thử nghiệm được ghi nhận trên 240 ảnh AFHQ cân bằng:

| Phương pháp | Macro-P@5 | Macro-MAP@10 |
|---|---:|---:|
| Trọng số học | **0.6983** | **0.5662** |
| Trọng số thủ công | 0.6608 | 0.5233 |
| HOG-only | 0.5350 | 0.4216 |
| Trọng số bằng nhau | 0.4383 | 0.2728 |

Thí nghiệm khác trên 3.090 ảnh cho thấy thay đổi trọng số và candidate K cải
thiện:

| Metric | Trước | Sau |
|---|---:|---:|
| Precision@5 | 0.3318 | 0.4679 |
| MAP@10 | 0.2195 | 0.3593 |

Không được trộn hai thí nghiệm thành một vì dataset/protocol khác nhau.

### 15.4. Phân tích từng query

Giao diện trả:

- Điểm cosine từng feature.
- Trọng số.
- Đóng góp `weight x cosine`.
- Fused score.

Ví dụ một ảnh xếp cao vì HOG và LBP cao nhưng HSV thấp cho thấy hệ thống dựa
vào hình dạng/kết cấu hơn màu.

### 15.5. “Quan trọng” không đồng nghĩa với “trọng số lớn”

Trọng số còn phụ thuộc scale và phân phối similarity. Sáu đặc trưng có số
chiều và phân phối cosine khác nhau. Muốn so sánh khoa học nên calibration
similarity trước khi học trọng số.

---

## 16. Làm sao nói ảnh giống và khác nhau?

Không nên chỉ nói “hai ảnh giống 80%”. Hãy tách theo nhóm:

### 16.1. Giống màu

- HSV cosine cao.
- Color Moments cosine cao.
- Có phân bố màu và độ sáng gần nhau.

Nhưng hai loài khác nhau vẫn có thể cùng màu. Ví dụ chó vàng và hổ/sư tử.

### 16.2. Giống kết cấu

- LBP cao: pattern cục bộ tương tự.
- GLCM cao: quan hệ mức xám theo hướng/khoảng cách tương tự.

Ví dụ lông mịn, lông xù, sọc, đốm.

### 16.3. Giống hình dạng

- HOG cao: bố cục cạnh cục bộ tương tự.
- Hu cao: moment toàn cục tương tự.

Ví dụ hình tai, chiều dài mõm, bố cục mắt-mũi.

### 16.4. Ví dụ giải thích một lỗi

Nếu ảnh chó trả về cáo:

- HOG có thể cao vì tai nhọn và mõm dài.
- Màu có thể tương tự.
- Metadata khác nhau nhưng metadata không tham gia similarity.

Đây không phải lỗi SQL. Đó là giới hạn của biểu diễn đặc trưng và định nghĩa
“giống nhau theo nội dung”.

---

## 17. Cấu trúc cơ sở dữ liệu

### 17.1. Có bao nhiêu bảng?

Schema ứng dụng hiện có **4 bảng chính**:

1. `images`.
2. `feature_sets`.
3. `search_runs`.
4. `evaluation_runs`.

Ngoài ra PostgreSQL có bảng quản lý migration `alembic_version`, nhưng đây
không phải bảng nghiệp vụ.

### 17.2. Sơ đồ quan hệ

```text
images
  PK id
  |
  | 1 -- 0..1
  v
feature_sets
  PK/FK image_id

images
  PK id
  |
  | 1 -- 0..N, query_image_id có thể NULL
  v
search_runs

evaluation_runs
  độc lập, lưu báo cáo đánh giá tổng thể
```

### 17.3. Bảng `images`

```text
images
├── id              BIGINT PK
├── sha256          VARCHAR(64) UNIQUE
├── filename        TEXT
├── storage_path    TEXT
├── animal_type     TEXT
├── width           INTEGER
├── height          INTEGER
├── size_bytes      INTEGER
├── role            TEXT CHECK corpus/query
└── uploaded_at     TIMESTAMPTZ
```

Index:

```text
idx_images_animal_type
idx_images_role
```

Ảnh binary không lưu trực tiếp trong DB. DB chỉ lưu đường dẫn, còn file nằm:

```text
storage/originals/<animal_type>/<filename>
```

### 17.4. Bảng `feature_sets`

```text
feature_sets
├── image_id        BIGINT PK, FK -> images.id ON DELETE CASCADE
├── extractor_ver   TEXT
├── vec_hog         vector(8100)
├── vec_hsv         vector(768)
├── vec_lbp         vector(18)
├── vec_glcm        vector(40)
├── vec_hu          vector(7)
└── vec_cm          vector(9)
```

`image_id` vừa là khóa chính vừa là khóa ngoại nên mỗi ảnh có tối đa một bộ
đặc trưng hiện hành.

`extractor_ver = v1.1` cho biết vector được tạo bằng phiên bản extractor nào.
Khi thay tham số, cần tăng version và trích xuất lại.

### 17.5. HNSW index

Các vector HSV, LBP, GLCM, Hu và CM có HNSW index dùng cosine.

HOG 8.100 chiều không có HNSW index trong migration vì vượt giới hạn 2.000
chiều được ghi nhận cho cấu hình pgvector đó. Tuy nhiên code `vector_search.py`
hiện vẫn phát truy vấn `ORDER BY vec_hog <=> query`; PostgreSQL có thể phải
exact scan cho cột này. Đây là điểm kỹ thuật cần nêu cẩn thận, không nói cả sáu
đều chắc chắn dùng HNSW index.

### 17.6. Bảng `search_runs`

```text
search_runs
├── id
├── query_image_id
├── weights          JSONB
├── results          JSONB
├── pipeline_trace   JSONB
├── elapsed_ms
└── created_at
```

Lưu:

- Trọng số đã dùng.
- Top-K kết quả.
- Điểm từng đặc trưng.
- Thời gian từng stage.
- Sub-score để rerank mà không extract lại.

Query upload hiện không nhất thiết được lưu vào `images`, nên `query_image_id`
có thể NULL.

### 17.7. Bảng `evaluation_runs`

```text
evaluation_runs
├── id
├── method
├── precision_at_5
├── map_at_10
├── per_class      JSONB
├── ablation       JSONB
├── started_at
└── finished_at
```

Bảng này lưu kết quả thí nghiệm, không lưu vector ảnh.

### 17.8. Vì sao tách ảnh và đặc trưng?

- Metadata và vector có vòng đời khác nhau.
- Xóa ảnh có thể cascade xóa vector.
- Dễ thay đổi storage file mà không sửa vector.
- Dễ truy vấn metadata mà không tải vector lớn.
- Vector có kiểu và index riêng.

---

## 18. Show cấu trúc và dữ liệu trong PostgreSQL

### 18.1. Liệt kê bảng

```sql
SELECT tablename
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;
```

### 18.2. Xem cột

Trong `psql`:

```text
\d images
\d feature_sets
\d search_runs
\d evaluation_runs
```

Hoặc SQL chuẩn:

```sql
SELECT
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;
```

### 18.3. Show metadata ảnh

```sql
SELECT
    id,
    filename,
    animal_type,
    width,
    height,
    size_bytes,
    storage_path,
    role,
    uploaded_at
FROM images
ORDER BY id
LIMIT 10;
```

### 18.4. Show kích thước vector

```sql
SELECT
    image_id,
    vector_dims(vec_hog)  AS hog_dim,
    vector_dims(vec_hsv)  AS hsv_dim,
    vector_dims(vec_lbp)  AS lbp_dim,
    vector_dims(vec_glcm) AS glcm_dim,
    vector_dims(vec_hu)   AS hu_dim,
    vector_dims(vec_cm)   AS cm_dim,
    extractor_ver
FROM feature_sets
LIMIT 10;
```

### 18.5. Show một vector

```sql
SELECT image_id, vec_cm
FROM feature_sets
WHERE image_id = 1;
```

HSV rất dài nên chỉ nên show đoạn đầu:

```sql
SELECT
    image_id,
    subvector(vec_hsv, 1, 20) AS hsv_first_20
FROM feature_sets
WHERE image_id = 1;
```

Tùy phiên bản pgvector, nếu `subvector` không có thì cast sang text hoặc lấy
vector qua API/Python.

### 18.6. Show index

```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('images', 'feature_sets')
ORDER BY tablename, indexname;
```

---

## 19. Muốn xem đặc tính của một file thì làm như nào?

### 19.1. Qua giao diện

Khởi động hệ thống:

```bash
docker compose up -d
```

Các trang:

```text
http://localhost:3000/upload
http://localhost:3000/search
http://localhost:3000/compare
http://localhost:8000/docs
```

Tại `/upload`:

1. Nhập nhãn.
2. Upload ảnh.
3. Mở “Xem trực quan đặc trưng”.
4. Show preprocess, HSV, CM, LBP, GLCM, HOG, Hu.

Tại `/search`:

1. Thả ảnh query.
2. Show ảnh gốc và sau tiền xử lý.
3. Mở từng stage feature.
4. Xem Top-K và điểm từng đặc trưng.

Tại `/compare`:

1. Upload hai ảnh.
2. Nhấn so sánh.
3. Show fused score và sáu cosine score.

### 19.2. Qua API với ảnh đã lưu

```text
GET /api/v1/images/{image_id}/raw
GET /api/v1/visualize/{image_id}/preprocess
GET /api/v1/visualize/{image_id}/hsv
GET /api/v1/visualize/{image_id}/cm
GET /api/v1/visualize/{image_id}/lbp
GET /api/v1/visualize/{image_id}/glcm
GET /api/v1/visualize/{image_id}/hog
GET /api/v1/visualize/{image_id}/hu
```

Ví dụ:

```bash
curl http://localhost:8000/api/v1/visualize/1/hsv \
  --output hsv.png
```

### 19.3. Visualize ảnh query chưa lưu

```bash
curl -X POST \
  'http://localhost:8000/api/v1/visualize/query?feature=hog' \
  -F 'file=@my_query.jpg' \
  --output query_hog.png
```

### 19.4. So sánh hai file

```bash
curl -X POST http://localhost:8000/api/v1/compare \
  -F 'left=@image_a.jpg' \
  -F 'right=@image_b.jpg'
```

Response:

```json
{
  "fused_score": 0.76,
  "per_feature": {
    "hsv": 0.71,
    "cm": 0.70,
    "lbp": 0.72,
    "glcm": 0.65,
    "hog": 0.80,
    "hu": 0.55
  },
  "left_dims": {
    "hsv": 768,
    "cm": 9,
    "lbp": 18,
    "glcm": 40,
    "hog": 8100,
    "hu": 7
  }
}
```

---

## 20. Trình bày trực quan từng đặc trưng

### 20.1. Preprocess panel

Show bốn ô:

1. Original và kích thước gốc.
2. Resized 128 x 128.
3. Gaussian blurred.
4. CLAHE.

Mục đích là chứng minh pipeline có các bước thật, không chỉ nói lý thuyết.

### 20.2. HSV

Biểu đồ cột 768 bin. Khi trình bày:

- Chọn 10 bin lớn nhất.
- Decode index thành H/S/V bin.
- Không cố đọc toàn bộ 768 cột.

### 20.3. Color Moments

Show chín cột có nhãn:

```text
H mean, H std, H third,
S mean, S std, S third,
V mean, V std, V third
```

### 20.4. LBP

Show histogram 18 bin. Có thể bổ sung ảnh LBP code-map để trực quan hơn vì
biểu đồ hiện tại chỉ show phân bố.

### 20.5. GLCM

Show heatmap 5 x 8:

- Hàng: năm property.
- Cột: hai khoảng cách x bốn góc.

### 20.6. HOG

Biểu đồ hiện tại lấy mẫu vector. Khi thuyết trình nên nói rõ “đây là biểu đồ
magnitude lấy mẫu”, không gọi là edge map. Nên bổ sung HOG glyph nếu có thời
gian.

### 20.7. Hu

Show bảy cột. Nêu rõ chúng đã log transform và L2-normalize, không phải Hu raw.

---

## 21. Câu hỏi phản biện thường gặp

### Câu 1: Hệ thống biết đây là mèo bằng cách nào?

> Hệ thống tìm kiếm không chạy một classifier để kết luận nhãn. Nó biểu diễn
> ảnh thành sáu vector rồi tìm các vector gần nhất. Nếu nhiều ảnh gần nhất có
> metadata `cat`, ta có thể diễn giải query giống nhóm mèo, nhưng nhãn không
> tham gia trực tiếp vào cosine.

### Câu 2: Nếu bỏ metadata thì còn tìm được không?

> Có. Search content-only chỉ cần vector. Metadata cần để hiển thị, quản lý và
> đánh giá đúng/sai theo lớp.

### Câu 3: Một phần tử HSV là màu gì?

> Nó không phải một mã màu RGB duy nhất. Nó là khối lượng pixel trong một
> khoảng Hue, một khoảng Saturation và một khoảng Value. Phải decode index
> thành ba bin mới giải thích được.

### Câu 4: Vì sao vector HSV có nhiều số 0?

> Ảnh chỉ sử dụng một phần nhỏ trong 768 tổ hợp màu-bão hòa-độ sáng. Những tổ
> hợp không có pixel sẽ có số đếm 0 và sau chuẩn hóa vẫn là 0.

### Câu 5: Vì sao vector có số âm?

> Histogram HSV, LBP và HOG không âm. Số âm thường đến từ moment bậc ba của
> Color Moments, log-Hu Moments hoặc cosine của vector có phần tử âm.

### Câu 6: HOG 8.100 chiều có phải 8.100 pixel không?

> Không. Mỗi phần tử là một bin hướng gradient tại một cell trong một block.
> Các block chồng lấn nên một vùng pixel có thể đóng góp vào nhiều block.

### Câu 7: Vì sao HOG lớn hơn mà không chiếm toàn bộ kết quả?

> Vì từng vector được chuẩn hóa L2 riêng và cosine được tính riêng. Sau đó hệ
> thống kết hợp sáu similarity score, không cộng trực tiếp 8.100 phần tử HOG
> với bảy phần tử Hu.

### Câu 8: Tại sao không chỉ dùng HOG?

> HOG mạnh về cạnh nhưng không mô tả tốt màu và một số texture. Kết quả thử
> nghiệm 240 ảnh cho thấy HOG-only Macro-MAP@10 là 0.4216, thấp hơn mô hình học
> trọng số 0.5662. Tuy nhiên mức lợi ích phụ thuộc dataset.

### Câu 9: Làm sao biết ảnh giống thật hay chỉ giống nền?

> Pipeline hiện tại chưa tách được chắc chắn. Có thể kiểm tra bằng visualization,
> crop/mask foreground và chạy lại. Nếu điểm giảm mạnh sau khi thay nền, feature
> đang nhạy với nền.

### Câu 10: Vì sao dùng cosine?

> Vector đã chuẩn hóa L2, cosine đo hướng tương đối và không phụ thuộc norm ban
> đầu. Nó cũng được pgvector hỗ trợ trực tiếp. Tuy nhiên histogram như HSV/LBP
> có thể thử Chi-square hoặc histogram intersection và chọn bằng validation.

### Câu 11: Cosine 0.8 có phải giống 80%?

> Không nên hiểu như xác suất 80%. Đó là độ gần theo góc vector. Ngưỡng “rất
> giống” phải được hiệu chỉnh trên dữ liệu gán nhãn.

### Câu 12: Vì sao resize hình vuông không giữ tỷ lệ?

> Đây là lựa chọn đơn giản để cố định số chiều HOG. Hạn chế là có thể làm méo
> hình. Cải tiến là resize giữ tỷ lệ rồi center crop hoặc padding.

### Câu 13: Vì sao Hu được đặt trọng số 0?

> Theo nhận xét ablation trong code hiện tại, Hu gây nhiễu trên corpus đã đánh
> giá. Lý do hợp lý là Hu được tính trên grayscale toàn ảnh thay vì silhouette
> foreground. Trọng số 0 là kết quả cấu hình hiện tại, không chứng minh Hu vô
> dụng trong mọi bài toán.

### Câu 14: Vì sao HSV cũng có trọng số 0?

> Màu lông không luôn tương quan với loài. Chó vàng có thể gần hổ/sư tử hơn chó
> đen về màu. Trên corpus hiện tại HSV làm giảm MAP nên backend tắt trong fusion
> mặc định, nhưng vẫn trích và lưu để visualize, phân tích hoặc đổi trọng số.

### Câu 15: Nếu trọng số bằng 0 sao vẫn lưu vector?

> Vì vector vẫn có giá trị giải thích, thử nghiệm, rerank và cấu hình cho dataset
> khác. Trọng số là chính sách tìm kiếm, không phải cấu trúc dữ liệu.

### Câu 16: Tìm kiếm có dùng tất cả 7.826 ảnh không?

> Corpus chính xác phải lấy từ `feature_sets` trong PostgreSQL. ANN lấy tối đa
> 200 candidate theo mỗi feature rồi hợp các candidate trước khi fusion. Vì vậy
> nó không brute-force đầy đủ toàn bộ corpus trong hot path.

### Câu 17: ANN có thể bỏ sót không?

> Có. Một ảnh không lọt Top-200 của bất kỳ feature nào sẽ không được fusion.
> Tăng candidate K tăng recall nhưng tốn thời gian. Đây là trade-off giữa tốc độ
> và chất lượng.

### Câu 18: Tại sao HOG không có HNSW index?

> Vector HOG 8.100 chiều vượt giới hạn index được ghi trong migration. Truy vấn
> HOG có thể exact scan. Có thể giảm chiều bằng PCA hoặc dùng halfvec/kiến trúc
> ANN khác nếu cần index HOG.

### Câu 19: SHA-256 có phải đặc trưng ảnh không?

> Không. SHA-256 chỉ phát hiện file trùng nội dung byte-for-byte. Hai ảnh nhìn
> giống nhau nhưng resize hoặc nén khác sẽ có SHA-256 khác; CBIR mới dùng để đo
> giống về nội dung.

### Câu 20: Làm sao chứng minh một feature quan trọng?

> Dùng ablation trên tập test cố định, báo cáo delta MAP/P@K và khoảng tin cậy.
> Không kết luận chỉ từ một query đẹp.

---

## 22. Các điểm lệch giữa backend và frontend cần biết

### 22.1. Trọng số

Backend thực tế:

```text
HOG 0.60, LBP 0.25, GLCM 0.10, CM 0.05, Hu 0, HSV 0
```

Frontend `weights.ts` và bảng `FusionTable` vẫn đang hiển thị bộ cũ:

```text
HOG 0.25, HSV 0.20, LBP 0.15,
GLCM 0.15, Hu 0.15, CM 0.10
```

API response trả `weights` thật từ backend, nhưng `FusionTable` hiện dùng
hằng số frontend cũ. Nếu trình bày đóng góp trọng số trên UI, phải sửa frontend
hoặc giải thích đây là sai lệch giao diện.

### 22.2. Kích thước preprocess

Một hint frontend nói 224 x 224, backend dùng 128 x 128. Phần preview chi tiết
đã ghi đúng 128 x 128.

### 22.3. Tài liệu cũ

`README.md`, `PLAN.md` và `docs/BASE.md` có một số mô tả kiến trúc cũ như JSONB
vector hoặc PostgreSQL không extension. Schema hiện hành đã migration sang
pgvector và xóa các cột JSONB `vectors`, `dims`, `extracted_at`.

Khi bảo vệ, nguồn sự thật ưu tiên:

1. Alembic migration mới nhất.
2. ORM model.
3. Service code đang chạy.
4. Tài liệu cũ chỉ để tham khảo lịch sử.

---

## 23. Kịch bản demo đề xuất

### Phần 1: Show dữ liệu

1. Mở cây `storage/originals`.
2. Chạy thống kê số ảnh theo lớp.
3. Mở 2-3 ảnh mỗi lớp để cho thấy biến thiên.
4. Chạy SQL `GROUP BY animal_type`.

### Phần 2: Show CSDL

1. Mở pgAdmin tại `http://localhost:5050`.
2. Show 4 bảng.
3. Show một dòng `images`.
4. Show `vector_dims` của một dòng `feature_sets`.
5. Show 9 số `vec_cm` và 20 số đầu `vec_hsv`.

### Phần 3: Show pipeline một ảnh

1. Mở `/upload` hoặc `/search`.
2. Chọn một ảnh rõ mặt.
3. Show original -> resize -> blur -> CLAHE.
4. Show sáu biểu đồ.
5. Giải thích một phần tử cụ thể của HSV và CM.

### Phần 4: Show so sánh

Chọn ba cặp:

1. Cùng loài, ngoại hình gần giống.
2. Cùng loài nhưng màu/bố cục khác.
3. Khác loài nhưng ngoại hình gần giống.

Với mỗi cặp:

- Show sáu cosine.
- Chỉ ra feature nào kéo điểm lên.
- Chỉ ra feature nào kéo điểm xuống.

### Phần 5: Show tìm kiếm

1. Upload query.
2. Show Top-5.
3. Mở một result.
4. Show per-feature score và fusion.
5. Thay trọng số, rerank.

### Phần 6: Show đánh giá

1. Chạy evaluation hoặc dùng report đã lưu.
2. Show P@5, MAP@10.
3. Show per-class.
4. Show ablation.
5. Nêu lớp dog khó vì biến thiên nội lớp cao.

---

## 24. Checklist trước khi bảo vệ

- [ ] Docker và PostgreSQL chạy.
- [ ] Migration ở revision mới nhất.
- [ ] Số ảnh DB đã kiểm tra, không chỉ đếm storage.
- [ ] Không có ảnh thiếu feature set.
- [ ] `extractor_ver` thống nhất `v1.1`.
- [ ] Chuẩn bị một image ID chắc chắn có đủ cached plot.
- [ ] Chuẩn bị hai file để `/compare`.
- [ ] Chuẩn bị một query cho kết quả tốt và một query lỗi để phân tích.
- [ ] Sửa hoặc ghi chú trọng số frontend cũ.
- [ ] Sửa hint 224 x 224 thành 128 x 128.
- [ ] Không nói hệ thống đã segment foreground.
- [ ] Không gọi cosine là xác suất.
- [ ] Không gọi signed third moment là skewness chuẩn.
- [ ] Không nói cả sáu vector đều có HNSW index.
- [ ] Phân biệt số ảnh storage với corpus DB.

---

## 25. Kết luận có thể dùng khi kết thúc

> Điểm chính của hệ thống là biến mỗi ảnh thành nhiều cách biểu diễn bổ sung
> nhau: màu, kết cấu và hình dạng. Mỗi con số trong vector đều có nguồn gốc từ
> một phép đo cụ thể trên pixel, không phải giá trị ngẫu nhiên. Việc tìm kiếm là
> so sánh các vector cùng loại bằng cosine rồi late-fusion theo trọng số. CSDL
> tách metadata, vector, lịch sử tìm kiếm và kết quả đánh giá thành bốn bảng.
> Hạn chế lớn nhất hiện tại là đặc trưng tính trên toàn ảnh nên chưa phân biệt
> chắc chắn foreground/background, và ANN candidate pruning có thể bỏ sót. Các
> hạn chế này có thể xử lý bằng detection/segmentation, alignment, metric phù
> hợp từng feature và đánh giá ablation trên tập test độc lập.

---

## 26. File mã nguồn cần mở khi bị hỏi

| Nội dung | File |
|---|---|
| Tiền xử lý | `backend/app/services/preprocess.py` |
| Danh sách feature và version | `backend/app/services/features/__init__.py` |
| HSV | `backend/app/services/features/hsv.py` |
| Color Moments | `backend/app/services/features/color_moments.py` |
| LBP | `backend/app/services/features/lbp.py` |
| GLCM | `backend/app/services/features/glcm.py` |
| HOG | `backend/app/services/features/hog.py` |
| Hu | `backend/app/services/features/hu.py` |
| Tìm kiếm và trọng số | `backend/app/services/search_engine.py` |
| ANN pgvector | `backend/app/services/vector_search.py` |
| ORM/schema | `backend/app/models.py` |
| Migration vector | `backend/alembic/versions/0002_pgvector_columns.py` |
| HNSW index | `backend/alembic/versions/0003_pgvector_notnull_indexes.py` |
| Xóa JSONB cũ | `backend/alembic/versions/0459248fa542_drop_old_jsonb_columns.py` |
| Visualization | `backend/app/services/plot.py` |
| So sánh hai ảnh | `backend/app/routers/compare.py` |
| Đánh giá | `backend/app/services/evaluator.py` |
| Học trọng số | `backend/scripts/learn_search_weights.py` |

