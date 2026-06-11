# Cơ sở lý thuyết đặc trưng và pipeline tìm kiếm ảnh

## 1. Mục tiêu hệ thống

Hệ thống thực hiện tìm kiếm ảnh dựa trên nội dung (Content-Based Image
Retrieval - CBIR). Đầu vào là một ảnh khuôn mặt động vật, đầu ra là các ảnh
trong cơ sở dữ liệu được sắp xếp giảm dần theo mức độ tương đồng nội dung.

Khác với tìm kiếm bằng metadata, CBIR không bắt buộc người dùng nhập tên loài.
Ảnh được biểu diễn bằng các thuộc tính thị giác mức thấp:

- Màu sắc: HSV Histogram, Color Moments.
- Kết cấu: Local Binary Pattern, Gray-Level Co-occurrence Matrix.
- Hình dạng và cấu trúc cạnh: Histogram of Oriented Gradients, Hu Moments.

Metadata `animal_type` được dùng làm nhãn để huấn luyện và đánh giá trọng số,
không được dùng làm bộ lọc ngầm khi đánh giá khả năng tìm kiếm bằng nội dung.

## 2. Pipeline tổng thể

```text
Ảnh đầu vào
  -> Giải mã thành ảnh BGR
  -> Resize về 128 x 128
  -> Gaussian blur 3 x 3
  -> CLAHE trên kênh L của không gian LAB
  -> Trích xuất 6 vector đặc trưng
  -> Chuẩn hóa L2 riêng từng vector
  -> Tính cosine similarity theo từng đặc trưng
  -> Hiệu chỉnh phân phối similarity
  -> Kết hợp điểm bằng trọng số (late fusion)
  -> Sắp xếp giảm dần
  -> Trả về Top 5
```

Các module hiện thực:

| Giai đoạn | File |
|---|---|
| Tiền xử lý | `backend/app/services/preprocess.py` |
| HSV Histogram | `backend/app/services/features/hsv.py` |
| Color Moments | `backend/app/services/features/color_moments.py` |
| LBP | `backend/app/services/features/lbp.py` |
| GLCM | `backend/app/services/features/glcm.py` |
| HOG | `backend/app/services/features/hog.py` |
| Hu Moments | `backend/app/services/features/hu.py` |
| Học trọng số | `backend/scripts/learn_search_weights.py` |

## 3. Tiền xử lý ảnh

### 3.1. Resize

Mọi ảnh được đưa về kích thước `128 x 128`. Việc cố định kích thước giúp HOG
và các đặc trưng phụ thuộc không gian luôn có cùng số chiều.

Nếu ảnh bị resize trực tiếp về hình vuông, tỷ lệ khuôn mặt có thể bị biến dạng.
Phương án tốt hơn là giữ nguyên tỷ lệ, sau đó padding hoặc crop vùng mặt.

### 3.2. Gaussian blur

Lọc Gaussian được biểu diễn bởi phép tích chập:

$$
I'(x,y)=(G_\sigma * I)(x,y)
$$

Kernel `3 x 3` làm giảm nhiễu cao tần và nhiễu do nén ảnh. Tuy nhiên, làm mờ
quá mạnh sẽ làm mất cạnh nhỏ và chi tiết lông mà LBP/HOG cần sử dụng.

### 3.3. CLAHE

Ảnh được chuyển từ BGR sang LAB. CLAHE chỉ được áp dụng lên kênh độ sáng `L`,
sau đó ảnh được chuyển lại về BGR.

CLAHE chia ảnh thành các vùng nhỏ, cân bằng histogram cục bộ và giới hạn độ
khuếch đại bằng `clipLimit`. Mục đích là:

- Tăng chi tiết ở vùng quá tối hoặc quá sáng.
- Giảm ảnh hưởng của điều kiện chiếu sáng.
- Hạn chế khuếch đại nhiễu so với histogram equalization thông thường.

CLAHE có thể làm thay đổi phân bố độ sáng của HSV và Color Moments. Trong một
pipeline nâng cao, nhóm màu sắc và nhóm texture/shape nên có nhánh tiền xử lý
riêng.

## 4. Tổng quan bộ đặc trưng

| Đặc trưng | Nhóm | Số chiều | Thông tin chính |
|---|---|---:|---|
| HSV Histogram | Màu sắc | 768 | Phân bố kết hợp H-S-V |
| Color Moments | Màu sắc | 9 | Trung bình, độ lệch chuẩn, moment bậc ba |
| LBP | Kết cấu | 18 | Pattern sáng-tối cục bộ |
| GLCM | Kết cấu | 40 | Quan hệ mức xám theo khoảng cách và hướng |
| HOG | Hình dạng | 8100 | Phân bố hướng gradient theo vùng |
| Hu Moments | Hình dạng toàn cục | 7 | Moment gần bất biến với dịch, xoay, tỷ lệ |

## 5. HSV Histogram

### 5.1. Cơ sở lý thuyết

Không gian HSV tách thông tin màu thành:

- `H` (Hue): loại màu.
- `S` (Saturation): độ bão hòa.
- `V` (Value): độ sáng.

So với BGR/RGB, HSV tách sắc màu khỏi độ sáng tốt hơn và phù hợp với mô tả màu
lông trong các điều kiện chiếu sáng khác nhau.

### 5.2. Cách tính

Không gian HSV được lượng tử hóa:

```text
H: 12 bins
S: 8 bins
V: 8 bins
```

Mỗi pixel đóng góp vào một ô của histogram ba chiều:

$$
h(i,j,k)=\#\{p \mid H(p)\in B_i,\ S(p)\in B_j,\ V(p)\in B_k\}
$$

Số chiều của vector:

$$
12 \times 8 \times 8=768
$$

Sau khi flatten, vector được chuẩn hóa L2.

### 5.3. Giá trị thông tin

HSV Histogram hữu ích khi các loài có màu lông khác biệt rõ, chẳng hạn hổ cam,
báo đen hoặc mèo trắng.

### 5.4. Hạn chế

- Không lưu vị trí không gian của màu.
- Màu nền có thể ảnh hưởng tương đương màu khuôn mặt.
- Nhiều loài có màu lông giống nhau.
- Histogram intersection hoặc Chi-square có thể phù hợp với histogram hơn
  cosine similarity.

## 6. Color Moments

### 6.1. Cơ sở lý thuyết

Phân bố màu có thể được xấp xỉ bằng một số moment thống kê. Với mỗi kênh
HSV, hệ thống tính ba đại lượng.

Trung bình:

$$
\mu=\frac{1}{N}\sum_{i=1}^{N}x_i
$$

Độ lệch chuẩn:

$$
\sigma=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2}
$$

Moment bậc ba có bảo toàn dấu:

$$
m_3=\sqrt[3]{\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^3}
$$

Ba đại lượng trên ba kênh tạo vector:

$$
3\ \text{kênh}\times3\ \text{moment}=9\ \text{chiều}
$$

### 6.2. Giá trị thông tin

- Mean mô tả màu chủ đạo.
- Standard deviation mô tả độ đa dạng màu.
- Moment bậc ba mô tả mức bất đối xứng của phân bố.

Color Moments rất gọn và bổ sung một mô tả thống kê tổng quát cho HSV
Histogram.

### 6.3. Hạn chế

Hue là biến tuần hoàn: giá trị `0` và `179` trong OpenCV biểu diễn hai màu rất
gần nhau nhưng trung bình số học lại xem chúng ở xa. Về lý thuyết, Hue nên dùng
circular mean và circular variance.

Moment bậc ba trong implementation hiện tại không phải skewness chuẩn hóa:

$$
\gamma_1=\frac{E[(X-\mu)^3]}{\sigma^3}
$$

Do đó tài liệu và giao diện nên gọi đúng là signed cube-root third central
moment thay vì skewness chuẩn.

## 7. Local Binary Pattern

### 7.1. Cơ sở lý thuyết

LBP mô tả cấu trúc sáng-tối xung quanh một pixel. Với pixel trung tâm
$g_c$ và $P$ điểm lân cận trên bán kính $R$:

$$
LBP_{P,R}=\sum_{p=0}^{P-1}s(g_p-g_c)2^p
$$

trong đó:

$$
s(x)=
\begin{cases}
1,&x\geq0\\
0,&x<0
\end{cases}
$$

Pipeline sử dụng:

```text
P = 16
R = 2
method = uniform
```

Uniform LBP gom các pattern có không quá hai lần chuyển `0 <-> 1`. Histogram
kết quả có:

$$
P+2=18\ \text{bins}
$$

### 7.2. Giá trị thông tin

LBP mô tả:

- Lông mịn hoặc lông thô.
- Pattern sọc, đốm và vùng đồng nhất.
- Các biên nhỏ quanh mắt, mũi và lông mặt.

Vì chỉ so sánh thứ tự sáng-tối giữa các pixel, LBP tương đối bền với biến đổi
độ sáng đơn điệu.

### 7.3. Hạn chế

Histogram toàn ảnh làm mất vị trí của texture. Hai ảnh có texture giống nhau
nhưng nằm ở các vùng khác nhau vẫn có thể cho histogram gần giống.

Cải tiến phù hợp:

- Multi-scale LBP với nhiều cặp `(P,R)`.
- Chia ảnh thành lưới `2 x 2` hoặc `4 x 4`.
- Tính histogram riêng từng vùng rồi nối các vector.

## 8. Gray-Level Co-occurrence Matrix

### 8.1. Cơ sở lý thuyết

GLCM mô tả xác suất một pixel mức xám $i$ xuất hiện cùng một pixel mức xám
$j$ tại khoảng cách $d$ và góc $\theta$:

$$
P_{d,\theta}(i,j)
$$

Ảnh grayscale được lượng tử hóa từ 256 xuống 32 mức để giảm nhiễu và chi phí
tính toán.

Pipeline sử dụng:

```text
Khoảng cách d: 1 và 3 pixel
Góc theta: 0, 45, 90 và 135 độ
```

### 8.2. Các thuộc tính Haralick

Contrast:

$$
\operatorname{Contrast}=\sum_{i,j}(i-j)^2P(i,j)
$$

Giá trị cao khi các pixel lân cận có độ chênh sáng lớn.

Dissimilarity:

$$
\operatorname{Dissimilarity}=\sum_{i,j}|i-j|P(i,j)
$$

Tương tự contrast nhưng tăng tuyến tính theo chênh lệch mức xám.

Homogeneity:

$$
\operatorname{Homogeneity}
=\sum_{i,j}\frac{P(i,j)}{1+(i-j)^2}
$$

Giá trị cao khi các pixel lân cận có mức xám gần nhau.

Energy:

$$
\operatorname{Energy}=\sum_{i,j}P(i,j)^2
$$

Giá trị cao khi texture đều và chỉ có một số quan hệ mức xám chiếm ưu thế.

Correlation:

$$
\operatorname{Correlation}
=\sum_{i,j}
\frac{(i-\mu_i)(j-\mu_j)P(i,j)}
{\sigma_i\sigma_j}
$$

Đại lượng này đo mức phụ thuộc tuyến tính giữa các pixel lân cận.

Số chiều:

$$
5\ \text{thuộc tính}
\times2\ \text{khoảng cách}
\times4\ \text{góc}
=40
$$

### 8.3. Giá trị thông tin

GLCM bổ sung cho LBP:

- LBP mô tả pattern rất cục bộ quanh từng pixel.
- GLCM mô tả quan hệ thống kê giữa các cặp mức xám theo hướng và khoảng cách.

### 8.4. Hạn chế

Các property có miền giá trị khác nhau trước khi chuẩn hóa. Vector GLCM toàn
ảnh cũng không biểu diễn vị trí texture. Có thể cải thiện bằng spatial GLCM
hoặc chuẩn hóa từng nhóm property trước khi ghép.

## 9. Histogram of Oriented Gradients

### 9.1. Cơ sở lý thuyết

HOG mô tả phân bố hướng cạnh. Trước hết tính gradient:

$$
G_x=\frac{\partial I}{\partial x},\qquad
G_y=\frac{\partial I}{\partial y}
$$

Độ lớn và hướng gradient:

$$
m=\sqrt{G_x^2+G_y^2}
$$

$$
\theta=\operatorname{atan2}(G_y,G_x)
$$

Mỗi pixel bỏ phiếu với trọng số $m$ vào histogram hướng tương ứng.

### 9.2. Cấu hình

```text
Ảnh đầu vào: 128 x 128 grayscale
Cell: 8 x 8
Block: 16 x 16 = 2 x 2 cells
Block stride: 8
Số orientation bins: 9
```

Số vị trí block trên mỗi chiều:

$$
\frac{128-16}{8}+1=15
$$

Số chiều HOG:

$$
15\times15\times4\times9=8100
$$

Các histogram trong block được chuẩn hóa để giảm ảnh hưởng của độ sáng và độ
tương phản.

### 9.3. Giá trị thông tin

HOG biểu diễn:

- Đường viền tổng thể của khuôn mặt.
- Hình dạng tai, mắt và mõm.
- Cấu trúc cạnh tại từng vùng trên ảnh.

Đây là đặc trưng đơn có kết quả tốt nhất trong các thí nghiệm ban đầu của dự
án.

### 9.4. Hạn chế

HOG phụ thuộc mạnh vào vị trí. Nếu mặt động vật không được crop và alignment
thống nhất, cùng một cell có thể mô tả tai ở ảnh này nhưng lại mô tả nền ở ảnh
khác.

Global L2 normalization sau HOG chỉ nhân toàn vector với một hệ số chung. Nó
không thay đổi tỷ lệ tương đối giữa các block và không thay đổi cosine
similarity.

## 10. Hu Moments

### 10.1. Moment không gian

Moment không gian bậc $(p,q)$:

$$
m_{pq}=\sum_x\sum_yx^py^qI(x,y)
$$

Tâm ảnh được xác định bởi:

$$
\bar{x}=\frac{m_{10}}{m_{00}},\qquad
\bar{y}=\frac{m_{01}}{m_{00}}
$$

Central moment:

$$
\mu_{pq}=\sum_x\sum_y
(x-\bar{x})^p(y-\bar{y})^qI(x,y)
$$

Normalized central moment:

$$
\eta_{pq}=
\frac{\mu_{pq}}
{\mu_{00}^{1+\frac{p+q}{2}}}
$$

Hu xây dựng bảy tổ hợp phi tuyến từ các $\eta_{pq}$. Các tổ hợp này gần bất
biến với tịnh tiến, thay đổi tỷ lệ và xoay.

### 10.2. Log transform

Hu Moments có thể trải qua nhiều bậc độ lớn. Hệ thống áp dụng:

$$
h'_i=-\operatorname{sign}(h_i)
\log_{10}(|h_i|+\epsilon)
$$

Sau biến đổi, vector bảy chiều được chuẩn hóa L2.

### 10.3. Hạn chế

Implementation hiện tại tính moment trực tiếp trên ảnh grayscale. Vì vậy Hu
không chỉ mô tả silhouette của khuôn mặt mà còn bị chi phối bởi phân bố cường
độ sáng và background.

Để Hu có ý nghĩa hình dạng rõ hơn, nên tính trên:

- Mask khuôn mặt đã tách nền.
- Ảnh nhị phân từ Otsu threshold.
- Edge map hoặc contour chính.

## 11. Chuẩn hóa vector đặc trưng

Mỗi vector được chuẩn hóa L2 riêng:

$$
\hat{x}_f=\frac{x_f}{\|x_f\|_2}
$$

Chuẩn hóa riêng là cần thiết vì các vector có số chiều rất khác nhau, từ Hu
7 chiều đến HOG 8100 chiều. Nếu nối tất cả vector rồi mới chuẩn hóa, nhóm có
số chiều lớn có thể chi phối kết quả.

## 12. Cosine similarity

Với query $q$ và ảnh cơ sở dữ liệu $d$, cosine similarity của đặc trưng $f$:

$$
s_f(q,d)=
\frac{x_{q,f}^{T}x_{d,f}}
{\|x_{q,f}\|_2\|x_{d,f}\|_2}
$$

Do vector đã được L2-normalize:

$$
s_f(q,d)=\hat{x}_{q,f}^{T}\hat{x}_{d,f}
$$

Cosine đo sự giống nhau về hướng của hai vector, không bị ảnh hưởng bởi độ lớn
ban đầu của vector.

## 13. Hiệu chỉnh phân phối similarity

Các đặc trưng có phân phối cosine khác nhau. Ví dụ GLCM và LBP có thể cho
cosine rất cao trên phần lớn cặp ảnh, trong khi HSV và HOG có miền phân tán
rộng hơn. Vì vậy không nên cộng trực tiếp raw cosine mà chưa calibration.

Với mỗi đặc trưng, mean và standard deviation được ước lượng trên tập train:

$$
z_f(q,d)=
\frac{s_f(q,d)-\mu_f}
{\sigma_f+\epsilon}
$$

Mọi tham số $\mu_f,\sigma_f$ phải được tính chỉ từ train fold. Nếu dùng toàn
bộ dữ liệu trước khi cross-validation, thông tin từ test sẽ bị rò rỉ.

## 14. Late fusion

Điểm cuối cùng là tổ hợp tuyến tính của sáu similarity đã hiệu chỉnh:

$$
S(q,d)=\sum_{f=1}^{6}w_fz_f(q,d)
$$

Late fusion phù hợp vì:

- Các đặc trưng khác nhau về số chiều và bản chất.
- Có thể kiểm soát đóng góp của từng đặc trưng.
- Có thể giải thích kết quả bằng per-feature similarity.
- Có thể re-rank mà không cần trích xuất lại vector.

## 15. Học trọng số bằng pairwise Learning-to-Rank

### 15.1. Xây dựng cặp huấn luyện

Với mỗi query $q$:

- $p$ là ảnh relevant, có cùng `animal_type`.
- $n$ là ảnh irrelevant, có `animal_type` khác.

Tạo vector chênh lệch:

$$
\Delta z=z(q,p)-z(q,n)
$$

Mục tiêu là:

$$
w^Tz(q,p)>w^Tz(q,n)
$$

hay tương đương:

$$
w^T\Delta z>0
$$

Ngoài negative ngẫu nhiên, pipeline chọn hard negative: ảnh khác lớp nhưng có
similarity cao, tức là những trường hợp hệ thống hiện tại dễ xếp sai.

### 15.2. RankNet logistic loss

Xác suất ảnh positive được xếp trên ảnh negative:

$$
P(p\succ n)=\sigma(w^T\Delta z)
=\frac{1}{1+\exp(-w^T\Delta z)}
$$

Hàm mất mát:

$$
\mathcal{L}(w)=
\frac{1}{M}
\sum_{i=1}^{M}
\log(1+\exp(-w^T\Delta z_i))
+\lambda\|w\|_2^2
$$

Thành phần logistic phạt các cặp bị xếp sai. Thành phần L2 regularization hạn
chế trọng số quá cực đoan và tăng độ ổn định.

### 15.3. Ràng buộc tất cả đặc trưng phải tham gia

Trọng số được tối ưu trên bounded simplex:

$$
\sum_{f=1}^{6}w_f=1
$$

$$
w_f\geq\delta>0
$$

Như vậy mọi đặc trưng đều có đóng góp, nhưng phần trọng số còn lại vẫn được
phân bổ dựa trên khả năng xếp hạng thực nghiệm.

Cả $\lambda$ và $\delta$ được chọn bằng inner cross-validation thay vì đặt
theo cảm tính.

## 16. Nested cross-validation

Quy trình đánh giá gồm hai vòng:

1. Outer cross-validation chia dữ liệu thành train và test để ước lượng khả
   năng tổng quát hóa.
2. Inner cross-validation trên outer-train chọn $\lambda$ và $\delta$.
3. Học trọng số trên toàn bộ outer-train với tham số đã chọn.
4. Đánh giá trên outer-test chưa từng tham gia chọn trọng số.
5. Lặp lại cho mọi outer fold và lấy trung bình.

Nested cross-validation tránh việc dùng cùng dữ liệu để vừa chọn mô hình vừa
báo cáo chất lượng, vốn tạo ra optimistic selection bias.

## 17. Kết quả thực nghiệm ban đầu

Thí nghiệm được chạy trên 240 ảnh AFHQ cân bằng:

```text
cat:  80 ảnh
dog:  80 ảnh
wild: 80 ảnh
```

Nested cross-validation chọn:

```text
lambda = 0.1
delta  = 0.05
```

Trọng số trong không gian similarity đã chuẩn hóa:

```python
STANDARDIZED_WEIGHTS = {
    "hsv":  0.050000,
    "cm":   0.050000,
    "lbp":  0.378633,
    "glcm": 0.096609,
    "hog":  0.374759,
    "hu":   0.050000,
}
```

Kết quả outer cross-validation:

| Phương pháp | Macro-P@5 | Macro-MAP@10 |
|---|---:|---:|
| Trọng số được học | **0.6983** | **0.5662** |
| Trọng số thủ công hiện tại | 0.6608 | 0.5233 |
| HOG-only | 0.5350 | 0.4216 |
| Trọng số bằng nhau | 0.4383 | 0.2728 |

Các hệ số trên chỉ có giá trị đối với tập dữ liệu, extractor version và cách
tiền xử lý đã dùng. Khi thay đổi dữ liệu hoặc thuật toán trích xuất, cần học
lại trọng số.

## 18. Chỉ số đánh giá

Precision@5:

$$
P@5=\frac{\text{số ảnh relevant trong Top 5}}{5}
$$

Precision tại vị trí $k$:

$$
P@k=\frac{\text{số ảnh relevant trong Top }k}{k}
$$

Average Precision@K:

$$
AP@K=
\frac{
\sum_{k=1}^{K}P@k\cdot rel(k)
}{
\min(K,\text{tổng số ảnh relevant})
}
$$

Mean Average Precision:

$$
MAP@K=\frac{1}{|Q|}\sum_{q\in Q}AP_q@K
$$

Nên dùng Macro-MAP để mỗi lớp động vật có đóng góp ngang nhau, tránh lớp nhiều
ảnh chi phối kết quả.

## 19. Vai trò của metadata

Metadata `animal_type` có ba cách sử dụng:

1. Làm ground truth để học trọng số.
2. Làm ground truth để tính Precision và MAP.
3. Làm bộ lọc tùy chọn nếu người dùng chủ động chọn loại động vật.

Không được lấy ground-truth `animal_type` của query test để tự động lọc
candidate trước khi đánh giá content-only retrieval. Cách làm đó gây data
leakage và khiến P@5 theo loài mất ý nghĩa.

Hệ thống nên công bố hai chế độ:

- Content-only: đánh giá chất lượng đặc trưng.
- Metadata-filtered: chức năng tìm kiếm kết hợp dành cho người dùng.

## 20. Các giới hạn cần nêu trong báo cáo

- Dataset AFHQ thử nghiệm ban đầu chỉ có ba nhãn rộng.
- Nhãn cùng loài chỉ thể hiện semantic relevance, không bảo đảm hai cá thể có
  ngoại hình thực sự giống nhau.
- Các đặc trưng hiện tại chủ yếu là global hoặc fixed-grid.
- Ảnh chưa có bước phát hiện và alignment khuôn mặt.
- Color Moments xử lý Hue như biến tuyến tính.
- Hu Moments được tính trên grayscale thay vì silhouette.
- Production ANN phải tính lại đầy đủ sáu similarity cho candidate trước khi
  áp dụng trọng số đã học.

## 21. Hướng cải tiến

### 21.1. Giữ tỷ lệ ảnh khi resize và padding về hình vuông

#### Vấn đề hiện tại

Hàm `_resize` đang biến mọi ảnh trực tiếp thành `128 x 128`. Với ảnh gốc có
kích thước `W x H`, hệ số co giãn theo hai trục là:

$$
s_x=\frac{128}{W},\qquad s_y=\frac{128}{H}
$$

Nếu `s_x` khác `s_y`, hình học của khuôn mặt bị biến dạng. Mặt có thể bị kéo
dài hoặc ép dẹt. Sai lệch này ảnh hưởng mạnh đến:

- HOG vì hướng và vị trí cạnh bị thay đổi.
- Hu Moments vì hình dạng phân bố điểm ảnh bị thay đổi.
- LBP và GLCM vì cấu trúc texture bị co giãn không đồng đều.

HSV Histogram ít nhạy hơn với biến dạng hình học, nhưng tỷ lệ diện tích giữa
động vật và nền vẫn có thể thay đổi.

#### Cách cải tiến

Dùng một hệ số scale duy nhất:

$$
s=\min\left(\frac{128}{W},\frac{128}{H}\right)
$$

Ảnh sau resize có kích thước:

$$
W'=\lfloor sW\rfloor,\qquad H'=\lfloor sH\rfloor
$$

Sau đó đặt ảnh vào giữa canvas `128 x 128`; phần còn thiếu được padding. Đây
thường được gọi là letterbox resize.

Không nên mặc định padding bằng màu đen nếu Hu, HSV hoặc GLCM được tính trên
toàn ảnh, vì vùng đen sẽ trở thành một tín hiệu giả. Có ba lựa chọn:

- Padding phản xạ (`BORDER_REFLECT`) để giảm đường biên nhân tạo.
- Padding bằng màu trung vị của ảnh.
- Tạo mask vùng ảnh thật và loại vùng padding khỏi các đặc trưng hỗ trợ mask.

#### Cách đánh giá

Giữ nguyên tập train/test và trọng số, chỉ thay phương pháp resize. So sánh
`Macro-P@5` và `Macro-MAP@10` giữa:

1. Resize trực tiếp `128 x 128`.
2. Letterbox với padding phản xạ.
3. Letterbox với mask vùng hợp lệ.

Sau khi đổi tiền xử lý, phải trích xuất lại toàn bộ đặc trưng và xây lại index;
không được so query đã letterbox với corpus được resize theo cách cũ.

### 21.2. Phát hiện, crop và alignment khuôn mặt

#### Vấn đề hiện tại

Các đặc trưng đang được tính trên toàn ảnh. Vì vậy chúng mô tả cả nền, thân,
vật thể phụ và điều kiện chụp. Hai ảnh cùng loài nhưng khác nền có thể nhận
điểm thấp; hai ảnh khác loài nhưng cùng nền có thể nhận điểm cao.

Đặc biệt:

- HSV và Color Moments dễ học màu nền.
- LBP và GLCM có thể mô tả cỏ, tường hoặc vải thay vì lông.
- HOG có thể tập trung vào đường biên của nền.
- Hu Moments trên toàn ảnh không đại diện riêng cho hình dạng khuôn mặt.

#### Pipeline đề xuất

```text
Ảnh gốc
  -> detector tìm bounding box khuôn mặt
  -> mở rộng bounding box thêm một biên nhỏ
  -> crop vùng mặt
  -> tìm landmark như hai mắt, mũi hoặc tâm tai
  -> ước lượng phép biến đổi similarity
  -> xoay và scale về một tư thế chuẩn
  -> letterbox về 128 x 128
  -> trích xuất đặc trưng
```

Phép biến đổi similarity gồm tịnh tiến, quay và scale đồng đều:

$$
\mathbf{x}'=sR\mathbf{x}+\mathbf{t}
$$

Nó chuẩn hóa vị trí và góc của khuôn mặt mà không làm méo tỷ lệ hình học như
phép affine tự do.

#### Lợi ích và rủi ro

Crop làm tăng tỷ lệ pixel thật sự thuộc về đối tượng. Alignment giúp các vùng
tương ứng như mắt, trán và mõm nằm gần cùng vị trí, do đó HOG và spatial
pyramid có ý nghĩa hơn.

Tuy nhiên, detector động vật có thể hoạt động không đồng đều giữa các loài.
Một detector chỉ tốt trên chó và mèo sẽ tạo thiên lệch khi thêm loài mới. Hệ
thống cần lưu `detection_confidence` và có fallback dùng ảnh gốc khi không tìm
được mặt. Nên đánh giá riêng ba cấu hình:

- Không crop.
- Chỉ crop.
- Crop và alignment.

### 21.3. Dùng spatial pyramid cho HSV, LBP và GLCM

#### Vấn đề của đặc trưng global

Histogram toàn ảnh có tính bất biến vị trí nhưng làm mất bố cục không gian.
Ví dụ, hai ảnh có cùng tỷ lệ vùng trắng và đen sẽ có HSV Histogram gần giống
nhau, dù ảnh thứ nhất có vùng trắng quanh mõm còn ảnh thứ hai có vùng trắng ở
nền.

#### Nguyên lý spatial pyramid

Spatial pyramid chia ảnh ở nhiều mức:

- Mức 0: `1 x 1`, gồm 1 vùng toàn ảnh.
- Mức 1: `2 x 2`, gồm 4 vùng.
- Mức 2: `4 x 4`, gồm 16 vùng.

Đặc trưng được tính riêng trong từng vùng rồi nối lại:

$$
\Phi(I)=
\left[
\alpha_0\Phi_{1\times1}(I),
\alpha_1\Phi_{2\times2}(I),
\alpha_2\Phi_{4\times4}(I)
\right]
$$

Trong đó $\alpha_l$ là trọng số của mức $l$. Có thể cho mức thô trọng số lớn
hơn để tránh các ô nhỏ quá nhạy với dịch chuyển, sau đó chuẩn hóa L2 vector
cuối cùng.

Với ba mức trên, tổng số vùng là:

$$
1+4+16=21
$$

Số chiều tương ứng sẽ tăng:

| Đặc trưng | Hiện tại | Pyramid 3 mức |
|---|---:|---:|
| HSV | 768 | 16,128 |
| LBP | 18 | 378 |
| GLCM | 40 | 840 |

#### Khuyến nghị cho dự án

Không nên áp dụng ngay ba mức cho mọi đặc trưng. Với ảnh `128 x 128`, ô ở mức
`4 x 4` chỉ còn `32 x 32`; GLCM 32 mức xám trên ô nhỏ có thể thưa và không ổn
định. Nên thử:

- HSV và LBP: mức `1 x 1`, `2 x 2`, `4 x 4`.
- GLCM: trước hết chỉ dùng `1 x 1` và `2 x 2`.
- Giảm số mức xám GLCM trong ô nhỏ nếu cần.

Spatial pyramid chỉ thực sự hiệu quả khi khuôn mặt đã được crop và alignment.
Nếu vị trí mặt thay đổi tự do, cùng một bộ phận có thể rơi vào các ô khác nhau
và làm similarity giảm.

### 21.4. Dùng circular statistics cho Hue

#### Vì sao Hue không phải biến tuyến tính?

Trong OpenCV, Hue nằm trong khoảng `[0, 179]`, nhưng hai đầu của khoảng biểu
diễn các màu nằm cạnh nhau trên vòng màu. Hue `1` và Hue `179` gần như cùng
màu đỏ, trong khi trung bình số học cho kết quả:

$$
\frac{1+179}{2}=90
$$

Hue `90` lại thuộc vùng màu hoàn toàn khác. Do đó mean, standard deviation và
skewness tuyến tính hiện dùng trong Color Moments không đúng về mặt hình học.

#### Cách tính thống kê vòng tròn

Đổi Hue $H_i$ thành góc:

$$
\theta_i=2\pi\frac{H_i}{180}
$$

Tính trung bình của các vector đơn vị:

$$
C=\frac{1}{N}\sum_i\cos\theta_i,\qquad
S=\frac{1}{N}\sum_i\sin\theta_i
$$

Circular mean là:

$$
\bar{\theta}=\operatorname{atan2}(S,C)
$$

Độ tập trung:

$$
R=\sqrt{C^2+S^2}
$$

và circular variance:

$$
V_c=1-R
$$

`R` gần 1 nghĩa là Hue tập trung quanh một màu; `R` gần 0 nghĩa là màu phân
tán hoặc các hướng màu triệt tiêu nhau.

#### Áp dụng

Trong Color Moments, thay ba thống kê tuyến tính của kênh H bằng các đại lượng
vòng tròn, chẳng hạn `sin(mean_angle)`, `cos(mean_angle)` và `1 - R`. Việc lưu
cặp sin/cos tránh điểm gián đoạn tại biên `0/179`. Các thống kê của S và V vẫn
có thể giữ dạng tuyến tính.

HSV Histogram đã xử lý Hue bằng các bin nhưng vẫn có đường biên giữa bin đầu
và bin cuối. Khi thiết kế metric, có thể dùng histogram có bin Hue tuần hoàn
hoặc khoảng cách cho phép dịch vòng theo trục Hue.

### 21.5. Tính Hu Moments trên mask hoặc contour khuôn mặt

#### Vấn đề hiện tại

Mã hiện tại gọi `cv2.moments(gray)`, tức xem cường độ grayscale như khối lượng.
Khi đó Hu Moments chịu ảnh hưởng bởi:

- Ánh sáng và bóng đổ trên mặt.
- Texture lông.
- Màu và cấu trúc nền.
- Vùng crop chứa nhiều hay ít background.

Điều này khác với mục tiêu thông thường của Hu Moments là mô tả hình dạng của
một miền nhị phân hoặc contour.

#### Pipeline đề xuất

```text
Ảnh đã crop
  -> semantic segmentation hoặc foreground segmentation
  -> mask nhị phân của đầu/khuôn mặt
  -> làm sạch mask bằng morphology
  -> giữ connected component chính
  -> lấy contour ngoài
  -> cv2.moments(mask hoặc contour)
  -> cv2.HuMoments
  -> biến đổi log và chuẩn hóa
```

Hu Moments được xây từ normalized central moments nên về lý thuyết bất biến
với tịnh tiến và scale; tổ hợp Hu còn được thiết kế để bất biến với quay.
Những tính chất này phù hợp hơn khi đầu vào là một silhouette ổn định.

#### Rủi ro và cách kiểm chứng

Hu rất nhạy với lỗi nhỏ ở contour. Tai bị cắt, lông xù hoặc mask thủng có thể
làm moment bậc cao thay đổi mạnh. Cần lưu confidence/chất lượng mask và có
fallback. So sánh ba biến thể:

1. Hu trên grayscale hiện tại.
2. Hu trên mask khuôn mặt.
3. Hu trên contour ngoài lớn nhất.

Query và toàn bộ corpus phải dùng cùng một mô hình segmentation và cùng quy
tắc hậu xử lý.

### 21.6. Thử Chi-square hoặc histogram intersection cho HSV/LBP

#### Vì sao cosine chưa chắc là metric tốt nhất?

Cosine đo góc giữa hai vector:

$$
s_{\cos}(p,q)=\frac{p^\top q}{\|p\|_2\|q\|_2}
$$

Metric này dùng được cho mọi vector nhưng không tận dụng đầy đủ bản chất của
histogram. HSV và LBP là phân phối tần suất không âm; sự trùng khớp hoặc chênh
lệch theo từng bin thường có ý nghĩa trực tiếp hơn góc vector.

#### Chi-square

Khoảng cách Chi-square đối xứng:

$$
d_{\chi^2}(p,q)=
\frac{1}{2}\sum_i
\frac{(p_i-q_i)^2}{p_i+q_i+\epsilon}
$$

Mẫu số làm cho sai lệch ở các bin hiếm được xem xét tương đối theo khối lượng
của bin. Vì đây là khoảng cách, có thể đổi thành similarity:

$$
s_{\chi^2}(p,q)=
\exp\left(-\frac{d_{\chi^2}(p,q)}{\tau}\right)
$$

Tham số $\tau$ phải được ước lượng chỉ từ tập train hoặc chọn bằng inner
cross-validation.

#### Histogram intersection

Với histogram đã chuẩn hóa L1:

$$
s_{\cap}(p,q)=\sum_i\min(p_i,q_i)
$$

Giá trị nằm trong `[0,1]`; nó đo trực tiếp phần khối lượng mà hai histogram
chia sẻ. Metric này đơn giản, nhanh và thường dễ diễn giải.

#### Cách chọn metric khoa học

Tạo các cấu hình `cosine`, `chi-square` và `intersection` cho HSV/LBP. Chọn
metric bằng nested cross-validation cùng lúc với trọng số. Tuyệt đối không
thử nhiều metric trên test rồi chọn metric có điểm test cao nhất, vì đó vẫn là
overfitting vào test.

Không nên mặc định dùng Chi-square cho Color Moments, GLCM, HOG hoặc Hu vì các
vector này không phải đều là histogram xác suất không âm. Mỗi loại đặc trưng
có thể sử dụng metric riêng trước bước hiệu chỉnh điểm và late fusion.

### 21.7. Đánh giá trên test độc lập và nhiều lớp động vật hơn

#### Hạn chế của thí nghiệm hiện tại

AFHQ thử nghiệm ban đầu chỉ có ba lớp rộng. Một mô hình phân biệt chó, mèo và
động vật hoang dã chưa chứng minh rằng nó tìm tốt giữa các giống gần nhau hoặc
giữa các cá thể cùng loài.

Nếu cùng một tập dữ liệu được dùng nhiều lần để sửa đặc trưng, chọn metric và
chọn trọng số, kết quả cuối cùng sẽ dần thích nghi với chính tập đó dù có
cross-validation.

#### Thiết kế dữ liệu đề xuất

Chia dữ liệu thành ba phần có vai trò rõ ràng:

- Train: học tham số và trọng số.
- Validation hoặc inner folds: chọn metric, $\lambda$, $\delta$ và cấu hình
  đặc trưng.
- Test độc lập: chỉ dùng một lần để báo cáo kết quả cuối.

Nếu có thông tin cá thể hoặc nguồn ảnh, nên dùng group split để ảnh của cùng
một cá thể, cùng video hoặc các bản gần trùng không xuất hiện ở cả train và
test. Nếu không, hệ thống có thể ghi nhớ ảnh thay vì học sự tương đồng.

Dataset nên có nhiều cấp nhãn:

- Loài: chó, mèo, hổ, sư tử, cáo.
- Giống hoặc nhóm con.
- Cá thể, nếu bài toán cần tìm đúng cá thể.

Định nghĩa relevance phải khớp mục tiêu. “Cùng loài” không tương đương “ngoại
hình giống nhau”. Nên báo cáo Macro-MAP theo lớp và điểm riêng từng lớp để
phát hiện hệ thống chỉ hoạt động tốt trên lớp đông hoặc dễ.

Metadata của query test không được dùng để lọc candidate trong thí nghiệm
content-only. Chế độ metadata-filtered phải được đánh giá và công bố riêng.

### 21.8. Báo cáo độ lệch chuẩn và khoảng tin cậy của trọng số

#### Tại sao một vector trọng số là chưa đủ?

Một lần huấn luyện chỉ cho một ước lượng điểm. Nếu thay đổi fold mà trọng số
HSV chuyển từ `0.05` thành `0.30`, chưa thể khẳng định hệ số tìm được là ổn
định. Các đặc trưng tương quan như LBP và GLCM có thể thay thế lẫn nhau: trọng
số của từng đặc trưng dao động dù chất lượng xếp hạng gần như không đổi.

Với $K$ outer folds và trọng số $w_f^{(k)}$ của đặc trưng $f$, báo cáo:

$$
\bar{w}_f=\frac{1}{K}\sum_{k=1}^{K}w_f^{(k)}
$$

$$
s_f=
\sqrt{
\frac{1}{K-1}
\sum_{k=1}^{K}
\left(w_f^{(k)}-\bar{w}_f\right)^2
}
$$

Bảng kết quả tối thiểu nên có:

| Feature | Mean weight | Standard deviation | 95% CI | Số fold chạm cận dưới |
|---|---:|---:|---:|---:|
| HSV | ... | ... | ... | ... |
| Color Moments | ... | ... | ... | ... |
| LBP | ... | ... | ... | ... |
| GLCM | ... | ... | ... | ... |
| HOG | ... | ... | ... | ... |
| Hu | ... | ... | ... | ... |

“Số fold chạm cận dưới” cho biết một đặc trưng có được giữ lại chủ yếu vì ràng
buộc $w_f\ge\delta$ hay thực sự được dữ liệu cấp trọng số lớn hơn.

#### Khoảng tin cậy

Các fold cross-validation không hoàn toàn độc lập vì tập train chồng lấn.
Do đó khoảng tin cậy t-test trực tiếp trên vài fold chỉ nên xem là mô tả gần
đúng. Cần dùng hai quy trình khác nhau:

1. Với trọng số: dùng repeated stratified group cross-validation hoặc
   bootstrap các group trong tập train, sau đó huấn luyện lại mô hình ở mỗi
   lần lấy mẫu.
2. Với metric: cố định mô hình, bootstrap các query cùng prediction của chúng
   trong outer-test hoặc test độc lập, rồi tính lại `Macro-MAP@K`.
3. Với mỗi đại lượng, lấy percentile `2.5%` và `97.5%` của phân phối thực
   nghiệm làm khoảng tin cậy 95%.

Cần phân biệt hai loại bất định:

- Khoảng tin cậy của metric: hệ thống tìm kiếm tốt đến mức nào.
- Độ ổn định của trọng số: cách phân bổ đóng góp giữa các đặc trưng có bền
  vững hay không.

Nếu trọng số dao động mạnh nhưng MAP ổn định, nguyên nhân thường là các đặc
trưng dư thừa hoặc tương quan. Khi đó nên báo cáo thêm ma trận tương quan giữa
similarity của sáu đặc trưng, không nên diễn giải từng trọng số như quan hệ
nhân quả.

### 21.9. Thứ tự triển khai khuyến nghị

Để đo được đóng góp của từng thay đổi, nên triển khai theo thứ tự:

1. Tạo tập test độc lập và cố định protocol đánh giá.
2. Đổi resize sang giữ tỷ lệ và chạy ablation.
3. Thêm crop, sau đó mới thêm alignment.
4. Sửa circular Hue và Hu trên mask.
5. Thử metric riêng cho histogram.
6. Thêm spatial pyramid sau khi vị trí khuôn mặt đã ổn định.
7. Học lại trọng số bằng nested cross-validation.
8. Báo cáo metric, độ lệch chuẩn và khoảng tin cậy.

Mỗi bước chỉ nên thay một nhóm yếu tố và phải lưu kết quả baseline. Nếu thay
đồng thời preprocessing, đặc trưng, metric và trọng số thì không thể kết luận
phần nào thực sự làm chất lượng tìm kiếm tăng.

## 22. Tài liệu tham khảo

1. N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human
   Detection," CVPR, 2005.
2. T. Ojala, M. Pietikäinen and T. Mäenpää, "Multiresolution Gray-Scale and
   Rotation Invariant Texture Classification with Local Binary Patterns,"
   IEEE TPAMI, 2002.
3. R. M. Haralick, K. Shanmugam and I. Dinstein, "Textural Features for Image
   Classification," IEEE Transactions on Systems, Man, and Cybernetics, 1973.
4. M. K. Hu, "Visual Pattern Recognition by Moment Invariants," IRE
   Transactions on Information Theory, 1962.
5. C. Burges et al., "Learning to Rank using Gradient Descent," ICML, 2005:
   <https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf>
6. A. Binder et al., "Insights from Classifying Visual Concepts with Multiple
   Kernel Learning," 2011: <https://arxiv.org/abs/1112.3697>
7. S. Varma and R. Simon, "Bias in Error Estimation When Using
   Cross-Validation for Model Selection," BMC Bioinformatics, 2006:
   <https://link.springer.com/article/10.1186/1471-2105-7-91>
8. S. Lazebnik, C. Schmid and J. Ponce, "Beyond Bags of Features: Spatial
   Pyramid Matching for Recognizing Natural Scene Categories," CVPR, 2006.
9. S. R. Jammalamadaka and A. SenGupta, "Topics in Circular Statistics,"
   World Scientific, 2001.
10. M. J. Swain and D. H. Ballard, "Color Indexing," International Journal
    of Computer Vision, 1991.
11. B. Efron and R. J. Tibshirani, "An Introduction to the Bootstrap,"
    Chapman and Hall/CRC, 1993.
