# Face_Recognition (nhận dạng khuôn mặt) và Face verification (Xác minh khuôn mặt)
- Xác minh khuôn mặt:
    - Input: Ta đưa đầu vào là hình ảnh của một người, tên/id của người đó.
    - Output: Khi chúng ta đưa 1 ảnh bất kỳ thì xem bức ảnh đó có phải là người ta đã xem ở input hay không?
- Nhận dạng khuôn mặt:
    - Chúng ta có dữ liệu khuôn mặt của K người (ví dụ công ty có 100 người)
    - Lấy 1 bức ảnh
    - Xem bức ảnh mà ta có liệu có phải là 1 người trong K người (trong công ty) hay ko?
- **Nhận dạng khuôn mặt khó hơn xác minh khuôn mặt. Vì sao?**
    - Ví dụ ta có mô hình xác minh khuôn mặt với độ chính xác 99% -> liệu đã ok?
    - Với độ chính xác 99% thì xác minh khuôn mặt ở công ty 100 người sẽ sai 1 người --> Có lẽ chưa phải là độ chính xác bạn mong muốn, có lẽ 99,9% phù hợp hơn!
# One-shot learning
-  Liệu ta có thể nhận ra 1 người chỉ với 1 bức ảnh của họ? One-shot learning thực hiện ra sao?
- Ví dụ công ty bạn có 100 người, và với 1 hình ảnh bạn có, bạn nghĩ nên áp dụng mô hình sau để xác định bức ảnh là 1 ai đó trong công ty: ảnh --> ConvNet --> y(softmax 101 - là ai trong công ty và không phải). Có thể đây là mô hình tốt, nhưng nếu công ty bạn có thêm người, hoặc có người nghỉ việc, khi đó bạn phải đào tạo lại mô hình? có cách nào hiệu quả hơn không?
- **Learning a 'similarity' fuction (hàm tương tự giữa 2 ảnh)**:
    - ta có: d(img1, img2) = biểu thị mức độ khác nhau giữa 2 hình ảnh.
    - ta xác định siêu tham số t: khi đó nếu d =< t, ta nói 2 hình ảnh chỉ 1 người; nếu d > t, ta nói 2 hình ảnh không chỉ 1 người.
- Khi đó nếu xác định bức ảnh có phải 1 người trong 100 người (hay 105 người), ta chỉ cần xem 100 (hoặc 105) công thức d xem có trường hợp nào thỏa mãn =< t hay không!
# Siamese Network:
- Nội dung: [DeepFace](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
- Siamese Network (deepface) sẽ có dang như dưới. Mô hình sẽ mã hóa bức ảnh xi --> f(xi) là vecto 128 chiều; khi đó d(xi, xj) = np.linalg.norm(f(xi) - f(xj)) **2. Việc học tập các tham số được thực hiện, sao cho với xi, xj chỉ cùng 1 người thì d nhỏ và ngược lại.
<img src = 'https://i.imgur.com/hSv2Mqi.png'>

# Triplet Loss
- Nội dung: [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)
- Làm thế nào để đào tạo tham số cho mô hình Face Recognition?
    - Ta sử dụng đầu vào là 3 bức ảnh: ảnh đối tượng (Anchor); 1 ảnh khác chỉ cùng đối tượng (Positive); 1 ảnh không phải đối tượng (Negative), khi đó ta có: **d(A,P) - d(A,N) + anpha =< 0 (1)** ***;Ta thêm anpha để đảm bảo không xảy ra trường hợp khi f(x) là vecto 0, bởi vì nếu không có anpha khi f(x) là vecto 0 sẽ thỏa mãn phương trình (1)!***
    <img src ='https://i.imgur.com/Ognsy3N.jpg'>
    - Loss function: L(A,P,N) = max(d(A,P) - d(A,N) + anpha, 0) --> J = tổng L --> ta cần tối ưu hóa J để xác định các tham số. Khi đó với tập các bức ảnh, ta cần tạo ra bộ 3 **A, P, N** để thực hiện đào tạo mô hình.
    <img src = 'https://i.imgur.com/XpOxXpc.jpg'>
    - Với 1 tập ảnh nếu chọn ngẫu nhiên thì phương trình (1) dễ dàng thỏa mãn do 2 bức ảnh khác nhau rất nhiều trong tập ảnh --> do đó ta sẽ lựa chọn những bộ 3 mà d(A,N) rất gần d(A,P) để training (bởi vì độ chính xác 95% lúc này sẽ rất khác so với chọn ngẫu nhiên, do có nhiều phương trình mặc nhiên đúng dù tham số ra sao). Khi đánh giá mô hình ta xem với các bộ 3 mà ta đưa vào, các tham số mà ta thu được, liệu có d(A,P) nhỏ và d(A,N) lớn hay không?

# Face Verification và Phân loại nhị phân.
- Ngoài cách sử dụng Triplet Loss để đào tạo tham số, ta có thể xem hàm similarity function (d) giống như 1 bài toán phân loại nhị phân d = 0/1.
<img src ='https://i.imgur.com/DwHY7k3.jpg'>

- Khi đó training mô hình ta có X là 2 bức ảnh , và y=0/1 xác minh xem có cùng 1 người hay không.

# Code
Bạn tham khảo tại các nguồn sau: 
- [Deep Face Recognition with Keras](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/)
- [Deep Face](https://github.com/serengil/deepface)

# Thực hành:
- Thực hành với [mô hình VGG-Face](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf): [.........click here.............](https://github.com/thien1892/Face_Recognition/blob/main/vgg_face.ipynb)
- Mô hình (input_shape = (224,224,3)):
<img src ='https://i.imgur.com/LIHs5ei.jpg'>
