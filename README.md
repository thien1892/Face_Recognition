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
