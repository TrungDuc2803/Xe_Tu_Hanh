import cv2
from ultralytics import YOLO
import os

# Đảm bảo tạo thư mục lưu trữ nếu chưa tồn tại
save_dir = "D:/Desktop/yolov8/yolov8_DATN/output"
os.makedirs(save_dir, exist_ok=True)

# Tải mô hình
model = YOLO("yolov8n.pt")  # xây dựng mqô hình mới từ đầu

# Mở camera (0 là ID của camera mặc định, thay đổi nếu cần)
cap = cv2.VideoCapture(2)

# Kiểm tra xem camera có mở được không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Đặt kích thước khung hình nếu cần
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Đọc và xử lý các khung hình từ camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận khung hình từ camera")
        break

    # Dự đoán trên khung hình
    results = model.predict(source=frame)

    # Vẽ các kết quả phát hiện lên khung hình
    annotated_frame = results[0].plot()

    # Hiển thị khung hình với các phát hiện
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

print("Kết quả được lưu tại:", save_dir)
