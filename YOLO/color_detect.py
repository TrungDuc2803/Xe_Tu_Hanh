from ultralytics import YOLO
import cv2
import torch
import numpy as np

# Tải mô hình đã huấn luyện và chuyển sang sử dụng GPU
model = YOLO('traffic_best2.pt')  # Thay thế đường dẫn này bằng đường dẫn tới tệp mô hình của bạn
model.to('cuda')

# Mở webcam
cap = cv2.VideoCapture(0)

# Kiểm tra nếu mở webcam thành công
if not cap.isOpened():
    print("Error: Không thể mở webcam")
    exit()

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Không thể đọc khung hình từ webcam")
        break

    # Chuyển khung hình sang định dạng tensor và chuyển sang GPU
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 640))  # Điều chỉnh kích thước khung hình thành 640x640
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0

    # Phát hiện đối tượng trong khung hình
    results = model(frame_tensor)

    # Hiển thị kết quả phát hiện
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Lấy tọa độ của hộp bao
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()

            # Chuyển đổi tọa độ về kích thước gốc của khung hình
            x1, y1, x2, y2 = x1 * frame.shape[1] / 640, y1 * frame.shape[0] / 640, x2 * frame.shape[1] / 640, y2 * frame.shape[0] / 640

            # Vẽ hình chữ nhật xung quanh đối tượng
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            print(model.names[int(cls)])
            # Hiển thị tên lớp và độ tin cậy
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # Hiển thị khung hình với các đối tượng được phát hiện
    cv2.imshow('Webcam YOLOv8 Detection', frame)

    # Nhấn 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
