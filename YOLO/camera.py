import cv2
import numpy as np

# Địa chỉ RTSP của camera, thường có dạng rtsp://username:password@ip_address:port/stream
# rtsp_url = "rtsp://admin:L2B274DC@192.168.1.112:554/cam/realmonitor?channel=1&subtype=0"

# Khởi tạo kết nối với camera
cap = cv2.VideoCapture(0)

# Kiểm tra kết nối
if not cap.isOpened():
    print("Không thể kết nối với camera")
    exit()

while True:
    # Đọc khung hình từ camera

    ret, frame = cap.read()
    frame = cv2.resize(frame, (int(800), int(600)))

    # Nếu không đọc được khung hình, thoát khỏi vòng lặp
    if not ret:
        print("Không thể đọc khung hình từ camera")
        break

    # Hiển thị khung hình
    color = (0, 0, 255)
    pts = np.array([[150, 600], [330, 280], [470, 280], [650, 600]], np.int32)
    cv2.line(frame, (300, 336), (500, 336), color, 2)
    cv2.line(frame, (240, 442), (560, 442), color, 2)
    cv2.polylines(frame, [pts], True, (0,255,0), 2)
    cv2.imshow('Camera Feed', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()