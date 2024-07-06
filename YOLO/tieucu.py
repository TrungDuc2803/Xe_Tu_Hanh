import cv2
import numpy as np
import glob

# Chuẩn bị các tham số cho bảng caro
chessboard_size = (9, 6)  # Kích thước của bảng caro (số ô vuông trong hàng ngang và hàng dọc)
square_size = 1.0  # Kích thước thực tế của một ô vuông trên bảng (ví dụ: 1.0 đơn vị là 1cm)

# Tạo các điểm đối tượng 3D
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Lưu trữ các điểm đối tượng và điểm hình ảnh từ tất cả các ảnh
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Đọc tất cả các ảnh hiệu chỉnh
images = glob.glob('calibration_images/*.jpg')  # Đường dẫn đến các ảnh hiệu chỉnh

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tìm các góc của bảng caro
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Nếu tìm thấy, thêm các điểm đối tượng và điểm ảnh
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Vẽ các góc và hiển thị
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Tính toán các tham số nội tại của camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Lưu kết quả
print("Tiêu cự (focal length) trong đơn vị pixel:")
print(f"F_x: {mtx[0,0]}")
print(f"F_y: {mtx[1,1]}")

print("\nMa trận camera:")
print(mtx)

print("\nĐộ méo ống kính (distortion coefficients):")
print(dist)
