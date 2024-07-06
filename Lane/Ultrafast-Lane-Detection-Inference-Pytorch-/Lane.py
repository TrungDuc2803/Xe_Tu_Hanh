import time

import cv2
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import requests

model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

def distance_point_to_line(point, line_start, line_end):
    """
    Tính khoảng cách từ một điểm đến một đoạn thẳng.

    Args:
    - point: tuple hoặc list chứa tọa độ (x, y) của điểm.
    - line_start: tuple hoặc list chứa tọa độ (x, y) của điểm đầu của đoạn thẳng.
    - line_end: tuple hoặc list chứa tọa độ (x, y) của điểm cuối của đoạn thẳng.

    Returns:
    - Khoảng cách từ điểm đến đoạn thẳng:
        - Dương nếu điểm ở bên trái đoạn thẳng.
        - Âm nếu điểm ở bên phải đoạn thẳng.
        - Bằng 0 nếu điểm nằm trên đoạn thẳng.
    """
    # Chuyển đổi các điểm sang numpy array để tính toán dễ dàng hơn
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    # Tính vector AB và AP
    vector_AB = line_end - line_start
    vector_AP = point - line_start

    # Tính cross product
    cross_product = np.cross(vector_AB, vector_AP)

    # Tính chiều dài đoạn thẳng AB
    length_AB = np.linalg.norm(vector_AB)

    # Nếu độ dài đoạn thẳng AB bằng 0 (điểm đầu và điểm cuối trùng nhau), trả về khoảng cách từ điểm đến điểm đầu
    if length_AB == 0:
        return np.linalg.norm(vector_AP)

    # Tính khoảng cách từ điểm đến đoạn thẳng
    distance = cross_product / length_AB

    return distance

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Initialize webcam
cap = cv2.VideoCapture("D:/DATN/Lane/Ultrafast-Lane-Detection-Inference-Pytorch-/lane1.mp4")
# cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

frame_count = 0
skip_frames = 3
checkforward = True
checkleft = True
checkright = True
checkbackward = True
checkstop = True

def call_api(endpoint):
    global front_Distance, behind_Distance, left_Distance, right_Distance
    # url = f'http://192.168.1.112:5000/{endpoint}'
    # response = requests.get(url)
    # if response.status_code == 200:
    #     if endpoint == 'Distance':
    #         data = response.json()
    #         front_Distance = data['front_Distance']
    #         behind_Distance = data['behind_Distance']
    #         right_Distance = data['right_Distance']
    #         left_Distance = data['left_Distance']
    # else:
    #     print(f'Failed calling {endpoint}:', response.status_code)
    pass
def runforward():
    global checkforward
    if checkforward:
        call_api('Forward')
        checkforward = False

def runbackward():
    global checkbackward
    if checkbackward:
        call_api('Backward')
        checkbackward = False

def runleft():
    global checkleft
    if checkleft:
        call_api('TurnLeft')
        checkleft = False

def runright():
    global checkright
    if checkright:
        call_api('TurnRight')
        checkright = False

def stop():
    global checkforward, checkright, checkleft, checkbackward, checkstop
    if checkstop:
        call_api('Stop')
        checkforward = True
        checkright = True
        checkleft = True
        checkbackward = True
        checkstop = False

def Distance():
    call_api('Distance')

starttime = time.time()
frame_count = 0
# Đọc hình ảnh từ camera
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Chỉ tính khung hình này để giảm thời gian chạy
    # if frame_count % (skip_frames + 1) == 0:
    output_img, lanes_points, lanes_detected, cfg, draw_points = lane_detector.detect_lanes(frame)
    frame_count += 1
    height, width = output_img.shape[:2]
    sides = []
    for lane_num, lane_points in enumerate(lanes_points):
        # Lấy những điểm làn chính giữa
        if lane_num == 1:
            line_start_left = (width // 2, height)
            line_end_left = (width // 2, 0)

            for point in lane_points:
                sides.append(distance_point_to_line(point, line_start_left, line_end_left))

    if sides:
        average_side = np.mean(sides[:-5])
        if average_side < -200:
            runleft()
            direction = 'left'
        elif average_side > -50:
            direction = 'right'
            runright()
        else:
            runforward()
            direction = 'forward'

        cv2.putText(output_img, f'Side: {average_side:.2f}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_img, f'FPS: {frame_count/(time.time() - starttime):.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(output_img, (width//2, height), (width//2, 0), (0, 255, 0), 2)
        cv2.putText(output_img, f'{direction}', (500, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print(sides)
    cv2.imshow("Detected lanes", output_img)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
stop()
cv2.destroyAllWindows()
