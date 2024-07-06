import time
from collections import deque
from ultralytics import YOLO
import cv2
import numpy as np
import requests
import torch

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

print(f"Using device: {device}")

def is_valid(x, y, rows, cols, matrix, visited):
    # Hàm kiểm tra ô (x, y) có hợp lệ để đi qua hay không
    return 0 <= x < rows and 0 <= y < cols and not visited[x][y] and matrix[x][y] == 0


def bfs(matrix, start, end):
    rows, cols = len(matrix), len(matrix[0])
    queue = deque([(start[0], start[1])])  # Hàng đợi ban đầu với điểm bắt đầu
    visited = [[False for _ in range(cols)] for _ in range(rows)]  # Ma trận đánh dấu các ô đã duyệt
    parent = [[None for _ in range(cols)] for _ in range(rows)]  # Ma trận lưu trữ cha để truy vết đường đi

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4 hướng di chuyển: lên, xuống, trái, phải

    while queue:
        x, y = queue.popleft()  # Lấy phần tử đầu tiên từ hàng đợi

        # Kiểm tra nếu đã đến điểm kết thúc
        if (x, y) == end:
            path = []
            # Truy vết từ điểm kết thúc về điểm bắt đầu
            while (x, y) != start:
                path.append((x, y))
                x, y = parent[x][y]
            path.append(start)
            path.reverse()  # Đảo ngược để có thứ tự từ điểm bắt đầu đến điểm kết thúc
            return path

        # Duyệt các ô láng giềng
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, rows, cols, matrix, visited):
                visited[nx][ny] = True  # Đánh dấu ô láng giềng là đã duyệt
                parent[nx][ny] = (x, y)  # Lưu cha của ô láng giềng để truy vết
                queue.append((nx, ny))  # Thêm vào hàng đợi

    # Nếu không tìm thấy đường đi từ điểm bắt đầu đến điểm kết thúc
    return None


# Ví dụ sử dụng
matrix = [
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0]
]
start = (6, 0)
end = (0, 0)

path = bfs(matrix, start, end)
matrix = []
if len(path) > 5 and path[4][0] == 5:
    matrix.append(["left", "Blue"])
if len(path) > 8 and path[7][0] == 3:
    if path[7][1] == 2:
        matrix.append(["left", "Red"])
    else:
        matrix.append(["right", "Red"])
if len(path) > 11:
    if path[10][1] == 4:
        matrix.append(["right", "Yellow"])
    else:
        matrix.append(["left", "Yellow"])

model = YOLO('traffic_best2.pt', device)  # Thay thế đường dẫn này bằng đường dẫn tới tệp mô hình của bạn
model.to('cuda')
video_path = "C:/Users/FPTSHOP/Videos/XSplit/Broadcaster/ngtrungduc2803@gmail.com/2024-06-26_17-47-45.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video.")
    exit()

height = 600
width = 800
front_Distance = 0
behind_Distance = 0
left_Distance = 0
right_Distance = 0
pts = np.array([[150, 600], [330, 280], [470, 280], [650, 600]], np.int32)

checkforward = True
checkleft = True
checkright = True
checkbackward = True
checkstop = True

checkrunleft = True
checkrunright = True
onrunleft = True
onrunright = True

def call_api(endpoint):
    global front_Distance, behind_Distance, left_Distance, right_Distance
    # url = f'http://192.168.1.112:5000/{endpoint}'
    # response = requests.get(url)
    # if response.status_code == 200:
    # 	if endpoint == 'Distance':
    #         data = response.json()
    #         front_Distance = data['front_Distance']
    #         behind_Distance = data['behind_Distance']
    #         right_Distance = data['right_Distance']
    #         left_Distance = data['left_Distance']
    #
    #     # print(f'Success calling {endpoint}:', response.status_code)
    # else:
    #     print(f'Failed calling {endpoint}:', response.status_code)
    pass

def point_side_of_line(line_start, line_end, point):
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    return np.sign((point[0] - line_start[0]) * dy - (point[1] - line_start[1]) * dx)

def runforward():
    # global checkforward
    # if checkforward:
    call_api('Forward')
        # checkforward = False

def runbackward():
    global checkbackward
    if checkbackward:
        call_api('Backward')
        checkbackward = False

def runleft():
    # global checkleft
    # if checkleft:
    call_api('TurnLeft')
    # checkleft = False

def runright():
    global checkright
    if checkright:
        call_api('TurnRight')
        checkright = False

def stop():
    global checkforward, checkright, checkleft, checkbackward, checkstop
    if checkstop :
        call_api('Stop')
        checkforward = True
        checkright = True
        checkleft = True
        checkbackward = True
        checkstop = False

def Distance():
    call_api('Distance')

index = 0
check = False
maxindexleft = 0
maxindexright = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if check == True:
        check = False
        time.sleep(1)
    frame = cv2.resize(frame, (int(width), int(height)))
    results = model.predict(source=frame, stream=True, save=False, imgsz=320, conf=0.25)

    lane = ''
    for result in results:
        boxes = result.boxes

        color = (0, 255, 0)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0]
            cls = box.cls[0]
            class_name = model.names[int(cls)]
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            Distance()
            if y2 > 500 and not check:
                line_start_left = (200, 600)
                line_end_left = (330, 280)
                line_start_right = (600, 600)
                line_end_right = (470, 280)

                point1 = (x1, y2)
                point2 = (x2, y2)

                sideleft1 = point_side_of_line(line_start_left, line_end_left, point1)
                sideleft2 = point_side_of_line(line_start_left, line_end_left, point2)
                sideright1 = point_side_of_line(line_start_right, line_end_right, point1)
                sideright2 = point_side_of_line(line_start_right, line_end_right, point2)

                if sideleft2 >= 0:
                    if maxindexleft < y2:
                        maxindexleft = y2

                elif sideright1 <= 0:
                    if maxindexright < y2:
                        maxindexright = y2
                else:
                    if model.names[int(cls)] == 'Red':
                        stop()
                        exit(0)
                    if (index < len(matrix))and (front_Distance < 20):
                        if model.names[int(cls)] == matrix[index][1]:
                            if matrix[index][0] == 'left':
                                check = True
                                runleft()
                                print(time.time())
                                time.sleep(3.8)
                                print(time.time())
                                check = False
                                lane = 'left'
                                print('left')
                            elif matrix[index][0] == 'right':
                                check = True
                                runright()
                                time.sleep(3.8)
                                check = False
                                lane = 'right'
                                print('right')
                            index = index + 1
            if not check:
                runforward()
                lane = 'forward'
                print('forward')
            cv2.putText(frame, f'{lane}', (500, 500),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2)

        cv2.line(frame, (300, 336), (500, 336), color, 2)
        cv2.line(frame, (240, 442), (560, 442),color, 2)
        cv2.polylines(frame, [pts], True, color, 2)

    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
checkstop = True
stop()
cv2.destroyAllWindows()