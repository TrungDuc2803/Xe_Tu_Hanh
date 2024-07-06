import time
import threading
from ultralytics import YOLO
import cv2
import torch
import requests
from collections import deque

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
end = (3, 0)

path = bfs(matrix, start, end)
matrix = []
if len(path) > 5 and path[4][0] == 5:
    matrix.append(["left", 'forward_left'])
if len(path) > 8 and path[7][0] == 3:
    if path[7][1] == 2:
        matrix.append(["left", 'round'])
    else:
        matrix.append(["right", 'round'])
if len(path) > 11:
    if path[10][1] == 4:
        matrix.append(["right", 'cross'])
    else:
        matrix.append(["left", 'cross'])
print(matrix)
def call_api(endpoint):
    global front_Distance, behind_Distance, left_Distance, right_Distance
    # url = f'http://192.168.1.112:5000/{endpoint}'
    # response = requests.get(url)
    # print(response.status_code)
    pass
def runforward():
    call_api('Forward')
def runleft():
    call_api('TurnLeft')
def runright():
    call_api('TurnRight')
def stop():
    call_api('Stop')

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

print(f"Using device: {device}")

model = YOLO('traffic_best2.pt', device)  # Thay thế đường dẫn này bằng đường dẫn tới tệp mô hình của bạn
model.to('cuda')
video_path = "C:/Users/FPTSHOP/Videos/XSplit/Broadcaster/ngtrungduc2803@gmail.com/2024-06-26_17-53-14.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Không thể mở webcam")
    exit()
image = []
def camera():
    # start_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # elapsed_time = time.time() - start_time
        image.append(frame)
        # if len(image) > 3:
        #     del image[0]
            # image1 = image[2]
            # cv2.putText(image1, f'FPS: {frame_count/elapsed_time:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow('Camera Frame', image1)
            # cv2.waitKey(1)
def lane():
    index = 0
    start_time = 0
    timesleep = 0
    while True:
        if len(image) > 2:
            frame = image[2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 640))  # Điều chỉnh kích thước khung hình thành 640x640
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0

            # Phát hiện đối tượng trong khung hình
            results = model(frame_tensor)
            # if start_time == 0:
            runforward()
            for result in results:
                # if start_time == 0:
                runforward()
                boxes = result.boxes
                for box in boxes:
                    # if start_time == 0:
                    runforward()
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()

                    x1, y1, x2, y2 = x1 * frame.shape[1] / 640, y1 * frame.shape[0] / 640, x2 * frame.shape[1] / 640, y2 * frame.shape[0] / 640

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    print(model.names[int(cls)])
                    print(x1, x2, y2)
                    cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    # if start_time != 0:
                    #     if time.time() - start_time < timesleep:
                    #         print('dang re')
                    #         print(timesleep)
                    #     else:
                    #         start_time = 0
                    if index <= len(matrix):
                        if y2 > 350 and 100 < x1 and x2 < 1200:
                            if model.names[int(cls)] == 'forward_left' and matrix[index] == ['left', 'forward_left'] and index == 0:
                                runleft()
                                print('runleft')
                                index += 1
                                # start_time = time.time()
                                # print('forward_left time',start_time)
                                time.sleep(3.9)
                                # time.sleep(3.7)
                            elif model.names[int(cls)] == 'cross' and index == 1:
                                if matrix[index] == ['left', 'cross']:
                                    runleft()
                                    print('runleft')
                                    # index += 1
                                    # start_time = time.time()
                                    # print('cross time',start_time)
                                    time.sleep(3.9)
                                elif matrix[index] == ['right', 'cross']:
                                    runright()
                                    print('runright')
                                    index += 1
                                    start_time = time.time()
                            elif model.names[int(cls)] == 'round' and index >= 1:
                                if matrix[index] == ['left', 'round']:
                                    runleft()
                                    print('runleft')
                                    index += 1
                                    start_time = time.time()
                                elif matrix[index] == ['right', 'round']:
                                    runright()
                                    print('runright')
                                    index += 1
                                    start_time = time.time()
                            elif model.names[int(cls)] == 'stop' and index == len(matrix):
                                stop()
                                print('runstop')
                                exit(0)
                            else:
                                runforward()
                                print('runforward')
            cv2.imshow('Webcam YOLOv8 Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop()
                exit(0)
read_thread = threading.Thread(target=camera)
read_thread.start()

process_thread = threading.Thread(target=lane)
process_thread.start()

read_thread.join()
process_thread.join()

cap.release()
stop()
cv2.destroyAllWindows()
