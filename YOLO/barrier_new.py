import time
import threading
from ultralytics import YOLO
import cv2
import numpy as np
import requests
import torch

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

print(f"Using device: {device}")
model = YOLO("best100.pt", device)
cap = cv2.VideoCapture(0)

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

check_forward = True
check_left = True
check_right = True
check_backward = True
check_stop = True

check_run_left = True
check_run_right = True
on_run_left = True
on_run_right = True


def call_api(endpoint):
    global front_Distance, behind_Distance, left_Distance, right_Distance
    url = f'http://192.168.1.112:5000/{endpoint}'
    response = requests.get(url)
    if response.status_code == 200:
        if endpoint == 'Distance':
            data = response.json()
            front_Distance = data['front_Distance']
            behind_Distance = data['behind_Distance']
            right_Distance = data['right_Distance']
            left_Distance = data['left_Distance']

    else:
        print(f'Failed calling {endpoint}:', response.status_code)


def point_side_of_line(line_start, line_end, point):
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    return np.sign((point[0] - line_start[0]) * dy - (point[1] - line_start[1]) * dx)


def run_forward():
    global check_forward
    if check_forward:
        call_api('Forward')
        check_forward = False


def run_backward():
    global check_backward
    if check_backward:
        call_api('Backward')
        check_backward = False


def run_left():
    global check_left
    if check_left:
        call_api('Left')
        check_left = False


def run_right():
    global check_right
    if check_right:
        call_api('Right')
        check_right = False


def stop():
    global check_forward, check_right, check_left, check_backward, check_stop
    if check_stop:
        call_api('Stop')
        check_forward = True
        check_right = True
        check_left = True
        check_backward = True
        check_stop = False


def distance():
    call_api('Distance')


image = []


def camera():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image.append(frame)
        if len(image) > 3:
            del image[0]


start_time = time.time()
frame_count = 0


def run_car():
    while cap.isOpened():
        global start_time, frame_count, front_Distance, behind_Distance, left_Distance, right_Distance, check_forward, \
            check_left, check_right, check_backward, check_stop, check_run_left, check_run_right, on_run_left, \
            on_run_right
        if len(image) > 2:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.resize(frame, (int(width), int(height)))
            results = model.predict(source=frame, stream=True, save=False, imgsz=320, conf=0.25)

            for result in results:
                boxes = result.boxes
                max_index_between = 0
                max_index_left = 0
                max_index_right = 0

                between = False

                color = (0, 255, 0)
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()
                    class_name = model.names[int(cls)]

                    if y2 > 280:
                        line_start_left = (150, 600)
                        line_end_left = (330, 280)
                        line_start_right = (650, 600)
                        line_end_right = (470, 280)

                        point1 = (x1, y2)
                        point2 = (x2, y2)

                        side_left1 = point_side_of_line(line_start_left, line_end_left, point1)
                        side_left2 = point_side_of_line(line_start_left, line_end_left, point2)
                        side_right1 = point_side_of_line(line_start_right, line_end_right, point1)
                        side_right2 = point_side_of_line(line_start_right, line_end_right, point2)

                        if side_left2 >= 0:
                            if max_index_left < y2:
                                max_index_left = y2

                        elif side_right1 <= 0:
                            if max_index_right < y2:
                                max_index_right = y2

                        else:
                            between = True

                            if side_left1 >= 0:
                                if max_index_left < y2:
                                    max_index_left = y2

                            if side_right2 <= 0:
                                if max_index_right < y2:
                                    max_index_right = y2

                            if max_index_between < y2:
                                max_index_between = y2

                    cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1 + 20), int(y1) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                distance()
                if between and max_index_between > 330:

                    if max_index_between > 330:
                        color = (0, 255, 255)

                    if max_index_between > 440:
                        color = (0, 0, 255)
                        colorx = (0, 0, 255)

                        if behind_Distance > 20:
                            run_backward()
                            check_right = True
                            check_forward = True
                            check_left = True
                            check_stop = True
                            colorx = (0, 255, 255)

                            if behind_Distance > 30:
                                colorx = (0, 255, 0)

                        else:
                            stop()

                        cv2.putText(frame, f"backward", (370, 580),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    colorx, 2)
                    elif max_index_between > 330:

                        if (((max_index_left < max_index_right and check_run_left) or check_run_right is False) and
                                on_run_left):
                            colorx = (0, 0, 255)
                            if left_Distance > 20:
                                run_left()
                                on_run_right = False
                                check_right = True
                                check_forward = True
                                check_backward = True
                                check_stop = True
                                colorx = (0, 255, 255)
                                if left_Distance > 30:
                                    colorx = (0, 255, 0)

                            else:
                                stop()
                                check_run_left = False
                                on_run_right = True

                            cv2.putText(frame, f"left", (370, 580),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        colorx, 2)

                        elif check_run_right and on_run_right:
                            colorx = (0, 0, 255)
                            if right_Distance > 20:
                                run_right()
                                on_run_left = False
                                check_left = True
                                check_forward = True
                                check_backward = True
                                check_stop = True
                                colorx = (0, 255, 255)

                                if right_Distance > 30:
                                    colorx = (0, 255, 0)

                            else:

                                stop()
                                check_run_right = False
                                on_run_left = True
                            cv2.putText(frame, f"right", (370, 580),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        colorx, 2)
                else:

                    color = (0, 255, 0)
                    colorx = (0, 0, 255)

                    if front_Distance > 20:
                        run_forward()
                        on_run_left = True
                        on_run_right = True
                        check_left = True
                        check_right = True
                        check_backward = True
                        check_stop = True
                        check_run_left = True
                        check_run_right = True
                        colorx = (0, 255, 255)

                        if front_Distance > 30:
                            colorx = (0, 255, 0)

                    else:
                        stop()

                    cv2.putText(frame, f"forward", (370, 580),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                colorx, 2)

                colorline = (0, 255, 0)
                if 10 < behind_Distance < 30:
                    colorline = (0, 255, 255)

                elif behind_Distance <= 10:
                    colorline = (0, 0, 255)

                cv2.arrowedLine(frame, (400, 500), (400, 550), colorline, 2, tipLength=0.2)
                cv2.putText(frame, f"{behind_Distance}", (403, 525),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            colorline, 2)

                colorline = (0, 255, 0)
                if 10 < front_Distance < 30:
                    colorline = (0, 255, 255)

                elif front_Distance <= 10:
                    colorline = (0, 0, 255)

                cv2.arrowedLine(frame, (400, 500), (400, 450), colorline, 2, tipLength=0.2)
                cv2.putText(frame, f"{front_Distance}", (403, 475),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            colorline, 2)

                colorline = (0, 255, 0)
                if 10 < left_Distance < 30:
                    colorline = (0, 255, 255)

                elif left_Distance <= 10:
                    colorline = (0, 0, 255)

                cv2.arrowedLine(frame, (400, 500), (350, 500), colorline, 2, tipLength=0.2)
                cv2.putText(frame, f"{left_Distance}", (360, 495),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            colorline, 2)
                colorline = (0, 255, 0)
                if 10 < right_Distance < 30:
                    colorline = (0, 255, 255)

                elif right_Distance <= 10:
                    colorline = (0, 0, 255)

                cv2.arrowedLine(frame, (400, 500), (450, 500), colorline, 2, tipLength=0.2)
                cv2.putText(frame, f"{right_Distance}", (410, 495),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            colorline, 2)

                cv2.line(frame, (300, 336), (500, 336), color, 2)
                cv2.line(frame, (240, 442), (560, 442), color, 2)
                cv2.polylines(frame, [pts], True, color, 2)

            cv2.putText(frame, f"FPS: {frame_count / (time.time() - start_time):.2f}", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('YOLOv8 Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop()
                exit(0)


read_thread = threading.Thread(target=camera)
read_thread.start()

process_thread = threading.Thread(target=run_car)
process_thread.start()

read_thread.join()
process_thread.join()

cap.release()
stop()
cv2.destroyAllWindows()
