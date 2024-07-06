import cv2
import threading
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import time
import torch

# Kiểm tra và sử dụng GPU nếu có
use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

print(f"Using device: {device}")

model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE

lane_detector = UltrafastLaneDetector(model_path, model_type, device)

image = []

cap = cv2.VideoCapture("C:/Users/FPTSHOP/Videos/XSplit/Broadcaster/ngtrungduc2803@gmail.com/2024-06-22_18-11-12.mp4")

def camera():
    start_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        elapsed_time = time.time() - start_time
        image.append(frame)
        if len(image) > 3:
            del image[0]
            image1 = image[2]
            cv2.putText(image1, f'FPS: {frame_count/elapsed_time:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera Frame', image1)
            cv2.waitKey(1)

def lane():
    start_time1 = time.time()
    frame_count1 = 0
    while True:
        if len(image) > 2:
            frame_count1 += 1
            elapsed_time1 = time.time() - start_time1
            output_img, lanes_points, lanes_detected, cfg, draw_points = lane_detector.detect_lanes(image[2])
            output_img = cv2.resize(output_img, (800, 400))
            cv2.putText(output_img, f'FPS: {frame_count1/elapsed_time1:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Lane Detection Frame', output_img)
            cv2.waitKey(1)

read_thread = threading.Thread(target=camera)
read_thread.start()

process_thread = threading.Thread(target=lane)
process_thread.start()

read_thread.join()
process_thread.join()

cap.release()
cv2.destroyAllWindows()
