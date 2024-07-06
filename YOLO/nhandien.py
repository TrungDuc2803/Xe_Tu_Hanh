from ultralytics import YOLO
import cv2
import numpy as np
import requests

model = YOLO("best100.pt")
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
pts = np.array([[150, 600], [330, 290], [470, 290], [650, 600]], np.int32)

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
    #     print(f'Success calling {endpoint}:', response.status_code)
    # else:
    #     print(f'Failed calling {endpoint}:', response.status_code)

def point_side_of_line(line_start, line_end, point):
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    return np.sign((point[0] - line_start[0]) * dy - (point[1] - line_start[1]) * dx)

def runforward():
    global checkforward
    if checkforward:
        # call_api('Forward')
        checkforward = False

def runbackward():
    global checkbackward
    if checkbackward:
        # call_api('Backward')
        checkbackward = False

def runleft():
    global checkleft
    if checkleft:
        # call_api('Left')
        checkleft = False

def runright():
    global checkright
    if checkright:
        call_api('Right')
        checkright = False

def stop():
    global checkforward, checkright, checkleft, checkbackward, checkstop
    if checkstop :
        # call_api('Stop')
        checkforward = True
        checkright = True
        checkleft = True
        checkbackward = True
        checkstop = False

def Distance():
      call_api('Distance')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (int(width), int(height)))
    results = model.predict(source=frame, stream=True, save=False, imgsz=320, conf=0.25)

    for result in results:
        boxes = result.boxes
        maxindexbetween = 0;
        maxindexleft = 0;
        maxindexright = 0;

        left = False
        right = False
        between = False

        color = (0, 255, 0)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            class_name = model.names[int(cls)]

            if (y2 > 280):
                line_start_left = (150, 600)
                line_end_left = (330, 280)
                line_start_right = (650, 600)
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
                    between = True

                    if sideleft1 >= 0:
                        if maxindexleft < y2:
                            maxindexleft = y2

                    if sideright2 <= 0:
                        if maxindexright < y2:
                            maxindexright = y2

                    if maxindexbetween < y2:
                        maxindexbetween = y2

            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1 + 20), int(y1) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # Distance()
        if (between == True and maxindexbetween > 330):

            if maxindexbetween > 330:
                color = (0, 255, 255)

            if maxindexbetween > 440:
                color = (0, 0, 255)
                colorx = (0, 0, 255)

                if behind_Distance > 20:
                    runbackward()
                    checkright = True
                    checkforward = True
                    checkleft = True
                    checkstop = True
                    colorx = (0, 255, 255)

                    if behind_Distance > 30:
                        colorx = (0, 255, 0)

                else:
                    stop()

                cv2.putText(frame, f"backward", (370, 580),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            colorx, 2)
            elif maxindexbetween > 330:

                if ((maxindexleft < maxindexright and checkrunleft) or checkrunright == False) and onrunleft:
                    colorx = (0, 0, 255)
                    if (left_Distance > 20):
                        runleft()
                        onrunright = False
                        checkright = True
                        checkforward = True
                        checkbackward = True
                        checkstop = True
                        colorx = (0, 255, 255)
                        if left_Distance > 30:
                            colorx = (0, 255, 0)

                    else:
                        stop()
                        checkrunleft = False
                        onrunright = True

                    cv2.putText(frame, f"left", (370, 580),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                colorx, 2)

                elif checkrunright and onrunright:
                    colorx = (0, 0, 255)
                    if right_Distance > 20 :
                        runright()
                        onrunleft = False
                        checkleft = True
                        checkforward = True
                        checkbackward = True
                        checkstop = True
                        colorx = (0, 255, 255)

                        if right_Distance > 30:
                            colorx = (0, 255, 0)

                    else:
                        stop()
                        checkrunright = False
                        onrunleft = True
                    cv2.putText(frame, f"right", (370, 580),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                colorx, 2)
        else:

            color = (0, 255, 0)
            colorx = (0, 0, 255)

            if front_Distance > 20:
                runforward()
                onrunleft = True
                onrunright = True
                checkleft = True
                checkright = True
                checkbackward = True
                checkstop = True
                checkrunleft = True
                checkrunright = True
                colorx = (0, 255, 255)

                if front_Distance > 30:
                    colorx = (0, 255, 0)

            else:
                stop()

            cv2.putText(frame, f"forward", (370, 580),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colorx, 2)

        colorline = (0, 255, 0)
        if behind_Distance < 30 and behind_Distance > 10:
            colorline = (0, 255, 255)

        if behind_Distance <= 10:
            colorline = (0, 0, 255)

        cv2.arrowedLine(frame, (400, 500), (400, 550), colorline, 2, tipLength=0.2)
        cv2.putText(frame, f"{behind_Distance}", (403, 525),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colorline, 2)

        colorline = (0, 255, 0)
        if front_Distance < 30 and front_Distance > 10:
            colorline = (0, 255, 255)

        if front_Distance <= 10:
            colorline = (0, 0, 255)

        cv2.arrowedLine(frame, (400, 500), (400, 450), colorline, 2, tipLength=0.2)
        cv2.putText(frame, f"{front_Distance}", (403, 475),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colorline, 2)

        colorline = (0, 255, 0)
        if left_Distance < 30 and left_Distance > 10:
            colorline = (0, 255, 255)

        if left_Distance <= 10:
            colorline = (0, 0, 255)

        cv2.arrowedLine(frame, (400, 500), (350, 500), colorline, 2, tipLength=0.2)
        cv2.putText(frame, f"{left_Distance}", (360, 495),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colorline, 2)
        colorline = (0, 255, 0)
        if right_Distance < 30 and right_Distance > 10:
            colorline = (0, 255, 255)

        if right_Distance <= 10:
            colorline = (0, 0, 255)

        cv2.arrowedLine(frame, (400, 500), (450, 500), colorline, 2, tipLength=0.2)
        cv2.putText(frame, f"{right_Distance}", (410, 495),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colorline, 2)

        cv2.line(frame, (310, 330), (490, 330), color, 2)
        cv2.line(frame, (240, 450), (560, 450),color, 2)
        cv2.polylines(frame, [pts], True, color, 2)

    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
checkstop = True
stop()
cv2.destroyAllWindows()
