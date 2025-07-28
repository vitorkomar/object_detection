import os
os.environ.pop("QT_QPA_PLATFORM", None)

import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame could not be read")
        break

    results = model(frame)
    result = results[0]
    
    xyxy = result.boxes.xyxy
    for box in xyxy:

        box = box.tolist()
        top_left_corner = (int(box[0]), int(box[1]))
        bottom_right_corner = (int(box[2]), int(box[3]))

        cv2.rectangle(frame, top_left_corner, bottom_right_corner, (0, 0, 255), 2)

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()