import cv2
from ultralytics import YOLO
from centroid_tracker import CentroidTracker

model = YOLO("yolov8n_drone_small.pt")

tracker = CentroidTracker(
    max_disappeared=40,
    max_distance=60
)

CONF_THRESH = 0.05
MIN_AREA = 60

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
 # webcam OR replace with video path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESH, iou=0.65, verbose=False)[0]

    rects = []
    if results.boxes is not None:
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            if (x2 - x1) * (y2 - y1) > MIN_AREA:
                rects.append((x1, y1, x2, y2))

    objects = tracker.update(rects)

    for (x1, y1, x2, y2) in rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)

    for object_id, (cx, cy) in objects.items():
        cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
        cv2.putText(frame, f"ID {object_id}", (cx-10, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("YOLOv8 + Centroid Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
