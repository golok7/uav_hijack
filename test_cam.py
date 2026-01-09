import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    cv2.imshow("Raw Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
