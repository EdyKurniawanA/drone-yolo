from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov5s.pt")  # replace with your trained weights

cap = cv2.VideoCapture(0)  # test with webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov5s.pt")  # replace with your trained weights

cap = cv2.VideoCapture(0)  # test with webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
