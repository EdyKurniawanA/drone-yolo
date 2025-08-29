# phase3_harness.py
import cv2
import torch
import queue
import threading
import time
from datetime import datetime

from quick_test_phase3.logger import Logger
from quick_test_phase3.class_counter import ClassCounter


# ----------------------------
# CONFIG
# ----------------------------
RTMP_URL = "rtmp://192.168.1.102/live"  # replace with DJI Flip stream if using
LOG_FILE = "logs/output.csv"
MODEL_PATH = "yolov5s.pt"  # change to your trained weights if needed

# ----------------------------
# THREAD-SAFE QUEUE
# ----------------------------
frame_queue = queue.Queue(maxsize=5)


# ----------------------------
# VIDEO CAPTURE THREAD
# ----------------------------
def capture_frames():
    cap = cv2.VideoCapture(RTMP_URL)

    if not cap.isOpened():
        print("❌ Cannot open video stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No frame received, retrying...")
            time.sleep(0.1)
            continue

        # put frames into queue (drop if full)
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()


# ----------------------------
# DETECTION + LOGGING THREAD
# ----------------------------
def process_frames():
    # Load YOLOv5
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)

    # Utilities
    class_counter = ClassCounter()
    logger = Logger(LOG_FILE)

    prev_time = time.time()
    frame_count = 0

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Run YOLO inference
        results = model(frame)

        # Update counter from YOLO
        class_counter.update_from_yolo(results)
        counts = class_counter.get_counts()

        # FPS calculation
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - prev_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Log
        logger.log(
            timestamp=datetime.now().isoformat(),
            system_uptime=time.time(),
            fps=fps,
            counts=counts,
        )

        # Show live preview (optional)
        annotated_frame = results.render()[0]
        cv2.imshow("YOLOv5 Phase 3", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)

    capture_thread.start()
    process_thread.start()

    try:
        while True:
            time.sleep(1)  # keep main alive
    except KeyboardInterrupt:
        print("\n🛑 Exiting gracefully...")
