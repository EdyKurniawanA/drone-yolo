import subprocess
import numpy as np
import cv2
import threading
import socket
import time
from ultralytics import YOLO

# --- CONFIG ---
RTMP_URL = "rtmp://192.168.1.102/live"
RESOLUTION = (1280, 720)  # Safe size to keep latency low
TARGET_FPS = 30  # Match Flip setting
GPS_HOST = "0.0.0.0"
GPS_PORT = 5000

# Load YOLO
model = YOLO("yolov5su.pt")  # Replace with your model path

running = True
latest_frame = None
lock = threading.Lock()  # For thread-safe frame sharing


def video_reader():
    """Continuously read frames from FFmpeg and store only the latest one"""
    global latest_frame, running

    ffmpeg_cmd = [
        "ffmpeg",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-i",
        RTMP_URL,
        "-vf",
        f"scale={RESOLUTION[0]}:{RESOLUTION[1]}",
        "-r",
        str(TARGET_FPS),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
    )

    frame_size = RESOLUTION[0] * RESOLUTION[1] * 3

    while running:
        raw_frame = proc.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            continue

        frame = np.frombuffer(raw_frame, np.uint8).reshape(
            (RESOLUTION[1], RESOLUTION[0], 3)
        )

        # Overwrite the latest frame
        with lock:
            latest_frame = frame

    proc.kill()


def video_processor():
    """Run YOLO inference on the latest frame and display it"""
    global latest_frame, running

    while running:
        frame = None
        with lock:
            if latest_frame is not None:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        # (Optional) downscale more before YOLO for speed
        small = cv2.resize(frame, (640, 360))

        # Run YOLO
        results = model(small, verbose=False)

        # Annotate results (on the original frame size for display)
        annotated = results[0].plot()
        cv2.imshow("YOLO RTMP Stream", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            break

    cv2.destroyAllWindows()


def gps_thread():
    """Listen for GPS UDP packets"""
    global running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((GPS_HOST, GPS_PORT))
    sock.settimeout(1)

    while running:
        try:
            data, addr = sock.recvfrom(1024)
            print(f"GPS Data from {addr}: {data.decode().strip()}")
        except socket.timeout:
            continue


if __name__ == "__main__":
    # Start threads
    t1 = threading.Thread(target=video_reader, daemon=True)
    t2 = threading.Thread(target=video_processor, daemon=True)
    t3 = threading.Thread(target=gps_thread, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    try:
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
        running = False

    t1.join()
    t2.join()
    t3.join()
