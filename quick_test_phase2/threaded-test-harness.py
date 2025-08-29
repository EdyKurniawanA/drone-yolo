import cv2
import socket
import threading
import signal
import sys

# Global stop flag
stop_event = threading.Event()


# === VIDEO THREAD ===
def video_thread(rtmp_url):
    cap = cv2.VideoCapture(rtmp_url)
    while not stop_event.is_set() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot grab frame")
            break

        cv2.imshow("RTMP Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()


# === GPS THREAD ===
def gps_thread(host="0.0.0.0", port=5000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(1.0)  # <-- important, so it won't block forever
    print(f"📡 Listening for GPS data on UDP {port}...")

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(1024)
            print(f"Received from {addr}: {data.decode('utf-8')}")
        except socket.timeout:
            continue

    sock.close()


# === Handle Ctrl+C ===
def signal_handler(sig, frame):
    print("\n Stopping all threads...")
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    rtmp_url = "rtmp://192.168.1.102/live"

    t1 = threading.Thread(target=video_thread, args=(rtmp_url,), daemon=True)
    t2 = threading.Thread(target=gps_thread, args=("0.0.0.0", 5000), daemon=True)

    t1.start()
    t2.start()

    # Wait for threads to finish
    t1.join()
    t2.join()
    print("Clean exit.")
