import cv2

# Replace with your RTMP URL (test with OBS or sample stream first)
rtmp_url = "rtmp://192.168.1.102/live"

cap = cv2.VideoCapture(rtmp_url)

if not cap.isOpened():
    print("❌ Cannot open RTMP stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    cv2.imshow("RTMP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
