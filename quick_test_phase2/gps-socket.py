import socket

UDP_IP = "0.0.0.0"  # listen on all interfaces
UDP_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"📡 Listening on UDP {UDP_PORT}... (Ctrl+C to stop)")
latest_gps = None

try:
    while True:
        data, addr = sock.recvfrom(1024)
        msg = data.decode()
        print(f"Received from {addr}: {msg}")
        latest_gps = msg
except KeyboardInterrupt:
    print("\n Stopped by user")
    sock.close()
