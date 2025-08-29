import sys
import time
import json
import threading
import queue
from typing import Optional, Dict, Any

import numpy as np
import cv2
import psutil
import torch
try:
    import serial  # pyserial
except Exception:  # optional at import-time
    serial = None

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QStackedWidget,
    QFormLayout,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
)


DEFAULT_RTMP = "rtmp://192.168.1.102/live"
DEFAULT_COM_PORT = "COM8"
DEFAULT_BAUD = "115200"
DEFAULT_MODEL = "yolov5s.pt"


class HomeScreen(QWidget):
    def __init__(self, on_start_callback):
        super().__init__()
        self.on_start_callback = on_start_callback

        title = QLabel("Drone YOLO - Connect Stream")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))

        form = QFormLayout()

        self.rtmp_input = QLineEdit(DEFAULT_RTMP)
        form.addRow("RTMP URL:", self.rtmp_input)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov5s.pt", "yolov5m.pt"])
        self.model_combo.setCurrentText(DEFAULT_MODEL)
        form.addRow("Model:", self.model_combo)

        self.com_input = QLineEdit(DEFAULT_COM_PORT)
        form.addRow("GPS COM Port:", self.com_input)

        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baud_combo.setCurrentText(DEFAULT_BAUD)
        form.addRow("GPS Baud:", self.baud_combo)

        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self._on_start_clicked)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(self.start_btn)
        layout.addStretch(1)
        self.setLayout(layout)

    def _is_valid_rtmp(self, url: str) -> bool:
        return url.startswith("rtmp://") and len(url.split("/")) >= 3

    def _on_start_clicked(self):
        url = self.rtmp_input.text().strip()
        model = self.model_combo.currentText().strip()
        com_port = self.com_input.text().strip()
        baud = int(self.baud_combo.currentText())

        if not self._is_valid_rtmp(url):
            QMessageBox.warning(self, "Invalid RTMP", "Please enter a valid RTMP URL (e.g., rtmp://host/live)")
            return

        if not com_port:
            QMessageBox.warning(self, "Invalid COM Port", "Please enter a COM port (e.g., COM8)")
            return

        self.on_start_callback({
            "rtmp_url": url,
            "model": model,
            "com_port": com_port,
            "baud": baud,
        })


class MainScreen(QWidget):
    def __init__(self):
        super().__init__()

        # Left: video placeholder
        self.video_label = QLabel("Video Stream")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedHeight(480)
        self.video_label.setStyleSheet("background-color: #202020; color: #CCCCCC; border: 1px solid #404040;")

        # Right: metrics + class counts
        self.metrics_list = QListWidget()
        self.metrics_list.addItem(QListWidgetItem("FPS: --"))
        self.metrics_list.addItem(QListWidgetItem("CPU: --"))
        self.metrics_list.addItem(QListWidgetItem("Memory: --"))
        self.metrics_list.addItem(QListWidgetItem("GPU: --"))

        self.class_list = QListWidget()
        self.class_list.addItem(QListWidgetItem("person: 0"))

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Metrics"))
        right_col.addWidget(self.metrics_list)
        right_col.addWidget(QLabel("Class Counts"))
        right_col.addWidget(self.class_list)
        right_col.addStretch(1)

        # Add stop button at bottom
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.setStyleSheet("background-color: #d32f2f; color: white; padding: 8px;")
        right_col.addWidget(self.stop_btn)

        top = QHBoxLayout()
        top.addWidget(self.video_label, 3)
        top.addLayout(right_col, 1)

        self.setLayout(top)

    # Placeholder API for updating UI later
    def update_metrics(self, fps: float = None, cpu: float = None, mem: float = None, gpu: float = None):
        def set_row(idx: int, label: str, value):
            if value is None:
                return
            item = self.metrics_list.item(idx)
            if item is not None:
                item.setText(f"{label}: {value}")

        set_row(0, "FPS", f"{fps:.1f}" if fps is not None else None)
        set_row(1, "CPU", f"{cpu:.1f}%" if cpu is not None else None)
        set_row(2, "Memory", f"{mem:.1f}%" if mem is not None else None)
        set_row(3, "GPU", f"{gpu:.1f}%" if gpu is not None else None)

    def update_class_counts(self, counts: dict):
        self.class_list.clear()
        for name, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            self.class_list.addItem(QListWidgetItem(f"{name}: {value}"))

    def add_stop_button(self, callback):
        self.stop_btn.clicked.connect(callback)

    def set_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        from PySide6.QtGui import QImage, QPixmap

        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.video_label.setPixmap(pix)


class PipelineManager:
    def __init__(self, rtmp_url: str, model_path: str, com_port: str, baud: int):
        self.rtmp_url = rtmp_url
        self.model_path = model_path
        self.com_port = com_port
        self.baud = baud

        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=3)
        
        # Thread-safe shared data
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_counts: Dict[str, int] = {}
        self.fps: float = 0.0
        self.cpu: float = 0.0
        self.mem: float = 0.0
        self.gpu: Optional[float] = None

        self.latest_gps: Dict[str, Any] = {}
        self.threads: list[threading.Thread] = []

    # ---------------- Capture ----------------
    def _capture_loop(self):
        """Use cv2.VideoCapture like phase2_harness.py for stability"""
        cap = cv2.VideoCapture(self.rtmp_url)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video stream: {self.rtmp_url}")
            return

        frame_count = 0
        last_time = time.time()
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No frame received, retrying...")
                time.sleep(0.1)
                continue

            # Put frames into queue (drop if full to prevent lag)
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
                frame_count += 1
                
                # Calculate FPS every second
                now = time.time()
                if now - last_time >= 1.0:
                    self.fps = frame_count / (now - last_time)
                    frame_count = 0
                    last_time = now

        cap.release()

    # ---------------- Inference ----------------
    def _inference_loop(self):
        """Use torch.hub.load like phase2_harness.py for stability"""
        try:
            model = torch.hub.load("ultralytics/yolov5", "custom", path=self.model_path)
        except Exception as e:
            print(f"⚠️ Failed to load custom model, using default: {e}")
            model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        
        # Move to CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Enable optimizations
        if device == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
                model.half()  # Use FP16 for speed
            except Exception:
                pass

        # Warmup
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        _ = model(dummy)

        from collections import Counter

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Run YOLO inference
            results = model(frame)
            
            # Get annotated frame
            annotated = results.render()[0]
            
            # Update class counts
            counts = {}
            try:
                # Handle YOLOv5 results format
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                    detections = results.xyxy[0]  # (N, 6): [x1, y1, x2, y2, conf, cls]
                    if len(detections) > 0:
                        names = results.names
                        cls_ids = detections[:, -1].cpu().numpy().astype(int)
                        labels = [names[int(c)] for c in cls_ids]
                        counts = dict(Counter(labels))
            except Exception:
                counts = {}

            # Put results in queue for UI thread
            if not self.result_queue.full():
                self.result_queue.put((annotated, counts))

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------------- GPS (NMEA) ----------------
    def _gps_loop(self):
        if serial is None:
            return
        try:
            ser = serial.Serial(self.com_port, self.baud, timeout=1)
        except Exception:
            ser = None
        last_gga = {}
        last_rmc = {}
        while not self.stop_event.is_set():
            if ser is None:
                time.sleep(1)
                continue
            try:
                line = ser.readline().decode(errors="ignore").strip()
            except Exception:
                time.sleep(0.05)
                continue
            if not line.startswith("$"):
                continue
            # Minimal parse for GGA/RMC
            # GGA: $GPGGA,time,lat,N,lon,E,fix,numsats,hdop,alt,M,...
            # RMC: $GPRMC,time,status,lat,N,lon,E,speed,course,date,...
            try:
                parts = line.split(",")
                talker = parts[0][3:6]
                if talker == "GGA" and len(parts) >= 10:
                    lat = self._parse_lat(parts[2], parts[3])
                    lon = self._parse_lon(parts[4], parts[5])
                    alt = self._safe_float(parts[9])
                    last_gga = {"lat": lat, "lon": lon, "alt": alt}
                elif talker == "RMC" and len(parts) >= 10:
                    utc = self._parse_time(parts[1], parts[9])  # hhmmss.sss and ddmmyy
                    last_rmc = {"utc": utc}
                # Merge
                merged = {}
                merged.update(last_rmc)
                merged.update(last_gga)
                if merged:
                    self.latest_gps = merged
            except Exception:
                continue
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass

    @staticmethod
    def _safe_float(s: str) -> Optional[float]:
        try:
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _parse_lat(lat_str: str, hemi: str) -> Optional[float]:
        if not lat_str:
            return None
        try:
            deg = float(lat_str[:2])
            minutes = float(lat_str[2:])
            val = deg + minutes / 60.0
            if hemi == "S":
                val = -val
            return val
        except Exception:
            return None

    @staticmethod
    def _parse_lon(lon_str: str, hemi: str) -> Optional[float]:
        if not lon_str:
            return None
        try:
            deg = float(lon_str[:3])
            minutes = float(lon_str[3:])
            val = deg + minutes / 60.0
            if hemi == "W":
                val = -val
            return val
        except Exception:
            return None

    @staticmethod
    def _parse_time(time_str: str, date_str: str) -> Optional[str]:
        # Return ISO-like string if possible
        if not time_str or not date_str or len(date_str) < 6:
            return None
        try:
            hh = int(time_str[0:2])
            mm = int(time_str[2:4])
            ss = float(time_str[4:])
            dd = int(date_str[0:2])
            mo = int(date_str[2:4])
            yy = int(date_str[4:6])
            # naive UTC (20xx assumption)
            iso = f"20{yy:02d}-{mo:02d}-{dd:02d}T{hh:02d}:{mm:02d}:{int(ss):02d}Z"
            return iso
        except Exception:
            return None

    # ---------------- Metrics ----------------
    def _metrics_loop(self):
        process = psutil.Process()
        while not self.stop_event.is_set():
            try:
                self.cpu = psutil.cpu_percent(interval=0.5)
                self.mem = psutil.virtual_memory().percent
            except Exception:
                pass

    # ---------------- Public API ----------------
    def start(self):
        self.stop_event.clear()
        self.threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._inference_loop, name="inference", daemon=True),
            threading.Thread(target=self._gps_loop, name="gps", daemon=True),
            threading.Thread(target=self._metrics_loop, name="metrics", daemon=True),
        ]
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass

    def snapshot(self):
        # Get latest frame and counts from result queue
        frame = None
        counts = {}
        
        try:
            while not self.result_queue.empty():
                frame, counts = self.result_queue.get_nowait()
        except queue.Empty:
            pass
            
        gps = dict(self.latest_gps)
        return frame, counts, gps, self.fps, self.cpu, self.mem, self.gpu


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone YOLO")
        self.resize(1100, 640)

        self.stack = QStackedWidget()
        self.home = HomeScreen(self._start_clicked)
        self.main = MainScreen()
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.main)

        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

        # Store chosen config
        self.config = {}
        self.pipeline: Optional[PipelineManager] = None

        # UI update timer - slower to prevent lag
        self.timer = QTimer(self)
        self.timer.setInterval(50)  # 20 Hz UI refresh (slower for stability)
        self.timer.timeout.connect(self._tick)

    def _start_clicked(self, config: dict):
        self.config = config
        # Start threads (RTMP capture, inference, GPS, metrics)
        self.pipeline = PipelineManager(
            rtmp_url=config["rtmp_url"],
            model_path=config["model"],
            com_port=config["com_port"],
            baud=config["baud"],
        )
        self.pipeline.start()
        self.stack.setCurrentWidget(self.main)
        self.timer.start()

        # Add stop button callback
        self.main.add_stop_button(self._stop_detection)

    def _stop_detection(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.timer.stop()
        self.stack.setCurrentWidget(self.home)

    def closeEvent(self, event):
        try:
            self.timer.stop()
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        super().closeEvent(event)

    def _tick(self):
        if self.pipeline is None:
            return
        frame, counts, gps, fps, cpu, mem, gpu = self.pipeline.snapshot()
        if frame is not None:
            self.main.set_frame(frame)
        self.main.update_class_counts(counts)
        self.main.update_metrics(fps=fps, cpu=cpu, mem=mem, gpu=gpu if gpu is not None else None)


def run():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()


