import sys
import time
import json
import threading
import queue
import os
import signal
import tracemalloc
from datetime import datetime
from typing import Optional, Dict, Any
import gc
import requests
import base64

import numpy as np
import cv2
import psutil
import torch

try:
    import serial  # pyserial
except Exception:  # optional at import-time
    serial = None

try:
    from inference import InferencePipeline
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except Exception:  # optional at import-time
    ROBOFLOW_AVAILABLE = False

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QPainter
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

# Import charts for performance visualization
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

# Import the existing logger and class counter
sys.path.append("..")
from quick_test_phase3.logger import Logger
from quick_test_phase3.class_counter import ClassCounter


DEFAULT_RTMP = "rtmp://192.168.1.102/live"
DEFAULT_COM_PORT = "COM8"
DEFAULT_BAUD = "115200"
DEFAULT_MODEL = "crack4.pt"
DEFAULT_ROBOFLOW_API_KEY = "Pwr60R16IPozPzElpd1Q"
DEFAULT_WORKSPACE = "edys-flow"
DEFAULT_WORKFLOW_ID = "road-damage-cv"
DEFAULT_LOCAL_INFERENCE_URL = "http://localhost:9001"

# Global configuration for stability
FRAME_SKIP_RATIO = 2  # Process every 2nd frame
MAX_QUEUE_SIZE = 3
INFERENCE_TIMEOUT = 0.1
CLEANUP_TIMEOUT = 2.0


class PerformanceChart(QWidget):
    """Real-time performance chart widget"""

    def __init__(self, title: str, y_max: float = 100.0, y_min: float = 0.0):
        super().__init__()
        self.y_max = y_max
        self.y_min = y_min

        # Create chart
        self.chart = QChart()
        self.chart.setTitle(title)
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().hide()

        # Create series for data
        self.series = QLineSeries()
        self.series.setName(title)
        self.chart.addSeries(self.series)

        # Setup axes
        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, 60)  # Show last 60 data points
        self.axis_x.setTitleText("Time (s)")

        self.axis_y = QValueAxis()
        self.axis_y.setRange(y_min, y_max)
        self.axis_y.setTitleText("Value")

        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)

        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)

        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.chart_view)
        # Add a small label to show current wall-clock time and value
        from PySide6.QtWidgets import QLabel
        self.value_label = QLabel("--:--:--  |  --")
        self.value_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.value_label)
        self.setLayout(layout)

        # Data storage
        self.data_points = []
        self.max_points = 60
        self.start_time = time.time()

    def update_value(self, value: float):
        """Add new data point to chart"""
        current_time = time.time() - self.start_time

        # Add new point
        self.data_points.append((current_time, value))

        # Keep only last max_points
        if len(self.data_points) > self.max_points:
            self.data_points = self.data_points[-self.max_points :]

        # Update chart
        self.series.clear()
        for t, v in self.data_points:
            self.series.append(t, v)

        # Auto-adjust Y axis if needed
        if value > self.y_max * 0.9:
            self.y_max = value * 1.1
            self.axis_y.setRange(self.y_min, self.y_max)
        elif value < self.y_min * 1.1:
            self.y_min = max(0, value * 0.9)
            self.axis_y.setRange(self.y_min, self.y_max)

        # Update label with wall clock time and current value
        try:
            now_str = datetime.now().strftime('%H:%M:%S')
            self.value_label.setText(f"{now_str}  |  {value:.1f}")
        except Exception:
            pass


class WorkerThread(QThread):
    """Separate thread for heavy processing to prevent GUI freezing"""
    frame_ready = Signal(np.ndarray, dict)  # frame, counts
    status_update = Signal(str)
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.log_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        
        # Performance tracking
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.frame_skip_counter = 0
        
        # Model and processing
        self.model = None
        self.roboflow_pipeline = None
        self.class_counter = ClassCounter(accumulate=True)
        
        # GPS data
        self.latest_gps = {}
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup CSV logging with timestamped filename"""
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/drone_yolo_{timestamp}.csv"
        
        fieldnames = [
            "timestamp", "system_uptime", "frame_id", "fps", "counts",
            "gps_utc", "gps_lat", "gps_lon", "gps_alt", "detection_count", "extra"
        ]
        
        self.logger = Logger(log_filename, fieldnames)
        self.frame_id = 0

    def run(self):
        """Main worker thread loop"""
        try:
            self.status_update.emit("🚀 Starting worker thread...")
            
            if self.config["model_type"] == "local":
                self._run_local_model()
            else:
                self._run_roboflow_model()
                
        except Exception as e:
            self.status_update.emit(f"❌ Worker thread error: {e}")
        finally:
            self._cleanup()

    def _run_local_model(self):
        """Run local YOLO model with optimizations"""
        try:
            # Load model
            self.status_update.emit("🔍 Loading local model...")
            model_path = self.config["model"]
            if not os.path.isabs(model_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, model_path)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
                
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
            
            # Optimize model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            
            if device == "cuda":
                torch.backends.cudnn.benchmark = True
                self.model.half()
                
            # Warmup
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.no_grad():
                _ = self.model(dummy)
                
            self.status_update.emit(f"✅ Model loaded on {device}")
            
            # Start capture thread
            capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            capture_thread.start()
            
            # Start GPS thread
            gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
            gps_thread.start()
            
            # Start logging thread
            logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
            logging_thread.start()
            
            # Main processing loop
            self._process_frames()
            
        except Exception as e:
            self.status_update.emit(f"❌ Local model error: {e}")

    def _run_roboflow_model(self):
        """Run Roboflow model with optimizations"""
        try:
            self.status_update.emit("🔍 Initializing Roboflow...")
            
            # Check if using local inference
            if self.config.get("inference_type") == "local":
                self._run_local_roboflow_inference()
            else:
                self._run_cloud_roboflow_inference()
                
        except Exception as e:
            self.status_update.emit(f"❌ Roboflow error: {e}")

    def _run_cloud_roboflow_inference(self):
        """Run cloud-based Roboflow inference"""
        self.roboflow_pipeline = InferencePipeline.init_with_workflow(
            api_key=self.config["api_key"],
            workspace_name=self.config["workspace"],
            workflow_id=self.config["workflow_id"],
            video_reference=self.config["rtmp_url"],
            max_fps=15,
            on_prediction=self._roboflow_callback,
        )
        
        self.roboflow_pipeline.start()
        self.status_update.emit("✅ Roboflow cloud started")
        
        # Start GPS and logging threads
        gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
        gps_thread.start()
        
        logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
        logging_thread.start()
        
        # Wait for stop event
        while not self.stop_event.is_set():
            time.sleep(0.1)

    def _run_local_roboflow_inference(self):
        """Run local Roboflow inference via SDK"""
        self.status_update.emit("🔍 Connecting to local Roboflow Inference...")
        
        try:
            # Initialize the inference client with API key
            self.inference_client = InferenceHTTPClient(
                api_url=self.config['inference_url'],
                api_key=self.config.get('api_key', '')  # Add API key for local inference
            )
            
            # Test connection by trying to get server info
            try:
                # This will test if the server is reachable
                server_info = self.inference_client.get_server_info()
                self.status_update.emit(f"✅ Connected to local Roboflow Inference: {server_info}")
            except Exception as e:
                self.status_update.emit(f"⚠️ Server info not available, but continuing: {e}")
            
            self.status_update.emit("✅ Connected to local Roboflow Inference")
            
            # Start capture thread
            capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            capture_thread.start()
            
            # Start GPS thread
            gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
            gps_thread.start()
            
            # Start logging thread
            logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
            logging_thread.start()
            
            # Main processing loop
            self._process_local_inference_frames()
            
        except Exception as e:
            self.status_update.emit(f"❌ Cannot connect to local inference: {e}")
            return

    def _process_local_inference_frames(self):
        """Process frames using local Roboflow inference SDK"""
        print("🔄 Starting local inference frame processing loop")
        frame_count = 0
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=INFERENCE_TIMEOUT)
                frame_count += 1
                print(f"📥 Got frame {frame_count} from queue, shape: {frame.shape}")
            except queue.Empty:
                print("⏳ No frames in queue, waiting...")
                continue
                
            # Frame skipping for performance
            self.frame_skip_counter += 1
            if self.frame_skip_counter % FRAME_SKIP_RATIO != 0:
                continue
                
            try:
                # Debug: Print frame info
                if self.frame_count % 10 == 0:  # Every 10 frames for more frequent updates
                    self.status_update.emit(f"🔄 Processing frame {self.frame_count}, shape: {frame.shape}")
                    print(f"🔄 Processing frame {self.frame_count}, shape: {frame.shape}")
                
                # Run inference using the SDK
                print(f"🚀 Running inference with workspace='{self.config.get('workspace', 'default')}', workflow_id='{self.config.get('workflow_id', 'default')}'")
                
                result = self.inference_client.run_workflow(
                    workspace_name=self.config.get("workspace", "default"),
                    workflow_id=self.config.get("workflow_id", "default"),
                    images={
                        "image": frame
                    }
                )
                
                # Debug: Print result info
                print(f"📊 Inference completed. Result type: {type(result)}")
                if isinstance(result, dict):
                    print(f"📊 Result keys: {list(result.keys())}")
                else:
                    print(f"📊 Result content: {result}")
                
                if self.frame_count % 10 == 0:
                    self.status_update.emit(f"📊 Inference completed. Result type: {type(result)}")
                
                # Extract counts from result
                counts = self._extract_counts_from_sdk_result(result)
                print(f"📊 Extracted counts: {counts}")
                
                # Create annotated frame
                annotated = frame.copy()
                
                # Try to get annotated image from workflow outputs
                if isinstance(result, dict) and "output_image" in result and result["output_image"] is not None:
                    try:
                        oi = result["output_image"]
                        annotated = oi.numpy_image if hasattr(oi, "numpy_image") else oi
                        print("🎨 Using output_image for annotated frame")
                    except Exception:
                        pass
                elif "output2" in result and result["output2"] is not None:
                    # output2 is label visualization image
                    print("🎨 Using output2 (label visualization) for annotated frame")
                    annotated = result["output2"]
                elif "output3" in result and result["output3"] is not None:
                    # output3 is bounding box visualization image
                    print("🎨 Using output3 (bounding box visualization) for annotated frame")
                    annotated = result["output3"]
                elif "image" in result and result["image"] is not None:
                    # If the SDK returns an annotated image
                    print("🎨 Using result image for annotated frame")
                    annotated = result["image"]
                else:
                    # Fallback: draw predictions manually
                    print("🎨 Drawing predictions manually")
                    predictions = None
                    if "output" in result and isinstance(result["output"], dict):
                        output = result["output"]
                        if "predictions" in output:
                            predictions = output["predictions"]
                        elif "detections" in output:
                            predictions = output["detections"]
                        elif "results" in output:
                            predictions = output["results"]
                    elif "predictions" in result:
                        predictions = result["predictions"]
                    
                    if predictions:
                        # Draw boxes from common prediction schemas
                        try:
                            for pred in predictions:
                                if not isinstance(pred, dict):
                                    continue
                                label = pred.get("class") or pred.get("class_name") or pred.get("label") or "object"
                                # Support xyxy or center-width-height
                                if all(k in pred for k in ("x1", "y1", "x2", "y2")):
                                    x1, y1, x2, y2 = int(pred["x1"]), int(pred["y1"]), int(pred["x2"]), int(pred["y2"])
                                else:
                                    # Assume center-based with width/height
                                    cx = pred.get("x") or pred.get("cx") or 0
                                    cy = pred.get("y") or pred.get("cy") or 0
                                    w = pred.get("width") or pred.get("w") or 0
                                    h = pred.get("height") or pred.get("h") or 0
                                    x1 = int(cx - w / 2)
                                    y1 = int(cy - h / 2)
                                    x2 = int(cx + w / 2)
                                    y2 = int(cy + h / 2)
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                print(f"🔎 Detected {label} at ({x1},{y1})-({x2},{y2})")
                        except Exception as _:
                            pass
                
                # Update accumulated class counts
                try:
                    if isinstance(counts, dict) and counts:
                        expanded_labels = []
                        for label, num in counts.items():
                            try:
                                expanded_labels.extend([label] * int(num))
                            except Exception:
                                continue
                        if expanded_labels:
                            self.class_counter.update_from_labels(expanded_labels)
                except Exception:
                    pass

                # Choose counts to display/log: prefer per-frame, fallback to accumulated
                try:
                    display_counts = counts if (isinstance(counts, dict) and counts) else self.class_counter.get_counts()
                except Exception:
                    display_counts = counts

                # Emit results to GUI
                print(f"📤 Emitting frame to GUI with counts: {display_counts}")
                self.frame_ready.emit(annotated, display_counts)
                
                # Log data
                if not self.log_queue.full():
                    self.log_queue.put((display_counts, frame))
                    
            except Exception as e:
                if hasattr(self, '_last_error_time'):
                    if time.time() - self._last_error_time > 5.0:
                        self.status_update.emit(f"⚠️ Local inference error: {e}")
                        self._last_error_time = time.time()
                else:
                    self._last_error_time = time.time()

    def _extract_counts_from_sdk_result(self, result):
        """Extract class counts from SDK inference result"""
        counts = {}
        try:
            # Handle different result formats from the SDK
            if isinstance(result, dict):
                # Debug: Print the full result structure
                print(f"🔍 Full result structure: {list(result.keys())}")
                
                # Try different possible locations for predictions
                predictions = None
                
                # Check for workflow outputs (your specific case)
                if "output" in result and isinstance(result["output"], dict):
                    output = result["output"]
                    print(f"🔍 Output structure: {list(output.keys()) if isinstance(output, dict) else type(output)}")
                    
                    if "predictions" in output:
                        predictions = output["predictions"]
                    elif "detections" in output:
                        predictions = output["detections"]
                    elif "results" in output:
                        predictions = output["results"]
                    elif isinstance(output, list):
                        predictions = output
                
                # Check for direct predictions
                elif "predictions" in result:
                    predictions = result["predictions"]
                elif "results" in result and isinstance(result["results"], list):
                    predictions = result["results"]
                elif "detections" in result:
                    predictions = result["detections"]
                
                # Extract counts from predictions
                if predictions:
                    from collections import Counter
                    labels = []
                    
                    if isinstance(predictions, list):
                        for pred in predictions:
                            if isinstance(pred, dict):
                                if "class" in pred:
                                    labels.append(pred["class"])
                                elif "class_name" in pred:
                                    labels.append(pred["class_name"])
                                elif "label" in pred:
                                    labels.append(pred["label"])
                            elif isinstance(pred, str):
                                labels.append(pred)
                    
                    counts = dict(Counter(labels))
                    print(f"🎯 Extracted labels: {labels}")
                    print(f"📊 Counts: {counts}")
                else:
                    print("⚠️ No predictions found in result")
                    
        except Exception as e:
            print(f"Error extracting counts: {e}")
            counts = {}
        return counts

    def _extract_counts_from_local_result(self, result):
        """Extract class counts from local inference result (legacy method)"""
        counts = {}
        try:
            if "predictions" in result:
                from collections import Counter
                labels = [pred.get("class", "") for pred in result["predictions"] if pred.get("class")]
                counts = dict(Counter(labels))
        except Exception:
            counts = {}
        return counts

    def _draw_predictions_on_frame(self, frame, predictions):
        """Draw predictions on frame (basic implementation)"""
        try:
            for pred in predictions:
                if "bbox" in pred:
                    bbox = pred["bbox"]
                    x1, y1, x2, y2 = int(bbox["x"]), int(bbox["y"]), int(bbox["x"] + bbox["width"]), int(bbox["y"] + bbox["height"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if "class" in pred:
                        cv2.putText(frame, pred["class"], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except Exception:
            pass
        return frame

    def _capture_frames(self):
        """Optimized RTMP capture with frame dropping"""
        try:
            # Use FFmpeg backend for better RTMP support
            cap = cv2.VideoCapture(self.config["rtmp_url"], cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
            
            if not cap.isOpened():
                self.status_update.emit(f"❌ Cannot open stream: {self.config['rtmp_url']}")
                return
                
            self.status_update.emit("📹 Video capture started")
            
            frame_count = 0
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if frame_count % 100 == 0:  # Log every 100 failed reads
                        print("⚠️ Failed to read frame from RTMP stream")
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"📹 Captured frame {frame_count}, shape: {frame.shape}")
                
                # Drop frames if queue is full (prevent memory buildup)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                    if frame_count % 30 == 0:
                        print(f"📤 Added frame {frame_count} to queue")
                else:
                    # Drop oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                        if frame_count % 30 == 0:
                            print(f"🔄 Replaced frame in queue (queue full)")
                    except queue.Empty:
                        pass
                
                # Update FPS
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = now
                    
        except Exception as e:
            self.status_update.emit(f"❌ Capture error: {e}")
        finally:
            try:
                cap.release()
            except:
                pass

    def _process_frames(self):
        """Process frames with optimizations"""
        from collections import Counter
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=INFERENCE_TIMEOUT)
            except queue.Empty:
                continue
                
            # Frame skipping for performance
            self.frame_skip_counter += 1
            if self.frame_skip_counter % FRAME_SKIP_RATIO != 0:
                continue
                
            # Run inference with optimizations
            with torch.no_grad():
                results = self.model(frame)
            
            # Get annotated frame
            annotated = results.render()[0]
            
            # Extract counts
            counts = {}
            try:
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                    detections = results.xyxy[0]
                    if len(detections) > 0:
                        names = results.names
                        cls_ids = detections[:, -1].cpu().numpy().astype(int)
                        labels = [names[int(c)] for c in cls_ids]
                        counts = dict(Counter(labels))
                        self.class_counter.update_from_yolo(results)
            except Exception:
                counts = {}
            
            # Emit results to GUI
            self.frame_ready.emit(annotated, counts)
            
            # Log data
            if not self.log_queue.full():
                self.log_queue.put((counts, frame))

    def _roboflow_callback(self, result, video_frame):
        """Roboflow prediction callback"""
        try:
            # Normalize input: sometimes callbacks return (result, frame) as a tuple
            if isinstance(result, tuple):
                try:
                    if len(result) == 2:
                        result, video_frame = result
                    elif len(result) == 1:
                        result = result[0]
                except Exception:
                    pass

            # Extract frame and counts
            annotated = None
            counts = {}
            
            # Prefer unified output image if present
            if isinstance(result, dict) and result.get("output_image") is not None:
                try:
                    oi = result["output_image"]
                    annotated = oi.numpy_image if hasattr(oi, "numpy_image") else oi
                except Exception:
                    annotated = result.get("output_image")
            # Get annotated frame from legacy field
            elif isinstance(result, dict) and result.get("label_visualization"):
                try:
                    annotated = result["label_visualization"].numpy_image
                except Exception:
                    annotated = result["label_visualization"]
            elif video_frame is not None:
                if hasattr(video_frame, 'image'):
                    annotated = video_frame.image
                elif hasattr(video_frame, 'numpy_image'):
                    annotated = video_frame.numpy_image
                else:
                    annotated = video_frame
            
            # Extract predictions from various schemas
            predictions = None
            if isinstance(result, dict):
                if "model_predictions" in result and isinstance(result["model_predictions"], dict) and "predictions" in result["model_predictions"]:
                    predictions = result["model_predictions"]["predictions"]
                elif "output" in result and isinstance(result["output"], dict):
                    out = result["output"]
                    if isinstance(out, dict):
                        predictions = out.get("predictions") or out.get("detections") or out.get("results")
                elif "predictions" in result:
                    predictions = result["predictions"]
                elif "detections" in result:
                    predictions = result["detections"]
            
            # Build counts and draw boxes if we only have raw predictions
            if predictions and annotated is not None:
                from collections import Counter
                labels = []
                try:
                    for pred in predictions:
                        if not isinstance(pred, dict):
                            continue
                        label = pred.get("class") or pred.get("class_name") or pred.get("label") or "object"
                        labels.append(label)
                        # Draw rectangle from either xyxy or center+wh
                        if all(k in pred for k in ("x1", "y1", "x2", "y2")):
                            x1, y1, x2, y2 = int(pred["x1"]), int(pred["y1"]), int(pred["x2"]), int(pred["y2"])
                        else:
                            cx = pred.get("x") or pred.get("cx") or 0
                            cy = pred.get("y") or pred.get("cy") or 0
                            w = pred.get("width") or pred.get("w") or 0
                            h = pred.get("height") or pred.get("h") or 0
                            x1 = int(cx - w / 2)
                            y1 = int(cy - h / 2)
                            x2 = int(cx + w / 2)
                            y2 = int(cy + h / 2)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        print(f"🔎 Detected {label} at ({x1},{y1})-({x2},{y2})")
                except Exception:
                    pass
                counts = dict(Counter(labels))
                # Update accumulated class counts
                try:
                    if labels:
                        self.class_counter.update_from_labels(labels)
                except Exception:
                    pass
            elif predictions:
                # Build counts even if we couldn't draw (no frame)
                from collections import Counter
                labels = []
                for pred in predictions:
                    if isinstance(pred, dict):
                        label = pred.get("class") or pred.get("class_name") or pred.get("label")
                        if label:
                            labels.append(label)
                counts = dict(Counter(labels))
                # Update accumulated class counts
                try:
                    if labels:
                        self.class_counter.update_from_labels(labels)
                except Exception:
                    pass
            
            # Ensure we emit some frame even without annotations
            if annotated is None and video_frame is not None:
                if hasattr(video_frame, 'image'):
                    annotated = video_frame.image
                elif hasattr(video_frame, 'numpy_image'):
                    annotated = video_frame.numpy_image
                else:
                    annotated = video_frame
            
            if annotated is not None:
                # Prefer per-frame counts; if empty, use accumulated
                try:
                    display_counts = counts if (isinstance(counts, dict) and counts) else self.class_counter.get_counts()
                except Exception:
                    display_counts = counts
                self.frame_ready.emit(annotated, display_counts)
                
            if not self.log_queue.full():
                self.log_queue.put((display_counts, video_frame))
                
        except Exception as e:
            if hasattr(self, '_last_error_time'):
                if time.time() - self._last_error_time > 5.0:
                    self.status_update.emit(f"⚠️ Roboflow callback error: {e}")
                    self._last_error_time = time.time()
            else:
                self._last_error_time = time.time()

    def _gps_loop(self):
        """GPS NMEA processing"""
        if serial is None:
            return
            
        try:
            ser = serial.Serial(self.config["com_port"], self.config["baud"], timeout=1)
        except Exception:
            ser = None
            
        last_gga = {}
        last_rmc = {}
        
        try:
            while not self.stop_event.is_set():
                if ser is None:
                    time.sleep(1)
                    continue
                    
                try:
                    line = ser.readline().decode(errors="ignore").strip()
                    if not line.startswith("$"):
                        continue
                        
                    parts = line.split(",")
                    talker = parts[0][3:6]
                    
                    if talker == "GGA" and len(parts) >= 10:
                        lat = self._parse_lat(parts[2], parts[3])
                        lon = self._parse_lon(parts[4], parts[5])
                        alt = self._safe_float(parts[9])
                        last_gga = {"lat": lat, "lon": lon, "alt": alt}
                    elif talker == "RMC" and len(parts) >= 10:
                        utc = self._parse_time(parts[1], parts[9])
                        last_rmc = {"utc": utc}
                    
                    # Merge GPS data
                    merged = {}
                    merged.update(last_rmc)
                    merged.update(last_gga)
                    if merged:
                        self.latest_gps = merged
                        
                except Exception:
                    continue
        finally:
            try:
                if ser is not None:
                    ser.close()
            except:
                pass

    def _logging_loop(self):
        """CSV logging thread"""
        while not self.stop_event.is_set():
            try:
                counts, frame = self.log_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Log detection data
            try:
                total_detections = int(sum(counts.values())) if isinstance(counts, dict) else 0
            except Exception:
                total_detections = 0
            self.logger.log(
                timestamp=datetime.now().isoformat(),
                system_uptime=time.perf_counter(),
                frame_id=self.frame_id,
                fps=self.fps,
                counts=counts,
                gps_utc=self.latest_gps.get("utc", ""),
                gps_lat=self.latest_gps.get("lat", ""),
                gps_lon=self.latest_gps.get("lon", ""),
                gps_alt=self.latest_gps.get("alt", ""),
                detection_count=total_detections,
                extra={
                    "accumulated_counts": self.class_counter.get_counts(),
                    "total_detections": total_detections
                }
            )
            self.frame_id += 1

    def _cleanup(self):
        """Cleanup resources"""
        try:
            self.status_update.emit("🧹 Cleaning up resources...")
            
            # Stop Roboflow
            if self.roboflow_pipeline:
                try:
                    self.roboflow_pipeline.join()
                except:
                    pass
                    
            # Clear model
            if self.model:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            # Close logger
            try:
                self.logger.close()
            except:
                pass
                
            # Force garbage collection
            gc.collect()
            
            self.status_update.emit("✅ Cleanup complete")
            
        except Exception as e:
            self.status_update.emit(f"⚠️ Cleanup error: {e}")

    def stop(self):
        """Stop the worker thread"""
        self.stop_event.set()
        self.quit()
        self.wait(CLEANUP_TIMEOUT * 1000)  # Wait up to 2 seconds

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
        if not time_str or not date_str or len(date_str) < 6:
            return None
        try:
            hh = int(time_str[0:2])
            mm = int(time_str[2:4])
            ss = float(time_str[4:])
            dd = int(date_str[0:2])
            mo = int(date_str[2:4])
            yy = int(date_str[4:6])
            iso = f"20{yy:02d}-{mo:02d}-{dd:02d}T{hh:02d}:{mm:02d}:{int(ss):02d}Z"
            return iso
        except Exception:
            return None


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

        # Model type selection
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Local Model", "Roboflow Cloud API", "Roboflow Local Inference"])
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        form.addRow("Model Type:", self.model_type_combo)

        # Local model selection
        self.model_combo = QComboBox()
        available_models = self._find_available_models()
        if available_models:
            self.model_combo.addItems(available_models)
            if "crack4.pt" in available_models:
                self.model_combo.setCurrentText("crack4.pt")
        else:
            self.model_combo.addItems(["crack4.pt", "yolov5s.pt", "yolov5m.pt"])
            self.model_combo.setCurrentText("crack4.pt")
        form.addRow("Local Model:", self.model_combo)

        # Roboflow API configuration
        self.roboflow_api_key = QLineEdit(DEFAULT_ROBOFLOW_API_KEY)
        form.addRow("Roboflow API Key:", self.roboflow_api_key)

        self.roboflow_workspace = QLineEdit(DEFAULT_WORKSPACE)
        form.addRow("Workspace:", self.roboflow_workspace)

        self.roboflow_workflow_id = QLineEdit(DEFAULT_WORKFLOW_ID)
        form.addRow("Workflow ID:", self.roboflow_workflow_id)

        # Local inference configuration
        self.local_inference_url = QLineEdit(DEFAULT_LOCAL_INFERENCE_URL)
        form.addRow("Local Inference URL:", self.local_inference_url)

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

        # Initialize UI state
        self._on_model_type_changed("Local Model")

    def _on_model_type_changed(self, model_type: str):
        """Enable/disable model configuration fields based on selected type"""
        is_local = model_type == "Local Model"
        is_roboflow_cloud = model_type == "Roboflow Cloud API"
        is_roboflow_local = model_type == "Roboflow Local Inference"
        
        self.model_combo.setEnabled(is_local)
        self.roboflow_api_key.setEnabled(is_roboflow_cloud or is_roboflow_local)
        self.roboflow_workspace.setEnabled(is_roboflow_cloud or is_roboflow_local)
        self.roboflow_workflow_id.setEnabled(is_roboflow_cloud or is_roboflow_local)
        self.local_inference_url.setEnabled(is_roboflow_local)

    def _is_valid_rtmp(self, url: str) -> bool:
        return url.startswith("rtmp://") and len(url.split("/")) >= 3

    def _on_start_clicked(self):
        url = self.rtmp_input.text().strip()
        model_type = self.model_type_combo.currentText()
        com_port = self.com_input.text().strip()
        baud = int(self.baud_combo.currentText())

        if not self._is_valid_rtmp(url):
            QMessageBox.warning(
                self,
                "Invalid RTMP",
                "Please enter a valid RTMP URL (e.g., rtmp://host/live)",
            )
            return

        if not com_port:
            QMessageBox.warning(
                self, "Invalid COM Port", "Please enter a COM port (e.g., COM8)"
            )
            return

        # Validate based on model type
        if model_type == "Local Model":
            model = self.model_combo.currentText().strip()
            if not self._validate_model_file(model):
                QMessageBox.warning(
                    self,
                    "Model File Not Found",
                    f"The selected model file '{model}' could not be found.\n"
                    f"Please ensure the model file is in the app directory.",
                )
                return
            config = {
                "rtmp_url": url,
                "model_type": "local",
                "model": model,
                "com_port": com_port,
                "baud": baud,
            }
        elif model_type == "Roboflow Cloud API":
            if not ROBOFLOW_AVAILABLE:
                QMessageBox.warning(
                    self,
                    "Roboflow Not Available",
                    "Roboflow inference package is not installed.\n"
                    "Please install it with: pip install inference",
                )
                return
            
            api_key = self.roboflow_api_key.text().strip()
            workspace = self.roboflow_workspace.text().strip()
            workflow_id = self.roboflow_workflow_id.text().strip()
            
            if not api_key or not workspace or not workflow_id:
                QMessageBox.warning(
                    self,
                    "Missing Roboflow Configuration",
                    "Please fill in all Roboflow API fields (API Key, Workspace, Workflow ID).",
                )
                return
            
            config = {
                "rtmp_url": url,
                "model_type": "roboflow",
                "inference_type": "cloud",
                "api_key": api_key,
                "workspace": workspace,
                "workflow_id": workflow_id,
                "com_port": com_port,
                "baud": baud,
            }
        elif model_type == "Roboflow Local Inference":
            inference_url = self.local_inference_url.text().strip()
            workflow_id = self.roboflow_workflow_id.text().strip()
            workspace = self.roboflow_workspace.text().strip()
            api_key = self.roboflow_api_key.text().strip()
            
            if not inference_url or not workflow_id or not workspace or not api_key:
                QMessageBox.warning(
                    self,
                    "Missing Local Inference Configuration",
                    "Please fill in Local Inference URL, Workspace, Workflow ID, and API Key fields.",
                )
                return
            
            if not inference_url.startswith("http://") and not inference_url.startswith("https://"):
                QMessageBox.warning(
                    self,
                    "Invalid URL",
                    "Please enter a valid URL starting with http:// or https://",
                )
                return
            
            config = {
                "rtmp_url": url,
                "model_type": "roboflow",
                "inference_type": "local",
                "inference_url": inference_url,
                "api_key": api_key,
                "workspace": workspace,
                "workflow_id": workflow_id,
                "com_port": com_port,
                "baud": baud,
            }

        self.on_start_callback(config)

    def _find_available_models(self) -> list:
        """Find all .pt model files in the app directory"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_files = []
            
            for file in os.listdir(script_dir):
                if file.endswith('.pt'):
                    model_files.append(file)
            
            model_files.sort()
            if "crack4.pt" in model_files:
                model_files.remove("crack4.pt")
                model_files.insert(0, "crack4.pt")
                
            return model_files
        except Exception:
            return []

    def _validate_model_file(self, model_name: str) -> bool:
        """Check if the model file exists in the app directory"""
        if not model_name:
            return False
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_name)
        return os.path.exists(model_path)


class MainScreen(QWidget):
    def __init__(self):
        super().__init__()

        # Left: video placeholder
        self.video_label = QLabel("Video Stream")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedHeight(480)
        self.video_label.setStyleSheet(
            "background-color: #202020; color: #CCCCCC; border: 1px solid #404040;"
        )

        # Right: Performance charts and class counts
        right_col = QVBoxLayout()

        # Performance charts section
        charts_label = QLabel("Performance Metrics")
        charts_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        charts_label.setAlignment(Qt.AlignCenter)
        right_col.addWidget(charts_label)

        # Create performance charts
        self.fps_chart = PerformanceChart("FPS", y_max=60.0, y_min=0.0)
        self.cpu_chart = PerformanceChart("CPU %", y_max=100.0, y_min=0.0)
        self.mem_chart = PerformanceChart("Memory %", y_max=100.0, y_min=0.0)
        self.gpu_chart = PerformanceChart("GPU %", y_max=100.0, y_min=0.0)

        # Add charts to layout (make them smaller to fit)
        charts_layout = QVBoxLayout()
        charts_layout.addWidget(self.fps_chart)
        charts_layout.addWidget(self.cpu_chart)
        charts_layout.addWidget(self.mem_chart)
        charts_layout.addWidget(self.gpu_chart)

        # Make charts smaller
        for chart in [self.fps_chart, self.cpu_chart, self.mem_chart, self.gpu_chart]:
            chart.setFixedHeight(120)

        right_col.addLayout(charts_layout)

        # Class counts section
        class_label = QLabel("Class Counts")
        class_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        class_label.setAlignment(Qt.AlignCenter)
        right_col.addWidget(class_label)

        self.class_list = QListWidget()
        self.class_list.addItem(QListWidgetItem("person: 0"))
        right_col.addWidget(self.class_list)

        right_col.addStretch(1)

        # Add stop button at bottom
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.setStyleSheet(
            "background-color: #d32f2f; color: white; padding: 8px;"
        )
        right_col.addWidget(self.stop_btn)

        top = QHBoxLayout()
        top.addWidget(self.video_label, 3)
        top.addLayout(right_col, 1)

        self.setLayout(top)

    def update_metrics(
        self, fps: float = None, cpu: float = None, mem: float = None, gpu: float = None
    ):
        """Update performance charts with new values"""
        if fps is not None:
            self.fps_chart.update_value(fps)
        if cpu is not None:
            self.cpu_chart.update_value(cpu)
        if mem is not None:
            self.mem_chart.update_value(mem)
        if gpu is not None:
            self.gpu_chart.update_value(gpu)

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
        from PySide6.QtGui import QImage, QPixmap, QPainter

        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.video_label.setPixmap(pix)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone YOLO")
        self.resize(1400, 800)

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
        self.worker_thread: Optional[WorkerThread] = None
        
        # Performance monitoring
        self.cpu = 0.0
        self.mem = 0.0
        self.gpu = 0.0
        
        # UI update timer - slower to prevent lag
        self.timer = QTimer(self)
        self.timer.setInterval(100)  # 10 Hz UI refresh for stability
        self.timer.timeout.connect(self._tick)
        
        # Performance monitoring timer
        self.perf_timer = QTimer(self)
        self.perf_timer.setInterval(1000)  # 1 Hz performance monitoring
        self.perf_timer.timeout.connect(self._update_performance)
        
        # Start memory tracing
        tracemalloc.start()

    def _start_clicked(self, config: dict):
        """Start detection with proper thread management"""
        self.config = config
        
        # Create and start worker thread
        self.worker_thread = WorkerThread(config)
        self.worker_thread.frame_ready.connect(self._on_frame_ready)
        self.worker_thread.status_update.connect(self._on_status_update)
        self.worker_thread.start()
        
        # Switch to main screen
        self.stack.setCurrentWidget(self.main)
        self.main.add_stop_button(self._stop_detection)
        
        # Start timers
        self.timer.start()
        self.perf_timer.start()

    def _stop_detection(self):
        """Stop detection with graceful cleanup"""
        if self.worker_thread is not None:
            self.worker_thread.stop()
            self.worker_thread = None
            
        self.timer.stop()
        self.perf_timer.stop()
        self.stack.setCurrentWidget(self.home)

    def _on_frame_ready(self, frame: np.ndarray, counts: dict):
        """Handle frame from worker thread"""
        self.main.set_frame(frame)
        self.main.update_class_counts(counts)

    def _on_status_update(self, message: str):
        """Handle status updates from worker thread"""
        print(f"Status: {message}")

    def _update_performance(self):
        """Update performance metrics"""
        try:
            self.cpu = psutil.cpu_percent()
            self.mem = psutil.virtual_memory().percent
            
            if torch.cuda.is_available():
                self.gpu = torch.cuda.utilization()
            else:
                self.gpu = 0.0
                
            # Get FPS from worker thread
            fps = 0.0
            if self.worker_thread:
                fps = self.worker_thread.fps
                
            self.main.update_metrics(fps=fps, cpu=self.cpu, mem=self.mem, gpu=self.gpu)
            
        except Exception as e:
            print(f"Performance update error: {e}")

    def _tick(self):
        """Main UI update loop"""
        # This is now handled by signals from worker thread
        pass

    def closeEvent(self, event):
        """Handle application close with graceful cleanup"""
        print("🛑 Application closing...")
        
        # Stop detection if running
        if self.worker_thread is not None:
            print("🛑 Stopping detection...")
            self.worker_thread.stop()
            self.worker_thread = None
            
        # Stop timers
        self.timer.stop()
        self.perf_timer.stop()
        
        # Print memory usage
        try:
            current, peak = tracemalloc.get_traced_memory()
            print(f"📊 Memory usage - Current: {current / 1024 / 1024:.1f} MB, Peak: {peak / 1024 / 1024:.1f} MB")
        except:
            pass
            
        # Print active threads
        try:
            active_threads = threading.active_count()
            print(f"🧵 Active threads: {active_threads}")
        except:
            pass
            
        # Force cleanup
        gc.collect()
        
        print("✅ Application closed gracefully")
        super().closeEvent(event)


def run():
    app = QApplication(sys.argv)
    
    # Set up signal handling for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        app.quit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    window = MainWindow()
    window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        print("✅ Application terminated")


if __name__ == "__main__":
    run()
