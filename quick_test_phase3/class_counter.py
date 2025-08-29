# class_counter.py
from collections import Counter
import torch


class ClassCounter:
    """
    Tracks YOLO detections per frame.
    Supports YOLOv5 or YOLOv8/11 outputs.
    """

    def __init__(self, accumulate: bool = False):
        self.accumulate = accumulate
        self.counter = Counter()

    def update_from_yolo(self, results):
        """
        Update the counter using YOLO results object.
        Handles YOLOv5 and YOLOv8/11.
        """

        if results is None:
            return

        # ---- YOLOv5 (results.xyxy, results.names) ----
        if hasattr(results, "xyxy"):
            names = results.names
            detections = results.xyxy[0]  # (N, 6): [x1, y1, x2, y2, conf, cls]
            cls_ids = detections[:, -1].cpu().numpy().astype(int)
            labels = [names[int(c)] for c in cls_ids]

        # ---- YOLOv8/11 (results[0].boxes.cls, results[0].names) ----
        elif isinstance(results, list) and hasattr(results[0], "boxes"):
            names = results[0].names
            classes = results[0].boxes.cls
            if isinstance(classes, torch.Tensor):
                classes = classes.cpu().numpy().astype(int)
            labels = [names[int(c)] for c in classes]

        else:
            return  # unsupported format

        if not self.accumulate:
            self.counter = Counter()  # reset per frame

        self.counter.update(labels)

    def get_counts(self):
        """Return dict with class counts, e.g. {"person": 3, "car": 1}"""
        return dict(self.counter)

    def reset(self):
        """Manually reset counts (useful if accumulate=True)."""
        self.counter = Counter()
