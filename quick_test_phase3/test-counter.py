# test_yolo5_counter.py
import torch
from class_counter import ClassCounter


class FakeYOLOv5Result:
    def __init__(self):
        self.names = {0: "person", 1: "car", 2: "dog"}
        # 3 detections: person, car, car
        self.xyxy = [
            torch.tensor(
                [
                    [0, 0, 10, 10, 0.9, 0],  # person
                    [0, 0, 10, 10, 0.8, 1],  # car
                    [0, 0, 10, 10, 0.7, 1],  # car
                ]
            )
        ]


counter = ClassCounter()
results = FakeYOLOv5Result()
counter.update_from_yolo(results)
print("Counts:", counter.get_counts())
