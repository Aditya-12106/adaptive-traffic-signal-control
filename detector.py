"""
detector.py
Thin wrapper around a YOLOv8 model that runs inference on a single BGR
frame and filters results down to the four vehicle classes used by this
project (car, bus, truck, motorcycle).

One shared VehicleDetector instance is used across all four approach feeds
in main.py so the model weights are loaded only once.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import config


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    class_name: str
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


class VehicleDetector:
    """Wraps a YOLOv8 model and returns only vehicle-class detections."""

    def __init__(self,
                 model_path: str = config.YOLO_MODEL_PATH,
                 confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
                 device: Optional[str] = None):
        # Lazy import so the rest of the code can be unit-tested without
        # ultralytics / torch installed.
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect(self, frame) -> List[Detection]:
        """Run detection on a single BGR numpy frame and return only
        vehicle-class detections above the confidence threshold."""
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in config.VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_name=config.VEHICLE_CLASSES[cls_id],
                    confidence=conf,
                )
            )
        return detections
