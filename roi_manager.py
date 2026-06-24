"""
roi_manager.py
Per-approach ROI management. Each approach (North / South / East / West)
gets its own ROIManager instance initialised with the two lane-group
polygons calibrated for that camera's angle and resolution.

The polygon coordinates live in config.APPROACHES[approach]["rois"] and
must be calibrated independently for each camera — they will not match
across different video feeds even if the intersection is the same.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector import Detection


class ROIManager:
    """Manages the ROI polygons for a single approach feed."""

    def __init__(self,
                 roi_polygons: Dict[str, List[Tuple[int, int]]],
                 approach: str = ""):
        """
        Args:
            roi_polygons: dict mapping lane group name -> list of (x,y) vertices,
                          e.g. {"north_left": [(x,y),...], "north_straight_right": [...]}
            approach:     human-readable label used in draw_rois() labels.
        """
        self.roi_polygons = roi_polygons
        self.approach = approach
        # Pre-convert to numpy int32 arrays for cv2.pointPolygonTest
        self._np_polygons: Dict[str, np.ndarray] = {
            name: np.array(pts, dtype=np.int32)
            for name, pts in roi_polygons.items()
        }

    # ------------------------------------------------------------------ #
    # Detection → lane assignment
    # ------------------------------------------------------------------ #

    def assign(self, detection: Detection) -> Optional[str]:
        """Return the lane-group name whose ROI contains the detection
        center, or None if it falls outside every ROI defined for this
        approach."""
        cx, cy = detection.center
        for name, polygon in self._np_polygons.items():
            if cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0:
                return name
        return None

    def assign_batch(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """Assign a list of detections to lane-group buckets.

        Every configured lane group is guaranteed to appear as a key
        (empty list if no detections). Detections outside all ROIs are
        silently dropped (e.g. vehicles on the far side of the intersection
        that are visible in the frame but belong to another approach).
        """
        buckets: Dict[str, List[Detection]] = {name: [] for name in self.roi_polygons}
        for det in detections:
            lane = self.assign(det)
            if lane is not None:
                buckets[lane].append(det)
        return buckets

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #

    def draw_rois(self,
                  frame,
                  color: Tuple[int, int, int] = (0, 255, 255),
                  thickness: int = 2):
        """Draw ROI polygon outlines and lane-group labels onto `frame`
        in place. Call this on each frame to verify calibration."""
        for name, polygon in self._np_polygons.items():
            cv2.polylines(frame, [polygon], isClosed=True,
                          color=color, thickness=thickness)
            label_pt = (int(polygon[0][0]), max(14, int(polygon[0][1]) - 5))
            cv2.putText(frame, name, label_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return frame
