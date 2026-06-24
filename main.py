"""
main.py
Entry point for the multi-video Adaptive Traffic Signal Control System.

Architecture:
    Four video feeds (North / South / East / West) are read in lockstep,
    one frame at a time. A single shared YOLOv8 model runs detection on
    each frame sequentially. Each approach has its own ROIManager with
    camera-specific polygon coordinates. Demand is aggregated across all
    four approaches and fed to a single AdaptiveTrafficController that
    drives the 4-phase signal logic.

    Output: a 2×2 tiled composite window (and optionally an output video)
    showing all four annotated feeds simultaneously, plus a controller
    dashboard panel.

Usage:
    python main.py

    Or override individual video paths on the command line:
    python main.py --north videos/north.mp4 --south videos/south.mp4 \
                   --east  videos/east.mp4  --west  videos/west.mp4

Video files default to those specified in config.APPROACHES.
"""

import argparse
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
import dashboard as dash
from controller import AdaptiveTrafficController
from demand_calculator import DemandCalculator
from detector import Detection, VehicleDetector
from roi_manager import ROIManager

# ---------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------- #

APPROACHES   = ["north", "south", "east", "west"]
TILE_W, TILE_H = 640, 360       # each approach tile in the composite display
COMPOSITE_W  = TILE_W * 2       # 1280
COMPOSITE_H  = TILE_H * 2       # 720

_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car":        (0, 200, 0),
    "truck":      (0, 140, 255),
    "bus":        (255, 0, 255),
    "motorcycle": (255, 255, 0),
}

# Tile positions in the 2×2 composite: top-left corner (x, y)
_TILE_ORIGINS: Dict[str, Tuple[int, int]] = {
    "north": (0,      0),
    "south": (TILE_W, 0),
    "east":  (0,      TILE_H),
    "west":  (TILE_W, TILE_H),
}

# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def draw_detections(frame, lane_buckets: Dict[str, List[Detection]]):
    """Draw bounding boxes + class labels for every detection in the frame."""
    for detections in lane_buckets.values():
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = _CLASS_COLORS.get(det.class_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{det.class_name} {det.confidence:.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def make_composite(tiles: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
    """Assemble the four approach tiles into a 2×2 composite frame.

    Each tile is resized to TILE_W × TILE_H. If a feed has ended or failed
    to open, a black placeholder with a label is used instead.
    """
    composite = np.zeros((COMPOSITE_H, COMPOSITE_W, 3), dtype=np.uint8)
    for approach, (tx, ty) in _TILE_ORIGINS.items():
        frame = tiles.get(approach)
        if frame is None:
            tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
            cv2.putText(tile, f"{approach.upper()} — no feed",
                        (TILE_W // 2 - 80, TILE_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        else:
            tile = cv2.resize(frame, (TILE_W, TILE_H))
        composite[ty:ty + TILE_H, tx:tx + TILE_W] = tile
    return composite


def open_capture(path: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {path}  — approach will be skipped.")
        return None
    return cap


# ---------------------------------------------------------------------- #
# Main loop
# ---------------------------------------------------------------------- #

def main(video_overrides: Optional[Dict[str, str]] = None):
    # ---- Build per-approach components ----
    approach_cfgs = {}
    for approach in APPROACHES:
        cfg      = config.APPROACHES[approach]
        vid_path = (video_overrides or {}).get(approach, cfg["video"])
        approach_cfgs[approach] = {
            "video":       vid_path,
            "roi_manager": ROIManager(cfg["rois"], approach=approach),
        }

    # ---- Single shared detector (model loaded once) ----
    detector    = VehicleDetector()
    demand_calc = DemandCalculator()
    controller  = AdaptiveTrafficController()

    # ---- Open all video captures ----
    caps: Dict[str, Optional[cv2.VideoCapture]] = {
        a: open_capture(approach_cfgs[a]["video"]) for a in APPROACHES
    }

    # Determine master FPS from the first available capture
    fps = config.SOURCE_FPS
    for cap in caps.values():
        if cap is not None:
            fps = cap.get(cv2.CAP_PROP_FPS) or config.SOURCE_FPS
            break
    dt = 1.0 / fps

    # ---- Optional composite output writer ----
    writer = None
    if config.SAVE_OUTPUT_VIDEO:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        composite_path = os.path.join(config.OUTPUT_DIR, "composite_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            composite_path, fourcc, fps, (COMPOSITE_W, COMPOSITE_H))
        print(f"Recording composite output to: {composite_path}")

    # ---- Per-approach state (lane buckets persist across skipped frames) ----
    lane_buckets_all: Dict[str, Dict[str, List[Detection]]] = {
        a: {lane: [] for lane in approach_cfgs[a]["roi_manager"].roi_polygons}
        for a in APPROACHES
    }

    frame_idx = 0
    active_caps = {a for a, cap in caps.items() if cap is not None}

    print(f"Starting — {len(active_caps)} active feed(s): {sorted(active_caps)}")
    print("Press 'q' to quit.")

    while active_caps:
        tiles: Dict[str, Optional[np.ndarray]] = {}
        finished_this_frame: List[str] = []

        for approach in APPROACHES:
            cap = caps.get(approach)

            # ----- Feed is unavailable / already finished -----
            if cap is None or approach not in active_caps:
                tiles[approach] = None
                continue

            ok, frame = cap.read()
            if not ok:
                print(f"[INFO] Feed '{approach}' ended at frame {frame_idx}.")
                cap.release()
                caps[approach] = None
                active_caps.discard(approach)
                tiles[approach] = None
                finished_this_frame.append(approach)
                continue

            roi_mgr = approach_cfgs[approach]["roi_manager"]

            # ----- Detection (every Nth frame; buckets held otherwise) -----
            if frame_idx % config.DETECTION_FRAME_SKIP == 0:
                detections = detector.detect(frame)
                lane_buckets_all[approach] = roi_mgr.assign_batch(detections)

            # ----- Annotate the individual approach frame -----
            roi_mgr.draw_rois(frame)
            draw_detections(frame, lane_buckets_all[approach])

            # Label the tile with the approach name
            cv2.putText(frame, approach.upper(), (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            tiles[approach] = frame

        # ----- Aggregate demand across all active approaches -----
        merged_lane_demands: Dict[str, float] = {}
        for approach in APPROACHES:
            lane_demands_a = demand_calc.lane_demands(lane_buckets_all[approach])
            merged_lane_demands.update(lane_demands_a)

        phase_demands    = demand_calc.phase_demands(merged_lane_demands)
        approach_demands = demand_calc.approach_demands(merged_lane_demands)

        per_approach_counts: Dict[str, Counter] = {
            a: demand_calc.vehicle_counts(lane_buckets_all[a]) for a in APPROACHES
        }
        total_vehicles = sum(
            sum(c.values()) for c in per_approach_counts.values()
        )

        # ----- Drive the controller -----
        controller.update(dt, phase_demands)
        ctrl_status      = controller.status()
        signal_snapshot  = controller.light.snapshot()

        # ----- Draw per-approach signal badge on each tile -----
        for approach in APPROACHES:
            if tiles[approach] is not None:
                dash.draw_approach_signal(tiles[approach], approach, signal_snapshot)

        # ----- Assemble composite & draw dashboard panel -----
        composite = make_composite(tiles)
        dash.draw_dashboard(
            composite,
            ctrl_status,
            approach_demands,
            per_approach_counts,
            total_vehicles,
        )

        # ----- Write & display -----
        if writer is not None:
            print("Writing frame", frame_idx)
            writer.write(composite)

        cv2.imshow("Adaptive Traffic Signal Controller", composite)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit signal received.")
            break

        frame_idx += 1

    # ---- Cleanup ----
    for cap in caps.values():
        if cap is not None:
            cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_idx} frame(s).")
    if writer is not None:
        print(f"Composite output saved to: {os.path.join(config.OUTPUT_DIR, 'composite_output.mp4')}")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Traffic Signal Control System — multi-video mode")
    for approach in APPROACHES:
        default = config.APPROACHES[approach]["video"]
        parser.add_argument(
            f"--{approach}",
            default=default,
            metavar="PATH",
            help=f"Path to {approach} approach video (default: {default})",
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = {a: getattr(args, a) for a in APPROACHES}
    main(video_overrides=overrides)
