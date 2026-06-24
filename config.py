"""
config.py
Central configuration for the Adaptive Traffic Signal Control System.

Architecture: 4 independent video feeds, one per approach (North / South /
East / West). Each feed has its own ROI polygons calibrated to that camera's
angle. The adaptive controller receives per-phase demand aggregated across
all four feeds.
"""

# ---------------------------------------------------------------------------
# Vehicle detection
# ---------------------------------------------------------------------------

# COCO class IDs for the vehicle types we care about.
# car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CONFIDENCE_THRESHOLD = 0.35

# "yolov8n.pt" auto-downloads on first use. Point at custom weights here.
YOLO_MODEL_PATH = "yolov8n.pt"

# ---------------------------------------------------------------------------
# Weighted demand calculation
# ---------------------------------------------------------------------------

VEHICLE_WEIGHTS = {
    "car":        1.0,
    "truck":      2.0,
    "bus":        2.0,
    "motorcycle": 0.5,
}

# ---------------------------------------------------------------------------
# Per-approach video feeds + ROI polygons
#
# Each entry in APPROACHES maps an approach name to:
#   video   : path to the video file for that approach
#   rois    : dict of lane_group -> polygon (list of (x,y) pixel coords)
#             calibrated for THAT camera's resolution & angle.
#
# Two lane groups per approach:
#   <approach>_left            — dedicated left-turn lane
#   <approach>_straight_right  — through + right lane(s)
#
# HOW TO CALIBRATE:
#   1. Grab a still frame:
#        python3 -c "
#          import cv2, sys
#          cap = cv2.VideoCapture(sys.argv[1])
#          ok, f = cap.read()
#          cv2.imwrite('calib_frame.png', f)
#        " videos/north.mp4
#   2. Open calib_frame.png in any image viewer; hover over polygon corners
#      to read pixel (x, y) coordinates.
#   3. Replace the PLACEHOLDER values below with your real coordinates.
#   4. Run python3 main.py and verify the drawn polygons align with the lanes.
# ---------------------------------------------------------------------------

APPROACHES = {
    "north": {
        "video": "videos/north.avi",
        "rois": {
            "north_left":           [(838,467), (852,256), (956,285), (1049,461)],
            "north_straight_right": [(838,467), (859,204), (779, 204), (744,225),(656,280),(374,475)],
        },
    },
    "south": {
        "video": "videos/south.avi",
        "rois": {
            "south_left":           [(806,327), (824,164), (926,194), (997,321)],
            "south_straight_right": [(407,362), (698,143), (824,143), (806,327)],
        },
    },
    "east": {
        "video": "videos/east.avi",
        "rois": {
            "east_left":           [(1229,401), (1210,345) ,(1136,263), (1094,234), (936,184),(962,212),(1006, 256),(1033,306),(1042, 385)],
            "east_straight_right": [(1042, 385), (1032,308), (1011, 256), (977,218), (919,186), (897,172), (827,150),(729,126),(694,134),(735, 150),(794,180),(820,196),(840,234),(845,327),(762,365),(918,376)],
        },
    },
    "west": {
        "video": "videos/west.avi",
        "rois": {
            "west_left":           [(1053, 399), (1009,366), (979,350), (961,332), (949, 316), (953, 300), (978,275), (1000,277), (1049,289), (1054,311), (1083,328), (1238,395)],
            "west_straight_right": [(1053, 399), (1009,366), (979,350), (961,332), (949, 316), (953, 300), (978,275), (919,271), (849,295), (815,314), (773,401)],
        },
    },
}

# Derived flat look-ups (built automatically — do not edit manually).
# Maps every lane group name -> approach name, e.g. "north_left" -> "north"
LANE_TO_APPROACH: dict = {}
# Flat dict of all ROI polygons across all approaches (used by ROIManager
# when it needs the global set, e.g. for demand aggregation).
ALL_ROI_POLYGONS: dict = {}
for _approach, _cfg in APPROACHES.items():
    for _lane, _poly in _cfg["rois"].items():
        LANE_TO_APPROACH[_lane] = _approach
        ALL_ROI_POLYGONS[_lane] = _poly

# ---------------------------------------------------------------------------
# Signal phases
# ---------------------------------------------------------------------------

PHASES = {
    1: {
        "name":      "North-South Straight + Right",
        "lanes":     ["north_straight_right", "south_straight_right"],
        "movements": {"north": "straight_right", "south": "straight_right"},
    },
    2: {
        "name":      "North-South Left Turn",
        "lanes":     ["north_left", "south_left"],
        "movements": {"north": "left", "south": "left"},
    },
    3: {
        "name":      "East-West Straight + Right",
        "lanes":     ["east_straight_right", "west_straight_right"],
        "movements": {"east": "straight_right", "west": "straight_right"},
    },
    4: {
        "name":      "East-West Left Turn",
        "lanes":     ["east_left", "west_left"],
        "movements": {"east": "left", "west": "left"},
    },
}

PHASE_ORDER = [1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Signal timing
# ---------------------------------------------------------------------------

MIN_GREEN       = 15    # seconds
MAX_GREEN       = 60    # seconds
YELLOW_TIME     = 5     # seconds (fixed)
DEMAND_FACTOR_K = 1.0   # green_time = min(MAX_GREEN, MIN_GREEN + k * demand)

# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

# Output directory for per-approach annotated videos.
OUTPUT_DIR = "outputs"
SAVE_OUTPUT_VIDEO = True

# Run YOLO every Nth frame. 1 = every frame. Higher = faster but laggier.
DETECTION_FRAME_SKIP = 1

# Fallback FPS when it cannot be read from the video file header.
SOURCE_FPS = 30
