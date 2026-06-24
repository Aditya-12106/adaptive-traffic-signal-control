"""
dashboard.py
Renders the live controller status overlay onto a composite display frame.

In the multi-video architecture the dashboard receives:
  - controller_status   : from AdaptiveTrafficController.status()
  - approach_demands    : {north/south/east/west: float}  aggregated demand
  - per_approach_counts : {approach: Counter}  vehicle counts per feed
  - total_vehicles      : int  sum across all feeds

The panel is drawn into the top-left corner of whatever frame is passed in
(typically the composite tiled frame built in main.py).
"""

from collections import Counter
from typing import Dict

import cv2

LINE_HEIGHT  = 22
PANEL_WIDTH  = 360

_STATE_COLORS = {
    "GREEN":  (0, 255, 0),
    "YELLOW": (0, 255, 255),
    "IDLE":   (0, 0, 255),
}

_APPROACH_COLORS = {
    "north": (102, 204, 255),
    "south": (102, 255, 153),
    "east":  (255, 178, 102),
    "west":  (255, 102, 178),
}


def _put_line(frame, text, x, y,
              color=(255, 255, 255), scale=0.5, thickness=1):
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_dashboard(frame,
                   controller_status:   dict,
                   approach_demands:    Dict[str, float],
                   per_approach_counts: Dict[str, Counter],
                   total_vehicles:      int) -> None:
    """Draw a semi-transparent info panel onto `frame` (in place).

    Args:
        frame               : BGR numpy array to draw on.
        controller_status   : dict from AdaptiveTrafficController.status().
        approach_demands    : aggregated demand per approach.
        per_approach_counts : vehicle class counts per approach feed.
        total_vehicles      : total vehicle count across all feeds.
    """
    panel_h = min(frame.shape[0] - 10, 460)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + PANEL_WIDTH, 5 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    x, y = 15, 25
    state = controller_status["state"]

    # --- Controller state ---
    phase_label = (f"Phase {controller_status['active_phase']}  "
                   f"({controller_status['phase_name'] or 'IDLE'})")
    _put_line(frame, phase_label, x, y, (0, 255, 255), scale=0.52, thickness=1)
    y += LINE_HEIGHT

    _put_line(frame, f"Signal: {state}", x, y,
              _STATE_COLORS.get(state, (255, 255, 255)))
    y += LINE_HEIGHT

    _put_line(frame, f"Green remaining : {controller_status['remaining_green']}s", x, y)
    y += LINE_HEIGHT
    _put_line(frame, f"Yellow remaining: {controller_status['remaining_yellow']}s", x, y)
    y += LINE_HEIGHT

    # When idle, show which phase the round-robin will resume after
    if state == "IDLE":
        last = controller_status.get("last_active_phase")
        resume_label = f"P{last}" if last is not None else "start"
        _put_line(frame, f"Idle — will resume after {resume_label}", x, y,
                  (160, 160, 160))
    y += int(LINE_HEIGHT * 0.6)

    # --- Phase demand breakdown ---
    _put_line(frame, "Phase Demand:", x, y, (200, 200, 0))
    y += LINE_HEIGHT
    for phase_id, demand in controller_status["phase_demands"].items():
        skipped = " (skip)" if phase_id in controller_status["skipped_phases"] else ""
        _put_line(frame, f"  Ph{phase_id}: {demand:.1f}{skipped}", x, y)
        y += LINE_HEIGHT
    y += LINE_HEIGHT // 2

    # --- Per-approach demand + vehicle counts ---
    _put_line(frame, "Approach Detail:", x, y, (200, 200, 0))
    y += LINE_HEIGHT
    for approach in ("north", "south", "east", "west"):
        col = _APPROACH_COLORS.get(approach, (255, 255, 255))
        demand  = approach_demands.get(approach, 0.0)
        counts  = per_approach_counts.get(approach, Counter())
        total_a = sum(counts.values())
        _put_line(frame,
                  f"  {approach.capitalize()}: demand={demand:.1f}  vehicles={total_a}",
                  x, y, col)
        y += int(LINE_HEIGHT * 0.9)
        if counts:
            detail = "  " + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            _put_line(frame, detail, x, y, col, scale=0.4)
            y += int(LINE_HEIGHT * 0.85)
        else:
            _put_line(frame, "    (none detected)", x, y, col, scale=0.4)
            y += int(LINE_HEIGHT * 0.85)
    y += LINE_HEIGHT // 2

    _put_line(frame, f"Total vehicles (all feeds): {total_vehicles}",
              x, y, (0, 255, 255))


# ---------------------------------------------------------------------------
# Signal-state indicator overlay (drawn on each individual approach tile)
# ---------------------------------------------------------------------------

def draw_approach_signal(frame, approach: str, signal_snapshot: Dict[str, str]):
    """Draw a small coloured signal-state badge in the top-right corner of
    a single approach frame, showing GREEN / YELLOW / RED for that approach's
    two movements (left and straight_right).

    Args:
        frame           : BGR numpy array for one approach tile.
        approach        : "north" / "south" / "east" / "west".
        signal_snapshot : from TrafficLight.snapshot(), e.g.
                          {"north_left": "GREEN", "north_straight_right": "RED", ...}
    """
    _SIGNAL_BGR = {
        "GREEN":  (0, 200, 0),
        "YELLOW": (0, 200, 200),
        "RED":    (0, 0, 200),
    }
    h, w = frame.shape[:2]
    movements = [f"{approach}_left", f"{approach}_straight_right"]
    labels    = ["L", "S/R"]

    box_w, box_h = 54, 20
    x0 = w - box_w - 6
    y0 = 6

    bg = frame.copy()
    cv2.rectangle(bg, (x0 - 4, y0 - 2),
                  (x0 + box_w + 4, y0 + len(movements) * box_h + 4),
                  (20, 20, 20), -1)
    cv2.addWeighted(bg, 0.55, frame, 0.45, 0, frame)

    for i, (mov, lbl) in enumerate(zip(movements, labels)):
        state = signal_snapshot.get(mov, "RED")
        color = _SIGNAL_BGR.get(state, (128, 128, 128))
        cy = y0 + i * box_h + box_h // 2
        cv2.circle(frame, (x0 + 8, cy), 7, color, -1)
        cv2.putText(frame, f"{lbl}: {state}", (x0 + 18, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
