"""
test_logic.py
Sanity-checks the full pipeline using synthetic detections only.
No dependency on ultralytics / torch or real video files required.

Run with:
    python test_logic.py

Tests cover:
    1. Per-approach ROI assignment
    2. Cross-approach demand merging
    3. Weighted demand formula
    4. Phase demand aggregation
    5. Adaptive green time formula
    6. Controller cycle: GREEN → YELLOW → next phase (phase skipping)
    7. IDLE entry and exit
    8. Round-robin wraparound
"""

from collections import Counter

import config
from controller import AdaptiveTrafficController, ControllerState
from demand_calculator import DemandCalculator
from detector import Detection
from roi_manager import ROIManager


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def make_roi_manager(approach: str) -> ROIManager:
    return ROIManager(config.APPROACHES[approach]["rois"], approach=approach)


def make_detection_at(polygon_pts, class_name: str, confidence: float = 0.9) -> Detection:
    """Return a Detection whose center sits at the centroid of the polygon."""
    xs = [p[0] for p in polygon_pts]
    ys = [p[1] for p in polygon_pts]
    cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)
    return Detection(bbox=(cx - 10, cy - 10, cx + 10, cy + 10),
                     class_name=class_name, confidence=confidence)


def make_detection_in_lane(approach: str, lane: str,
                            class_name: str, confidence: float = 0.9) -> Detection:
    """Return a Detection whose center is inside the given approach + lane ROI."""
    poly = config.APPROACHES[approach]["rois"][lane]
    return make_detection_at(poly, class_name, confidence)


def check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, f"FAILED: {label}"


# ---------------------------------------------------------------------- #
# Test 1 — Per-approach ROI assignment
# ---------------------------------------------------------------------- #

def test_roi_assignment():
    print("\n-- 1. Per-approach ROI assignment --")
    for approach in ("north", "south", "east", "west"):
        roi_mgr = make_roi_manager(approach)
        rois    = config.APPROACHES[approach]["rois"]

        lane_left = f"{approach}_left"
        lane_sr   = f"{approach}_straight_right"

        det_left = make_detection_in_lane(approach, lane_left,   "car")
        det_sr   = make_detection_in_lane(approach, lane_sr, "truck")

        check(f"{approach}: det in {lane_left} assigned correctly",
              roi_mgr.assign(det_left) == lane_left)
        check(f"{approach}: det in {lane_sr} assigned correctly",
              roi_mgr.assign(det_sr) == lane_sr)

        outside = Detection(bbox=(-50, -50, -40, -40),
                            class_name="car", confidence=0.9)
        check(f"{approach}: det outside all ROIs → None",
              roi_mgr.assign(outside) is None)

        batch   = [det_left, det_left, det_sr]
        buckets = roi_mgr.assign_batch(batch)
        check(f"{approach}: batch bucket counts correct",
              len(buckets[lane_left]) == 2 and len(buckets[lane_sr]) == 1)
        check(f"{approach}: all configured lanes present as keys",
              set(buckets.keys()) == set(rois.keys()))


# ---------------------------------------------------------------------- #
# Test 2 — Cross-approach demand merging
# ---------------------------------------------------------------------- #

def test_cross_approach_demand_merge():
    print("\n-- 2. Cross-approach demand merging --")
    calc = DemandCalculator()

    # Build synthetic lane_buckets for each approach
    merged: dict = {}
    for approach in ("north", "south", "east", "west"):
        roi_mgr = make_roi_manager(approach)
        lane_sr = f"{approach}_straight_right"
        # 2 cars in straight_right for every approach
        dets    = [make_detection_in_lane(approach, lane_sr, "car")] * 2
        buckets = roi_mgr.assign_batch(dets)
        merged.update(calc.lane_demands(buckets))

    # Phase 1 = north_straight_right + south_straight_right = 2+2 = 4 cars = 4.0
    phase_demands = calc.phase_demands(merged)
    check(f"Phase 1 demand = 4.0 (2 cars N + 2 cars S), got {phase_demands[1]}",
          phase_demands[1] == 4.0)
    check(f"Phase 3 demand = 4.0 (2 cars E + 2 cars W), got {phase_demands[3]}",
          phase_demands[3] == 4.0)
    check("Phase 2 demand = 0.0 (no left-turn vehicles)", phase_demands[2] == 0.0)
    check("Phase 4 demand = 0.0 (no left-turn vehicles)", phase_demands[4] == 0.0)

    approach_demands = calc.approach_demands(merged)
    for a in ("north", "south", "east", "west"):
        check(f"{a} approach demand = 2.0, got {approach_demands[a]}",
              approach_demands[a] == 2.0)


# ---------------------------------------------------------------------- #
# Test 3 — Weighted demand formula
# ---------------------------------------------------------------------- #

def test_demand_formula():
    print("\n-- 3. Weighted demand formula --")
    calc = DemandCalculator()

    dets = [
        Detection((0, 0, 1, 1), "car",        0.9),   # 1.0
        Detection((0, 0, 1, 1), "car",        0.9),   # 1.0
        Detection((0, 0, 1, 1), "truck",      0.9),   # 2.0
        Detection((0, 0, 1, 1), "motorcycle", 0.9),   # 0.5
    ]
    demand = calc.lane_demand(dets)
    check(f"2 cars + 1 truck + 1 moto = 4.5, got {demand}", demand == 4.5)

    check("Bus weight = 2.0",
          calc.lane_demand([Detection((0,0,1,1),"bus",0.9)]) == 2.0)
    check("Empty lane = 0.0",
          calc.lane_demand([]) == 0.0)


# ---------------------------------------------------------------------- #
# Test 4 — Vehicle counts helper
# ---------------------------------------------------------------------- #

def test_vehicle_counts():
    print("\n-- 4. Vehicle counts helper --")
    calc = DemandCalculator()

    roi_mgr_n = make_roi_manager("north")
    roi_mgr_s = make_roi_manager("south")

    dets_n = [make_detection_in_lane("north", "north_straight_right", "car")] * 3
    dets_s = [make_detection_in_lane("south", "south_left",           "bus")] * 2

    counts_n = calc.vehicle_counts(roi_mgr_n.assign_batch(dets_n))
    counts_s = calc.vehicle_counts(roi_mgr_s.assign_batch(dets_s))

    check(f"North: 3 cars, got {counts_n['car']}", counts_n["car"] == 3)
    check(f"South: 2 buses, got {counts_s['bus']}", counts_s["bus"] == 2)


# ---------------------------------------------------------------------- #
# Test 5 — Adaptive green time formula
# ---------------------------------------------------------------------- #

def test_green_time_formula():
    print("\n-- 5. Adaptive green time formula --")
    ctrl = AdaptiveTrafficController()
    check("demand= 5 → green=20s", ctrl.green_time_for_demand(5)  == 20.0)
    check("demand=20 → green=35s", ctrl.green_time_for_demand(20) == 35.0)
    check("demand=60 → green=60s (capped)", ctrl.green_time_for_demand(60) == 60.0)
    check("demand= 0 → green=MIN_GREEN",
          ctrl.green_time_for_demand(0) == config.MIN_GREEN)


# ---------------------------------------------------------------------- #
# Test 6 — Controller cycle, phase skipping, and IDLE
# ---------------------------------------------------------------------- #

def test_controller_cycle():
    print("\n-- 6. Controller cycle, phase skipping, and IDLE entry --")
    ctrl = AdaptiveTrafficController()

    # Only Phase 3 has demand; 1, 2, 4 should be skipped.
    demands = {1: 0.0, 2: 0.0, 3: 10.0, 4: 0.0}
    ctrl.update(0.0, demands)

    check("Starts GREEN on Phase 3 (first non-zero)",
          ctrl.state == ControllerState.GREEN and ctrl.current_phase == 3)
    check("Phases 1 and 2 recorded as skipped",
          1 in ctrl.skipped_phases and 2 in ctrl.skipped_phases)

    expected_green = ctrl.green_time_for_demand(10.0)   # 15 + 10 = 25s
    check(f"Green duration = {expected_green}s",
          ctrl.current_green_duration == expected_green)

    # Expire the green
    ctrl.update(expected_green, demands)
    check("Transitions to YELLOW after green expires",
          ctrl.state == ControllerState.YELLOW)

    # Expire yellow with zero demand → IDLE
    ctrl.update(config.YELLOW_TIME, {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})
    check("Enters IDLE when all phases zero",
          ctrl.state == ControllerState.IDLE)
    check("IDLE status: no active phase",
          ctrl.status()["active_phase"] is None)
    check("last_active_phase recorded as P3",
          ctrl.last_active_phase == 3)


# ---------------------------------------------------------------------- #
# Test 6b — Idle exit: exact spec scenario
# ---------------------------------------------------------------------- #

def test_idle_exit_round_robin():
    """Spec scenario:
        P1 → P2 → P3 each complete in turn, then all demand drops → IDLE.
        New demand: P1=10, P2=0, P3=0, P4=2.
        Round-robin after P3: P4 → P1 → P2 → P3.
        P4 must get green first even though P1 has higher demand.
    """
    print("\n-- 6b. Idle exit: round-robin resumes after last_active_phase --")
    ctrl = AdaptiveTrafficController()

    # Drive P1 → P2 → P3 to completion
    for phase_id in (1, 2, 3):
        d = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        d[phase_id] = 5.0
        ctrl.update(0.0, d)
        check(f"Phase {phase_id} gets green",
              ctrl.current_phase == phase_id and ctrl.state == ControllerState.GREEN)
        ctrl.update(ctrl.current_green_duration, d)          # expire green
        ctrl.update(config.YELLOW_TIME, {1:0.0,2:0.0,3:0.0,4:0.0})  # expire yellow

    check("After P3 completes with no further demand, controller is IDLE",
          ctrl.state == ControllerState.IDLE)
    check("last_active_phase = 3",
          ctrl.last_active_phase == 3)
    check("last_active_phase visible in status()",
          ctrl.status()["last_active_phase"] == 3)

    # New demand: P1=10 (bigger), P4=2 (smaller but comes first in sequence)
    new_demand = {1: 10.0, 2: 0.0, 3: 0.0, 4: 2.0}
    ctrl.update(0.0, new_demand)

    check("Exits IDLE to Phase 4 (next after P3), NOT highest-demand P1",
          ctrl.state == ControllerState.GREEN and ctrl.current_phase == 4)

    # Green duration must be based on P4's demand (2), not P1's (10)
    expected_p4 = ctrl.green_time_for_demand(2.0)   # 15 + 2 = 17s
    check(f"Green time = {expected_p4}s (P4 demand=2, not P1 demand=10)",
          ctrl.current_green_duration == expected_p4)

    # After P4 finishes, next valid phase is P1 (P2=0, P3=0 skipped)
    ctrl.update(ctrl.current_green_duration, new_demand)
    ctrl.update(config.YELLOW_TIME, new_demand)
    check("After P4, advances to P1 (round-robin continues normally)",
          ctrl.current_phase == 1 and ctrl.state == ControllerState.GREEN)

    # Verify green time now reflects P1's demand (10)
    expected_p1 = ctrl.green_time_for_demand(10.0)  # 15 + 10 = 25s
    check(f"P1 green time = {expected_p1}s (P1 demand=10)",
          ctrl.current_green_duration == expected_p1)


# ---------------------------------------------------------------------- #
# Test 6c — Idle exit on first startup (last_active_phase is None)
# ---------------------------------------------------------------------- #

def test_idle_exit_first_startup():
    """On very first startup last_active_phase is None.
    The controller must scan from phase_order[0] — not crash."""
    print("\n-- 6c. Idle exit on first startup (last_active_phase=None) --")
    ctrl = AdaptiveTrafficController()
    check("last_active_phase starts as None", ctrl.last_active_phase is None)

    # P1=0, P2=5 — first non-zero from the beginning is P2
    ctrl.update(0.0, {1: 0.0, 2: 5.0, 3: 0.0, 4: 0.0})
    check("First non-zero phase from beginning of order is P2",
          ctrl.current_phase == 2 and ctrl.state == ControllerState.GREEN)


# ---------------------------------------------------------------------- #
# Test 7 — Round-robin wraparound
# ---------------------------------------------------------------------- #

def test_round_robin_wraparound():
    print("\n-- 7. Round-robin wraparound --")
    ctrl = AdaptiveTrafficController()

    ctrl.update(0.0, {1: 0.0, 2: 0.0, 3: 0.0, 4: 5.0})
    check("Starts on Phase 4", ctrl.current_phase == 4)

    green_dur = ctrl.current_green_duration
    ctrl.update(green_dur, {1: 5.0, 2: 0.0, 3: 0.0, 4: 0.0})
    check("YELLOW after Phase 4 green expires",
          ctrl.state == ControllerState.YELLOW)

    ctrl.update(config.YELLOW_TIME, {1: 5.0, 2: 0.0, 3: 0.0, 4: 0.0})
    check("Wraps around to Phase 1",
          ctrl.current_phase == 1 and ctrl.state == ControllerState.GREEN)


# ---------------------------------------------------------------------- #
# Test 8 — Status dict completeness
# ---------------------------------------------------------------------- #

def test_status_dict():
    print("\n-- 8. Status dict completeness --")
    ctrl = AdaptiveTrafficController()
    ctrl.update(0.0, {1: 3.0, 2: 0.0, 3: 7.0, 4: 0.0})
    s = ctrl.status()
    for key in ("state", "active_phase", "phase_name",
                "remaining_green", "remaining_yellow",
                "skipped_phases", "phase_demands", "last_active_phase"):
        check(f"Status dict has key '{key}'", key in s)
    check("phase_demands covers all 4 phases",
          set(s["phase_demands"].keys()) == {1, 2, 3, 4})


# ---------------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    test_roi_assignment()
    test_cross_approach_demand_merge()
    test_demand_formula()
    test_vehicle_counts()
    test_green_time_formula()
    test_controller_cycle()
    test_idle_exit_round_robin()
    test_idle_exit_first_startup()
    test_round_robin_wraparound()
    test_status_dict()
    print("\n✓ All tests passed.")
