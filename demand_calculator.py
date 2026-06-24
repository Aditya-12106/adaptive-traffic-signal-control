"""
demand_calculator.py
Converts per-lane vehicle detections into weighted demand scores, then
aggregates those into per-phase demand (for the controller) and
per-approach demand (for the dashboard).

In the multi-video architecture this module is stateless and shared: main.py
calls it once per frame *per approach* to get per-lane demands, then merges
all four approaches' lane demands into a single flat dict before computing
phase demands and approach totals.
"""

from collections import Counter
from typing import Dict, List, Optional

import config
from detector import Detection


class DemandCalculator:
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 phases: Optional[Dict[int, dict]] = None,
                 lane_to_approach: Optional[Dict[str, str]] = None):
        self.weights        = weights        or config.VEHICLE_WEIGHTS
        self.phases         = phases         or config.PHASES
        self.lane_to_approach = lane_to_approach or config.LANE_TO_APPROACH

    # ------------------------------------------------------------------ #
    # Per-lane demand
    # ------------------------------------------------------------------ #

    def lane_demand(self, detections: List[Detection]) -> float:
        """Weighted demand for a single lane group.
        demand = 1.0*cars + 2.0*trucks + 2.0*buses + 0.5*motorcycles
        """
        return sum(self.weights.get(det.class_name, 0.0) for det in detections)

    def lane_demands(self,
                     lane_buckets: Dict[str, List[Detection]]) -> Dict[str, float]:
        """Return {lane_name: demand_score} for every lane bucket supplied."""
        return {lane: self.lane_demand(dets) for lane, dets in lane_buckets.items()}

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    def approach_demands(self,
                         lane_demands: Dict[str, float]) -> Dict[str, float]:
        """Roll up lane-group demand into North/South/East/West totals.

        `lane_demands` is the *merged* dict covering all four approaches
        (built in main.py by combining each approach's lane_demands dict).
        """
        approach_totals: Dict[str, float] = {
            a: 0.0 for a in set(self.lane_to_approach.values())
        }
        for lane, demand in lane_demands.items():
            approach = self.lane_to_approach.get(lane)
            if approach:
                approach_totals[approach] += demand
        return approach_totals

    def phase_demands(self,
                      lane_demands: Dict[str, float]) -> Dict[int, float]:
        """Sum lane-group demand into each of the 4 signal phases:
            Phase 1 = north_straight_right + south_straight_right
            Phase 2 = north_left           + south_left
            Phase 3 = east_straight_right  + west_straight_right
            Phase 4 = east_left            + west_left
        """
        return {
            phase_id: sum(lane_demands.get(lane, 0.0)
                          for lane in phase_cfg["lanes"])
            for phase_id, phase_cfg in self.phases.items()
        }

    # ------------------------------------------------------------------ #
    # Vehicle counts (for the dashboard)
    # ------------------------------------------------------------------ #

    def vehicle_counts(self,
                       lane_buckets: Dict[str, List[Detection]]) -> Counter:
        """Raw (unweighted) count of each vehicle class across all lanes,
        used for the dashboard's 'Vehicle Class Counts' display."""
        counts: Counter = Counter()
        for dets in lane_buckets.values():
            for det in dets:
                counts[det.class_name] += 1
        return counts
