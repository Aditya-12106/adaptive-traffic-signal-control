"""
controller.py
Adaptive 4-phase traffic signal controller.

Decision loop (called every frame via update()):
    1. Aggregate per-phase demand received from all four approach feeds.
    2. Skip any phase whose demand == 0.
    3. Compute adaptive green time: green = min(MAX_GREEN, MIN_GREEN + k*demand)
    4. Execute phase (GREEN state).
    5. Fixed yellow transition (YELLOW_TIME seconds).
    6. Advance round-robin to next non-zero-demand phase.
    7. If ALL phases have zero demand → IDLE (all-red) until demand returns.

The controller is purely demand-driven: main.py feeds it aggregated
phase_demands each tick; the controller never reads sensors itself, making
it trivially unit-testable with synthetic values.
"""

from enum import Enum
from typing import Dict, List, Optional

import config
from traffic_light import TrafficLight, SignalState


class ControllerState(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    IDLE   = "IDLE"


class AdaptiveTrafficController:
    def __init__(self,
                 phase_order: Optional[List[int]] = None,
                 min_green:   float = config.MIN_GREEN,
                 max_green:   float = config.MAX_GREEN,
                 yellow_time: float = config.YELLOW_TIME,
                 k:           float = config.DEMAND_FACTOR_K):
        self.phase_order  = phase_order or list(config.PHASE_ORDER)
        self.min_green    = min_green
        self.max_green    = max_green
        self.yellow_time  = yellow_time
        self.k            = k

        self.light = TrafficLight()

        self.state: ControllerState      = ControllerState.IDLE
        self.current_phase: Optional[int] = None
        self.timer: float                 = 0.0   # elapsed seconds in current state
        self.current_green_duration: float = 0.0

        self._phase_demands_cache: Dict[int, float] = {p: 0.0 for p in self.phase_order}
        self.skipped_phases: List[int] = []

        # Tracks the most recently *completed* phase (i.e. the phase whose
        # yellow interval just finished).  Persists across idle periods so
        # that when the controller exits idle it resumes the round-robin
        # sequence immediately *after* this phase rather than restarting
        # from Phase 1 or jumping to the highest-demand phase.
        # None only on first startup (before any phase has ever run).
        self.last_active_phase: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def green_time_for_demand(self, demand: float) -> float:
        """green_time = min(MAX_GREEN, MIN_GREEN + k * demand)"""
        return min(self.max_green, self.min_green + self.k * demand)

    def update(self, dt: float, phase_demands: Dict[int, float]):
        """Advance the controller by `dt` seconds.

        Args:
            dt:            Time elapsed since the last call (seconds).
                           Typically 1/fps when driven frame-by-frame.
            phase_demands: Current measured demand per phase, e.g.
                           {1: 12.5, 2: 0.0, 3: 8.0, 4: 0.0}.
                           Computed by DemandCalculator from all four feeds.
        """
        self._phase_demands_cache = dict(phase_demands)
        self.timer += dt

        if self.state == ControllerState.IDLE:
            self._handle_idle(phase_demands)
        elif self.state == ControllerState.GREEN:
            self._handle_green()
        elif self.state == ControllerState.YELLOW:
            self._handle_yellow(phase_demands)

    # ------------------------------------------------------------------ #
    # Internal state handlers
    # ------------------------------------------------------------------ #

    def _handle_idle(self, phase_demands: Dict[int, float]):
        self.light.all_red()
        # Resume round-robin *after* the last completed phase so the
        # sequence is continuous across idle gaps.  On first startup
        # last_active_phase is None and the scan begins at phase_order[0].
        next_phase = self._find_next_demanding_phase(
            phase_demands, start_after=self.last_active_phase)
        if next_phase is not None:
            self._start_green(next_phase, phase_demands[next_phase])

    def _handle_green(self):
        if self.timer >= self.current_green_duration:
            self._start_yellow()

    def _handle_yellow(self, phase_demands: Dict[int, float]):
        if self.timer >= self.yellow_time:
            self._advance_to_next_phase(phase_demands)

    def _start_green(self, phase_id: int, demand: float):
        self.current_phase          = phase_id
        self.current_green_duration = self.green_time_for_demand(demand)
        self.state                  = ControllerState.GREEN
        self.timer                  = 0.0
        self.light.set_phase_state(phase_id, SignalState.GREEN)

    def _start_yellow(self):
        self.state = ControllerState.YELLOW
        self.timer = 0.0
        self.light.set_phase_state(self.current_phase, SignalState.YELLOW)

    def _advance_to_next_phase(self, phase_demands: Dict[int, float]):
        next_phase = self._find_next_demanding_phase(
            phase_demands, start_after=self.current_phase)
        if next_phase is None:
            # Remember where we stopped so idle exit resumes in sequence.
            self.last_active_phase = self.current_phase
            self.current_phase = None
            self.state         = ControllerState.IDLE
            self.timer         = 0.0
            self.light.all_red()
        else:
            self._start_green(next_phase, phase_demands[next_phase])

    def _find_next_demanding_phase(self,
                                   phase_demands: Dict[int, float],
                                   start_after: Optional[int]) -> Optional[int]:
        """Round-robin scan starting *after* `start_after` (or from the
        beginning if None). Returns the first phase with demand > 0, and
        records all zero-demand phases encountered as skipped.
        Returns None only when every phase is zero (→ IDLE)."""
        n = len(self.phase_order)
        start_idx = (
            0 if start_after is None
            else (self.phase_order.index(start_after) + 1) % n
        )
        self.skipped_phases = []
        for offset in range(n):
            idx      = (start_idx + offset) % n
            phase_id = self.phase_order[idx]
            if phase_demands.get(phase_id, 0.0) > 0:
                return phase_id
            self.skipped_phases.append(phase_id)
        return None

    # ------------------------------------------------------------------ #
    # Dashboard helpers
    # ------------------------------------------------------------------ #

    def remaining_green(self) -> float:
        if self.state != ControllerState.GREEN:
            return 0.0
        return max(0.0, self.current_green_duration - self.timer)

    def remaining_yellow(self) -> float:
        if self.state != ControllerState.YELLOW:
            return 0.0
        return max(0.0, self.yellow_time - self.timer)

    def status(self) -> dict:
        return {
            "state":             self.state.value,
            "active_phase":      self.current_phase,
            "phase_name":        (config.PHASES[self.current_phase]["name"]
                                  if self.current_phase else None),
            "remaining_green":   round(self.remaining_green(), 1),
            "remaining_yellow":  round(self.remaining_yellow(), 1),
            "skipped_phases":    list(self.skipped_phases),
            "phase_demands":     dict(self._phase_demands_cache),
            "last_active_phase": self.last_active_phase,
        }
