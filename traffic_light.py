"""
traffic_light.py
Tracks the RED / YELLOW / GREEN state of every approach+movement pair at
the intersection under standard 4-phase exclusive control. Only the
movements belonging to the active phase may be non-RED at any given time.
"""

from enum import Enum
from typing import Dict, Tuple

import config


class SignalState(Enum):
    RED    = "RED"
    YELLOW = "YELLOW"
    GREEN  = "GREEN"


class TrafficLight:
    def __init__(self):
        self.states: Dict[Tuple[str, str], SignalState] = {}
        for phase_cfg in config.PHASES.values():
            for approach, movement in phase_cfg["movements"].items():
                self.states[(approach, movement)] = SignalState.RED

    def all_red(self):
        """Set every movement to RED (idle / all-red clearance interval)."""
        for key in self.states:
            self.states[key] = SignalState.RED

    def set_phase_state(self, phase_id: int, state: SignalState):
        """Set every movement in phase_id to `state`; all others go RED."""
        self.all_red()
        phase_cfg = config.PHASES[phase_id]
        for approach, movement in phase_cfg["movements"].items():
            self.states[(approach, movement)] = state

    def snapshot(self) -> Dict[str, str]:
        """Human-readable signal state map for the dashboard, e.g.
        {'north_left': 'RED', 'north_straight_right': 'GREEN', ...}"""
        return {
            f"{approach}_{movement}": state.value
            for (approach, movement), state in self.states.items()
        }
