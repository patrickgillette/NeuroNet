
from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple, List, Set
from dataclasses import dataclass
from enum import Enum, auto

from screen_interface import (
    SimTime, Encoder, Decoder, Environment, ScreenAction, ScreenActionType,
    Frame, ScreenEnvironment, IOCoordinator
)

# ----------------------------
# Concrete ScreenEnvironment
# ----------------------------
class SimpleScreen(ScreenEnvironment):
    """
    Implements apply_action with a tiny drawing model.
    Value '1' marks a drawn cell; '0' is empty; higher values are allowed.
    """
    def apply_action(self, t: SimTime, action: ScreenAction) -> None:
        if action.kind == ScreenActionType.CLEAR:
            self._clear()
            return
        if action.kind == ScreenActionType.PUT_CHAR:
            if action.x is not None and action.y is not None and action.ch is not None:
                code = ord(action.ch) if len(action.ch) > 0 else 1
                self._put_char(action.x, action.y, code)
            return
        if action.kind == ScreenActionType.DRAW_DOT:
            if action.x is not None and action.y is not None:
                self._put_char(action.x, action.y, 1)
            return
        if action.kind == ScreenActionType.MOVE:
            dx = action.dx or 0
            dy = action.dy or 0
            self._move(dx, dy, draw=action.draw, ch_code=1)
            return

# ----------------------------
# Minimal encoders/decoders
# ----------------------------
class NullEncoder(Encoder):
    """No sensory spikes; useful for a motor demo."""
    def __init__(self, target_neuron_ids: Sequence[int]):
        self.targets = list(target_neuron_ids)
    def encode(self, t: SimTime, observation: Frame) -> Iterable[Tuple[int, float]]:
        return []  # no input spikes

class FirstToSpikeMoveDecoder(Decoder):
    """
    Picks MOVE by whichever direction population spikes first within a window.
    """
    def __init__(self,
                 up_ids: Sequence[int],
                 down_ids: Sequence[int],
                 left_ids: Sequence[int],
                 right_ids: Sequence[int],
                 readout_period_ms: float = 10.0,
                 min_action_delay_ms: float = 1.0,
                 step: int = 1):
        self.up: Set[int] = set(up_ids)
        self.down: Set[int] = set(down_ids)
        self.left: Set[int] = set(left_ids)
        self.right: Set[int] = set(right_ids)
        self.period = float(readout_period_ms)
        self.min_delay = float(min_action_delay_ms)
        self.step = int(step)
        self._window_start: Optional[SimTime] = None
        self._first: Optional[Tuple[str, SimTime]] = None

    def reset(self) -> None:
        self._window_start = None
        self._first = None

    def on_spike(self, t: SimTime, neuron_id: int) -> None:
        if self._window_start is None:
            self._window_start = t
            self._first = None
        # record first direction if not yet set
        if self._first is None:
            if neuron_id in self.up:
                self._first = ("up", t)
            elif neuron_id in self.down:
                self._first = ("down", t)
            elif neuron_id in self.left:
                self._first = ("left", t)
            elif neuron_id in self.right:
                self._first = ("right", t)

    def readout(self, t: SimTime) -> Optional[ScreenAction]:
        if self._window_start is None:
            self._window_start = t
            return None
        if (t - self._window_start) < self.period:
            return None

        # end of window: emit action if we saw a first spike
        action = None
        if self._first is not None:
            dir_, _ = self._first
            if dir_ == "up":
                action = ScreenAction(kind=ScreenActionType.MOVE, dx=0, dy=-self.step)
            elif dir_ == "down":
                action = ScreenAction(kind=ScreenActionType.MOVE, dx=0, dy=self.step)
            elif dir_ == "left":
                action = ScreenAction(kind=ScreenActionType.MOVE, dx=-self.step, dy=0)
            elif dir_ == "right":
                action = ScreenAction(kind=ScreenActionType.MOVE, dx=self.step, dy=0)

        # reset window
        self._window_start = t
        self._first = None
        return action
