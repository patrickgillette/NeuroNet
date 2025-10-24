from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Set

from ...core.interfaces import Decoder, SimTime, ScreenAction, ScreenActionType

@dataclass
class FirstToSpikeMoveDecoder(Decoder):
    up_ids: Iterable[int]
    down_ids: Iterable[int]
    left_ids: Iterable[int]
    right_ids: Iterable[int]
    readout_period_ms: float = 100.0
    min_action_delay_ms: float = 1.0
    step: int = 1

    def __post_init__(self):
        self._t_next: float = 0.0
        self._buf: Set[int] = set()
        self._up, self._down = set(self.up_ids), set(self.down_ids)
        self._left, self._right = set(self.left_ids), set(self.right_ids)

    def reset(self) -> None:
        self._t_next = 0.0
        self._buf.clear()

    def on_spike(self, t: SimTime, neuron_id: int) -> None:
        self._buf.add(neuron_id)

    def readout(self, t: SimTime) -> Optional[ScreenAction]:
        if t < self._t_next:
            return None
        self._t_next = t + max(self.min_action_delay_ms, self.readout_period_ms)

        if self._buf & self._up:
            self._buf.clear()
            return ScreenAction(kind=ScreenActionType.MOVE, dx=0, dy=-self.step)
        if self._buf & self._down:
            self._buf.clear()
            return ScreenAction(kind=ScreenActionType.MOVE, dx=0, dy=+self.step)
        if self._buf & self._left:
            self._buf.clear()
            return ScreenAction(kind=ScreenActionType.MOVE, dx=-self.step, dy=0)
        if self._buf & self._right:
            self._buf.clear()
            return ScreenAction(kind=ScreenActionType.MOVE, dx=+self.step, dy=0)

        self._buf.clear()
        return None
