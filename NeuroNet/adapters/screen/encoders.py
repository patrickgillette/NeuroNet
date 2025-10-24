from __future__ import annotations
from typing import Iterable, Tuple

from ...core.interfaces import SimTime, Frame
from ...core.interfaces import Encoder  # base contract

class PositionEncoder(Encoder):
    """Encode a single lit pixel (dot) -> one active input neuron y*W + x."""
    def __init__(self, width: int, height: int, base_id: int, min_interval_ms: float = 5.0):
        self.W, self.H = width, height
        self.base = base_id
        self.min_interval = float(min_interval_ms)
        self._last_emit = -1e9

    def encode(self, t: SimTime, observation: Frame) -> Iterable[Tuple[int, float]]:
        if t - self._last_emit < self.min_interval:
            return []
        for y in range(observation.height):
            for x in range(observation.width):
                if observation.get(x, y) != 0:
                    nid = self.base + (y * self.W + x)
                    self._last_emit = t
                    return [(nid, 0.0)]
        return []
