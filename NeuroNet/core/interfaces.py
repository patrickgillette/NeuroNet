
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterable, Optional, Sequence, Tuple, List

SimTime = float  # milliseconds

# ---- I/O contracts ----
class Encoder(ABC):
    @abstractmethod
    def encode(self, t: SimTime, observation: Any) -> Iterable[Tuple[int, float]]:
        ...

class Decoder(ABC):
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def on_spike(self, t: SimTime, neuron_id: int) -> None: ...
    @abstractmethod
    def readout(self, t: SimTime) -> Optional[Any]:
        ...

class Environment(ABC):
    @abstractmethod
    def apply_action(self, t: SimTime, action: Any) -> None: ...
    @abstractmethod
    def observe(self, t: SimTime) -> Any: ...

# ---- Minimal screen model kept as generic example types ----
class ScreenActionType(Enum):
    PUT_CHAR = auto()
    MOVE = auto()
    DRAW_DOT = auto()
    CLEAR = auto()

@dataclass(frozen=True)
class ScreenAction:
    kind: ScreenActionType
    x: Optional[int] = None
    y: Optional[int] = None
    ch: Optional[str] = None
    dx: Optional[int] = None
    dy: Optional[int] = None
    draw: bool = True

class Frame:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._cells: List[List[int]] = [[0 for _ in range(width)] for _ in range(height)]
    def get(self, x: int, y: int) -> int: return self._cells[y][x]
    def set(self, x: int, y: int, v: int) -> None: self._cells[y][x] = v
    def to_readonly(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self._cells)

# ---- Glue: I/O coordinator ----
class IOCoordinator:
    def __init__(self, env: Environment, encoder: Encoder, decoder: Decoder):
        self._env, self._encoder, self._decoder = env, encoder, decoder
    def on_output_spike(self, t: SimTime, neuron_id: int) -> None:
        self._decoder.on_spike(t, neuron_id)
    def maybe_emit_action(self, t: SimTime) -> Optional[Any]:
        action = self._decoder.readout(t)
        if action is not None:
            self._env.apply_action(t, action)
        return action
    def encode_observation(self, t: SimTime) -> Iterable[Tuple[int, float]]:
        obs = self._env.observe(t)
        return self._encoder.encode(t, obs)