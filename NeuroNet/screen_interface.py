
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Protocol

SimTime = float  # milliseconds

# -----------------------------------------------------------------------------
# Core I/O Interfaces
# -----------------------------------------------------------------------------

class Encoder(ABC):
    """External observation -> spikes into InputArea (neuron IDs in that area)."""

    @abstractmethod
    def encode(self, t: SimTime, observation: Any) -> Iterable[Tuple[int, float]]:
        """
        Produce spikes as (target_neuron_id, spike_time_offset_ms >= 0).
        The scheduler will schedule deliveries at (t + offset + synaptic_delay).
        """


class Decoder(ABC):
    """Spikes from an OutputArea -> external action/value (for the environment)."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state (windows, accumulators, filters)."""

    @abstractmethod
    def on_spike(self, t: SimTime, neuron_id: int) -> None:
        """Receive a spike from an OutputArea neuron at simulated time t."""

    @abstractmethod
    def readout(self, t: SimTime) -> Optional[Any]:
        """
        Convert recent spike activity into an action/value.
        May return None if not enough evidence yet (windowing, cadence, etc.).
        """


class Environment(ABC):
    """World model or device the network interacts with."""

    @abstractmethod
    def apply_action(self, t: SimTime, action: Any) -> None:
        """Apply an action at simulated time t; may internally queue effects with delay."""

    @abstractmethod
    def observe(self, t: SimTime) -> Any:
        """Return the current observable state for encoders to transform into spikes."""


# -----------------------------------------------------------------------------
# Terminal "screen" environment
# -----------------------------------------------------------------------------

class ScreenActionType(Enum):
    """High-level console actions (extend as needed)."""
    PUT_CHAR = auto()     # place a character at (x, y)
    MOVE = auto()         # move a cursor by (dx, dy) and optionally draw
    DRAW_DOT = auto()     # set a pixel/char at (x, y)
    CLEAR = auto()        # clear the buffer


@dataclass(frozen=True)
class ScreenAction:
    """Typed action destined for the ScreenEnvironment."""
    kind: ScreenActionType
    x: Optional[int] = None
    y: Optional[int] = None
    ch: Optional[str] = None
    dx: Optional[int] = None
    dy: Optional[int] = None
    draw: bool = True     # for MOVE: whether to draw at the new position


class Frame:
    """
    A minimal, implementation-agnostic frame.
    Backing storage can be a list of lists or a numpy array in a concrete impl.
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Represent as integers (0..N) or ASCII codes in a real impl
        self._cells: List[List[int]] = [[0 for _ in range(width)] for _ in range(height)]

    def get(self, x: int, y: int) -> int:
        return self._cells[y][x]

    def set(self, x: int, y: int, value: int) -> None:
        self._cells[y][x] = value

    def to_readonly(self) -> Tuple[Tuple[int, ...], ...]:
        """Expose an immutable view for encoders."""
        return tuple(tuple(row) for row in self._cells)


class ScreenEnvironment(Environment):
    """
    Virtual terminal screen: source of truth for what's displayed.
    The CLI renderer will render frames from here; encoders 'see' this buffer.
    """

    def __init__(self, width: int = 32, height: int = 18, render_latency_ms: float = 1.0):
        self._frame = Frame(width, height)
        self._cursor_x = width // 2
        self._cursor_y = height // 2
        self._render_latency = float(render_latency_ms)

    # --- Environment API ---

    def apply_action(self, t: SimTime, action: ScreenAction) -> None:
        """
        Mutate the buffer deterministically. In a concrete impl, you may schedule
        effects at t + self._render_latency to model display lag.
        """
        # Skeleton only; implement in concrete class.
        raise NotImplementedError

    def observe(self, t: SimTime) -> Frame:
        """Return the current frame; encoders will downsample/encode as spikes."""
        return self._frame

    # --- Helpers for concrete implementations ---

    def _put_char(self, x: int, y: int, ch_code: int) -> None:
        if 0 <= x < self._frame.width and 0 <= y < self._frame.height:
            self._frame.set(x, y, ch_code)

    def _move(self, dx: int, dy: int, draw: bool = True, ch_code: int = 1) -> None:
        self._cursor_x = max(0, min(self._frame.width - 1, self._cursor_x + dx))
        self._cursor_y = max(0, min(self._frame.height - 1, self._cursor_y + dy))
        if draw:
            self._put_char(self._cursor_x, self._cursor_y, ch_code)

    def _clear(self) -> None:
        for y in range(self._frame.height):
            for x in range(self._frame.width):
                self._frame.set(x, y, 0)


# -----------------------------------------------------------------------------
# Encoders: screen -> input spikes
# -----------------------------------------------------------------------------

class ScreenEncoder(Encoder):
    """
    Base encoder from a Frame to spikes targeting input-area neuron IDs.
    Concrete subclasses implement a coding scheme (e.g., Poisson per tile, event-delta).
    """

    def __init__(self, target_neuron_ids: Sequence[int], min_latency_ms: float = 1.0):
        self._targets = list(target_neuron_ids)
        self._min_latency = float(min_latency_ms)

    def encode(self, t: SimTime, observation: Frame) -> Iterable[Tuple[int, float]]:
        raise NotImplementedError


class TilePoissonEncoder(ScreenEncoder):
    """
    Example interface: Downsample the frame into tiles; map tile intensity to rate,
    produce Poisson spikes with at least _min_latency. Leave implementation empty here.
    """
    def __init__(
        self,
        target_neuron_ids: Sequence[int],
        grid_w: int,
        grid_h: int,
        window_ms: float = 20.0,
        max_rate_hz: float = 200.0,
        min_latency_ms: float = 1.0,
    ):
        super().__init__(target_neuron_ids, min_latency_ms=min_latency_ms)
        self._grid_w = grid_w
        self._grid_h = grid_h
        self._window_ms = float(window_ms)
        self._max_rate_hz = float(max_rate_hz)

    def encode(self, t: SimTime, observation: Frame) -> Iterable[Tuple[int, float]]:
        # Skeleton only; implement sampling, downsampling, and Poisson sampling in a real impl.
        raise NotImplementedError


class EventDeltaEncoder(ScreenEncoder):
    """
    Example interface: emit spikes only for cells that changed since last observation.
    """
    def __init__(self, target_neuron_ids: Sequence[int], min_latency_ms: float = 1.0):
        super().__init__(target_neuron_ids, min_latency_ms=min_latency_ms)
        self._prev: Optional[Tuple[Tuple[int, ...], ...]] = None

    def encode(self, t: SimTime, observation: Frame) -> Iterable[Tuple[int, float]]:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Decoders: output spikes -> console actions
# -----------------------------------------------------------------------------

class ConsoleDecoder(Decoder):
    """
    Base decoder mapping OutputArea spikes to ScreenAction(s).
    Concrete subclasses define readout policy (first-to-spike, rate window, etc.).
    """

    def __init__(self, readout_period_ms: float = 10.0, min_action_delay_ms: float = 1.0):
        self._readout_period = float(readout_period_ms)
        self._min_action_delay = float(min_action_delay_ms)

    def reset(self) -> None:
        raise NotImplementedError

    def on_spike(self, t: SimTime, neuron_id: int) -> None:
        raise NotImplementedError

    def readout(self, t: SimTime) -> Optional[ScreenAction]:
        raise NotImplementedError


class FirstToSpikeMoveDecoder(ConsoleDecoder):
    """
    Example interface: A small population controls MOVE{up,down,left,right}.
    The first spike in a readout window picks the action.
    """

    def __init__(
        self,
        up_ids: Sequence[int],
        down_ids: Sequence[int],
        left_ids: Sequence[int],
        right_ids: Sequence[int],
        readout_period_ms: float = 10.0,
        min_action_delay_ms: float = 1.0,
        step: int = 1,
    ):
        super().__init__(readout_period_ms, min_action_delay_ms)
        self._up = set(up_ids)
        self._down = set(down_ids)
        self._left = set(left_ids)
        self._right = set(right_ids)
        self._step = int(step)

    def reset(self) -> None:
        # Skeleton only; clear internal buffers/timestamps.
        raise NotImplementedError

    def on_spike(self, t: SimTime, neuron_id: int) -> None:
        # Skeleton only; record first spike per direction within the window.
        raise NotImplementedError

    def readout(self, t: SimTime) -> Optional[ScreenAction]:
        # Skeleton only; choose direction with earliest spike; return a MOVE action.
        raise NotImplementedError


class RateWindowPutCharDecoder(ConsoleDecoder):
    """
    Example interface: Use rate over a window to select (x, y, char).
    Typically you would have three populations or a population code per dimension.
    """

    def __init__(
        self,
        x_pop_ids: Sequence[int],
        y_pop_ids: Sequence[int],
        char_pop_ids: Sequence[int],
        width: int,
        height: int,
        readout_period_ms: float = 20.0,
        min_action_delay_ms: float = 1.0,
    ):
        super().__init__(readout_period_ms, min_action_delay_ms)
        self._x_ids = list(x_pop_ids)
        self._y_ids = list(y_pop_ids)
        self._c_ids = list(char_pop_ids)
        self._width = int(width)
        self._height = int(height)

    def reset(self) -> None:
        raise NotImplementedError

    def on_spike(self, t: SimTime, neuron_id: int) -> None:
        raise NotImplementedError

    def readout(self, t: SimTime) -> Optional[ScreenAction]:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# CLI Renderer (purely cosmetic; not part of causality)
# -----------------------------------------------------------------------------

class CLIRenderer:
    """
    Renders a Frame to the terminal (stdout). The Environment remains the source of truth.
    You can call this at a slower cadence than the sim loop (e.g., every 50–100 ms).
    """

    def __init__(self):
        pass

    def render(self, t: SimTime, frame: Frame) -> None:
        """
        Pretty-print the frame. In a concrete impl, clear screen and draw.
        Keep this separate from Environment to avoid coupling rendering to state.
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Glue: minimal I/O coordinator facade (for your scheduler)
# -----------------------------------------------------------------------------

class IOCoordinator:
    """
    Thin adapter the scheduler can call to advance the I/O loop at chosen times.
    - Pull spikes from OutputArea -> feed decoder.on_spike
    - At readout times, ask decoder.readout -> env.apply_action
    - Sample env.observe -> pass to encoder.encode -> schedule input spikes

    The SimulationKernel should own *when* to call these; this class just wires calls.
    """

    def __init__(self, env: Environment, encoder: Encoder, decoder: Decoder):
        self._env = env
        self._encoder = encoder
        self._decoder = decoder

    # The following methods are invoked by your scheduler at appropriate times:

    def on_output_spike(self, t: SimTime, neuron_id: int) -> None:
        self._decoder.on_spike(t, neuron_id)

    def maybe_emit_action(self, t: SimTime) -> None:
        action = self._decoder.readout(t)
        if action is not None:
            self._env.apply_action(t, action)

    def encode_observation(self, t: SimTime) -> Iterable[Tuple[int, float]]:
        obs = self._env.observe(t)
        return self._encoder.encode(t, obs)
