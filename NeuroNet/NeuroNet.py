from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Protocol, Tuple

SimTime = float  # milliseconds
#This is a SNN meant to take arbitrary external data as input and output actions/values.
# Define abstract base classes for encoders, decoders, environments, and I/O adapters.
# These can be subclassed to implement specific functionality.
class Encoder(ABC):
    """External data -> spikes into a target InputArea."""
    @abstractmethod
    def encode(self, t: SimTime, observation: Any) -> Iterable[Tuple[int, float]]:
        """
        Return an iterable of (target_neuron_id, spike_time_offset_ms) to schedule.
        spike_time_offset_ms >= 0 allows within-encoder latency/delay encoding.
        """

class Decoder(ABC):
    """Spikes from an OutputArea -> external action/value."""
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def on_spike(self, t: SimTime, neuron_id: int) -> None: ...
    @abstractmethod
    def readout(self, t: SimTime) -> Any:
        """
        Convert recent spikes to an action/value (e.g., rate over a window,
        first-to-spike, population vector). Called when the scheduler requests it.
        """

class Environment(ABC):
    """Closed-loop world model or real device interface."""
    @abstractmethod
    def step(self, t: SimTime, action: Any) -> Dict[str, Any]:
        """
        Apply action at time t, advance internal state (can be discrete or continuous),
        and return a dict of named observations, e.g. {"camera": frame, "pos": (x,y)}.
        """

class IOAdapter(ABC):
    """
    Binds encoders/decoders to concrete brain areas and routes observations/actions.
    This stays agnostic to network internals; it only needs neuron IDs of I/O areas.
    """
    @abstractmethod
    def pull_observations(self, t: SimTime) -> Dict[str, Any]: ...
    @abstractmethod
    def push_action(self, t: SimTime, action: Any) -> None: ...

class PortSpec(Protocol):
    name: str       # "camera", "audio", "reward", "action"
    shape: Tuple[int, ...] | None
    dtype: Any

class IOSchema:
    inputs: Dict[str, PortSpec]   # declared observable streams
    outputs: Dict[str, PortSpec]  # declared action streams
