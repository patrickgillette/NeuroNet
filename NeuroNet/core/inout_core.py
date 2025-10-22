
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Protocol, Set

# -----------------------------------------------------------------------------
# Core types
# -----------------------------------------------------------------------------
SimTime = float  # milliseconds
NeuronId = int

"""
io_core.py — Generic I/O interfaces and a multi-port coordinator for SNN loops.

This module defines:
  • Encoder / Decoder / Environment base classes using a common SimTime alias
  • IOSchema and PortSpec for declaring named input/output streams
  • IOPort bindings to map encoders/decoders to neuron-id ranges
  • IOCoordinator that:
      - fan-in: pulls observations from an Environment, feeds them to multiple Encoders,
                and returns spikes to inject into the network
      - fan-out: collects output spikes, routes them to appropriate Decoders,
                 and emits per-port actions at each decoder's readout cadence

Design goals:
  • Decouple I/O from any particular environment (e.g., screen, robotics, audio)
  • Support multiple independent modalities and actuator streams concurrently
  • Keep scheduling policy simple: the simulation loop controls when to call us

This is single-threaded and simulation-time oriented; no wall-clock sleeps here.
"""

# -----------------------------------------------------------------------------
# Base interfaces
# -----------------------------------------------------------------------------
class Encoder(ABC):
    """External observation -> spikes targeting input-area neuron IDs.

    Implementations must be pure w.r.t. side effects on the network; they only
    return a list/iterable of (target_neuron_id, spike_time_offset_ms >= 0).
    """

    @abstractmethod
    def encode(self, t: SimTime, observation: Any) -> Iterable[Tuple[NeuronId, float]]:
        raise NotImplementedError


class Decoder(ABC):
    """Spikes from an output population -> typed action/value for a named port."""

    @abstractmethod
    def reset(self) -> None:
        """Clear any internal state (windows, accumulators, traces)."""
        raise NotImplementedError

    @abstractmethod
    def on_spike(self, t: SimTime, neuron_id: NeuronId) -> None:
        """Receive an output spike at time t from a neuron bound to this decoder."""
        raise NotImplementedError

    @abstractmethod
    def readout(self, t: SimTime) -> Optional[Any]:
        """Optionally produce an action/value at time t (may return None)."""
        raise NotImplementedError


class Environment(ABC):
    """World model or device the network interacts with.

    The coordinator will call observe() to get a dict of named input observations
    and apply_action() to submit a dict of named output actions.
    """

    @abstractmethod
    def observe(self, t: SimTime) -> Dict[str, Any]:
        """Return observations keyed by input port name."""
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, t: SimTime, actions: Dict[str, Any]) -> None:
        """Apply per-port actions (possibly a partial dict)."""
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Port schema (optional, for validation & tooling)
# -----------------------------------------------------------------------------
class PortSpec(Protocol):
    name: str
    shape: Tuple[int, ...] | None
    dtype: Any


@dataclass
class IOSchema:
    inputs: Dict[str, PortSpec] = field(default_factory=dict)
    outputs: Dict[str, PortSpec] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Bindings
# -----------------------------------------------------------------------------
@dataclass
class InputBinding:
    port: str
    encoder: Encoder
    target_ids: Sequence[NeuronId]


@dataclass
class OutputBinding:
    port: str
    decoder: Decoder
    source_ids: Set[NeuronId]
    readout_period_ms: float = 10.0
    _next_readout_at: Optional[SimTime] = None

    def schedule_if_needed(self, t: SimTime) -> None:
        if self._next_readout_at is None:
            self._next_readout_at = t + self.readout_period_ms

    def due(self, t: SimTime) -> bool:
        return self._next_readout_at is not None and t >= self._next_readout_at

    def advance(self) -> None:
        if self._next_readout_at is not None:
            self._next_readout_at += self.readout_period_ms


# -----------------------------------------------------------------------------
# IO Coordinator (multi-port)
# -----------------------------------------------------------------------------
class IOCoordinator:
    """Coordinates multiple encoders/decoders with a single Environment.

    Usage in a simulation loop (pseudo):
        io = IOCoordinator(env)
        io.bind_input("camera", CameraEncoder(...), ids_cam)
        io.bind_input("proprio", ProprioEncoder(...), ids_prop)
        io.bind_output("nav", NavDecoder(...), ids_nav, readout_period_ms=50)
        io.bind_output("gripper", GripperDecoder(...), ids_grip, readout_period_ms=20)

        while running:
            # 1) Encode current observations -> schedule spikes
            for nid, offset in io.encode_observations(t):
                net.inject_spike(t + offset, nid, current_strength)

            # 2) Step network -> collect spikes
            spikes = net.step(t, dt_ms)
            for nid in spikes:
                io.on_output_spike(t, nid)

            # 3) Periodic readouts -> environment actions
            actions = io.maybe_emit_actions(t)
            if actions:
                env.apply_action(t, actions)

            t += dt_ms
    """

    def __init__(self, env: Environment, schema: Optional[IOSchema] = None):
        self._env = env
        self._schema = schema
        self._inputs: List[InputBinding] = []
        self._outputs: List[OutputBinding] = []
        # Fast routing table: neuron id -> list of output binding indices
        self._route: Dict[NeuronId, List[int]] = {}

    # ---- Binding API ----
    def bind_input(self, port: str, encoder: Encoder, target_ids: Sequence[NeuronId]) -> None:
        self._inputs.append(InputBinding(port=port, encoder=encoder, target_ids=tuple(target_ids)))

    def bind_output(
        self,
        port: str,
        decoder: Decoder,
        source_ids: Sequence[NeuronId],
        readout_period_ms: float = 10.0,
    ) -> None:
        ob = OutputBinding(
            port=port,
            decoder=decoder,
            source_ids=set(source_ids),
            readout_period_ms=float(readout_period_ms),
        )
        self._outputs.append(ob)
        idx = len(self._outputs) - 1
        for nid in source_ids:
            self._route.setdefault(nid, []).append(idx)

    def reset(self) -> None:
        for ob in self._outputs:
            ob.decoder.reset()
            ob._next_readout_at = None

    # ---- Simulation-loop hooks ----
    def encode_observations(self, t: SimTime) -> Iterable[Tuple[NeuronId, float]]:
        """Observe environment and fan-out to all encoders.

        Returns an iterable of (target_neuron_id, offset_ms) spikes to be scheduled
        by the caller into the SNN. This function does not mutate network state.
        """
        obs_by_port = self._env.observe(t)
        # If schema provided, you could validate keys/types here.
        for ib in self._inputs:
            if ib.port not in obs_by_port:
                # Port missing -> skip silently to allow partial envs
                continue
            spikes = ib.encoder.encode(t, obs_by_port[ib.port])
            # Encoders are responsible for targeting neuron ids within their range;
            # however, some encoders may only compute offsets; we pass through as-is.
            for nid, off in spikes:
                yield (nid, float(off))

    def on_output_spike(self, t: SimTime, neuron_id: NeuronId) -> None:
        """Route a spike to all decoders that listen to this neuron id."""
        for idx in self._route.get(neuron_id, []):
            ob = self._outputs[idx]
            # Lazy schedule readout windows per decoder
            ob.schedule_if_needed(t)
            ob.decoder.on_spike(t, neuron_id)

    def maybe_emit_actions(self, t: SimTime) -> Dict[str, Any]:
        """Poll decoders that are due; coalesce non-None actions by port.

        Returns a possibly-empty dict mapping output port names to actions.
        Decoders are polled only when their readout period elapses.
        """
        actions: Dict[str, Any] = {}
        for ob in self._outputs:
            if not ob.due(t):
                continue
            ob.advance()
            act = ob.decoder.readout(t)
            if act is not None:
                actions[ob.port] = act
        return actions


# -----------------------------------------------------------------------------
# Convenience: trivial passthrough Environment (optional)
# -----------------------------------------------------------------------------
@dataclass
class SimpleDictEnvironment(Environment):
    """A minimal environment that stores an internal observation dict and
    accepts action dicts via apply_action(). Useful for tests.
    """
    state: Dict[str, Any] = field(default_factory=dict)

    def observe(self, t: SimTime) -> Dict[str, Any]:
        return self.state

    def apply_action(self, t: SimTime, actions: Dict[str, Any]) -> None:
        # Default behavior: record last actions under reserved key
        self.state.setdefault("_last_actions", {})
        self.state["_last_actions"].update(actions)


# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
# • This module is intentionally generic. Concrete encoders/decoders/environments
#   for screens, cameras, audio, robots, etc. should live in separate modules and
#   implement the base Encoder/Decoder/Environment here.
# • The coordinator does not validate neuron id ranges; keep your mappings
#   consistent (e.g., via a ProcessingCore that owns id allocation).
# • If you need per-decoder min-action delays or overlapping windows, encode that
#   logic inside the Decoder implementation. The coordinator simply honors
#   each decoder's readout cadence.
