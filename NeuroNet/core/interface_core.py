
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Protocol, Set

from .interfaces import SimTime, NeuronId, Encoder, Decoder, Environment

class PortSpec(Protocol):
    name: str
    shape: Tuple[int, ...] | None
    dtype: Any


@dataclass
class IOSchema:
    inputs: Dict[str, PortSpec] = field(default_factory=dict)
    outputs: Dict[str, PortSpec] = field(default_factory=dict)

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

class IOCoordinator:

    def __init__(self, env: Environment, schema: Optional[IOSchema] = None):
        self._env = env
        self._schema = schema
        self._inputs: List[InputBinding] = []
        self._outputs: List[OutputBinding] = []
        self._route: Dict[NeuronId, List[int]] = {}

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

    def encode_observations(self, t: SimTime) -> Iterable[Tuple[NeuronId, float]]:
        obs_by_port = self._env.observe(t)
        # If schema provided, you could validate keys/types here.
        for ib in self._inputs:
            if ib.port not in obs_by_port:
                continue
            spikes = ib.encoder.encode(t, obs_by_port[ib.port])

            for nid, off in spikes:
                yield (nid, float(off))

    def on_output_spike(self, t: SimTime, neuron_id: NeuronId) -> None:
        for idx in self._route.get(neuron_id, []):
            ob = self._outputs[idx]
            # Lazy schedule readout windows per decoder
            ob.schedule_if_needed(t)
            ob.decoder.on_spike(t, neuron_id)

    def maybe_emit_actions(self, t: SimTime) -> Dict[str, Any]:
        actions: Dict[str, Any] = {}
        for ob in self._outputs:
            if not ob.due(t):
                continue
            ob.advance()
            act = ob.decoder.readout(t)
            if act is not None:
                actions[ob.port] = act
        return actions


@dataclass
class SimpleDictEnvironment(Environment):
    state: Dict[str, Any] = field(default_factory=dict)

    def observe(self, t: SimTime) -> Dict[str, Any]:
        return self.state

    def apply_action(self, t: SimTime, actions: Dict[str, Any]) -> None:
        # Default behavior: record last actions under reserved key
        self.state.setdefault("_last_actions", {})
        self.state["_last_actions"].update(actions)