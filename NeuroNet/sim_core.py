from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Iterable, Optional, DefaultDict
from collections import defaultdict

SimTime = float  # ms
NeuronId = int

# ----------------------------
# LIF neuron (discrete time)
# ----------------------------
@dataclass
class LIFConfig:
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_thresh: float = 1.0
    tau_m_ms: float = 20.0      # membrane time constant
    r_m: float = 1.0            # membrane resistance (scales current -> voltage)
    tau_ref_ms: float = 2.0     # refractory period

@dataclass
class LIFState:
    v: float = 0.0
    ref_until: SimTime = -1e9

@dataclass
class Neuron:
    cfg: LIFConfig = field(default_factory=LIFConfig)
    st: LIFState = field(default_factory=LIFState)

    def reset(self) -> None:
        self.st = LIFState(v=self.cfg.v_rest, ref_until=-1e9)

    def step(self, t: SimTime, dt_ms: float, i_ext: float) -> bool:
        """Return True if spikes at end of step."""
        if t < self.st.ref_until:
            # hold at reset during refractory
            self.st.v = self.cfg.v_reset
            return False

        # Euler update: dv = dt/tau * (-(v - v_rest) + R*I)
        alpha = dt_ms / self.cfg.tau_m_ms
        dv = alpha * (-(self.st.v - self.cfg.v_rest) + self.cfg.r_m * i_ext)
        self.st.v += dv

        if self.st.v >= self.cfg.v_thresh:
            self.st.v = self.cfg.v_reset
            self.st.ref_until = t + self.cfg.tau_ref_ms
            return True
        return False

# ----------------------------
# Synapses with delay/weight
# ----------------------------
@dataclass(frozen=True)
class Synapse:
    pre: NeuronId
    post: NeuronId
    w: float            # current injected (arbitrary units)
    delay_ms: float     # axonal delay

# ----------------------------
# Network + scheduler
# ----------------------------
class SNN:
    def __init__(self, num_neurons: int, lif_template: Optional[LIFConfig] = None):
        self.neurons: List[Neuron] = [
            Neuron(cfg=(lif_template or LIFConfig())) for _ in range(num_neurons)
        ]
        for n in self.neurons:
            n.reset()
        self.outgoing: DefaultDict[int, List[Synapse]] = defaultdict(list)
        # ring buffer for future currents (ms resolution)
        self._pending_currents: DefaultDict[int, float] = defaultdict(float)

    def add_synapse(self, s: Synapse) -> None:
        self.outgoing[s.pre].append(s)

    def _schedule_current(self, deliver_at_ms: int, target: int, current: float) -> None:
        # Accumulate current that will be delivered at that integer ms to target
        key = (deliver_at_ms << 16) | target  # combine time and neuron id
        self._pending_currents[key] += current

    def inject_spike(self, t: SimTime, post: int, current: float) -> None:
        # external spike -> immediate (this step) current accumulation
        key = (int(round(t)) << 16) | post
        self._pending_currents[key] += current

    def step(self, t: SimTime, dt_ms: float) -> List[int]:
        """
        Advance network by dt_ms. Returns list of spiking neuron IDs at end of step.
        Current delivery is at integer-millisecond bins for simplicity.
        """
        now_bin = int(round(t))
        # Gather input currents scheduled for this bin
        i_in: List[float] = [0.0] * len(self.neurons)
        # pull and clear all entries whose time == now_bin
        keys_to_delete = []
        for key, cur in self._pending_currents.items():
            time_bin = key >> 16
            if time_bin == now_bin:
                nid = key & 0xFFFF
                i_in[nid] += cur
                keys_to_delete.append(key)
        for k in keys_to_delete:
            del self._pending_currents[k]

        spikes: List[int] = []
        for nid, neuron in enumerate(self.neurons):
            if neuron.step(t, dt_ms, i_in[nid]):
                spikes.append(nid)

        # propagate spikes through synapses
        for pre in spikes:
            for s in self.outgoing.get(pre, []):
                deliver_at = int(round(t + s.delay_ms))
                self._schedule_current(deliver_at, s.post, s.w)

        return spikes

