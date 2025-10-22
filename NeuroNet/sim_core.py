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


@dataclass
class PlasticityConfig:
    eta: float = 0.002        # learning rate
    tau_trace_ms: float = 200.0
    tau_elig_ms: float = 300.0
    A_pre: float = 1.0
    A_post: float = 1.0
    w_min: float = -1.0
    w_max: float = 1.0

# ----------------------------
# Synapses with delay/weight
# ----------------------------
@dataclass(frozen=True)
class Synapse:
    pre: NeuronId
    post: NeuronId
    w: float
    delay_ms: float
    plastic: bool = False

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

        self._syn_list: List[Synapse] = []          # flat list keeps stable indices
        self._outgoing_idx: DefaultDict[int, List[int]] = defaultdict(list)
        self.plastic_cfg = PlasticityConfig()

        # traces and eligibilities (by synapse index)
        self._pre_tr: DefaultDict[int, float] = defaultdict(float)
        self._post_tr: DefaultDict[int, float] = defaultdict(float)
        self._elig: DefaultDict[int, float] = defaultdict(float)
        self._t_last = 0.0

    def add_synapse(self, s: Synapse) -> None:
        idx = len(self._syn_list)
        self._syn_list.append(s)
        self.outgoing[s.pre].append(s)          # keep old mapping (for delivery weights)
        self._outgoing_idx[s.pre].append(idx) 

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

        # propagate through synapses (as before)
        for pre in spikes:
            for s in self.outgoing.get(pre, []):
                deliver_at = int(round(t + s.delay_ms))
                if deliver_at <= now_bin:
                    deliver_at = now_bin + 1
                self._schedule_current(deliver_at, s.post, s.w)

        # --- Plasticity bookkeeping ---
        # decay traces and eligibilities
        if self._syn_list:
            dt = t - self._t_last
            self._t_last = t
            pc = self.plastic_cfg
            decay_pre = pow(2.718281828, -dt / pc.tau_trace_ms)
            decay_post = pow(2.718281828, -dt / pc.tau_trace_ms)
            decay_elig = pow(2.718281828, -dt / pc.tau_elig_ms)
            for idx, s in enumerate(self._syn_list):
                if s.plastic:
                    self._pre_tr[idx] *= decay_pre
                    self._post_tr[idx] *= decay_post
                    self._elig[idx] *= decay_elig

        # on pre spikes: bump pre-trace and eligibility by post-trace
        for pre in spikes:
            for idx in self._outgoing_idx.get(pre, []):
                s = self._syn_list[idx]
                if not s.plastic: 
                    continue
                self._pre_tr[idx] += self.plastic_cfg.A_pre
                self._elig[idx] += self.plastic_cfg.A_pre * self._post_tr[idx]

        # on post spikes: bump post-trace and eligibility by pre-trace
        for post in spikes:
            # find all synapses that target this post
            for idx, s in enumerate(self._syn_list):
                if s.plastic and s.post == post:
                    self._post_tr[idx] += self.plastic_cfg.A_post
                    self._elig[idx] += self.plastic_cfg.A_post * self._pre_tr[idx]

        return spikes

    def apply_reward(self, r: float) -> None:
        pc = self.plastic_cfg
        # write back updated weights with clipping
        new_list = []
        for idx, s in enumerate(self._syn_list):
            if s.plastic:
                dw = pc.eta * r * self._elig[idx]
                w = max(pc.w_min, min(pc.w_max, s.w + dw))
                s = Synapse(pre=s.pre, post=s.post, w=w, delay_ms=s.delay_ms, plastic=True)
            new_list.append(s)
        self._syn_list = new_list
        # also refresh the easy-to-use outgoing with new weights
        self.outgoing.clear()
        for s in self._syn_list:
            self.outgoing[s.pre].append(s)



