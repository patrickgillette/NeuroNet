# NeuroNet/core/runtime.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional
from .interfaces import IOCoordinator, SimTime
from ..sim.sim_core import SNN
from ..sim.processing_core import ProcessingCore
from .goals import Goal

class Runtime:
    """
    Owns the closed loop:
      encode(obs_t) -> inject -> step SNN -> decoder spikes -> action -> env.apply
      -> observe -> goal.evaluate -> apply_reward -> repeat
    The demo supplies env/encoder/decoder/goal and the built ProcessingCore.
    """

    def __init__(self, core: ProcessingCore, io: IOCoordinator, net: Optional[SNN] = None):
        assert core.net is not None, "Call core.build() first."
        self.core = core
        self.net = core.net if net is None else net
        self.io = io

    def tick(self, t: SimTime, dt_ms: float, goal: Goal,
             inject_scale: float = 1.3) -> Tuple[Optional[float], Iterable[int]]:
        # 1) Encode current observation -> inject spikes
        for nid, offset in self.io.encode_observation(t):
            self.net.inject_spike(t + offset, nid, inject_scale)

        # 2) Step SNN
        spikes = self.net.step(t, dt_ms=dt_ms)

        # 3) Feed output spikes to decoder
        for nid in spikes:
            self.io.on_output_spike(t, nid)

        # 4) Readout and apply action
        obs_before = self.io._env.observe(t)
        action = self.io.maybe_emit_action(t)  # returns action or None
        obs_after = self.io._env.observe(t)

        # 5) Goal evaluation -> apply reward
        r = 0.0
        if goal and action is not None:
            r = goal.evaluate(t, obs_before, action, obs_after)
            if r != 0.0:
                self.net.apply_reward(r)

        return (r if r != 0.0 else None, spikes)
