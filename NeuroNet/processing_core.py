
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Dict

from sim_core import SNN, Synapse, LIFConfig

NeuronRange = Tuple[int, int]  # [start, end) neuron ids

@dataclass
class Pop:
    name: str
    start: int
    size: int
    @property
    def ids(self) -> range:
        return range(self.start, self.start + self.size)

class ProcessingCore:
    """
    Owns neuron id layout and connectivity between:
      - input population(s) (targets for Encoders)
      - processing population(s) (plastic)
      - output population(s) (readout for Decoders)
    Encoders/Decoders only receive/populate ids via this object.
    """
    def __init__(self,
                 lif_input: Optional[LIFConfig] = None,
                 lif_hidden: Optional[LIFConfig] = None,
                 lif_output: Optional[LIFConfig] = None):
        self._lif_in = lif_input or LIFConfig()
        self._lif_h = lif_hidden or LIFConfig()
        self._lif_out = lif_output or LIFConfig()

        self._pops: Dict[str, Pop] = {}
        self._order: List[Pop] = []

        # Backing SNN will be created after we know total size
        self.net: Optional[SNN] = None

    # ---- declare layout ----
    def add_population(self, name: str, size: int) -> Pop:
        assert self.net is None, "Add populations before build()."
        start = sum(p.size for p in self._order)
        pop = Pop(name=name, start=start, size=size)
        self._pops[name] = pop
        self._order.append(pop)
        return pop

    def get(self, name: str) -> Pop:
        return self._pops[name]

    def build(self) -> None:
        assert self.net is None
        total = sum(p.size for p in self._order)
        # For simplicity, use one LIF template; if you want per-pop, split SNN or extend it.
        self.net = SNN(num_neurons=total, lif_template=LIFConfig())

    # ---- connect helpers ----
    def dense(self, pre: Pop, post: Pop, w: Tuple[float, float], delay_ms: float,
              plastic: bool) -> None:
        assert self.net is not None
        import random
        for i in pre.ids:
            for j in post.ids:
                w0 = random.uniform(w[0], w[1])
                self.net.add_synapse(Synapse(pre=i, post=j, w=w0, delay_ms=delay_ms, plastic=plastic))

    def lateral_inhibition(self, pop: Pop, w_inh: float, delay_ms: float = 0.0) -> None:
        assert self.net is not None
        for i in pop.ids:
            for j in pop.ids:
                if i != j:
                    self.net.add_synapse(Synapse(pre=i, post=j, w=w_inh, delay_ms=delay_ms, plastic=False))

    # ---- API for I/O adapters ----
    def input_ids(self, name: str) -> range:
        return self.get(name).ids

    def output_ids(self, name: str) -> range:
        return self.get(name).ids
