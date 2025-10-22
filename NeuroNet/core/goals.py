# NeuroNet/core/goals.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
from .interfaces import SimTime

class Goal(ABC):
    """
    Computes scalar reward (or any training signal) from the agent/environment
    transition at time t. The runtime will call evaluate() after an action.
    """
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def evaluate(self, t: SimTime, obs_before: Any, action: Any, obs_after: Any) -> float:
        ...

class NullGoal(Goal):
    def reset(self) -> None: ...
    def evaluate(self, t: SimTime, obs_before: Any, action: Any, obs_after: Any) -> float:
        return 0.0
