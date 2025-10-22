# NeuroNet/adapters/screen/goals.py
from __future__ import annotations
from typing import Optional, Tuple
from ...core.interfaces import Frame, SimTime
from ...core.goals import Goal

def _find_dot(frame: Frame) -> Optional[Tuple[int,int]]:
    for y in range(frame.height):
        for x in range(frame.width):
            if frame.get(x, y) != 0:
                return (x, y)
    return None

def _dist_to_wall(x:int, y:int, w:int, h:int) -> int:
    return min(x, y, w-1-x, h-1-y)

class CenterSeekingGoal(Goal):
    """
    Reward moving away from walls (your current demo logic), reusable.
    """
    def reset(self) -> None: ...
    def evaluate(self, t: SimTime, obs_before: Frame, action, obs_after: Frame) -> float:
        p0 = _find_dot(obs_before)
        p1 = _find_dot(obs_after)
        if not p0 or not p1:
            return 0.0
        (x0,y0), (x1,y1) = p0, p1
        if (x0,y0) == (x1,y1):
            return -1.0
        w, h = obs_after.width, obs_after.height
        gain = _dist_to_wall(x1,y1,w,h) - _dist_to_wall(x0,y0,w,h)
        return 0.2 + 0.1 * gain
