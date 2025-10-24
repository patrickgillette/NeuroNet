from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

from ...core.interfaces import Frame, SimTime, ScreenAction, ScreenActionType
from ...core.inout_core import Environment  # use the base contract from interfaces via inout_core

@dataclass
class SimpleScreen(Environment):
    width: int
    height: int

    def __post_init__(self):
        self._frame = Frame(self.width, self.height)
        self._dot: Optional[tuple[int, int]] = None

    # Multi-port observe: return dict keyed by input port name.
    def observe(self, t: SimTime) -> Dict[str, Any]:
        return {"screen": self._frame}

    # Multi-port apply_action: expect dict keyed by output port name(s).
    def apply_action(self, t: SimTime, actions: Dict[str, Any]) -> None:
        act = actions.get("nav")
        if isinstance(act, ScreenAction):
            self._apply_screen_action(act)

    # Helpers
    def _apply_screen_action(self, action: ScreenAction) -> None:
        if action.kind == ScreenActionType.CLEAR:
            self._frame = Frame(self.width, self.height)
            self._dot = None
            return

        if action.kind == ScreenActionType.DRAW_DOT and action.x is not None and action.y is not None:
            self._set_dot(action.x, action.y)
            return

        if action.kind == ScreenActionType.MOVE and self._dot is not None:
            dx = action.dx or 0
            dy = action.dy or 0
            x = max(0, min(self.width - 1, self._dot[0] + dx))
            y = max(0, min(self.height - 1, self._dot[1] + dy))
            self._set_dot(x, y)
            return

    def _set_dot(self, x: int, y: int) -> None:
        # clear prior dot
        for yy in range(self.height):
            for xx in range(self.width):
                if self._frame.get(xx, yy) != 0:
                    self._frame.set(xx, yy, 0)
        self._frame.set(x, y, 1)
        self._dot = (x, y)
