"""Frame transport: every frame carries identity and age so staleness is expressible."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class Frame:
    """A single captured frame.

    `seq` and `ts` exist so a consumer can tell a fresh frame from a frozen one. The
    previous implementation returned `(True, last_frame.copy())` forever after the
    stream died, which turned the bot into a blind tap generator.
    """

    seq: int
    ts: float
    bgr: np.ndarray

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def wh(self) -> tuple[int, int]:
        return self.width, self.height

    def age(self, now: Optional[float] = None) -> float:
        return (time.perf_counter() if now is None else now) - self.ts


class FrameSource(Protocol):
    """Anything that can yield frames: scrcpy, a directory of PNGs, a fake."""

    def read(self) -> Optional[Frame]:
        """Return the newest frame, or None if none is available or it is too stale."""
        ...

    def healthy(self) -> bool:
        """False once the source is known to be dead and cannot recover itself."""
        ...

    def release(self) -> None:
        ...
