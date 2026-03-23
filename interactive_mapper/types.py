"""Shared data structures for the interactive mapper package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class SensorSnapshot:
    """Latest sensor state known to the mapper."""

    timestamp: float
    frame: Optional[np.ndarray] = None
    imu_data: Optional[object] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


SensorCallback = Callable[[SensorSnapshot], None]
