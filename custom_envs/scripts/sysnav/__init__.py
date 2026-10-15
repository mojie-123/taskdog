"""SysNav-style semantic object search components for taskdog.

The package turns language, RGB-D observations and the anonymous room map into
decisions executed by ``navigation/navigate_mujoco_sysnav.py`` with the
existing MuJoCo/Nav2/N2M stack.
"""

from .config import SysNavConfig, load_sysnav_config
from .runtime import SysNavComponents, build_sysnav_components
from .search_runner import SearchResult, SysNavSearchRunner
from .types import (
    Detection2D,
    EarlyStopDecision,
    LocalizedDetection,
    RoomDescription,
    RoomSelection,
    SearchInstruction,
    TargetVerification,
)

__all__ = [
    "Detection2D",
    "EarlyStopDecision",
    "LocalizedDetection",
    "RoomDescription",
    "RoomSelection",
    "SearchInstruction",
    "SearchResult",
    "SysNavConfig",
    "SysNavComponents",
    "SysNavSearchRunner",
    "TargetVerification",
    "load_sysnav_config",
    "build_sysnav_components",
]
