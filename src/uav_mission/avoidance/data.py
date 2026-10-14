from __future__ import annotations
import math
from enum import Enum
from dataclasses import dataclass

from src.uav_mission.geo import LocalPoint
from src.uav_mission.utils.enums import DecisionKind

@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    detour: LocalPoint | None = None
    reason: str = ""


@dataclass(frozen=True)
class AvoidanceConfig:
    detection_range_m: float = 20.0      # jak daleko przed dronem reagujemy
    corridor_half_width_m: float = 3.0   # pół szerokości drogi, która musi być wolna
    detour_clearance_m: float = 5.0      # odstęp od skraju przeszkody przy objeździe
    confirm_scans: int = 3               # ile kolejnych skanów musi potwierdzić przeszkodę
    max_detour_offset_m: float = 30.0    # szerszy objazd uznajemy za niewykonalny

    def __post_init__(self):
        if self.confirm_scans < 1:
            raise ValueError("confirm_scans musi być co najmniej 1")
        if self.detour_clearance_m < self.corridor_half_width_m:
            raise ValueError("detour_clearance_m nie może być mniejsze niż korytarz")
        for name in ("detection_range_m", "corridor_half_width_m", "max_detour_offset_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} musi być dodatnie")


@dataclass(frozen=True)
class _Frame:
    """Układ związany z kierunkiem lotu: `along` do przodu, `lateral` w lewo."""
    origin: LocalPoint
    forward: tuple[float, float]  # (north, east)
    left: tuple[float, float]

    @classmethod
    def towards(cls, origin: LocalPoint, target: LocalPoint) -> _Frame:
        dn, de = target.north - origin.north, target.east - origin.east
        length = math.hypot(dn, de)
        forward = (dn / length, de / length)
        left = (forward[1], -forward[0])
        return cls(origin, forward, left)

    def to_frame(self, point: LocalPoint) -> tuple[float, float]:
        dn, de = point.north - self.origin.north, point.east - self.origin.east
        return (dn * self.forward[0] + de * self.forward[1],
                dn * self.left[0] + de * self.left[1])

    def to_local(self, along: float, lateral: float) -> LocalPoint:
        return LocalPoint(
            self.origin.north + along * self.forward[0] + lateral * self.left[0],
            self.origin.east + along * self.forward[1] + lateral * self.left[1])