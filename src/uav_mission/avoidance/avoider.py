from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from src.uav_mission.geo import LocalPoint
from src.uav_mission.sensors.lidar import LidarScan
from src.uav_mission.avoidance.data import AvoidanceConfig, Decision, _Frame, DecisionKind

class ObstacleAvoider:
    def __init__(self, config: AvoidanceConfig = AvoidanceConfig()):
        self.config = config
        self._consecutive_blocked = 0

    def reset(self) -> None:
        """Wywoływane przy zmianie celu: potwierdzenia liczymy od nowa."""
        self._consecutive_blocked = 0

    def check(self, scan: LidarScan, position: LocalPoint, target: LocalPoint) -> Decision:
        cfg = self.config
        distance_to_target = position.distance_to(target)
        if distance_to_target < 1e-6:
            return Decision(DecisionKind.CLEAR, reason="dron jest w celu")

        frame = _Frame.towards(position, target)
        hits = [frame.to_frame(h) for h in scan.hits()]
        reach = min(cfg.detection_range_m, distance_to_target)
        blocking = [(a, l) for a, l in hits
                    if 0.0 < a <= reach and abs(l) < cfg.corridor_half_width_m]

        if not blocking:
            self._consecutive_blocked = 0
            return Decision(DecisionKind.CLEAR)

        self._consecutive_blocked += 1
        if self._consecutive_blocked < cfg.confirm_scans:
            return Decision(DecisionKind.CLEAR, reason=(
                f"możliwa przeszkoda, potwierdzenie "
                f"{self._consecutive_blocked}/{cfg.confirm_scans}"))

        cluster = _cluster(blocking, hits, 2 * cfg.corridor_half_width_m)
        front = min(a for a, _ in blocking)
        options = []
        for side, offset in (("lewej", max(l for _, l in cluster) + cfg.detour_clearance_m),
                             ("prawej", min(l for _, l in cluster) - cfg.detour_clearance_m)):
            if abs(offset) > cfg.max_detour_offset_m:
                continue
            detour = frame.to_local(front, offset)
            if not _segment_clear(position, detour, scan.hits(), cfg.corridor_half_width_m):
                continue
            cost = position.distance_to(detour) + detour.distance_to(target)
            options.append((cost, side, detour))

        if not options:
            return Decision(DecisionKind.NO_PATH, reason=(
                f"przeszkoda {front:.1f} m przed dronem, obie strony zablokowane"))

        cost, side, detour = min(options, key=lambda o: o[0])
        return Decision(DecisionKind.DETOUR, detour, reason=(
            f"przeszkoda {front:.1f} m przed dronem, objazd z {side} strony"))


def _cluster(seeds: Sequence[tuple[float, float]], points: Sequence[tuple[float, float]],
             max_gap: float) -> list[tuple[float, float]]:
    """Punkty połączone z `seeds` łańcuchem przerw mniejszych niż `max_gap`."""
    members = list(dict.fromkeys(seeds))
    remaining = [p for p in points if p not in set(members)]
    index = 0
    while index < len(members):
        a, l = members[index]
        still_remaining = []
        for p in remaining:
            if math.hypot(p[0] - a, p[1] - l) < max_gap:
                members.append(p)
            else:
                still_remaining.append(p)
        remaining = still_remaining
        index += 1
    return members


def _segment_clear(start: LocalPoint, end: LocalPoint, points: Sequence[LocalPoint],
                   min_distance: float) -> bool:
    """Czy żaden punkt nie leży bliżej odcinka start-end niż `min_distance`."""
    dn, de = end.north - start.north, end.east - start.east
    length_sq = dn * dn + de * de
    for p in points:
        pn, pe = p.north - start.north, p.east - start.east
        t = 0.0 if length_sq == 0 else max(0.0, min(1.0, (pn * dn + pe * de) / length_sq))
        if math.hypot(pn - t * dn, pe - t * de) < min_distance:
            return False
    return True
