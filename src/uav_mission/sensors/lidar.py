from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from src.uav_mission.geo import LocalPoint
from src.uav_mission.simulation.world import World
from src.uav_mission.utils.errors import MissionError


class SensorError(MissionError):
    """Czujnik nie dostarczył danych. Lot "na ślepo" jest niedopuszczalny."""


@dataclass(frozen=True)
class LidarScan:
    position: LocalPoint             # skąd wykonano skan
    bearings_deg: tuple[float, ...]
    ranges_m: tuple[float | None, ...]
    max_range_m: float

    def hits(self) -> list[LocalPoint]:
        """Punkty odbicia w lokalnym układzie metrycznym."""
        points = []
        for bearing, distance in zip(self.bearings_deg, self.ranges_m):
            if distance is not None:
                points.append(LocalPoint(
                    self.position.north + distance * math.cos(math.radians(bearing)),
                    self.position.east + distance * math.sin(math.radians(bearing))))
        return points

    def closest(self) -> float | None:
        valid = [r for r in self.ranges_m if r is not None]
        return min(valid, default=None)


class Lidar(Protocol):
    def scan(self, position: LocalPoint, altitude_m: float) -> LidarScan:
        """Wykonuje skan. Rzuca SensorError, gdy czujnik nie działa."""


@dataclass(frozen=True)
class LidarConfig:
    num_beams: int = 72              # co 5°
    max_range_m: float = 30.0
    noise_std_m: float = 0.05        # szum gaussowski odczytu
    dropout_prob: float = 0.0        # szansa zgubienia wiązki
    spurious_prob: float = 0.0       # szansa fałszywego krótkiego odczytu
    fail_after_scans: int | None = None  # awaria czujnika po tylu skanach
    seed: int = 0

    def __post_init__(self):
        if self.num_beams < 1:
            raise ValueError("num_beams musi być dodatnie")
        if self.max_range_m <= 0:
            raise ValueError("max_range_m musi być dodatnie")
        if self.noise_std_m < 0:
            raise ValueError("noise_std_m nie może być ujemne")
        for name in ("dropout_prob", "spurious_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} musi być w przedziale [0, 1]")


class SimulatedLidar:
    def __init__(self, world: World, config: LidarConfig = LidarConfig()):
        self._world = world
        self._config = config
        self._rng = random.Random(config.seed)
        self._scans_done = 0
        # Tylko dla symulacji: najmniejsza odległość drona od przeszkody
        # w chwilach wykonywania skanów. Dron jej nie zna; służy do oceny testu.
        self.min_clearance_m = math.inf
        step = 360.0 / config.num_beams
        self._bearings = tuple(i * step for i in range(config.num_beams))

    def scan(self, position: LocalPoint, altitude_m: float) -> LidarScan:
        cfg = self._config
        if cfg.fail_after_scans is not None and self._scans_done >= cfg.fail_after_scans:
            raise SensorError("LiDAR nie odpowiada (symulowana awaria)")
        self._scans_done += 1
        self.min_clearance_m = min(self.min_clearance_m,
                                   self._world.clearance(position, altitude_m))

        ranges = tuple(self._measure(position, bearing, altitude_m)
                       for bearing in self._bearings)
        return LidarScan(position, self._bearings, ranges, cfg.max_range_m)

    def _measure(self, position: LocalPoint, bearing: float, altitude_m: float) -> float | None:
        cfg = self._config
        if self._rng.random() < cfg.dropout_prob:
            return None
        if self._rng.random() < cfg.spurious_prob:
            return self._rng.uniform(0.5, cfg.max_range_m)

        distance = self._world.ray_distance(position, bearing, altitude_m, cfg.max_range_m)
        if distance is None:
            return None
        if cfg.noise_std_m > 0:
            distance += self._rng.gauss(0.0, cfg.noise_std_m)
        return min(max(distance, 0.0), cfg.max_range_m)
