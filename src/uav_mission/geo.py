from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class GeoPoint:
    """Punkt we współrzędnych geograficznych (stopnie)."""
    lat: float
    lon: float


@dataclass(frozen=True)
class LocalPoint:
    """Punkt w lokalnym układzie metrycznym."""
    north: float
    east: float

    def distance_to(self, other: LocalPoint) -> float:
        return math.hypot(self.north - other.north, self.east - other.east)

    def norm(self) -> float:
        return math.hypot(self.north, self.east)


def to_local(origin: GeoPoint, point: GeoPoint) -> LocalPoint:
    """Przelicza punkt GPS na metry względem `origin`."""
    north = math.radians(point.lat - origin.lat) * EARTH_RADIUS_M
    east = (math.radians(point.lon - origin.lon)
            * EARTH_RADIUS_M * math.cos(math.radians(origin.lat)))
    return LocalPoint(north, east)


def to_global(origin: GeoPoint, point: LocalPoint) -> GeoPoint:
    """Odwrotność `to_local`."""
    lat = origin.lat + math.degrees(point.north / EARTH_RADIUS_M)
    lon = origin.lon + math.degrees(
        point.east / (EARTH_RADIUS_M * math.cos(math.radians(origin.lat))))
    return GeoPoint(lat, lon)
