"""Świat symulacji: przeszkody, o których dron nie wie przed lotem.

Przeszkody są zapisane w osobnym pliku scenariusza, a nie w pliku misji,
bo planer trasy nie może ich znać. Dron dowiaduje się o nich wyłącznie
z LiDAR-u. Ten moduł to "prawda" symulacji: korzysta z niego symulowany
czujnik oraz pomiar bezpieczeństwa (najmniejsza odległość od przeszkody).

Model: przeszkoda to pionowy walec (środek, promień, wysokość), co dobrze
oddaje drzewa, słupy i maszty. Współrzędne w pliku są w GPS, a w pamięci
w lokalnym układzie metrycznym względem miejsca startu drona.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.uav_mission.geo import GeoPoint, LocalPoint, to_local
from src.uav_mission.utils.errors import MissionError


class WorldFileError(MissionError):
    """Niepoprawny plik scenariusza. Wszystkie problemy w `errors`."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        details = "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(f"Niepoprawny scenariusz:\n{details}")


@dataclass(frozen=True)
class Obstacle:
    id: str
    center: LocalPoint
    radius_m: float
    height_m: float = math.inf  # domyślnie wyższa niż jakikolwiek lot


class World:
    def __init__(self, obstacles: Sequence[Obstacle] = ()):
        self.obstacles = tuple(obstacles)

    def ray_distance(self, origin: LocalPoint, bearing_deg: float, altitude_m: float,
                     max_range_m: float) -> float | None:
        """Odległość do najbliższej przeszkody wzdłuż promienia albo None.

        Kierunek jak w nawigacji: 0° = północ, 90° = wschód. Przeszkody
        niższe niż `altitude_m` są pomijane (promień przechodzi nad nimi).
        """
        d_north = math.cos(math.radians(bearing_deg))
        d_east = math.sin(math.radians(bearing_deg))
        nearest = None
        for obstacle in self.obstacles:
            if altitude_m >= obstacle.height_m:
                continue
            t = _ray_circle(origin, d_north, d_east, obstacle.center, obstacle.radius_m)
            if t is not None and t <= max_range_m and (nearest is None or t < nearest):
                nearest = t
        return nearest

    def clearance(self, position: LocalPoint, altitude_m: float = 0.0) -> float:
        """Odległość od powierzchni najbliższej przeszkody (ujemna = wewnątrz).

        Służy do oceny bezpieczeństwa lotu, a nie do sterowania: dron tej
        wartości nie zna. Bez przeszkód zwraca nieskończoność."""
        return min((position.distance_to(o.center) - o.radius_m
                    for o in self.obstacles if altitude_m < o.height_m),
                   default=math.inf)


def _ray_circle(origin: LocalPoint, d_north: float, d_east: float,
                center: LocalPoint, radius: float) -> float | None:
    """Najmniejsze t >= 0, dla którego origin + t*d leży na okręgu.
    Gdy origin jest wewnątrz okręgu, zwraca 0 (czujnik "zasłonięty")."""
    f_north = origin.north - center.north
    f_east = origin.east - center.east
    c = f_north ** 2 + f_east ** 2 - radius ** 2
    if c <= 0:
        return 0.0
    b = f_north * d_north + f_east * d_east
    discriminant = b * b - c
    if discriminant < 0:
        return None
    t = -b - math.sqrt(discriminant)
    return t if t >= 0 else None

def load_world(path: str | Path, home: GeoPoint) -> World:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorldFileError([f"nie można odczytać pliku {path}: {exc}"]) from exc
    except json.JSONDecodeError as exc:
        raise WorldFileError(
            [f"niepoprawny JSON (linia {exc.lineno}, kolumna {exc.colno}): {exc.msg}"]) from exc
    return parse_world(data, home)


def parse_world(data: Any, home: GeoPoint) -> World:
    if not isinstance(data, dict):
        raise WorldFileError(["główny element pliku musi być obiektem JSON"])
    raw = data.get("obstacles", [])
    if not isinstance(raw, list):
        raise WorldFileError(["pole 'obstacles' musi być listą"])

    errors: list[str] = []
    obstacles: list[Obstacle] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        where = f"obstacles[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where}: oczekiwano obiektu")
            continue
        obstacle_id = item.get("id")
        if not isinstance(obstacle_id, str) or not obstacle_id.strip():
            errors.append(f"{where}: pole 'id' musi być niepustym tekstem")
            obstacle_id = None
        elif obstacle_id in seen:
            errors.append(f"{where}: identyfikator '{obstacle_id}' powtarza się")
        else:
            seen.add(obstacle_id)
            where = f"{where} ('{obstacle_id}')"

        lat = _number(item, "lat", where, errors, low=-90, high=90)
        lon = _number(item, "lon", where, errors, low=-180, high=180)
        radius = _number(item, "radius_m", where, errors, low=0, exclusive_low=True)
        height = _number(item, "height_m", where, errors, low=0, exclusive_low=True,
                         default=math.inf)
        if None not in (obstacle_id, lat, lon, radius, height):
            center = to_local(home, GeoPoint(lat, lon))
            obstacles.append(Obstacle(obstacle_id, center, radius, height))

    if errors:
        raise WorldFileError(errors)
    return World(obstacles)


def _number(item: dict, key: str, where: str, errors: list[str], *,
            low: float | None = None, high: float | None = None,
            exclusive_low: bool = False, default: float | None = None) -> float | None:
    if key not in item:
        if default is None:
            errors.append(f"{where}: brak pola '{key}'")
        return default
    value = item[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append(f"{where}: pole '{key}' musi być liczbą, otrzymano {value!r}")
        return None
    too_low = low is not None and (value <= low if exclusive_low else value < low)
    if too_low or (high is not None and value > high):
        errors.append(f"{where}: pole '{key}' poza dozwolonym zakresem: {value}")
        return None
    return float(value)
