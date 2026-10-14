from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.uav_mission.geo import GeoPoint, to_local
from src.uav_mission.utils.errors import MissionFileError

DEFAULT_ALTITUDE_M = 20.0
DEFAULT_MAX_DISTANCE_M = 1000.0

@dataclass(frozen=True)
class MissionPoint:
    id: str
    position: GeoPoint


@dataclass(frozen=True)
class MissionSpec:
    name: str
    altitude_m: float
    max_distance_from_home_m: float
    home: GeoPoint | None
    free_points: tuple[MissionPoint, ...]
    ordered_points: tuple[MissionPoint, ...]

    @property
    def all_points(self) -> tuple[MissionPoint, ...]:
        return self.free_points + self.ordered_points


def load_mission(path: str | Path) -> MissionSpec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MissionFileError([f"nie można odczytać pliku {path}: {exc}"]) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MissionFileError(
            [f"niepoprawny JSON (linia {exc.lineno}, kolumna {exc.colno}): {exc.msg}"]) from exc
    return parse_mission(data)


def parse_mission(data: Any) -> MissionSpec:
    if not isinstance(data, dict):
        raise MissionFileError(["główny element pliku musi być obiektem JSON"])

    errors: list[str] = []

    name = data.get("name", "bez nazwy")
    if not isinstance(name, str):
        errors.append("pole 'name' musi być tekstem")

    altitude = _number(data, "altitude_m", "misja", errors,
                       default=DEFAULT_ALTITUDE_M, positive=True)
    max_distance = _number(data, "max_distance_from_home_m", "misja", errors,
                           default=DEFAULT_MAX_DISTANCE_M, positive=True)
    home = _geo_point(data["home"], "home", errors) if "home" in data else None

    free = _point_list(data, "free_points", errors)
    ordered = _point_list(data, "ordered_points", errors)

    if not errors and not free and not ordered:
        errors.append("misja musi zawierać co najmniej jeden punkt")

    seen: set[str] = set()
    for point in free + ordered:
        if point.id in seen:
            errors.append(f"identyfikator punktu '{point.id}' powtarza się")
        seen.add(point.id)

    if errors:
        raise MissionFileError(errors)

    return MissionSpec(name, altitude, max_distance, home, tuple(free), tuple(ordered))


def check_distances(spec: MissionSpec, home: GeoPoint) -> None:
    """Sprawdza, czy wszystkie punkty leżą w dozwolonej odległości od startu."""
    errors = []
    for point in spec.all_points:
        distance = to_local(home, point.position).norm()
        if distance > spec.max_distance_from_home_m:
            errors.append(f"punkt '{point.id}' leży {distance:.0f} m od startu "
                          f"(limit {spec.max_distance_from_home_m:.0f} m)")
    if errors:
        raise MissionFileError(errors)


def _number(container: dict, key: str, where: str, errors: list[str], *,
            default: float | None = None, low: float | None = None,
            high: float | None = None, positive: bool = False) -> float | None:
    if key not in container:
        if default is None:
            errors.append(f"{where}: brak pola '{key}'")
        return default

    value = container[key]
    # bool jest w Pythonie podklasą int, więc odrzucamy go jawnie.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append(f"{where}: pole '{key}' musi być liczbą, otrzymano {value!r}")
        return None
    if positive and value <= 0:
        errors.append(f"{where}: pole '{key}' musi być dodatnie, otrzymano {value}")
        return None
    if (low is not None and value < low) or (high is not None and value > high):
        errors.append(f"{where}: pole '{key}' poza zakresem [{low}, {high}]: {value}")
        return None
    return float(value)


def _geo_point(raw: Any, where: str, errors: list[str]) -> GeoPoint | None:
    if not isinstance(raw, dict):
        errors.append(f"{where}: oczekiwano obiektu z polami 'lat' i 'lon'")
        return None
    lat = _number(raw, "lat", where, errors, low=-90, high=90)
    lon = _number(raw, "lon", where, errors, low=-180, high=180)
    if lat is None or lon is None:
        return None
    return GeoPoint(lat, lon)


def _point_list(data: dict, key: str, errors: list[str]) -> list[MissionPoint]:
    raw = data.get(key, [])
    if not isinstance(raw, list):
        errors.append(f"pole '{key}' musi być listą")
        return []

    points = []
    for index, item in enumerate(raw):
        where = f"{key}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where}: oczekiwano obiektu")
            continue
        point_id = item.get("id")
        if not isinstance(point_id, str) or not point_id.strip():
            errors.append(f"{where}: pole 'id' musi być niepustym tekstem")
            point_id = None
        else:
            where = f"{where} ('{point_id}')"
        position = _geo_point(item, where, errors)
        if point_id is not None and position is not None:
            points.append(MissionPoint(point_id, position))
    return points
