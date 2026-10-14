from __future__ import annotations

import math

from src.uav_mission.utils.errors import MissionError

EARTH_RADIUS_M = 6_378_137.0


def sitl_home_argument(lat: float, lon: float, ground_alt_m: float = 0.0,
                       heading_deg: float = 0.0) -> str:
    """Linijka do pola 'Extra command line' w zakładce Simulation."""
    return f"--home={lat:.7f},{lon:.7f},{ground_alt_m:g},{heading_deg:g}"


def check_start_position(actual: tuple[float, float], expected: tuple[float, float],
                         max_distance_m: float = 50.0) -> float:
    distance = _distance_m(*actual, *expected)
    if distance > max_distance_m:
        raise MissionError(
            f"Dron stoi {distance / 1000:.1f} km od miejsca startu z pliku misji.\n"
            "W SITL: Mission Planner -> Simulation -> 'Extra command line' wpisz\n"
            f"    {sitl_home_argument(*expected)}\n"
            "zaznacz 'Wipe' i uruchom symulację ponownie (albo użyj --relocate).")
    return distance


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return math.hypot(x, y) * EARTH_RADIUS_M