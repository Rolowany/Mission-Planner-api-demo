from dataclasses import dataclass
from src.uav_mission.geo import GeoPoint

@dataclass(frozen=True)
class Waypoint:
    id: str
    position: GeoPoint
    ordered: bool


@dataclass(frozen=True)
class ControllerConfig:
    altitude_m: float = 15.0
    accept_radius_m: float = 2.0
    cruise_speed_mps: float = 5.0
    takeoff_timeout_s: float = 60.0
    leg_base_timeout_s: float = 20.0
    land_timeout_s: float = 300.0
    max_detour_depth: int = 3