import math

import pytest

from src.uav_mission.geo import GeoPoint, LocalPoint, to_global, to_local

ORIGIN = GeoPoint(50.061, 19.908)


def test_origin_maps_to_zero():
    assert to_local(ORIGIN, ORIGIN) == LocalPoint(0.0, 0.0)


@pytest.mark.parametrize("north, east", [(30, 40), (-120.5, 300), (0, -500)])
def test_roundtrip(north, east):
    back = to_local(ORIGIN, to_global(ORIGIN, LocalPoint(north, east)))
    assert back.north == pytest.approx(north, abs=1e-6)
    assert back.east == pytest.approx(east, abs=1e-6)


def test_known_distance():
    point = to_global(ORIGIN, LocalPoint(30, 40))
    assert to_local(ORIGIN, point).norm() == pytest.approx(50.0, abs=1e-6)


def test_one_degree_of_latitude_is_about_111_km():
    north = to_local(ORIGIN, GeoPoint(ORIGIN.lat + 1, ORIGIN.lon)).north
    assert north == pytest.approx(111_320, rel=0.01)


def test_longitude_shrinks_with_latitude():
    at_equator = to_local(GeoPoint(0, 0), GeoPoint(0, 0.001)).east
    at_60 = to_local(GeoPoint(60, 0), GeoPoint(60, 0.001)).east
    assert at_60 == pytest.approx(at_equator * math.cos(math.radians(60)))
