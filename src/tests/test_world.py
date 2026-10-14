import json
import math
from pathlib import Path

import pytest

from src.uav_mission.geo import GeoPoint, LocalPoint, to_global
from src.uav_mission.simulation.world import (
    Obstacle, World, WorldFileError, load_world, parse_world)
from src.uav_mission.geo import to_local


HOME = GeoPoint(50.061, 19.908)
SCENARIOS = Path(__file__).parent.parent.parent / "scenarios"


def tree(north, east, radius=2.0, height=math.inf, name="t"):
    return Obstacle(name, LocalPoint(north, east), radius, height)


def test_ray_hits_obstacle_surface():
    world = World([tree(20, 0, radius=2)])
    assert world.ray_distance(LocalPoint(0, 0), 0, 10, 30) == pytest.approx(18.0)


@pytest.mark.parametrize("bearing", [90, 180, 270])
def test_ray_in_other_direction_misses(bearing):
    world = World([tree(20, 0)])
    assert world.ray_distance(LocalPoint(0, 0), bearing, 10, 30) is None


def test_bearing_90_points_east():
    world = World([tree(0, 15, radius=1)])
    assert world.ray_distance(LocalPoint(0, 0), 90, 10, 30) == pytest.approx(14.0)


def test_oblique_ray():
    # Przeszkoda w kierunku 45°, w odległości 10*sqrt(2) od środka.
    world = World([tree(10, 10, radius=1)])
    distance = world.ray_distance(LocalPoint(0, 0), 45, 10, 30)
    assert distance == pytest.approx(10 * math.sqrt(2) - 1)


def test_obstacle_beyond_range_is_not_seen():
    world = World([tree(50, 0)])
    assert world.ray_distance(LocalPoint(0, 0), 0, 10, 30) is None


def test_nearest_of_two_obstacles_is_reported():
    world = World([tree(25, 0, name="far"), tree(10, 0, name="near")])
    assert world.ray_distance(LocalPoint(0, 0), 0, 10, 30) == pytest.approx(8.0)


def test_ray_grazing_side_of_obstacle():
    # Promień na północ mija środek o 1.5 m, promień 2 m: przecięcie istnieje.
    world = World([tree(20, 1.5, radius=2)])
    expected = 20 - math.sqrt(2 ** 2 - 1.5 ** 2)
    assert world.ray_distance(LocalPoint(0, 0), 0, 10, 30) == pytest.approx(expected)


def test_inside_obstacle_reads_zero():
    world = World([tree(0, 0, radius=3)])
    assert world.ray_distance(LocalPoint(1, 0), 180, 10, 30) == 0.0


def test_flying_above_obstacle_is_not_seen():
    world = World([tree(20, 0, height=15)])
    assert world.ray_distance(LocalPoint(0, 0), 0, 20, 30) is None
    assert world.ray_distance(LocalPoint(0, 0), 0, 10, 30) == pytest.approx(18.0)


# --- miara bezpieczeństwa ---------------------------------------------

def test_clearance_measures_distance_to_surface():
    world = World([tree(10, 0, radius=2), tree(0, 30, radius=5)])
    assert world.clearance(LocalPoint(0, 0)) == pytest.approx(8.0)
    assert world.clearance(LocalPoint(10, 1)) == pytest.approx(-1.0)  # wewnątrz


def test_clearance_without_obstacles_is_infinite():
    assert World().clearance(LocalPoint(0, 0)) == math.inf


# --- plik scenariusza -------------------------------------------------

def obstacle_json(**overrides):
    base = {"id": "a", "lat": 50.0612, "lon": 19.908, "radius_m": 3}
    base.update(overrides)
    return base


def errors_of(data):
    with pytest.raises(WorldFileError) as info:
        parse_world(data, HOME)
    return info.value.errors


def test_obstacle_is_converted_to_local_coordinates():
    position = to_global(HOME, LocalPoint(30, -12))
    world = parse_world({"obstacles": [obstacle_json(lat=position.lat, lon=position.lon)]}, HOME)
    center = world.obstacles[0].center
    assert center.north == pytest.approx(30, abs=1e-6)
    assert center.east == pytest.approx(-12, abs=1e-6)
    assert world.obstacles[0].height_m == math.inf


def test_empty_scenario_is_valid():
    assert parse_world({"obstacles": []}, HOME).obstacles == ()


@pytest.mark.parametrize("bad, fragment", [
    (obstacle_json(radius_m=0), "radius_m"),
    (obstacle_json(radius_m=-1), "radius_m"),
    (obstacle_json(height_m=0), "height_m"),
    (obstacle_json(lat=95), "lat"),
    (obstacle_json(lon="x"), "liczbą"),
    ({"id": "a", "lat": 50, "lon": 19}, "brak pola 'radius_m'"),
    (obstacle_json(id=""), "'id'"),
])
def test_invalid_obstacle_is_rejected(bad, fragment):
    assert any(fragment in e for e in errors_of({"obstacles": [bad]}))


def test_duplicate_ids_and_multiple_errors_are_reported():
    errors = errors_of({"obstacles": [obstacle_json(), obstacle_json(), obstacle_json(id="b", radius_m=-2)]})
    assert any("powtarza" in e for e in errors)
    assert any("radius_m" in e for e in errors)


def test_invalid_json_reports_location(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"obstacles": [', encoding="utf-8")
    with pytest.raises(WorldFileError, match="linia 1"):
        load_world(path, HOME)


@pytest.mark.parametrize("name", ["no_obstacles.json", "single_tree.json"])
def test_repository_scenarios_load(name):
    load_world(SCENARIOS / name, HOME)


def test_single_tree_lies_on_planned_route_leg():
    # Drzewo stoi na środku odcinka K2 -> F7 trasy planowanej dla missions/example.json.
    mission = json.loads((SCENARIOS.parent / "missions" / "example.json").read_text())
    by_id = {p["id"]: GeoPoint(p["lat"], p["lon"])
             for p in mission["free_points"] + mission["ordered_points"]}
    k2, f7 = to_local(HOME, by_id["K2"]), to_local(HOME, by_id["F7"])
    middle = LocalPoint((k2.north + f7.north) / 2, (k2.east + f7.east) / 2)
    world = load_world(SCENARIOS / "single_tree.json", HOME)
    assert world.clearance(middle) < 0  # środek odcinka jest wewnątrz drzewa
