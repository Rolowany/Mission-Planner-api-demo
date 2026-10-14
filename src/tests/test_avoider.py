"""Testy decyzji omijania na skanach z symulowanego LiDAR-u (bez SITL)."""
import math

import pytest

from src.uav_mission.avoidance.avoider import (
    AvoidanceConfig, DecisionKind, ObstacleAvoider)
from src.uav_mission.geo import LocalPoint
from src.uav_mission.sensors.lidar import LidarConfig, LidarScan, SimulatedLidar
from src.uav_mission.simulation.world import Obstacle, World

ALT = 20.0
HERE = LocalPoint(0, 0)
FAR_NORTH = LocalPoint(100, 0)
IMMEDIATE = AvoidanceConfig(confirm_scans=1)


def tree(north, east, radius=2.0, name="t"):
    return Obstacle(name, LocalPoint(north, east), radius)


def decide(world, position=HERE, target=FAR_NORTH, config=IMMEDIATE, beams=360):
    scan = SimulatedLidar(world, LidarConfig(num_beams=beams, noise_std_m=0.0)).scan(position, ALT)
    return ObstacleAvoider(config).check(scan, position, target)


def min_clearance(world, start, end, step=0.25):
    """Najmniejsza odległość od przeszkód wzdłuż odcinka (prawda symulacji)."""
    length = start.distance_to(end)
    n = max(1, int(length / step))
    return min(world.clearance(LocalPoint(start.north + (end.north - start.north) * i / n,
                                          start.east + (end.east - start.east) * i / n))
               for i in range(n + 1))


def rotate(point, degrees):
    a = math.radians(degrees)
    return LocalPoint(point.north * math.cos(a) - point.east * math.sin(a),
                      point.north * math.sin(a) + point.east * math.cos(a))


def test_empty_world_is_clear():
    assert decide(World()).kind is DecisionKind.CLEAR


def test_obstacle_beside_path_is_ignored():
    assert decide(World([tree(10, 7, radius=1)])).kind is DecisionKind.CLEAR


def test_obstacle_behind_target_is_ignored():
    assert decide(World([tree(14, 0)]), target=LocalPoint(10, 0)).kind is DecisionKind.CLEAR


def test_obstacle_behind_drone_is_ignored():
    assert decide(World([tree(-8, 0)])).kind is DecisionKind.CLEAR


def test_obstacle_beyond_detection_range_waits_until_closer():
    world = World([tree(25, 0)])
    assert decide(world).kind is DecisionKind.CLEAR
    assert decide(world, position=LocalPoint(12, 0)).kind is DecisionKind.DETOUR


def test_drone_at_target_is_clear():
    assert decide(World([tree(5, 0)]), target=HERE).kind is DecisionKind.CLEAR


@pytest.mark.parametrize("heading", [0, 90, 135, 225, 300])
def test_detour_around_tree_keeps_safe_distance(heading):
    obstacle_center = rotate(LocalPoint(12, 0), heading)
    target = rotate(FAR_NORTH, heading)
    world = World([Obstacle("t", obstacle_center, 4.0)])

    decision = decide(world, target=target)
    assert decision.kind is DecisionKind.DETOUR
    assert min_clearance(world, HERE, decision.detour) >= IMMEDIATE.corridor_half_width_m - 0.5
    assert min_clearance(world, decision.detour, target) > 1.0


def test_detour_goes_to_the_shorter_side():
    left_of_path = decide(World([tree(12, -1.5, radius=3)]))   # przesunięte na zachód
    right_of_path = decide(World([tree(12, 1.5, radius=3)]))   # przesunięte na wschód
    assert right_of_path.detour.east < 0 < left_of_path.detour.east


def test_blocked_side_is_avoided_even_if_shorter():
    world = World([tree(12, 1.0, radius=3, name="na_trasie"),
                   tree(10, -9, radius=2, name="blokuje_zachod")])
    decision = decide(world)
    assert decision.kind is DecisionKind.DETOUR
    assert decision.detour.east > 0
    assert min_clearance(world, HERE, decision.detour) > 2.0


def test_trees_too_close_to_pass_between_are_one_obstacle():
    world = World([tree(12, 0, radius=2, name="a"), tree(12, 6, radius=2, name="b")])
    decision = decide(world)
    assert decision.kind is DecisionKind.DETOUR
    detour = decision.detour
    assert detour.east < -2 or detour.east > 8
    assert min_clearance(world, HERE, detour) > 2.0


def test_wall_without_gap_gives_no_path():
    wall = World([tree(12, east, radius=2, name=f"w{east}") for east in range(-40, 41, 3)])
    decision = decide(wall)
    assert decision.kind is DecisionKind.NO_PATH
    assert "zablokowane" in decision.reason


def single_reading_scan(distance, bearing=0.0):
    bearings = tuple(float(b) for b in range(0, 360, 5))
    ranges = tuple(distance if b == bearing else None for b in bearings)
    return LidarScan(HERE, bearings, ranges, 30.0)


def empty_scan():
    return single_reading_scan(None)


def test_detour_requires_consecutive_confirmations():
    avoider = ObstacleAvoider(AvoidanceConfig(confirm_scans=3))
    scan = single_reading_scan(8.0)
    first, second, third = (avoider.check(scan, HERE, FAR_NORTH) for _ in range(3))
    assert first.kind is second.kind is DecisionKind.CLEAR
    assert "1/3" in first.reason and "2/3" in second.reason
    assert third.kind is DecisionKind.DETOUR


def test_single_spurious_reading_does_not_trigger_detour():
    avoider = ObstacleAvoider(AvoidanceConfig(confirm_scans=3))
    sequence = [single_reading_scan(6.0), empty_scan(), single_reading_scan(6.0), empty_scan()]
    assert all(avoider.check(s, HERE, FAR_NORTH).kind is DecisionKind.CLEAR for s in sequence)


def test_reset_restarts_confirmation():
    avoider = ObstacleAvoider(AvoidanceConfig(confirm_scans=2))
    scan = single_reading_scan(8.0)
    avoider.check(scan, HERE, FAR_NORTH)
    avoider.reset()
    assert avoider.check(scan, HERE, FAR_NORTH).kind is DecisionKind.CLEAR


def test_noisy_lidar_with_spurious_readings_still_detects_tree():
    world = World([tree(12, 0, radius=3)])
    lidar = SimulatedLidar(world, LidarConfig(noise_std_m=0.1, spurious_prob=0.02,
                                              dropout_prob=0.1, seed=3))
    avoider = ObstacleAvoider(AvoidanceConfig(confirm_scans=3))
    decisions = [avoider.check(lidar.scan(HERE, ALT), HERE, FAR_NORTH) for _ in range(3)]
    assert decisions[-1].kind is DecisionKind.DETOUR
    assert min_clearance(world, HERE, decisions[-1].detour) > 2.0


@pytest.mark.parametrize("kwargs", [
    {"confirm_scans": 0}, {"detection_range_m": 0},
    {"corridor_half_width_m": 4, "detour_clearance_m": 3},
])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        AvoidanceConfig(**kwargs)
