import math
import statistics

import pytest

from src.uav_mission.geo import LocalPoint
from src.uav_mission.sensors.lidar import LidarConfig, SensorError, SimulatedLidar
from src.uav_mission.simulation.world import Obstacle, World

ORIGIN = LocalPoint(0, 0)
ALT = 20.0
EXACT = LidarConfig(noise_std_m=0.0)  # bez szumu: odczyty dokładne


def tree(north, east, radius=2.0, height=math.inf):
    return Obstacle("t", LocalPoint(north, east), radius, height)


def reading(scan, bearing):
    return scan.ranges_m[scan.bearings_deg.index(bearing)]


def test_beams_cover_full_circle_evenly():
    scan = SimulatedLidar(World(), LidarConfig(num_beams=8)).scan(ORIGIN, ALT)
    assert scan.bearings_deg == (0, 45, 90, 135, 180, 225, 270, 315)


def test_empty_world_gives_no_readings():
    scan = SimulatedLidar(World(), EXACT).scan(ORIGIN, ALT)
    assert all(r is None for r in scan.ranges_m)
    assert scan.closest() is None


def test_obstacle_ahead_is_measured_exactly():
    scan = SimulatedLidar(World([tree(20, 0)]), EXACT).scan(ORIGIN, ALT)
    assert reading(scan, 0) == pytest.approx(18.0)
    assert reading(scan, 180) is None
    assert scan.closest() == pytest.approx(18.0)


def test_scan_follows_drone_position():
    lidar = SimulatedLidar(World([tree(20, 0)]), EXACT)
    assert reading(lidar.scan(LocalPoint(10, 0), ALT), 0) == pytest.approx(8.0)


def test_low_obstacle_is_not_seen_from_above():
    lidar = SimulatedLidar(World([tree(20, 0, height=10)]), EXACT)
    assert reading(lidar.scan(ORIGIN, 20.0), 0) is None


def test_hits_lie_on_obstacle_surface():
    obstacle = tree(15, 5, radius=3)
    scan = SimulatedLidar(World([obstacle]), EXACT).scan(ORIGIN, ALT)
    hits = scan.hits()
    assert hits
    for hit in hits:
        assert hit.distance_to(obstacle.center) == pytest.approx(obstacle.radius_m, abs=1e-6)


def test_noise_is_unbiased_and_reproducible():
    world = World([tree(20, 0)])
    config = LidarConfig(noise_std_m=0.1, seed=7)
    values = [reading(SimulatedLidar(world, config).scan(ORIGIN, ALT), 0) for _ in range(3)]
    assert values[0] == values[1] == values[2]  # to samo ziarno -> te same odczyty

    lidar = SimulatedLidar(world, config)
    samples = [reading(lidar.scan(ORIGIN, ALT), 0) for _ in range(500)]
    assert statistics.mean(samples) == pytest.approx(18.0, abs=0.02)
    assert 0.08 < statistics.stdev(samples) < 0.12


def test_noisy_reading_never_negative_or_beyond_range():
    lidar = SimulatedLidar(World([tree(0, 0, radius=1)]),
                           LidarConfig(noise_std_m=0.5, max_range_m=30))
    for _ in range(50):
        scan = lidar.scan(LocalPoint(1.01, 0), ALT)
        assert all(r is None or 0.0 <= r <= 30.0 for r in scan.ranges_m)


def test_full_dropout_gives_no_readings():
    scan = SimulatedLidar(World([tree(20, 0)]), LidarConfig(dropout_prob=1.0)).scan(ORIGIN, ALT)
    assert all(r is None for r in scan.ranges_m)


def test_spurious_readings_appear_in_empty_world():
    lidar = SimulatedLidar(World(), LidarConfig(spurious_prob=0.05, seed=1))
    readings = [r for _ in range(20) for r in lidar.scan(ORIGIN, ALT).ranges_m if r is not None]
    assert readings  # są fałszywe odczyty, choć nie ma przeszkód
    assert all(0.5 <= r <= 30.0 for r in readings)


def test_sensor_failure_after_configured_scans():
    lidar = SimulatedLidar(World(), LidarConfig(fail_after_scans=2))
    lidar.scan(ORIGIN, ALT)
    lidar.scan(ORIGIN, ALT)
    with pytest.raises(SensorError):
        lidar.scan(ORIGIN, ALT)


@pytest.mark.parametrize("kwargs", [
    {"num_beams": 0}, {"max_range_m": 0}, {"noise_std_m": -1},
    {"dropout_prob": 1.5}, {"spurious_prob": -0.1},
])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LidarConfig(**kwargs)


def test_min_clearance_tracks_closest_approach():
    lidar = SimulatedLidar(World([tree(20, 0, radius=2)]), EXACT)
    assert lidar.min_clearance_m == math.inf
    for north in (0, 10, 15, 5):
        lidar.scan(LocalPoint(north, 0), ALT)
    assert lidar.min_clearance_m == pytest.approx(3.0)  # z pozycji 15 m do powierzchni 18 m
