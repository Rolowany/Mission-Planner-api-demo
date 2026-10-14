import json
from pathlib import Path

import pytest

from src.uav_mission.geo import GeoPoint
from src.uav_mission.mission_file import (
    DEFAULT_ALTITUDE_M, check_distances, load_mission, parse_mission)
from src.uav_mission.utils.errors import MissionFileError

EXAMPLE = Path(__file__).parent.parent.parent / "missions" / "blonia.json"


def point(pid, lat=50.0611, lon=19.908):
    return {"id": pid, "lat": lat, "lon": lon}


def errors_of(data):
    with pytest.raises(MissionFileError) as info:
        parse_mission(data)
    return info.value.errors


def test_example_mission_is_valid():
    spec = load_mission(EXAMPLE)
    assert spec.home is not None
    assert len(spec.ordered_points) < len(spec.free_points)
    check_distances(spec, spec.home)


def test_defaults_are_applied():
    spec = parse_mission({"free_points": [point("A")]})
    assert spec.altitude_m == DEFAULT_ALTITUDE_M
    assert spec.home is None
    assert spec.ordered_points == ()


def test_all_points_lists_free_then_ordered():
    spec = parse_mission({"free_points": [point("A")], "ordered_points": [point("K")]})
    assert [p.id for p in spec.all_points] == ["A", "K"]


def test_duplicate_ids_across_lists_are_rejected():
    errors = errors_of({"free_points": [point("A")], "ordered_points": [point("A")]})
    assert any("powtarza" in e for e in errors)


@pytest.mark.parametrize("bad_point, fragment", [
    (point("A", lat=91), "zakresem"),
    (point("A", lon=-181), "zakresem"),
    ({"id": "A", "lat": "50", "lon": 19.9}, "liczbą"),
    ({"id": "A", "lat": True, "lon": 19.9}, "liczbą"),
    ({"id": "A", "lon": 19.9}, "brak pola 'lat'"),
    ({"id": "", "lat": 50, "lon": 19.9}, "'id'"),
    ("A", "obiektu"),
])
def test_invalid_point_is_rejected(bad_point, fragment):
    assert any(fragment in e for e in errors_of({"free_points": [bad_point]}))


@pytest.mark.parametrize("altitude", [0, -5, "high", float("nan")])
def test_invalid_altitude_is_rejected(altitude):
    errors = errors_of({"free_points": [point("A")], "altitude_m": altitude})
    assert any("altitude_m" in e for e in errors)


def test_all_errors_are_reported_at_once():
    errors = errors_of({"altitude_m": -1, "free_points": [point("A", lat=100), point("A")],
                        "ordered_points": "nie lista"})
    assert len(errors) >= 3


def test_invalid_json_reports_location(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"free_points": [}', encoding="utf-8")
    with pytest.raises(MissionFileError, match="linia 1"):
        load_mission(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(MissionFileError, match="nie można odczytać"):
        load_mission(tmp_path / "brak.json")


def test_points_too_far_from_home_are_rejected():
    spec = parse_mission({"max_distance_from_home_m": 100,
                          "free_points": [point("BLISKO", lat=50.0611), point("DALEKO", lat=50.07)]})
    with pytest.raises(MissionFileError) as info:
        check_distances(spec, GeoPoint(50.061, 19.908))
    assert len(info.value.errors) == 1
    assert "DALEKO" in info.value.errors[0]
