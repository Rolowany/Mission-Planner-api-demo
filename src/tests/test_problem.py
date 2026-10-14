import pytest

from src.uav_mission.geo import LocalPoint
from src.uav_mission.planning import RoutingProblem

# Kwadrat 10x10 m: start w (0,0), punkty w pozostałych narożnikach.
SQUARE = [LocalPoint(0, 10), LocalPoint(10, 10), LocalPoint(10, 0)]


def square_problem(ordered=()):
    return RoutingProblem.from_points(LocalPoint(0, 0), SQUARE, ordered)


def test_route_cost_includes_return_home():
    assert square_problem().route_cost([0, 1, 2]) == pytest.approx(40.0)


def test_empty_problem_has_zero_cost():
    problem = RoutingProblem.from_points(LocalPoint(0, 0), [])
    assert problem.num_points == 0
    assert problem.route_cost([]) == 0.0
    assert problem.is_feasible([])


def test_free_and_ordered_partition():
    problem = square_problem(ordered=[2, 0])
    assert problem.ordered == (2, 0)
    assert problem.free == (1,)


def test_valid_route_has_no_violations():
    assert square_problem(ordered=[2, 0]).violations([2, 1, 0]) == []


def test_wrong_order_is_detected():
    problem = square_problem(ordered=[2, 0])
    assert not problem.is_order_respected([0, 1, 2])
    assert any("kolejność" in v for v in problem.violations([0, 1, 2]))


def test_missing_point_is_detected():
    assert any("brakujące" in v for v in square_problem().violations([0, 1]))


def test_duplicate_point_is_detected():
    assert any("więcej niż raz" in v for v in square_problem().violations([0, 1, 1, 2]))


@pytest.mark.parametrize("bad", [3, -1, "A", 1.0])
def test_unknown_point_is_detected(bad):
    assert any("nieznany" in v for v in square_problem().violations([0, 1, 2, bad]))


@pytest.mark.parametrize("distances, ordered, message", [
    ([], (), "startu"),
    ([[0, 1], [1]], (), "kwadratowa"),
    ([[0, -1], [1, 0]], (), "ujemne"),
    ([[0, 1], [1, 0]], (0, 0), "powtarzać"),
    ([[0, 1], [1, 0]], (1,), "zakresem"),
])
def test_invalid_construction_is_rejected(distances, ordered, message):
    with pytest.raises(ValueError, match=message):
        RoutingProblem(distances, ordered)
