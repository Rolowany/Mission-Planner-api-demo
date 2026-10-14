import pytest

from src.tests.helpers import random_problem
from src.uav_mission.geo import LocalPoint
from src.uav_mission.planning import RoutingProblem, nearest_neighbor


@pytest.mark.parametrize("seed", range(30))
def test_result_is_always_feasible(seed):
    n_points = 3 + seed % 20
    problem = random_problem(seed, n_points, n_ordered=seed % 4)
    assert problem.violations(nearest_neighbor(problem)) == []


def test_visits_nearest_point_first_on_a_line():
    points = [LocalPoint(0, 30), LocalPoint(0, 10), LocalPoint(0, 20)]
    problem = RoutingProblem.from_points(LocalPoint(0, 0), points)
    assert nearest_neighbor(problem) == [1, 2, 0]


def test_waits_for_earlier_ordered_point():
    points = [LocalPoint(0, 5), LocalPoint(0, 50)]
    problem = RoutingProblem.from_points(LocalPoint(0, 0), points, ordered=[1, 0])
    assert nearest_neighbor(problem) == [1, 0]


def test_all_points_ordered_gives_the_only_valid_route():
    problem = random_problem(seed=1, n_points=6, n_ordered=6)
    assert nearest_neighbor(problem) == list(problem.ordered)


def test_no_points_gives_empty_route():
    problem = RoutingProblem.from_points(LocalPoint(0, 0), [])
    assert nearest_neighbor(problem) == []
