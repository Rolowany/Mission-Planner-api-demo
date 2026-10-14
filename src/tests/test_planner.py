import pytest

from src.tests.helpers import random_problem
from src.uav_mission.geo import GeoPoint
from src.uav_mission.mission_file import parse_mission
from src.uav_mission.planning import nearest_neighbor, plan_route, problem_from_mission


@pytest.fixture
def problem():
    return random_problem(seed=5, n_points=12, n_ordered=3)


def test_default_plan_is_feasible_and_beats_greedy(problem):
    result = plan_route(problem)
    assert problem.violations(result.route) == []
    assert result.cost <= problem.route_cost(nearest_neighbor(problem))


def test_without_improvers_greedy_is_used(problem):
    result = plan_route(problem, improvers=[])
    assert result.solver == "greedy"
    assert result.route == nearest_neighbor(problem)


def test_failing_improver_falls_back_to_greedy(problem):
    def broken(_problem, _route):
        raise RuntimeError("solwer niedostępny")

    result = plan_route(problem, improvers=[("broken", broken)])
    assert result.solver == "greedy"
    assert result.attempts[-1].status == "error"
    assert "niedostępny" in result.attempts[-1].message


def test_infeasible_improver_result_is_discarded(problem):
    def cheater(_problem, route):
        return route[:-1]

    result = plan_route(problem, improvers=[("cheater", cheater)])
    assert result.solver == "greedy"
    assert result.attempts[-1].status == "infeasible"
    assert problem.violations(result.route) == []


def test_worse_improver_result_is_not_used(problem):
    def worse(p, _route):
        return list(p.ordered) + list(p.free)

    greedy_cost = problem.route_cost(nearest_neighbor(problem))
    assert problem.route_cost(worse(problem, None)) > greedy_cost
    result = plan_route(problem, improvers=[("worse", worse)])
    assert result.solver == "greedy"


def test_improvers_are_chained_and_best_wins(problem):
    improved = plan_route(problem).route
    received = []

    def recording(_problem, route):
        received.append(list(route))
        return route

    result = plan_route(problem, improvers=[
        ("first", lambda _p, _r: improved), ("recording", recording)])
    assert received == [improved]  # druga metoda dostała wynik pierwszej
    assert result.solver == "first"  # równy wynik nie zastępuje wcześniejszego


def test_problem_from_mission_maps_indices():
    spec = parse_mission({
        "free_points": [{"id": "A", "lat": 50.0611, "lon": 19.908}],
        "ordered_points": [{"id": "K1", "lat": 50.0612, "lon": 19.908},
                           {"id": "K2", "lat": 50.0613, "lon": 19.908}],
    })
    problem = problem_from_mission(spec, GeoPoint(50.061, 19.908))
    assert problem.free == (0,)
    assert problem.ordered == (1, 2)
    assert problem.distance(problem.home, 0) == pytest.approx(11.1, abs=0.1)
