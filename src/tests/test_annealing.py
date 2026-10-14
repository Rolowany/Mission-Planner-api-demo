import time

import pytest

from src.tests.helpers import brute_force_optimum, random_problem
from src.uav_mission.planning import AnnealingConfig, anneal, nearest_neighbor


@pytest.mark.parametrize("seed", range(20))
def test_result_is_feasible_and_not_worse_than_start(seed):
    problem = random_problem(seed, n_points=5 + seed, n_ordered=seed % 5)
    initial = nearest_neighbor(problem)
    result = anneal(problem, initial, AnnealingConfig(iterations=5_000, seed=seed))
    assert problem.violations(result) == []
    assert problem.route_cost(result) <= problem.route_cost(initial) + 1e-9


@pytest.mark.parametrize("seed", range(15))
def test_finds_optimum_on_small_instances(seed):
    problem = random_problem(seed, n_points=7, n_ordered=seed % 4)
    result = anneal(problem, nearest_neighbor(problem), AnnealingConfig(seed=seed))
    assert problem.route_cost(result) == pytest.approx(brute_force_optimum(problem))


def test_same_seed_gives_same_route():
    problem = random_problem(seed=42, n_points=15, n_ordered=3)
    initial = nearest_neighbor(problem)
    config = AnnealingConfig(iterations=3_000, seed=123)
    assert anneal(problem, initial, config) == anneal(problem, initial, config)


def test_all_ordered_route_is_unchanged():
    problem = random_problem(seed=3, n_points=8, n_ordered=8)
    assert anneal(problem, list(problem.ordered)) == list(problem.ordered)


def test_infeasible_start_is_rejected():
    problem = random_problem(seed=0, n_points=4, n_ordered=2)
    wrong_order = list(reversed(problem.ordered)) + list(problem.free)
    with pytest.raises(ValueError, match="niepoprawna"):
        anneal(problem, wrong_order)


@pytest.mark.parametrize("n_points", [0, 1])
def test_trivial_instances(n_points):
    problem = random_problem(seed=0, n_points=n_points, n_ordered=0)
    assert anneal(problem, list(range(n_points))) == list(range(n_points))


def test_time_limit_stops_early():
    problem = random_problem(seed=0, n_points=40, n_ordered=5)
    config = AnnealingConfig(iterations=10_000_000, time_limit_s=0.2)
    start = time.monotonic()
    result = anneal(problem, nearest_neighbor(problem), config)
    assert time.monotonic() - start < 2.0
    assert problem.violations(result) == []
