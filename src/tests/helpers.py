from __future__ import annotations

import itertools
import random

from src.uav_mission.geo import LocalPoint
from src.uav_mission.planning import RoutingProblem


def random_problem(seed: int, n_points: int, n_ordered: int,
                   size_m: float = 500.0) -> RoutingProblem:
    rng = random.Random(seed)
    points = [LocalPoint(rng.uniform(-size_m, size_m), rng.uniform(-size_m, size_m))
              for _ in range(n_points)]
    ordered = rng.sample(range(n_points), n_ordered)
    return RoutingProblem.from_points(LocalPoint(0, 0), points, ordered)


def brute_force_optimum(problem: RoutingProblem) -> float:
    """Najkrótsza poprawna trasa przez sprawdzenie wszystkich permutacji.
    Tylko dla małych instancji (do ok. 8 punktów)."""
    assert problem.num_points <= 8, "za duża instancja na brute force"
    return min(problem.route_cost(perm)
               for perm in itertools.permutations(range(problem.num_points))
               if problem.is_order_respected(perm))
