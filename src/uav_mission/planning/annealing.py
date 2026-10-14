from __future__ import annotations

import math
import random
import time

from src.uav_mission.planning.problem import Route, RoutingProblem
from src.uav_mission.planning.data import AnnealingConfig

def anneal(problem: RoutingProblem, initial: Route,
           config: AnnealingConfig = AnnealingConfig()) -> Route:
    violations = problem.violations(initial)
    if violations:
        raise ValueError(f"trasa startowa jest niepoprawna: {violations}")

    n = len(initial)
    current = list(initial)
    current_cost = problem.route_cost(current)
    if n < 2 or config.iterations <= 0 or current_cost == 0:
        return current

    rng = random.Random(config.seed)
    best, best_cost = current, current_cost

    cache: dict[tuple, float | None] = {tuple(current): current_cost}

    avg_edge = current_cost / (n + 1)
    temp = config.start_temp_factor * avg_edge
    end_temp = config.end_temp_factor * avg_edge
    cooling = (end_temp / temp) ** (1.0 / config.iterations)

    deadline = None
    if config.time_limit_s is not None:
        deadline = time.monotonic() + config.time_limit_s

    for iteration in range(config.iterations):
        if deadline is not None and iteration % 256 == 0 and time.monotonic() > deadline:
            break

        candidate = _random_neighbor(current, rng)
        candidate_tuple = tuple(candidate)

        if candidate_tuple in cache:
            cost = cache[candidate_tuple]
            is_valid = cost is not None
        else:
            is_valid = problem.is_order_respected(candidate)
            if is_valid:
                cost = problem.route_cost(candidate)
                cache[candidate_tuple] = cost
            else:
                cache[candidate_tuple] = None
                cost = None

        if problem.is_order_respected(candidate):
            cost = problem.route_cost(candidate)
            delta = cost - current_cost
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                current, current_cost = candidate, cost
                if cost < best_cost:
                    best, best_cost = candidate, cost
        temp *= cooling

    return list(best)


def _random_neighbor(route: Route, rng: random.Random) -> Route:
    n = len(route)
    i, j = sorted(rng.sample(range(n), 2))
    move = rng.randrange(3)

    if move == 0:  # 2-opt: odwrócenie fragmentu i..j
        return route[:i] + route[i:j + 1][::-1] + route[j + 1:]

    candidate = list(route)
    if move == 1:  # zamiana dwóch punktów
        candidate[i], candidate[j] = candidate[j], candidate[i]
    else:  # przeniesienie punktu z pozycji i na pozycję j
        candidate.insert(j, candidate.pop(i))
    return candidate
