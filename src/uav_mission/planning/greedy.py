from __future__ import annotations

from src.uav_mission.planning.problem import Route, RoutingProblem


def nearest_neighbor(problem: RoutingProblem) -> Route:
    remaining_free = set(problem.free)
    next_ordered = 0
    current = problem.home
    route: Route = []

    while remaining_free or next_ordered < len(problem.ordered):
        candidates = list(remaining_free)
        if next_ordered < len(problem.ordered):
            candidates.append(problem.ordered[next_ordered])

        nxt = min(candidates, key=lambda node: (problem.distance(current, node), node))

        if nxt in remaining_free:
            remaining_free.remove(nxt)
        else:
            next_ordered += 1
        route.append(nxt)
        current = nxt

    return route
