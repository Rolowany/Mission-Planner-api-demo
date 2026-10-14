from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.uav_mission.geo import GeoPoint, LocalPoint, to_local
from src.uav_mission.mission_file import MissionSpec
from src.uav_mission.planning.annealing import AnnealingConfig, anneal
from src.uav_mission.planning.greedy import nearest_neighbor
from src.uav_mission.planning.problem import Route, RoutingProblem
from src.uav_mission.planning.data import SolverAttempt, PlanResult

log = logging.getLogger(__name__)

Improver = Callable[[RoutingProblem, Route], Route]

def default_improvers(config: AnnealingConfig = AnnealingConfig()) -> list[tuple[str, Improver]]:
    return [("simulated_annealing", lambda problem, route: anneal(problem, route, config))]


def plan_route(problem: RoutingProblem,
               improvers: Sequence[tuple[str, Improver]] | None = None) -> PlanResult:
    if improvers is None:
        improvers = default_improvers()

    start = time.monotonic()
    best_route = nearest_neighbor(problem)
    best_cost = problem.route_cost(best_route)
    best_solver = "greedy"
    attempts = [SolverAttempt("greedy", "ok", best_cost, time.monotonic() - start)]

    for name, improve in improvers:
        start = time.monotonic()
        try:
            candidate = improve(problem, list(best_route))
        except Exception as exc:  # noqa: BLE001 - awaria optymalizatora nie może zatrzymać misji
            log.warning("Metoda %s zakończyła się błędem: %s", name, exc)
            attempts.append(SolverAttempt(name, "error", None,
                                          time.monotonic() - start, str(exc)))
            continue
        duration = time.monotonic() - start

        violations = problem.violations(candidate)
        if violations:
            log.warning("Metoda %s zwróciła niepoprawną trasę: %s", name, violations)
            attempts.append(SolverAttempt(name, "infeasible", None, duration,
                                          "; ".join(violations)))
            continue

        cost = problem.route_cost(candidate)
        attempts.append(SolverAttempt(name, "ok", cost, duration))
        if cost < best_cost:
            best_route, best_cost, best_solver = list(candidate), cost, name

    return PlanResult(best_route, best_cost, best_solver, tuple(attempts))


def problem_from_mission(spec: MissionSpec, home: GeoPoint) -> RoutingProblem:
    """Buduje problem dla misji. Indeksy punktów odpowiadają `spec.all_points`
    (najpierw punkty dowolne, potem uporządkowane)."""
    local = [to_local(home, point.position) for point in spec.all_points]
    first_ordered = len(spec.free_points)
    ordered = range(first_ordered, len(local))
    return RoutingProblem.from_points(LocalPoint(0.0, 0.0), local, ordered)
