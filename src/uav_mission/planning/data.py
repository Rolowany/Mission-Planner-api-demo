from dataclasses import dataclass

from src.uav_mission.planning.problem import Route

@dataclass(frozen=True)
class SolverAttempt:
    name: str
    status: str
    cost: float | None
    duration_s: float
    message: str = ""


@dataclass(frozen=True)
class PlanResult:
    route: Route
    cost: float
    solver: str
    attempts: tuple[SolverAttempt, ...]

@dataclass(frozen=True)
class AnnealingConfig:
    iterations: int = 20_000
    seed: int | None = 0
    time_limit_s: float | None = None
    start_temp_factor: float = 0.5
    end_temp_factor: float = 0.001