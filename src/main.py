import argparse
import logging
import math
import sys

from src.uav_mission.avoidance.avoider import ObstacleAvoider
from src.uav_mission.control.mission_controller import (
    ControllerConfig, MissionController, Waypoint, connect, wait_for_gps)
from src.uav_mission.control.start_position import check_start_position
from src.uav_mission.geo import GeoPoint
from src.uav_mission.mission_file import MissionSpec, check_distances, load_mission
from src.uav_mission.planning import (
    AnnealingConfig, PlanResult, RoutingProblem, default_improvers, plan_route,
    problem_from_mission)
from src.uav_mission.sensors.lidar import LidarConfig, SimulatedLidar
from src.uav_mission.simulation.world import load_world
from src.uav_mission.utils.errors import MissionError, MissionFileError

log = logging.getLogger("mission")


def _print_result(spec: MissionSpec, problem: RoutingProblem, result: PlanResult) -> None:
    points = spec.all_points
    n_free = len(spec.free_points)

    print(f"Misja: {spec.name}")
    print(f"Trasa: {len(result.route)} punktów, długość {result.cost:.1f} m "
          f"(najlepszy wynik: {result.solver})\n")
    print("    0. START")
    previous = problem.home
    for step, index in enumerate(result.route, start=1):
        kind = f"uporządkowany #{index - n_free + 1}" if index >= n_free else "dowolny"
        leg = problem.distance(previous, index)
        print(f"{step:5d}. {points[index].id:<10} {kind:<18} +{leg:7.1f} m")
        previous = index
    print(f"{'':7}START{'':24} +{problem.distance(previous, problem.home):7.1f} m\n")

    print("Metody:")
    for attempt in result.attempts:
        cost = f"{attempt.cost:.1f} m" if attempt.cost is not None else "-"
        print(f"  {attempt.name:<22} {attempt.status:<11} {cost:>10} "
              f"{attempt.duration_s * 1000:9.1f} ms  {attempt.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Misja przez punkty z omijaniem przeszkód")
    parser.add_argument("mission", help="plik misji (JSON)")
    parser.add_argument("--connect", default="tcp:127.0.0.1:5762",
                        help="adres połączenia MAVLink (domyślnie %(default)s)")
    parser.add_argument("--scenario",
                        help="plik scenariusza z przeszkodami; włącza symulowany LiDAR "
                             "i omijanie (bez niego dron leci bez omijania)")
    parser.add_argument("--lidar-fail-after", type=int, default=None,
                        help="symulowana awaria LiDAR-u po tylu skanach (scenariusz awarii)")
    parser.add_argument("--iterations", type=int, default=20_000,
                        help="liczba iteracji wyżarzania (domyślnie %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="ziarno losowania")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="limit czasu wyżarzania w sekundach")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    try:
        spec = load_mission(args.mission)
        master = connect(args.connect)
        lat, lon, _ = wait_for_gps(master)
        home = GeoPoint(lat, lon)
        if spec.home is not None:
            check_start_position((lat, lon), (spec.home.lat, spec.home.lon))
        check_distances(spec, home)
        world = load_world(args.scenario, home) if args.scenario else None
    except (MissionError, MissionFileError) as exc:
        log.error("%s", exc)
        return 1

    problem = problem_from_mission(spec, home)
    config = AnnealingConfig(iterations=args.iterations, seed=args.seed,
                             time_limit_s=args.time_limit)

    result = plan_route(problem, default_improvers(config))
    _print_result(spec, problem, result)

    points = spec.all_points
    n_free = len(spec.free_points)
    waypoints = [Waypoint(points[i].id, points[i].position, ordered=i >= n_free)
                 for i in result.route]

    lidar = avoider = None
    if world is not None:
        lidar = SimulatedLidar(world, LidarConfig(fail_after_scans=args.lidar_fail_after))
        avoider = ObstacleAvoider()
        log.info("Symulowany LiDAR włączony, przeszkód w scenariuszu: %d", len(world.obstacles))
    else:
        log.warning("Brak scenariusza - lot bez omijania przeszkód")

    controller = MissionController(master, waypoints, home,
                                   ControllerConfig(altitude_m=spec.altitude_m),
                                   lidar, avoider)

    exit_code = 0
    try:
        controller.run_mission()
        log.info("Misja zakończona pomyślnie")
    except MissionError as exc:
        log.error("Błąd misji: %s", exc)
        controller.emergency_rtl()
        exit_code = 1
    except KeyboardInterrupt:
        log.warning("Przerwano przez użytkownika")
        controller.emergency_rtl()
        exit_code = 1

    log.info("Objazdy przeszkód: %d", controller.detours)
    if lidar is not None and lidar.min_clearance_m != math.inf:
        log.info("Najmniejsza odległość od przeszkody (prawda symulacji): %.1f m",
                 lidar.min_clearance_m)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
