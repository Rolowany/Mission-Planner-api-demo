from __future__ import annotations

from collections.abc import Sequence

from src.uav_mission.geo import LocalPoint

Route = list[int]


class RoutingProblem:
    def __init__(self, distances: Sequence[Sequence[float]], ordered: Sequence[int] = ()):
        size = len(distances)
        if size < 1:
            raise ValueError("macierz odległości musi zawierać przynajmniej miejsce startu")
        if any(len(row) != size for row in distances):
            raise ValueError("macierz odległości musi być kwadratowa")
        if any(d < 0 for row in distances for d in row):
            raise ValueError("odległości nie mogą być ujemne")

        n = size - 1
        if len(set(ordered)) != len(ordered):
            raise ValueError("punkty uporządkowane nie mogą się powtarzać")
        if any(not 0 <= i < n for i in ordered):
            raise ValueError("indeks punktu uporządkowanego poza zakresem")

        self._dist = [[float(d) for d in row] for row in distances]
        self.num_points = n
        self.home = n
        self.ordered: tuple[int, ...] = tuple(ordered)
        self._rank = {node: rank for rank, node in enumerate(self.ordered)}
        self.free: tuple[int, ...] = tuple(i for i in range(n) if i not in self._rank)

    @classmethod
    def from_points(cls, home: LocalPoint, points: Sequence[LocalPoint],
                    ordered: Sequence[int] = ()) -> RoutingProblem:
        """Tworzy problem z odległościami euklidesowymi."""
        nodes = [*points, home]
        distances = [[a.distance_to(b) for b in nodes] for a in nodes]
        return cls(distances, ordered)

    def distance(self, a: int, b: int) -> float:
        return self._dist[a][b]

    def route_cost(self, route: Sequence[int]) -> float:
        """Długość cyklu: start -> punkty z trasy -> start."""
        if not route:
            return 0.0
        d = self._dist
        total = d[self.home][route[0]] + d[route[-1]][self.home]
        for a, b in zip(route, route[1:]):
            total += d[a][b]
        return total

    def is_order_respected(self, route: Sequence[int]) -> bool:
        """Szybkie sprawdzenie kolejności; zakłada, że trasa jest permutacją."""
        expected = 0
        for node in route:
            rank = self._rank.get(node)
            if rank is not None:
                if rank != expected:
                    return False
                expected += 1
        return True

    def violations(self, route: Sequence[int]) -> list[str]:
        """Pełna walidacja trasy. Pusta lista oznacza trasę poprawną."""
        problems: list[str] = []
        seen: set[int] = set()
        for node in route:
            if not isinstance(node, int) or not 0 <= node < self.num_points:
                problems.append(f"nieznany punkt: {node!r}")
            elif node in seen:
                problems.append(f"punkt {node} występuje więcej niż raz")
            else:
                seen.add(node)

        missing = sorted(set(range(self.num_points)) - seen)
        if missing:
            problems.append(f"brakujące punkty: {missing}")

        ranks = [self._rank[n] for n in route if n in self._rank]
        if ranks != sorted(ranks):
            problems.append("naruszona kolejność punktów uporządkowanych")
        return problems

    def is_feasible(self, route: Sequence[int]) -> bool:
        return not self.violations(route)
