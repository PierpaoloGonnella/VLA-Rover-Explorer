from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field


Cell = tuple[int, int]


@dataclass
class ObstacleGrid:
    """Short-lived ultrasonic obstacle hits represented in image space."""

    frame_shape: tuple[int, ...]
    cols: int = 12
    rows: int = 8
    ttl_cycles: int = 40
    hits: dict[Cell, int] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.frame_shape[1]

    @property
    def height(self) -> int:
        return self.frame_shape[0]

    def cell_for(self, point: tuple[float, float]) -> Cell:
        x, y = point
        col = min(self.cols - 1, max(0, int(x * self.cols / self.width)))
        row = min(self.rows - 1, max(0, int(y * self.rows / self.height)))
        return col, row

    def centre(self, cell: Cell) -> tuple[float, float]:
        col, row = cell
        return ((col + .5) * self.width / self.cols, (row + .5) * self.height / self.rows)

    def _ray_cells(
        self,
        origin: tuple[float, float],
        angle: float,
        distance_px: float,
    ) -> list[Cell]:
        step = max(2.0, min(self.width / self.cols, self.height / self.rows) / 3)
        count = max(1, math.ceil(max(0.0, distance_px) / step))
        cells: list[Cell] = []
        for index in range(1, count + 1):
            distance = distance_px * index / count
            point = (
                origin[0] + math.cos(angle) * distance,
                origin[1] + math.sin(angle) * distance,
            )
            if not (0 <= point[0] < self.width and 0 <= point[1] < self.height):
                break
            cell = self.cell_for(point)
            if not cells or cells[-1] != cell:
                cells.append(cell)
        return cells

    def observe_ray(
        self,
        origin: tuple[float, float],
        angle: float,
        distance_px: float,
        *,
        hit: bool,
        cycle: int,
    ) -> Cell | None:
        cells = self._ray_cells(origin, angle, distance_px)
        clear_cells = cells[:-1] if hit else cells
        for cell in clear_cells:
            self.hits.pop(cell, None)
        hit_cell = cells[-1] if hit and cells else None
        if hit_cell is not None:
            self.hits[hit_cell] = cycle
        self.prune(cycle)
        return hit_cell

    def prune(self, cycle: int) -> None:
        oldest = cycle - max(1, self.ttl_cycles)
        self.hits = {cell: seen for cell, seen in self.hits.items() if seen >= oldest}

    def occupied(self, inflation_cells: int = 0) -> set[Cell]:
        result: set[Cell] = set()
        radius = max(0, inflation_cells)
        for col, row in self.hits:
            for dc in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    if dc * dc + dr * dr > radius * radius:
                        continue
                    candidate = col + dc, row + dr
                    if 0 <= candidate[0] < self.cols and 0 <= candidate[1] < self.rows:
                        result.add(candidate)
        return result

    def astar(
        self,
        start: Cell,
        goal: Cell,
        blocked: set[Cell],
    ) -> list[Cell] | None:
        """Find a conservative four-neighbour route without corner cutting."""
        blocked = set(blocked)
        blocked.discard(start)
        if goal in blocked:
            return None
        frontier: list[tuple[int, int, Cell]] = [(0, 0, start)]
        serial = 0
        came_from: dict[Cell, Cell | None] = {start: None}
        cost: dict[Cell, int] = {start: 0}
        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                path: list[Cell] = []
                node: Cell | None = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                return list(reversed(path))
            col, row = current
            for neighbour in ((col + 1, row), (col - 1, row), (col, row + 1), (col, row - 1)):
                if not (0 <= neighbour[0] < self.cols and 0 <= neighbour[1] < self.rows):
                    continue
                if neighbour in blocked:
                    continue
                new_cost = cost[current] + 1
                if neighbour in cost and new_cost >= cost[neighbour]:
                    continue
                cost[neighbour] = new_cost
                came_from[neighbour] = current
                heuristic = abs(goal[0] - neighbour[0]) + abs(goal[1] - neighbour[1])
                serial += 1
                heapq.heappush(frontier, (new_cost + heuristic, serial, neighbour))
        return None
