from __future__ import annotations

import time
from dataclasses import dataclass, field

from .localize import RoverPose


@dataclass
class CoverageTracker:
    frame_shape: tuple[int, ...]
    cols: int = 6
    rows: int = 4
    visited: set[tuple[int, int]] = field(default_factory=set)
    excluded: set[tuple[int, int]] = field(default_factory=set)
    localized_updates: int = 0
    revisits: int = 0
    lost_frames: int = 0
    boundary_guard_vetoes: int = 0
    ultrasonic_guard_vetoes: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def cell_for(self, centre: tuple[float, float]) -> tuple[int, int]:
        height, width = self.frame_shape[:2]
        col = min(self.cols - 1, max(0, int(centre[0] * self.cols / width)))
        row = min(self.rows - 1, max(0, int(centre[1] * self.rows / height)))
        return col, row

    def update(self, pose: RoverPose | None) -> None:
        if pose is None:
            self.lost_frames += 1
            return
        self.localized_updates += 1
        cell = self.cell_for(pose.centre)
        if cell in self.visited:
            self.revisits += 1
        self.visited.add(cell)

    def exclude(self, cell: tuple[int, int]) -> None:
        """Remove a confirmed obstacle cell from the reachable coverage goal."""
        self.excluded.add(cell)

    def add_vetoes(self, count: int) -> None:
        self.boundary_guard_vetoes += max(0, count)

    def add_ultrasonic_vetoes(self, count: int) -> None:
        self.ultrasonic_guard_vetoes += max(0, count)

    @property
    def fraction(self) -> float:
        reachable = max(1, self.cols * self.rows - len(self.excluded))
        visited_reachable = len(self.visited - self.excluded)
        return min(1.0, visited_reachable / reachable)

    def report(self) -> dict[str, float | int]:
        return {
            "fraction_visited": self.fraction,
            "cells_visited": len(self.visited),
            "total_cells": self.cols * self.rows,
            "reachable_cells": self.cols * self.rows - len(self.excluded),
            "blocked_cells": len(self.excluded),
            "lost_frames": self.lost_frames,
            "boundary_guard_vetoes": self.boundary_guard_vetoes,
            "ultrasonic_guard_vetoes": self.ultrasonic_guard_vetoes,
            "revisit_ratio": self.revisits / self.localized_updates if self.localized_updates else 0.0,
            "wall_clock_seconds": time.monotonic() - self.started_at,
        }
