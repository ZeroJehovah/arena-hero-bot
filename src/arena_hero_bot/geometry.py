"""Deterministic geometry and local pathfinding helpers."""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from typing import Final

from arena_hero import Direction, Position

DIRECTIONS: Final[tuple[Direction, ...]] = (
    Direction.UP,
    Direction.RIGHT,
    Direction.DOWN,
    Direction.LEFT,
)


def add(position: Position, direction: Direction) -> Position:
    """Return the cell one step from ``position`` in ``direction``."""

    dx, dy = direction.delta
    return position[0] + dx, position[1] + dy


def manhattan(left: Position, right: Position) -> int:
    """Return Manhattan distance between two cells."""

    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def direction_between(origin: Position, destination: Position) -> Direction | None:
    """Return the cardinal direction for an adjacent destination."""

    delta = destination[0] - origin[0], destination[1] - origin[1]
    for direction in DIRECTIONS:
        if direction.delta == delta:
            return direction
    return None


def line_of_fire(
    origin: Position,
    target: Position,
    obstacles: set[Position] | frozenset[Position],
) -> bool:
    """Return whether a Ranger cell shot is legal under v0.14 geometry."""

    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    distance = max(abs(dx), abs(dy))
    if distance not in {1, 2, 3}:
        return False
    if dx != 0 and dy != 0 and abs(dx) != abs(dy):
        return False
    if dx == 0 and dy == 0:
        return False

    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    return all(
        (origin[0] + step_x * step, origin[1] + step_y * step) not in obstacles
        for step in range(1, distance)
    )


def firing_positions(target: Position) -> tuple[Position, ...]:
    """Return every cell that can geometrically fire at ``target``."""

    offsets = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
    )
    return tuple(
        (target[0] + dx * distance, target[1] + dy * distance)
        for distance in range(1, 4)
        for dx, dy in offsets
    )


def adjacent_positions(position: Position) -> tuple[Position, ...]:
    """Return cardinal neighbors in stable order."""

    return tuple(add(position, direction) for direction in DIRECTIONS)


def next_step(
    origin: Position,
    goal: Position,
    *,
    blocked: set[Position],
    recent: tuple[Position, ...] = (),
    direction_offset: int = 0,
    max_expansions: int = 4096,
    allow_goal: bool = False,
    require_path: bool = False,
) -> Direction | None:
    """Find one deterministic A* step with a penalty for recent cells.

    When no route to ``goal`` exists the greedy fallback below still returns
    the neighbour closest to it.  That is the right answer while the blockage
    is transient, because units shuffle and one step of pressure resolves it.
    It is the wrong answer when the goal is walled off for good: the caller
    cannot tell progress from an orbit around the wall, so it keeps asking and
    the unit circles forever.  ``require_path=True`` returns ``None`` in that
    case so the caller can choose a reachable goal instead.
    """

    if origin == goal:
        return None

    blocked = blocked - {origin}
    if allow_goal:
        blocked.discard(goal)
    recent_penalty = {position: index + 1 for index, position in enumerate(recent)}
    rotated = DIRECTIONS[direction_offset % 4 :] + DIRECTIONS[: direction_offset % 4]
    sequence = count()
    frontier: list[tuple[int, int, int, Position]] = []
    heappush(frontier, (manhattan(origin, goal), manhattan(origin, goal), 0, origin))
    came_from: dict[Position, Position] = {}
    cost_so_far: dict[Position, int] = {origin: 0}
    expansions = 0

    while frontier and expansions < max_expansions:
        _, _, _, current = heappop(frontier)
        expansions += 1
        if current == goal:
            break

        for direction in rotated:
            neighbor = add(current, direction)
            if neighbor in blocked:
                continue
            step_cost = 1 + recent_penalty.get(neighbor, 0) * 3
            new_cost = cost_so_far[current] + step_cost
            if new_cost >= cost_so_far.get(neighbor, 2**63 - 1):
                continue
            cost_so_far[neighbor] = new_cost
            came_from[neighbor] = current
            heuristic = manhattan(neighbor, goal)
            heappush(
                frontier,
                (new_cost + heuristic, heuristic, next(sequence), neighbor),
            )

    if goal not in came_from:
        if require_path:
            return None
        candidates = [
            add(origin, direction)
            for direction in rotated
            if add(origin, direction) not in blocked
        ]
        if not candidates:
            return None
        first = min(
            candidates,
            key=lambda cell: (
                manhattan(cell, goal) + recent_penalty.get(cell, 0) * 3,
                manhattan(cell, goal),
                cell,
            ),
        )
        return direction_between(origin, first)

    current = goal
    while came_from[current] != origin:
        current = came_from[current]
    return direction_between(origin, current)
