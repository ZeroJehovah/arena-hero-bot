"""Bounded, information-driven exploration for an intact travelling squad."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import hypot

from arena_hero import Position

from .geometry import adjacent_positions, manhattan

TILE_SIZE = 16
MAX_SEEN_TILES = 8192
SEARCH_RADIUS = 32
MAX_SEARCH_CELLS = 4096
MAX_CANDIDATES = 24
ROUTE_LOOKAHEAD = 6
RECHECK_TICKS = 6
STALLED_TICKS = 10
GOAL_COOLDOWN_TICKS = 40
MAX_COOLDOWNS = 8
MIN_INFORMATION_GAIN = 4


def _supercover(dx: int, dy: int) -> tuple[Position, ...]:
    """Cells crossed before the endpoint, including both sides of a corner."""

    nx, ny = abs(dx), abs(dy)
    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    x = y = ix = iy = 0
    cells: list[Position] = []
    while ix < nx or iy < ny:
        crossing = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if crossing == 0:
            cells.extend(((x + sx, y), (x, y + sy)))
            x, y, ix, iy = x + sx, y + sy, ix + 1, iy + 1
        elif crossing < 0:
            x, ix = x + sx, ix + 1
        else:
            y, iy = y + sy, iy + 1
        if (x, y) != (dx, dy):
            cells.append((x, y))
    return tuple(cells)


@lru_cache(maxsize=6)
def _vision_rays(radius: int) -> tuple[tuple[Position, tuple[Position, ...]], ...]:
    return tuple(
        ((dx, dy), _supercover(dx, dy))
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if abs(dx) + abs(dy) <= radius
    )


def visible_cells(
    origin: Position, radius: int, obstacles: set[Position]
) -> frozenset[Position]:
    """Match Manhattan vision and the server's integer supercover occlusion.

    An obstacle endpoint is visible. Units, Cores and resources never occlude
    vision. Predictions use known terrain; unknown rocks can shorten the view
    when the next authoritative snapshot arrives.
    """

    x, y = origin
    return frozenset(
        (x + dx, y + dy)
        for (dx, dy), ray in _vision_rays(radius)
        if all((x + rx, y + ry) not in obstacles for rx, ry in ray)
    )


@dataclass(slots=True)
class ExplorationMap:
    """Exact observed-cell bits in a bounded set of recently observed tiles."""

    tiles: dict[Position, tuple[int, int]] = field(default_factory=dict)
    last_tick: int = -1
    new_cells: int = 0

    def seen(self, cell: Position) -> bool:
        x, y = cell
        mask, _tick = self.tiles.get((x // TILE_SIZE, y // TILE_SIZE), (0, 0))
        return bool(mask & (1 << ((y % TILE_SIZE) * TILE_SIZE + x % TILE_SIZE)))

    def observe(
        self,
        observers: list[tuple[Position, int]],
        obstacles: set[Position],
        tick: int,
    ) -> None:
        """Only actual vision becomes known; intended routes never do."""

        cells: set[Position] = set()
        for position, radius in observers:
            cells.update(visible_cells(position, radius, obstacles))
        self.new_cells = sum(
            not self.seen(cell) and cell not in obstacles for cell in cells
        )
        self.mark(cells, tick)
        self.last_tick = tick

    def mark(self, cells: set[Position], tick: int) -> None:
        """Merge authoritative observations and evict the oldest remote tiles."""

        for x, y in cells:
            key = x // TILE_SIZE, y // TILE_SIZE
            mask, _last_seen = self.tiles.get(key, (0, tick))
            bit = 1 << ((y % TILE_SIZE) * TILE_SIZE + x % TILE_SIZE)
            self.tiles[key] = mask | bit, tick
        if len(self.tiles) > MAX_SEEN_TILES:
            oldest = sorted(self.tiles, key=lambda key: (self.tiles[key][1], key))
            for key in oldest[: len(self.tiles) - MAX_SEEN_TILES]:
                del self.tiles[key]


@dataclass(frozen=True, slots=True)
class ExplorationRoute:
    """A shared route, progress evidence and a bounded failed-approach cooldown."""

    path: tuple[Position, ...] = ()
    assigned_tick: int = 0
    checked_tick: int = 0
    progress_tick: int = 0
    remaining: int = 0
    expected_gain: float = 0.0
    cooldowns: tuple[tuple[Position, int], ...] = ()

    @property
    def goal(self) -> Position | None:
        return self.path[-1] if self.path else None

    @property
    def waypoint(self) -> Position | None:
        return (
            self.path[min(ROUTE_LOOKAHEAD, len(self.path) - 1)] if self.path else None
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    path: tuple[Position, ...]
    gain: float
    score: float


class ExpeditionNavigator:
    """Compare new visible ground per travel Tick, without a home-distance term.

    One bounded terrain search supplies all candidate paths. Only a small,
    directionally diverse shortlist receives a full route visibility score.
    Shared reservations discount duplicate scouting without blocking traffic.
    """

    def __init__(
        self,
        exploration: ExplorationMap,
        obstacles: set[Position],
        blocked: set[Position],
        radius: int,
    ) -> None:
        self.exploration = exploration
        self.obstacles = obstacles
        self.blocked = blocked
        self.radius = radius
        self._views: dict[Position, frozenset[Position]] = {}

    def view(self, position: Position) -> frozenset[Position]:
        if position not in self._views:
            self._views[position] = visible_cells(position, self.radius, self.obstacles)
        return self._views[position]

    def reservation(self, route: ExplorationRoute) -> set[Position]:
        """Reserve prospective views, never traversable cells or actual memory."""

        return {cell for position in route.path for cell in self.view(position)}

    def _value(self, cell: Position, reserved: set[Position]) -> float:
        if cell in self.obstacles or self.exploration.seen(cell):
            return 0.0
        return 0.15 if cell in reserved else 1.0

    def _evaluate(
        self, path: tuple[Position, ...], reserved: set[Position]
    ) -> _Candidate:
        covered = set(self.view(path[0]))
        gain = 0.0
        for step, position in enumerate(path[1:], 1):
            view = self.view(position)
            gain += 0.97**step * sum(
                self._value(cell, reserved) for cell in view - covered
            )
            covered.update(view)
        return _Candidate(path, gain, gain / (4 + len(path) - 1))

    def _search(
        self, origin: Position, cooldowns: tuple[tuple[Position, int], ...]
    ) -> dict[Position, Position | None]:
        parents: dict[Position, Position | None] = {origin: None}
        resting = {cell for cell, _until in cooldowns}
        queue = deque([origin])
        while queue and len(parents) < MAX_SEARCH_CELLS:
            cell = queue.popleft()
            for neighbor in adjacent_positions(cell):
                if (
                    neighbor in parents
                    or neighbor in self.blocked
                    or neighbor in resting
                    or manhattan(origin, neighbor) > SEARCH_RADIUS
                ):
                    continue
                parents[neighbor] = cell
                queue.append(neighbor)
                if len(parents) >= MAX_SEARCH_CELLS:
                    break
        return parents

    @staticmethod
    def _path(
        goal: Position, parents: dict[Position, Position | None]
    ) -> tuple[Position, ...]:
        result = [goal]
        current = parents[goal]
        while current is not None:
            result.append(current)
            current = parents[current]
        return tuple(reversed(result))

    @staticmethod
    def _alignment(origin: Position, goal: Position, bearing: Position) -> float:
        dx, dy = goal[0] - origin[0], goal[1] - origin[1]
        return (dx * bearing[0] + dy * bearing[1]) / max(1.0, hypot(dx, dy))

    def _candidates(
        self,
        origin: Position,
        parents: dict[Position, Position | None],
        reserved: set[Position],
        cooldowns: tuple[tuple[Position, int], ...],
        bearing: Position,
    ) -> list[_Candidate]:
        sectors: dict[Position, list[tuple[float, float, Position]]] = {}
        for cell in parents:
            dx, dy = cell[0] - origin[0], cell[1] - origin[1]
            if manhattan(cell, origin) < 4 or any(
                manhattan(cell, failed) <= 2 for failed, _until in cooldowns
            ):
                continue
            # A coarse sampling grid plus corridor ends avoids paying for
            # every cell while still finding bent, one-cell-wide passages.
            if (dx % 4 or dy % 4) and sum(
                neighbor in parents for neighbor in adjacent_positions(cell)
            ) > 1:
                continue
            gain = sum(self._value(point, reserved) for point in self.view(cell))
            if gain < MIN_INFORMATION_GAIN:
                continue
            path = self._path(cell, parents)
            sector = (
                (0 if dx == 0 else (1 if dx > 0 else -1)),
                (0 if dy == 0 else (1 if dy > 0 else -1)),
            )
            sectors.setdefault(sector, []).append(
                (
                    gain / (4 + len(path) - 1),
                    self._alignment(origin, cell, bearing),
                    cell,
                )
            )
        shortlist = [
            item
            for candidates in sectors.values()
            for item in sorted(candidates, reverse=True)[:3]
        ]
        return [
            self._evaluate(self._path(cell, parents), reserved)
            for _score, _alignment, cell in sorted(shortlist, reverse=True)[
                :MAX_CANDIDATES
            ]
        ]

    def _transit(
        self,
        origin: Position,
        parents: dict[Position, Position | None],
        reserved: set[Position],
        bearing: Position,
    ) -> tuple[Position, ...]:
        """Cross a known basin toward its nearest useful unexplored boundary."""

        exits = [cell for cell in parents if manhattan(origin, cell) >= SEARCH_RADIUS]
        if not exits:
            return ()
        probes: list[Position] = []
        for distance in (40, 64, 96, 128, 192, 256, 512, 1024):
            probes = [
                (origin[0] + dx * distance, origin[1] + dy * distance)
                for dx, dy in (
                    (1, 0),
                    (1, 1),
                    (0, 1),
                    (-1, 1),
                    (-1, 0),
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                )
                if sum(
                    self._value(cell, reserved)
                    for cell in self.view(
                        (origin[0] + dx * distance, origin[1] + dy * distance)
                    )
                )
                >= MIN_INFORMATION_GAIN
            ]
            if probes:
                break
        if not probes:
            return ()
        goal = min(
            exits,
            key=lambda cell: (
                min(manhattan(cell, probe) for probe in probes)
                + len(self._path(cell, parents)),
                -self._alignment(origin, cell, bearing),
                cell,
            ),
        )
        return self._path(goal, parents)

    def plan(
        self,
        origin: Position,
        bearing: Position,
        tick: int,
        previous: ExplorationRoute,
        reserved: set[Position],
    ) -> ExplorationRoute:
        cooldowns = tuple(item for item in previous.cooldowns if item[1] > tick)
        path = previous.path
        if origin in path:
            path = path[path.index(origin) :]
        remaining = len(path) - 1 + manhattan(origin, path[0]) if path else 0
        progress_tick = previous.progress_tick
        best_remaining = previous.remaining
        if path and remaining < best_remaining:
            best_remaining, progress_tick = remaining, tick
        stalled = bool(path) and tick - progress_tick >= STALLED_TICKS
        if stalled and len(path) > 1:
            # Rest the failed approach, not just its distant endpoint: merely
            # choosing a farther goal on the same blocked route changes nothing.
            cooldowns = (*cooldowns, (path[1], tick + GOAL_COOLDOWN_TICKS))[
                -MAX_COOLDOWNS:
            ]
        valid = (
            len(path) > 2
            and not stalled
            and manhattan(origin, path[0]) <= 2
            and all(cell not in self.blocked for cell in path[1:])
            and manhattan(origin, path[-1]) > 2
        )
        if valid and tick - previous.checked_tick < RECHECK_TICKS:
            return replace(
                previous,
                path=path,
                remaining=best_remaining,
                progress_tick=progress_tick,
                cooldowns=cooldowns,
            )
        parents = self._search(origin, cooldowns)
        candidates = self._candidates(origin, parents, reserved, cooldowns, bearing)
        best = max(
            candidates,
            key=lambda candidate: (
                candidate.score,
                self._alignment(origin, candidate.path[-1], bearing),
                candidate.gain,
                candidate.path[-1],
            ),
            default=None,
        )
        if valid and path[-1] in parents:
            current = self._evaluate(self._path(path[-1], parents), reserved)
            # A useful commitment survives small changes in the map or which
            # teammate currently lies nearest the squad centroid.
            if current.gain >= MIN_INFORMATION_GAIN and (
                best is None or best.score <= current.score * 1.3
            ):
                return replace(
                    previous,
                    path=current.path,
                    checked_tick=tick,
                    remaining=best_remaining,
                    progress_tick=progress_tick,
                    expected_gain=current.gain,
                    cooldowns=cooldowns,
                )
        if best is not None and best.gain >= MIN_INFORMATION_GAIN:
            path, gain = best.path, best.gain
        else:
            path, gain = self._transit(origin, parents, reserved, bearing), 0.0
        return ExplorationRoute(
            path=path,
            assigned_tick=tick,
            checked_tick=tick,
            progress_tick=tick,
            remaining=max(0, len(path) - 1),
            expected_gain=gain,
            cooldowns=cooldowns,
        )
