"""Planning and control files for long-distance Core migrations."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena_hero import Position

from .geometry import manhattan, next_step
from .memory import WorldMemory
from .strategy import (
    SYMMETRIC_HOME_CLEAR_RADIUS,
    SYMMETRIC_RANGER_OFFSETS,
    SYMMETRIC_RESOURCE_BLOCK_SIZE,
    SYMMETRIC_VANGUARD_OFFSETS,
    _resource_block_center_cells,
)

MIGRATION_REQUEST_VERSION = 1
_DEFENSE_OFFSETS = (*SYMMETRIC_VANGUARD_OFFSETS, *SYMMETRIC_RANGER_OFFSETS)


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """One deterministic next-block recommendation toward a quota goal."""

    current_block: Position
    current_ring: int
    current_quota: int
    target_block: Position
    target_ring: int
    target_quota: int
    goal_ring: int
    goal_quota: int
    target: Position
    estimated_cells: int
    block_rocks: int
    clear_cells: int
    blocked_defense_posts: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_block": list(self.current_block),
            "current_ring": self.current_ring,
            "current_quota": self.current_quota,
            "target_block": list(self.target_block),
            "target_ring": self.target_ring,
            "target_quota": self.target_quota,
            "goal_ring": self.goal_ring,
            "goal_quota": self.goal_quota,
            "target": list(self.target),
            "estimated_cells": self.estimated_cells,
            "block_rocks": self.block_rocks,
            "clear_cells": self.clear_cells,
            "blocked_defense_posts": self.blocked_defense_posts,
        }


@dataclass(frozen=True, slots=True)
class MigrationRequest:
    """A durable operator request consumed by the running bot."""

    target: Position
    requested_at: str
    note: str = ""
    version: int = MIGRATION_REQUEST_VERSION


def _axis(coordinate: int) -> int:
    return coordinate if coordinate >= 0 else -coordinate - 1


def resource_block_index(position: Position) -> Position:
    """Return the signed 32x32 block containing ``position``."""

    size = SYMMETRIC_RESOURCE_BLOCK_SIZE
    return position[0] // size, position[1] // size


def resource_block_ring(block: Position) -> int:
    """Return the symmetric block-ring distance from the world center."""

    return _axis(block[0]) + _axis(block[1])


def resource_block_quota(block: Position) -> int:
    """Return the natural-resource quota for one 32x32 block."""

    ring = resource_block_ring(block)
    return _resource_quota_at_ring(ring)


def _resource_quota_at_ring(ring: int) -> int:
    return max(2, (16 * 8) // (8 + ring))


def next_resource_tier_ring(current_ring: int) -> int | None:
    """Return the highest inner ring whose quota exceeds the current one."""

    current_quota = _resource_quota_at_ring(current_ring)
    higher = [
        ring
        for ring in range(current_ring)
        if _resource_quota_at_ring(ring) > current_quota
    ]
    return max(higher) if higher else None


def resource_quota_outer_ring(target_quota: int) -> int | None:
    """Return the outermost ring that still provides ``target_quota``.

    Moving to that ring crosses the fewest outer cells while guaranteeing the
    requested quota.  The tier boundaries are intentionally the same ones the
    game uses: ring 34 still has quota 3, ring 24 has quota 4.
    """

    if target_quota <= 2:
        return None
    rings = [
        ring for ring in range(128) if _resource_quota_at_ring(ring) >= target_quota
    ]
    return max(rings) if rings else None


def campaign_stop_ring(target_quota: int) -> int | None:
    """Return the outermost campaign ring allowed at one formation tier.

    Tiers 2 and 3 approach the next quota boundary without crossing it:
    quota 2 stops at ring 35 beside quota 3, and quota 3 stops at ring 25
    beside quota 4.  The max formation may enter quota 4 and travels all the
    way to the outer quota 8 ring.
    """

    if target_quota <= 2:
        boundary = resource_quota_outer_ring(3)
        return None if boundary is None else boundary + 1
    if target_quota < 8:
        boundary = resource_quota_outer_ring(target_quota + 1)
        return None if boundary is None else boundary + 1
    return resource_quota_outer_ring(target_quota)


def cardinal_block_neighbors(block: Position) -> tuple[Position, ...]:
    """Return the four blocks sharing an edge with ``block``."""

    x, y = block
    return (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)


def blocks_are_cardinally_adjacent(left: Position, right: Position) -> bool:
    """Return whether two block indices differ by one cardinal step."""

    return abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1


def _block_cells(block: Position) -> tuple[Position, ...]:
    size = SYMMETRIC_RESOURCE_BLOCK_SIZE
    min_x = block[0] * size
    min_y = block[1] * size
    return tuple(
        (x, y) for x in range(min_x, min_x + size) for y in range(min_y, min_y + size)
    )


def plan_next_resource_tier(
    memory: WorldMemory,
    *,
    current_position: Position | None = None,
) -> MigrationPlan | None:
    """Pick one clean adjacent block center toward the next quota tier."""

    origin = (
        current_position
        or memory.core_migration_last_position
        or memory.core_home_position
    )
    if origin is None:
        return None
    current_block = resource_block_index(origin)
    current_ring = resource_block_ring(current_block)
    target_ring = next_resource_tier_ring(current_ring)
    if target_ring is None:
        return None
    target_quota = _resource_quota_at_ring(target_ring)
    return _plan_adjacent_block(
        memory,
        origin=origin,
        current_block=current_block,
        current_ring=current_ring,
        goal_ring=target_ring,
        goal_quota=target_quota,
    )


def plan_resource_quota(
    memory: WorldMemory,
    target_quota: int,
    *,
    current_position: Position | None = None,
) -> MigrationPlan | None:
    """Pick one clean adjacent block center toward ``target_quota``.

    The quota boundary remains the durable campaign goal, while each published
    ``core_home_position`` is only the next 32-cell block hop.  Repeating these
    hops prevents a respawned or restarted Core from inheriting a distant final
    coordinate as if it were the next local waypoint.
    """

    origin = (
        current_position
        or memory.core_migration_last_position
        or memory.core_home_position
    )
    if origin is None:
        return None
    current_block = resource_block_index(origin)
    current_ring = resource_block_ring(current_block)
    current_quota = _resource_quota_at_ring(current_ring)
    if target_quota <= current_quota:
        return None
    target_ring = resource_quota_outer_ring(target_quota)
    if target_ring is None or target_ring >= current_ring:
        return None
    return _plan_adjacent_block(
        memory,
        origin=origin,
        current_block=current_block,
        current_ring=current_ring,
        goal_ring=target_ring,
        goal_quota=target_quota,
    )


def plan_resource_cap(
    memory: WorldMemory,
    *,
    target_quota: int,
    stop_ring: int,
    current_position: Position | None = None,
    excluded_blocks: frozenset[Position] = frozenset(),
    allow_lateral: bool = False,
) -> MigrationPlan | None:
    """Plan one hop without crossing a formation tier's quota ceiling.

    The planner first uses an inward cardinal neighbor.  Lateral same-ring
    neighbors are considered only when explicitly allowed and no inward
    candidate remains, which gives blocked scouting targets a bounded detour
    without ever increasing the block ring.
    """

    origin = (
        current_position
        or memory.core_migration_last_position
        or memory.core_home_position
    )
    if origin is None:
        return None
    current_block = resource_block_index(origin)
    current_ring = resource_block_ring(current_block)
    current_quota = resource_block_quota(current_block)
    if current_quota > target_quota or current_ring <= stop_ring:
        return None

    def candidates(
        *, inward: bool
    ) -> list[tuple[int, int, int, int, int, int, Position, Position]]:
        def has_reachable_inward_exit(block: Position, target: Position) -> bool:
            """Whether a same-ring detour can still make inward progress."""

            for next_block in cardinal_block_neighbors(block):
                if (
                    resource_block_ring(next_block) >= current_ring
                    or next_block in excluded_blocks
                    or resource_block_quota(next_block) > target_quota
                ):
                    continue
                for next_target in _resource_block_center_cells(
                    (
                        next_block[0] * SYMMETRIC_RESOURCE_BLOCK_SIZE,
                        next_block[1] * SYMMETRIC_RESOURCE_BLOCK_SIZE,
                    )
                ):
                    if next_target in memory.obstacles:
                        continue
                    if (
                        next_step(
                            target,
                            next_target,
                            blocked=memory.obstacles,
                            require_path=True,
                        )
                        is not None
                    ):
                        return True
            return False

        found: list[tuple[int, int, int, int, int, int, Position, Position]] = []
        for block in cardinal_block_neighbors(current_block):
            ring = resource_block_ring(block)
            if block in excluded_blocks or resource_block_quota(block) > target_quota:
                continue
            if inward and ring >= current_ring:
                continue
            if not inward and ring != current_ring:
                continue
            block_rocks = sum(cell in memory.obstacles for cell in _block_cells(block))
            for target in _resource_block_center_cells(
                (
                    block[0] * SYMMETRIC_RESOURCE_BLOCK_SIZE,
                    block[1] * SYMMETRIC_RESOURCE_BLOCK_SIZE,
                )
            ):
                if target in memory.obstacles:
                    continue
                clear_cells = sum(
                    (target[0] + dx, target[1] + dy) not in memory.obstacles
                    for dx in range(
                        -SYMMETRIC_HOME_CLEAR_RADIUS,
                        SYMMETRIC_HOME_CLEAR_RADIUS + 1,
                    )
                    for dy in range(
                        -SYMMETRIC_HOME_CLEAR_RADIUS,
                        SYMMETRIC_HOME_CLEAR_RADIUS + 1,
                    )
                    if abs(dx) + abs(dy) <= SYMMETRIC_HOME_CLEAR_RADIUS
                )
                blocked_posts = sum(
                    (target[0] + dx, target[1] + dy) in memory.obstacles
                    for dx, dy in _DEFENSE_OFFSETS
                )
                if not inward and not has_reachable_inward_exit(block, target):
                    continue
                found.append(
                    (
                        ring,
                        int(target in memory.resource_cells),
                        blocked_posts,
                        block_rocks,
                        -clear_cells,
                        manhattan(origin, target),
                        block,
                        target,
                    )
                )
        return found

    pools = [sorted(candidates(inward=True))]
    if allow_lateral:
        pools.append(sorted(candidates(inward=False)))
    chosen = next(
        (
            candidate
            for pool in pools
            for candidate in pool
            if next_step(
                origin,
                candidate[-1],
                blocked=memory.obstacles,
                require_path=True,
            )
            is not None
        ),
        None,
    )
    if chosen is None:
        return None
    (
        _ring,
        _known_resource,
        blocked_posts,
        block_rocks,
        negative_clear,
        distance,
        block,
        target,
    ) = chosen
    return MigrationPlan(
        current_block=current_block,
        current_ring=current_ring,
        current_quota=current_quota,
        target_block=block,
        target_ring=resource_block_ring(block),
        target_quota=resource_block_quota(block),
        goal_ring=stop_ring,
        goal_quota=target_quota,
        target=target,
        estimated_cells=distance,
        block_rocks=block_rocks,
        clear_cells=-negative_clear,
        blocked_defense_posts=blocked_posts,
    )


def _plan_adjacent_block(
    memory: WorldMemory,
    *,
    origin: Position,
    current_block: Position,
    current_ring: int,
    goal_ring: int,
    goal_quota: int,
) -> MigrationPlan | None:
    """Search the center cells of the inward cardinal neighboring blocks."""

    candidates: list[tuple[int, int, int, int, int, int, Position, Position]] = []
    for block in cardinal_block_neighbors(current_block):
        ring = resource_block_ring(block)
        if ring >= current_ring:
            continue
        block_rocks = sum(cell in memory.obstacles for cell in _block_cells(block))
        for target in _resource_block_center_cells(
            (
                block[0] * SYMMETRIC_RESOURCE_BLOCK_SIZE,
                block[1] * SYMMETRIC_RESOURCE_BLOCK_SIZE,
            )
        ):
            if target in memory.obstacles:
                continue
            clear_cells = sum(
                (target[0] + dx, target[1] + dy) not in memory.obstacles
                for dx in range(
                    -SYMMETRIC_HOME_CLEAR_RADIUS, SYMMETRIC_HOME_CLEAR_RADIUS + 1
                )
                for dy in range(
                    -SYMMETRIC_HOME_CLEAR_RADIUS, SYMMETRIC_HOME_CLEAR_RADIUS + 1
                )
                if abs(dx) + abs(dy) <= SYMMETRIC_HOME_CLEAR_RADIUS
            )
            blocked_posts = sum(
                (target[0] + dx, target[1] + dy) in memory.obstacles
                for dx, dy in _DEFENSE_OFFSETS
            )
            candidates.append(
                (
                    ring,
                    int(target in memory.resource_cells),
                    blocked_posts,
                    block_rocks,
                    -clear_cells,
                    manhattan(origin, target),
                    block,
                    target,
                )
            )
    chosen = next(
        (
            candidate
            for candidate in sorted(candidates)
            if next_step(
                origin,
                candidate[-1],
                blocked=memory.obstacles,
                require_path=True,
            )
            is not None
        ),
        None,
    )
    if chosen is None:
        return None
    (
        _ring,
        _known_resource,
        blocked_posts,
        block_rocks,
        negative_clear,
        distance,
        block,
        target,
    ) = chosen
    return MigrationPlan(
        current_block=current_block,
        current_ring=current_ring,
        current_quota=resource_block_quota(current_block),
        target_block=block,
        target_ring=resource_block_ring(block),
        target_quota=resource_block_quota(block),
        goal_ring=goal_ring,
        goal_quota=goal_quota,
        target=target,
        estimated_cells=distance,
        block_rocks=block_rocks,
        clear_cells=-negative_clear,
        blocked_defense_posts=blocked_posts,
    )


def load_migration_request(path: Path) -> MigrationRequest | None:
    """Load and validate an optional hot migration request."""

    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("migration request must be a JSON object")
    version = int(raw.get("version", 0))
    if version != MIGRATION_REQUEST_VERSION:
        raise ValueError(f"unsupported migration request version: {version}")
    target_raw = raw.get("target")
    if not isinstance(target_raw, list) or len(target_raw) != 2:
        raise ValueError("migration request target must contain [x, y]")
    target = int(target_raw[0]), int(target_raw[1])
    if target not in _resource_block_center_cells(target):
        raise ValueError("migration target must be one of a chunk's center cells")
    return MigrationRequest(
        target=target,
        requested_at=str(raw.get("requested_at", "")),
        note=str(raw.get("note", "")),
        version=version,
    )


def write_migration_request(
    path: Path,
    target: Position,
    *,
    note: str = "",
) -> MigrationRequest:
    """Atomically publish a validated migration request."""

    if target not in _resource_block_center_cells(target):
        raise ValueError("migration target must be one of a chunk's center cells")
    request = MigrationRequest(
        target=target,
        requested_at=datetime.now(UTC).isoformat(),
        note=note,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": request.version,
        "target": list(request.target),
        "requested_at": request.requested_at,
        "note": request.note,
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return request
