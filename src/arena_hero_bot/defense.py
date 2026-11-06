"""Bounded formation and damage calculations using the server combat rules."""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from uuid import UUID

from arena_hero import Direction, Position, Ranger, UnitType, UnitView

from .geometry import DIRECTIONS, add, adjacent_positions, line_of_fire, manhattan

FRONT_DEPTH = 4
REAR_DEPTH = 3
MIN_RANK_SIZE = 2


def projection(origin: Position, cell: Position, direction: Direction) -> int:
    """Signed distance along a threatened bearing."""
    dx, dy = add((0, 0), direction)
    return (cell[0] - origin[0]) * dx + (cell[1] - origin[1]) * dy


def threatened_sides(
    core: Position, enemies: Sequence[UnitView]
) -> dict[Direction, tuple[UnitView, ...]]:
    """Diagonal approaches threaten both adjoining sides."""
    result: dict[Direction, tuple[UnitView, ...]] = {}
    for direction in DIRECTIONS:
        contacts = tuple(
            enemy
            for enemy in enemies
            if projection(core, enemy.position, direction) > 0
            and 2 * projection(core, enemy.position, direction)
            >= max(
                abs(enemy.position[0] - core[0]),
                abs(enemy.position[1] - core[1]),
            )
        )
        if contacts:
            result[direction] = contacts
    return result


def can_attack(enemy: UnitView, target: Position, rocks: Container[Position]) -> bool:
    if enemy.unit_type is UnitType.RANGER:
        return line_of_fire(enemy.position, target, rocks)
    return (
        enemy.unit_type is UnitType.VANGUARD and manhattan(enemy.position, target) == 1
    )


def precision_volley(
    positions: Mapping[UUID, Position],
    enemies: Sequence[UnitView],
    core: Position,
    rocks: Container[Position],
    shooter_hp: Mapping[UUID, int] | None = None,
) -> dict[UUID, UnitView]:
    """Allocate real target UUIDs, including multiple occupants of one cell.

    Spare healthy shooters replace wounded ones without taking arrows from
    another kill. Necessary wounded shooters still participate in the volley.
    """
    free = dict(positions)
    result: dict[UUID, UnitView] = {}
    for enemy in sorted(
        enemies,
        key=lambda target: (
            not can_attack(target, core, rocks),
            target.hp,
            manhattan(core, target.position),
            target.id.int,
        ),
    ):
        shooters = [
            actor_id
            for actor_id, position in free.items()
            if line_of_fire(position, enemy.position, rocks)
        ]
        if len(shooters) < enemy.hp:
            continue
        shooters.sort(
            key=lambda actor_id: (
                sum(
                    line_of_fire(free[actor_id], target.position, rocks)
                    for target in enemies
                ),
                actor_id.int,
            )
        )
        for actor_id in shooters[: enemy.hp]:
            result[actor_id] = enemy
            del free[actor_id]
    if shooter_hp is not None:
        for actor_id, enemy in tuple(result.items()):
            if shooter_hp[actor_id] > 1:
                continue
            replacements = [
                spare_id
                for spare_id, position in free.items()
                if shooter_hp[spare_id] > 1
                and line_of_fire(position, enemy.position, rocks)
            ]
            if replacements:
                replacement = min(replacements, key=lambda spare_id: spare_id.int)
                result[replacement] = result.pop(actor_id)
                del free[replacement]
    return result


def casualty_bound(
    positions: Mapping[UUID, tuple[Position, int]],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
) -> int:
    """Conservative losses from shared arrows and a single sweep per enemy.

    A Ranger contributes one arrow, however many allies it can target. A
    Vanguard contributes damage to the largest one-cell group it can sweep.
    Distributing that budget onto the lowest HPs is an upper casualty bound.
    """
    exposed = {
        actor_id: (position, hp)
        for actor_id, (position, hp) in positions.items()
        if any(can_attack(enemy, position, rocks) for enemy in enemies)
    }
    damage = 0
    for enemy in enemies:
        victims = [
            position
            for position, _hp in exposed.values()
            if can_attack(enemy, position, rocks)
        ]
        if not victims:
            continue
        if enemy.unit_type is UnitType.RANGER:
            damage += 1
        else:
            damage += max(victims.count(position) for position in set(victims))
    losses = 0
    for hp in sorted(hp for _position, hp in exposed.values()):
        if damage < hp:
            break
        losses += 1
        damage -= hp
    return losses


def maximum_core_damage(
    core_positions: Sequence[Position],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
) -> int:
    """Worst damage over a short Core move, with MOVE/attack exclusivity.

    Enemy collision constraints are ignored conservatively. The bounded
    dynamic program considers either shooting/sweeping or moving each Tick;
    it never lets one hostile do both. Four-Tick routes have at most 41 cells.
    """
    total = 0
    for enemy in enemies:
        states = {enemy.position: 0}
        for core_position in core_positions:
            next_states: dict[Position, int] = {}
            for position, damage in states.items():
                attacks = (
                    line_of_fire(position, core_position, rocks)
                    if enemy.unit_type is UnitType.RANGER
                    else manhattan(position, core_position) == 1
                )
                next_states[position] = max(
                    next_states.get(position, 0), damage + int(attacks)
                )
                for neighbor in adjacent_positions(position):
                    if neighbor not in rocks and neighbor not in core_positions:
                        next_states[neighbor] = max(
                            next_states.get(neighbor, 0), damage
                        )
            states = next_states
        total += max(states.values(), default=0)
    return total


def layered_posts(
    core: Position,
    allies: Sequence[Ranger],
    enemies: Sequence[UnitView],
    previous: Mapping[str, Position],
    unavailable: Container[Position],
    rocks: Container[Position],
) -> dict[UUID, Position]:
    """Two stable firing ranks per threatened side before surplus allocation.

    Slots are terrain-aware, unique, and scored by actual firing geometry.
    Nearby previous holders retain useful slots. Only two obsolete firing
    slots may rotate per Tick; casualties refill their own gaps.
    """
    sides = threatened_sides(core, enemies)
    if not sides or not allies:
        return {}
    remaining = {ally.id: ally for ally in allies}
    claimed: set[Position] = set()
    assignments: dict[UUID, Position] = {}
    rotations = 0

    def candidates(direction: Direction, depth: int) -> list[Position]:
        dx, dy = add((0, 0), direction)
        width = 4 if depth >= REAR_DEPTH else 8
        return [
            (core[0] + dx * depth - dy * offset, core[1] + dy * depth + dx * offset)
            for offset in range(-width, width + 1)
            if (
                core[0] + dx * depth - dy * offset,
                core[1] + dy * depth + dx * offset,
            )
            not in unavailable
            and (
                core[0] + dx * depth - dy * offset,
                core[1] + dy * depth + dx * offset,
            )
            not in rocks
        ]

    def score(cell: Position, direction: Direction) -> int:
        return sum(
            (8 if can_attack(enemy, core, rocks) else 3)
            * int(line_of_fire(cell, enemy.position, rocks))
            + sum(
                line_of_fire(cell, neighbor, rocks)
                for neighbor in adjacent_positions(enemy.position)
                if neighbor not in rocks
            )
            for enemy in sides.get(direction, ())
        )

    def assign(direction: Direction, depth: int) -> bool:
        nonlocal rotations
        choices = [cell for cell in candidates(direction, depth) if cell not in claimed]
        if not choices or not remaining:
            return False
        best = max(score(cell, direction) for cell in choices)
        holders = [
            ally for ally in remaining.values() if previous.get(str(ally.id)) in choices
        ]
        if holders:
            holder = min(
                holders,
                key=lambda ally: (
                    -score(previous[str(ally.id)], direction),
                    manhattan(ally.position, previous[str(ally.id)]),
                    ally.id.int,
                ),
            )
            old = previous[str(holder.id)]
            if score(old, direction) >= best or rotations >= 2:
                goal = old
            else:
                goal = max(
                    choices,
                    key=lambda cell: (
                        score(cell, direction),
                        -manhattan(holder.position, cell),
                        cell,
                    ),
                )
                rotations += 1
        else:
            goal = max(
                choices,
                key=lambda cell: (
                    score(cell, direction),
                    -min(manhattan(ally.position, cell) for ally in remaining.values()),
                    -manhattan(core, cell),
                    cell,
                ),
            )
            holder = min(
                remaining.values(),
                key=lambda ally: (manhattan(ally.position, goal), ally.id.int),
            )
        assignments[holder.id] = goal
        del remaining[holder.id]
        claimed.add(goal)
        return True

    counts = dict.fromkeys(sides, 0)
    ordered = sorted(
        sides, key=lambda direction: (-len(sides[direction]), direction.value)
    )
    for depth in (FRONT_DEPTH, REAR_DEPTH):
        for _ in range(MIN_RANK_SIZE):
            for direction in ordered:
                counts[direction] += int(assign(direction, depth))
    # Keep a small watch on other bearings; allocate every remaining useful
    # position to the real incoming forces rather than a symmetric army split.
    for direction in DIRECTIONS:
        if direction not in sides and len(remaining) > len(sides):
            assign(direction, REAR_DEPTH)
    while remaining:
        direction = max(
            ordered,
            key=lambda bearing: len(sides[bearing]) / (counts[bearing] + 1),
        )
        depth = FRONT_DEPTH if counts[direction] % 2 else REAR_DEPTH
        if not any(
            assign(direction, rank)
            for rank in (
                depth,
                REAR_DEPTH if depth == FRONT_DEPTH else FRONT_DEPTH,
                REAR_DEPTH - 1,
                REAR_DEPTH - 2,
            )
        ):
            ordered.remove(direction)
            if not ordered:
                break
            continue
        counts[direction] += 1
    return assignments
