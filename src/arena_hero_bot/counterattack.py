"""Bounded, simultaneous attacks against a small entrenched local force."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Container, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any
from uuid import UUID

from arena_hero import Position, Ranger, Turn, UnitType, UnitView

from .defense import can_attack, casualty_bound
from .geometry import (
    adjacent_positions,
    firing_positions,
    line_of_fire,
    manhattan,
    overlay_blockers,
)

STANDOFF_TICKS = 64
CONTACT_RADIUS = 12
CONTACT_GAP = 8
THREAT_MEMORY_TICKS = 32
MAX_ENEMIES = 4
ARMY_RATIO = 3
MIN_RESERVE = 4
MIN_PARTY = 4
MAX_PARTY = 8
MAX_ROUTE = 8
ASSEMBLY_TICKS = 16
FIRE_TICKS = 4
COOLDOWN_TICKS = 32
RETRY_TICKS = 8


@dataclass(slots=True)
class StandoffContact:
    anchor: Position
    position: Position
    since: int
    seen: int
    still: int = 1


@dataclass(frozen=True, slots=True)
class AssaultSlot:
    stage: Position
    firing: Position


@dataclass(slots=True)
class Counterattack:
    target_ids: tuple[str, ...]
    target: Position
    started: int
    slots: dict[str, AssaultSlot]
    entered: int | None = None


@dataclass(slots=True)
class CounterattackState:
    core_id: str | None = None
    core_position: Position | None = None
    contacts: dict[str, StandoffContact] = field(default_factory=dict)
    operation: Counterattack | None = None
    cooldown_until: int = -1
    retry_after: int = -1
    reason: str = "observing standoff"

    def observe(self, turn: Turn) -> None:
        core = turn.core
        if core is None:
            self.operation = None
            self.contacts.clear()
            return
        respawned = any(
            e.event_type == "CORE_RESPAWNED" and e.target_id in {None, core.id}
            for e in turn.events
        )
        if (
            self.core_id != str(core.id)
            or self.core_position != core.position
            or respawned
        ):
            self.core_id = str(core.id)
            self.core_position = core.position
            self.contacts.clear()
            self.operation = None
            self.cooldown_until = self.retry_after = -1
        destroyed = {
            str(e.target_id)
            for e in turn.events
            if e.event_type in {"DESTRUCTION_PARTICIPATION", "UNIT_DESTROYED"}
        }
        self.contacts = {
            key: contact
            for key, contact in self.contacts.items()
            if turn.tick - contact.seen <= CONTACT_GAP and key not in destroyed
        }
        for enemy in turn.visible_enemies:
            if not isinstance(enemy, UnitView) or enemy.unit_type not in {
                UnitType.RANGER,
                UnitType.VANGUARD,
            }:
                continue
            key = str(enemy.id)
            if manhattan(core.position, enemy.position) > CONTACT_RADIUS:
                self.contacts.pop(key, None)
                continue
            old = self.contacts.get(key)
            if old is None or manhattan(old.anchor, enemy.position) > 2:
                self.contacts[key] = StandoffContact(
                    enemy.position, enemy.position, turn.tick, turn.tick
                )
            elif turn.tick > old.seen:
                old.still = (
                    old.still + 1
                    if old.position == enemy.position and turn.tick == old.seen + 1
                    else 1
                )
                old.position = enemy.position
                old.seen = turn.tick

    def finish(self, tick: int, reason: str) -> None:
        self.operation = None
        self.cooldown_until = tick + COOLDOWN_TICKS
        self.retry_after = -1
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CounterattackState:
        operation = raw.get("operation")
        return cls(
            core_id=raw.get("core_id"),
            core_position=tuple(raw["core_position"])
            if raw.get("core_position") is not None
            else None,
            contacts={
                key: StandoffContact(
                    tuple(value["anchor"]),
                    tuple(value["position"]),
                    int(value["since"]),
                    int(value["seen"]),
                    int(value["still"]),
                )
                for key, value in raw.get("contacts", {}).items()
            },
            operation=Counterattack(
                tuple(operation["target_ids"]),
                tuple(operation["target"]),
                int(operation["started"]),
                {
                    key: AssaultSlot(tuple(value["stage"]), tuple(value["firing"]))
                    for key, value in operation["slots"].items()
                },
                operation.get("entered"),
            )
            if operation
            else None,
            cooldown_until=int(raw.get("cooldown_until", -1)),
            retry_after=int(raw.get("retry_after", -1)),
            reason=str(raw.get("reason", "observing standoff")),
        )


def paired_cells(positions: Sequence[Position]) -> bool:
    """Two Rangers per ingress cell; adjacent cells on one ray are legal."""
    counts = Counter(positions)
    return len(counts) >= 2 and all(count == 2 for count in counts.values())


def safe_routes(
    origin: Position,
    core: Position,
    blocked: Container[Position],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
) -> dict[Position, tuple[Position, ...]]:
    """At most 145 cells per Ranger; no assembly step enters hostile fire."""
    routes: dict[Position, tuple[Position, ...]] = {origin: ()}
    frontier = deque([origin])
    while frontier:
        cell = frontier.popleft()
        route = routes[cell]
        if len(route) == MAX_ROUTE:
            continue
        for neighbor in adjacent_positions(cell):
            if (
                neighbor in routes
                or neighbor in blocked
                or manhattan(core, neighbor) > CONTACT_RADIUS + 4
                or any(can_attack(enemy, neighbor, rocks) for enemy in enemies)
            ):
                continue
            routes[neighbor] = (*route, neighbor)
            frontier.append(neighbor)
    return routes


def entry_viable(
    positions: Mapping[UUID, tuple[Position, int]],
    targets: Sequence[UnitView],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
    *,
    army_size: int | None = None,
) -> bool:
    """Require advantage after the MOVE Tick, including adjacent enemy moves.

    Other current firing lines are excluded outright. An enemy able to move
    into a firing line next Tick counts towards opposition, but cannot also
    shoot on the ingress Tick. Arrival losses can exceed the eventual kills;
    enough survivors must still be able to execute the target volley.
    """
    if not targets or len(positions) < MIN_PARTY:
        return False
    target_ids = {target.id for target in targets}
    cells = tuple(position for position, _hp in positions.values())
    if not paired_cells(cells):
        return False
    if any(
        not line_of_fire(cell, target.position, rocks)
        for cell in cells
        for target in targets
    ):
        return False
    if any(
        enemy.id not in target_ids
        and any(can_attack(enemy, cell, rocks) for cell in cells)
        for enemy in enemies
    ):
        return False
    opposition = reachable_opposition(cells, enemies, rocks)
    if opposition > MAX_ENEMIES or (
        army_size is not None and army_size < ARMY_RATIO * opposition
    ):
        return False
    survivors = len(positions) - casualty_bound(positions, enemies, rocks)
    return survivors >= max(opposition + 2, sum(target.hp for target in targets))


def reachable_opposition(
    cells: Sequence[Position],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
) -> int:
    """Count identities that can cover the group now or after one legal MOVE."""
    opposition = 0
    for enemy in enemies:
        possible = [enemy]
        possible.extend(
            enemy.model_copy(update={"position": cell})
            for cell in adjacent_positions(enemy.position)
            if cell not in rocks
        )
        if any(can_attack(foe, cell, rocks) for foe in possible for cell in cells):
            opposition += 1
    return opposition


def choose_formation(
    turn: Turn,
    allies: Sequence[Ranger],
    targets: Sequence[UnitView],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
    dangerous: Container[Position],
    max_party: int,
    *,
    army_size: int | None = None,
) -> dict[str, AssaultSlot]:
    """Bounded matching of safe staging capacity to paired firing cells.

    No angular separation is required: two stacks may enter the same ray at
    different distances. A pair may share a safe stage before entering together.
    """
    core = turn.core
    if core is None or not targets or max_party < MIN_PARTY:
        return {}
    target = targets[0].position
    target_ids = {enemy.id for enemy in targets}
    unavailable = {core.position, *turn.resource_cells}
    unavailable.update(enemy.position for enemy in turn.visible_enemies)
    occupied = {unit.position for unit in turn.units}
    occupants: dict[Position, set[UUID]] = {}
    for unit in turn.units:
        occupants.setdefault(unit.position, set()).add(unit.id)
    candidates = sorted(
        allies, key=lambda ally: (manhattan(ally.position, target), ally.id)
    )[:24]
    paths = {
        ally.id: safe_routes(
            ally.position,
            core.position,
            overlay_blockers(rocks, unavailable, dangerous),
            enemies,
            rocks,
        )
        for ally in candidates
    }
    options: list[tuple[int, UUID, Position, Position]] = []
    for firing in firing_positions(target):
        if (
            firing in rocks
            or firing in unavailable
            or firing in occupied
            or firing in dangerous
            or manhattan(core.position, firing) > CONTACT_RADIUS + 4
            or not line_of_fire(firing, target, rocks)
            or any(
                enemy.id not in target_ids and can_attack(enemy, firing, rocks)
                for enemy in enemies
            )
        ):
            continue
        for stage in adjacent_positions(firing):
            if stage in rocks or stage in unavailable or stage in dangerous:
                continue
            if any(can_attack(enemy, stage, rocks) for enemy in enemies):
                continue
            for ally in candidates:
                if len(occupants.get(stage, set()) - {ally.id}) >= 2:
                    continue
                route = paths[ally.id].get(stage)
                if route is not None:
                    options.append((len(route), ally.id, stage, firing))
    options.sort()
    pairs: list[tuple[tuple[int, UUID, Position, Position], ...]] = []
    for firing in sorted({option[3] for option in options}):
        # Retain several alternative identities per stage so another stack
        # can use the closest Ranger without making this pair impossible.
        choices = []
        for stage in adjacent_positions(firing):
            choices.extend([o for o in options if o[2:] == (stage, firing)][:8])
        candidates_for_cell = [
            (left, right)
            for left, right in combinations(choices, 2)
            if left[1] != right[1]
            and all(
                count + len(occupants.get(stage, set()) - {left[1], right[1]}) <= 2
                for stage, count in Counter((left[2], right[2])).items()
            )
        ]
        candidates_for_cell.sort(
            key=lambda pair: (max(o[0] for o in pair), sum(o[0] for o in pair), pair)
        )
        pairs.extend(candidates_for_cell[:16])
    hp = {ally.id: ally.hp for ally in candidates}
    # A small beam bounds matching work independently of the full army/map.
    beam: list[tuple[tuple[int, UUID, Position, Position], ...]] = [()]
    for size in range(2, min(MAX_PARTY, max_party) + 1, 2):
        extended = set()
        for selected in beam:
            ids = {o[1] for o in selected}
            used = {cell for o in selected for cell in o[2:]}
            for pair in pairs:
                if any(o[1] in ids or o[2] in used or o[3] in used for o in pair):
                    continue
                extended.add(tuple(sorted((*selected, *pair))))
        beam = sorted(
            extended,
            key=lambda group: (
                max(o[0] for o in group),
                sum(o[0] for o in group),
                group,
            ),
        )[:48]
        for selected in beam:
            if size >= MIN_PARTY and entry_viable(
                {
                    actor_id: (firing, hp[actor_id])
                    for _, actor_id, _, firing in selected
                },
                targets,
                enemies,
                rocks,
                army_size=army_size,
            ):
                return {
                    str(actor_id): AssaultSlot(stage, firing)
                    for _, actor_id, stage, firing in selected
                }
    return {}
