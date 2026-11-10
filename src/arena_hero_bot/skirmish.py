"""Short local clearance with fixed membership and conservative ingress."""

from __future__ import annotations

from collections import Counter
from collections.abc import Container, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from arena_hero import Position, Ranger, Turn, UnitType, UnitView

from .counterattack import AssaultSlot, reachable_opposition, safe_routes
from .defense import can_attack
from .geometry import adjacent_positions, firing_positions, line_of_fire, manhattan

MAX_STEPS = 6
MAX_TICKS = 12
LOST_TICKS = 8
COOLDOWN_TICKS = 32
CONTACT_RADIUS = 12


@dataclass(slots=True)
class Skirmish:
    """One bounded clearance; disappearance never means a confirmed kill."""

    core_id: str
    core_position: Position
    target_ids: tuple[str, ...]
    target_position: Position
    started: int
    last_seen: int
    slots: dict[str, AssaultSlot]
    origins: dict[str, Position]
    entered: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Skirmish:
        return cls(
            core_id=str(raw["core_id"]),
            core_position=tuple(raw["core_position"]),
            target_ids=tuple(raw["target_ids"]),
            target_position=tuple(raw["target_position"]),
            started=int(raw["started"]),
            last_seen=int(raw["last_seen"]),
            slots={
                key: AssaultSlot(tuple(value["stage"]), tuple(value["firing"]))
                for key, value in raw["slots"].items()
            },
            origins={key: tuple(value) for key, value in raw["origins"].items()},
            entered=raw.get("entered"),
        )


def entry_viable(
    members: Sequence[Ranger],
    slots: dict[str, AssaultSlot],
    targets: Sequence[UnitView],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
) -> bool:
    """Count the MOVE exposure before promising next-Tick killing arrows."""
    positions = {
        member.id: (slots[str(member.id)].firing, member.hp) for member in members
    }
    if not targets or len(positions) < 2:
        return False
    target_ids = {target.id for target in targets}
    cells = tuple(position for position, _hp in positions.values())
    if any(
        not line_of_fire(cell, target.position, rocks)
        for cell in cells
        for target in targets
    ) or any(
        enemy.id not in target_ids
        and any(can_attack(enemy, cell, rocks) for cell in cells)
        for enemy in enemies
    ):
        return False
    opposition = reachable_opposition(cells, enemies, rocks)
    if opposition > 2 or any(
        enemy.unit_type is not UnitType.RANGER
        and reachable_opposition(cells, (enemy,), rocks)
        for enemy in enemies
    ):
        # This short two-volley calculation models arrows, not Vanguard sweeps.
        # Mixed forces retain the existing defensive/counterattack rules.
        return False
    ingress_arrows = sum(
        any(can_attack(enemy, cell, rocks) for cell in cells) for enemy in enemies
    )
    original_size = len(members)
    target_hp = tuple(sorted(target.hp for target in targets))
    for arrived_hp in _retaliation_states(
        tuple(sorted(member.hp for member in members)), ingress_arrows
    ):
        if len(arrived_hp) < opposition + 1:
            return False
        states = {(arrived_hp, target_hp)}
        # Retaliation is simultaneous: an enemy killed by this volley still
        # contributes its arrow. Enumerate concentrated AND split damage so
        # surviving wounded shooters are not silently treated as full HP.
        for _ in range(2):
            following = set()
            for hp, remaining in states:
                arrows = len(hp)
                if arrows < remaining[0]:
                    return False
                next_targets = []
                for health in remaining:
                    hit = min(arrows, health)
                    arrows -= hit
                    if health > hit:
                        next_targets.append(health - hit)
                retaliation = opposition - (len(target_hp) - len(remaining))
                for survivors in _retaliation_states(hp, retaliation):
                    if original_size - len(survivors) > len(target_hp):
                        return False
                    if next_targets:
                        following.add((survivors, tuple(next_targets)))
            states = following
        if states:
            return False
    return True


def _retaliation_states(hp: tuple[int, ...], arrows: int) -> set[tuple[int, ...]]:
    """All HP distributions for at most six allies and two hostile arrows."""
    states = {hp}
    for _ in range(arrows):
        following = set(states)
        for health in states:
            for index, value in enumerate(health):
                changed = (*health[:index], *health[index + 1 :])
                if value > 1:
                    changed = (*changed, value - 1)
                following.add(tuple(sorted(changed)))
        states = following
    return states


def choose_slots(
    turn: Turn,
    allies: Sequence[Ranger],
    targets: Sequence[UnitView],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
    blocked: Container[Position],
    max_party: int,
    *,
    origins: Mapping[str, Position] | None = None,
    min_party: int = 2,
    route_budget: int = MAX_STEPS,
) -> dict[str, AssaultSlot]:
    """Bounded beam search for a safe assembly and simultaneous arrival."""
    if turn.core is None or not targets or max_party < 2:
        return {}
    core = turn.core.position
    target = targets[0].position
    candidates = sorted(
        allies, key=lambda ally: (manhattan(ally.position, target), ally.id)
    )[:12]
    routes = {
        ally.id: safe_routes(ally.position, core, blocked, enemies, rocks)
        for ally in candidates
    }
    occupants: dict[Position, set[UUID]] = {}
    for unit in turn.units:
        occupants.setdefault(unit.position, set()).add(unit.id)
    options: list[tuple[int, UUID, Position, Position]] = []
    for firing in firing_positions(target):
        if (
            firing in blocked
            or manhattan(core, firing) > CONTACT_RADIUS + 4
            or not all(line_of_fire(firing, foe.position, rocks) for foe in targets)
        ):
            continue
        for stage in adjacent_positions(firing):
            if stage in blocked or any(can_attack(e, stage, rocks) for e in enemies):
                continue
            for ally in candidates:
                route = routes[ally.id].get(stage)
                origin = origins[str(ally.id)] if origins is not None else ally.position
                if (
                    route is not None
                    and len(route) + 1 <= route_budget
                    and manhattan(origin, firing) <= MAX_STEPS
                    and all(manhattan(origin, cell) <= MAX_STEPS for cell in route)
                ):
                    options.append((len(route), ally.id, stage, firing))
    options.sort()
    # A few choices per UUID/cell retain alternatives without exponential
    # growth in army size or the known map.
    counts: Counter[tuple[UUID, Position]] = Counter()
    per_actor: Counter[UUID] = Counter()
    bounded: list[tuple[int, UUID, Position, Position]] = []
    for option in options:
        key = option[1], option[3]
        if counts[key] < 2 and per_actor[option[1]] < 16:
            bounded.append(option)
            counts[key] += 1
            per_actor[option[1]] += 1
    beam: list[tuple[tuple[int, UUID, Position, Position], ...]] = [()]
    by_id = {ally.id: ally for ally in candidates}
    for size in range(1, min(6, max_party) + 1):
        extended = set()
        for selected in beam:
            ids = {option[1] for option in selected}
            for option in bounded:
                if option[1] in ids:
                    continue
                group = tuple(sorted((*selected, option)))
                group_ids = ids | {option[1]}
                stages = Counter(item[2] for item in group)
                firing_cells = Counter(item[3] for item in group)
                if stages.keys() & firing_cells.keys() or any(
                    count + len(occupants.get(cell, set()) - group_ids) > 2
                    for cells in (stages, firing_cells)
                    for cell, count in cells.items()
                ):
                    continue
                extended.add(group)
        beam = sorted(
            extended,
            key=lambda group: (
                max(item[0] for item in group),
                sum(item[0] for item in group),
                group,
            ),
        )[:32]
        if size not in (2, 4, 6) or size < min_party:
            continue
        for group in beam:
            slots = {
                str(actor_id): AssaultSlot(stage, firing)
                for _, actor_id, stage, firing in group
            }
            if entry_viable(
                [by_id[item[1]] for item in group], slots, targets, enemies, rocks
            ):
                return slots
    return {}
