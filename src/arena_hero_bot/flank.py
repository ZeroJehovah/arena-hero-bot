"""A bounded execution option for an existing local Ranger mission.

Observation never owns fighters. An entry permit covers exactly one group's
last MOVE in one Tick; firing and all other movement use real health and fire.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import combinations
from time import monotonic
from typing import Any
from uuid import UUID

from arena_hero import CoreView, Position, Ranger, Turn, UnitType, UnitView

from .counterattack import AssaultSlot, paired_cells
from .defense import can_attack, fire_groups, precision_volley
from .exploration import can_see
from .geometry import adjacent_positions, firing_positions, line_of_fire, manhattan

STABLE_TICKS = 8
CONTACT_LIMIT = 128
CONTACT_TTL = 32
GROUP_LIMIT = 8
EXTENSION_LIMIT = 256
BEAM_WIDTH = 32
PATH_LIMIT = 64
SEARCH_SECONDS = 0.2
PROJECTION_STATE_LIMIT = 4096
PREFIRE_TTL = 8


class Kind(StrEnum):
    COUNTERATTACK = "counterattack"
    SKIRMISH = "skirmish"
    PROTECTION = "worker_area_protection"
    FIRELINE = "local_fireline"


class Phase(StrEnum):
    ASSEMBLE = "ASSEMBLE"
    ENTER = "ENTER"
    OPENING = "OPENING"
    FINISH = "FINISH"


def group_key(kind: Kind, targets: Sequence[UnitView]) -> str:
    return (
        kind.value
        + ":"
        + ";".join(
            f"{target.id}@{target.position[0]},{target.position[1]}"
            for target in sorted(targets, key=lambda enemy: enemy.id)
        )
    )


def observer_radius(observer: CoreView | UnitView) -> int:
    if isinstance(observer, CoreView):
        return 5
    return {UnitType.WORKER: 3, UnitType.VANGUARD: 4, UnitType.RANGER: 5}[
        observer.unit_type
    ]


def mutual_front(
    targets: Sequence[UnitView],
    rangers: Sequence[Ranger],
    rocks: Container[Position],
) -> tuple[Ranger, ...]:
    """Every target needs an actual Ranger observer on both sides of the fog."""
    front: dict[UUID, Ranger] = {}
    for target in targets:
        observers = [
            ranger
            for ranger in rangers
            if can_see(ranger.position, target.position, 5, rocks)
            and can_see(target.position, ranger.position, 5, rocks)
        ]
        if not observers:
            return ()
        front.update((ranger.id, ranger) for ranger in observers)
    return tuple(front[key] for key in sorted(front))


@dataclass(slots=True)
class Contact:
    kind: Kind
    since: int
    seen: int
    front_ids: tuple[str, ...]

    @property
    def ticks(self) -> int:
        return self.seen - self.since + 1


@dataclass(frozen=True, slots=True)
class Parent:
    kind: Kind
    task_id: str
    started: int
    deadline: int
    assembly_deadline: int
    fire_ticks: int
    max_steps: int
    party_sizes: tuple[int, ...]
    paired: bool = False
    kill_all: bool = False
    fixed_ids: frozenset[UUID] = frozenset()


@dataclass(slots=True)
class Operation:
    parent: Parent
    core_id: str
    core_position: Position
    target_ids: tuple[str, ...]
    target_positions: dict[str, Position]
    member_ids: tuple[str, ...]
    front_ids: tuple[str, ...]
    origins: dict[str, Position]
    slots: dict[str, AssaultSlot]
    routes: dict[str, tuple[Position, ...]]
    stable_ticks: int
    created_tick: int = 0
    stable_since: int = 0
    route_validated_tick: int = 0
    phase: Phase = Phase.ASSEMBLE
    entry_tick: int | None = None
    first_shot: int | None = None
    last_seen: int = 0
    confirmed_destroyed: set[str] = field(default_factory=set)
    front_losses: set[str] = field(default_factory=set)
    flank_losses: set[str] = field(default_factory=set)
    other_losses: set[str] = field(default_factory=set)
    front_damage: int = 0
    flank_damage: int = 0
    actual_hits: int = 0
    healing_cost: int = 0
    deposits: int = 0
    spawns: int = 0
    core_damage: int = 0
    worker_losses: set[str] = field(default_factory=set)
    worker_ids: frozenset[str] = frozenset()
    last_event_tick: int = -1
    actual_hp: dict[str, int] = field(default_factory=dict)
    actual_arrivals: tuple[str, ...] = ()
    opening_allocation: dict[str, str] = field(default_factory=dict)
    events_seen: dict[str, int] = field(default_factory=dict)
    friendly_ids: frozenset[str] = frozenset()

    @property
    def remaining_ids(self) -> tuple[str, ...]:
        return tuple(
            key for key in self.target_ids if key not in self.confirmed_destroyed
        )

    def summary(self, tick: int) -> dict[str, Any]:
        return {
            "id": self.parent.task_id + ":flank",
            "parent_kind": self.parent.kind.value,
            "parent_id": self.parent.task_id,
            "phase": self.phase.value,
            "stable_ticks": self.stable_ticks,
            "stable_since": self.stable_since,
            "created_tick": self.created_tick,
            "route_validated_tick": self.route_validated_tick,
            "original_targets": list(self.target_ids),
            "remaining_targets": list(self.remaining_ids),
            "members": list(self.member_ids),
            "front_members": list(self.front_ids),
            "actual_arrivals": list(self.actual_arrivals),
            "actual_hp": dict(self.actual_hp),
            "opening_allocation": dict(self.opening_allocation),
            "arrival_model": "qualified_zero_damage",
            "entry_tick": self.entry_tick,
            "first_shot": self.first_shot,
            "started": self.parent.started,
            "deadline": self.parent.deadline,
            "remaining_ticks": max(0, self.parent.deadline - tick),
            "confirmed_destroyed": sorted(self.confirmed_destroyed),
            "front_losses": sorted(self.front_losses),
            "flank_losses": sorted(self.flank_losses),
            "other_losses": sorted(self.other_losses),
            "front_damage": self.front_damage,
            "flank_damage": self.flank_damage,
            "actual_hits": self.actual_hits,
            "healing_cost": self.healing_cost,
            "deposits": self.deposits,
            "spawns": self.spawns,
            "core_damage": self.core_damage,
            "worker_losses": sorted(self.worker_losses),
            "worker_ids": sorted(self.worker_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "confirmed_destroyed": sorted(self.confirmed_destroyed),
            "front_losses": sorted(self.front_losses),
            "flank_losses": sorted(self.flank_losses),
            "other_losses": sorted(self.other_losses),
            "friendly_ids": sorted(self.friendly_ids),
            "worker_losses": sorted(self.worker_losses),
            "worker_ids": sorted(self.worker_ids),
            "parent": asdict(self.parent)
            | {"fixed_ids": sorted(str(key) for key in self.parent.fixed_ids)},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Operation:
        parent = raw["parent"]
        result = cls(
            parent=Parent(
                kind=Kind(parent["kind"]),
                task_id=str(parent["task_id"]),
                started=int(parent["started"]),
                deadline=int(parent["deadline"]),
                assembly_deadline=int(parent["assembly_deadline"]),
                fire_ticks=int(parent["fire_ticks"]),
                max_steps=int(parent["max_steps"]),
                party_sizes=tuple(int(size) for size in parent["party_sizes"]),
                paired=bool(parent.get("paired", False)),
                kill_all=bool(parent.get("kill_all", False)),
                fixed_ids=frozenset(UUID(key) for key in parent.get("fixed_ids", ())),
            ),
            core_id=str(raw["core_id"]),
            core_position=tuple(raw["core_position"]),
            target_ids=tuple(raw["target_ids"]),
            target_positions={
                key: tuple(cell) for key, cell in raw["target_positions"].items()
            },
            member_ids=tuple(raw["member_ids"]),
            front_ids=tuple(raw["front_ids"]),
            origins={key: tuple(cell) for key, cell in raw["origins"].items()},
            slots={
                key: AssaultSlot(tuple(value["stage"]), tuple(value["firing"]))
                for key, value in raw["slots"].items()
            },
            routes={
                key: tuple(tuple(cell) for cell in path)
                for key, path in raw["routes"].items()
            },
            stable_ticks=int(raw["stable_ticks"]),
            created_tick=int(raw["created_tick"]),
            stable_since=int(raw["stable_since"]),
            route_validated_tick=int(raw.get("route_validated_tick", 0)),
            phase=Phase(raw["phase"]),
            entry_tick=int(raw["entry_tick"])
            if raw.get("entry_tick") is not None
            else None,
            first_shot=int(raw["first_shot"])
            if raw.get("first_shot") is not None
            else None,
            last_seen=int(raw.get("last_seen", 0)),
            confirmed_destroyed=set(raw.get("confirmed_destroyed", ())),
            front_losses=set(raw.get("front_losses", ())),
            flank_losses=set(raw.get("flank_losses", ())),
            other_losses=set(raw.get("other_losses", ())),
            front_damage=int(raw.get("front_damage", 0)),
            flank_damage=int(raw.get("flank_damage", 0)),
            actual_hits=int(raw.get("actual_hits", 0)),
            healing_cost=int(raw.get("healing_cost", 0)),
            deposits=int(raw.get("deposits", 0)),
            spawns=int(raw.get("spawns", 0)),
            core_damage=int(raw.get("core_damage", 0)),
            worker_losses=set(raw.get("worker_losses", ())),
            worker_ids=frozenset(raw.get("worker_ids", ())),
            last_event_tick=int(raw.get("last_event_tick", -1)),
            actual_hp={key: int(hp) for key, hp in raw.get("actual_hp", {}).items()},
            actual_arrivals=tuple(raw.get("actual_arrivals", ())),
            opening_allocation=dict(raw.get("opening_allocation", {})),
            events_seen={
                key: int(tick) for key, tick in raw.get("events_seen", {}).items()
            },
            friendly_ids=frozenset(raw.get("friendly_ids", ())),
        )
        ids = (
            *result.target_ids,
            *result.member_ids,
            *result.front_ids,
            result.core_id,
        )
        for key in ids:
            UUID(key)
        size_limits = {
            Kind.SKIRMISH: ((2, 4, 6), 2, 6, 12, 3),
            Kind.COUNTERATTACK: ((4, 6, 8), 4, 8, 20, 4),
            Kind.PROTECTION: ((2, 4, 6), 2, 24, 32, 4),
            Kind.FIRELINE: ((1, 2), 1, 3, 8, 3),
        }
        sizes, target_limit, route_limit, total_ticks, firing_ticks = size_limits[
            result.parent.kind
        ]
        cells = (
            result.core_position,
            *result.origins.values(),
            *result.target_positions.values(),
            *(
                cell
                for slot in result.slots.values()
                for cell in (slot.stage, slot.firing)
            ),
            *(cell for route in result.routes.values() for cell in route),
        )
        if (
            not result.target_ids
            or len(set(result.target_ids)) != len(result.target_ids)
            or len(result.target_ids) > target_limit
            or len(set(result.member_ids)) != len(result.member_ids)
            or not set(result.parent.party_sizes) <= set(sizes)
            or len(result.member_ids) not in result.parent.party_sizes
            or not 1 <= result.parent.max_steps <= route_limit
            or set(result.slots) != set(result.member_ids)
            or set(result.origins) != set(result.member_ids)
            or set(result.routes) != set(result.member_ids)
            or set(result.target_ids) != set(result.target_positions)
            or not result.confirmed_destroyed <= set(result.target_ids)
            or result.parent.deadline != result.parent.started + total_ticks
            or result.parent.fire_ticks != firing_ticks
            or not result.parent.started
            <= result.created_tick
            < result.parent.assembly_deadline
            <= result.parent.deadline
            or result.created_tick - result.stable_since + 1 < STABLE_TICKS
            or any(
                len(cell) != 2 or any(type(value) is not int for value in cell)
                for cell in cells
            )
            or any(
                manhattan(slot.stage, slot.firing) != 1
                or len(result.routes[key]) >= result.parent.max_steps
                or (
                    result.routes[key][-1]
                    if result.routes[key]
                    else result.origins[key]
                )
                != slot.stage
                or any(
                    manhattan(a, b) != 1
                    for a, b in zip(
                        (result.origins[key], *result.routes[key]),
                        result.routes[key],
                        strict=False,
                    )
                )
                for key, slot in result.slots.items()
            )
            or (
                result.parent.kind is Kind.COUNTERATTACK
                and (
                    not result.parent.paired
                    or not result.parent.kill_all
                    or not paired_cells(
                        tuple(slot.firing for slot in result.slots.values())
                    )
                )
            )
            or (
                result.parent.kind is Kind.PROTECTION
                and (
                    not result.parent.kill_all
                    or {str(key) for key in result.parent.fixed_ids}
                    != set(result.member_ids)
                )
            )
            or (result.phase is not Phase.ASSEMBLE and result.entry_tick is None)
            or (result.phase is Phase.FINISH and result.first_shot is None)
            or (
                result.entry_tick is not None
                and not result.created_tick
                <= result.entry_tick
                < result.parent.deadline
            )
            or (
                result.first_shot is not None
                and (
                    result.entry_tick is None
                    or not result.entry_tick
                    < result.first_shot
                    < result.parent.deadline
                )
            )
        ):
            raise ValueError("invalid flank identity or budget")
        return result


@dataclass(slots=True)
class State:
    core_id: str | None = None
    contacts: dict[str, Contact] = field(default_factory=dict)
    operation: Operation | None = None
    returns: dict[str, int] = field(default_factory=dict)
    prefire_cells: dict[Position, int] = field(default_factory=dict)
    last_result: dict[str, Any] | None = None

    def observe(
        self,
        turn: Turn,
        groups: Sequence[tuple[Kind, Sequence[UnitView]]],
        rangers: Sequence[Ranger],
        rocks: Container[Position],
    ) -> None:
        core_id = str(turn.core.id) if turn.core is not None else None
        if core_id != self.core_id or any(
            event.event_type == "CORE_RESPAWNED" for event in turn.events
        ):
            if self.operation is not None:
                self.finish(turn.tick, "core_changed")
            self.core_id = core_id
            self.contacts.clear()
            self.returns.clear()
            self.prefire_cells.clear()
        self.contacts = {
            key: value
            for key, value in self.contacts.items()
            if 0 <= turn.tick - value.seen <= CONTACT_TTL
        }
        for kind, targets in groups[:GROUP_LIMIT]:
            if not targets:
                continue
            key = group_key(kind, targets)
            front = mutual_front(targets, rangers, rocks)
            old = self.contacts.get(key)
            if not front:
                self.contacts.pop(key, None)
                continue
            if old is None or old.seen not in {turn.tick - 1, turn.tick}:
                old = Contact(kind, turn.tick, turn.tick, ())
                self.contacts[key] = old
            old.seen = turn.tick
            old.front_ids = tuple(str(ranger.id) for ranger in front)
        if len(self.contacts) > CONTACT_LIMIT:
            self.contacts = dict(
                sorted(self.contacts.items(), key=lambda item: (item[1].seen, item[0]))[
                    -CONTACT_LIMIT:
                ]
            )
        self.prefire_cells = {
            cell: tick
            for cell, tick in self.prefire_cells.items()
            if 0 <= turn.tick - tick <= PREFIRE_TTL
        }
        hostile_ids = {str(enemy.id) for enemy in turn.visible_enemies}
        for event in turn.events:
            if (
                str(event.actor_id) in hostile_ids
                and event.position is not None
                and event.event_type in {"SHOT_HIT", "SHOT_MISSED", "UNIT_DAMAGED"}
                and 0 <= turn.tick - event.tick <= PREFIRE_TTL
            ):
                self.prefire_cells[event.position] = event.tick
        if len(self.prefire_cells) > CONTACT_LIMIT:
            self.prefire_cells = dict(
                sorted(self.prefire_cells.items(), key=lambda item: (item[1], item[0]))[
                    -CONTACT_LIMIT:
                ]
            )
        self._settlement(turn)
        live = {str(ranger.id): ranger for ranger in turn.rangers}
        self.returns = {
            key: tick
            for key, tick in self.returns.items()
            if key in live
            and (
                turn.core is None
                or manhattan(live[key].position, turn.core.position) > 11
            )
        }

    def _settlement(self, turn: Turn) -> None:
        operation = self.operation
        if operation is None:
            return
        live = {str(ranger.id): ranger for ranger in turn.rangers}
        operation.actual_hp = {
            key: live[key].hp
            for key in (*operation.member_ids, *operation.front_ids)
            if key in live
        }
        operation.actual_arrivals = tuple(
            key
            for key in operation.member_ids
            if key in live and live[key].position == operation.slots[key].firing
        )
        operation.friendly_ids |= frozenset(str(unit.id) for unit in turn.units)
        operation.worker_ids |= frozenset(str(worker.id) for worker in turn.workers)
        previous_event_tick = operation.last_event_tick
        for event in turn.events:
            key = str(event.event_id)
            if (
                key in operation.events_seen
                or event.tick <= previous_event_tick
                or event.tick < operation.parent.started
            ):
                continue
            operation.last_event_tick = max(operation.last_event_tick, event.tick)
            operation.events_seen[key] = event.tick
            target_id = str(event.target_id)
            if event.event_type in {"UNIT_DESTROYED", "DESTRUCTION_PARTICIPATION"}:
                if target_id in operation.target_ids:
                    operation.confirmed_destroyed.add(target_id)
                elif target_id in operation.member_ids:
                    operation.flank_losses.add(target_id)
                elif target_id in operation.front_ids:
                    operation.front_losses.add(target_id)
                elif target_id in operation.friendly_ids:
                    operation.other_losses.add(target_id)
                if target_id in operation.worker_ids:
                    operation.worker_losses.add(target_id)
            values = event.values or {}
            damage = values.get("damage", 0)
            if (
                event.event_type == "UNIT_DAMAGED"
                and isinstance(damage, int)
                and damage > 0
            ):
                if target_id in operation.member_ids:
                    operation.flank_damage += damage
                elif target_id in operation.front_ids:
                    operation.front_damage += damage
                if target_id in operation.friendly_ids and values.get("hp") == 0:
                    if target_id in operation.member_ids:
                        operation.flank_losses.add(target_id)
                    elif target_id in operation.front_ids:
                        operation.front_losses.add(target_id)
                    else:
                        operation.other_losses.add(target_id)
                    if target_id in operation.worker_ids:
                        operation.worker_losses.add(target_id)
            if event.event_type == "CORE_DAMAGED" and target_id == operation.core_id:
                operation.core_damage += max(0, int(damage))
            if (
                event.event_type == "SHOT_HIT"
                and target_id in operation.target_ids
                and str(event.actor_id) in operation.member_ids
                and isinstance(damage, int)
                and not isinstance(damage, bool)
                and damage > 0
            ):
                operation.actual_hits += damage
            if event.healing is not None and (
                str(event.actor_id) in operation.friendly_ids
                or target_id in operation.friendly_ids
                or str(event.actor_id) == operation.core_id
            ):
                operation.healing_cost += event.healing.cost
            operation.deposits += int(
                event.event_type == "DEPOSIT_SUCCEEDED"
                and target_id == operation.core_id
                and str(event.actor_id) in operation.friendly_ids
            )
            operation.spawns += int(
                event.event_type == "CORE_SPAWN_SUCCEEDED"
                and str(event.actor_id) == operation.core_id
            )
        operation.events_seen = dict(
            sorted(operation.events_seen.items(), key=lambda item: (item[1], item[0]))[
                -CONTACT_LIMIT:
            ]
        )

    def finish(self, tick: int, reason: str) -> None:
        operation = self.operation
        if operation is not None:
            self.last_result = operation.summary(tick) | {
                "phase": "DONE",
                "reason": reason,
                "ended": tick,
            }
            self.returns.update(dict.fromkeys(operation.member_ids, tick))
        self.operation = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_id": self.core_id,
            "contacts": {key: asdict(value) for key, value in self.contacts.items()},
            "operation": self.operation.to_dict() if self.operation else None,
            "returns": dict(self.returns),
            "prefire_cells": [
                [*cell, tick] for cell, tick in self.prefire_cells.items()
            ],
            "last_result": self.last_result,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> State:
        if not isinstance(raw, dict):
            return cls()
        result = cls(core_id=raw.get("core_id"), last_result=raw.get("last_result"))
        try:
            result.contacts = {
                key: Contact(
                    Kind(value["kind"]),
                    int(value["since"]),
                    int(value["seen"]),
                    tuple(value["front_ids"]),
                )
                for key, value in list(raw.get("contacts", {}).items())[-CONTACT_LIMIT:]
                if isinstance(value, dict) and int(value["since"]) <= int(value["seen"])
            }
            result.returns = {
                str(UUID(key)): int(tick)
                for key, tick in list(raw.get("returns", {}).items())[-CONTACT_LIMIT:]
            }
            result.prefire_cells = {
                (int(x), int(y)): int(tick)
                for x, y, tick in raw.get("prefire_cells", ())[-CONTACT_LIMIT:]
            }
            if raw.get("operation"):
                result.operation = Operation.from_dict(raw["operation"])
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            result.operation = None
            result.contacts.clear()
        return result


@dataclass(frozen=True, slots=True)
class EntryPermit:
    tick: int
    task_id: str
    destinations: Mapping[UUID, Position]

    def allows(
        self, tick: int, operation: Operation | None, actor_id: UUID, cell: Position
    ) -> bool:
        return (
            operation is not None
            and operation.phase is Phase.ASSEMBLE
            and self.tick == tick
            and self.task_id == operation.parent.task_id
            and self.destinations.get(actor_id) == cell
            and {str(key) for key in self.destinations} == set(operation.member_ids)
            and all(
                self.destinations[UUID(key)] == operation.slots[key].firing
                for key in operation.member_ids
            )
            and str(actor_id) in operation.member_ids
            and operation.slots[str(actor_id)].firing == cell
        )


@dataclass(slots=True)
class SearchBudget:
    deadline: float
    groups: int = 0
    extensions: int = 0
    paths: int = 0
    started: float = field(default_factory=monotonic)
    spent: float = 0.0
    reasons: Counter[str] = field(default_factory=Counter)

    def exhausted(self) -> bool:
        return monotonic() >= self.deadline or self.extensions >= EXTENSION_LIMIT

    def reject(self, reason: str) -> None:
        self.reasons[reason] += 1


def prefire_cells(
    targets: Sequence[UnitView],
    front: Sequence[Ranger],
    rocks: Container[Position],
    planned_positions: Mapping[UUID, Position] | None = None,
) -> frozenset[Position]:
    positions = {ranger.position for ranger in front}
    positions.update((planned_positions or {}).values())
    return frozenset(
        cell
        for position in positions
        for cell in (position, *adjacent_positions(position))
        if any(line_of_fire(target.position, cell, rocks) for target in targets)
    )


def cell_rejection(
    cell: Position,
    targets: Sequence[UnitView],
    observers: Sequence[CoreView | UnitView],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
    prefire: Container[Position],
) -> str | None:
    if cell in rocks:
        return "terrain_blocked"
    if cell in prefire:
        return "front_prefire_overlap"
    if any(
        can_see(observer.position, cell, observer_radius(observer), rocks)
        for observer in observers
    ):
        return "known_observer"
    target_ids = {target.id for target in targets}
    for enemy in enemies:
        if enemy.unit_type is UnitType.VANGUARD:
            if any(
                manhattan(origin, cell) == 1
                for origin in (
                    enemy.position,
                    *(
                        neighbor
                        for neighbor in adjacent_positions(enemy.position)
                        if neighbor not in rocks
                    ),
                )
            ):
                return "vanguard_risk"
        elif enemy.id not in target_ids and can_attack(enemy, cell, rocks):
            return "third_party_fire"
    return None


def _match_volley(
    required: Sequence[UnitView],
    choices: Mapping[UUID, tuple[UUID, ...]],
    free: frozenset[UUID],
    chosen: dict[UUID, UnitView],
) -> dict[UUID, UnitView] | None:
    if not required:
        return chosen.copy()
    target = required[0]
    for actor_id in choices[target.id]:
        if actor_id not in free:
            continue
        chosen[actor_id] = target
        result = _match_volley(required[1:], choices, free - {actor_id}, chosen)
        if result is not None:
            return result
        chosen.pop(actor_id)
    return None


def allocate_fire(
    positions: Mapping[UUID, Position],
    hp: Mapping[UUID, int],
    targets: Sequence[UnitView],
    rocks: Container[Position],
    committed: Mapping[str, int] | None = None,
) -> dict[UUID, UnitView]:
    """Choose lethal UUID allocations before spending spare arrows on damage."""
    remaining = [
        target.model_copy(
            update={"hp": target.hp - (committed or {}).get(str(target.id), 0)}
        )
        for target in targets
        if target.hp > (committed or {}).get(str(target.id), 0)
    ]
    # Enumerate target subsets, never the full army. Matching scarce rays
    # before flexible shooters avoids greedy assignments losing a second kill.
    best: dict[UUID, UnitView] = {}
    best_score = (-1, -1)
    flexibility = {
        key: sum(line_of_fire(cell, target.position, rocks) for target in remaining)
        for key, cell in positions.items()
    }
    ordered_ids = sorted(
        positions, key=lambda actor_id: (-hp[actor_id], flexibility[actor_id], actor_id)
    )
    choices = {
        target.id: tuple(
            actor_id
            for actor_id in ordered_ids
            if line_of_fire(positions[actor_id], target.position, rocks)
        )
        for target in remaining
    }
    for count in range(len(remaining), 0, -1):
        for subset in combinations(remaining, count):
            required = sorted(
                (target for target in subset for _ in range(target.hp)),
                key=lambda target: (
                    sum(
                        line_of_fire(position, target.position, rocks)
                        for position in positions.values()
                    ),
                    target.hp,
                    target.id,
                ),
            )

            if len(required) > len(positions):
                continue
            result = _match_volley(required, choices, frozenset(positions), {})
            score = (count, sum(hp[key] for key in result or {}))
            if result is not None and score > best_score:
                best, best_score = result, score
        if best:
            break
    allocated = Counter(target.id for target in best.values())
    for actor_id in sorted(
        set(positions) - best.keys(), key=lambda key: (-hp[key], key)
    ):
        candidates = [
            target
            for target in remaining
            if allocated[target.id] < target.hp
            and line_of_fire(positions[actor_id], target.position, rocks)
        ]
        if candidates:
            target = min(
                candidates, key=lambda enemy: (enemy.hp - allocated[enemy.id], enemy.id)
            )
            best[actor_id] = target
            allocated[target.id] += 1
    originals = {target.id: target for target in targets}
    return {key: originals[target.id] for key, target in best.items()}


@dataclass(frozen=True, slots=True)
class Exchange:
    viable: bool
    rounds: int
    kills: int
    losses: int
    reason: str = ""


def exchange(
    positions: Mapping[UUID, Position],
    hp: Mapping[UUID, int],
    targets: Sequence[UnitView],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
    rounds: int,
    *,
    front: Mapping[UUID, tuple[Position, int]] | None = None,
    first_on_front: bool = False,
    confirmed_kills: int = 0,
    confirmed_losses: int = 0,
    committed: Mapping[str, int] | None = None,
    deadline: float | None = None,
) -> Exchange:
    """Do not subsidize a losing local fire group with unrelated kills."""
    all_positions = dict(positions) | {
        key: cell for key, (cell, _hp) in (front or {}).items()
    }
    shots = allocate_fire(positions, hp, targets, rocks)
    groups = [
        (members, opponents)
        for members, opponents in fire_groups(
            all_positions, enemies, rocks, shots=shots
        )
        if members & positions.keys()
    ]
    if len(groups) <= 1:
        return _exchange(
            positions,
            hp,
            targets,
            enemies,
            rocks,
            rounds,
            front=front,
            first_on_front=first_on_front,
            confirmed_kills=confirmed_kills,
            confirmed_losses=confirmed_losses,
            committed=committed,
            deadline=deadline,
        )
    results = []
    covered: set[UUID] = set()
    for members, opponents in groups:
        opponent_ids = {enemy.id for enemy in opponents}
        local_targets = tuple(target for target in targets if target.id in opponent_ids)
        if not local_targets:
            return Exchange(False, 0, 0, 0, "exchange_unfavorable")
        covered.update(target.id for target in local_targets)
        result = _exchange(
            {key: cell for key, cell in positions.items() if key in members},
            {key: value for key, value in hp.items() if key in members},
            local_targets,
            enemies,
            rocks,
            rounds,
            front={
                key: value for key, value in (front or {}).items() if key in members
            },
            first_on_front=first_on_front,
            committed=committed,
            deadline=deadline,
        )
        if not result.viable:
            return result
        results.append(result)
    if (
        covered != {target.id for target in targets}
        or sum(result.losses for result in results) + confirmed_losses
        > len(targets) + confirmed_kills
    ):
        return Exchange(False, rounds, 0, 0, "exchange_unfavorable")
    return Exchange(
        True,
        max(result.rounds for result in results),
        len(targets),
        sum(result.losses for result in results),
    )


def _exchange(
    positions: Mapping[UUID, Position],
    hp: Mapping[UUID, int],
    targets: Sequence[UnitView],
    enemies: Sequence[UnitView],
    rocks: Container[Position],
    rounds: int,
    *,
    front: Mapping[UUID, tuple[Position, int]] | None = None,
    first_on_front: bool = False,
    confirmed_kills: int = 0,
    confirmed_losses: int = 0,
    committed: Mapping[str, int] | None = None,
    deadline: float | None = None,
) -> Exchange:
    """Bounded simultaneous volleys with one arrow per hostile UUID per Tick.

    No HP is removed for entry here. Third parties only gain a moved firing
    ray on a subsequent Tick, and kills never erase this Tick's retaliation.
    """
    if not targets or not positions or rounds <= 0:
        return Exchange(False, 0, 0, 0, "budget_exhausted")
    allies = tuple(sorted(positions))
    fronts = tuple(sorted(key for key in front or {} if key not in positions))
    cells = tuple(positions[key] for key in allies) + tuple(
        (front or {})[key][0] for key in fronts
    )
    initial_hp = tuple(hp[key] for key in allies) + tuple(
        (front or {})[key][1] for key in fronts
    )
    initial_targets = tuple(target.hp for target in targets)
    target_indexes = {target.id: index for index, target in enumerate(targets)}
    states = {(initial_hp, initial_targets)}
    worst_losses = 0
    for round_index in range(rounds):
        following = set()
        for health, target_hp in states:
            if deadline is not None and monotonic() >= deadline:
                return Exchange(
                    False, round_index, 0, worst_losses, "search_budget_exhausted"
                )
            live_positions = {
                key: positions[key]
                for index, key in enumerate(allies)
                if health[index] > 0
            }
            live_hp = {
                key: health[index]
                for index, key in enumerate(allies)
                if health[index] > 0
            }
            live_targets = [
                target.model_copy(update={"hp": target_hp[index]})
                for index, target in enumerate(targets)
                if target_hp[index] > 0
            ]
            allocation = allocate_fire(
                live_positions,
                live_hp,
                live_targets,
                rocks,
                committed if round_index == 0 else None,
            )
            arrows = Counter(target.id for target in allocation.values())
            if round_index == 0:
                for target in live_targets:
                    arrows[target.id] += (committed or {}).get(str(target.id), 0)
            next_targets = tuple(
                max(0, value - arrows[targets[index].id])
                for index, value in enumerate(target_hp)
            )
            retaliation = {health}
            for enemy in {enemy.id: enemy for enemy in enemies}.values():
                index = target_indexes.get(enemy.id)
                if index is not None and target_hp[index] <= 0:
                    continue
                origins = {enemy.position}
                # Targets stay on their qualified ray. A third party may
                # MOVE instead of firing on an earlier Tick, never both.
                if index is None:
                    for _ in range(round_index):
                        origins |= {
                            neighbor
                            for origin in tuple(origins)
                            for neighbor in adjacent_positions(origin)
                            if neighbor not in rocks
                        }
                if enemy.unit_type is UnitType.VANGUARD:
                    if any(
                        health[i] > 0
                        and any(manhattan(origin, cell) == 1 for origin in origins)
                        for i, cell in enumerate(cells)
                    ):
                        # A sweep can hit several neighbors simultaneously;
                        # leave that contact to the mother's melee policy.
                        return Exchange(
                            False, round_index, 0, worst_losses, "vanguard_risk"
                        )
                    continue
                if enemy.unit_type is not UnitType.RANGER:
                    continue
                victims = [
                    i
                    for i, cell in enumerate(cells)
                    if health[i] > 0
                    and (not first_on_front or round_index != 0 or i >= len(allies))
                    and any(line_of_fire(origin, cell, rocks) for origin in origins)
                ]
                if not victims:
                    continue
                retaliation = {
                    tuple(
                        max(0, value - int(i == victim))
                        for i, value in enumerate(current)
                    )
                    for current in retaliation
                    for victim in victims
                }
                if len(retaliation) > PROJECTION_STATE_LIMIT:
                    return Exchange(
                        False, round_index, 0, worst_losses, "search_budget_exhausted"
                    )
            for survivors in retaliation:
                losses = sum(value == 0 for value in survivors)
                worst_losses = max(worst_losses, losses)
                kills = sum(value == 0 for value in next_targets)
                if losses + confirmed_losses > kills + confirmed_kills and not any(
                    next_targets
                ):
                    return Exchange(
                        False,
                        round_index + 1,
                        kills,
                        worst_losses,
                        "exchange_unfavorable",
                    )
                if any(next_targets):
                    if not any(survivors[: len(allies)]):
                        return Exchange(
                            False,
                            round_index + 1,
                            kills,
                            worst_losses,
                            "exchange_unfavorable",
                        )
                    following.add((survivors, next_targets))
        if not following:
            return Exchange(True, round_index + 1, len(targets), worst_losses)
        if len(following) > PROJECTION_STATE_LIMIT:
            return Exchange(
                False, round_index + 1, 0, worst_losses, "search_budget_exhausted"
            )
        states = following
    return Exchange(False, rounds, 0, worst_losses, "exchange_unfavorable")


def choose(
    turn: Turn,
    state: State,
    parent: Parent,
    candidates: Sequence[Ranger],
    front_candidates: Sequence[Ranger],
    targets: Sequence[UnitView],
    enemies: Sequence[UnitView],
    observers: Sequence[CoreView | UnitView],
    rocks: Container[Position],
    blocked: Container[Position],
    budget: SearchBudget,
    *,
    allowed: Callable[[Ranger, Position], bool],
    known: Callable[[Position], bool],
) -> Operation | None:
    """Produce a complete proposal; callers own all mutations and reservations."""
    contact = state.contacts.get(group_key(parent.kind, targets))
    if contact is None or contact.seen != turn.tick or contact.ticks < STABLE_TICKS:
        budget.reject("contact_unstable")
        return None
    if (
        state.operation is not None
        or turn.core is None
        or budget.groups >= GROUP_LIMIT
        or budget.exhausted()
    ):
        budget.reject(
            "reserved_by_other_task" if state.operation else "search_budget_exhausted"
        )
        return None
    budget.groups += 1
    if any(target.unit_type is not UnitType.RANGER for target in targets):
        budget.reject("target_type")
        return None
    front = mutual_front(targets, front_candidates, rocks)
    if not front:
        budget.reject("observer_lost")
        return None
    # Prefer a low-cost front, without forbidding a sustainable exposed
    # observer. Its actual health and retaliation remain in both projections.
    possible_front = tuple(ranger for ranger in front if ranger.hp == 2)
    front = ()
    uncovered = {target.id for target in targets}
    candidate_ids = {ranger.id for ranger in candidates}
    while uncovered:
        coverage = {
            ranger.id: {
                target.id
                for target in targets
                if target.id in uncovered
                and can_see(ranger.position, target.position, 5, rocks)
                and can_see(target.position, ranger.position, 5, rocks)
            }
            for ranger in possible_front
        }
        useful = [ranger for ranger in possible_front if coverage[ranger.id]]
        if not useful:
            break
        observer = min(
            useful,
            key=lambda ranger: (
                any(can_attack(enemy, ranger.position, rocks) for enemy in enemies),
                ranger.id in candidate_ids,
                -len(coverage[ranger.id]),
                ranger.id,
            ),
        )
        front = (*front, observer)
        uncovered -= coverage[observer.id]
    if not mutual_front(targets, front, rocks):
        budget.reject("front_unsafe")
        return None
    front_ids = {ranger.id for ranger in front}
    candidates = tuple(
        sorted(
            (
                ranger
                for ranger in candidates
                if ranger.hp == 2 and ranger.id not in front_ids
            ),
            key=lambda ranger: (
                min(manhattan(ranger.position, target.position) for target in targets),
                ranger.id,
            ),
        )
    )
    if parent.fixed_ids:
        candidates = tuple(
            ranger for ranger in candidates if ranger.id in parent.fixed_ids
        )
        if {ranger.id for ranger in candidates} != parent.fixed_ids:
            budget.reject("reserved_by_other_task")
            return None
    if len(candidates) < min(parent.party_sizes, default=len(targets)):
        budget.reject("insufficient_flank_shooters")
        return None
    # Existing current shots win when they can already finish the targets.
    current = precision_volley(
        {ranger.id: ranger.position for ranger in (*front_candidates, *candidates)},
        targets,
        turn.core.position,
        rocks,
    )
    if {target.id for target in current.values()} == {target.id for target in targets}:
        budget.reject("existing_volley")
        return None
    prefire = prefire_cells(targets, front, rocks) | state.prefire_cells.keys()
    firing = []
    for cell in sorted(
        {cell for target in targets for cell in firing_positions(target.position)}
    ):
        reason = cell_rejection(cell, targets, observers, enemies, rocks, prefire)
        if reason is not None:
            budget.reject(reason)
            continue
        if cell in blocked or not known(cell):
            budget.reject("terrain_unknown" if not known(cell) else "terrain_blocked")
            continue
        valid_targets = [
            target for target in targets if line_of_fire(cell, target.position, rocks)
        ]
        # Unknown intermediate terrain cannot certify a firing ray.
        if any(
            not all(
                known(
                    (
                        cell[0]
                        + (target.position[0] - cell[0])
                        // max(
                            abs(target.position[0] - cell[0]),
                            abs(target.position[1] - cell[1]),
                        )
                        * step,
                        cell[1]
                        + (target.position[1] - cell[1])
                        // max(
                            abs(target.position[0] - cell[0]),
                            abs(target.position[1] - cell[1]),
                        )
                        * step,
                    )
                )
                for step in range(
                    1,
                    max(
                        abs(target.position[0] - cell[0]),
                        abs(target.position[1] - cell[1]),
                    ),
                )
            )
            for target in valid_targets
        ):
            budget.reject("terrain_unknown")
            continue
        if valid_targets:
            firing.append(cell)
    routes: dict[UUID, dict[Position, tuple[Position, ...]]] = {}
    options: list[tuple[int, UUID, Position, Position]] = []
    for ranger in candidates:
        if budget.exhausted() or budget.paths >= PATH_LIMIT:
            budget.reject("search_budget_exhausted")
            break
        budget.paths += 1
        wanted = {
            stage
            for cell in firing
            if allowed(ranger, cell)
            for stage in adjacent_positions(cell)
            if stage not in blocked
            and known(stage)
            and stage not in prefire
            and allowed(ranger, stage)
            and not any(can_attack(enemy, stage, rocks) for enemy in enemies)
        }
        if not wanted:
            continue
        found = {ranger.position: ()}
        frontier = deque([ranger.position])
        while frontier and len(found) < 2048 and monotonic() < budget.deadline:
            if wanted <= found.keys():
                break
            cell = frontier.popleft()
            route = found[cell]
            if len(route) >= parent.max_steps - 1:
                continue
            for neighbor in adjacent_positions(cell):
                if (
                    neighbor in found
                    or neighbor in blocked
                    or not known(neighbor)
                    or neighbor in prefire
                    or not allowed(ranger, neighbor)
                    or any(
                        can_attack(enemy, neighbor, rocks)
                        or (
                            enemy.unit_type is UnitType.VANGUARD
                            and any(
                                manhattan(origin, neighbor) == 1
                                for origin in adjacent_positions(enemy.position)
                                if origin not in rocks
                            )
                        )
                        for enemy in enemies
                    )
                ):
                    continue
                found[neighbor] = (*route, neighbor)
                frontier.append(neighbor)
        routes[ranger.id] = found
        per_actor = []
        for cell in firing:
            if not allowed(ranger, cell):
                continue
            for stage in adjacent_positions(cell):
                route = found.get(stage)
                if (
                    route is None
                    or stage in blocked
                    or any(can_attack(enemy, stage, rocks) for enemy in enemies)
                    or stage in prefire
                ):
                    continue
                ticks = len(route) + 1
                if (
                    turn.tick + ticks >= parent.assembly_deadline
                    or turn.tick + ticks + parent.fire_ticks > parent.deadline
                ):
                    continue
                per_actor.append((len(route), ranger.id, stage, cell))
        options.extend(sorted(per_actor)[:16])
    if not options:
        budget.reject("no_safe_route")
        return None
    occupants: dict[Position, set[UUID]] = {}
    for actor in (*turn.units, turn.core):
        occupants.setdefault(actor.position, set()).add(actor.id)
    beam: list[tuple[tuple[int, UUID, Position, Position], ...]] = [()]
    by_id = {ranger.id: ranger for ranger in candidates}
    best_proposal: Operation | None = None
    best_score: tuple[int, ...] | None = None
    for size in range(1, max(parent.party_sizes, default=0) + 1):
        extended = set()
        for group in beam:
            if budget.exhausted():
                break
            budget.extensions += 1
            ids = {item[1] for item in group}
            # Bound expansions across the whole Tick, not independently per
            # Ranger. Retain alternative pairs rather than a lone best mover.
            for option in sorted(
                options, key=lambda item: (item[1] in ids, item[0], item)
            ):
                if option[1] in ids:
                    continue
                if budget.exhausted():
                    break
                proposed = tuple(sorted((*group, option)))
                proposed_ids = ids | {option[1]}
                stages, cells = (
                    Counter(item[2] for item in proposed),
                    Counter(item[3] for item in proposed),
                )
                if stages.keys() & cells.keys() or any(
                    count + len(occupants.get(cell, set()) - proposed_ids) > 2
                    for counts in (stages, cells)
                    for cell, count in counts.items()
                ):
                    continue
                extended.add(proposed)
            if budget.exhausted():
                break
        beam = sorted(
            extended,
            key=lambda group: (
                max(item[0] for item in group),
                sum(item[0] for item in group),
                group,
            ),
        )[:BEAM_WIDTH]
        if size not in parent.party_sizes or size < len(targets):
            continue
        for group in beam:
            positions = {actor_id: cell for _, actor_id, _, cell in group}
            health = {key: by_id[key].hp for key in positions}
            if parent.fixed_ids and positions.keys() != parent.fixed_ids:
                continue
            if parent.paired and not paired_cells(tuple(positions.values())):
                continue
            allocation = allocate_fire(positions, health, targets, rocks)
            hits = Counter(target.id for target in allocation.values())
            if not any(hits[target.id] >= target.hp for target in targets):
                budget.reject("no_opening_kill")
                continue
            if parent.kill_all and any(
                hits[target.id] < target.hp for target in targets
            ):
                budget.reject("no_complete_volley")
                continue
            projections = [
                exchange(
                    positions,
                    health,
                    targets,
                    enemies,
                    rocks,
                    parent.fire_ticks,
                    front={ranger.id: (ranger.position, ranger.hp) for ranger in front},
                    first_on_front=expected,
                    deadline=budget.deadline,
                )
                for expected in (True, False)
            ]
            if not all(result.viable for result in projections):
                budget.reject(projections[-1].reason or "exchange_unfavorable")
                continue
            member_ids = tuple(str(key) for key in sorted(positions))
            score = (
                sum(hits[target.id] >= target.hp for target in targets),
                -max(result.losses for result in projections),
                -max(result.rounds for result in projections),
                -max(item[0] for item in group),
                -sum(item[0] for item in group),
                -len(group),
            )
            if best_score is not None and score <= best_score:
                continue
            best_score = score
            best_proposal = Operation(
                parent=parent,
                core_id=str(turn.core.id),
                core_position=turn.core.position,
                target_ids=tuple(
                    str(target.id)
                    for target in sorted(targets, key=lambda target: target.id)
                ),
                target_positions={
                    str(target.id): target.position for target in targets
                },
                member_ids=member_ids,
                front_ids=tuple(str(ranger.id) for ranger in front),
                origins={key: by_id[UUID(key)].position for key in member_ids},
                slots={
                    str(actor_id): AssaultSlot(stage, cell)
                    for _, actor_id, stage, cell in group
                },
                routes={
                    str(actor_id): routes[actor_id][stage]
                    for _, actor_id, stage, _ in group
                },
                stable_ticks=contact.ticks,
                created_tick=turn.tick,
                stable_since=contact.since,
                route_validated_tick=turn.tick,
                last_seen=turn.tick,
                actual_hp={key: by_id[UUID(key)].hp for key in member_ids},
                opening_allocation={
                    str(key): str(target.id) for key, target in allocation.items()
                },
                friendly_ids=frozenset(str(actor.id) for actor in turn.units),
                worker_ids=frozenset(str(worker.id) for worker in turn.workers),
            )
        if budget.exhausted():
            break
    if best_proposal is not None:
        return best_proposal
    budget.reject(
        "search_budget_exhausted" if budget.exhausted() else "no_complete_formation"
    )
    return None
