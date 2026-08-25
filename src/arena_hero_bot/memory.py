"""Persistent observations that are safe to carry across complete Turns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arena_hero import CoreView, Position, Turn, UnitView

SCHEMA_VERSION = 1
POSITION_HISTORY_LIMIT = 8
ENEMY_POSITION_HISTORY_LIMIT = 4
ENEMY_DRIFT_MAX_GAP = 3
CONTESTED_CELL_TTL = 80
RESOURCE_MEMORY_TTL = 65536
RESOURCE_MEMORY_LIMIT = 2048
RESOURCE_ABSENCE_RADIUS = 1
RESOURCE_ABSENCE_COOLDOWN = 512
_ABSENCE_OFFSETS = tuple(
    (dx, dy)
    for dx in range(-RESOURCE_ABSENCE_RADIUS, RESOURCE_ABSENCE_RADIUS + 1)
    for dy in range(-RESOURCE_ABSENCE_RADIUS, RESOURCE_ABSENCE_RADIUS + 1)
    if abs(dx) + abs(dy) <= RESOURCE_ABSENCE_RADIUS
)


@dataclass(frozen=True, slots=True)
class EnemySighting:
    """The last authoritative observation of one enemy object."""

    object_id: str
    kind: str
    position: Position
    hp: int
    tick: int
    unit_type: str | None = None
    owner_username: str | None = None
    shield: int | None = None

    @classmethod
    def from_view(cls, enemy: CoreView | UnitView, tick: int) -> EnemySighting:
        """Create a sighting without inventing hidden ownership data."""

        if isinstance(enemy, CoreView):
            return cls(
                object_id=str(enemy.id),
                kind="CORE",
                position=enemy.position,
                hp=enemy.hp,
                shield=enemy.shield,
                owner_username=enemy.owner_username,
                tick=tick,
            )
        return cls(
            object_id=str(enemy.id),
            kind="UNIT",
            position=enemy.position,
            hp=enemy.hp,
            unit_type=enemy.unit_type.value,
            tick=tick,
        )


@dataclass(frozen=True, slots=True)
class UnitGoal:
    """A stable exploration goal used to avoid per-Tick direction churn."""

    position: Position
    assigned_tick: int
    purpose: str
    last_progress_position: Position | None = None


@dataclass(slots=True)
class WorldMemory:
    """Durable obstacle, enemy, movement, and exploration observations."""

    obstacles: set[Position] = field(default_factory=set)
    enemies: dict[str, EnemySighting] = field(default_factory=dict)
    enemy_position_history: dict[str, list[tuple[int, Position]]] = field(
        default_factory=dict
    )
    position_history: dict[str, list[Position]] = field(default_factory=dict)
    goals: dict[str, UnitGoal] = field(default_factory=dict)
    pending_move_targets: dict[str, Position] = field(default_factory=dict)
    contested_positions: dict[Position, int] = field(default_factory=dict)
    resource_cells: dict[Position, int] = field(default_factory=dict)
    resource_absences: dict[Position, int] = field(default_factory=dict)
    last_tick: int = 0

    def observe(self, turn: Turn) -> None:
        """Apply one complete authoritative Turn to persistent memory."""

        self.last_tick = turn.tick
        self.obstacles.update(turn.obstacle_cells)
        self._observe_resource_cells(turn)
        destroyed_enemy_ids = {
            str(event.target_id)
            for event in turn.events
            if event.target_id is not None
            and event.event_type
            in {
                "DESTRUCTION_PARTICIPATION",
                "UNIT_DESTROYED",
                "CORE_DESTROYED",
            }
        }
        for enemy_id in destroyed_enemy_ids:
            self.enemies.pop(enemy_id, None)
            self.enemy_position_history.pop(enemy_id, None)
        for event in turn.events:
            actor_id = str(event.actor_id) if event.actor_id is not None else None
            if actor_id is None or event.event_type == "UNIT_MOVE_SUCCEEDED":
                if actor_id is not None:
                    self.pending_move_targets.pop(actor_id, None)
                continue
            if event.event_type != "UNIT_MOVE_FAILED":
                continue
            attempted = self.pending_move_targets.pop(actor_id, None)
            if event.reason_code == "MOVE_CONTESTED" and attempted is not None:
                self.contested_positions[attempted] = turn.tick
        self.contested_positions = {
            position: observed_tick
            for position, observed_tick in self.contested_positions.items()
            if turn.tick - observed_tick <= CONTESTED_CELL_TTL
        }
        for enemy in turn.visible_enemies:
            enemy_id = str(enemy.id)
            self.enemies[enemy_id] = EnemySighting.from_view(enemy, turn.tick)
            history = self.enemy_position_history.setdefault(enemy_id, [])
            if not history or history[-1][0] != turn.tick:
                history.append((turn.tick, enemy.position))
                del history[:-ENEMY_POSITION_HISTORY_LIMIT]

        active_ids = {str(unit.id) for unit in turn.units}
        if turn.core is not None:
            active_ids.add(str(turn.core.id))
        for unit in turn.units:
            unit_id = str(unit.id)
            history = self.position_history.setdefault(unit_id, [])
            if not history or history[-1] != unit.position:
                history.append(unit.position)
                del history[:-POSITION_HISTORY_LIMIT]
        if turn.core is not None:
            core_id = str(turn.core.id)
            history = self.position_history.setdefault(core_id, [])
            if not history or history[-1] != turn.core.position:
                history.append(turn.core.position)
                del history[:-POSITION_HISTORY_LIMIT]

        for unit_id in set(self.position_history) - active_ids:
            self.position_history.pop(unit_id, None)
            self.goals.pop(unit_id, None)
            self.pending_move_targets.pop(unit_id, None)

    def _closely_observed_cells(self, turn: Turn) -> set[Position]:
        """Return the cells close enough this Tick to trust an absence."""

        observers = [unit.position for unit in turn.units]
        if turn.core is not None:
            observers.append(turn.core.position)
        return {(x + dx, y + dy) for x, y in observers for dx, dy in _ABSENCE_OFFSETS}

    def _observe_resource_cells(self, turn: Turn) -> None:
        """Track known harvest sites and when each was last found empty.

        A cell drops out of ``turn.resource_cells`` as soon as no unit of ours
        stands close enough to see it, and nothing else recorded it, so a Core
        that has mined out its own neighbourhood had no way back to the cells
        its Workers walked past minutes earlier.  Over 43,196 recorded Ticks
        this Core found 1,960 distinct cells and 1,030 of them - 52.6% - were
        already beyond the local harvest radius when first seen, which is
        exactly the population that used to vanish.

        An empty cell is not a dead cell.  Measuring the recording by how long
        ago a cell was last confirmed empty from close range, and asking how
        often the next close look found a resource there again, gives 0.03%
        within 24 Ticks, 0.60% by 120, 1.35% by 512, then a plateau: 1.79% out
        to 2,048 Ticks and 1.63% beyond.  So an absence is never allowed to
        delete a site, only to rest it for ``RESOURCE_ABSENCE_COOLDOWN`` Ticks,
        which is where that curve flattens.  Flicker costs nothing, because a
        cell we are standing next to is in the live pool anyway.

        That plateau also sets ``RESOURCE_MEMORY_TTL``.  Past roughly 512 Ticks
        a sighting's age carries no signal - a site last seen 8,000 Ticks ago is
        as likely to hold a resource as one seen 600 Ticks ago - so expiring
        old sites only shrinks the candidate pool.  What earns the walk is that
        1.8%: only 13% of the cells around this Core have ever held a resource,
        so checking a remembered site beats sweeping an unknown one by about
        7x.  ``RESOURCE_MEMORY_LIMIT`` alone keeps the pool bounded, set above
        the 1,960 distinct cells found in 43,196 Ticks so that a Core which has
        exhausted its neighbourhood keeps every candidate it has ever seen.

        ``RESOURCE_ABSENCE_RADIUS`` stays at one cell for the same reason.
        Believing absence from further out would rest live cells, while missing
        one costs only the trip a Worker was already making: it records the
        absence on arrival.
        """

        for cell in turn.resource_cells:
            self.resource_cells[cell] = turn.tick
            self.resource_absences.pop(cell, None)
        for cell in self._closely_observed_cells(turn):
            if cell in self.resource_cells and cell not in turn.resource_cells:
                self.resource_absences[cell] = turn.tick
        for cell, seen_tick in list(self.resource_cells.items()):
            if turn.tick - seen_tick > RESOURCE_MEMORY_TTL:
                del self.resource_cells[cell]
                self.resource_absences.pop(cell, None)
        if len(self.resource_cells) > RESOURCE_MEMORY_LIMIT:
            self.resource_cells = dict(
                sorted(
                    self.resource_cells.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:RESOURCE_MEMORY_LIMIT]
            )
            self.resource_absences = {
                cell: absent_tick
                for cell, absent_tick in self.resource_absences.items()
                if cell in self.resource_cells
            }

    def remembered_resource_cells(self, tick: int) -> frozenset[Position]:
        """Return known harvest sites worth walking to right now.

        Sites found empty from close range within the cooldown are left out;
        everything else this Core has seen holding a resource is offered, since
        an out-of-sight cell is invisible rather than gone.
        """

        return frozenset(
            cell
            for cell in self.resource_cells
            if tick - self.resource_absences.get(cell, -RESOURCE_ABSENCE_COOLDOWN - 1)
            > RESOURCE_ABSENCE_COOLDOWN
        )

    def recent_positions(self, unit_id: str, limit: int = 4) -> tuple[Position, ...]:
        """Return recent cells with newest first, excluding the current cell."""

        history = self.position_history.get(unit_id, [])
        return tuple(reversed(history[:-1][-limit:]))

    def recent_enemies(self, tick: int, ttl: int) -> tuple[EnemySighting, ...]:
        """Return unexpired sightings, preferring Cores and newer observations."""

        fresh = [enemy for enemy in self.enemies.values() if tick - enemy.tick <= ttl]
        return tuple(
            sorted(
                fresh,
                key=lambda enemy: (
                    enemy.kind != "CORE",
                    -enemy.tick,
                    enemy.object_id,
                ),
            )
        )

    def predicted_enemy_position(
        self,
        enemy_id: str,
        tick: int,
    ) -> Position | None:
        """Predict one cardinal step after two consecutive matching moves."""

        history = self.enemy_position_history.get(enemy_id, [])
        if len(history) < 3:
            return None
        older_tick, older = history[-3]
        previous_tick, previous = history[-2]
        current_tick, current = history[-1]
        if (
            current_tick != tick
            or previous_tick - older_tick != 1
            or current_tick - previous_tick != 1
        ):
            return None
        previous_delta = previous[0] - older[0], previous[1] - older[1]
        current_delta = current[0] - previous[0], current[1] - previous[1]
        if (
            previous_delta != current_delta
            or abs(current_delta[0]) + abs(current_delta[1]) != 1
        ):
            return None
        return current[0] + current_delta[0], current[1] + current_delta[1]

    def enemy_drift_position(
        self,
        enemy_id: str,
        tick: int,
        steps: int = 1,
    ) -> Position | None:
        """Extrapolate a hostile's recent drift ``steps`` Ticks ahead.

        ``predicted_enemy_position`` only fires after three consecutive
        same-delta observations, which almost never happens for an intruder
        that keeps flickering in and out of fleet vision.  Combat resolves on
        the post-movement snapshot, so a chaser that aims at the cell a target
        currently stands in always swings at empty ground.  This looser
        estimate uses the oldest sample still inside ``ENEMY_DRIFT_MAX_GAP``
        and returns a lead cell, which is what interception needs.
        """

        history = self.enemy_position_history.get(enemy_id, [])
        if steps <= 0 or len(history) < 2:
            return None
        current_tick, current = history[-1]
        if current_tick != tick:
            return None
        for older_tick, older in reversed(history[:-1]):
            span = current_tick - older_tick
            if not 1 <= span <= ENEMY_DRIFT_MAX_GAP:
                continue
            delta = current[0] - older[0], current[1] - older[1]
            if delta == (0, 0):
                return None
            return (
                current[0] + _scaled_delta(delta[0], steps, span),
                current[1] + _scaled_delta(delta[1], steps, span),
            )
        return None

    def goal_for(self, unit_id: str) -> UnitGoal | None:
        """Return a Unit's current durable exploration goal."""

        return self.goals.get(unit_id)

    def set_goal(self, unit_id: str, goal: UnitGoal) -> None:
        """Remember a Unit goal."""

        self.goals[unit_id] = goal

    def clear_goal(self, unit_id: str) -> None:
        """Forget a completed or invalid Unit goal."""

        self.goals.pop(unit_id, None)

    def save(self, path: Path) -> None:
        """Atomically save non-secret memory as JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable on-disk representation."""

        return {
            "schema_version": SCHEMA_VERSION,
            "last_tick": self.last_tick,
            "obstacles": sorted([list(position) for position in self.obstacles]),
            "enemies": {
                key: asdict(value) for key, value in sorted(self.enemies.items())
            },
            "enemy_position_history": {
                key: [
                    {"tick": tick, "position": list(position)}
                    for tick, position in value
                ]
                for key, value in sorted(self.enemy_position_history.items())
            },
            "position_history": {
                key: [list(position) for position in value]
                for key, value in sorted(self.position_history.items())
            },
            "goals": {key: asdict(value) for key, value in sorted(self.goals.items())},
            "pending_move_targets": {
                key: list(position)
                for key, position in sorted(self.pending_move_targets.items())
            },
            "contested_positions": [
                {"position": list(position), "tick": observed_tick}
                for position, observed_tick in sorted(self.contested_positions.items())
            ],
            "resource_cells": [
                {"position": list(position), "tick": seen_tick}
                for position, seen_tick in sorted(self.resource_cells.items())
            ],
            "resource_absences": [
                {"position": list(position), "tick": absent_tick}
                for position, absent_tick in sorted(self.resource_absences.items())
            ],
        }

    @classmethod
    def load(cls, path: Path) -> WorldMemory:
        """Load memory, or start clean when no memory file exists."""

        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported memory schema in {path}")
        return cls(
            obstacles={_position(position) for position in raw.get("obstacles", [])},
            enemies={
                key: EnemySighting(
                    object_id=str(value["object_id"]),
                    kind=str(value["kind"]),
                    position=_position(value["position"]),
                    hp=int(value["hp"]),
                    tick=int(value["tick"]),
                    unit_type=_optional_string(value.get("unit_type")),
                    owner_username=_optional_string(value.get("owner_username")),
                    shield=_optional_integer(value.get("shield")),
                )
                for key, value in raw.get("enemies", {}).items()
            },
            enemy_position_history={
                key: [
                    (int(item["tick"]), _position(item["position"])) for item in value
                ]
                for key, value in raw.get("enemy_position_history", {}).items()
            },
            position_history={
                key: [_position(position) for position in value]
                for key, value in raw.get("position_history", {}).items()
            },
            goals={
                key: UnitGoal(
                    position=_position(value["position"]),
                    assigned_tick=int(value["assigned_tick"]),
                    purpose=str(value["purpose"]),
                    last_progress_position=_optional_position(
                        value.get("last_progress_position")
                    ),
                )
                for key, value in raw.get("goals", {}).items()
            },
            pending_move_targets={
                key: _position(position)
                for key, position in raw.get("pending_move_targets", {}).items()
            },
            contested_positions={
                _position(value["position"]): int(value["tick"])
                for value in raw.get("contested_positions", [])
            },
            resource_cells={
                _position(value["position"]): int(value["tick"])
                for value in raw.get("resource_cells", [])
            },
            resource_absences={
                _position(value["position"]): int(value["tick"])
                for value in raw.get("resource_absences", [])
            },
            last_tick=int(raw.get("last_tick", 0)),
        )


def _position(value: object) -> Position:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(type(coordinate) is int for coordinate in value)
    ):
        raise ValueError(f"invalid stored position: {value!r}")
    return int(value[0]), int(value[1])


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"invalid stored string: {value!r}")
    return value


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"invalid stored integer: {value!r}")
    return value


def _optional_position(value: object) -> Position | None:
    if value is None:
        return None
    return _position(value)


def _scaled_delta(delta: int, steps: int, span: int) -> int:
    """Scale one axis of an observed delta, rounding half away from zero."""

    if delta == 0:
        return 0
    sign = 1 if delta > 0 else -1
    return sign * ((abs(delta) * steps + span // 2) // span)
