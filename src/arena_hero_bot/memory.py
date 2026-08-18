"""Persistent observations that are safe to carry across complete Turns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arena_hero import CoreView, Position, Turn, UnitView

SCHEMA_VERSION = 1
POSITION_HISTORY_LIMIT = 8
CONTESTED_CELL_TTL = 80


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
    position_history: dict[str, list[Position]] = field(default_factory=dict)
    goals: dict[str, UnitGoal] = field(default_factory=dict)
    pending_move_targets: dict[str, Position] = field(default_factory=dict)
    contested_positions: dict[Position, int] = field(default_factory=dict)
    last_tick: int = 0

    def observe(self, turn: Turn) -> None:
        """Apply one complete authoritative Turn to persistent memory."""

        self.last_tick = turn.tick
        self.obstacles.update(turn.obstacle_cells)
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
            self.enemies[str(enemy.id)] = EnemySighting.from_view(enemy, turn.tick)

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
