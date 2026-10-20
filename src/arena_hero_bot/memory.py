"""Persistent observations that are safe to carry across complete Turns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arena_hero import CoreView, Position, Turn, UnitType, UnitView

from .exploration import ExplorationMap, ExplorationRoute, visible_cells

SCHEMA_VERSION = 1
POSITION_HISTORY_LIMIT = 8
ENEMY_POSITION_HISTORY_LIMIT = 4
ENEMY_DRIFT_MAX_GAP = 3
CONTESTED_CELL_TTL = 80
RESOURCE_MEMORY_TTL = 65536
RESOURCE_MEMORY_LIMIT = 2048
WORKER_VISIT_LIMIT = 4096
RESOURCE_ABSENCE_RADIUS = 1
RESOURCE_CLAIM_PURPOSE = "resource-claim-v1"
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


@dataclass(frozen=True, slots=True)
class ExpeditionSquad:
    """A persisted expedition membership, current heading and shared route."""

    serial: int
    members: tuple[str, ...]
    bearing: Position
    pursuit_target_id: str | None = None
    pursuit_position: Position | None = None
    pursuit_direction: Position = (0, 0)
    pursuit_distance: int = 0
    exploration_route: ExplorationRoute = field(default_factory=ExplorationRoute)
    regroup_order: tuple[str, ...] = ()


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
    # Combat identities are durable: a live Unit keeps its assigned role when
    # a newly spawned UUID sorts ahead of it.  Values are ``defense``,
    # ``patrol-1`` .. ``patrol-4``, ``staged`` or ``expedition-N``.
    unit_roles: dict[str, str] = field(default_factory=dict)
    # Exact defensive posts, independent of temporary combat/healing goals.
    # The anchor lets the strategy translate every post together if the Core
    # moves, without reallocating surviving Units to different slots.
    defense_posts: dict[str, Position] = field(default_factory=dict)
    defense_anchor: Position | None = None
    # Legacy memory files predate durable role assignment.  The strategy uses
    # this marker to run one spatial sanity pass, then never reclassifies live
    # UUIDs merely because their positions change.
    unit_roles_initialized: bool = False
    expedition_squads: list[ExpeditionSquad] = field(default_factory=list)
    exploration: ExplorationMap = field(default_factory=ExplorationMap)
    next_expedition_serial: int = 0
    # Normal high-tier production is deliberately throttled.  Keep its
    # payback marker with the rest of the durable strategy observations so a
    # service restart cannot immediately spend the same bank again.
    last_normal_growth_tick: int | None = None
    last_normal_growth_resources: int | None = None
    # Selected once for the new symmetric defensive posture and retained
    # across restarts while the Core walks there.
    core_home_position: Position | None = None
    # The four-block economy is a durable identity assignment, separate from
    # combat roles.  A replacement Worker fills a vacancy in its own region
    # instead of following the nearest live resource across a block boundary.
    worker_blocks: dict[str, int] = field(default_factory=dict)
    # Arrival ticks, not merely sightings, drive the least-recently-visited
    # patrol sweep.  Keeping this separate from ``resource_cells`` preserves
    # confirmed-empty deletion while still restoring local coverage.
    worker_visit_ticks: dict[Position, int] = field(default_factory=dict)
    # The eight patrol Rangers share one clockwise ring.  Slots and phase are
    # durable so a restart does not collapse the formation onto one corner.
    patrol_ring_slots: dict[str, int] = field(default_factory=dict)
    patrol_ring_phase: int = 0
    patrol_ring_origin: Position | None = None
    last_tick: int = 0

    def observe(self, turn: Turn) -> None:
        """Apply one complete authoritative Turn to persistent memory."""

        self.last_tick = turn.tick
        self.obstacles.update(turn.obstacle_cells)
        self.observe_exploration(turn)
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
        for worker in turn.workers:
            self.worker_visit_ticks[worker.position] = turn.tick
        if len(self.worker_visit_ticks) > WORKER_VISIT_LIMIT:
            retained = sorted(
                self.worker_visit_ticks.items(),
                key=lambda item: (item[1], item[0]),
            )[-WORKER_VISIT_LIMIT:]
            self.worker_visit_ticks = dict(retained)
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
        self.defense_posts = {
            unit_id: position
            for unit_id, position in self.defense_posts.items()
            if unit_id in active_ids
        }
        live_worker_ids = {str(worker.id) for worker in turn.workers}
        self.worker_blocks = {
            unit_id: block
            for unit_id, block in self.worker_blocks.items()
            if unit_id in live_worker_ids
        }
        live_combat_ids = {str(unit.id) for unit in (*turn.vanguards, *turn.rangers)}
        self.patrol_ring_slots = {
            unit_id: slot
            for unit_id, slot in self.patrol_ring_slots.items()
            if unit_id in live_combat_ids
        }

    def observe_exploration(self, turn: Turn) -> None:
        """Share authoritative vision, including empty ground, across squads."""

        if self.exploration.last_tick == turn.tick:
            return
        radii = {UnitType.WORKER: 3, UnitType.VANGUARD: 4, UnitType.RANGER: 5}
        observers = [(unit.position, radii[unit.unit_type]) for unit in turn.units]
        if turn.core is not None:
            observers.append((turn.core.position, 5))
        obstacles = self.obstacles | set(turn.obstacle_cells)
        if not self.exploration.tiles and self.position_history:
            # Old memory has no empty-ground atlas. Seed only the short paths
            # actually observed by surviving units, not imagined travel lines
            # or the empty space between distant expeditions.
            cells: set[Position] = set()
            for unit in turn.units:
                for position in self.position_history.get(str(unit.id), ()):
                    cells.update(
                        visible_cells(position, radii[unit.unit_type], obstacles)
                    )
            self.exploration.mark(cells, turn.tick - 1)
        self.exploration.observe(observers, obstacles, turn.tick)

    def _closely_observed_cells(self, turn: Turn) -> set[Position]:
        """Return the cells close enough this Tick to trust an absence."""

        observers = [unit.position for unit in turn.units]
        if turn.core is not None:
            observers.append(turn.core.position)
        return {(x + dx, y + dy) for x, y in observers for dx, dy in _ABSENCE_OFFSETS}

    def _observe_resource_cells(self, turn: Turn) -> None:
        """Remember visible resources and forget cells confirmed empty.

        A cell drops out of ``turn.resource_cells`` as soon as no friendly unit
        stands close enough to see it, so an out-of-sight cell is merely
        unconfirmed rather than gone.  Keep those sightings for
        ``RESOURCE_MEMORY_TTL`` so Workers can finish a trip after the resource
        leaves vision.

        An empty cell seen within ``RESOURCE_ABSENCE_RADIUS`` is different: the
        observation is authoritative.  Delete the coordinate immediately,
        together with any durable goal aimed at it.  The server selects
        replacement cells without any documented advantage for a former
        resource site, so retaining the coordinate only sends Workers on stale
        trips.  If the same cell is later seen holding a resource again, that
        fresh sighting records it anew.
        """

        visible = set(turn.resource_cells)
        for cell in visible:
            self.resource_cells[cell] = turn.tick
        emptied = {
            cell
            for cell in self.resource_cells
            if cell not in visible and cell in self._closely_observed_cells(turn)
        }
        for cell in emptied:
            del self.resource_cells[cell]
        if emptied:
            # Resource memory cannot import strategy purposes without a cycle,
            # so keep the persisted resource-claim purpose here. Other goals
            # (for example exploration) may legitimately share this position.
            self.goals = {
                unit_id: goal
                for unit_id, goal in self.goals.items()
                if not (
                    goal.purpose == RESOURCE_CLAIM_PURPOSE and goal.position in emptied
                )
            }
        for cell, seen_tick in list(self.resource_cells.items()):
            if turn.tick - seen_tick > RESOURCE_MEMORY_TTL:
                del self.resource_cells[cell]
        if len(self.resource_cells) > RESOURCE_MEMORY_LIMIT:
            self.resource_cells = dict(
                sorted(
                    self.resource_cells.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:RESOURCE_MEMORY_LIMIT]
            )

    def remembered_resource_cells(self) -> frozenset[Position]:
        """Return remembered resource cells not confirmed empty."""

        return frozenset(self.resource_cells)

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
        """Predict repeated straight motion or a two-cell cardinal patrol."""

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
        if abs(current_delta[0]) + abs(current_delta[1]) != 1:
            return None
        if previous_delta == current_delta:
            return current[0] + current_delta[0], current[1] + current_delta[1]
        # Four consecutive A-B-A-B observations establish a repeated reversal.
        # Extrapolating the last direction instead aims outside the patrol on
        # every Tick, even after hundreds of missed shots.
        if (
            len(history) >= 4
            and older == current
            and history[-4] == (older_tick - 1, previous)
        ):
            return previous
        return None

    def enemy_drift_position(
        self,
        enemy_id: str,
        tick: int,
        steps: int = 1,
    ) -> Position | None:
        """Extrapolate a hostile's recent drift ``steps`` Ticks ahead.

        ``predicted_enemy_position`` needs consecutive observations of repeated
        motion, which almost never happens for an intruder
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
        if steps == 1 and len(history) >= 3:
            before_tick, before = history[-3]
            previous_tick, previous = history[-2]
            if previous_tick - before_tick == 1 and current_tick - previous_tick == 1:
                first_step = previous[0] - before[0], previous[1] - before[1]
                second_step = current[0] - previous[0], current[1] - previous[1]
                if (
                    first_step != (0, 0)
                    and first_step[0] == -second_step[0]
                    and first_step[1] == -second_step[1]
                ):
                    # A one-cell reversal means the target is oscillating; the
                    # cell it just came from is the only useful lead, while
                    # extrapolating the last direction aims where it turned.
                    return previous
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

    def enemy_recently_moved(self, enemy_id: str, tick: int) -> bool:
        """Whether a hostile's last two observations disagree.

        A fresh first-contact target has no prior observation, so this stays
        ``False`` and the guard may still test the standing cell.  A target
        that moved recently but whose current-tick drift is unavailable (a
        one-tick vision gap is enough) is known to have left some previous
        cell, so the standing cell is no longer a reliable sweep target.
        """

        history = self.enemy_position_history.get(enemy_id, [])
        if not history or history[-1][0] != tick:
            return False
        current = history[-1][1]
        for older_tick, older in reversed(history[:-1]):
            if 1 <= tick - older_tick <= ENEMY_DRIFT_MAX_GAP:
                return older != current
        return False

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
            "unit_roles": dict(sorted(self.unit_roles.items())),
            "defense_posts": {
                unit_id: list(position)
                for unit_id, position in sorted(self.defense_posts.items())
            },
            "defense_anchor": (
                list(self.defense_anchor) if self.defense_anchor is not None else None
            ),
            "unit_roles_initialized": self.unit_roles_initialized,
            "expedition_squads": [
                {
                    "serial": squad.serial,
                    "members": list(squad.members),
                    "bearing": list(squad.bearing),
                    "pursuit_target_id": squad.pursuit_target_id,
                    "pursuit_position": (
                        list(squad.pursuit_position)
                        if squad.pursuit_position is not None
                        else None
                    ),
                    "pursuit_direction": list(squad.pursuit_direction),
                    "pursuit_distance": squad.pursuit_distance,
                    "exploration_route": asdict(squad.exploration_route),
                    "regroup_order": list(squad.regroup_order),
                }
                for squad in self.expedition_squads
            ],
            "exploration": {
                "tiles": [
                    {"position": list(position), "mask": hex(mask), "tick": tick}
                    for position, (mask, tick) in sorted(self.exploration.tiles.items())
                ],
                "last_tick": self.exploration.last_tick,
                "new_cells": self.exploration.new_cells,
            },
            "next_expedition_serial": self.next_expedition_serial,
            "last_normal_growth_tick": self.last_normal_growth_tick,
            "last_normal_growth_resources": self.last_normal_growth_resources,
            "core_home_position": (
                list(self.core_home_position)
                if self.core_home_position is not None
                else None
            ),
            "worker_blocks": dict(sorted(self.worker_blocks.items())),
            "worker_visit_ticks": [
                {"position": list(position), "tick": tick}
                for position, tick in sorted(self.worker_visit_ticks.items())
            ],
            "patrol_ring_slots": dict(sorted(self.patrol_ring_slots.items())),
            "patrol_ring_phase": self.patrol_ring_phase,
            "patrol_ring_origin": (
                list(self.patrol_ring_origin)
                if self.patrol_ring_origin is not None
                else None
            ),
        }

    @classmethod
    def load(cls, path: Path) -> WorldMemory:
        """Load memory, or start clean when no memory file exists."""

        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported memory schema in {path}")
        resource_cells = {
            _position(value["position"]): int(value["tick"])
            for value in raw.get("resource_cells", [])
        }
        goals = {
            key: UnitGoal(
                position=_position(value["position"]),
                assigned_tick=int(value["assigned_tick"]),
                purpose=str(value["purpose"]),
                last_progress_position=_optional_position(
                    value.get("last_progress_position")
                ),
            )
            for key, value in raw.get("goals", {}).items()
        }
        # Older memory files recorded a cooldown for close-confirmed empty
        # cells.  The current policy deletes those coordinates immediately, so
        # do not resurrect them on load, and drop any goal that still points at
        # one of them.
        absent_cells = {
            _position(value["position"]) for value in raw.get("resource_absences", [])
        }
        if absent_cells:
            resource_cells = {
                position: seen_tick
                for position, seen_tick in resource_cells.items()
                if position not in absent_cells
            }
            goals = {
                key: goal
                for key, goal in goals.items()
                if not (
                    goal.purpose == RESOURCE_CLAIM_PURPOSE
                    and goal.position in absent_cells
                )
            }
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
            goals=goals,
            pending_move_targets={
                key: _position(position)
                for key, position in raw.get("pending_move_targets", {}).items()
            },
            contested_positions={
                _position(value["position"]): int(value["tick"])
                for value in raw.get("contested_positions", [])
            },
            resource_cells=resource_cells,
            unit_roles={
                str(unit_id): str(role)
                for unit_id, role in raw.get("unit_roles", {}).items()
            },
            defense_posts={
                str(unit_id): _position(position)
                for unit_id, position in raw.get("defense_posts", {}).items()
            },
            defense_anchor=_optional_position(raw.get("defense_anchor")),
            unit_roles_initialized=bool(raw.get("unit_roles_initialized", False)),
            expedition_squads=[
                ExpeditionSquad(
                    serial=int(value.get("serial", index + 1)),
                    members=tuple(str(member) for member in value.get("members", [])),
                    bearing=_position(value["bearing"]),
                    pursuit_target_id=(
                        str(value["pursuit_target_id"])
                        if value.get("pursuit_target_id") is not None
                        else None
                    ),
                    pursuit_position=(
                        _optional_position(value.get("pursuit_position"))
                    ),
                    pursuit_direction=_position(value.get("pursuit_direction", [0, 0])),
                    pursuit_distance=int(value.get("pursuit_distance", 0)),
                    exploration_route=_exploration_route(
                        value.get("exploration_route", {})
                    ),
                    regroup_order=tuple(
                        str(member) for member in value.get("regroup_order", [])
                    ),
                )
                for index, value in enumerate(raw.get("expedition_squads", []))
            ],
            exploration=ExplorationMap(
                tiles={
                    _position(value["position"]): (
                        int(value["mask"], 16),
                        int(value["tick"]),
                    )
                    for value in raw.get("exploration", {}).get("tiles", [])
                },
                last_tick=int(raw.get("exploration", {}).get("last_tick", -1)),
                new_cells=int(raw.get("exploration", {}).get("new_cells", 0)),
            ),
            next_expedition_serial=int(raw.get("next_expedition_serial", 0)),
            last_normal_growth_tick=_optional_integer(
                raw.get("last_normal_growth_tick")
            ),
            last_normal_growth_resources=_optional_integer(
                raw.get("last_normal_growth_resources")
            ),
            core_home_position=_optional_position(raw.get("core_home_position")),
            worker_blocks={
                str(unit_id): int(block)
                for unit_id, block in raw.get("worker_blocks", {}).items()
            },
            worker_visit_ticks={
                _position(value["position"]): int(value["tick"])
                for value in raw.get("worker_visit_ticks", [])
            },
            patrol_ring_slots={
                str(unit_id): int(slot)
                for unit_id, slot in raw.get("patrol_ring_slots", {}).items()
            },
            patrol_ring_phase=int(raw.get("patrol_ring_phase", 0)),
            patrol_ring_origin=_optional_position(raw.get("patrol_ring_origin")),
            last_tick=int(raw.get("last_tick", 0)),
        )


def _exploration_route(raw: dict[str, Any]) -> ExplorationRoute:
    return ExplorationRoute(
        path=tuple(_position(position) for position in raw.get("path", [])),
        assigned_tick=int(raw.get("assigned_tick", 0)),
        checked_tick=int(raw.get("checked_tick", 0)),
        progress_tick=int(raw.get("progress_tick", 0)),
        remaining=int(raw.get("remaining", 0)),
        expected_gain=float(raw.get("expected_gain", 0.0)),
        cooldowns=tuple(
            (_position(position), int(tick))
            for position, tick in raw.get("cooldowns", [])
        ),
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
