"""Aggressive v1 tactic built on the official Arena Hero Turn controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2
from uuid import UUID

from arena_hero import (
    BeaconStatus,
    CoreState,
    CoreView,
    Direction,
    Position,
    Ranger,
    Turn,
    Unit,
    UnitType,
    UnitView,
    Vanguard,
    Worker,
    unit_cost,
)

from .geometry import (
    DIRECTIONS,
    add,
    adjacent_positions,
    direction_between,
    firing_positions,
    line_of_fire,
    manhattan,
    next_step,
)
from .memory import EnemySighting, UnitGoal, WorldMemory
from .models import DecisionReport

EXPLORATION_PURPOSE = "explore-center-v3"
RESOURCE_PATROL_PURPOSE = "resource-patrol-v3"
COMBAT_PATROL_PURPOSE = "combat-patrol-v1"
RESOURCE_PATROL_SPACING = 6
COMBAT_PATROL_SPACING = 12
DEFENSIVE_PERIMETER_MIN_RADIUS = 2
VANGUARD_VISION_RADIUS = 4
RANGER_VISION_RADIUS = 5
CORE_CAPACITY_FAST_EXPANSION = 50
CORE_CAPACITY_MEDIUM_RESERVE = 95
CORE_CAPACITY_HIGH_RESERVE = 150


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Parameters that define the tactic posture.

    ``max_population=None`` enables the live, safety-first growth mode.  A
    positive value keeps the older explicit population cap available for
    experiments and backwards-compatible callers.
    """

    target_workers: int = 2
    max_population: int | None = 12
    resource_target: int = 0
    safety_reserve: int = 10
    resource_patrol_radius: int = 30
    offensive_patrol_radius: int = 60
    offensive_patrol_goal_ttl: int = 160
    offensive_core_memory_ttl: int = 512
    offensive_min_combat_units: int = 8
    enemy_memory_ttl: int = 160
    enemy_core_memory_ttl: int = 4096
    exploration_goal_ttl: int = 80
    exploration_radius: int = 24
    worker_threat_radius: int = 6
    worker_threat_memory_ttl: int = 24
    resource_guard_min_workers: int = 6


@dataclass(slots=True)
class _TurnContext:
    turn: Turn
    report: DecisionReport
    occupied: set[Position]
    enemy_positions: set[Position]
    reserved: set[Position] = field(default_factory=set)
    departures: set[str] = field(default_factory=set)
    resource_assignments: dict[UUID, Position] = field(default_factory=dict)
    defensive_assignments: dict[UUID, Position] | None = None
    remaining_resources: int = 0
    remaining_resource_space: int = 0
    beacon_claimed: bool = False


class AggressiveStrategy:
    """Seek combat early while retaining a minimal resource engine."""

    def __init__(
        self,
        memory: WorldMemory,
        config: StrategyConfig | None = None,
    ) -> None:
        self.memory = memory
        self.config = config or StrategyConfig()

    def decide(self, turn: Turn) -> DecisionReport:
        """Queue one complete aggressive plan for the current Turn."""

        turn.clear()
        self.memory.observe(turn)
        recent_enemies = self.memory.recent_enemies(
            turn.tick, self.config.enemy_memory_ttl
        )
        tactical_enemies = self._tactical_enemies(turn, recent_enemies)
        report = DecisionReport(
            tick=turn.tick,
            visible_enemies=len(turn.visible_enemies),
            remembered_enemies=len(recent_enemies),
        )
        occupied = {unit.position for unit in turn.units}
        occupied.update(enemy.position for enemy in turn.visible_enemies)
        if turn.core is not None:
            occupied.add(turn.core.position)
        context = _TurnContext(
            turn=turn,
            report=report,
            occupied=occupied,
            enemy_positions={enemy.position for enemy in turn.visible_enemies},
            resource_assignments=self._assign_resources(turn),
            remaining_resources=turn.resources,
            remaining_resource_space=turn.resource_space,
        )

        for ranger in sorted(turn.rangers, key=lambda unit: unit.id.bytes):
            self._decide_ranger(
                ranger,
                context,
                tactical_enemies,
                offensive=self._is_offensive_combat_unit(ranger, turn),
            )
        for vanguard in sorted(turn.vanguards, key=lambda unit: unit.id.bytes):
            self._decide_vanguard(
                vanguard,
                context,
                tactical_enemies,
                offensive=self._is_offensive_combat_unit(vanguard, turn),
            )
        core_position = turn.core.position if turn.core is not None else None
        for worker in sorted(
            turn.workers,
            key=lambda unit: self._worker_priority(unit, core_position),
        ):
            self._decide_worker(worker, context)
        self._decide_core(context)
        return report

    def _decide_ranger(
        self,
        ranger: Ranger,
        context: _TurnContext,
        recent_enemies: tuple[EnemySighting, ...],
        *,
        offensive: bool = False,
    ) -> None:
        if self._recover_if_critical(ranger, maximum_hp=2, context=context):
            return
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        shootable = [
            enemy
            for enemy in context.turn.visible_enemies
            if line_of_fire(ranger.position, enemy.position, obstacles)
        ]
        if shootable:
            target = max(
                shootable,
                key=lambda enemy: self._enemy_score(
                    enemy, ranger.position, context.turn
                ),
            )
            ranger.shoot_cell(target.position)
            context.report.add(
                actor_id=str(ranger.id),
                actor_kind="RANGER",
                action="SHOOT",
                reason=f"fire at high-value {self._enemy_label(target)} cell",
                target=target.position,
            )
            return

        if self._pickup_beacon(ranger, context):
            return
        visible_target = self._best_visible_target(ranger.position, context.turn)
        if visible_target is not None:
            goal = self._ranger_approach_goal(ranger, visible_target, context)
            if goal is not None and self._move(
                ranger,
                goal,
                context,
                reason=f"close firing angle on {self._enemy_label(visible_target)}",
            ):
                return

        remembered_enemies = (
            self._offensive_enemies(context.turn) if offensive else recent_enemies
        )
        remembered = self._best_remembered_target(
            ranger.position,
            remembered_enemies,
        )
        if remembered is not None and self._move(
            ranger,
            remembered.position,
            context,
            reason=f"hunt last seen {remembered.kind.lower()}",
        ):
            return

        goal, reason = self._idle_combat_goal(
            ranger,
            context.turn,
            offensive=offensive,
            context=context,
        )
        self._move_or_wait(
            ranger,
            goal,
            context,
            reason=reason,
            wait_at_goal=not offensive and self._preserves_resources(),
        )

    def _decide_vanguard(
        self,
        vanguard: Vanguard,
        context: _TurnContext,
        recent_enemies: tuple[EnemySighting, ...],
        *,
        offensive: bool = False,
    ) -> None:
        if self._recover_if_critical(vanguard, maximum_hp=4, context=context):
            return
        adjacent_groups: dict[Direction, list[CoreView | UnitView]] = {}
        for enemy in context.turn.visible_enemies:
            direction = direction_between(vanguard.position, enemy.position)
            if direction is not None:
                adjacent_groups.setdefault(direction, []).append(enemy)
        if adjacent_groups:
            direction, targets = max(
                adjacent_groups.items(),
                key=lambda item: sum(
                    self._enemy_score(enemy, vanguard.position, context.turn)
                    for enemy in item[1]
                ),
            )
            vanguard.sweep(direction)
            context.report.add(
                actor_id=str(vanguard.id),
                actor_kind="VANGUARD",
                action="SWEEP",
                reason=f"hit {len(targets)} adjacent hostile object(s)",
                target=add(vanguard.position, direction),
            )
            return

        if self._pickup_beacon(vanguard, context):
            return
        visible_target = self._best_visible_target(vanguard.position, context.turn)
        if visible_target is not None:
            candidates = [
                position
                for position in adjacent_positions(visible_target.position)
                if position not in self.memory.obstacles
                and position not in context.enemy_positions
                and (
                    position == vanguard.position
                    or position not in context.occupied | context.reserved
                )
            ]
            if candidates:
                goal = min(
                    candidates,
                    key=lambda position: (
                        manhattan(vanguard.position, position),
                        position,
                    ),
                )
                if self._move(
                    vanguard,
                    goal,
                    context,
                    reason=f"rush {self._enemy_label(visible_target)}",
                ):
                    return

        remembered_enemies = (
            self._offensive_enemies(context.turn) if offensive else recent_enemies
        )
        remembered = self._best_remembered_target(
            vanguard.position,
            remembered_enemies,
        )
        if remembered is not None and self._move(
            vanguard,
            remembered.position,
            context,
            reason=f"hunt last seen {remembered.kind.lower()}",
        ):
            return

        goal, reason = self._idle_combat_goal(
            vanguard,
            context.turn,
            offensive=offensive,
            context=context,
        )
        self._move_or_wait(
            vanguard,
            goal,
            context,
            reason=reason,
            wait_at_goal=not offensive and self._preserves_resources(),
        )

    def _decide_worker(self, worker: Worker, context: _TurnContext) -> None:
        core = context.turn.core
        if core is None:
            self._record_wait(worker, context, "no Core while respawning")
            return

        worker_threats = self._worker_threats(worker, context)
        if worker.cargo > 0 and worker_threats:
            retreat_goal = core.position
            if (
                core.view.state is CoreState.MOVING
                and core.view.destination is not None
            ):
                retreat_goal = core.view.destination
            if self._retreat_worker(
                worker,
                retreat_goal,
                context,
                worker_threats,
            ):
                return
            self._record_wait(
                worker,
                context,
                "no safe path for: retreat carried resources from enemy pressure",
            )
            return

        if worker.cargo > 0:
            self.memory.clear_goal(str(worker.id))
            if core.view.state is CoreState.MOVING:
                destination = core.view.destination
                if (
                    destination is not None
                    and worker.position != destination
                    and destination not in context.occupied | context.reserved
                    and self._move(
                        worker,
                        destination,
                        context,
                        reason=(
                            "stage carried resources at migrating Core destination"
                        ),
                        allow_goal=True,
                    )
                ):
                    return
                self._record_wait(
                    worker,
                    context,
                    "Core migration is in progress; wait before depositing cargo",
                )
                return

            if worker.position == core.position:
                if context.remaining_resource_space > 0:
                    deposited = min(worker.cargo, context.remaining_resource_space)
                    worker.deposit()
                    context.remaining_resources += deposited
                    context.remaining_resource_space -= deposited
                    context.report.add(
                        actor_id=str(worker.id),
                        actor_kind="WORKER",
                        action="DEPOSIT",
                        reason="return combat-economy resources to Core",
                        target=core.position,
                    )
                else:
                    if self._unbounded_growth():
                        departure_goal = self._resource_patrol_goal(
                            worker,
                            context.turn,
                        )
                        if departure_goal != worker.position and self._move(
                            worker,
                            departure_goal,
                            context,
                            reason=(
                                "clear the full Core cell for safe storage expansion"
                            ),
                            allow_goal=True,
                        ):
                            return
                    self._record_wait(worker, context, "Core storage is full")
                return
            allow_core = self._core_has_room_for(worker, context)
            if allow_core and self._move(
                worker,
                core.position,
                context,
                reason="return carried resources to Core",
                allow_goal=True,
            ):
                return
            self._record_wait(worker, context, "Core cell is not currently reachable")
            return

        if worker_threats:
            if not self._retreat_worker(
                worker,
                core.position,
                context,
                worker_threats,
            ):
                self._record_wait(
                    worker,
                    context,
                    "no safe path for: retreat from visible enemy pressure",
                )
            return

        assigned_resource = context.resource_assignments.get(worker.id)
        if worker.position == assigned_resource:
            worker.harvest()
            self.memory.clear_goal(str(worker.id))
            context.report.add(
                actor_id=str(worker.id),
                actor_kind="WORKER",
                action="HARVEST",
                reason="harvest current visible resource",
                target=worker.position,
            )
            return

        if self._pickup_beacon(worker, context):
            return
        if self._recover_if_critical(worker, maximum_hp=2, context=context):
            return

        if assigned_resource is not None and self._move(
            worker,
            assigned_resource,
            context,
            reason="claim nearest unassigned visible resource",
        ):
            self.memory.clear_goal(str(worker.id))
            return

        if self._preserves_resources():
            goal = self._resource_patrol_goal(worker, context.turn)
            reason = "patrol near the stationary Core for resources"
        else:
            goal = self._exploration_goal(worker, context.turn.tick)
            reason = "scout for resources and enemies"
        self._move_or_wait(
            worker,
            goal,
            context,
            reason=reason,
        )

    def _decide_core(self, context: _TurnContext) -> None:
        core = context.turn.core
        if core is None:
            return
        if core.view.state is CoreState.MOVING:
            context.report.add(
                actor_id=str(core.id),
                actor_kind="CORE",
                action="WAIT",
                reason="Core migration is already progressing",
            )
            return

        nearby_enemy = any(
            manhattan(core.position, enemy.position) <= 4
            for enemy in context.turn.visible_enemies
        )
        if core.hp <= 2 and context.remaining_resources > 0:
            core.heal()
            context.report.add(
                actor_id=str(core.id),
                actor_kind="CORE",
                action="HEAL",
                reason="prevent imminent Core destruction",
                target=core.position,
            )
            return
        if nearby_enemy and core.shield <= 1 and context.remaining_resources > 0:
            core.repair_shield()
            context.report.add(
                actor_id=str(core.id),
                actor_kind="CORE",
                action="REPAIR_SHIELD",
                reason="repair critical shield under local pressure",
                target=core.position,
            )
            return
        if (
            not context.beacon_claimed
            and core.position == context.turn.beacon.position
            and context.turn.beacon.status is BeaconStatus.GROUND
        ):
            core.pickup_beacon()
            context.beacon_claimed = True
            context.report.add(
                actor_id=str(core.id),
                actor_kind="CORE",
                action="PICKUP_BEACON",
                reason="secure Champion Beacon with Core",
                target=core.position,
            )
            return

        if self._core_can_spawn(context):
            unit_type = self._choose_spawn(context.turn, context.remaining_resources)
            if unit_type is not None:
                core.spawn(unit_type)
                context.report.add(
                    actor_id=str(core.id),
                    actor_kind="CORE",
                    action="SPAWN",
                    reason=self._spawn_reason(unit_type),
                    target=core.position,
                )
                return

        preserving_resources = self._preserves_resources()
        if not preserving_resources and core.hp < 5 and context.remaining_resources > 0:
            core.heal()
            context.report.add(
                actor_id=str(core.id),
                actor_kind="CORE",
                action="HEAL",
                reason="restore Core HP between engagements",
                target=core.position,
            )
            return
        if (
            not preserving_resources
            and core.shield < 5
            and context.remaining_resources > 0
        ):
            core.repair_shield()
            context.report.add(
                actor_id=str(core.id),
                actor_kind="CORE",
                action="REPAIR_SHIELD",
                reason="restore shield between engagements",
                target=core.position,
            )
            return

        if (
            not preserving_resources
            and not nearby_enemy
            and context.turn.workers
            and all(worker.cargo == 0 for worker in context.turn.workers)
        ):
            direction = self._core_migration_direction(context)
            if direction is not None:
                core.start_move(direction)
                context.report.add(
                    actor_id=str(core.id),
                    actor_kind="CORE",
                    action="START_MOVE",
                    reason="advance mobile base toward the resource-rich center",
                    target=add(core.position, direction),
                )

    def _pickup_beacon(self, unit: Unit, context: _TurnContext) -> bool:
        if (
            context.beacon_claimed
            or unit.position != context.turn.beacon.position
            or context.turn.beacon.status is not BeaconStatus.GROUND
        ):
            return False
        unit.pickup_beacon()
        context.beacon_claimed = True
        context.report.add(
            actor_id=str(unit.id),
            actor_kind=unit.unit_type.value,
            action="PICKUP_BEACON",
            reason="secure Champion Beacon",
            target=unit.position,
        )
        return True

    def _heal_if_critical(
        self,
        unit: Unit,
        *,
        maximum_hp: int,
        context: _TurnContext,
    ) -> bool:
        core = context.turn.core
        missing_hp = maximum_hp - unit.hp
        if (
            core is None
            or core.view.state is not CoreState.NORMAL
            or unit.position != core.position
            or unit.hp > maximum_hp // 2
            or missing_hp <= 0
            or context.remaining_resources < missing_hp
        ):
            return False
        unit.heal()
        context.remaining_resources -= missing_hp
        context.report.add(
            actor_id=str(unit.id),
            actor_kind=unit.unit_type.value,
            action="HEAL",
            reason="recover critical combat asset before redeployment",
            target=unit.position,
        )
        return True

    def _recover_if_critical(
        self,
        unit: Unit,
        *,
        maximum_hp: int,
        context: _TurnContext,
    ) -> bool:
        if unit.hp > maximum_hp // 2:
            return False
        if self._heal_if_critical(unit, maximum_hp=maximum_hp, context=context):
            return True

        core = context.turn.core
        if core is None:
            return False
        reason = "return critical unit to Core for healing"
        if unit.position == core.position:
            self._record_wait(unit, context, "wait at Core for healing resources")
            return True
        if self._core_has_room_for(unit, context) and self._move(
            unit,
            core.position,
            context,
            reason=reason,
            allow_goal=True,
        ):
            return True

        staging_cells = [
            position
            for position in adjacent_positions(core.position)
            if position not in self.memory.obstacles
            and position not in context.turn.obstacle_cells
            and position not in context.occupied | context.reserved
        ]
        if staging_cells:
            goal = min(
                staging_cells,
                key=lambda position: (manhattan(unit.position, position), position),
            )
            if self._move(unit, goal, context, reason=reason):
                return True
        self._record_wait(unit, context, f"no safe path for: {reason}")
        return True

    def _move_or_wait(
        self,
        unit: Unit,
        goal: Position,
        context: _TurnContext,
        *,
        reason: str,
        allow_goal: bool = False,
        wait_at_goal: bool = False,
    ) -> None:
        if unit.position == goal:
            self._record_wait(unit, context, reason)
            if wait_at_goal:
                unit.wait()
            return
        if not self._move(
            unit,
            goal,
            context,
            reason=reason,
            allow_goal=allow_goal,
        ):
            self._record_wait(unit, context, f"no safe path for: {reason}")
            if wait_at_goal and unit.position == goal:
                unit.wait()

    def _retreat_worker(
        self,
        worker: Worker,
        core_position: Position,
        context: _TurnContext,
        threats: tuple[Position, ...],
    ) -> bool:
        """Choose a one-step retreat that never needlessly closes on an enemy."""

        blocked = set(self.memory.obstacles)
        blocked.update(self.memory.contested_positions)
        blocked.update(context.turn.obstacle_cells)
        blocked.update(context.occupied)
        blocked.update(context.reserved)
        blocked.discard(worker.position)
        blocked.discard(core_position)
        threat_positions = set(threats)
        directions = (
            DIRECTIONS[self._direction_offset(worker.id) :]
            + DIRECTIONS[: self._direction_offset(worker.id)]
        )
        recent_history = self.memory.recent_positions(str(worker.id))
        recent = set(recent_history)
        candidates: list[tuple[int, int, int, int, int, Direction]] = []
        for order, direction in enumerate(directions):
            destination = add(worker.position, direction)
            if (
                destination in blocked
                or destination in context.enemy_positions
                or destination in threat_positions
            ):
                continue
            nearest_enemy = min(manhattan(destination, enemy) for enemy in threats)
            direct_away = int(
                any(
                    (
                        worker.position[0] == enemy[0] == destination[0]
                        and abs(destination[1] - enemy[1])
                        > abs(worker.position[1] - enemy[1])
                    )
                    or (
                        worker.position[1] == enemy[1] == destination[1]
                        and abs(destination[0] - enemy[0])
                        > abs(worker.position[0] - enemy[0])
                    )
                    for enemy in threats
                )
            )
            candidates.append(
                (
                    nearest_enemy,
                    direct_away,
                    int(destination not in recent),
                    -manhattan(destination, core_position),
                    -order,
                    direction,
                )
            )

        if len(recent_history) >= 2 and recent_history[1] == worker.position:
            previous = recent_history[0]
            non_looping = [
                candidate
                for candidate in candidates
                if add(worker.position, candidate[-1]) != previous
            ]
            if non_looping:
                candidates = non_looping

        if candidates:
            _, _, _, _, _, direction = max(candidates)
            return self._queue_move(
                worker,
                direction,
                context,
                reason="retreat from visible enemy pressure",
                target=core_position,
            )
        return self._move(
            worker,
            core_position,
            context,
            reason="retreat from visible enemy pressure",
            allow_goal=True,
        )

    def _worker_threats(
        self,
        worker: Worker,
        context: _TurnContext,
    ) -> tuple[Position, ...]:
        visible = [
            enemy.position
            for enemy in context.turn.visible_enemies
            if (
                isinstance(enemy, CoreView)
                or (
                    isinstance(enemy, UnitView)
                    and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                )
            )
            and manhattan(worker.position, enemy.position)
            <= self.config.worker_threat_radius
        ]
        remembered = [
            enemy.position
            for enemy in self.memory.recent_enemies(
                context.turn.tick,
                self.config.enemy_core_memory_ttl,
            )
            if (
                enemy.kind == "CORE"
                or (
                    enemy.kind == "UNIT"
                    and enemy.unit_type
                    in {UnitType.VANGUARD.value, UnitType.RANGER.value}
                    and context.turn.tick - enemy.tick
                    <= self.config.worker_threat_memory_ttl
                )
            )
            and manhattan(worker.position, enemy.position)
            <= self.config.worker_threat_radius
        ]
        contested = [
            position
            for position in self.memory.contested_positions
            if manhattan(worker.position, position) <= self.config.worker_threat_radius
        ]
        return tuple(dict.fromkeys((*visible, *remembered, *contested)))

    def _move(
        self,
        unit: Unit,
        goal: Position,
        context: _TurnContext,
        *,
        reason: str,
        allow_goal: bool = False,
    ) -> bool:
        if unit.position == goal:
            return False
        blocked = set(self.memory.obstacles)
        blocked.update(self.memory.contested_positions)
        blocked.update(context.turn.obstacle_cells)
        blocked.update(context.occupied)
        blocked.update(context.reserved)
        if isinstance(unit, Worker):
            blocked.update(self._worker_threat_exclusion_cells(context.turn))
        direction = next_step(
            unit.position,
            goal,
            blocked=blocked,
            recent=self.memory.recent_positions(str(unit.id)),
            direction_offset=self._direction_offset(unit.id),
            allow_goal=allow_goal,
        )
        if direction is None:
            return False
        destination = add(unit.position, direction)
        if destination in context.reserved:
            return False
        return self._queue_move(
            unit,
            direction,
            context,
            reason=reason,
            target=goal,
        )

    def _queue_move(
        self,
        unit: Unit,
        direction: Direction,
        context: _TurnContext,
        *,
        reason: str,
        target: Position,
    ) -> bool:
        destination = add(unit.position, direction)
        if destination in context.reserved:
            return False
        unit.move(direction)
        self.memory.pending_move_targets[str(unit.id)] = destination
        context.reserved.add(destination)
        # A queued move vacates the unit's current cell for the rest of this
        # plan.  Keeping every origin in ``occupied`` made a dense defensive
        # cluster behave like a solid wall: no guard could step through the
        # cells that another guard was leaving, and nearby Workers could stay
        # trapped indefinitely.  The Core itself remains occupied even when
        # a unit departs from its cell.
        if context.turn.core is None or unit.position != context.turn.core.position:
            context.occupied.discard(unit.position)
        if (
            context.turn.core is not None
            and unit.position == context.turn.core.position
        ):
            context.departures.add(str(unit.id))
        context.report.add(
            actor_id=str(unit.id),
            actor_kind=unit.unit_type.value,
            action="MOVE",
            reason=reason,
            target=target,
        )
        return True

    def _record_wait(self, unit: Unit, context: _TurnContext, reason: str) -> None:
        context.report.add(
            actor_id=str(unit.id),
            actor_kind=unit.unit_type.value,
            action="WAIT",
            reason=reason,
            target=unit.position,
        )

    def _best_visible_target(
        self, origin: Position, turn: Turn
    ) -> CoreView | UnitView | None:
        enemies = turn.visible_enemies
        if not enemies:
            return None
        target = max(
            enemies,
            key=lambda enemy: self._enemy_score(enemy, origin, turn),
        )
        if target is None:
            return None
        return target

    def _best_remembered_target(
        self,
        origin: Position,
        enemies: tuple[EnemySighting, ...],
    ) -> EnemySighting | None:
        if not enemies:
            return None
        return max(
            enemies,
            key=lambda enemy: (
                500 if enemy.kind == "CORE" else 100,
                -manhattan(origin, enemy.position),
                enemy.tick,
            ),
        )

    def _tactical_enemies(
        self,
        turn: Turn,
        enemies: tuple[EnemySighting, ...],
    ) -> tuple[EnemySighting, ...]:
        """Keep goal-oriented defenders near the Core instead of roaming away."""

        if not self._preserves_resources() or turn.core is None:
            return enemies
        return tuple(
            enemy
            for enemy in enemies
            if manhattan(turn.core.position, enemy.position)
            <= self.config.worker_threat_radius * 2
        )

    def _is_offensive_combat_unit(
        self,
        unit: Ranger | Vanguard,
        turn: Turn,
    ) -> bool:
        """Assign a stable minority of combat units to the roaming squad."""

        if not self._offensive_patrol_enabled(turn):
            return False
        # UUID assignment keeps roles stable across ticks without adding
        # private state.  With the live roster this sends roughly one third
        # of Rangers/Vanguards away while leaving a substantial Core guard.
        return unit.id.int % 3 == 0

    def _offensive_patrol_enabled(self, turn: Turn) -> bool:
        """Allow roaming combat only while the economy can absorb the risk."""

        core = turn.core
        return (
            self._unbounded_growth()
            and core is not None
            and len(turn.vanguards) + len(turn.rangers)
            >= self.config.offensive_min_combat_units
            and turn.resources >= self._spawn_safety_reserve(turn) + 5
            and core.hp >= 5
            and core.shield >= 5
        )

    def _offensive_enemies(self, turn: Turn) -> tuple[EnemySighting, ...]:
        """Return recent targets suitable for a roaming squad to pursue."""

        if not self._offensive_patrol_enabled(turn):
            return ()
        enemies = self.memory.recent_enemies(
            turn.tick,
            self.config.enemy_core_memory_ttl,
        )
        return tuple(
            enemy
            for enemy in enemies
            if (
                enemy.kind == "CORE"
                and turn.tick - enemy.tick <= self.config.offensive_core_memory_ttl
            )
            or (
                enemy.kind == "UNIT"
                and turn.tick - enemy.tick <= self.config.enemy_memory_ttl
            )
        )

    def _idle_combat_goal(
        self,
        unit: Ranger | Vanguard,
        turn: Turn,
        *,
        offensive: bool = False,
        context: _TurnContext | None = None,
    ) -> tuple[Position, str]:
        if offensive and self._offensive_patrol_enabled(turn):
            return self._combat_patrol_goal(unit, turn)
        if not self._preserves_resources() or turn.core is None:
            return turn.beacon.position, "advance toward the public Beacon battle zone"
        if context is None:
            # Keep this helper useful for callers that only have a Turn.  The
            # normal decision path passes the context so assignments are
            # calculated once and can account for same-Tick reservations.
            context = _TurnContext(
                turn=turn,
                report=DecisionReport(tick=turn.tick),
                occupied={item.position for item in turn.units},
                enemy_positions={item.position for item in turn.visible_enemies},
            )
        if context.defensive_assignments is None:
            context.defensive_assignments = self._defensive_perimeter_assignments(
                context
            )
        return (
            context.defensive_assignments.get(unit.id, unit.position),
            "hold a defensive perimeter around the resource Core",
        )

    def _defensive_perimeter_assignments(
        self,
        context: _TurnContext,
    ) -> dict[UUID, Position]:
        """Place non-roaming combat units on one obstacle-aware vision ring.

        The game uses Manhattan vision, so the most useful discrete circle is
        the set of cells at one Manhattan distance from the Core.  A ring at
        radius ``r`` contains ``8r`` cells.  Each guard covers approximately
        ``2 * vision`` consecutive ring cells; starting from the largest
        radius supported by the roster and shrinking only when obstacles make
        a gap un-coverable gives a stable, count-and-vision-derived perimeter.
        """

        core = context.turn.core
        if core is None:
            return {}
        core_position = core.position
        guards = tuple(
            sorted(
                (
                    unit
                    for unit in (*context.turn.rangers, *context.turn.vanguards)
                    if not self._is_offensive_combat_unit(unit, context.turn)
                ),
                key=lambda unit: unit.id.bytes,
            )
        )
        if not guards:
            return {}

        obstacles = set(self.memory.obstacles)
        obstacles.update(context.turn.obstacle_cells)
        guard_ids = {unit.id for unit in guards}
        unavailable = set(context.enemy_positions) | set(context.reserved)
        unavailable.add(core.position)
        unavailable.update(
            unit.position for unit in context.turn.units if unit.id not in guard_ids
        )
        visions = {unit.id: _combat_vision_radius(unit) for unit in guards}
        base_radius = max(
            DEFENSIVE_PERIMETER_MIN_RADIUS,
            sum(visions.values()) // 4,
        )

        def ring_for(radius: int) -> tuple[Position, ...]:
            return tuple(
                (core_position[0] + dx, core_position[1] + dy)
                for dx, dy in _defensive_ring_offsets(radius)
                if (core_position[0] + dx, core_position[1] + dy) not in obstacles
                and _clear_manhattan_path(
                    core_position,
                    (core_position[0] + dx, core_position[1] + dy),
                    obstacles,
                )
            )

        # A Worker or a recently observed hostile can temporarily occupy a
        # ring cell.  Expand the ring until every guard still has a unique
        # legal slot, while keeping the count/vision radius as the primary
        # choice.
        max_radius = base_radius
        search_limit = base_radius + max(8, len(guards) * 2)
        while max_radius < search_limit:
            available = [
                position
                for position in ring_for(max_radius)
                if position not in unavailable
            ]
            if len(available) >= len(guards):
                break
            max_radius += 1

        best: dict[UUID, Position] = {}
        for radius in range(max_radius, DEFENSIVE_PERIMETER_MIN_RADIUS - 1, -1):
            coverage_cells = ring_for(radius)
            available = [
                position for position in coverage_cells if position not in unavailable
            ]
            if len(available) < len(guards):
                continue
            assignments = _assign_defensive_ring_slots(available, guards, visions)
            if not best:
                best = assignments
            if _ring_is_covered(
                coverage_cells,
                assignments,
                visions,
                obstacles,
            ):
                return assignments
        return best

    def _combat_patrol_goal(
        self,
        unit: Ranger | Vanguard,
        turn: Turn,
    ) -> tuple[Position, str]:
        """Choose a stable outward search point for one roaming combat unit."""

        core = turn.core
        if core is None:
            return unit.position, "no Core available for offensive patrol"
        unit_id = str(unit.id)
        current = self.memory.goal_for(unit_id)
        claimed_positions = {
            goal.position
            for other in (*turn.rangers, *turn.vanguards)
            if other.id != unit.id
            if (goal := self.memory.goal_for(str(other.id))) is not None
            and goal.purpose == COMBAT_PATROL_PURPOSE
        }
        if (
            current is not None
            and current.purpose == COMBAT_PATROL_PURPOSE
            and unit.position != current.position
            and current.position not in self.memory.obstacles
            and current.position not in claimed_positions
            and manhattan(core.position, current.position)
            <= self.config.offensive_patrol_radius * 2
            and turn.tick - current.assigned_tick
            <= self.config.offensive_patrol_goal_ttl
        ):
            return current.position, "search outward for enemy units and Cores"

        offsets = _resource_patrol_offsets(
            self.config.offensive_patrol_radius,
            COMBAT_PATROL_SPACING,
        )
        combat_units = sorted(
            (*turn.rangers, *turn.vanguards),
            key=lambda item: item.id.bytes,
        )
        unit_index = next(
            index for index, item in enumerate(combat_units) if item.id == unit.id
        )
        phase = turn.tick // self.config.offensive_patrol_goal_ttl
        unit_spacing = max(1, len(offsets) // len(combat_units))
        offset_index = (unit_index * unit_spacing + phase) % len(offsets)
        if current is not None and current.purpose == COMBAT_PATROL_PURPOSE:
            current_offset = (
                current.position[0] - core.position[0],
                current.position[1] - core.position[1],
            )
            if current_offset in offsets:
                offset_index = (offsets.index(current_offset) + unit_spacing) % len(
                    offsets
                )

        patrol_position = core.position
        for step in range(len(offsets)):
            dx, dy = offsets[(offset_index + step) % len(offsets)]
            candidate = core.position[0] + dx, core.position[1] + dy
            if (
                candidate != unit.position
                and candidate not in self.memory.obstacles
                and candidate not in claimed_positions
            ):
                patrol_position = candidate
                break
        goal = UnitGoal(
            position=patrol_position,
            assigned_tick=turn.tick,
            purpose=COMBAT_PATROL_PURPOSE,
            last_progress_position=unit.position,
        )
        self.memory.set_goal(unit_id, goal)
        return patrol_position, "search outward for enemy units and Cores"

    def _enemy_score(
        self,
        enemy: CoreView | UnitView,
        origin: Position,
        turn: Turn,
    ) -> int:
        if isinstance(enemy, CoreView):
            score = 600 + (5 - enemy.hp) * 45 + (5 - min(enemy.shield, 5)) * 10
        else:
            base = {
                UnitType.WORKER: 220,
                UnitType.RANGER: 200,
                UnitType.VANGUARD: 170,
            }[enemy.unit_type]
            maximum_hp = {
                UnitType.WORKER: 2,
                UnitType.RANGER: 2,
                UnitType.VANGUARD: 4,
            }[enemy.unit_type]
            score = base + (maximum_hp - enemy.hp) * 35
        if turn.core is not None:
            score += max(0, 8 - manhattan(turn.core.position, enemy.position)) * 12
        return score - manhattan(origin, enemy.position) * 3

    def _ranger_approach_goal(
        self,
        ranger: Ranger,
        target: CoreView | UnitView,
        context: _TurnContext,
    ) -> Position | None:
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        candidates = [
            position
            for position in firing_positions(target.position)
            if position not in obstacles
            and position not in context.enemy_positions
            and (
                position == ranger.position
                or position not in context.occupied | context.reserved
            )
            and line_of_fire(position, target.position, obstacles)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda position: (manhattan(ranger.position, position), position),
        )

    def _assign_resources(self, turn: Turn) -> dict[UUID, Position]:
        """Assign every visible resource to the nearest empty Worker."""

        workers = {worker.id: worker for worker in turn.workers if worker.cargo == 0}
        resources = set(turn.resource_cells) - {
            enemy.position for enemy in turn.visible_enemies
        }
        assignments: dict[UUID, Position] = {}
        while workers and resources:
            _, worker_id, resource = min(
                (
                    manhattan(worker.position, resource),
                    worker.id,
                    resource,
                )
                for worker in workers.values()
                for resource in resources
            )
            assignments[worker_id] = resource
            workers.pop(worker_id)
            resources.remove(resource)
        return assignments

    def _core_migration_direction(self, context: _TurnContext) -> Direction | None:
        """Choose one currently legal-looking Core step toward the origin."""

        core = context.turn.core
        if core is None or core.position == (0, 0):
            return None
        blocked = set(self.memory.obstacles)
        blocked.update(context.turn.obstacle_cells)
        blocked.update(context.turn.resource_cells)
        blocked.update(context.occupied)
        blocked.update(context.reserved)
        return next_step(
            core.position,
            (0, 0),
            blocked=blocked,
            recent=self.memory.recent_positions(str(core.id)),
            direction_offset=self._direction_offset(core.id),
        )

    def _exploration_goal(self, worker: Worker, tick: int) -> Position:
        unit_id = str(worker.id)
        current = self.memory.goal_for(unit_id)
        if (
            current is not None
            and current.purpose == EXPLORATION_PURPOSE
            and worker.position != current.position
            and current.position not in self.memory.obstacles
        ):
            if current.last_progress_position is None:
                current = UnitGoal(
                    position=current.position,
                    assigned_tick=current.assigned_tick,
                    purpose=current.purpose,
                    last_progress_position=worker.position,
                )
                self.memory.set_goal(unit_id, current)
            age = tick - current.assigned_tick
            if age <= self.config.exploration_goal_ttl:
                return current.position

            progress_reference = current.last_progress_position
            if progress_reference is not None and manhattan(
                worker.position, current.position
            ) < manhattan(progress_reference, current.position):
                renewed = UnitGoal(
                    position=current.position,
                    assigned_tick=tick,
                    purpose=current.purpose,
                    last_progress_position=worker.position,
                )
                self.memory.set_goal(unit_id, renewed)
                return renewed.position

        phase = tick // self.config.exploration_goal_ttl
        radius = self.config.exploration_radius + (phase % 3) * 6
        position = _inward_goal(worker.position, radius)
        if position == worker.position:
            vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
            dx, dy = vectors[(worker.id.int + phase) % len(vectors)]
            position = dx * radius, dy * radius
        goal = UnitGoal(
            position=position,
            assigned_tick=tick,
            purpose=EXPLORATION_PURPOSE,
            last_progress_position=worker.position,
        )
        self.memory.set_goal(unit_id, goal)
        return goal.position

    def _resource_patrol_goal(self, worker: Worker, turn: Turn) -> Position:
        """Assign a stable patrol point close enough for efficient deposits."""

        core = turn.core
        if core is None:
            return worker.position
        unit_id = str(worker.id)
        current = self.memory.goal_for(unit_id)
        claimed_positions = {
            goal.position
            for other in turn.workers
            if other.id != worker.id
            if (goal := self.memory.goal_for(str(other.id))) is not None
            and goal.purpose == RESOURCE_PATROL_PURPOSE
        }
        if (
            current is not None
            and current.purpose == RESOURCE_PATROL_PURPOSE
            and worker.position != current.position
            and current.position not in self.memory.obstacles
            and current.position not in claimed_positions
            and not self._near_remembered_worker_danger(current.position, turn)
            and manhattan(core.position, current.position)
            <= self.config.resource_patrol_radius * 2
            and turn.tick - current.assigned_tick <= self.config.exploration_goal_ttl
        ):
            return current.position

        phase = turn.tick // self.config.exploration_goal_ttl
        offsets = _resource_patrol_offsets(
            self.config.resource_patrol_radius,
            RESOURCE_PATROL_SPACING,
        )
        ordered_workers = sorted(turn.workers, key=lambda item: item.id.bytes)
        worker_index = next(
            index for index, item in enumerate(ordered_workers) if item.id == worker.id
        )
        worker_spacing = max(1, len(offsets) // len(ordered_workers))
        offset_index = (worker_index * worker_spacing + phase) % len(offsets)
        if current is not None and current.purpose == RESOURCE_PATROL_PURPOSE:
            current_offset = (
                current.position[0] - core.position[0],
                current.position[1] - core.position[1],
            )
            if current_offset in offsets:
                offset_index = (offsets.index(current_offset) + 1) % len(offsets)
        patrol_position = core.position
        for step in range(len(offsets)):
            dx, dy = offsets[(offset_index + step) % len(offsets)]
            candidate = core.position[0] + dx, core.position[1] + dy
            if (
                candidate != worker.position
                and candidate not in claimed_positions
                and not self._near_remembered_worker_danger(candidate, turn)
            ):
                patrol_position = candidate
                break
        goal = UnitGoal(
            position=patrol_position,
            assigned_tick=turn.tick,
            purpose=RESOURCE_PATROL_PURPOSE,
            last_progress_position=worker.position,
        )
        self.memory.set_goal(unit_id, goal)
        return goal.position

    def _remembered_worker_danger_positions(
        self,
        turn: Turn,
    ) -> tuple[Position, ...]:
        return tuple(
            enemy.position
            for enemy in self.memory.recent_enemies(
                turn.tick,
                self.config.enemy_core_memory_ttl,
            )
            if enemy.kind == "CORE"
            or (
                enemy.kind == "UNIT"
                and enemy.unit_type in {UnitType.VANGUARD.value, UnitType.RANGER.value}
                and turn.tick - enemy.tick <= self.config.worker_threat_memory_ttl
            )
        )

    def _near_remembered_worker_danger(
        self,
        position: Position,
        turn: Turn,
    ) -> bool:
        return any(
            manhattan(position, enemy) <= self.config.worker_threat_radius
            for enemy in self._remembered_worker_danger_positions(turn)
        )

    def _worker_threat_exclusion_cells(self, turn: Turn) -> set[Position]:
        radius = self.config.worker_threat_radius
        return {
            (center[0] + dx, center[1] + dy)
            for center in self._remembered_worker_danger_positions(turn)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if abs(dx) + abs(dy) <= radius
        }

    def _core_has_room_for(self, worker: Unit, context: _TurnContext) -> bool:
        core = context.turn.core
        if core is None:
            return False
        if core.position in context.reserved:
            return False
        occupants = [
            unit
            for unit in context.turn.units
            if unit.position == core.position
            and unit.id != worker.id
            and str(unit.id) not in context.departures
        ]
        return not occupants

    def _core_can_spawn(self, context: _TurnContext) -> bool:
        core = context.turn.core
        growth_target = self._growth_population_target()
        if (
            core is None
            or (
                growth_target is not None
                and context.turn.state.population >= growth_target
            )
            or (
                self.config.resource_target > 0
                and context.remaining_resources >= self.config.resource_target
            )
        ):
            return False
        if core.position in context.reserved:
            return False
        occupants = [
            unit
            for unit in context.turn.units
            if unit.position == core.position and str(unit.id) not in context.departures
        ]
        return not occupants

    def _choose_spawn(self, turn: Turn, resources: int) -> UnitType | None:
        growth_target = self._growth_population_target()
        if growth_target is not None and turn.state.population >= growth_target:
            return None
        if self._needs_resource_guard(turn):
            candidates = (UnitType.VANGUARD,)
        elif len(turn.workers) < self.config.target_workers:
            candidates = (UnitType.WORKER,)
        elif not turn.vanguards:
            candidates = (UnitType.VANGUARD, UnitType.RANGER)
        elif len(turn.rangers) < len(turn.vanguards) * 2:
            candidates = (UnitType.RANGER, UnitType.VANGUARD)
        else:
            candidates = (UnitType.VANGUARD, UnitType.RANGER)
        return next(
            (
                unit_type
                for unit_type in candidates
                if unit_cost(unit_type, turn.state.population)
                + self._spawn_safety_reserve(turn)
                <= resources
            ),
            None,
        )

    def _needs_resource_guard(self, turn: Turn) -> bool:
        if not self._preserves_resources() or turn.core is None or turn.vanguards:
            return False
        # Establish a first combat guard before the economy reaches its full
        # worker target.  A resource Core can be rushed long before all
        # workers are online; waiting for a remembered enemy leaves the Core
        # undefended during that vulnerable growth window.
        guard_threshold = min(
            self.config.target_workers,
            self.config.resource_guard_min_workers,
        )
        if len(turn.workers) == guard_threshold:
            return True
        local_radius = (
            self.config.resource_patrol_radius + self.config.worker_threat_radius
        )
        return any(
            (
                enemy.kind == "CORE"
                or (
                    enemy.unit_type in {UnitType.VANGUARD.value, UnitType.RANGER.value}
                    and turn.tick - enemy.tick <= self.config.enemy_memory_ttl
                )
            )
            and manhattan(turn.core.position, enemy.position) <= local_radius
            for enemy in self.memory.recent_enemies(
                turn.tick,
                self.config.enemy_core_memory_ttl,
            )
        )

    def _growth_population_target(self) -> int | None:
        if self.config.resource_target <= 0:
            return self.config.max_population
        if self.config.resource_target <= 10:
            if self.config.max_population is None:
                return 1
            return min(1, self.config.max_population)
        required = (self.config.resource_target + 4) // 5
        if self.config.max_population is None:
            return required + 1
        return min(required + 1, self.config.max_population)

    def _preserves_resources(self) -> bool:
        """Return whether this tactic should prioritize a safe Core reserve."""

        return self.config.resource_target > 0 or self.config.max_population is None

    def _spawn_safety_reserve(self, turn: Turn) -> int:
        """Keep enough Core resources for a full emergency recovery cycle.

        The reserve is only added by the unbounded live mode. Explicit legacy
        resource-target configurations retain their historical exact-cost
        behavior so callers can reproduce prior experiments. In the live mode,
        the stockpile follows the Core capacity tiers used by the shop: small
        Cores expand quickly, then preserve 50, 95, or 150 resources as storage
        grows. The base reserve and any missing recovery resources always win.
        """

        if not self._unbounded_growth():
            return 0
        core = turn.core
        if core is None:
            return max(0, self.config.safety_reserve)
        missing_recovery = max(0, 5 - core.hp) + max(0, 5 - core.shield)
        return max(
            0,
            self.config.safety_reserve,
            missing_recovery,
            self._stockpile_target(turn),
        )

    def _stockpile_target(self, turn: Turn) -> int:
        """Return the shop-aligned live stockpile target for Core capacity."""

        capacity = turn.resource_capacity
        if capacity < CORE_CAPACITY_FAST_EXPANSION:
            return 0
        if capacity < CORE_CAPACITY_MEDIUM_RESERVE:
            return 50
        if capacity < CORE_CAPACITY_HIGH_RESERVE:
            return 95
        return 150

    def _unbounded_growth(self) -> bool:
        """Return whether the live strategy has no fixed population/stockpile goal."""

        return self.config.max_population is None and self.config.resource_target <= 0

    def _spawn_reason(self, unit_type: UnitType) -> str:
        if self._unbounded_growth():
            return (
                "expand storage while preserving the Core safety reserve "
                f"with {unit_type.value.lower()}"
            )
        if self.config.resource_target > 0:
            return (
                f"expand capacity toward {self.config.resource_target} Core resources "
                f"with {unit_type.value.lower()}"
            )
        return f"expand aggressive {unit_type.value.lower()} roster"

    @staticmethod
    def _direction_offset(unit_id: UUID) -> int:
        return unit_id.int % 4

    @staticmethod
    def _worker_priority(
        worker: Worker, core_position: Position | None
    ) -> tuple[bool, bytes]:
        vacates_core = worker.position == core_position and worker.cargo == 0
        return not vacates_core, worker.id.bytes

    @staticmethod
    def _enemy_label(enemy: CoreView | UnitView) -> str:
        if isinstance(enemy, CoreView):
            return f"Core @{enemy.owner_username}"
        return enemy.unit_type.value


def _inward_goal(position: Position, distance: int) -> Position:
    """Move each coordinate toward the resource-rich world origin."""

    def inward(coordinate: int) -> int:
        if coordinate > 0:
            return max(0, coordinate - distance)
        return min(0, coordinate + distance)

    return inward(position[0]), inward(position[1])


def _resource_patrol_offsets(radius: int, spacing: int) -> tuple[Position, ...]:
    """Return a deterministic sweep route with no gaps in Worker vision."""

    coordinates = list(range(-radius, radius + 1, spacing))
    if coordinates[-1] != radius:
        coordinates.append(radius)
    route: list[Position] = []
    for row, dy in enumerate(coordinates):
        columns = coordinates if row % 2 == 0 else reversed(coordinates)
        route.extend((dx, dy) for dx in columns if (dx, dy) != (0, 0))
    return tuple(route)


def _combat_vision_radius(unit: Ranger | Vanguard) -> int:
    """Return the authoritative Manhattan vision radius for a combat unit."""

    if isinstance(unit, Ranger):
        return RANGER_VISION_RADIUS
    return VANGUARD_VISION_RADIUS


def _defensive_ring_offsets(radius: int) -> tuple[Position, ...]:
    """Return a deterministic Manhattan-distance ring in angular order."""

    offsets = [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if abs(dx) + abs(dy) == radius
    ]
    return tuple(sorted(offsets, key=lambda position: atan2(position[1], position[0])))


def _assign_defensive_ring_slots(
    cells: list[Position],
    guards: tuple[Ranger | Vanguard, ...],
    visions: dict[UUID, int],
) -> dict[UUID, Position]:
    """Divide ring cells into vision-sized sectors and pick each sector's center."""

    ordered = tuple(
        sorted(
            guards,
            key=lambda unit: (-visions[unit.id], unit.id.bytes),
        )
    )
    total_capacity = sum(2 * visions[unit.id] for unit in ordered)
    assignments: dict[UUID, Position] = {}
    cursor = 0
    cumulative_capacity = 0
    for index, unit in enumerate(ordered):
        cumulative_capacity += 2 * visions[unit.id]
        remaining_units = len(ordered) - index - 1
        target_end = (
            len(cells) * cumulative_capacity + total_capacity // 2
        ) // total_capacity
        end = max(cursor + 1, target_end)
        end = min(end, len(cells) - remaining_units)
        center = cursor + (end - cursor - 1) // 2
        assignments[unit.id] = cells[center]
        cursor = end
    return assignments


def _ring_is_covered(
    cells: tuple[Position, ...],
    assignments: dict[UUID, Position],
    visions: dict[UUID, int],
    obstacles: set[Position],
) -> bool:
    """Return whether every traversable ring cell is visible from a guard."""

    return all(
        any(
            manhattan(position, slot) <= visions[unit_id]
            and _clear_manhattan_path(slot, position, obstacles)
            for unit_id, slot in assignments.items()
        )
        for position in cells
    )


def _clear_manhattan_path(
    origin: Position,
    target: Position,
    obstacles: set[Position] | frozenset[Position],
) -> bool:
    """Return whether a shortest Manhattan path avoids known obstacles.

    Visibility is evaluated over shortest cardinal paths.  If one such path
    remains open, an obstacle does not hide the target; if every shortest path
    is cut, the target is treated as being behind the obstacle.  This mirrors
    the map rule that an obstacle cell is visible but cells behind it are not,
    while allowing paths around a single isolated obstacle.
    """

    if origin == target:
        return True
    if target in obstacles:
        return False
    frontier = {origin}
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    for _ in range(abs(dx) + abs(dy)):
        next_frontier: set[Position] = set()
        for position in frontier:
            if position[0] != target[0]:
                candidate = (position[0] + step_x, position[1])
                if candidate not in obstacles:
                    next_frontier.add(candidate)
            if position[1] != target[1]:
                candidate = (position[0], position[1] + step_y)
                if candidate not in obstacles:
                    next_frontier.add(candidate)
        frontier = next_frontier
        if not frontier:
            return False
        if target in frontier:
            return True
    return target in frontier
