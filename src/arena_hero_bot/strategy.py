"""Aggressive v1 tactic built on the official Arena Hero Turn controls."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Parameters that define the aggressive v1 posture."""

    target_workers: int = 2
    max_population: int = 12
    enemy_memory_ttl: int = 160
    exploration_goal_ttl: int = 80
    exploration_radius: int = 24
    worker_threat_radius: int = 6


@dataclass(slots=True)
class _TurnContext:
    turn: Turn
    report: DecisionReport
    occupied: set[Position]
    enemy_positions: set[Position]
    reserved: set[Position] = field(default_factory=set)
    departures: set[str] = field(default_factory=set)
    resource_assignments: dict[UUID, Position] = field(default_factory=dict)
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
            self._decide_ranger(ranger, context, recent_enemies)
        for vanguard in sorted(turn.vanguards, key=lambda unit: unit.id.bytes):
            self._decide_vanguard(vanguard, context, recent_enemies)
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
    ) -> None:
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
        if self._heal_if_critical(ranger, maximum_hp=2, context=context):
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

        remembered = self._best_remembered_target(ranger.position, recent_enemies)
        if remembered is not None and self._move(
            ranger,
            remembered.position,
            context,
            reason=f"hunt last seen {remembered.kind.lower()}",
        ):
            return

        self._move_or_wait(
            ranger,
            context.turn.beacon.position,
            context,
            reason="advance toward the public Beacon battle zone",
        )

    def _decide_vanguard(
        self,
        vanguard: Vanguard,
        context: _TurnContext,
        recent_enemies: tuple[EnemySighting, ...],
    ) -> None:
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
        if self._heal_if_critical(vanguard, maximum_hp=4, context=context):
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

        remembered = self._best_remembered_target(vanguard.position, recent_enemies)
        if remembered is not None and self._move(
            vanguard,
            remembered.position,
            context,
            reason=f"hunt last seen {remembered.kind.lower()}",
        ):
            return

        self._move_or_wait(
            vanguard,
            context.turn.beacon.position,
            context,
            reason="advance toward the public Beacon battle zone",
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
        if self._heal_if_critical(worker, maximum_hp=2, context=context):
            return

        if assigned_resource is not None and self._move(
            worker,
            assigned_resource,
            context,
            reason="claim nearest unassigned visible resource",
        ):
            self.memory.clear_goal(str(worker.id))
            return

        goal = self._exploration_goal(worker, context.turn.tick)
        self._move_or_wait(
            worker, goal, context, reason="scout for resources and enemies"
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
                    reason=f"expand aggressive {unit_type.value.lower()} roster",
                    target=core.position,
                )
                return

        if core.hp < 5 and context.remaining_resources > 0:
            core.heal()
            context.report.add(
                actor_id=str(core.id),
                actor_kind="CORE",
                action="HEAL",
                reason="restore Core HP between engagements",
                target=core.position,
            )
            return
        if core.shield < 5 and context.remaining_resources > 0:
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
            not nearby_enemy
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

    def _move_or_wait(
        self,
        unit: Unit,
        goal: Position,
        context: _TurnContext,
        *,
        reason: str,
        allow_goal: bool = False,
    ) -> None:
        if not self._move(
            unit,
            goal,
            context,
            reason=reason,
            allow_goal=allow_goal,
        ):
            self._record_wait(unit, context, f"no safe path for: {reason}")

    def _retreat_worker(
        self,
        worker: Worker,
        core_position: Position,
        context: _TurnContext,
        threats: tuple[UnitView, ...],
    ) -> bool:
        """Choose a one-step retreat that never needlessly closes on an enemy."""

        blocked = set(self.memory.obstacles)
        blocked.update(context.turn.obstacle_cells)
        blocked.update(context.occupied)
        blocked.update(context.reserved)
        blocked.discard(worker.position)
        blocked.discard(core_position)
        directions = (
            DIRECTIONS[self._direction_offset(worker.id) :]
            + DIRECTIONS[: self._direction_offset(worker.id)]
        )
        candidates: list[tuple[int, int, int, Direction]] = []
        for order, direction in enumerate(directions):
            destination = add(worker.position, direction)
            if destination in blocked or destination in context.enemy_positions:
                continue
            nearest_enemy = min(
                manhattan(destination, enemy.position) for enemy in threats
            )
            candidates.append(
                (
                    nearest_enemy,
                    -manhattan(destination, core_position),
                    -order,
                    direction,
                )
            )

        if candidates:
            _, _, _, direction = max(candidates)
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
    ) -> tuple[UnitView, ...]:
        return tuple(
            enemy
            for enemy in context.turn.visible_enemies
            if isinstance(enemy, UnitView)
            and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(worker.position, enemy.position)
            <= self.config.worker_threat_radius
        )

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
        blocked.update(context.turn.obstacle_cells)
        blocked.update(context.occupied)
        blocked.update(context.reserved)
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
        context.reserved.add(destination)
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
            direction_offset=self._direction_offset(core.id),
        )

    def _exploration_goal(self, worker: Worker, tick: int) -> Position:
        unit_id = str(worker.id)
        current = self.memory.goal_for(unit_id)
        if (
            current is not None
            and current.purpose == EXPLORATION_PURPOSE
            and worker.position != current.position
            and tick - current.assigned_tick <= self.config.exploration_goal_ttl
        ):
            return current.position

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
        )
        self.memory.set_goal(unit_id, goal)
        return goal.position

    def _core_has_room_for(self, worker: Worker, context: _TurnContext) -> bool:
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
        if core is None or context.turn.state.population >= self.config.max_population:
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
        if turn.state.population >= self.config.max_population:
            return None
        if len(turn.workers) < self.config.target_workers:
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
                if unit_cost(unit_type, turn.state.population) <= resources
            ),
            None,
        )

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
