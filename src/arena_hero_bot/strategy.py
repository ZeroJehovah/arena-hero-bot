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

from .combat_policy import (
    CombatPolicy,
    CombatTargetLedger,
    ThreatAssessment,
    ThreatLevel,
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
RESOURCE_CLAIM_PURPOSE = "resource-claim-v1"
RESOURCE_PATROL_PURPOSE = "resource-patrol-v3"
COMBAT_PATROL_PURPOSE = "combat-patrol-v1"
RESOURCE_CLAIM_TTL = 4
RESOURCE_PATROL_SPACING = 6
COMBAT_PATROL_SPACING = 12
DEFENSIVE_PERIMETER_MIN_RADIUS = 2
VANGUARD_VISION_RADIUS = 4
RANGER_VISION_RADIUS = 5
RANGER_STANDOFF_RANGE = 3
COMBAT_SCREEN_RADIUS = 3
CORE_ESCAPE_MIN_ENEMIES = 3
MAX_VANGUARD_ATTACKERS = 4
CORE_WORKER_INTERCEPT_RADIUS = 4
CORE_INTRUDER_INTERCEPTORS = 4
INTRUDER_LEAD_STEPS = 2
SWEEP_STAY_WEIGHT = 1
SWEEP_LIKELY_WEIGHT = 3
CORE_CAPACITY_FAST_EXPANSION = 50
CORE_CAPACITY_MEDIUM_RESERVE = 95
CORE_CAPACITY_HIGH_RESERVE = 100
CORE_STOCKPILE_CAPACITY_PERCENT = 70
CORE_BANKING_CAPACITY_PERCENT = 90
CORE_EVASION_MIN_BREAKAWAY_DISTANCE = 5
EARLY_COMBAT_MIN_UNITS = 2
EARLY_COMBAT_GUARD_WORKERS = 4
GARRISON_MIN_GUARDS = 3
GARRISON_ROSTER_SHARE = 4
RAID_ABORT_SCAN_RADIUS = 6


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
    growth_slowdown_population: int | None = 40
    safety_reserve: int = 10
    resource_patrol_radius: int = 14
    offensive_patrol_radius: int = 60
    offensive_patrol_goal_ttl: int = 160
    offensive_core_memory_ttl: int = 512
    offensive_unit_memory_ttl: int = 32
    offensive_min_combat_units: int = 8
    enemy_memory_ttl: int = 160
    enemy_core_memory_ttl: int = 4096
    exploration_goal_ttl: int = 80
    exploration_radius: int = 24
    worker_threat_radius: int = 6
    worker_threat_memory_ttl: int = 24
    intruder_hunt_ttl: int = 24
    defensive_perimeter_max_radius: int = 12
    resource_guard_min_workers: int = 6
    combat_alert_radius: int = 14
    core_intruder_radius: int = 16
    core_assault_radius: int = 8
    core_escape_enemy_count: int = CORE_ESCAPE_MIN_ENEMIES
    combat_pursuit_radius: int = 20
    raid_squad_size: int = 4
    raid_radius: int = 48
    raid_opportunity_radius: int = 30
    raid_max_ticks: int = 160
    raid_trigger_kills: int = 3
    raid_kill_window: int = 24


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
    focus_target: CoreView | UnitView | None = None
    vanguard_attack_ids: frozenset[UUID] = field(default_factory=frozenset)
    assault_enemies: tuple[CoreView | UnitView, ...] = ()
    screen_assignments: dict[UUID, Position] = field(default_factory=dict)
    core_escape_direction: Direction | None = None
    combat_assault: bool = False
    threat: ThreatAssessment = field(default_factory=ThreatAssessment)
    damage_ledger: CombatTargetLedger = field(default_factory=CombatTargetLedger)
    squad_return_ids: frozenset[UUID] = field(default_factory=frozenset)
    intruder_intercept_ids: frozenset[UUID] = field(default_factory=frozenset)
    emergency: bool = False
    garrison_ids: frozenset[UUID] = field(default_factory=frozenset)
    raid_ids: frozenset[UUID] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class _DefensiveLayout:
    """A defensive ring layout kept stable while the roster and Core stay put."""

    core_id: UUID
    core_position: Position
    guard_ids: tuple[UUID, ...]
    radius: int
    assignments: dict[UUID, Position]


class AggressiveStrategy:
    """Seek combat early while retaining a minimal resource engine."""

    def __init__(
        self,
        memory: WorldMemory,
        config: StrategyConfig | None = None,
    ) -> None:
        self.memory = memory
        self.config = config or StrategyConfig()
        # Defensive positions are intentionally strategy state rather than a
        # per-Tick context value.  Recomputing them from newly observed
        # obstacle/occupancy cells made guards oscillate between adjacent
        # radii even when there was no enemy to react to.
        self._defensive_layout: _DefensiveLayout | None = None
        self._combat_focus_id: str | None = None
        self._intruder_hunt_id: str | None = None
        self._combat_policy = CombatPolicy()
        self._squad_return_until: dict[UUID, int] = {}
        # Raid state deliberately lives on the strategy instead of
        # ``WorldMemory``: a restart then aborts the raid and walks the
        # detachment home, which is the safe direction to fail in.
        self._combat_kills: list[tuple[int, Position]] = []
        self._raid_ids: frozenset[UUID] = frozenset()
        self._raid_target: Position | None = None
        self._raid_until_tick: int = -1
        self._raid_cooldown_until: int = -1

    def decide(self, turn: Turn) -> DecisionReport:
        """Queue one complete aggressive plan for the current Turn."""

        turn.clear()
        # ``observe`` drops destroyed enemies from memory, taking the only
        # record of what they were with them, so log the kills first.
        self._record_combat_kills(turn)
        self.memory.observe(turn)
        obstacles = self.memory.obstacles | set(turn.obstacle_cells)
        threat = self._combat_policy.assess(turn, obstacles)
        recent_enemies = self.memory.recent_enemies(
            turn.tick, self.config.enemy_memory_ttl
        )
        tactical_enemies = self._tactical_enemies(turn, recent_enemies)
        emergency = (
            self._emergency_combat_mode(turn, recent_enemies)
            or threat.requires_coordination
        )
        report = DecisionReport(
            tick=turn.tick,
            visible_enemies=len(turn.visible_enemies),
            remembered_enemies=len(recent_enemies),
            threat_level=threat.level.value,
            threat_reason=threat.reason,
            minimum_ticks_to_range=threat.minimum_ticks_to_range,
            projected_core_damage=threat.projected_core_damage,
        )
        occupied = {unit.position for unit in turn.units}
        occupied.update(enemy.position for enemy in turn.visible_enemies)
        if turn.core is not None:
            occupied.add(turn.core.position)
        focus_target = self._focus_target(turn)
        raid_ids = self._update_raid(turn, threat)
        squad_return_ids = self._update_squad_returns(turn, threat, raid_ids)
        garrison_ids = self._garrison_ids(turn, raid_ids)
        context = _TurnContext(
            turn=turn,
            report=report,
            occupied=occupied,
            enemy_positions={enemy.position for enemy in turn.visible_enemies},
            resource_assignments=self._assign_resources(turn),
            remaining_resources=turn.resources,
            remaining_resource_space=turn.resource_space,
            focus_target=focus_target,
            threat=threat,
            squad_return_ids=squad_return_ids,
            vanguard_attack_ids=self._vanguard_attack_ids(
                turn,
                focus_target,
                emergency,
            ),
            intruder_intercept_ids=self._intruder_intercept_ids(
                turn,
                self._intruder_anchor(turn, focus_target, recent_enemies),
            ),
            emergency=emergency,
            garrison_ids=garrison_ids,
            raid_ids=raid_ids,
        )
        assault_enemies = self._core_assault_enemies(turn)
        if threat.requires_coordination or self._needs_preemptive_ranger_evasion(
            turn,
            self._visible_combat_enemies(turn),
            threat,
        ):
            assault_enemies = self._visible_combat_enemies(turn)
        context.assault_enemies = assault_enemies
        context.combat_assault = self._combat_defense_required(
            turn,
            assault_enemies,
            threat,
        )
        if context.combat_assault and self._should_evacuate_core(
            turn,
            assault_enemies,
            threat,
        ):
            context.core_escape_direction = self._core_escape_direction(
                context,
                assault_enemies,
            )
            if context.core_escape_direction is not None and turn.core is not None:
                # Reserve the escape lane before any unit receives a move.
                # The previous controller calculated this after the roster had
                # already occupied every useful exit.
                context.reserved.add(
                    add(turn.core.position, context.core_escape_direction)
                )
            context.screen_assignments = self._combat_screen_assignments(context)
            self._clear_core_escape_lane(context)

        for ranger in sorted(turn.rangers, key=lambda unit: unit.id.bytes):
            self._decide_ranger(
                ranger,
                context,
                tactical_enemies,
                offensive=self._is_offensive_combat_unit(ranger, turn, threat),
            )
        for vanguard in sorted(turn.vanguards, key=lambda unit: unit.id.bytes):
            self._decide_vanguard(
                vanguard,
                context,
                tactical_enemies,
                offensive=self._is_offensive_combat_unit(vanguard, turn, threat),
            )
        core_position = turn.core.position if turn.core is not None else None
        for worker in sorted(
            turn.workers,
            key=lambda unit: self._worker_priority(unit, core_position),
        ):
            self._decide_worker(worker, context)
        self._decide_core(context)
        report.planned_damage = dict(context.damage_ledger.planned_damage)
        return report

    def _decide_ranger(
        self,
        ranger: Ranger,
        context: _TurnContext,
        recent_enemies: tuple[EnemySighting, ...],
        *,
        offensive: bool = False,
    ) -> None:
        visible_enemies = self._visible_combat_targets(
            ranger,
            context.turn,
            fleet_wide=(context.emergency and ranger.id not in context.garrison_ids)
            or ranger.id in context.raid_ids,
            intercept_ids=context.intruder_intercept_ids,
        )
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        # A Ranger outranges a Worker by three cells and is the cheapest way
        # to finish a thief off, but it is far too fragile to join the chase.
        # Thieves are offered to the firing solution only; where the Ranger
        # stands is still decided by the rest of its logic below.
        firing_pool = visible_enemies + self._opportunity_intruders(
            context,
            visible_enemies,
            offensive=offensive,
        )
        shootable = [
            (enemy, shot_cell)
            for enemy in firing_pool
            if (
                shot_cell := self._ranger_shot_cell(
                    ranger,
                    enemy,
                    context.turn,
                    obstacles,
                )
            )
        ]
        if shootable:
            focused = self._preferred_target(context.focus_target, firing_pool)
            target = context.damage_ledger.select(
                tuple(item[0] for item in shootable),
                focused,
                ranger.position,
                context.turn.core.position if context.turn.core is not None else None,
            )
            shot_cell = next(
                (
                    cell
                    for candidate, cell in shootable
                    if target is not None and candidate.id == target.id
                ),
                None,
            )
            if target is None or shot_cell is None:
                shootable = []
            else:
                if (
                    isinstance(target, UnitView)
                    and target.unit_type is not UnitType.WORKER
                ):
                    standoff = self._ranger_approach_goal(ranger, target, context)
                    if (
                        not context.combat_assault
                        and standoff is not None
                        and standoff != ranger.position
                        and self._ranger_range(ranger.position, target.position)
                        < RANGER_STANDOFF_RANGE
                        # Deliberately unleashed: backing off to max range
                        # is a retreat from the target, not a push towards
                        # it, and a Ranger must never be denied that step.
                        and self._move(
                            ranger,
                            standoff,
                            context,
                            reason=(
                                f"disengage to max-range firing line on "
                                f"{self._enemy_label(target)}"
                            ),
                        )
                    ):
                        return
                ranger.shoot(target, expected_cell=shot_cell)
                context.damage_ledger.record(target)
                context.report.add(
                    actor_id=str(ranger.id),
                    actor_kind="RANGER",
                    action="SHOOT",
                    reason=(
                        (
                            f"focus fire at {self._enemy_label(target)}"
                            if focused is None or target.id == focused.id
                            else f"supporting fire at {self._enemy_label(target)}"
                        )
                        + (
                            " with one-cell lead"
                            if shot_cell != target.position
                            else ""
                        )
                        + (
                            " via damage ledger"
                            if focused is not None and target.id != focused.id
                            else ""
                        )
                    ),
                    target=shot_cell,
                )
                return

        # A Ranger at one HP is still a live firing platform.  Let it take a
        # legal shot before withdrawing; otherwise a whole damaged fireteam
        # can collapse into the Core while an enemy remains in range.
        if self._recover_if_critical(ranger, maximum_hp=2, context=context):
            return

        if (
            ranger.id in context.squad_return_ids
            and context.turn.core is not None
            and self._move(
                ranger,
                context.turn.core.position,
                context,
                reason="return intercepted expedition Ranger to Core",
                allow_goal=True,
            )
        ):
            return

        if self._pickup_beacon(ranger, context):
            return
        visible_target = self._preferred_target(context.focus_target, visible_enemies)
        if visible_target is None:
            visible_target = self._best_visible_target(
                ranger.position,
                context.turn,
                visible_enemies,
            )
        if visible_target is not None:
            goal = self._ranger_approach_goal(ranger, visible_target, context)
            if goal is not None and self._move_within_leash(
                ranger,
                goal,
                context,
                offensive=offensive,
                reason=f"close firing angle on {self._enemy_label(visible_target)}",
            ):
                return

        remembered_enemies = (
            self._offensive_enemies(
                context.turn,
                raiding=ranger.id in context.raid_ids,
            )
            if offensive
            else self._defensive_memory_targets(ranger, context, recent_enemies)
        )
        remembered = self._best_remembered_target(
            ranger.position,
            remembered_enemies,
        )
        if remembered is not None and self._move_within_leash(
            ranger,
            remembered.position,
            context,
            offensive=offensive,
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
        if not context.combat_assault and self._recover_if_critical(
            vanguard,
            maximum_hp=4,
            context=context,
        ):
            return
        visible_enemies = self._visible_combat_targets(
            vanguard,
            context.turn,
            fleet_wide=(context.emergency and vanguard.id not in context.garrison_ids)
            or vanguard.id in context.raid_ids,
            intercept_ids=context.intruder_intercept_ids,
        )
        if context.combat_assault and vanguard.id not in context.vanguard_attack_ids:
            self._decide_vanguard_screen(vanguard, context, visible_enemies)
            return
        sweep_groups = self._sweep_groups(vanguard, context, visible_enemies)
        if sweep_groups:
            focused_group = next(
                (
                    item
                    for item in sweep_groups.items()
                    if context.focus_target is not None
                    and any(enemy.id == context.focus_target.id for enemy in item[1][0])
                    and item[1][1] >= SWEEP_LIKELY_WEIGHT
                ),
                None,
            )
            direction, (targets, weight) = focused_group or max(
                sweep_groups.items(),
                key=lambda item: (
                    item[1][1],
                    sum(
                        self._enemy_score(enemy, vanguard.position, context.turn)
                        + (100 if context.damage_ledger.remaining(enemy) > 0 else -100)
                        for enemy in item[1][0]
                    ),
                ),
            )
            # A sweep is spent on the cell as it looks after movement.  Only
            # commit the Tick when a hostile is expected to still be standing
            # there; otherwise repositioning to cut the target off is worth
            # strictly more than swinging at the cell it is walking out of.
            if weight >= SWEEP_LIKELY_WEIGHT:
                vanguard.sweep(direction)
                for target in targets:
                    context.damage_ledger.record(target)
                context.report.add(
                    actor_id=str(vanguard.id),
                    actor_kind="VANGUARD",
                    action="SWEEP",
                    reason=f"hit {len(targets)} adjacent hostile object(s)",
                    target=add(vanguard.position, direction),
                )
                return

        if (
            vanguard.id in context.squad_return_ids
            and context.turn.core is not None
            and self._move(
                vanguard,
                context.turn.core.position,
                context,
                reason="return intercepted expedition Vanguard to Core",
                allow_goal=True,
            )
        ):
            return

        if self._pickup_beacon(vanguard, context):
            return
        visible_target = self._preferred_target(context.focus_target, visible_enemies)
        if visible_target is None:
            visible_target = self._best_visible_target(
                vanguard.position,
                context.turn,
                visible_enemies,
            )
        if visible_target is not None:
            if (
                context.focus_target is not None
                and visible_target.id == context.focus_target.id
            ):
                goal = self._vanguard_block_goal(vanguard, visible_target, context)
                if goal is not None and self._move_within_leash(
                    vanguard,
                    goal,
                    context,
                    offensive=offensive,
                    reason=(
                        f"close escape routes around "
                        f"{self._enemy_label(visible_target)}"
                    ),
                ):
                    return
            # Stepping into a cell next to where the target stands *now*
            # lands the chaser exactly where the target just was, so an
            # equal-speed runner keeps a permanent two-cell gap.  Aim at the
            # cell its recent drift leads to instead and the gap can close.
            intercept = self._intercept_cell(visible_target, context)
            for anchor in dict.fromkeys((intercept, visible_target.position)):
                candidates = [
                    position
                    for position in adjacent_positions(anchor)
                    if position not in self.memory.obstacles
                    and position not in context.enemy_positions
                    and (
                        position == vanguard.position
                        or position not in context.occupied | context.reserved
                    )
                ]
                if not candidates:
                    continue
                goal = min(
                    candidates,
                    key=lambda position: (
                        manhattan(vanguard.position, position),
                        manhattan(intercept, position),
                        position,
                    ),
                )
                if self._move_within_leash(
                    vanguard,
                    goal,
                    context,
                    offensive=offensive,
                    reason=f"rush {self._enemy_label(visible_target)}",
                ):
                    return

        if context.combat_assault and self._recover_if_critical(
            vanguard,
            maximum_hp=4,
            context=context,
        ):
            return

        remembered_enemies = (
            self._offensive_enemies(
                context.turn,
                raiding=vanguard.id in context.raid_ids,
            )
            if offensive
            else self._defensive_memory_targets(vanguard, context, recent_enemies)
        )
        remembered = self._best_remembered_target(
            vanguard.position,
            remembered_enemies,
        )
        if remembered is not None and self._move_within_leash(
            vanguard,
            remembered.position,
            context,
            offensive=offensive,
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

    def _decide_vanguard_screen(
        self,
        vanguard: Vanguard,
        context: _TurnContext,
        visible_enemies: tuple[CoreView | UnitView, ...],
    ) -> None:
        """Hold a compact screen while a bounded strike team attacks."""

        adjacent_groups: dict[Direction, list[CoreView | UnitView]] = {}
        for enemy in visible_enemies:
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
                reason=f"screen Core against {len(targets)} adjacent hostile object(s)",
                target=add(vanguard.position, direction),
            )
            return

        # A screening Vanguard used to hold its ring cell no matter how much
        # damage it had taken.  Ranged attackers outrange the sweep above, so a
        # critical screen kept standing on its post, unable to answer, until it
        # died: three Vanguards were lost that way in a single fleet attack,
        # one of them waiting motionless for four Ticks at 2 HP.  Rotating it
        # out costs one ring cell and saves the Unit.
        if self._recover_if_critical(vanguard, maximum_hp=4, context=context):
            return

        goal = self._combat_screen_goal(vanguard, context)
        self._move_or_wait(
            vanguard,
            goal,
            context,
            reason="hold the Core screen while the strike team attacks",
            wait_at_goal=True,
        )

    def _combat_screen_goal(
        self,
        vanguard: Vanguard,
        context: _TurnContext,
    ) -> Position:
        return context.screen_assignments.get(vanguard.id, vanguard.position)

    def _combat_screen_assignments(
        self,
        context: _TurnContext,
    ) -> dict[UUID, Position]:
        """Assign unique threat-facing ring cells to the defensive Vanguards."""

        core = context.turn.core
        if core is None:
            return {}
        core_position = core.position
        screens = tuple(
            sorted(
                (
                    unit
                    for unit in context.turn.vanguards
                    if unit.id not in context.vanguard_attack_ids
                ),
                key=lambda unit: unit.id.bytes,
            )
        )
        if not screens:
            return {}
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        enemy_positions = {enemy.position for enemy in context.assault_enemies}
        screen_positions = {unit.position for unit in screens}
        available = [
            (core_position[0] + dx, core_position[1] + dy)
            for dx, dy in _defensive_ring_offsets(COMBAT_SCREEN_RADIUS)
            if (core_position[0] + dx, core_position[1] + dy) not in obstacles
            and (core_position[0] + dx, core_position[1] + dy) not in enemy_positions
            and (core_position[0] + dx, core_position[1] + dy) not in context.reserved
            and (
                (core_position[0] + dx, core_position[1] + dy) in screen_positions
                or (core_position[0] + dx, core_position[1] + dy)
                not in context.occupied
            )
        ]
        assignments: dict[UUID, Position] = {}
        for screen in screens:
            if not available:
                assignments[screen.id] = screen.position
                continue
            position = max(
                available,
                key=lambda candidate: (
                    sum(
                        manhattan(core_position, enemy.position)
                        - manhattan(candidate, enemy.position)
                        for enemy in context.assault_enemies
                    ),
                    -min(
                        manhattan(candidate, enemy.position)
                        for enemy in context.assault_enemies
                    ),
                    -manhattan(screen.position, candidate),
                    candidate,
                ),
            )
            assignments[screen.id] = position
            available.remove(position)
        return assignments

    def _intercept_cell(
        self,
        enemy: CoreView | UnitView,
        context: _TurnContext,
        steps: int = INTRUDER_LEAD_STEPS,
    ) -> Position:
        """Return the cell to converge on, leading a moving target."""

        lead = self.memory.enemy_drift_position(
            str(enemy.id),
            context.turn.tick,
            steps,
        )
        if lead is None or lead in self.memory.obstacles:
            return enemy.position
        if lead in context.turn.obstacle_cells:
            return enemy.position
        return lead

    def _sweep_groups(
        self,
        vanguard: Vanguard,
        context: _TurnContext,
        visible_enemies: tuple[CoreView | UnitView, ...],
    ) -> dict[Direction, tuple[tuple[CoreView | UnitView, ...], int]]:
        """Group reachable hostiles per sweep direction with a hit confidence.

        Combat resolves on the snapshot taken after movement, so the useful
        question is not "who stands next to me" but "who will stand in the
        cell I am about to sweep".  A hostile whose drift leads into the cell,
        and a hostile with no movement evidence at all, are both likely to be
        there.  One that is visibly walking out of the cell is not.
        """

        collected: dict[Direction, dict[str, CoreView | UnitView]] = {}
        weights: dict[Direction, int] = {}
        for enemy in visible_enemies:
            lead = self.memory.enemy_drift_position(
                str(enemy.id),
                context.turn.tick,
                1,
            )
            reachable: dict[Direction, int] = {}
            here = direction_between(vanguard.position, enemy.position)
            if here is not None:
                reachable[here] = (
                    SWEEP_STAY_WEIGHT
                    if lead is not None and lead != enemy.position
                    else SWEEP_LIKELY_WEIGHT
                )
            if lead is not None and lead != enemy.position:
                ahead = direction_between(vanguard.position, lead)
                if ahead is not None:
                    reachable[ahead] = SWEEP_LIKELY_WEIGHT
            for direction, weight in reachable.items():
                collected.setdefault(direction, {})[str(enemy.id)] = enemy
                weights[direction] = weights.get(direction, 0) + weight
        return {
            direction: (tuple(targets.values()), weights[direction])
            for direction, targets in collected.items()
        }

    def _vanguard_block_goal(
        self,
        vanguard: Vanguard,
        target: CoreView | UnitView,
        context: _TurnContext,
    ) -> Position | None:
        """Assign the focus target's adjacent escape cells across Vanguards.

        The ring is placed around the cell the target is heading for, not the
        one it is standing in, so the escort arrives where the escape routes
        actually are.  Only the detached escort takes part in the rotation;
        spreading the slots over the whole roster gave each blocker an
        effectively random preference.
        """

        anchor = self._intercept_cell(target, context)
        slots = [
            position
            for position in adjacent_positions(anchor)
            if position not in self.memory.obstacles
            and position not in context.turn.obstacle_cells
            and position not in context.enemy_positions
            and position not in context.occupied | context.reserved
        ]
        if not slots:
            return None
        escort = context.intruder_intercept_ids
        ordered_vanguards = sorted(
            (
                unit
                for unit in context.turn.vanguards
                if not escort or unit.id in escort
            ),
            key=lambda unit: unit.id.bytes,
        )
        index = next(
            (
                index
                for index, unit in enumerate(ordered_vanguards)
                if unit.id == vanguard.id
            ),
            0,
        )
        preferred = adjacent_positions(anchor)
        rotation = index % len(preferred)
        rotated = preferred[rotation:] + preferred[:rotation]
        return min(
            slots,
            key=lambda position: (
                next(slot for slot, cell in enumerate(rotated) if cell == position),
                manhattan(vanguard.position, position),
                position,
            ),
        )

    def _decide_worker(self, worker: Worker, context: _TurnContext) -> None:
        core = context.turn.core
        if core is None:
            self._record_wait(worker, context, "no Core while respawning")
            return

        # ``_emergency_worker_goal`` places Workers on a ring around the Core
        # at the assault radius.  Applying it to a Worker that is already
        # further out dragged distant economy Units back toward the fight and
        # abandoned resources they were about to claim, so only Workers inside
        # the contested zone are pulled out of it.
        if context.combat_assault and (
            manhattan(worker.position, core.position) <= self.config.core_assault_radius
        ):
            goal = self._emergency_worker_goal(worker, context)
            if goal == worker.position:
                self._record_wait(
                    worker,
                    context,
                    "hold outside the Core assault zone",
                )
            elif self._move(
                worker,
                goal,
                context,
                reason="clear the Core assault zone with economy unit",
                allow_goal=True,
            ):
                return
            else:
                worker_threats = self._worker_threats(worker, context)
                if worker_threats and self._retreat_worker(
                    worker,
                    core.position,
                    context,
                    worker_threats,
                ):
                    return
                self._record_wait(
                    worker,
                    context,
                    "no safe path for: clear the Core assault zone",
                )
            return

        worker_threats = self._worker_threats(worker, context)
        combat_threats = self._worker_combat_threats(worker, context)
        if (
            worker.hp <= 1
            and worker_threats
            and self._recover_if_critical(worker, maximum_hp=2, context=context)
        ):
            return
        if worker.cargo > 0 and combat_threats:
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
                combat_threats,
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
            # Only one Worker can hold the Core cell per Tick and the guard
            # ring adds more contention, so a loaded Worker regularly lost the
            # race.  Waiting in place left it idle wherever it happened to
            # stand; closing the remaining distance instead means the deposit
            # lands on the Tick the cell frees up.
            if manhattan(worker.position, core.position) > 1 and self._move(
                worker,
                core.position,
                context,
                reason="stage carried resources next to the busy Core",
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
            self.memory.set_goal(
                str(worker.id),
                UnitGoal(
                    position=assigned_resource,
                    assigned_tick=context.turn.tick,
                    purpose=RESOURCE_CLAIM_PURPOSE,
                    last_progress_position=worker.position,
                ),
            )
            return

        resource_goal = self.memory.goal_for(str(worker.id))
        if (
            assigned_resource is None
            and resource_goal is not None
            and resource_goal.purpose == RESOURCE_CLAIM_PURPOSE
        ):
            if resource_goal.position in context.turn.resource_cells:
                self.memory.clear_goal(str(worker.id))
            else:
                age = context.turn.tick - resource_goal.assigned_tick
                progress_reference = resource_goal.last_progress_position
                progressing = progress_reference is not None and manhattan(
                    worker.position, resource_goal.position
                ) < manhattan(progress_reference, resource_goal.position)
                if age <= RESOURCE_CLAIM_TTL or progressing:
                    if worker.position == resource_goal.position:
                        self._record_wait(
                            worker,
                            context,
                            "wait for recently seen resource to reappear",
                        )
                        return
                    if self._move(
                        worker,
                        resource_goal.position,
                        context,
                        reason="continue toward recently seen resource",
                    ):
                        if progressing and age > RESOURCE_CLAIM_TTL:
                            # A visible resource can be several travel Ticks
                            # away.  Keep a stale observation only while the
                            # Worker is demonstrably closing in; a stalled
                            # route still expires at the short TTL.
                            self.memory.set_goal(
                                str(worker.id),
                                UnitGoal(
                                    position=resource_goal.position,
                                    assigned_tick=context.turn.tick,
                                    purpose=resource_goal.purpose,
                                    last_progress_position=worker.position,
                                ),
                            )
                        return
                self.memory.clear_goal(str(worker.id))

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

    def _emergency_worker_goal(
        self,
        worker: Worker,
        context: _TurnContext,
    ) -> Position:
        """Move Workers away from the fight instead of clogging the Core exit."""

        core = context.turn.core
        if core is None:
            return worker.position
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        enemy_positions = {enemy.position for enemy in context.assault_enemies}
        candidates = [
            (core.position[0] + dx, core.position[1] + dy)
            for dx, dy in _defensive_ring_offsets(
                max(5, self.config.core_assault_radius)
            )
        ]
        legal = [
            position
            for position in candidates
            if position not in obstacles
            and position not in enemy_positions
            and position not in context.reserved
            and (position == worker.position or position not in context.occupied)
        ]
        if not legal:
            return worker.position
        return max(
            legal,
            key=lambda position: (
                min(manhattan(position, enemy) for enemy in enemy_positions)
                if enemy_positions
                else 0,
                manhattan(position, core.position),
                position,
            ),
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

        if context.core_escape_direction is not None:
            core.start_move(context.core_escape_direction)
            context.report.add(
                actor_id=str(core.id),
                actor_kind="CORE",
                action="START_MOVE",
                reason=(
                    "evacuate Core from overwhelming enemy assault before "
                    "the screen breaks"
                ),
                target=add(core.position, context.core_escape_direction),
            )
            return

        nearby_enemy = any(
            manhattan(core.position, enemy.position) <= 4
            for enemy in context.turn.visible_enemies
        )
        nearby_combat_enemies = tuple(
            enemy
            for enemy in context.turn.visible_enemies
            if isinstance(enemy, CoreView)
            or (
                enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                and manhattan(core.position, enemy.position) <= 4
            )
        )
        overwhelmed = (
            nearby_enemy
            and self._can_break_away(context.turn, nearby_combat_enemies)
            and (
                len(nearby_combat_enemies) >= self.config.core_escape_enemy_count
                or (core.shield <= 2 and len(nearby_combat_enemies) >= 2)
            )
        )
        if overwhelmed:
            direction = self._core_escape_direction(context, nearby_combat_enemies)
            if direction is not None:
                core.start_move(direction)
                context.report.add(
                    actor_id=str(core.id),
                    actor_kind="CORE",
                    action="START_MOVE",
                    reason="evacuate Core from overwhelming enemy assault",
                    target=add(core.position, direction),
                )
                return
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
        if (
            (nearby_enemy or (self._unbounded_growth() and context.emergency))
            and core.shield < 5
            and context.remaining_resources > 0
        ):
            core.repair_shield()
            context.report.add(
                actor_id=str(core.id),
                actor_kind="CORE",
                action="REPAIR_SHIELD",
                reason="repair damaged shield before the next enemy strike",
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
        combat_threats = self._worker_combat_threats(worker, context)
        contested = [
            position
            for position in self.memory.contested_positions
            if manhattan(worker.position, position) <= self.config.worker_threat_radius
        ]
        return tuple(dict.fromkeys((*combat_threats, *contested)))

    def _worker_combat_threats(
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
        return tuple(dict.fromkeys((*visible, *remembered)))

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
        self,
        origin: Position,
        turn: Turn,
        enemies: tuple[CoreView | UnitView, ...] | None = None,
    ) -> CoreView | UnitView | None:
        enemies = turn.visible_enemies if enemies is None else enemies
        if not enemies:
            return None
        target = max(
            enemies,
            key=lambda enemy: (
                self._combat_target_priority(enemy),
                self._enemy_score(enemy, origin, turn),
            ),
        )
        if target is None:
            return None
        return target

    def _opportunity_intruders(
        self,
        context: _TurnContext,
        known: tuple[CoreView | UnitView, ...],
        *,
        offensive: bool,
    ) -> tuple[UnitView, ...]:
        """Return in-zone thieves a Ranger may shoot without giving chase."""

        if offensive:
            return ()
        seen = {enemy.id for enemy in known}
        return tuple(
            enemy
            for enemy in self._core_intruders(context.turn)
            if enemy.id not in seen
        )

    def _visible_combat_targets(
        self,
        unit: Ranger | Vanguard,
        turn: Turn,
        *,
        fleet_wide: bool = False,
        intercept_ids: frozenset[UUID] = frozenset(),
    ) -> tuple[CoreView | UnitView, ...]:
        """Limit reactions to enemies in that unit's own local area.

        ``turn.visible_enemies`` is fleet-wide, so an escort member is allowed
        to act on an intruder that only a Worker can currently see.  Every
        other unit keeps the strict own-vision rule and stays on station.

        ``fleet_wide`` is granted to the raid detachment and, during an
        emergency, to every guard outside the garrison.  It used to be handed
        to the roaming squad too, which meant a single Worker sighting forty
        cells out pulled the whole roaming third onto it -- exactly the
        open-ended commitment the movement leash exists to prevent.
        """

        if fleet_wide:
            return turn.visible_enemies
        escorting = unit.id in intercept_ids
        return tuple(
            enemy
            for enemy in turn.visible_enemies
            if self._combat_target_is_local(unit, enemy.position)
            or (
                escorting
                and isinstance(enemy, UnitView)
                and enemy.unit_type is UnitType.WORKER
                and turn.core is not None
                and manhattan(turn.core.position, enemy.position)
                <= self.config.core_intruder_radius
            )
        )

    def _core_intruders(self, turn: Turn) -> tuple[UnitView, ...]:
        """Return visible enemy Workers loitering inside the economy zone.

        An enemy Worker cannot attack, so it never belongs in the threat
        ladder that decides evacuation or combat posture.  It does steal the
        resource cells this Core lives on and scouts for its own army, so it
        still has to be hunted down.  Ordering by Core distance keeps the
        closest thief as the focus while it stays inside the zone.
        """

        core = turn.core
        if core is None:
            return ()
        radius = self.config.core_intruder_radius
        return tuple(
            sorted(
                (
                    enemy
                    for enemy in turn.visible_enemies
                    if isinstance(enemy, UnitView)
                    and enemy.unit_type is UnitType.WORKER
                    and manhattan(core.position, enemy.position) <= radius
                ),
                key=lambda enemy: (
                    manhattan(core.position, enemy.position),
                    str(enemy.id),
                ),
            )
        )

    def _intruder_anchor(
        self,
        turn: Turn,
        focus_target: CoreView | UnitView | None,
        recent_enemies: tuple[EnemySighting, ...],
    ) -> Position | None:
        """Return the cell the intruder escort should converge on.

        Vision over a thief inside the economy zone flickers, because the
        perimeter guards face outward and the cell is usually only covered by
        a passing Worker.  The escort used to disband on the first dark Tick,
        so the hunt restarted from scratch every few Ticks and the Worker
        strolled away.  Holding the last known cell keeps the same squad
        committed across the gap; the hunt is dropped once that sighting goes
        stale or leaves the zone.
        """

        core = turn.core
        if core is None:
            self._intruder_hunt_id = None
            return None
        if (
            isinstance(focus_target, UnitView)
            and focus_target.unit_type is UnitType.WORKER
        ):
            self._intruder_hunt_id = str(focus_target.id)
            return focus_target.position
        if focus_target is not None:
            # Anything that can actually shoot outranks a thief, but the hunt
            # is only paused: the escort resumes once the fight is over.
            return None
        if self._intruder_hunt_id is None:
            return None
        radius = self.config.core_intruder_radius
        sighting = next(
            (
                enemy
                for enemy in recent_enemies
                if enemy.object_id == self._intruder_hunt_id
                and turn.tick - enemy.tick <= self.config.intruder_hunt_ttl
                and manhattan(core.position, enemy.position) <= radius
            ),
            None,
        )
        if sighting is None:
            self._intruder_hunt_id = None
            return None
        return sighting.position

    def _intruder_intercept_ids(
        self,
        turn: Turn,
        anchor: Position | None,
    ) -> frozenset[UUID]:
        """Reserve a bounded escort for one intruder.

        A lone chaser can never catch an equal-speed target: it steps into the
        cell the target just left and the gap stays at two forever.  Killing
        one needs its escape cells covered, so a small squad is detached while
        every other guard keeps its perimeter slot instead of stampeding
        across the map after a single Worker.
        """

        if anchor is None or not turn.vanguards:
            return frozenset()
        ordered = sorted(
            turn.vanguards,
            key=lambda unit: (
                manhattan(unit.position, anchor),
                unit.hp <= 2,
                unit.id.bytes,
            ),
        )
        return frozenset(unit.id for unit in ordered[:CORE_INTRUDER_INTERCEPTORS])

    def _focus_target(self, turn: Turn) -> CoreView | UnitView | None:
        enemies: list[CoreView | UnitView] = [
            enemy
            for enemy in turn.visible_enemies
            if isinstance(enemy, CoreView)
            or (
                isinstance(enemy, UnitView)
                and enemy.unit_type in {UnitType.RANGER, UnitType.VANGUARD}
            )
        ]
        if not enemies:
            intruders = self._core_intruders(turn) if turn.vanguards else ()
            if not intruders:
                self._combat_focus_id = None
                return None
            # Stay on the intruder already being hunted while it remains
            # inside the zone; swapping focus every Tick restarted the pincer
            # and let each thief walk away untouched.
            target = next(
                (
                    enemy
                    for enemy in intruders
                    if str(enemy.id) == self._combat_focus_id
                ),
                intruders[0],
            )
            self._combat_focus_id = str(target.id)
            return target
        obstacles = self.memory.obstacles | set(turn.obstacle_cells)
        shootable = tuple(
            enemy
            for enemy in enemies
            if any(
                self._ranger_shot_cell(ranger, enemy, turn, obstacles) is not None
                for ranger in turn.rangers
            )
        )
        candidates = shootable or tuple(enemies)
        origin = turn.core.position if turn.core is not None else (0, 0)
        best = max(
            candidates,
            key=lambda enemy: (
                self._combat_target_priority(enemy),
                self._enemy_score(enemy, origin, turn),
                str(enemy.id),
            ),
        )
        sticky = next(
            (enemy for enemy in enemies if str(enemy.id) == self._combat_focus_id),
            None,
        )
        if sticky is not None and self._combat_target_priority(sticky) >= (
            self._combat_target_priority(best)
        ):
            return sticky
        self._combat_focus_id = str(best.id)
        return best

    @staticmethod
    def _combat_target_priority(enemy: CoreView | UnitView) -> int:
        """Prefer the ranged threat that caused the last large-battle loss."""

        if isinstance(enemy, CoreView):
            return 1
        return {
            UnitType.RANGER: 3,
            UnitType.VANGUARD: 2,
            UnitType.WORKER: 0,
        }[enemy.unit_type]

    def _vanguard_attack_ids(
        self,
        turn: Turn,
        focus_target: CoreView | UnitView | None,
        emergency: bool,
    ) -> frozenset[UUID]:
        """Reserve only a bounded strike team; keep the rest as a Core screen."""

        if not emergency or focus_target is None or not turn.vanguards:
            return frozenset()
        ordered = tuple(
            sorted(
                turn.vanguards,
                key=lambda unit: (
                    unit.hp <= 2,
                    manhattan(unit.position, focus_target.position),
                    unit.id.bytes,
                ),
            )
        )
        if len(ordered) <= MAX_VANGUARD_ATTACKERS:
            count = len(ordered)
        else:
            count = min(MAX_VANGUARD_ATTACKERS, max(2, (len(ordered) + 1) // 2))
        return frozenset(unit.id for unit in ordered[:count])

    @staticmethod
    def _preferred_target(
        focus: CoreView | UnitView | None,
        enemies: tuple[CoreView | UnitView, ...],
    ) -> CoreView | UnitView | None:
        if focus is None:
            return None
        return next((enemy for enemy in enemies if enemy.id == focus.id), None)

    def _core_assault_enemies(
        self,
        turn: Turn,
    ) -> tuple[CoreView | UnitView, ...]:
        core = turn.core
        if core is None:
            return ()
        return tuple(
            enemy
            for enemy in turn.visible_enemies
            if (
                isinstance(enemy, CoreView)
                or enemy.unit_type in {UnitType.RANGER, UnitType.VANGUARD}
            )
            and manhattan(core.position, enemy.position)
            <= self.config.core_assault_radius
        )

    def _should_evacuate_core(
        self,
        turn: Turn,
        enemies: tuple[CoreView | UnitView, ...],
        threat: ThreatAssessment | None = None,
    ) -> bool:
        core = turn.core
        if core is None or core.view.state is CoreState.MOVING or not enemies:
            return False
        if not self._can_break_away(turn, enemies):
            return False
        if self._guardless_low_capacity_assault(turn, enemies):
            # A freshly respawned Core can spend several minutes below the
            # first Vanguard threshold.  Once a combat enemy enters the local
            # assault ring, moving immediately is safer than waiting for the
            # first shot while the economy is still undefended.
            return True
        if self._needs_preemptive_ranger_evasion(turn, enemies, threat):
            # A distant combat unit cannot protect the Core during its
            # four-Tick migration. Evacuate an outer-ring Ranger approach
            # instead of waiting for pursuit memory or the first hit.
            return True
        if threat is not None and threat.should_evacuate_core:
            return True
        # A group already inside the pre-evade ring is dangerous even when it
        # has not crossed attack range yet. Waiting for the first hit leaves a
        # four-Tick Core migration too late against a melee group.
        if (
            threat is not None
            and threat.level is ThreatLevel.PRE_EVADE
            and len(enemies) >= 2
            and min(manhattan(core.position, enemy.position) for enemy in enemies) <= 3
        ):
            return True
        if len(enemies) >= self.config.core_escape_enemy_count:
            return True
        if core.shield <= 3 and len(enemies) >= 2:
            return True
        return core.hp <= 3

    def _combat_defense_required(
        self,
        turn: Turn,
        assault_enemies: tuple[CoreView | UnitView, ...],
        threat: ThreatAssessment | None = None,
    ) -> bool:
        """Raise the coordinated defense posture before the Core is hit."""

        core = turn.core
        if core is None or core.view.state is CoreState.MOVING:
            return False
        if self._guardless_low_capacity_assault(turn, assault_enemies):
            return True
        if self._needs_preemptive_ranger_evasion(turn, assault_enemies, threat):
            return True
        if threat is not None and threat.requires_coordination:
            return True
        if len(assault_enemies) >= 2:
            return True
        nearby = tuple(
            enemy
            for enemy in turn.visible_enemies
            if (
                isinstance(enemy, CoreView)
                or enemy.unit_type in {UnitType.RANGER, UnitType.VANGUARD}
            )
            and manhattan(core.position, enemy.position)
            <= self.config.combat_alert_radius
        )
        return len(nearby) >= max(4, self.config.core_escape_enemy_count * 2)

    def _guardless_low_capacity_assault(
        self,
        turn: Turn,
        enemies: tuple[CoreView | UnitView, ...],
    ) -> bool:
        """Evacuate an undefended low-capacity Core before its first hit."""

        return (
            self._unbounded_growth()
            and turn.resource_capacity < CORE_CAPACITY_FAST_EXPANSION
            and not turn.vanguards
            and not turn.rangers
            and bool(enemies)
        )

    def _can_break_away(
        self,
        turn: Turn,
        enemies: tuple[CoreView | UnitView, ...],
    ) -> bool:
        """Return whether a Core migration can still outrun this pursuit.

        A Core needs four Ticks per cell while Units cover one cell per Tick,
        and a ``MOVING`` Core can neither ``HEAL`` nor ``REPAIR_SHIELD``.  An
        Core that starts migrating with an unmatched pursuer already close
        therefore trades its only repair option for a retreat it cannot win:
        three of six recorded Core losses were chipped down mid-migration by
        one or two pursuers.  When nearby guards at least match the pursuit
        they can trade Ticks for the migration, so the usual evacuation
        criteria still apply.
        """

        core = turn.core
        if core is None:
            return False
        if not enemies:
            return True
        screen = [
            unit
            for unit in (*turn.vanguards, *turn.rangers)
            if manhattan(core.position, unit.position)
            <= self.config.core_assault_radius
        ]
        if len(screen) >= len(enemies):
            return True
        return (
            min(manhattan(core.position, enemy.position) for enemy in enemies)
            >= CORE_EVASION_MIN_BREAKAWAY_DISTANCE
        )

    def _has_local_combat_guard(self, turn: Turn) -> bool:
        """Return whether a combat unit is close enough to screen the Core."""

        core = turn.core
        return core is not None and any(
            manhattan(core.position, unit.position) <= COMBAT_SCREEN_RADIUS
            for unit in (*turn.vanguards, *turn.rangers)
        )

    def _needs_preemptive_ranger_evasion(
        self,
        turn: Turn,
        enemies: tuple[CoreView | UnitView, ...],
        threat: ThreatAssessment | None,
    ) -> bool:
        """Detect an outer-ring Ranger approach without a local Core screen."""

        core = turn.core
        if (
            core is None
            or threat is None
            or threat.level is not ThreatLevel.PRE_EVADE
            or not threat.near_core_enemy_ids
            or self._has_local_combat_guard(turn)
        ):
            return False
        return any(
            isinstance(enemy, UnitView)
            and enemy.unit_type is UnitType.RANGER
            and self.config.core_assault_radius
            < manhattan(core.position, enemy.position)
            <= self.config.core_assault_radius + 4
            for enemy in enemies
        )

    def _emergency_combat_mode(
        self,
        turn: Turn,
        recent_enemies: tuple[EnemySighting, ...] | None = None,
    ) -> bool:
        core = turn.core
        if core is None:
            return False
        if core.hp < 5 or core.shield < 5:
            return True
        if turn.visible_enemies:
            combat_enemies = tuple(
                enemy
                for enemy in turn.visible_enemies
                if isinstance(enemy, CoreView)
                or enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            )
            nearby = tuple(
                enemy
                for enemy in combat_enemies
                if manhattan(core.position, enemy.position)
                <= self.config.combat_alert_radius
            )
            return len(nearby) >= 2
        enemies = recent_enemies or self.memory.recent_enemies(
            turn.tick,
            self.config.enemy_memory_ttl,
        )
        nearby_combat = tuple(
            enemy
            for enemy in enemies
            if (
                enemy.kind == "CORE"
                or enemy.unit_type in {UnitType.VANGUARD.value, UnitType.RANGER.value}
            )
            and turn.tick - enemy.tick <= 16
            and manhattan(core.position, enemy.position) <= 12
        )
        return len(nearby_combat) >= 3

    def _ranger_shot_cell(
        self,
        ranger: Ranger,
        enemy: CoreView | UnitView,
        turn: Turn,
        obstacles: set[Position] | frozenset[Position],
    ) -> Position | None:
        # Movement resolves before the shot, so aiming at the cell a moving
        # target currently occupies is a guaranteed miss.  ``drift`` is the
        # loose one-step estimate and fires far more often than the strict
        # three-Tick predictor, which almost never triggers while vision over
        # the target keeps flickering.  At range 1 the target is close enough
        # to stall or collide, so the standing cell is still the better guess
        # there; from range 2 outwards the lead is preferred.
        strict = self.memory.predicted_enemy_position(str(enemy.id), turn.tick)
        drift = self.memory.enemy_drift_position(str(enemy.id), turn.tick, 1)
        predicted = strict or drift
        leads = (
            predicted is not None
            and predicted != enemy.position
            and self._ranger_range(ranger.position, enemy.position) > 1
        )
        candidates = (predicted, enemy.position) if leads else (enemy.position,)
        for cell in dict.fromkeys(cell for cell in candidates if cell is not None):
            if line_of_fire(ranger.position, cell, obstacles):
                return cell
        return None

    def _combat_target_is_local(
        self,
        unit: Ranger | Vanguard,
        target: Position,
    ) -> bool:
        return manhattan(unit.position, target) <= _combat_vision_radius(unit)

    def _defensive_memory_targets(
        self,
        unit: Ranger | Vanguard,
        context: _TurnContext,
        recent_enemies: tuple[EnemySighting, ...],
    ) -> tuple[EnemySighting, ...]:
        """Pick the last-seen hostiles this guard is allowed to walk towards.

        Vision over an intruder flickers constantly, because the perimeter
        guards face outward and the cell is usually only covered by a passing
        Worker.  Dropping remembered Workers outright therefore cancelled the
        hunt every few Ticks and reset it from scratch, which is why thieves
        survived hundreds of Ticks inside the zone.  The detached escort keeps
        the last known cell of an intruder; everyone else still ignores
        remembered Workers and stays on station.
        """

        core = context.turn.core
        escorting = unit.id in context.intruder_intercept_ids
        radius = self.config.core_intruder_radius
        targets: list[EnemySighting] = []
        for enemy in recent_enemies:
            if self._is_combat_memory_target(enemy):
                if self._combat_target_is_local(unit, enemy.position):
                    targets.append(enemy)
                continue
            if (
                escorting
                and core is not None
                and enemy.kind == "UNIT"
                and enemy.unit_type == UnitType.WORKER.value
                and manhattan(core.position, enemy.position) <= radius
            ):
                targets.append(enemy)
        return tuple(targets)

    @staticmethod
    def _is_combat_memory_target(enemy: EnemySighting) -> bool:
        """Whether a remembered hostile can attack and is worth intercepting."""

        return enemy.kind == "CORE" or enemy.unit_type in {
            UnitType.RANGER.value,
            UnitType.VANGUARD.value,
        }

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
        threat: ThreatAssessment | None = None,
    ) -> bool:
        """Assign a stable minority of combat units to the roaming squad."""

        if (
            self._emergency_combat_mode(turn)
            or not self._offensive_patrol_enabled(turn)
            or (threat is not None and threat.level is not ThreatLevel.NORMAL)
        ):
            return False
        # UUID assignment keeps roles stable across ticks without adding
        # private state.  With the live roster this sends roughly one third
        # of Rangers/Vanguards away while leaving a substantial Core guard.
        return unit.id.int % 3 == 0

    def _offensive_patrol_enabled(self, turn: Turn) -> bool:
        """Keep a minority roaming while the Core itself remains safe.

        Core stockpile tiers govern production and high-capacity patrol safety.
        Small Cores still retain the original minority patrol while they are
        in the fast-expansion tier, but a large Core below its reserve should
        not leave combat units exposed far from the recovery point.  The
        patrol is also suspended when the Core is damaged or the combat roster
        is too small to leave a meaningful guard.
        """

        core = turn.core
        return (
            self._unbounded_growth()
            and core is not None
            and len(turn.vanguards) + len(turn.rangers)
            >= self.config.offensive_min_combat_units
            and core.hp >= 5
            and core.shield >= 5
            and turn.resources >= self._stockpile_target(turn)
        )

    def _offensive_enemies(
        self,
        turn: Turn,
        *,
        raiding: bool = False,
    ) -> tuple[EnemySighting, ...]:
        """Return recent targets suitable for a roaming squad to pursue.

        A remembered enemy Core is offered to the raid detachment only.
        Enemy Cores sit a median 59 cells away, so letting the whole
        roaming third walk at one is a two-hundred-Tick commitment of a
        third of the army - the open-ended kind of push the leash exists
        to stop.  The bounded four-unit raid takes that job instead.
        """

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
                raiding
                and enemy.kind == "CORE"
                and turn.tick - enemy.tick <= self.config.offensive_core_memory_ttl
            )
            or (
                enemy.kind == "UNIT"
                and turn.tick - enemy.tick <= self.config.offensive_unit_memory_ttl
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
        if (
            context is not None
            and self._raid_target is not None
            and unit.id in context.raid_ids
        ):
            return (
                self._raid_target,
                "raid the enemy Core along the attack bearing",
            )
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
        radius ``r`` contains ``4r`` cells.  Obstacle cells are removed, but
        the ring is not required to be connected to the Core: guards can walk
        around an interior obstacle to reach a legal perimeter cell.  Each
        guard covers approximately ``2 * vision`` consecutive ring cells;
        shrinking only when visibility gaps cannot be covered gives a stable,
        count-and-vision-derived perimeter.
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
            self._defensive_layout = None
            return {}

        obstacles = set(self.memory.obstacles)
        obstacles.update(context.turn.obstacle_cells)
        ordered_guard_ids = tuple(unit.id for unit in guards)
        cached = self._defensive_layout
        if (
            cached is not None
            and cached.core_id == core.id
            and cached.core_position == core_position
            and cached.guard_ids == ordered_guard_ids
            and not any(slot in obstacles for slot in cached.assignments.values())
        ):
            return dict(cached.assignments)

        # Transient unit occupancy is deliberately not a layout constraint.
        # Workers naturally pass through the perimeter and guards can wait
        # for a slot to clear; treating those cells as unavailable would make
        # the radius and cardinal anchors depend on traffic at one Tick.
        unavailable = set(context.enemy_positions)
        unavailable.add(core.position)
        visions = {unit.id: _combat_vision_radius(unit) for unit in guards}
        # The count-and-vision radius grows linearly with the army, so a
        # 29-guard fleet parked its whole perimeter 30 cells out and left the
        # interior empty: anything already inside the ring could only be
        # chased from behind, and an equal-speed runner is uncatchable that
        # way.  Capping the radius keeps the guards between the intruder and
        # the Core, where they are also closer to home if a real attack lands.
        base_radius = max(
            DEFENSIVE_PERIMETER_MIN_RADIUS,
            min(
                self.config.defensive_perimeter_max_radius,
                sum(visions.values()) // 4,
            ),
        )

        def ring_for(radius: int) -> tuple[Position, ...]:
            return tuple(
                (core_position[0] + dx, core_position[1] + dy)
                for dx, dy in _defensive_ring_offsets(radius)
                if (core_position[0] + dx, core_position[1] + dy) not in obstacles
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
        best_radius = max_radius
        for radius in range(max_radius, DEFENSIVE_PERIMETER_MIN_RADIUS - 1, -1):
            coverage_cells = ring_for(radius)
            available = [
                position for position in coverage_cells if position not in unavailable
            ]
            if len(available) < len(guards):
                continue
            # Keep the vertical axis covered whenever those cells are
            # traversable.  Obstacles on a cardinal ring cell are omitted by
            # ``ring_for`` and therefore do not create an artificial gap.
            cardinal_positions = tuple(
                position
                for dx, dy in (
                    (0, -radius),
                    (0, radius),
                    (radius, 0),
                    (-radius, 0),
                )
                for position in ((core_position[0] + dx, core_position[1] + dy),)
                if position in available
            )
            assignments = _assign_defensive_ring_slots(
                available,
                guards,
                visions,
                anchor_positions=cardinal_positions,
            )
            assignments = _repair_defensive_ring_slots(
                coverage_cells,
                available,
                assignments,
                visions,
                obstacles,
                set(cardinal_positions),
            )
            if not best:
                best = assignments
                best_radius = radius
            if _ring_is_covered(
                coverage_cells,
                assignments,
                visions,
                obstacles,
            ):
                self._defensive_layout = _DefensiveLayout(
                    core_id=core.id,
                    core_position=core_position,
                    guard_ids=ordered_guard_ids,
                    radius=radius,
                    assignments=dict(assignments),
                )
                return dict(assignments)
        if best:
            self._defensive_layout = _DefensiveLayout(
                core_id=core.id,
                core_position=core_position,
                guard_ids=ordered_guard_ids,
                radius=best_radius,
                assignments=dict(best),
            )
        else:
            self._defensive_layout = None
        return dict(best)

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
                UnitType.WORKER: 120,
                UnitType.RANGER: 260,
                UnitType.VANGUARD: 190,
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
            key=lambda position: (
                -self._ranger_range(position, target.position),
                manhattan(ranger.position, position),
                position,
            ),
        )

    @staticmethod
    def _ranger_range(origin: Position, target: Position) -> int:
        return max(abs(target[0] - origin[0]), abs(target[1] - origin[1]))

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

    def _core_escape_direction(
        self,
        context: _TurnContext,
        enemies: tuple[CoreView | UnitView, ...],
    ) -> Direction | None:
        core = context.turn.core
        if core is None:
            return None
        blocked = (
            set(self.memory.obstacles)
            | set(context.turn.obstacle_cells)
            | set(context.turn.resource_cells)
            | context.reserved
        )
        # A friendly unit may vacate the lane in this same plan.  The lane is
        # cleared before the normal unit passes, so treating all transient
        # occupancy as hard terrain would recreate the old boxed-Core failure.
        for enemy in context.turn.visible_enemies:
            blocked.add(enemy.position)
        return self._combat_policy.escape_direction(
            core.position,
            enemies,
            self.memory.obstacles | set(context.turn.obstacle_cells),
            blocked,
            beacon_position=context.turn.beacon.position,
            previous_direction=None,
        )

    def _visible_combat_enemies(
        self,
        turn: Turn,
    ) -> tuple[CoreView | UnitView, ...]:
        """Return visible hostile combat objects relevant to Core survival."""

        return tuple(
            enemy
            for enemy in turn.visible_enemies
            if isinstance(enemy, CoreView)
            or (
                isinstance(enemy, UnitView)
                and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            )
        )

    def _clear_core_escape_lane(self, context: _TurnContext) -> None:
        """Evacuate friendly occupants from the reserved Core escape cell."""

        core = context.turn.core
        direction = context.core_escape_direction
        if core is None or direction is None:
            return
        escape_cell = add(core.position, direction)
        occupants = sorted(
            (unit for unit in context.turn.units if unit.position == escape_cell),
            key=lambda unit: unit.id.bytes,
        )
        if not occupants:
            return
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        enemy_positions = {enemy.position for enemy in context.turn.visible_enemies}
        for unit in occupants:
            for move_direction in DIRECTIONS:
                destination = add(unit.position, move_direction)
                if (
                    destination == core.position
                    or destination in obstacles
                    or destination in enemy_positions
                    or destination in context.reserved
                    or destination in context.occupied
                ):
                    continue
                if self._queue_move(
                    unit,
                    move_direction,
                    context,
                    reason="clear the reserved Core escape lane",
                    target=destination,
                ):
                    break

    def _record_combat_kills(self, turn: Turn) -> None:
        """Log where enemy fighters died, before memory forgets what they were.

        ``WorldMemory.observe`` pops a destroyed enemy at the top of every
        Turn, so the sighting that carries ``unit_type`` only exists until
        then.  Worker kills are ignored: a thief dying inside our own zone
        says nothing about where a hostile fleet came from.
        """

        window = self.config.raid_kill_window
        self._combat_kills = [
            entry for entry in self._combat_kills if turn.tick - entry[0] <= window
        ]
        combat_types = {UnitType.VANGUARD.value, UnitType.RANGER.value}
        for event in turn.events:
            if event.event_type != "DESTRUCTION_PARTICIPATION":
                continue
            if event.target_id is None or event.position is None:
                continue
            sighting = self.memory.enemies.get(str(event.target_id))
            if sighting is None:
                continue
            if sighting.kind == "UNIT" and sighting.unit_type not in combat_types:
                continue
            self._combat_kills.append((turn.tick, event.position))

    def _recent_combat_kills(self, turn: Turn) -> int:
        """Count fighters killed inside the raid trigger window."""

        window = self.config.raid_kill_window
        return sum(1 for tick, _ in self._combat_kills if turn.tick - tick <= window)

    def _combat_leash_radius(
        self,
        unit: Ranger | Vanguard,
        context: _TurnContext,
        *,
        offensive: bool,
    ) -> int:
        """Return how far from the Core this unit may walk to reach a target."""

        if unit.id in context.raid_ids:
            return self.config.raid_radius
        if offensive:
            return self.config.offensive_patrol_radius
        return self.config.combat_pursuit_radius

    def _within_combat_leash(
        self,
        unit: Ranger | Vanguard,
        goal: Position,
        context: _TurnContext,
        *,
        offensive: bool,
    ) -> bool:
        """Whether walking to ``goal`` keeps the combat zone bounded.

        The defensive leash is set by a return race, not by vision.  The
        perimeter caps at twelve cells and a guard on it sees to sixteen, so
        a hostile that enters vision needs about fifteen Ticks to reach the
        Core.  At the measured 0.61 cells/Tick a guard needs roughly 1.64
        Ticks per cell to come home, which runs out of margin at twenty.
        Twenty also stays clear of the enemy Core distances observed at
        twenty-three and beyond, so cracking one stays a job for the raid
        detachment rather than for the whole army.

        Firing is deliberately never leashed - a Ranger standing on the
        boundary must still be able to answer fire from beyond it - so this
        gates movement goals only.
        """

        core = context.turn.core
        if core is None:
            return True
        distance = manhattan(core.position, goal)
        if distance <= self._combat_leash_radius(unit, context, offensive=offensive):
            return True
        # A unit that already drifted outside its leash keeps the right to
        # walk inwards, or it would stall out there for the rest of its life.
        return distance < manhattan(core.position, unit.position)

    def _move_within_leash(
        self,
        unit: Ranger | Vanguard,
        goal: Position,
        context: _TurnContext,
        *,
        offensive: bool,
        reason: str,
    ) -> bool:
        """Move towards ``goal`` only while it stays inside the combat zone."""

        if not self._within_combat_leash(unit, goal, context, offensive=offensive):
            return False
        return self._move(unit, goal, context, reason=reason)

    def _garrison_ids(
        self,
        turn: Turn,
        raid_ids: frozenset[UUID] = frozenset(),
    ) -> frozenset[UUID]:
        """Keep a fixed share of guards on the ring through any emergency.

        ``_visible_combat_targets`` hands every guard the fleet-wide enemy
        list once an emergency starts, so one contact used to pull all of
        them off station.  Contacts arrive from all eight bearings, so that
        leaves the far side of the Core open to the next attacker.  Garrison
        membership comes from the stable UUID split so it cannot flicker
        between Ticks, and it never overlaps the roaming/raid third.
        """

        guards = sorted(
            (
                unit
                for unit in (*turn.rangers, *turn.vanguards)
                if unit.id.int % 3 != 0 and unit.id not in raid_ids
            ),
            key=lambda unit: unit.id.bytes,
        )
        if not guards:
            return frozenset()
        roster = len(turn.rangers) + len(turn.vanguards)
        size = min(
            len(guards),
            max(GARRISON_MIN_GUARDS, roster // GARRISON_ROSTER_SHARE),
        )
        return frozenset(unit.id for unit in guards[:size])

    def _update_raid(self, turn: Turn, threat: ThreatAssessment) -> frozenset[UUID]:
        """Run the bounded counter-attack that hunts a nearby enemy Core.

        The engagement this rule comes from ended with one Vanguard adjacent
        to an enemy Core and two Rangers in range, so four units are enough
        to crack a 5/5 Core once its escort is dead.  Committing the whole
        fleet emptied the perimeter for no extra damage, so the raid is
        capped in size, in distance (``raid_radius``) and in time
        (``raid_max_ticks``), and it turns around rather than fight a
        garrison it cannot eat.
        """

        core = turn.core
        live = frozenset(unit.id for unit in (*turn.vanguards, *turn.rangers))
        self._raid_ids &= live
        if core is None:
            self._abort_raid(turn)
            return frozenset()
        if self._raid_ids:
            objective = self._raid_target_for(turn, self.config.raid_radius)
            if (
                turn.tick > self._raid_until_tick
                or threat.combat_pressure
                or core.hp < 5
                or core.shield < 5
                or self._raid_outmatched(turn)
                or self._raid_objective_spent(turn)
            ):
                self._abort_raid(turn)
                return frozenset()
            if objective is not None:
                self._raid_target = objective
            return self._raid_ids
        if (
            turn.tick <= self._raid_cooldown_until
            or threat.combat_pressure
            or self._emergency_combat_mode(turn)
            or not self._offensive_patrol_enabled(turn)
        ):
            return frozenset()
        objective = self._raid_target_for(
            turn,
            self.config.raid_opportunity_radius,
        )
        if objective is None:
            return frozenset()
        squad = self._raid_squad(turn)
        if len(squad) < self.config.raid_squad_size:
            return frozenset()
        self._raid_ids = frozenset(unit.id for unit in squad)
        self._raid_target = objective
        self._raid_until_tick = turn.tick + self.config.raid_max_ticks
        self._combat_kills.clear()
        return self._raid_ids

    def _raid_target_for(self, turn: Turn, radius: int) -> Position | None:
        """Pick the objective: a known Core inside ``radius``, else the bearing.

        The two callers pass different radii on purpose.  Launching on a
        Core we merely happen to remember is an opportunistic decision, so it
        stays inside the tight ``raid_opportunity_radius``; enemy Core
        memory lives for thousands of Ticks, and a wide trigger here would
        turn the detachment into a permanent standing expedition instead of
        the answer to a fleet we just beat.  Refreshing the objective of a
        raid that is already committed uses the full ``raid_radius``.
        """

        core = turn.core
        if core is None:
            return None
        known = self._remembered_enemy_cores(turn, radius)
        if known:
            return min(
                known,
                key=lambda sighting: (
                    manhattan(core.position, sighting.position),
                    sighting.object_id,
                ),
            ).position
        if self._recent_combat_kills(turn) < self.config.raid_trigger_kills:
            return None
        return self._kill_bearing_goal(turn)

    def _remembered_enemy_cores(
        self,
        turn: Turn,
        radius: int,
    ) -> tuple[EnemySighting, ...]:
        """Return remembered enemy Cores within ``radius`` of our own Core."""

        core = turn.core
        if core is None:
            return ()
        return tuple(
            sighting
            for sighting in self.memory.recent_enemies(
                turn.tick,
                self.config.enemy_core_memory_ttl,
            )
            if sighting.kind == "CORE"
            and manhattan(core.position, sighting.position) <= radius
        )

    def _kill_bearing_goal(self, turn: Turn) -> Position | None:
        """Extend the Core-to-kill-centroid bearing out to the raid radius.

        Contacts arrive from every bearing, so there is no standing direction
        worth fortifying; the only defensible heading for a search is the one
        the fleet we just beat actually came from.  The reach is
        ``raid_radius``: measured enemy Core distances cluster at 20-30 and
        again at 42-47 with nothing in between, so stopping at 30 gave up the
        second cluster for no saving, while 48 keeps the round trip inside
        ``raid_max_ticks`` and never sends the detachment past the ring the
        roaming squad already patrols.
        """

        core = turn.core
        if core is None:
            return None
        window = self.config.raid_kill_window
        cells = [
            position
            for tick, position in self._combat_kills
            if turn.tick - tick <= window
        ]
        if not cells:
            return None
        offset_x = sum(cell[0] for cell in cells) / len(cells) - core.position[0]
        offset_y = sum(cell[1] for cell in cells) / len(cells) - core.position[1]
        span = abs(offset_x) + abs(offset_y)
        if span == 0:
            return None
        scale = self.config.raid_radius / span
        return (
            core.position[0] + round(offset_x * scale),
            core.position[1] + round(offset_y * scale),
        )

    def _raid_members(self, turn: Turn) -> tuple[Ranger | Vanguard, ...]:
        """Return the live units currently assigned to the raid."""

        return tuple(
            unit
            for unit in (*turn.rangers, *turn.vanguards)
            if unit.id in self._raid_ids
        )

    def _raid_squad(self, turn: Turn) -> tuple[Ranger | Vanguard, ...]:
        """Pick the smallest detachment that can still crack a Core.

        One Ranger buys the Manhattan-5 vision disc a search needs; the rest
        are Vanguards, which carry twice the hit points and did the actual
        Core damage in the engagement this rule was derived from.
        """

        def eligible(unit: Ranger | Vanguard) -> bool:
            return (
                self._is_offensive_combat_unit(unit, turn)
                and unit.id not in self._squad_return_until
            )

        rangers = sorted(
            (unit for unit in turn.rangers if eligible(unit)),
            key=lambda unit: unit.id.bytes,
        )
        vanguards = sorted(
            (unit for unit in turn.vanguards if eligible(unit)),
            key=lambda unit: unit.id.bytes,
        )
        size = self.config.raid_squad_size
        squad: list[Ranger | Vanguard] = list(rangers[:1])
        squad.extend(vanguards[: size - len(squad)])
        squad.extend(rangers[1 : size - len(squad) + 1])
        return tuple(squad[:size])

    def _raid_outmatched(self, turn: Turn) -> bool:
        """Whether the objective is held by more fighters than we sent.

        A four-unit detachment trades evenly at best away from home, so an
        equal count is already a reason to turn around rather than a reason
        to commit.
        """

        members = self._raid_members(turn)
        if not members:
            return True
        hostiles = sum(
            1
            for enemy in turn.visible_enemies
            if isinstance(enemy, UnitView)
            and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and any(
                manhattan(member.position, enemy.position) <= RAID_ABORT_SCAN_RADIUS
                for member in members
            )
        )
        return hostiles >= len(members)

    def _raid_objective_spent(self, turn: Turn) -> bool:
        """Whether the detachment arrived and found nothing left to kill."""

        target = self._raid_target
        if target is None:
            return True
        members = self._raid_members(turn)
        if not any(
            manhattan(member.position, target) <= RANGER_VISION_RADIUS
            for member in members
        ):
            return False
        return not self._remembered_enemy_cores(turn, self.config.raid_radius)

    def _abort_raid(self, turn: Turn) -> None:
        """Send the detachment home and hold off relaunching for a while."""

        for unit_id in self._raid_ids:
            self._squad_return_until[unit_id] = max(
                self._squad_return_until.get(unit_id, 0),
                turn.tick + self.config.raid_max_ticks,
            )
        if self._raid_ids:
            self._raid_cooldown_until = turn.tick + self.config.raid_max_ticks // 2
        self._raid_ids = frozenset()
        self._raid_target = None
        self._raid_until_tick = -1

    def _update_squad_returns(
        self,
        turn: Turn,
        threat: ThreatAssessment,
        raid_ids: frozenset[UUID] = frozenset(),
    ) -> frozenset[UUID]:
        """Recall an intercepted roaming pair for a bounded return window.

        The raid detachment is exempt: making contact is the point of the
        raid, so the generic intercept recall would cancel it on arrival.
        ``_update_raid`` owns that decision instead.
        """

        live_ids = {unit.id for unit in (*turn.vanguards, *turn.rangers)}
        self._squad_return_until = {
            unit_id: until
            for unit_id, until in self._squad_return_until.items()
            if unit_id in live_ids and until >= turn.tick
        }
        if turn.core is None:
            self._squad_return_until.clear()
            return frozenset()

        visible_combat = tuple(
            enemy
            for enemy in turn.visible_enemies
            if isinstance(enemy, UnitView)
            and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
        )
        if threat.level is not ThreatLevel.NORMAL:
            for unit in (*turn.vanguards, *turn.rangers):
                # Keep the deterministic one-third expedition identity used by
                # the existing patrol, but recall it as soon as it is locally
                # intercepted instead of chasing through the contact.
                if unit.id.int % 3 != 0 or unit.id in raid_ids:
                    continue
                if any(
                    manhattan(unit.position, enemy.position) <= 5
                    for enemy in visible_combat
                ):
                    self._squad_return_until[unit.id] = max(
                        self._squad_return_until.get(unit.id, 0),
                        turn.tick + 8,
                    )

        for unit_id in tuple(self._squad_return_until):
            unit = next(
                (
                    candidate
                    for candidate in (*turn.vanguards, *turn.rangers)
                    if candidate.id == unit_id
                ),
                None,
            )
            if unit is not None and manhattan(unit.position, turn.core.position) <= 3:
                self._squad_return_until.pop(unit_id, None)
        return frozenset(self._squad_return_until)

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
        ):
            if current.last_progress_position is None:
                current = UnitGoal(
                    position=current.position,
                    assigned_tick=current.assigned_tick,
                    purpose=current.purpose,
                    last_progress_position=worker.position,
                )
                self.memory.set_goal(unit_id, current)
            if turn.tick - current.assigned_tick <= self.config.exploration_goal_ttl:
                return current.position
            # A patrol point can be further away in travel Ticks than the goal
            # TTL allows.  Expiring it mid-transit made Workers rotate to the
            # next point before ever arriving, so the sweep degenerated into
            # walking.  Renew the goal while the Worker is still closing in.
            progress_reference = current.last_progress_position
            if progress_reference is not None and manhattan(
                worker.position, current.position
            ) < manhattan(progress_reference, current.position):
                renewed = UnitGoal(
                    position=current.position,
                    assigned_tick=turn.tick,
                    purpose=current.purpose,
                    last_progress_position=worker.position,
                )
                self.memory.set_goal(unit_id, renewed)
                return renewed.position

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
        emergency = self._emergency_combat_mode(turn)
        if emergency:
            candidates = (
                (UnitType.RANGER, UnitType.VANGUARD)
                if len(turn.rangers) * 2 < len(turn.vanguards)
                else (UnitType.VANGUARD, UnitType.RANGER)
            )
        elif self._needs_resource_guard(turn):
            if not turn.vanguards:
                candidates = (UnitType.VANGUARD,)
            elif not turn.rangers:
                # Once the first melee guard exists, add ranged damage before
                # spending the rest of the low-capacity economy on Workers.
                # Fall back to a second Vanguard when the cheaper candidate
                # is the only affordable combat option.
                candidates = (UnitType.RANGER, UnitType.VANGUARD)
            else:
                candidates = (UnitType.VANGUARD, UnitType.RANGER)
        elif len(turn.workers) < self.config.target_workers:
            candidates = (UnitType.WORKER,)
        elif not turn.vanguards:
            candidates = (UnitType.VANGUARD, UnitType.RANGER)
        elif len(turn.rangers) < len(turn.vanguards) * 2:
            candidates = (UnitType.RANGER, UnitType.VANGUARD)
        else:
            candidates = (UnitType.VANGUARD, UnitType.RANGER)
        return self._affordable_spawn(
            turn,
            resources,
            candidates,
            strict_preference=not emergency,
        )

    def _affordable_spawn(
        self,
        turn: Turn,
        resources: int,
        candidates: tuple[UnitType, ...],
        *,
        strict_preference: bool,
    ) -> UnitType | None:
        """Return the first candidate the Core can fund without losing intent.

        Taking the first affordable candidate silently inverts the policy: a
        Ranger costs more than a Vanguard, so the melee fallback always won
        the race and the roster drifted to almost pure Vanguards even while
        the policy asked for twice as many Rangers.  Outside emergencies the
        Core saves for its preferred Unit instead, and only falls back when
        that Unit cannot fit in Core storage at all.
        """

        def affordable(unit_type: UnitType) -> bool:
            cost = unit_cost(unit_type, turn.state.population)
            reserve = self._spawn_safety_reserve(
                turn,
                production_cost=cost,
                available_resources=resources,
            )
            return cost + reserve <= resources

        for index, unit_type in enumerate(candidates):
            if affordable(unit_type):
                return unit_type
            if (
                index == 0
                and strict_preference
                and unit_cost(unit_type, turn.state.population)
                <= turn.resource_capacity
            ):
                return None
        return None

    def _needs_resource_guard(self, turn: Turn) -> bool:
        if not self._preserves_resources() or turn.core is None:
            return False
        combat_units = len(turn.vanguards) + len(turn.rangers)
        if (
            self._unbounded_growth()
            and turn.resource_capacity < CORE_CAPACITY_FAST_EXPANSION
            and not combat_units
            and turn.workers
        ):
            # After a Core respawn, spending the first five resources on a
            # second Worker leaves the base with no combat answer while it
            # waits for the next ten-resource Vanguard.  Establish the first
            # guard before widening the economy beyond its first Worker.
            return True
        if (
            self._unbounded_growth()
            and turn.resource_capacity < CORE_CAPACITY_FAST_EXPANSION
            and 0 < combat_units < EARLY_COMBAT_MIN_UNITS
            and self.config.target_workers > 0
            and len(turn.workers)
            >= min(
                self.config.target_workers,
                self.config.resource_guard_min_workers,
                EARLY_COMBAT_GUARD_WORKERS,
            )
        ):
            return True
        if turn.vanguards:
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
                (
                    enemy.kind == "CORE"
                    and turn.tick - enemy.tick <= self.config.enemy_memory_ttl
                )
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

    def _spawn_safety_reserve(
        self,
        turn: Turn,
        *,
        production_cost: int | None = None,
        available_resources: int | None = None,
    ) -> int:
        """Keep enough Core resources for a full emergency recovery cycle.

        The reserve is only added by the unbounded live mode. Explicit legacy
        resource-target configurations retain their historical exact-cost
        behavior so callers can reproduce prior experiments. In the live mode,
        the stockpile follows the live Core capacity tiers: small Cores expand
        quickly, then preserve 50, 95, or 100 resources as storage grows. The
        base reserve and any missing recovery resources always win.
        """

        if not self._unbounded_growth():
            return 0
        core = turn.core
        if core is None:
            return max(0, self.config.safety_reserve)
        missing_recovery = max(0, 5 - core.hp) + max(0, 5 - core.shield)
        if self._emergency_combat_mode(turn):
            return max(self.config.safety_reserve, missing_recovery)
        reserve = max(
            0,
            self.config.safety_reserve,
            missing_recovery,
            self._stockpile_target(turn),
        )
        if production_cost is not None:
            # In the low-capacity growth tier, a healthy Core should spend
            # every surplus resource on the next Unit.  Keeping the generic
            # ten-resource reserve here strands a 15-capacity Core at
            # 13/15 or 14/15: it has enough for a Worker, but the reserve
            # makes the production check fail.  Emergency combat mode already
            # returned above, so a damaged or actively threatened Core keeps
            # its recovery reserve.
            if (
                turn.resource_capacity < CORE_CAPACITY_FAST_EXPANSION
                and available_resources is not None
                and available_resources <= turn.resource_capacity
            ):
                return max(missing_recovery, self._stockpile_target(turn))
            # At an exact capacity-tier boundary, ``reserve + cost`` can be
            # larger than the entire Core.  Permit the smallest transition
            # spend needed to cross that boundary; otherwise a no-cap strategy
            # would deadlock permanently at capacities 50, 95, or 100.  A
            # freshly respawned Core has capacity 10, where the base reserve
            # itself would make the first 5-resource Worker impossible to
            # produce.  Lower that reserve only when the Core can already
            # afford its full recovery cycle; damaged Cores must still wait
            # for recovery resources.
            max_affordable_reserve = max(
                0,
                turn.resource_capacity - production_cost,
            )
            if missing_recovery <= max_affordable_reserve and (
                available_resources is None
                or available_resources <= turn.resource_capacity
            ):
                reserve = min(reserve, max_affordable_reserve)
        return reserve

    def _stockpile_target(self, turn: Turn) -> int:
        """Return the shop-aligned live stockpile target for Core capacity."""

        capacity = turn.resource_capacity
        if capacity < CORE_CAPACITY_FAST_EXPANSION:
            return 0
        if capacity < CORE_CAPACITY_MEDIUM_RESERVE:
            return 50
        if capacity <= CORE_CAPACITY_HIGH_RESERVE:
            return 95
        # Above the high tier a flat target pinned the live stockpile just
        # over 100 forever: every deposit past ``target + unit_cost`` funded
        # the next Unit immediately, so almost all income bought roster while
        # the bank -- the only source of HEAL and REPAIR_SHIELD -- never grew.
        # Scaling with the storage the roster already paid for keeps a real
        # emergency reserve, and the banking tier slows growth once the army
        # is large enough to hold the Core on its own.
        percent = (
            CORE_BANKING_CAPACITY_PERCENT
            if self._growth_slowdown_active(turn)
            else CORE_STOCKPILE_CAPACITY_PERCENT
        )
        return max(CORE_CAPACITY_HIGH_RESERVE, capacity * percent // 100)

    def _growth_slowdown_active(self, turn: Turn) -> bool:
        """Return whether growth should yield to banking at this population.

        Unit prices rise 1.3x per five population while each Unit only adds
        five storage, so past the soft threshold every additional guard costs
        much more and adds very little to a Core that already holds a full
        defensive ring.  Real enemy pressure lifts the slowdown so the bank
        can be spent on defenders at once.
        """

        threshold = self.config.growth_slowdown_population
        return (
            threshold is not None
            and self._unbounded_growth()
            and turn.state.population >= threshold
            and not self._emergency_combat_mode(turn)
        )

    def _unbounded_growth(self) -> bool:
        """Return whether the live strategy has no fixed population/stockpile goal."""

        return self.config.max_population is None and self.config.resource_target <= 0

    def _spawn_reason(self, unit_type: UnitType) -> str:
        if self._unbounded_growth():
            return (
                "expand storage with the next available Core resources "
                f"using {unit_type.value.lower()}"
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
    *,
    anchor_positions: tuple[Position, ...] = (),
) -> dict[UUID, Position]:
    """Divide ring cells into vision-sized sectors and pick each sector's center.

    The vertical cardinal cells are reserved first (when supplied by the
    obstacle-aware ring builder).  This prevents a perfectly covered
    Manhattan ring from looking open at its top or bottom simply because a
    sector's mathematical center landed on a diagonal cell.
    """

    ordered = tuple(
        sorted(
            guards,
            key=lambda unit: (-visions[unit.id], unit.id.bytes),
        )
    )
    assignments: dict[UUID, Position] = {}
    available_anchors = tuple(
        position for position in anchor_positions if position in cells
    )
    anchor_count = min(len(ordered), len(available_anchors))
    if anchor_count:
        # Start with evenly spaced ring indices, then replace the nearest
        # non-anchor slots with the required cardinal cells.  This keeps the
        # sectors balanced while making the vertical axis explicit.
        ordered_cells = list(cells)
        cell_index = {position: index for index, position in enumerate(ordered_cells)}
        selected = {
            ordered_cells[(index * len(ordered_cells)) // len(ordered)]
            for index in range(len(ordered))
        }
        fixed = set(available_anchors[:anchor_count])
        for anchor in available_anchors[:anchor_count]:
            if anchor in selected:
                continue
            anchor_index = cell_index[anchor]
            replacement = min(
                (position for position in selected if position not in fixed),
                key=lambda position: min(
                    (cell_index[position] - anchor_index) % len(ordered_cells),
                    (anchor_index - cell_index[position]) % len(ordered_cells),
                ),
            )
            selected.remove(replacement)
            selected.add(anchor)
        slots = list(available_anchors[:anchor_count])
        slots.extend(
            position
            for position in ordered_cells
            if position in selected and position not in fixed
        )
        for unit, position in zip(ordered, slots, strict=True):
            assignments[unit.id] = position
        return assignments

    total_capacity = sum(2 * visions[unit.id] for unit in ordered)
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


def _repair_defensive_ring_slots(
    cells: tuple[Position, ...],
    available: list[Position],
    assignments: dict[UUID, Position],
    visions: dict[UUID, int],
    obstacles: set[Position],
    anchors: set[Position],
) -> dict[UUID, Position]:
    """Repair a cardinal layout when sector rounding leaves a small gap."""

    if not assignments:
        return assignments
    visibility_cache: dict[tuple[Position, int], set[Position]] = {}

    def visible_from(position: Position, vision: int) -> set[Position]:
        key = position, vision
        if key not in visibility_cache:
            visibility_cache[key] = {
                cell
                for cell in cells
                if manhattan(position, cell) <= vision
                and _clear_manhattan_path(position, cell, obstacles)
            }
        return visibility_cache[key]

    def coverage_count(layout: dict[UUID, Position]) -> int:
        covered: set[Position] = set()
        for unit_id, position in layout.items():
            covered.update(visible_from(position, visions[unit_id]))
        return len(covered)

    current = coverage_count(assignments)
    while current < len(cells):
        used = set(assignments.values())
        best: tuple[int, bytes, Position, UUID] | None = None
        for unit_id, position in assignments.items():
            if position in anchors:
                continue
            for candidate in available:
                if candidate in used:
                    continue
                trial = dict(assignments)
                trial[unit_id] = candidate
                score = coverage_count(trial)
                choice = (score, unit_id.bytes, candidate, unit_id)
                if best is None or choice[:3] > best[:3]:
                    best = choice
        if best is None or best[0] <= current:
            break
        current, _, candidate, unit_id = best
        assignments[unit_id] = candidate
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
