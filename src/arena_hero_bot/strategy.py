"""Aggressive v1 tactic built on the official Arena Hero Turn controls."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from itertools import pairwise
from math import atan2
from uuid import UUID

from arena_hero import (
    BeaconStatus,
    Core,
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
    core_resource_capacity,
    unit_cost,
)

from .combat_policy import (
    CombatPolicy,
    CombatTargetLedger,
    ThreatAssessment,
    ThreatLevel,
)
from .exploration import ExpeditionNavigator, ExplorationRoute
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
from .memory import (
    RESOURCE_MEMORY_LIMIT,
    RESOURCE_RECHECK_FLOOR,
    EnemySighting,
    ExpeditionSquad,
    UnitGoal,
    WorldMemory,
)
from .models import DecisionReport

EXPLORATION_PURPOSE = "explore-center-v3"
RESOURCE_CLAIM_PURPOSE = "resource-claim-v1"
RESOURCE_PATROL_PURPOSE = "resource-patrol-v3"
COMBAT_PATROL_PURPOSE = "combat-patrol-v1"
RESOURCE_CLAIM_TTL = 4
CLAIM_STALL_BUDGET = 3
UNREACHABLE_CLAIM_COOLDOWN = 128
# Resource replenishment resolves every four Ticks.  On a saturated map,
# reserve one Worker for a refresh every twelve Ticks; between refreshes it can
# service the known resource pool instead of paying a full Worker of income
# for continuous long-range scouting.
RESOURCE_SCOUT_INTERVAL = 12
# A remote claim is still valid, but its loaded return to the Core is the
# limiting leg of a refill cycle.  Count that leg with a higher weight for sites outside
# the local patrol ring; this remains a soft preference and never removes a
# far site from the candidate pool.
REMOTE_RETURN_WEIGHT = 81
# Keep the established local-ring round-trip price independent of the remote
# fallback tuning above; local pairing also covers unreachable-route guards.
LOCAL_RETURN_WEIGHT = 3
RESOURCE_PATROL_SPACING = 6
COMBAT_PATROL_SPACING = 12
DEFENSIVE_PERIMETER_MIN_RADIUS = 2
# Guards sit three cells apart along a ring and the rings themselves are three
# cells apart, so a Ranger (vision 5) still sees both neighbours and the next
# ring inward.  One slot per cell used to push the whole roster onto a single
# circle: 34 of 49 live combat units stood at exactly Manhattan 12 with nothing
# behind them, so anything that slipped through the circle met an empty
# interior.
DEFENSIVE_RING_SPACING = 3
VANGUARD_VISION_RADIUS = 4
RANGER_VISION_RADIUS = 5
RANGER_STANDOFF_RANGE = 3
RANGER_RETURN_THREAT_RADIUS = 5
COMBAT_SCREEN_RADIUS = 3
CORE_ESCAPE_MIN_ENEMIES = 3
MAX_VANGUARD_ATTACKERS = 4
CORE_WORKER_INTERCEPT_RADIUS = 4
CORE_INTRUDER_INTERCEPTORS = 4
PERIMETER_INTERCEPTORS = 6
PERIMETER_INTERCEPT_MEMORY_TICKS = 8
PERIMETER_INTERCEPT_MAX_ENEMIES = 2
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
# Active-offense ("expedition") posture: a fixed formation saturates the Core
# neighbourhood while everything built beyond it is sent out on independent,
# non-returning expeditions.  Formation target is 12 Workers / 16 Vanguards /
# Legacy posture: 32 Rangers; 13 Vanguards + 26 Rangers hold the defensive
# ring, and three independent patrol teams (one Vanguard + two Rangers each)
# patrol just
# outside it.  Any surplus is staged at the ring edge until a full expedition
# (2 Vanguards + 2 Rangers) can depart.
EXPEDITION_FORMATION_VANGUARDS = 16
EXPEDITION_FORMATION_RANGERS = 32
EXPEDITION_SQUAD_VANGUARDS = 2
EXPEDITION_SQUAD_RANGERS = 2
PATROL_TEAM_COUNT = 3
PATROL_TEAM_SIZE = 3
PATROL_TEAM_VANGUARDS = 1
PATROL_TEAM_RANGERS = 2
DEFENSE_VANGUARDS = EXPEDITION_FORMATION_VANGUARDS - (
    PATROL_TEAM_COUNT * PATROL_TEAM_VANGUARDS
)
DEFENSE_RANGERS = EXPEDITION_FORMATION_RANGERS - (
    PATROL_TEAM_COUNT * PATROL_TEAM_RANGERS
)
DEFENSE_ROLE = "defense"
STAGED_ROLE = "staged"
PATROL_ROLE_PREFIX = "patrol-"
EXPEDITION_ROLE_PREFIX = "expedition-"
EXPEDITION_HORIZON = 256
EXPEDITION_PURSUIT_MIN_DISTANCE = 100
# Staging cells sit just outside the defensive ring's maximum radius.
EXPEDITION_STAGING_RADIUS = 13
# A legacy expedition member can be hundreds of cells away when it rejoins the
# staging pool.  The default 4,096-node A* budget was measured stopping inside
# a six-cell loop on such a return, while 8,192 found the real route in about
# 50 ms.  Apply the larger budget only to staged Units so normal fleet planning
# keeps its existing bound.
EXPEDITION_STAGING_PATH_EXPANSIONS = 8192
# A separated expedition's rejoining members need the same bounded long-route
# budget. Quiet pursuit otherwise falls into a greedy loop before reaching
# the teammate it is meant to accompany.
EXPEDITION_REJOIN_PATH_EXPANSIONS = 8192
# Maximum Manhattan distance between a member and its nearest teammate before
# the member stops to let the gap close.  This keeps *adjacent* members within
# sight (Vanguard vision is 4, Ranger vision is 5) so the squad stays a
# connected chain, without collapsing the whole squad onto one cell.
EXPEDITION_LINK_RADIUS = 4
# An expedition member under fire closes on a hostile that close (inside the
# squad's own sight line) instead of standing still for cohesion; when the
# attacker stays beyond sweep reach and keeps firing, the member breaks
# contact instead of absorbing shots.  A 1HP enemy Ranger kiting a
# full-HP Vanguard from range 2-3 used to kill it in place because expedition
# members skipped all the visible-target combat logic.  This radius is the
# window a member may still close on to answer that fire.
# Three combat units inside the expedition's counter window can cover every
# useful approach cell.  Treat that as an encirclement before the normal
# close-on-target branch, even while the member is still at full HP.

# Revised live posture.  The legacy constants above remain available for
# explicit small test configurations; ``target_workers >= 16`` selects this
# formation and its four quadrant patrol teams.
SYMMETRIC_FORMATION_VANGUARDS = 12
SYMMETRIC_FORMATION_RANGERS = 24
SYMMETRIC_PATROL_TEAM_COUNT = 4
SYMMETRIC_DEFENSE_VANGUARDS = 8
SYMMETRIC_DEFENSE_RANGERS = 16
SYMMETRIC_CORE_RADIUS = 6
CORE_QUEUE_RADIUS = 4


class _MoveIntent(StrEnum):
    """Movement class used by the transient friendly-traffic ledger."""

    TRANSIT = "transit"
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    STAND = "stand"


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
    growth_slowdown_population: int | None = 38
    safety_reserve: int = 10
    resource_patrol_radius: int = 14
    resource_outreach_radius: int = 48
    offensive_patrol_radius: int = 60
    offensive_squad_size: int = 8
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
    expedition_mode: bool = False


@dataclass(slots=True)
class _TurnContext:
    turn: Turn
    report: DecisionReport
    occupied: set[Position]
    enemy_positions: set[Position]
    reserved: set[Position] = field(default_factory=set)
    friendly_occupancy: dict[Position, int] = field(default_factory=dict)
    arrival_counts: dict[Position, int] = field(default_factory=dict)
    departure_counts: dict[Position, int] = field(default_factory=dict)
    standing_reserved: set[Position] = field(default_factory=set)
    planned_ids: set[UUID] = field(default_factory=set)
    core_inbound_id: UUID | None = None
    core_service_action: str | None = None
    resource_assignments: dict[UUID, Position] = field(default_factory=dict)
    defensive_assignments: dict[UUID, Position] | None = None
    remaining_resources: int = 0
    remaining_resource_space: int = 0
    beacon_claimed: bool = False
    focus_target: CoreView | UnitView | None = None
    vanguard_attack_ids: frozenset[UUID] = field(default_factory=frozenset)
    assault_enemies: tuple[CoreView | UnitView, ...] = ()
    screen_assignments: dict[UUID, Position] = field(default_factory=dict)
    combat_assault: bool = False
    threat: ThreatAssessment = field(default_factory=ThreatAssessment)
    damage_ledger: CombatTargetLedger = field(default_factory=CombatTargetLedger)
    squad_return_ids: frozenset[UUID] = field(default_factory=frozenset)
    intruder_intercept_ids: frozenset[UUID] = field(default_factory=frozenset)
    perimeter_intercept_ids: frozenset[UUID] = field(default_factory=frozenset)
    perimeter_intercept_anchor: Position | None = None
    emergency: bool = False
    garrison_ids: frozenset[UUID] = field(default_factory=frozenset)
    raid_ids: frozenset[UUID] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Seed the friendly occupancy ledger for hand-built test contexts."""

        if self.friendly_occupancy:
            return
        for unit in self.turn.units:
            self.friendly_occupancy[unit.position] = (
                self.friendly_occupancy.get(unit.position, 0) + 1
            )
        if self.turn.core is not None:
            self.friendly_occupancy[self.turn.core.position] = (
                self.friendly_occupancy.get(self.turn.core.position, 0) + 1
            )


@dataclass(slots=True)
class _ExpeditionPursuit:
    target_id: str | None = None
    position: Position | None = None
    direction: Position = (0, 0)
    distance: int = 0
    last_positions: dict[UUID, Position] = field(default_factory=dict)


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
        # Cache the legacy perimeter layout.  The symmetric formation also
        # stores each Unit's exact post in WorldMemory so assignments survive
        # combat, casualties, and service restarts.
        self._defensive_layout: _DefensiveLayout | None = None
        # The roaming squad, cached against the roster it was drawn from so
        # membership only changes when a combat unit is built or dies.
        self._offensive_squad: tuple[frozenset[UUID], frozenset[UUID]] | None = None
        self._combat_focus_id: str | None = None
        self._intruder_hunt_id: str | None = None
        self._combat_policy = CombatPolicy()
        self._squad_return_until: dict[UUID, int] = {}
        # The last cell a lone fighter was seen in inside the defensive zone,
        # kept so the converging squad does not disband on the first dark Tick.
        self._perimeter_intercept: tuple[str, Position, int] | None = None
        # Claim hysteresis, kept off ``WorldMemory`` for the same reason as
        # the raid state below: a restart forgets it and the next Tick simply
        # re-matches every Worker once.
        self._claim_stalls: dict[UUID, int] = {}
        # Set for the current Tick only by ``_assign_resources`` when a
        # Worker is deliberately left out of known-site pairing.  Keeping the
        # identity here lets the selected Worker be non-minimal without
        # turning an ordinary remote-resource fallback into a scout.
        self._resource_scout_id: UUID | None = None
        # Older memory files have no production marker.  On the first
        # high-tier live Tick, conservatively start the marker at the observed
        # bank so a deployment cannot immediately repeat the last spend; the
        # marker is then durable for all later restarts.
        self._needs_growth_marker_migration = self.memory.last_tick > 0 and (
            self.memory.last_normal_growth_tick is None
            or self.memory.last_normal_growth_resources is None
        )
        # Cells a Worker could not route to, kept off ``WorldMemory`` for the
        # same reason: reachability is a property of the current obstacle and
        # threat picture, so a restart should re-ask the question rather than
        # inherit an answer.
        self._unreachable_claims: dict[Position, int] = {}
        # Raid state deliberately lives on the strategy instead of
        # ``WorldMemory``: a restart then aborts the raid and walks the
        # detachment home, which is the safe direction to fail in.
        self._combat_kills: list[tuple[int, Position]] = []
        self._raid_ids: frozenset[UUID] = frozenset()
        self._raid_target: Position | None = None
        self._raid_until_tick: int = -1
        self._raid_cooldown_until: int = -1
        # Expedition posture state.  Surplus combat units are staged at the
        # defensive ring edge until a full expedition can depart.  Each
        # expedition keeps its membership, pursuit and shared exploration
        # route. Its bearing is the current formation heading, not a mandate
        # to march ever farther in the original direction.
        self._staged_ids: frozenset[UUID] = frozenset()
        self._expedition_squads: list[tuple[frozenset[UUID], Position]] = []
        self._expedition_serial_by_members: dict[frozenset[UUID], int] = {}
        self._expedition_pursuits: dict[frozenset[UUID], _ExpeditionPursuit] = {}
        self._expedition_routes: dict[frozenset[UUID], ExplorationRoute] = {}
        self._expedition_regroup_orders: dict[frozenset[UUID], tuple[UUID, ...]] = {}
        self._expedition_navigation_turn: Turn | None = None
        self._expedition_serial: int = self.memory.next_expedition_serial
        for persisted in self.memory.expedition_squads:
            try:
                members = frozenset(UUID(member) for member in persisted.members)
            except (TypeError, ValueError):
                continue
            if members:
                self._expedition_squads.append((members, persisted.bearing))
                self._expedition_serial_by_members[members] = persisted.serial
                self._expedition_routes[members] = persisted.exploration_route
                self._expedition_regroup_orders[members] = tuple(
                    UUID(member)
                    for member in persisted.regroup_order
                    if member in persisted.members
                )
                self._expedition_pursuits[members] = _ExpeditionPursuit(
                    target_id=persisted.pursuit_target_id,
                    position=persisted.pursuit_position,
                    direction=persisted.pursuit_direction,
                    distance=persisted.pursuit_distance,
                )
        self._expedition_serial = max(
            self._expedition_serial,
            max((squad.serial for squad in self.memory.expedition_squads), default=0),
        )

    def decide(self, turn: Turn) -> DecisionReport:
        """Queue one complete aggressive plan for the current Turn."""

        turn.clear()
        # ``observe`` drops destroyed enemies from memory, taking the only
        # record of what they were with them, so log the kills first.
        self._record_combat_kills(turn)
        self.memory.observe(turn)
        if self.config.expedition_mode:
            self._reconcile_unit_roles(turn)
            self._update_expeditions(turn)
            self._refresh_expedition_pursuits(turn)
            self._refresh_expedition_navigation(turn)
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
        friendly_occupancy: dict[Position, int] = {}
        for unit in turn.units:
            friendly_occupancy[unit.position] = (
                friendly_occupancy.get(unit.position, 0) + 1
            )
        if turn.core is not None:
            friendly_occupancy[turn.core.position] = (
                friendly_occupancy.get(turn.core.position, 0) + 1
            )
        focus_target = self._focus_target(turn)
        raid_ids = self._update_raid(turn, threat)
        garrison_ids = self._garrison_ids(turn, raid_ids)
        perimeter_anchor = self._perimeter_intercept_anchor(turn, emergency)
        perimeter_intercept_ids = self._perimeter_intercept_ids(
            turn,
            perimeter_anchor,
            garrison_ids,
            raid_ids,
        )
        squad_return_ids = self._update_squad_returns(
            turn,
            threat,
            raid_ids,
            perimeter_intercept_ids,
        )
        context = _TurnContext(
            turn=turn,
            report=report,
            occupied=occupied,
            enemy_positions={enemy.position for enemy in turn.visible_enemies},
            friendly_occupancy=friendly_occupancy,
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
            perimeter_intercept_ids=perimeter_intercept_ids,
            perimeter_intercept_anchor=perimeter_anchor,
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
        if self._symmetric_posture():
            # Establish home posts even when every defender is fighting or
            # healing and none will reach the idle movement branch this Tick.
            context.defensive_assignments = self._defensive_perimeter_assignments(
                context
            )
        if context.combat_assault and context.assault_enemies:
            context.screen_assignments = self._combat_screen_assignments(context)

        for ranger in sorted(turn.rangers, key=lambda unit: unit.id.bytes):
            if ranger.id in context.planned_ids:
                continue
            self._decide_ranger(
                ranger,
                context,
                tactical_enemies,
                offensive=self._is_offensive_combat_unit(ranger, turn, threat),
            )
        for vanguard in sorted(turn.vanguards, key=lambda unit: unit.id.bytes):
            if vanguard.id in context.planned_ids:
                continue
            self._decide_vanguard(
                vanguard,
                context,
                tactical_enemies,
                offensive=self._is_offensive_combat_unit(vanguard, turn, threat),
            )
        self._prepare_core_departure(context)
        core_position = turn.core.position if turn.core is not None else None
        for worker in sorted(
            turn.workers,
            key=lambda unit: self._worker_priority(unit, core_position),
        ):
            if worker.id in context.planned_ids:
                continue
            self._decide_worker(worker, context)
        self._decide_core(context)
        if self.config.expedition_mode:
            self._persist_expedition_pursuits()
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
        expedition_member = self._is_expedition_member(ranger, context.turn)
        if expedition_member and self._decide_expedition_ranger(ranger, context):
            return
        visible_enemies = self._visible_combat_targets(
            ranger,
            context.turn,
            fleet_wide=(context.emergency and ranger.id not in context.garrison_ids)
            or ranger.id in context.raid_ids,
            intercept_ids=context.intruder_intercept_ids,
            converge_ids=context.perimeter_intercept_ids,
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
        # A one-HP Ranger can still trade a shot safely when the engagement is
        # remote, but it must not spend its last Tick firing while the Core is
        # already in a pre-evacuation or engaged posture.  Combat resolves
        # after movement, so the incoming shot lands before the next decision;
        # at the Core's edge that used to turn a defensive Ranger's final shot
        # into a preventable death during Core migration.
        if (
            not expedition_member
            and ranger.hp <= 1
            and context.threat.level
            in {ThreatLevel.PRE_EVADE, ThreatLevel.ENGAGED, ThreatLevel.BREAKOUT}
            and self._recover_if_critical(
                ranger,
                maximum_hp=2,
                context=context,
            )
        ):
            return
        # A roaming Ranger marked for recall has already been intercepted by
        # the local threat policy.  Return it before the firing/standoff
        # branch can pull it back into the same remote duel and make it
        # oscillate between disengaging and heading home.
        if (
            not expedition_member
            and ranger.id in context.squad_return_ids
            and context.turn.core is not None
            and self._move_ranger_toward_core(
                ranger,
                context,
                reason="return intercepted expedition Ranger to Core",
            )
        ):
            return
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
                if not expedition_member and self._decline_ranger_duel(
                    ranger, target, context
                ):
                    return
                if (
                    not expedition_member
                    and ranger.hp <= 1
                    and self._ranger_would_take_return_fire(
                        ranger,
                        target,
                        obstacles,
                    )
                    and self._recover_if_critical(
                        ranger,
                        maximum_hp=2,
                        context=context,
                    )
                ):
                    return
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

        # Away from a local Core threat, a Ranger at one HP is still a live
        # firing platform.  Let it take a legal shot before withdrawing;
        # otherwise a whole damaged fireteam can collapse into the Core while
        # an enemy remains in range.
        if not expedition_member and self._recover_if_critical(
            ranger, maximum_hp=2, context=context
        ):
            return

        if self._pickup_beacon(ranger, context):
            return
        if not expedition_member:
            visible_target = self._preferred_target(
                context.focus_target, visible_enemies
            )
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
        if self._is_expedition_member(
            vanguard, context.turn
        ) and self._decide_expedition_vanguard(vanguard, context):
            return
        remote_assault = (
            context.combat_assault
            and context.turn.core is not None
            and manhattan(vanguard.position, context.turn.core.position)
            > self.config.core_assault_radius
        )
        if (not context.combat_assault or remote_assault) and self._recover_if_critical(
            vanguard,
            maximum_hp=4,
            critical_hp=3,
            context=context,
        ):
            return
        visible_enemies = self._visible_combat_targets(
            vanguard,
            context.turn,
            fleet_wide=(context.emergency and vanguard.id not in context.garrison_ids)
            or vanguard.id in context.raid_ids,
            intercept_ids=context.intruder_intercept_ids,
            converge_ids=context.perimeter_intercept_ids,
        )
        if (
            context.combat_assault
            and vanguard.id not in context.vanguard_attack_ids
            and not self._is_expedition_member(vanguard, context.turn)
        ):
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
        expedition_member = self._is_expedition_member(vanguard, context.turn)
        if not expedition_member:
            visible_target = self._preferred_target(
                context.focus_target, visible_enemies
            )
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
                critical_hp=3,
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
        if self._recover_if_critical(
            vanguard,
            maximum_hp=4,
            critical_hp=3,
            context=context,
        ):
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
        escort = context.intruder_intercept_ids or context.perimeter_intercept_ids
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
        local_core_assault = context.threat.recent_core_attack or any(
            manhattan(core.position, enemy.position) <= self.config.core_assault_radius
            for enemy in context.assault_enemies
        )
        if (
            context.combat_assault
            and local_core_assault
            and (
                manhattan(worker.position, core.position)
                <= self.config.core_assault_radius
            )
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
                if (
                    context.remaining_resource_space > 0
                    and context.core_service_action is None
                ):
                    deposited = min(worker.cargo, context.remaining_resource_space)
                    worker.deposit()
                    context.remaining_resources += deposited
                    context.remaining_resource_space -= deposited
                    context.core_service_action = "DEPOSIT"
                    context.report.add(
                        actor_id=str(worker.id),
                        actor_kind="WORKER",
                        action="DEPOSIT",
                        reason="return combat-economy resources to Core",
                        target=core.position,
                    )
                else:
                    if not self._move_to_core_queue(
                        worker,
                        context,
                        reason=(
                            "leave full Core service cell for a unique cargo queue slot"
                        ),
                    ):
                        self._record_wait(
                            worker,
                            context,
                            "Core storage is full and no outbound queue slot is free",
                        )
                return
            if (
                context.remaining_resource_space <= 0
                or context.core_service_action is not None
                or context.core_inbound_id is not None
            ):
                if not self._move_to_core_queue(
                    worker,
                    context,
                    reason="queue carried resources outside the Core service cell",
                ):
                    self._record_wait(
                        worker,
                        context,
                        "Core service cell is occupied and no queue slot is free",
                    )
                return
            if self._move(
                worker,
                core.position,
                context,
                reason="return carried resources to Core",
                allow_goal=True,
                intent=_MoveIntent.INBOUND,
            ):
                return
            # A blocked home approach can end just outside the danger zone.
            # Keep withdrawing from that nearby threat instead of treating
            # the pathfinder's lack of progress as a normal cargo queue.
            approach_threats = tuple(
                position
                for position in self._remembered_worker_danger_positions(context.turn)
                if manhattan(worker.position, position)
                <= self.config.worker_threat_radius + 1
            )
            if approach_threats and self._retreat_worker(
                worker, core.position, context, approach_threats
            ):
                return
            if not self._move_to_core_queue(
                worker,
                context,
                reason="queue carried resources after a full Core approach",
            ):
                self._record_wait(worker, context, "no safe path to Core")
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

        if assigned_resource is not None and not self._has_static_route(
            worker,
            assigned_resource,
            context,
            allow_goal=True,
        ):
            # The claim is walled off rather than merely crowded, so every
            # approach to it is an orbit: the route never closes, the Worker
            # never stands on the cell, no absence is ever recorded, and the
            # cell therefore stays the nearest candidate forever.  Rest it and
            # fall through to patrol, which already avoids remembered danger.
            self._release_unreachable_claim(worker, assigned_resource, context)
            assigned_resource = None

        if assigned_resource is not None and self._move(
            worker,
            assigned_resource,
            context,
            reason="claim nearest unassigned known resource",
            allow_goal=True,
        ):
            existing_goal = self.memory.goal_for(str(worker.id))
            if (
                existing_goal is None
                or existing_goal.purpose != RESOURCE_CLAIM_PURPOSE
                or existing_goal.position != assigned_resource
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
            elif existing_goal.last_progress_position is None or manhattan(
                worker.position, assigned_resource
            ) < manhattan(existing_goal.last_progress_position, assigned_resource):
                # Keep the best-known progress for a held claim.  Replacing
                # this with the current cell every Tick makes a route that
                # circles around a blocked cluster look healthy forever: one
                # occasional step closer resets the stall budget before the
                # Worker ever reaches the resource.
                self.memory.set_goal(
                    str(worker.id),
                    UnitGoal(
                        position=existing_goal.position,
                        assigned_tick=existing_goal.assigned_tick,
                        purpose=existing_goal.purpose,
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
            elif not self._has_static_route(
                worker,
                resource_goal.position,
                context,
                allow_goal=True,
            ):
                self._release_unreachable_claim(
                    worker,
                    resource_goal.position,
                    context,
                )
            else:
                age = context.turn.tick - resource_goal.assigned_tick
                progress_reference = resource_goal.last_progress_position
                progressing = progress_reference is not None and manhattan(
                    worker.position, resource_goal.position
                ) < manhattan(progress_reference, resource_goal.position)
                if age <= RESOURCE_CLAIM_TTL or progressing:
                    if worker.position == resource_goal.position:
                        # Arrival is already a close observation, so an empty
                        # site is resting in memory.  Do not pin the Worker
                        # there for the whole recheck floor; the cooldown is a
                        # preference and the Worker can patrol another known
                        # site in this same Tick.
                        self.memory.clear_goal(str(worker.id))
                        self._claim_stalls.pop(worker.id, None)
                    else:
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
            if (
                self._is_resource_scout(worker, context.turn)
                and context.threat.level is ThreatLevel.NORMAL
            ):
                goal = self._exploration_goal(worker, context.turn.tick)
                reason = "scout beyond the local patrol ring for resources"
            else:
                goal = self._resource_patrol_goal(worker, context.turn, context)
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

    @staticmethod
    def _predicted_friendly_occupancy(
        position: Position,
        context: _TurnContext,
    ) -> int:
        """Return occupancy after the moves already accepted this Tick."""

        return (
            context.friendly_occupancy.get(position, 0)
            + context.arrival_counts.get(position, 0)
            - context.departure_counts.get(position, 0)
        )

    @staticmethod
    def _has_unit_action(unit: Unit, context: _TurnContext) -> bool:
        """Inspect queued actions without building a Pydantic command plan."""

        builder = getattr(context.turn, "_builder", None)
        actions = getattr(builder, "unit_actions", {})
        return unit.id in context.planned_ids or unit.id in actions

    def _core_queue_positions(
        self,
        unit: Unit,
        context: _TurnContext,
    ) -> tuple[Position, ...]:
        """Return empty, reachable cells near the Core in stable order."""

        core = context.turn.core
        if core is None:
            return ()
        blockers = self._static_blockers(unit, context)
        candidates: list[Position] = []
        for radius in range(1, CORE_QUEUE_RADIUS + 1):
            for dx in range(-radius, radius + 1):
                dy = radius - abs(dx)
                offsets = ((dx, dy),) if dy == 0 else ((dx, dy), (dx, -dy))
                for offset_x, offset_y in offsets:
                    position = (
                        core.position[0] + offset_x,
                        core.position[1] + offset_y,
                    )
                    if (
                        position in blockers
                        or position in context.enemy_positions
                        or position == core.position
                        or position in context.standing_reserved
                        or self._predicted_friendly_occupancy(position, context) != 0
                        or not self._has_static_route(
                            unit,
                            position,
                            context,
                            allow_goal=True,
                        )
                    ):
                        continue
                    candidates.append(position)
        return tuple(
            sorted(
                set(candidates),
                key=lambda position: (
                    manhattan(position, core.position),
                    manhattan(unit.position, position),
                    position,
                ),
            )
        )

    def _move_to_core_queue(
        self,
        unit: Unit,
        context: _TurnContext,
        *,
        reason: str,
    ) -> bool:
        """Move a unit toward a unique temporary Core queue slot."""

        if self._has_unit_action(unit, context):
            return False
        core = context.turn.core
        if core is not None:
            distance = manhattan(unit.position, core.position)
            if (
                0 < distance <= CORE_QUEUE_RADIUS
                and self._predicted_friendly_occupancy(unit.position, context) == 1
                and unit.position not in self.memory.obstacles
                and unit.position not in context.enemy_positions
            ):
                self._record_wait(unit, context, "hold a unique Core queue slot")
                return True
        for goal in self._core_queue_positions(unit, context):
            context.standing_reserved.add(goal)
            if self._move(
                unit,
                goal,
                context,
                reason=reason,
                allow_goal=True,
                intent=_MoveIntent.STAND,
            ):
                return True
            context.standing_reserved.discard(goal)
        return False

    def _move_core_unit_via_neighbor(
        self,
        unit: Unit,
        context: _TurnContext,
        *,
        reason: str,
    ) -> bool:
        """Release one full neighbour so a Core unit can leave its cell."""

        core = context.turn.core
        if core is None:
            return False
        blockers = self._static_blockers(unit, context)
        for neighbor in adjacent_positions(core.position):
            if neighbor in blockers or neighbor in context.enemy_positions:
                continue
            occupants = [
                candidate
                for candidate in context.turn.units
                if candidate.position == neighbor and candidate.id != unit.id
            ]
            predicted = self._predicted_friendly_occupancy(neighbor, context)
            if predicted >= 2 and occupants:
                movable = [
                    candidate
                    for candidate in occupants
                    if candidate.id not in context.planned_ids
                    and not self._has_unit_action(candidate, context)
                ]
                for occupant in sorted(
                    movable,
                    key=lambda candidate: candidate.id.bytes,
                ):
                    for escape in adjacent_positions(neighbor):
                        if (
                            escape == core.position
                            or escape in blockers
                            or escape in context.enemy_positions
                            or escape in context.standing_reserved
                            or self._predicted_friendly_occupancy(escape, context) != 0
                        ):
                            continue
                        escape_direction = direction_between(occupant.position, escape)
                        if escape_direction is None:
                            continue
                        context.standing_reserved.add(escape)
                        if self._queue_move(
                            occupant,
                            escape_direction,
                            context,
                            reason="clear a full Core approach cell",
                            target=escape,
                            intent=_MoveIntent.STAND,
                        ):
                            predicted = self._predicted_friendly_occupancy(
                                neighbor, context
                            )
                            break
                        context.standing_reserved.discard(escape)
                    if predicted < 2:
                        break
            if predicted >= 2:
                continue
            direction = direction_between(unit.position, neighbor)
            if direction is None:
                continue
            if self._queue_move(
                unit,
                direction,
                context,
                reason=reason,
                target=neighbor,
                intent=_MoveIntent.OUTBOUND,
            ):
                return True
        return False

    def _prepare_core_departure(self, context: _TurnContext) -> None:
        """Free the Core service cell before accepting new inbound traffic."""

        core = context.turn.core
        if core is None or core.view.state is CoreState.MOVING:
            return
        core_units = [
            unit for unit in context.turn.units if unit.position == core.position
        ]
        if not core_units:
            return
        service_candidate: Unit | None = None
        if context.remaining_resource_space > 0:
            service_candidate = next(
                (
                    unit
                    for unit in core_units
                    if isinstance(unit, Worker) and unit.cargo > 0
                ),
                None,
            )
        if service_candidate is None:
            service_candidate = next(
                (
                    unit
                    for unit in core_units
                    if unit.hp <= 1
                    and self._heal_available(
                        unit,
                        maximum_hp=2 if isinstance(unit, (Worker, Ranger)) else 4,
                        context=context,
                    )
                ),
                None,
            )
        for unit in sorted(core_units, key=lambda candidate: candidate.id.bytes):
            if unit is service_candidate:
                continue
            critical_without_service = (
                unit.hp <= (1 if isinstance(unit, (Worker, Ranger)) else 2)
                and context.remaining_resources <= 0
            )
            full_loaded_worker = isinstance(unit, Worker) and (
                unit.cargo > 0 and context.remaining_resource_space <= 0
            )
            extra_service_unit = service_candidate is not None
            if not (
                critical_without_service or full_loaded_worker or extra_service_unit
            ):
                continue
            if self._has_unit_action(unit, context):
                continue
            if self._move_to_core_queue(
                unit,
                context,
                reason="leave Core service cell for unique traffic",
            ):
                continue
            self._move_core_unit_via_neighbor(
                unit,
                context,
                reason="leave Core service cell through a full approach",
            )

    def _emergency_worker_goal(
        self,
        worker: Worker,
        context: _TurnContext,
    ) -> Position:
        """Move Workers away from the Core fight."""

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
            if position not in obstacles and position not in enemy_positions
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
            unit_type = (
                None
                if self._normal_growth_on_cooldown(context)
                else self._choose_spawn(context.turn, context.remaining_resources)
            )
            if unit_type is not None:
                core.spawn(unit_type)
                context.core_service_action = "SPAWN"
                if self._growth_slowdown_active(context.turn) and not context.emergency:
                    self.memory.last_normal_growth_tick = context.turn.tick
                    self.memory.last_normal_growth_resources = (
                        context.remaining_resources
                        - unit_cost(unit_type, context.turn.state.population)
                    )
                context.report.add(
                    actor_id=str(core.id),
                    actor_kind="CORE",
                    action="SPAWN",
                    reason=self._spawn_reason(unit_type),
                    target=core.position,
                )
                return

        if self._symmetric_posture() and not nearby_enemy and context.turn.workers:
            # Workers already know how to stage carried resources at the
            # Core's current destination while it is moving.  Requiring the
            # whole economy to be empty before starting the revised live
            # relocation deadlocks as soon as a remote Worker is returning.
            direction = self._core_home_direction(context)
            if direction is not None:
                core.start_move(direction)
                context.report.add(
                    actor_id=str(core.id),
                    actor_kind="CORE",
                    action="START_MOVE",
                    reason="relocate Core to the selected symmetric-defense home",
                    target=add(core.position, direction),
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

    def _heal_available(
        self,
        unit: Unit,
        *,
        maximum_hp: int,
        critical_hp: int | None = None,
        context: _TurnContext,
    ) -> bool:
        """Report whether the Core would fund a heal for ``unit``.

        The heal gate has to use the same threshold that sent the Unit home.
        When the withdrawal threshold was raised above ``maximum_hp // 2`` and
        this gate was left behind, a Vanguard at three HP counted as critical,
        walked onto the Core cell, and then never qualified for the heal it
        came for. Friendly overlap does not remove the need to use the same
        threshold for return and healing.
        """

        core = context.turn.core
        heal_threshold = maximum_hp // 2 if critical_hp is None else critical_hp
        missing_hp = maximum_hp - unit.hp
        return (
            core is not None
            and core.view.state is CoreState.NORMAL
            and unit.hp <= heal_threshold
            and missing_hp > 0
            and context.remaining_resources >= missing_hp
        )

    def _heal_if_critical(
        self,
        unit: Unit,
        *,
        maximum_hp: int,
        critical_hp: int | None = None,
        context: _TurnContext,
    ) -> bool:
        core = context.turn.core
        missing_hp = maximum_hp - unit.hp
        if core is None or unit.position != core.position:
            return False
        if not self._heal_available(
            unit,
            maximum_hp=maximum_hp,
            critical_hp=critical_hp,
            context=context,
        ):
            return False
        if context.core_service_action is not None:
            return False
        unit.heal()
        context.remaining_resources -= missing_hp
        context.core_service_action = "HEAL"
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
        critical_hp: int | None = None,
        context: _TurnContext,
    ) -> bool:
        if (
            self.config.expedition_mode
            and isinstance(unit, (Ranger, Vanguard))
            and unit.id in self._expedition_member_ids(context.turn)
        ):
            # Expeditions are one-way combat units. Low health never enrolls
            # them in defensive recall or Core healing.
            return False
        retreat_threshold = maximum_hp // 2 if critical_hp is None else critical_hp
        if unit.hp > retreat_threshold:
            return False
        if self._heal_if_critical(
            unit,
            maximum_hp=maximum_hp,
            critical_hp=critical_hp,
            context=context,
        ):
            return True

        core = context.turn.core
        if core is None:
            return False
        reason = "return critical unit to Core for healing"
        if unit.position == core.position:
            if not self._move_to_core_queue(
                unit,
                context,
                reason="leave Core service cell while another unit is healing",
            ):
                self._record_wait(unit, context, "wait at Core for healing resources")
            return True
        moved = (
            self._move_ranger_toward_core(unit, context, reason=reason)
            if isinstance(unit, Ranger)
            else self._move(
                unit,
                core.position,
                context,
                reason=reason,
                allow_goal=True,
                intent=_MoveIntent.INBOUND,
            )
        )
        if not moved and not self._move_to_core_queue(
            unit,
            context,
            reason="queue critical unit outside the Core service cell",
        ):
            self._record_wait(unit, context, f"no safe path for: {reason}")
        return True

    def _move_ranger_toward_core(
        self,
        ranger: Ranger,
        context: _TurnContext,
        *,
        reason: str,
    ) -> bool:
        """Return a Ranger home without stepping closer into enemy fire."""

        core = context.turn.core
        if core is None or ranger.position == core.position:
            return False
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        threats = tuple(
            enemy
            for enemy in context.turn.visible_enemies
            if isinstance(enemy, UnitView)
            and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(ranger.position, enemy.position)
            <= RANGER_RETURN_THREAT_RADIUS
        )
        if not threats:
            return self._move(
                ranger,
                core.position,
                context,
                reason=reason,
                allow_goal=True,
                intent=_MoveIntent.INBOUND,
            )

        # The static blocker helper may return the durable obstacle set
        # directly. Copy only on this threat-specific branch, which mutates
        # the local view to exempt the Ranger and Core.
        blocked = set(self._static_blockers(ranger, context))
        blocked.update(context.enemy_positions)
        blocked.discard(ranger.position)
        blocked.discard(core.position)
        candidates = [
            position
            for position in adjacent_positions(ranger.position)
            if position not in blocked and position not in context.enemy_positions
        ]
        if not candidates:
            return self._move(
                ranger,
                core.position,
                context,
                reason=reason,
                allow_goal=True,
                intent=_MoveIntent.INBOUND,
            )

        def attack_distance(enemy: UnitView, position: Position) -> int:
            if enemy.unit_type is UnitType.VANGUARD:
                return manhattan(enemy.position, position)
            return self._ranger_range(enemy.position, position)

        current_distance = min(
            attack_distance(enemy, ranger.position) for enemy in threats
        )
        non_closing = [
            position
            for position in candidates
            if min(attack_distance(enemy, position) for enemy in threats)
            >= current_distance
        ]
        protected = [
            position
            for position in non_closing
            if not any(
                self._enemy_can_attack_position(enemy, position, obstacles)
                for enemy in threats
            )
        ]
        preferred = protected or non_closing
        if not preferred:
            return self._move(
                ranger,
                core.position,
                context,
                reason=reason,
                allow_goal=True,
                intent=_MoveIntent.INBOUND,
            )
        goal = min(
            preferred,
            key=lambda position: (
                manhattan(position, core.position),
                -min(attack_distance(enemy, position) for enemy in threats),
                position,
            ),
        )
        direction = direction_between(ranger.position, goal)
        if direction is None:
            return self._move(
                ranger,
                core.position,
                context,
                reason=reason,
                allow_goal=True,
                intent=_MoveIntent.INBOUND,
            )
        return self._queue_move(
            ranger,
            direction,
            context,
            reason=reason,
            target=core.position,
            intent=(
                _MoveIntent.INBOUND
                if add(ranger.position, direction) == core.position
                else _MoveIntent.TRANSIT
            ),
        )

    def _move_or_wait(
        self,
        unit: Unit,
        goal: Position,
        context: _TurnContext,
        *,
        reason: str,
        allow_goal: bool = False,
        wait_at_goal: bool = False,
        intent: _MoveIntent = _MoveIntent.STAND,
    ) -> None:
        if unit.position == goal:
            self._record_wait(unit, context, reason)
            if wait_at_goal:
                unit.wait()
        elif not self._move(
            unit,
            goal,
            context,
            reason=reason,
            allow_goal=allow_goal,
            intent=intent,
        ):
            self._record_wait(unit, context, f"no safe path for: {reason}")

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
        blocked.update(context.enemy_positions)
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
                (
                    enemy.kind == "CORE"
                    and context.turn.tick - enemy.tick <= self.config.enemy_memory_ttl
                )
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

    def _static_blockers(self, unit: Unit, context: _TurnContext) -> set[Position]:
        """Return the blockers for ``unit`` that outlive this Tick.

        Friendly occupancy and queued destinations are left out because units
        may overlap and swap.  Terrain, contested cells, and Worker danger
        zones still constrain the route.
        """

        blocked = self.memory.obstacles
        if self.memory.contested_positions:
            blocked = blocked | set(self.memory.contested_positions)
        if not context.turn.obstacle_cells <= blocked:
            blocked = blocked | set(context.turn.obstacle_cells)
        if isinstance(unit, Worker):
            threat_exclusion = self._worker_threat_exclusion_cells(context.turn)
            if threat_exclusion:
                blocked = blocked | threat_exclusion
        return blocked

    def _has_static_route(
        self,
        unit: Unit,
        goal: Position,
        context: _TurnContext,
        *,
        allow_goal: bool = False,
    ) -> bool:
        """Return whether ``goal`` is reachable ignoring our own traffic.

        A partial path toward a goal does not prove the goal is reachable.
        Resource claims require the complete route so a Worker does not keep
        approaching a permanently sealed site without ever harvesting it.
        """

        if goal in self.memory.obstacles or goal in context.turn.obstacle_cells:
            return False
        if unit.position == goal:
            return True
        return (
            next_step(
                unit.position,
                goal,
                blocked=self._static_blockers(unit, context),
                allow_goal=allow_goal,
                require_path=True,
            )
            is not None
        )

    def _move(
        self,
        unit: Unit,
        goal: Position,
        context: _TurnContext,
        *,
        reason: str,
        allow_goal: bool = False,
        intent: _MoveIntent = _MoveIntent.TRANSIT,
        path_expansions: int = 4096,
        require_path: bool = False,
    ) -> bool:
        if unit.position == goal:
            return False
        blocked = self._static_blockers(unit, context)
        # Friendly units may overlap and swap; they never block travel.
        if goal in self.memory.obstacles or goal in context.turn.obstacle_cells:
            return False
        extra_blocked: set[Position] = set()
        core = context.turn.core
        if (
            intent is _MoveIntent.STAND
            and core is not None
            and goal != core.position
            and unit.position != core.position
        ):
            # A queueing unit must not use the service cell as a shortcut.
            extra_blocked.add(core.position)
        extra_blocked.update(context.enemy_positions)
        if extra_blocked:
            blocked = blocked | extra_blocked
        max_expansions = (
            EXPEDITION_STAGING_PATH_EXPANSIONS
            if self.config.expedition_mode and unit.id in self._staged_ids
            else path_expansions
        )
        direction = next_step(
            unit.position,
            goal,
            blocked=blocked,
            recent=self.memory.recent_positions(str(unit.id)),
            direction_offset=self._direction_offset(unit.id),
            max_expansions=max_expansions,
            allow_goal=allow_goal,
            require_path=require_path,
        )
        if direction is None:
            return False
        destination = add(unit.position, direction)
        if destination in blocked and not (allow_goal and destination == goal):
            return False
        return self._queue_move(
            unit,
            direction,
            context,
            reason=reason,
            target=goal,
            intent=intent if destination == goal else _MoveIntent.TRANSIT,
        )

    def _queue_move(
        self,
        unit: Unit,
        direction: Direction,
        context: _TurnContext,
        *,
        reason: str,
        target: Position,
        intent: _MoveIntent = _MoveIntent.TRANSIT,
    ) -> bool:
        destination = add(unit.position, direction)
        if self._has_unit_action(unit, context):
            return False
        # Also protect direct retreat moves and permissive goal paths.
        if (
            destination in self.memory.obstacles
            or destination in context.turn.obstacle_cells
        ):
            return False
        predicted = self._predicted_friendly_occupancy(destination, context)
        if intent is _MoveIntent.STAND:
            if predicted != 0:
                return False
        elif predicted >= 2:
            return False
        unit.move(direction)
        self.memory.pending_move_targets[str(unit.id)] = destination
        context.reserved.add(destination)
        context.arrival_counts[destination] = (
            context.arrival_counts.get(destination, 0) + 1
        )
        context.departure_counts[unit.position] = (
            context.departure_counts.get(unit.position, 0) + 1
        )
        context.planned_ids.add(unit.id)
        if intent is _MoveIntent.INBOUND:
            core = context.turn.core
            if core is not None and destination == core.position:
                context.core_inbound_id = unit.id
        if intent is _MoveIntent.STAND and destination == target:
            context.standing_reserved.add(destination)
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
        converge_ids: frozenset[UUID] = frozenset(),
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

        ``converge_ids`` is the same narrow grant for a lone fighter already
        inside the defensive leash.  Surrounding an equal-speed target needs
        the squad aimed at where it stands now rather than at the cell it left,
        and one Vanguard's four-cell vision is not enough to organise that.
        """

        if fleet_wide:
            return turn.visible_enemies
        escorting = unit.id in intercept_ids
        converging = unit.id in converge_ids
        core = turn.core
        return tuple(
            enemy
            for enemy in turn.visible_enemies
            if self._combat_target_is_local(unit, enemy.position)
            or (
                escorting
                and isinstance(enemy, UnitView)
                and enemy.unit_type is UnitType.WORKER
                and core is not None
                and manhattan(core.position, enemy.position)
                <= self.config.core_intruder_radius
            )
            or (
                converging
                and isinstance(enemy, UnitView)
                and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                and core is not None
                and manhattan(core.position, enemy.position)
                <= self.config.combat_pursuit_radius
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

    def _perimeter_intercept_anchor(
        self,
        turn: Turn,
        emergency: bool,
    ) -> Position | None:
        """Return the cell a lone fighter inside the defensive zone occupies.

        A single enemy fighter walking through the defensive zone is the one
        case the rest of the combat logic used to answer with nothing at all.
        It raises the threat to ``ALERT``, which demotes the whole roaming
        third to ring guards; it sits beyond ``worker_threat_radius * 2``, so
        no guard is handed it as a memory target; and the one unit that does
        make contact is recalled home as an intercepted expedition.  The three
        rules cancel each other and the intruder strolls past a stationary
        army.

        Evidence is deliberately Core-local, exactly as ``_raid_recall_needed``
        requires of the mirrored decision: only a fighter already inside
        ``combat_pursuit_radius`` counts, so this can never reach past the
        defensive leash.  A real attack is left alone: an emergency, a damaged
        Core or a roster too small to spare a squad all cancel the convergence,
        and so does a third fighter, because past two the answer is a defence
        rather than a chase.  The last known cell is held for a few Ticks
        because vision over a mover flickers and the squad must not restart its
        convergence every other Tick.
        """

        core = turn.core
        if core is None or emergency or not self._offensive_patrol_enabled(turn):
            self._perimeter_intercept = None
            return None
        radius = self.config.combat_pursuit_radius
        inside = tuple(
            enemy
            for enemy in turn.visible_enemies
            if isinstance(enemy, UnitView)
            and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(core.position, enemy.position) <= radius
        )
        if len(inside) > PERIMETER_INTERCEPT_MAX_ENEMIES:
            self._perimeter_intercept = None
            return None
        if inside:
            target = min(
                inside,
                key=lambda enemy: (
                    manhattan(core.position, enemy.position),
                    str(enemy.id),
                ),
            )
            self._perimeter_intercept = (
                str(target.id),
                target.position,
                turn.tick,
            )
            return target.position
        remembered = self._perimeter_intercept
        if remembered is None:
            return None
        _, position, tick = remembered
        arrived = any(
            manhattan(unit.position, position) <= 1
            for unit in (*turn.vanguards, *turn.rangers)
        )
        if (
            arrived
            or turn.tick - tick > PERIMETER_INTERCEPT_MEMORY_TICKS
            or manhattan(core.position, position) > radius
        ):
            # Standing on the cell with nothing in sight ends the hunt: the
            # target broke contact, and a squad that keeps walking at an empty
            # cell is the open-ended pursuit the leash exists to prevent.
            self._perimeter_intercept = None
            return None
        return position

    def _perimeter_intercept_ids(
        self,
        turn: Turn,
        anchor: Position | None,
        garrison_ids: frozenset[UUID],
        raid_ids: frozenset[UUID],
    ) -> frozenset[UUID]:
        """Reserve the nearest bounded squad to surround a lone fighter.

        An equal-speed target cannot be run down by a single chaser, so the
        squad has to be large enough to cover the escape cells; it is still
        capped, and the garrison and raid detachment are excluded, so the ring
        keeps its cover on the other seven bearings while this happens.
        """

        if anchor is None:
            return frozenset()
        candidates = sorted(
            (
                unit
                for unit in (*turn.vanguards, *turn.rangers)
                if unit.id not in garrison_ids
                and unit.id not in raid_ids
                and manhattan(unit.position, anchor)
                <= self.config.combat_pursuit_radius
            ),
            key=lambda unit: (
                manhattan(unit.position, anchor),
                unit.hp <= 2,
                unit.id.bytes,
            ),
        )
        return frozenset(unit.id for unit in candidates[:PERIMETER_INTERCEPTORS])

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
            # ``should_evacuate_core`` also fires on a lone fighter merely
            # entering the preemptive horizon.  When the local combat guard
            # already outmatches the approaching enemy, moving the Core
            # trades four Ticks of production and healing for a fight the
            # guards should win in place.  Only stand down while the enemy
            # has not yet reached attack range or landed a hit.
            outmatched = (
                not threat.recent_core_attack
                and not threat.threatening_core_enemy_ids
                and self._local_combat_guard_count(turn) >= 2 * len(enemies)
            )
            return not outmatched
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

    def _local_combat_guard_count(self, turn: Turn) -> int:
        """Count combat Units close enough to defend a standing Core."""

        core = turn.core
        if core is None:
            return 0
        return sum(
            1
            for unit in (*turn.vanguards, *turn.rangers)
            if manhattan(core.position, unit.position)
            <= self.config.combat_alert_radius
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

    def _decline_ranger_duel(
        self,
        ranger: Ranger,
        target: CoreView | UnitView,
        context: _TurnContext,
    ) -> bool:
        """Avoid a one-for-one trade with a lone enemy Ranger.

        Two full-health Rangers at range 1-3 exchange equal damage, so a solo
        Ranger firing first dies with its target.  At max range a single step
        can break contact and buy time for a wingman to form a killing volley.
        """

        if not isinstance(target, UnitView) or target.unit_type is not UnitType.RANGER:
            return False
        if ranger.hp < 2 or target.hp < 2:
            return False
        if str(target.id) in context.threat.threatening_core_enemy_ids:
            return False
        if self._ranger_volley_count(target, context) >= 2:
            return False
        if self._ranger_range(ranger.position, target.position) < RANGER_STANDOFF_RANGE:
            return False
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        candidates = [
            position
            for position in adjacent_positions(ranger.position)
            if position not in obstacles and position not in context.enemy_positions
        ]
        if not candidates:
            return False
        goal = max(
            candidates,
            key=lambda position: (
                self._ranger_range(position, target.position),
                -manhattan(ranger.position, position),
                position,
            ),
        )
        if self._ranger_range(goal, target.position) <= self._ranger_range(
            ranger.position, target.position
        ):
            return False
        return self._move_within_leash(
            ranger,
            goal,
            context,
            offensive=False,
            reason=f"hold for a killing volley on {self._enemy_label(target)}",
        )

    def _ranger_volley_count(
        self,
        target: CoreView | UnitView,
        context: _TurnContext,
    ) -> int:
        """Count friendly Rangers already in range to volley ``target``."""

        return sum(
            1
            for ranger in context.turn.rangers
            if self._ranger_range(ranger.position, target.position)
            <= RANGER_STANDOFF_RANGE
        )

    @staticmethod
    def _ranger_would_take_return_fire(
        ranger: Ranger,
        target: CoreView | UnitView,
        obstacles: set[Position] | frozenset[Position],
    ) -> bool:
        """Return whether a last-HP Ranger is inside the target's attack."""

        if not isinstance(target, UnitView):
            return False
        return AggressiveStrategy._enemy_can_attack_position(
            target,
            ranger.position,
            obstacles,
        )

    @staticmethod
    def _enemy_can_attack_position(
        enemy: UnitView,
        position: Position,
        obstacles: set[Position] | frozenset[Position],
    ) -> bool:
        if enemy.unit_type is UnitType.VANGUARD:
            return manhattan(enemy.position, position) == 1
        return enemy.unit_type is UnitType.RANGER and line_of_fire(
            enemy.position,
            position,
            obstacles,
        )

    def _ranger_shot_cell(
        self,
        ranger: Ranger,
        enemy: CoreView | UnitView,
        turn: Turn,
        obstacles: set[Position] | frozenset[Position],
    ) -> Position | None:
        # Movement resolves before the shot. Prefer a repeated straight or
        # two-cell patrol track over extrapolating just the last direction.
        # The looser ``drift`` estimate also works across flickering vision.
        # At range 1 the target is close enough
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
        converging = unit.id in context.perimeter_intercept_ids
        radius = self.config.core_intruder_radius
        pursuit_radius = self.config.combat_pursuit_radius
        targets: list[EnemySighting] = []
        for enemy in recent_enemies:
            if self._is_combat_memory_target(enemy):
                if self._combat_target_is_local(unit, enemy.position) or (
                    converging
                    and core is not None
                    and manhattan(core.position, enemy.position) <= pursuit_radius
                ):
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
        """Keep goal-oriented defenders near the Core instead of roaming away.

        A hostile that can shoot is kept while it stays inside the defensive
        leash rather than inside the tighter economy ring.  The old single
        radius was twelve cells, so a fighter loitering at eighteen or twenty
        was erased from every defender's target list before the per-unit rules
        ever saw it: the ring had nothing to react to and the roaming third,
        already demoted by the raised threat level, had nothing either.
        Twenty is the distance a guard can still walk to and come home from,
        so admitting it here cannot widen the engagement zone.  Everything
        that cannot shoot back keeps the economy radius.
        """

        core = turn.core
        if not self._preserves_resources() or core is None:
            return enemies
        economy_radius = self.config.worker_threat_radius * 2
        pursuit_radius = self.config.combat_pursuit_radius
        return tuple(
            enemy
            for enemy in enemies
            if manhattan(core.position, enemy.position)
            <= (
                pursuit_radius
                if self._is_combat_memory_target(enemy)
                else economy_radius
            )
        )

    @staticmethod
    def _patrol_team_from_role(role: str | None) -> int | None:
        """Return a patrol team number encoded in a durable role."""

        if role is None or not role.startswith(PATROL_ROLE_PREFIX):
            return None
        try:
            team = int(role.removeprefix(PATROL_ROLE_PREFIX))
        except ValueError:
            return None
        return team if 1 <= team <= SYMMETRIC_PATROL_TEAM_COUNT else None

    @staticmethod
    def _is_expedition_role(role: str | None) -> bool:
        """Whether a role is one of the persisted expedition labels."""

        if role is None or not role.startswith(EXPEDITION_ROLE_PREFIX):
            return False
        try:
            return int(role.removeprefix(EXPEDITION_ROLE_PREFIX)) > 0
        except ValueError:
            return False

    def _symmetric_posture(self) -> bool:
        """Whether the revised 16-Worker live formation is active."""

        return self.config.expedition_mode and self.config.target_workers >= 16

    def _patrol_team_count(self) -> int:
        return (
            SYMMETRIC_PATROL_TEAM_COUNT
            if self._symmetric_posture()
            else PATROL_TEAM_COUNT
        )

    def _formation_quota(self) -> tuple[int, int, int, int]:
        if self._symmetric_posture():
            return (
                SYMMETRIC_FORMATION_VANGUARDS,
                SYMMETRIC_FORMATION_RANGERS,
                SYMMETRIC_DEFENSE_VANGUARDS,
                SYMMETRIC_DEFENSE_RANGERS,
            )
        return (
            EXPEDITION_FORMATION_VANGUARDS,
            EXPEDITION_FORMATION_RANGERS,
            DEFENSE_VANGUARDS,
            DEFENSE_RANGERS,
        )

    def _reconcile_unit_roles(self, turn: Turn) -> None:
        """Assign stable combat identities without reshuffling live units.

        Roles are allocated only to previously unassigned UUIDs.  A new Unit
        can fill a genuinely vacant slot, but it cannot displace a live
        defender or patrol member merely because its UUID sorts earlier.
        """

        live_units = {str(unit.id): unit for unit in (*turn.vanguards, *turn.rangers)}
        live_ids = set(live_units)
        legacy_migration = not self.memory.unit_roles_initialized
        roles = {
            unit_id: role
            for unit_id, role in self.memory.unit_roles.items()
            if unit_id in live_ids
            and (
                role in {DEFENSE_ROLE, STAGED_ROLE}
                or self._patrol_team_from_role(role) is not None
                or self._is_expedition_role(role)
            )
        }

        # The squad list is authoritative for expedition members.  Reapply
        # those roles before filling any defensive or patrol vacancy.
        expedition_ids: set[str] = set()
        for index, (members, _bearing) in enumerate(self._expedition_squads, 1):
            serial = self._expedition_serial_by_members.get(members, index)
            for member_id in members:
                unit_id = str(member_id)
                if unit_id in live_ids:
                    roles[unit_id] = f"{EXPEDITION_ROLE_PREFIX}{serial}"
                    expedition_ids.add(unit_id)

        migration_excluded_ids: set[str] = set()
        if legacy_migration and turn.core is not None:
            # The previous in-memory tactic called every outer UUID a patrol
            # member.  A restart could therefore leave an old expedition
            # remnant hundreds of cells away while it still carried a patrol
            # label.  Keep such remnants staged during this one migration;
            # the normal vacancy pass below supplies a nearby replacement.
            for unit_id, role in tuple(roles.items()):
                if (
                    self._patrol_team_from_role(role) is not None
                    and manhattan(turn.core.position, live_units[unit_id].position)
                    > self.config.offensive_patrol_radius
                ):
                    roles[unit_id] = STAGED_ROLE
                    migration_excluded_ids.add(unit_id)

        if self._symmetric_posture():
            # The revised formation deliberately shrinks the old 13V+26R
            # garrison.  Stable roles normally never displace a survivor, but
            # that rule cannot preserve an explicitly superseded formation:
            # retain a deterministic 8V+16R core and release every excess
            # defender into staging, where normal expedition assembly can use
            # it without losing the UUID's identity.
            _, _, defense_vanguards, defense_rangers = self._formation_quota()
            for unit_type, quota in (
                (UnitType.VANGUARD, defense_vanguards),
                (UnitType.RANGER, defense_rangers),
            ):
                assigned = sorted(
                    (
                        unit_id
                        for unit_id, role in roles.items()
                        if role == DEFENSE_ROLE
                        and live_units[unit_id].unit_type is unit_type
                    ),
                    key=lambda unit_id: UUID(unit_id).bytes,
                )
                for unit_id in assigned[quota:]:
                    roles[unit_id] = STAGED_ROLE

        def candidate_key(unit_id: str) -> tuple[object, ...]:
            return (roles.get(unit_id) != STAGED_ROLE, UUID(unit_id).bytes)

        def assign_available(unit_type: UnitType, role: str, quota: int) -> None:
            assigned = sum(
                1
                for unit_id, current_role in roles.items()
                if current_role == role and live_units[unit_id].unit_type is unit_type
            )
            if assigned >= quota:
                return
            candidates = sorted(
                (
                    unit_id
                    for unit_id, unit in live_units.items()
                    if unit.unit_type is unit_type
                    and unit_id not in expedition_ids
                    and unit_id not in migration_excluded_ids
                    and (unit_id not in roles or roles[unit_id] == STAGED_ROLE)
                ),
                # A staged Unit is eligible to replace a dead formation
                # member.  Prefer that already-built reserve before assigning
                # a never-seen UUID, while leaving every surviving fixed role
                # untouched.
                key=candidate_key,
            )
            for unit_id in candidates[: quota - assigned]:
                roles[unit_id] = role

        # Keep the long-lived defensive ring identities first, then fill the
        # fixed patrol teams.  The quotas are intentionally separate so
        # a new Ranger never moves an existing patrol member into defence.
        _, _, defense_vanguards, defense_rangers = self._formation_quota()
        assign_available(UnitType.VANGUARD, DEFENSE_ROLE, defense_vanguards)
        assign_available(UnitType.RANGER, DEFENSE_ROLE, defense_rangers)
        for team in range(1, self._patrol_team_count() + 1):
            role = f"{PATROL_ROLE_PREFIX}{team}"
            assign_available(UnitType.VANGUARD, role, PATROL_TEAM_VANGUARDS)
            assign_available(UnitType.RANGER, role, PATROL_TEAM_RANGERS)

        for unit_id in sorted(
            live_ids - set(roles), key=lambda value: UUID(value).bytes
        ):
            roles[unit_id] = STAGED_ROLE
        self.memory.unit_roles = roles
        self.memory.defense_posts = {
            unit_id: position
            for unit_id, position in self.memory.defense_posts.items()
            if roles.get(unit_id) == DEFENSE_ROLE
        }
        self.memory.unit_roles_initialized = True
        self._staged_ids = frozenset(
            UUID(unit_id) for unit_id, role in roles.items() if role == STAGED_ROLE
        )

    def _ensure_unit_roles(self, turn: Turn) -> None:
        """Refresh role assignments when a Turn introduces or removes UUIDs."""

        self._reconcile_unit_roles(turn)

    def _formation_vanguard_ids(self, turn: Turn) -> frozenset[UUID]:
        """Return live Vanguards assigned to defence or a patrol team."""

        self._ensure_unit_roles(turn)
        return frozenset(
            UUID(unit_id)
            for unit_id, role in self.memory.unit_roles.items()
            if role == DEFENSE_ROLE or self._patrol_team_from_role(role) is not None
            if unit_id in {str(unit.id) for unit in turn.vanguards}
        )

    def _formation_ranger_ids(self, turn: Turn) -> frozenset[UUID]:
        """Return live Rangers assigned to defence or a patrol team."""

        self._ensure_unit_roles(turn)
        return frozenset(
            UUID(unit_id)
            for unit_id, role in self.memory.unit_roles.items()
            if role == DEFENSE_ROLE or self._patrol_team_from_role(role) is not None
            if unit_id in {str(unit.id) for unit in turn.rangers}
        )

    def _formation_combat_ids(self, turn: Turn) -> frozenset[UUID]:
        """Return all live combat units assigned to the fixed formation."""

        return self._formation_vanguard_ids(turn) | self._formation_ranger_ids(turn)

    def _expedition_member_ids(self, turn: Turn) -> frozenset[UUID]:
        """Return every live combat unit already bound to an expedition."""
        live = {unit.id for unit in (*turn.vanguards, *turn.rangers)}
        return frozenset(
            unit_id
            for members, _ in self._expedition_squads
            for unit_id in members
            if unit_id in live
        )

    def _is_expedition_member(
        self,
        unit: Ranger | Vanguard,
        turn: Turn,
    ) -> bool:
        """True when the unit fights as part of an expedition squad."""
        return self.config.expedition_mode and unit.id in self._expedition_member_ids(
            turn
        )

    def _patrol_ids(self, turn: Turn) -> frozenset[UUID]:
        """Return the nine live members of the three fixed patrol teams."""

        self._ensure_unit_roles(turn)
        live_ids = {str(unit.id) for unit in (*turn.vanguards, *turn.rangers)}
        return frozenset(
            UUID(unit_id)
            for unit_id, role in self.memory.unit_roles.items()
            if unit_id in live_ids and self._patrol_team_from_role(role) is not None
        )

    def _patrol_team_for(self, unit_id: UUID, turn: Turn) -> int | None:
        """Return the stable patrol team owning ``unit_id``."""

        if not self.config.expedition_mode:
            offensive = sorted(self._offensive_ids(turn), key=lambda value: value.bytes)
            if unit_id in offensive:
                return offensive.index(unit_id) % self._patrol_team_count() + 1
            return None
        self._ensure_unit_roles(turn)
        return self._patrol_team_from_role(self.memory.unit_roles.get(str(unit_id)))

    def _patrol_team_members(
        self, team: int, turn: Turn
    ) -> tuple[Ranger | Vanguard, ...]:
        """Return one patrol team's live members in stable UUID order."""

        if not self.config.expedition_mode:
            offensive = self._offensive_ids(turn)
            combat = sorted(
                (
                    unit
                    for unit in (*turn.vanguards, *turn.rangers)
                    if unit.id in offensive
                ),
                key=lambda unit: unit.id.bytes,
            )
            return tuple(
                unit
                for index, unit in enumerate(combat)
                if index % self._patrol_team_count() + 1 == team
            )  # type: ignore[return-value]
        role = f"{PATROL_ROLE_PREFIX}{team}"
        return tuple(
            sorted(
                (
                    unit
                    for unit in (*turn.vanguards, *turn.rangers)
                    if self.memory.unit_roles.get(str(unit.id)) == role
                ),
                key=lambda unit: unit.id.bytes,
            )
        )  # type: ignore[return-value]

    def _update_expeditions(self, turn: Turn) -> None:
        """Maintain staging and expedition squads for the active-offense mode.

        Surplus combat units are staged at the ring edge as soon as they are
        built.  Once a full expedition (2 Vanguards + 2 Rangers) is available
        it departs with one shared exploration route. Existing
        defence, patrol, and expedition UUIDs are never reclassified merely
        because another UUID sorts before them.
        """

        self._reconcile_unit_roles(turn)
        core = turn.core
        live = {unit.id for unit in (*turn.vanguards, *turn.rangers)}
        survivors = [
            (members, members & live, bearing)
            for members, bearing in self._expedition_squads
            if members & live
        ]
        self._expedition_squads = [
            (remaining, bearing) for _original, remaining, bearing in survivors
        ]
        self._expedition_pursuits = {
            remaining: self._expedition_pursuits.get(original, _ExpeditionPursuit())
            for original, remaining, _bearing in survivors
        }
        self._expedition_serial_by_members = {
            remaining: self._expedition_serial_by_members.get(original, index + 1)
            for index, (original, remaining, _bearing) in enumerate(survivors)
        }
        self._expedition_routes = {
            remaining: self._expedition_routes.get(original, ExplorationRoute())
            for original, remaining, _bearing in survivors
        }
        self._expedition_regroup_orders = {
            remaining: tuple(
                member
                for member in self._expedition_regroup_orders.get(original, ())
                if member in remaining
            )
            for original, remaining, _bearing in survivors
        }
        staged_vanguards = sorted(
            (
                unit.id
                for unit in turn.vanguards
                if self.memory.unit_roles.get(str(unit.id)) == STAGED_ROLE
            ),
            key=lambda unit_id: unit_id.bytes,
        )
        staged_rangers = sorted(
            (
                unit.id
                for unit in turn.rangers
                if self.memory.unit_roles.get(str(unit.id)) == STAGED_ROLE
            ),
            key=lambda unit_id: unit_id.bytes,
        )

        while (
            len(staged_vanguards) >= EXPEDITION_SQUAD_VANGUARDS
            and len(staged_rangers) >= EXPEDITION_SQUAD_RANGERS
        ):
            members = frozenset(
                staged_vanguards[:EXPEDITION_SQUAD_VANGUARDS]
                + staged_rangers[:EXPEDITION_SQUAD_RANGERS]
            )
            bearing = self._expedition_bearing(core)
            self._expedition_squads.append((members, bearing))
            self._expedition_pursuits[members] = _ExpeditionPursuit()
            self._expedition_routes[members] = ExplorationRoute()
            self._expedition_serial += 1
            self._expedition_serial_by_members[members] = self._expedition_serial
            role = f"{EXPEDITION_ROLE_PREFIX}{self._expedition_serial}"
            for member_id in members:
                self.memory.unit_roles[str(member_id)] = role
            del staged_vanguards[:EXPEDITION_SQUAD_VANGUARDS]
            del staged_rangers[:EXPEDITION_SQUAD_RANGERS]
        self._staged_ids = frozenset(staged_vanguards) | frozenset(staged_rangers)
        self._persist_expedition_pursuits()
        self.memory.next_expedition_serial = self._expedition_serial

    def _expedition_bearing(self, core: Core | None) -> Position:
        """Choose a deterministic initial heading to break exploration ties.

        The bearing is derived from the Core identity and a running squad
        serial. Actual information gain and other squads' route reservations
        decide where to travel after assembly.
        """

        offsets = (
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        )
        seed = (core.id.int if core is not None else 0) + self._expedition_serial * 7
        return offsets[seed % len(offsets)]

    def _expedition_squad_for(
        self, unit_id: UUID
    ) -> tuple[frozenset[UUID], Position] | None:
        return next(
            (
                (members, bearing)
                for members, bearing in self._expedition_squads
                if unit_id in members
            ),
            None,
        )

    def _expedition_goal(self, unit: Ranger | Vanguard, turn: Turn) -> Position:
        """Return the same short route checkpoint to every squad member."""

        self._refresh_expedition_navigation(turn)
        squad = self._expedition_squad_for(unit.id)
        if squad is None:
            return unit.position
        route = self._expedition_routes.get(squad[0], ExplorationRoute())
        return route.waypoint if route.waypoint is not None else unit.position

    def _refresh_expedition_navigation(self, turn: Turn) -> None:
        """Select and reserve one information-gathering route per whole squad."""

        if self._expedition_navigation_turn is turn:
            return
        self._expedition_navigation_turn = turn
        self.memory.observe_exploration(turn)
        obstacles = self.memory.obstacles | set(turn.obstacle_cells)
        blocked = (
            obstacles
            | set(self.memory.contested_positions)
            | {enemy.position for enemy in turn.visible_enemies}
        )
        for enemy in turn.visible_enemies:
            if isinstance(enemy, UnitView) and enemy.unit_type is UnitType.RANGER:
                blocked.update(
                    cell
                    for cell in firing_positions(enemy.position)
                    if line_of_fire(enemy.position, cell, obstacles)
                )
            elif isinstance(enemy, UnitView) and enemy.unit_type is UnitType.VANGUARD:
                blocked.update(adjacent_positions(enemy.position))
        alive = {unit.id: unit for unit in (*turn.vanguards, *turn.rangers)}
        navigators: dict[frozenset[UUID], ExpeditionNavigator] = {}
        reservations: dict[frozenset[UUID], set[Position]] = {}
        for members, _bearing in self._expedition_squads:
            live_members = [alive[mid] for mid in members if mid in alive]
            if not live_members:
                continue
            navigator = ExpeditionNavigator(
                self.memory.exploration,
                obstacles,
                blocked,
                max(_combat_vision_radius(member) for member in live_members),
            )
            navigators[members] = navigator
            reservations[members] = navigator.reservation(
                self._expedition_routes.get(members, ExplorationRoute())
            )

        headings: dict[frozenset[UUID], Position] = {}
        for members, bearing in self._expedition_squads:
            if members not in navigators:
                continue
            ordered = self._ordered_expedition_members(turn, members, bearing)
            previous = self._expedition_routes.get(members, ExplorationRoute())
            pursuit = self._expedition_pursuits.get(members)
            if pursuit is not None and pursuit.target_id is not None:
                # Contact has its existing survival/combat priority. A route
                # selected before a chase must not later drag the team back.
                self._expedition_routes[members] = ExplorationRoute(
                    cooldowns=previous.cooldowns
                )
                reservations[members] = set()
                continue
            if any(
                manhattan(left.position, right.position) > EXPEDITION_LINK_RADIUS
                for left, right in pairwise(ordered)
            ):
                # Finish the existing forward regrouping chain before turning;
                # no detached member receives its own exploration assignment.
                self._expedition_routes[members] = replace(
                    previous, progress_tick=turn.tick
                )
                continue
            centroid = (
                sum(member.position[0] for member in ordered) // len(ordered),
                sum(member.position[1] for member in ordered) // len(ordered),
            )
            origin = min(
                ordered,
                key=lambda member: (
                    manhattan(member.position, centroid),
                    member.id.bytes,
                ),
            ).position
            reserved = {
                cell
                for other, cells in reservations.items()
                if other != members
                for cell in cells
            }
            navigator = navigators[members]
            route = navigator.plan(origin, bearing, turn.tick, previous, reserved)
            self._expedition_routes[members] = route
            reservations[members] = navigator.reservation(route)
            if route.goal is not None and route.goal != previous.goal:
                delta = route.goal[0] - origin[0], route.goal[1] - origin[1]
                headings[members] = (
                    0 if delta[0] == 0 else (1 if delta[0] > 0 else -1),
                    0 if delta[1] == 0 else (1 if delta[1] > 0 else -1),
                )
        self._expedition_squads = [
            (members, headings.get(members, bearing))
            for members, bearing in self._expedition_squads
        ]
        self._persist_expedition_pursuits()

    def _refresh_expedition_pursuits(self, turn: Turn) -> None:
        """Update each squad's durable target without allowing silent disengage."""

        alive = {unit.id: unit for unit in (*turn.vanguards, *turn.rangers)}
        visible_by_squad: dict[frozenset[UUID], dict[str, CoreView | UnitView]] = {}
        for members, _bearing in self._expedition_squads:
            pursuit = self._expedition_pursuits.setdefault(
                members, _ExpeditionPursuit()
            )
            live_members = [
                alive[member_id] for member_id in members if member_id in alive
            ]
            visible: dict[str, CoreView | UnitView] = {}
            for member in live_members:
                visible.update(
                    {
                        str(enemy.id): enemy
                        for enemy in self._visible_combat_targets(member, turn)
                    }
                )
            visible_by_squad[members] = visible
            previous_positions = pursuit.last_positions
            if previous_positions:
                pursuit.distance += max(
                    (
                        manhattan(unit.position, previous_positions[unit.id])
                        for unit in live_members
                        if unit.id in previous_positions
                    ),
                    default=0,
                )
            pursuit.last_positions = {unit.id: unit.position for unit in live_members}
            current = pursuit.target_id
            new_target_ids = [
                target_id for target_id in visible if target_id != current
            ]
            if current is None or new_target_ids:
                candidates = tuple(visible.values())
                if candidates:
                    chosen = max(
                        candidates,
                        key=lambda enemy: (
                            self._combat_target_priority(enemy),
                            self._enemy_score(enemy, live_members[0].position, turn),
                            str(enemy.id),
                        ),
                    )
                    pursuit.target_id = str(chosen.id)
                    pursuit.position = chosen.position
                    pursuit.direction = (0, 0)
                    pursuit.distance = 0
            elif current in visible:
                current_view = visible[current]
                if pursuit.position is not None:
                    delta = (
                        current_view.position[0] - pursuit.position[0],
                        current_view.position[1] - pursuit.position[1],
                    )
                    if delta != (0, 0):
                        pursuit.direction = (
                            0 if delta[0] == 0 else (1 if delta[0] > 0 else -1),
                            0 if delta[1] == 0 else (1 if delta[1] > 0 else -1),
                        )
                pursuit.position = current_view.position
            elif pursuit.position is not None and pursuit.direction != (0, 0):
                pursuit.position = (
                    pursuit.position[0] + pursuit.direction[0],
                    pursuit.position[1] + pursuit.direction[1],
                )
            elif pursuit.position is not None:
                history = self.memory.enemy_position_history.get(current or "", [])
                if len(history) >= 2:
                    previous = history[-2][1]
                    latest = history[-1][1]
                    delta = latest[0] - previous[0], latest[1] - previous[1]
                    if delta != (0, 0):
                        pursuit.direction = (
                            0 if delta[0] == 0 else (1 if delta[0] > 0 else -1),
                            0 if delta[1] == 0 else (1 if delta[1] > 0 else -1),
                        )
                if pursuit.direction == (0, 0) and live_members:
                    centroid = (
                        sum(unit.position[0] for unit in live_members)
                        // len(live_members),
                        sum(unit.position[1] for unit in live_members)
                        // len(live_members),
                    )
                    delta = (
                        pursuit.position[0] - centroid[0],
                        pursuit.position[1] - centroid[1],
                    )
                    pursuit.direction = (
                        0 if delta[0] == 0 else (1 if delta[0] > 0 else -1),
                        0 if delta[1] == 0 else (1 if delta[1] > 0 else -1),
                    )
            if (
                pursuit.target_id is not None
                and pursuit.position is not None
                and pursuit.distance >= EXPEDITION_PURSUIT_MIN_DISTANCE
                and not visible
            ):
                pursuit.target_id = None
                pursuit.position = None
                pursuit.direction = (0, 0)
                pursuit.distance = 0
        self._persist_expedition_pursuits()

    def _persist_expedition_pursuits(self) -> None:
        """Copy in-memory pursuit state into the durable squad records."""

        self.memory.expedition_squads = [
            ExpeditionSquad(
                serial=self._expedition_serial_by_members.get(members, index + 1),
                members=tuple(
                    str(member_id)
                    for member_id in sorted(members, key=lambda value: value.bytes)
                ),
                bearing=bearing,
                pursuit_target_id=self._expedition_pursuits.get(
                    members, _ExpeditionPursuit()
                ).target_id,
                pursuit_position=self._expedition_pursuits.get(
                    members, _ExpeditionPursuit()
                ).position,
                pursuit_direction=self._expedition_pursuits.get(
                    members, _ExpeditionPursuit()
                ).direction,
                pursuit_distance=self._expedition_pursuits.get(
                    members, _ExpeditionPursuit()
                ).distance,
                exploration_route=self._expedition_routes.get(
                    members, ExplorationRoute()
                ),
                regroup_order=tuple(
                    str(member)
                    for member in self._expedition_regroup_orders.get(members, ())
                ),
            )
            for index, (members, bearing) in enumerate(self._expedition_squads)
        ]

    def _expedition_pursuit_for(
        self, unit: Ranger | Vanguard
    ) -> _ExpeditionPursuit | None:
        squad = self._expedition_squad_for(unit.id)
        return self._expedition_pursuits.get(squad[0]) if squad is not None else None

    def _expedition_chase_goal(
        self, unit: Ranger | Vanguard, context: _TurnContext
    ) -> Position | None:
        pursuit = self._expedition_pursuit_for(unit)
        if pursuit is None or pursuit.target_id is None or pursuit.position is None:
            return None
        return pursuit.position

    def _regroup_quiet_expedition(
        self, unit: Ranger | Vanguard, context: _TurnContext
    ) -> bool:
        """Reconnect before quiet travel; callers handle immediate combat first."""

        goal = self._expedition_rendezvous_goal(unit, context.turn, whole_squad=True)
        if goal is None:
            return False
        if goal == unit.position:
            self._record_wait(
                unit, context, "hold for the expedition's laggards to close up"
            )
        elif not self._move(
            unit,
            goal,
            context,
            reason="close up so the expedition's line of sight stays connected",
            path_expansions=EXPEDITION_REJOIN_PATH_EXPANSIONS,
            require_path=True,
        ):
            self._record_wait(unit, context, "no safe route to rejoin the expedition")
        return True

    def _expedition_attack_goal(
        self,
        ranger: Ranger,
        target: CoreView | UnitView,
        context: _TurnContext,
    ) -> Position | None:
        """Return a free cell that advances a lone Ranger on its target.

        A surviving Ranger keeps its expedition identity, but it no longer has
        a squad route to follow.  Firing from a legal cell is normally enough
        for a full squad; for a singleton, repeatedly firing at a nearby
        moving target can leave it pinned to one cell indefinitely.  Approach
        the target through a free adjacent cell so the residual expedition
        continues attacking without inventing a return-to-Core path.
        """

        current_distance = manhattan(ranger.position, target.position)
        if current_distance <= 1:
            return None
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        candidates = [
            position
            for position in adjacent_positions(target.position)
            if position != ranger.position
            and position not in obstacles
            and position not in context.enemy_positions
            and manhattan(position, target.position) < current_distance
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda position: (
                manhattan(ranger.position, position),
                manhattan(position, target.position),
                position,
            ),
        )

    def _decide_expedition_ranger(self, ranger: Ranger, context: _TurnContext) -> bool:
        pursuit = self._expedition_pursuit_for(ranger)
        visible = self._visible_combat_targets(ranger, context.turn)
        target = None
        if pursuit is not None and pursuit.target_id is not None:
            target = next(
                (enemy for enemy in visible if str(enemy.id) == pursuit.target_id),
                None,
            )
        if target is None:
            target = self._best_visible_target(ranger.position, context.turn, visible)
        if not visible and self._regroup_quiet_expedition(ranger, context):
            return True
        if pursuit is None or pursuit.target_id is None:
            return False
        target = next(
            (enemy for enemy in visible if str(enemy.id) == pursuit.target_id), None
        )
        if target is not None:
            squad = self._expedition_squad_for(ranger.id)
            if (
                squad is not None
                and len(squad[0]) == 1
                and (
                    isinstance(target, CoreView)
                    or target.unit_type is not UnitType.WORKER
                )
            ):
                attack_goal = self._expedition_attack_goal(ranger, target, context)
                if attack_goal is not None and self._move(
                    ranger,
                    attack_goal,
                    context,
                    reason="advance on visible expedition target",
                ):
                    return True
            shot_cell = self._ranger_shot_cell(
                ranger, target, context.turn, self.memory.obstacles
            )
            if shot_cell is not None:
                ranger.shoot(target, expected_cell=shot_cell)
                context.damage_ledger.record(target)
                context.report.add(
                    actor_id=str(ranger.id),
                    actor_kind="RANGER",
                    action="SHOOT",
                    reason="aggressive expedition fire",
                    target=shot_cell,
                )
                return True
            goal = self._ranger_approach_goal(ranger, target, context)
            if goal is not None and self._move(
                ranger, goal, context, reason="pursue expedition target"
            ):
                return True
        goal = self._expedition_chase_goal(ranger, context)
        return goal is not None and self._move(
            ranger, goal, context, reason="continue expedition pursuit after lost sight"
        )

    def _decide_expedition_vanguard(
        self, vanguard: Vanguard, context: _TurnContext
    ) -> bool:
        pursuit = self._expedition_pursuit_for(vanguard)
        visible = self._visible_combat_targets(vanguard, context.turn)
        if pursuit is None or pursuit.target_id is None:
            if not visible and self._regroup_quiet_expedition(vanguard, context):
                return True
            target = self._best_visible_target(vanguard.position, context.turn, visible)
            return target is not None and self._expedition_close_to_engage(
                vanguard, target, context, offensive=True
            )
        pursued_target = next(
            (enemy for enemy in visible if str(enemy.id) == pursuit.target_id), None
        )
        if any(
            weight >= SWEEP_LIKELY_WEIGHT
            for _targets, weight in self._sweep_groups(
                vanguard, context, visible
            ).values()
        ):
            return False
        if not visible and self._regroup_quiet_expedition(vanguard, context):
            return True
        target = pursued_target
        if target is not None and self._expedition_close_to_engage(
            vanguard, target, context, offensive=True
        ):
            return True
        goal = self._expedition_chase_goal(vanguard, context)
        return goal is not None and self._move(
            vanguard,
            goal,
            context,
            reason="continue expedition pursuit after lost sight",
        )

    def _ordered_expedition_members(
        self,
        turn: Turn,
        members: frozenset[UUID],
        bearing: Position,
    ) -> list[Ranger | Vanguard]:
        """Keep the same leaders throughout a detour until the chain closes.

        Projection orders an intact travelling squad. Reapplying that order
        while disconnected makes detouring members swap leaders every few
        steps and abandon the path around the wall. Persist the broken chain
        by UUID, including across a restart, until its links reconnect.
        """

        alive = {
            member.id: member
            for member in (*turn.vanguards, *turn.rangers)
            if member.id in members
        }
        previous = self._expedition_regroup_orders.get(members, ())
        previous_order = (
            [alive[member] for member in previous]
            if len(previous) == len(alive) and set(previous) == alive.keys()
            else []
        )
        ordered = sorted(
            alive.values(),
            key=lambda member: (
                member.position[0] * bearing[0] + member.position[1] * bearing[1],
                member.position,
                member.id.bytes,
            ),
        )
        if previous_order:
            previous_broken = any(
                manhattan(left.position, right.position) > EXPEDITION_LINK_RADIUS
                for left, right in pairwise(previous_order)
            )
            canonical_broken = any(
                manhattan(left.position, right.position) > EXPEDITION_LINK_RADIUS
                for left, right in pairwise(ordered)
            )
            # A connected chain can have a different projection order whose
            # adjacent links are disconnected. Switching immediately between
            # those two orders makes the middle pair trade leaders forever.
            # Keep the proven connected order until the canonical order is
            # itself safe to adopt.
            if previous_broken or canonical_broken:
                return previous_order
        if any(
            manhattan(left.position, right.position) > EXPEDITION_LINK_RADIUS
            for left, right in pairwise(ordered)
        ):
            self._expedition_regroup_orders[members] = tuple(
                member.id for member in ordered
            )
        else:
            self._expedition_regroup_orders.pop(members, None)
        return ordered

    def _expedition_rendezvous_goal(
        self,
        unit: Ranger | Vanguard,
        turn: Turn,
        *,
        whole_squad: bool = False,
    ) -> Position | None:
        """Return where a member must move to keep the squad connected.

        Squad members follow a chain that stays fixed while reconnecting,
        and each member only enforces the gap to its two *adjacent*
        neighbours in that chain.  When its gap to the neighbour behind it
        exceeds ``EXPEDITION_LINK_RADIUS`` the member has raced ahead and
        holds in place; when its gap to the neighbour ahead exceeds the radius
        it closes forward.  A leader therefore stops to wait the moment the
        rearmost links fall out of sight, instead of sprinting on unaware.
        ``None`` means the member is still connected and should keep
        exploring; otherwise the returned cell is the neighbour to close on, or
        its own cell to wait in.
        """

        squad = self._expedition_squad_for(unit.id)
        if squad is None:
            return None
        members, bearing = squad
        ordered = self._ordered_expedition_members(turn, members, bearing)
        if len(ordered) < 2:
            return None

        own_index = next(
            (index for index, member in enumerate(ordered) if member.id == unit.id),
            None,
        )
        if own_index is None:
            return None
        laggard = ordered[own_index - 1].position if own_index > 0 else None
        leader = (
            ordered[own_index + 1].position if own_index + 1 < len(ordered) else None
        )

        # A gap opened ahead of us: the member lags and must close forward.
        # This takes precedence when both gaps are open.  A middle member can
        # be separated from the tail by an obstacle detour while also falling
        # behind the front; waiting for the tail in that state deadlocks the
        # chain because the tail is trying to close on the waiting member.
        if leader is not None and (
            manhattan(unit.position, leader) > EXPEDITION_LINK_RADIUS
        ):
            return self._expedition_close_cell(unit, leader, turn)
        # A gap opened behind us: the member raced ahead, so it stops and lets
        # the tail close up rather than sprinting away from it.
        if laggard is not None and (
            manhattan(unit.position, laggard) > EXPEDITION_LINK_RADIUS
        ):
            return unit.position
        if whole_squad and any(
            manhattan(left.position, right.position) > EXPEDITION_LINK_RADIUS
            for left, right in pairwise(ordered)
        ):
            # The member's immediate links can be intact while another link
            # is open. Follow its local leader instead of independently
            # chasing a distant enemy on the other side of the broken chain.
            return (
                self._expedition_close_cell(unit, leader, turn)
                if leader is not None
                else unit.position
            )
        return None

    def _expedition_close_cell(
        self,
        unit: Ranger | Vanguard,
        leader: Position,
        turn: Turn,
    ) -> Position:
        """Return a standable cell beside ``leader`` for a member to close on.

        Prefer the leader's most forward reachable neighbour to keep the
        squad spread along its bearing while closing gaps in sight coverage.
        Teammate positions influence the destination, but never block the
        route to it; members may overlap and pass through one another.
        """

        static = (
            self.memory.obstacles
            | set(self.memory.contested_positions)
            | set(turn.obstacle_cells)
            | {enemy.position for enemy in turn.visible_enemies}
        )
        teammates = {
            member.id: member.position for member in (*turn.vanguards, *turn.rangers)
        }
        squad = self._expedition_squad_for(unit.id)
        members, bearing = squad if squad is not None else (frozenset(), (0, 0))
        occupied_by_team = {teammates[mid] for mid in members if mid in teammates}
        candidates = [
            cell
            for cell in adjacent_positions(leader)
            if cell not in static
            and cell not in occupied_by_team
            and cell != unit.position
        ]
        if not candidates:
            return leader
        reachable = [
            cell
            for cell in candidates
            if next_step(
                unit.position,
                cell,
                blocked=static,
                require_path=True,
                max_expansions=EXPEDITION_REJOIN_PATH_EXPANSIONS,
            )
            is not None
        ]
        pool = reachable if reachable else candidates
        bear_x, bear_y = bearing

        def forward(cell: Position) -> int:
            return cell[0] * bear_x + cell[1] * bear_y

        return min(
            pool,
            key=lambda cell: (
                -forward(cell),
                manhattan(unit.position, cell),
                manhattan(cell, leader),
                cell,
            ),
        )

    def _expedition_shared_target(
        self,
        turn: Turn,
        squad_members: frozenset[UUID],
        *,
        include_remembered: bool = True,
    ) -> Position | None:
        """Return the single enemy cell the whole squad should converge on.

        A squad fights as one body: members never scatter after different
        targets.  The best enemy visible to any member becomes the shared
        target that pulls the entire squad forward together, and when only a
        memory remains the nearest remembered hostile is followed so the squad
        advances on one heading instead of splitting.
        """

        alive = {member.id: member for member in (*turn.vanguards, *turn.rangers)}
        members = [
            member for member_id in squad_members if (member := alive.get(member_id))
        ]
        if not members:
            return None
        centroid = (
            sum(member.position[0] for member in members) // len(members),
            sum(member.position[1] for member in members) // len(members),
        )
        best: tuple[tuple[int, int, int, str], CoreView | UnitView] | None = None
        for member in members:
            visible = self._visible_combat_targets(member, turn)
            for enemy in visible:
                score = (
                    self._combat_target_priority(enemy),
                    self._enemy_score(enemy, centroid, turn),
                    0,
                    str(enemy.id),
                )
                if best is None or score > best[0]:
                    best = (score, enemy)
        if best is not None:
            return best[1].position
        if not include_remembered:
            return None
        remembered = (
            self._offensive_enemies(turn)
            if self._offensive_patrol_enabled(turn)
            else ()
        )
        if remembered:
            target = self._best_remembered_target(centroid, remembered)
            if target is not None:
                return target.position
        return None

    def _expedition_close_to_engage(
        self,
        vanguard: Vanguard,
        target: CoreView | UnitView,
        context: _TurnContext,
        *,
        offensive: bool,
    ) -> bool:
        """Advance onto ``target`` like the non-expedition rush, overriding cohesion."""
        intercept = self._intercept_cell(target, context)
        for anchor in dict.fromkeys((intercept, target.position)):
            candidates = [
                position
                for position in adjacent_positions(anchor)
                if position not in self.memory.obstacles
                and position not in context.enemy_positions
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
            return self._move_within_leash(
                vanguard,
                goal,
                context,
                offensive=offensive,
                reason="close on the ranged attacker instead of waiting",
            )
        return False

    def _staging_goal(self, unit: Ranger | Vanguard, turn: Turn) -> Position:
        """Hold a surplus unit at a unique cell outside the defensive ring.

        Resolve hash collisions in stable order to spread sight coverage.
        Preserve Units that have already reached a valid staging cell so new
        arrivals do not rearrange the ring.  These distinct destinations do
        not prevent friendly units from sharing cells along their routes.
        """
        core = turn.core
        if core is None:
            return unit.position
        offsets = _defensive_ring_offsets(EXPEDITION_STAGING_RADIUS)
        obstacles = self.memory.obstacles | set(turn.obstacle_cells)
        cells = tuple(
            (core.position[0] + dx, core.position[1] + dy)
            for dx, dy in offsets
            if (core.position[0] + dx, core.position[1] + dy) not in obstacles
        )
        if not cells:
            return unit.position

        staged = sorted(
            (
                candidate
                for candidate in (*turn.vanguards, *turn.rangers)
                if candidate.id in self._staged_ids
            ),
            key=lambda candidate: candidate.id.bytes,
        )
        assignments: dict[UUID, Position] = {}
        claimed: set[Position] = set()

        # A Unit already on the staging ring has finished the journey.  Let it
        # keep that cell before assigning hashed preferences to Units in transit.
        valid_cells = set(cells)
        for candidate in staged:
            if candidate.position in valid_cells and candidate.position not in claimed:
                assignments[candidate.id] = candidate.position
                claimed.add(candidate.position)

        for candidate in staged:
            if candidate.id in assignments:
                continue
            preferred = candidate.id.int % len(cells)
            goal = next(
                (
                    cells[index % len(cells)]
                    for index in range(preferred, preferred + len(cells))
                    if cells[index % len(cells)] not in claimed
                ),
                None,
            )
            if goal is None:
                assignments[candidate.id] = candidate.position
                continue
            assignments[candidate.id] = goal
            claimed.add(goal)

        return assignments.get(unit.id, unit.position)

    def _is_offensive_combat_unit(
        self,
        unit: Ranger | Vanguard,
        turn: Turn,
        threat: ThreatAssessment | None = None,
    ) -> bool:
        """Assign a stable minority of combat units to the roaming squad."""
        if self.config.expedition_mode:
            # Expedition members never return; patrol members roam outside the
            # ring.  Everything else (formation defenders and staged surplus)
            # holds station.
            if unit.id in self._expedition_member_ids(turn):
                return True
            if unit.id in self._patrol_ids(turn):
                return not (
                    self._emergency_combat_mode(turn)
                    or not self._offensive_patrol_enabled(turn)
                    or (threat is not None and threat.level is not ThreatLevel.NORMAL)
                )
            return False

        if (
            self._emergency_combat_mode(turn)
            or not self._offensive_patrol_enabled(turn)
            or (threat is not None and threat.level is not ThreatLevel.NORMAL)
        ):
            return False
        return unit.id in self._offensive_ids(turn)

    def _offensive_ids(self, turn: Turn) -> frozenset[UUID]:
        """Return the capped, stable roaming squad.

        The share used to be an open-ended UUID third, which grows with the
        roster: at 49 combat units that sent 15 of them out, and because the
        patrol route is a bounding box they stood a median 60 and up to 87
        cells from the Core -- far past any radius they could be recalled
        across.  A fixed squad keeps scouting coverage while every further
        unit built adds to the defence instead of the expedition.  UUID
        membership still decides *who* goes, so roles stay stable between
        Ticks without extra state.
        """

        if self.config.expedition_mode:
            return self._patrol_ids(turn) | self._expedition_member_ids(turn)

        cache = self._offensive_squad
        roster = frozenset(unit.id for unit in (*turn.rangers, *turn.vanguards))
        if cache is not None and cache[0] == roster:
            return cache[1]
        preferred = sorted(
            (unit_id for unit_id in roster if unit_id.int % 3 == 0),
            key=lambda unit_id: unit_id.bytes,
        )
        squad = frozenset(preferred[: max(0, self.config.offensive_squad_size)])
        self._offensive_squad = (roster, squad)
        return squad

    def _offensive_patrol_enabled(self, turn: Turn) -> bool:
        """Keep a minority roaming while the Core itself remains safe.

        Patrol is the standing posture whenever the Core is intact and the
        combat roster can spare a roaming squad.  The resource floor is set
        to the previous population tier's full capacity minus the cost of one
        Ranger at that tier, so a normal full-capacity production lands
        exactly on the floor and keeps the patrol running.  Emergency combat
        spending falls below the floor and recalls the squad; an intact Core
        and a combat roster below the minimum keep everyone home as before.
        """

        core = turn.core
        return (
            self._unbounded_growth()
            and core is not None
            and len(turn.vanguards) + len(turn.rangers)
            >= self.config.offensive_min_combat_units
            and core.hp >= 5
            and core.shield >= 5
            and turn.resources >= self._patrol_reserve_floor(turn)
        )

    def _patrol_reserve_floor(self, turn: Turn) -> int:
        """Return the resource level below which the roaming squad recalls.

        The floor is ``previous_population``'s Core capacity minus one Ranger
        price at that same tier.  The previous tier is used because production
        first raises population and then capacity by five, while a full-capacity
        normal production spends exactly one Ranger: the post-spawn balance
        equals this floor, so routine growth does not toggle patrol off.
        Emergency combat spending, which can occur below full capacity, drops
        resources under the floor and thus pulls the roaming squad home.
        """

        previous_population = max(0, turn.state.population - 1)
        return max(
            0,
            core_resource_capacity(previous_population)
            - unit_cost(UnitType.RANGER, previous_population),
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
        if self.config.expedition_mode:
            if unit.id in self._expedition_member_ids(turn):
                self._refresh_expedition_navigation(turn)
                squad = self._expedition_squad_for(unit.id)
                if squad is not None:
                    members, _ = squad
                    visible_shared = self._expedition_shared_target(
                        turn, members, include_remembered=False
                    )
                    if visible_shared is not None:
                        return (
                            visible_shared,
                            "advance together on the squad's enemy target",
                        )
                rendezvous = self._expedition_rendezvous_goal(unit, turn)
                if rendezvous is not None:
                    if rendezvous == unit.position:
                        return (
                            unit.position,
                            "hold for the expedition's laggards to close up",
                        )
                    return (
                        rendezvous,
                        "close up so the expedition's line of sight stays connected",
                    )
                # A squad may only acquire a new target from its own local
                # sight.  Global enemy memory belongs to the defensive and
                # patrol policies; feeding it here can pull an expedition
                # hundreds of cells sideways after its bounded pursuit has
                # expired, leaving the formation circling an obstacle pocket.
                return (
                    self._expedition_goal(unit, turn),
                    "explore new ground along the squad's shared route",
                )
            if unit.id in self._staged_ids:
                return (
                    self._staging_goal(unit, turn),
                    "hold at the defensive ring edge awaiting expedition",
                )
        if (
            context is not None
            and self._raid_target is not None
            and unit.id in context.raid_ids
        ):
            return (
                self._raid_target,
                "raid the enemy Core along the attack bearing",
            )
        if (
            context is not None
            and context.perimeter_intercept_anchor is not None
            and unit.id in context.perimeter_intercept_ids
        ):
            return (
                context.perimeter_intercept_anchor,
                "converge on the intruder inside the defensive perimeter",
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
        goal = context.defensive_assignments.get(unit.id, unit.position)
        if goal in self.memory.obstacles or goal in turn.obstacle_cells:
            # Retain the intended post until the Core reaches a clear home.
            # Moving just this seat would silently break the mirrored shape.
            return unit.position, "hold while the assigned defensive post is obstructed"
        return (
            goal,
            "hold a defensive perimeter around the resource Core",
        )

    @staticmethod
    def _symmetric_defense_offsets() -> tuple[
        tuple[Position, ...], tuple[Position, ...]
    ]:
        """Return eight mirrored Vanguard spokes and two Ranger posts each."""

        vanguards = (
            (0, -6),
            (4, -4),
            (6, 0),
            (4, 4),
            (0, 6),
            (-4, 4),
            (-6, 0),
            (-4, -4),
        )
        rangers = (
            (0, -8),
            (0, -10),
            (6, -6),
            (8, -8),
            (8, 0),
            (10, 0),
            (6, 6),
            (8, 8),
            (0, 8),
            (0, 10),
            (-6, 6),
            (-8, 8),
            (-8, 0),
            (-10, 0),
            (-6, -6),
            (-8, -8),
        )
        return vanguards, rangers

    def _symmetric_home_candidate(self, context: _TurnContext) -> Position | None:
        """Pick the nearest clear center with the complete symmetric footprint."""

        core = context.turn.core
        if core is None:
            return None
        vanguard_offsets, ranger_offsets = self._symmetric_defense_offsets()
        offsets = (*vanguard_offsets, *ranger_offsets)
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        resources = set(context.turn.resource_cells) | set(self.memory.resource_cells)
        occupied = set(context.occupied) - {core.position}
        candidates: list[tuple[int, int, int, Position]] = []
        for dx in range(-24, 25):
            for dy in range(-24, 25):
                if not (dx or dy):
                    continue
                position = (core.position[0] + dx, core.position[1] + dy)
                footprint = {(position[0] + ox, position[1] + oy) for ox, oy in offsets}
                blocked = len(footprint & obstacles)
                if blocked:
                    continue
                if (
                    position in obstacles
                    or position in resources
                    or position in occupied
                ):
                    continue
                local_obstacles = sum(
                    (position[0] + ox, position[1] + oy) in obstacles
                    for ox in range(-2, 3)
                    for oy in range(-2, 3)
                )
                nearby_resources = sum(
                    manhattan(position, resource)
                    <= self.config.resource_outreach_radius
                    for resource in resources
                )
                candidates.append(
                    (local_obstacles, -nearby_resources, abs(dx) + abs(dy), position)
                )
        return min(candidates)[-1] if candidates else None

    def _core_home_direction(self, context: _TurnContext) -> Direction | None:
        """Walk the Core to the persisted clear center for the new formation."""

        core = context.turn.core
        if core is None:
            return None
        target = self.memory.core_home_position
        obstacles = self.memory.obstacles | set(context.turn.obstacle_cells)
        footprint_offsets = (
            *self._symmetric_defense_offsets()[0],
            *self._symmetric_defense_offsets()[1],
        )
        if target is None or any(
            (target[0] + dx, target[1] + dy) in obstacles
            for dx, dy in footprint_offsets
        ):
            target = self._symmetric_home_candidate(context)
            self.memory.core_home_position = target
        if target is None or core.position == target:
            return None
        blocked = (
            obstacles
            | set(context.turn.resource_cells)
            | set(context.occupied)
            | context.reserved
        ) - {core.position}
        return next_step(
            core.position,
            target,
            blocked=blocked,
            recent=self.memory.recent_positions(str(core.id)),
            direction_offset=self._direction_offset(core.id),
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
            if self._symmetric_posture():
                self.memory.defense_posts.clear()
                self.memory.defense_anchor = None
                self._defensive_layout = None
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
        if self._symmetric_posture():
            defense_ids = {
                unit_id
                for unit_id, role in self.memory.unit_roles.items()
                if role == DEFENSE_ROLE
            }
            guards = tuple(unit for unit in guards if str(unit.id) in defense_ids)
        if not guards:
            self._defensive_layout = None
            if self._symmetric_posture():
                self.memory.defense_posts.clear()
                self.memory.defense_anchor = core_position
            return {}
        ordered_guard_ids = tuple(unit.id for unit in guards)

        if self._symmetric_posture():
            # Posts belong to persistent UUIDs, not to each Tick's sorted
            # roster.  A casualty leaves one vacancy; its replacement inherits
            # only that slot.  An incomplete roster keeps the same template.
            anchor = self.memory.defense_anchor or core_position
            previous = {
                unit_id: (
                    position[0] + core_position[0] - anchor[0],
                    position[1] + core_position[1] - anchor[1],
                )
                for unit_id, position in self.memory.defense_posts.items()
            }
            assignments: dict[UUID, Position] = {}
            vanguard_offsets, ranger_offsets = self._symmetric_defense_offsets()
            for unit_type, offsets in (
                (UnitType.VANGUARD, vanguard_offsets),
                (UnitType.RANGER, ranger_offsets),
            ):
                # Friendly traffic can pass through a guard.  Do not nudge
                # individual posts away from pocket exits or occupied cells.
                assignments.update(
                    _assign_fixed_defense_posts(
                        tuple(unit for unit in guards if unit.unit_type is unit_type),
                        tuple(
                            (core_position[0] + dx, core_position[1] + dy)
                            for dx, dy in offsets
                        ),
                        previous,
                    )
                )
            self.memory.defense_posts = {
                str(unit_id): position for unit_id, position in assignments.items()
            }
            self.memory.defense_anchor = core_position
            self._defensive_layout = _DefensiveLayout(
                core_id=core.id,
                core_position=core_position,
                guard_ids=ordered_guard_ids,
                radius=SYMMETRIC_CORE_RADIUS,
                assignments=assignments,
            )
            return assignments

        obstacles = set(self.memory.obstacles)
        obstacles.update(context.turn.obstacle_cells)
        cached = self._defensive_layout
        if (
            cached is not None
            and cached.core_id == core.id
            and cached.core_position == core_position
            and cached.guard_ids == ordered_guard_ids
            and not any(slot in obstacles for slot in cached.assignments.values())
        ):
            return dict(cached.assignments)

        # Workers can pass through stationary guards, including at pocket
        # exits.  Choose posts for coverage without reserving traffic lanes.
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

        ring_cache: dict[int, tuple[Position, ...]] = {}
        free_cache: dict[int, list[Position]] = {}

        def ring_for(radius: int) -> tuple[Position, ...]:
            if radius not in ring_cache:
                ring_cache[radius] = tuple(
                    (core_position[0] + dx, core_position[1] + dy)
                    for dx, dy in _defensive_ring_offsets(radius)
                    if (core_position[0] + dx, core_position[1] + dy) not in obstacles
                )
            return ring_cache[radius]

        def free_cells(radius: int) -> list[Position]:
            if radius not in free_cache:
                free_cache[radius] = [
                    position
                    for position in ring_for(radius)
                    if position not in unavailable
                ]
            return free_cache[radius]

        def ring_plan(outer: int) -> tuple[tuple[tuple[int, int], ...], int]:
            """Spread the guards over rings from ``outer`` inward.

            Guards beyond the outer ring's spaced slots fill the interior
            instead of pushing the perimeter further out.  Each ring first
            takes only its spaced slots, so neighbours stay
            ``DEFENSIVE_RING_SPACING`` apart and every guard still sees the
            next ring inward; if the whole roster still does not fit, the
            rings tighten from the outside in.  The second return value is
            how many guards found no legal cell at all.
            """

            radii = [
                radius
                for radius in range(
                    outer,
                    DEFENSIVE_PERIMETER_MIN_RADIUS - 1,
                    -DEFENSIVE_RING_SPACING,
                )
                if free_cells(radius)
            ]
            counts: dict[int, int] = {}
            remaining = len(guards)
            # First pass: every ring takes only its spaced slots, outermost
            # first, so the ring that has to see an approach coming is filled
            # before the depth behind it.
            for radius in radii:
                if remaining <= 0:
                    break
                room = min(
                    len(free_cells(radius)),
                    max(1, len(free_cells(radius)) // DEFENSIVE_RING_SPACING),
                )
                share = min(remaining, room)
                if share > 0:
                    counts[radius] = counts.get(radius, 0) + share
                    remaining -= share
            # Second pass: a roster too large for spaced slots tightens every
            # ring together instead of packing the outer one wall-to-wall,
            # which would spend the surplus on overlapping vision rather than
            # on depth.
            while remaining > 0:
                seated = 0
                for radius in radii:
                    if remaining <= 0:
                        break
                    if counts.get(radius, 0) >= len(free_cells(radius)):
                        continue
                    counts[radius] = counts.get(radius, 0) + 1
                    remaining -= 1
                    seated += 1
                if not seated:
                    break
            plan = tuple(
                (radius, counts[radius]) for radius in radii if radius in counts
            )
            return plan, remaining

        # Expand only while terrain and hostile positions leave too few
        # distinct posts, keeping the count/vision radius as the primary choice.
        max_radius = base_radius
        search_limit = base_radius + max(8, len(guards) * 2)
        while max_radius < search_limit and ring_plan(max_radius)[1]:
            max_radius += 1

        plan, _ = ring_plan(max_radius)
        if len(plan) > 1:
            # More guards than the outer ring's spaced slots: the surplus goes
            # inward.  Depth is what a single circle cannot buy -- an intruder
            # that crosses it can only be chased from behind, and an
            # equal-speed runner is uncatchable that way.
            outer_radius = plan[0][0]
            assignments = _assign_layered_ring_slots(
                plan,
                {radius: free_cells(radius) for radius, _ in plan},
                guards,
                visions,
                tuple(free_cells(outer_radius)),
                core_position,
            )
            assignments = _repair_layered_outer_ring(
                assignments,
                ring_for(outer_radius),
                free_cells(outer_radius),
                visions,
                obstacles,
                core_position,
                outer_radius,
            )
            self._defensive_layout = _DefensiveLayout(
                core_id=core.id,
                core_position=core_position,
                guard_ids=ordered_guard_ids,
                radius=plan[0][0],
                assignments=dict(assignments),
            )
            return dict(assignments)

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
        """Choose a waypoint for the Unit's independent three-member team."""

        core = turn.core
        if core is None:
            return unit.position, "no Core available for offensive patrol"
        team = self._patrol_team_for(unit.id, turn)
        if team is None:
            return unit.position, "no patrol team assigned"
        unit_id = str(unit.id)
        purpose = f"{COMBAT_PATROL_PURPOSE}-{team}"
        current = self.memory.goal_for(unit_id)
        claimed_positions = {
            goal.position
            for other in (*turn.rangers, *turn.vanguards)
            if other.id != unit.id
            if (goal := self.memory.goal_for(str(other.id))) is not None
            and goal.purpose.startswith(COMBAT_PATROL_PURPOSE)
            and self._patrol_team_for(other.id, turn) != team
        }
        if (
            current is not None
            and current.purpose == purpose
            and unit.position != current.position
            and current.position not in self.memory.obstacles
            and current.position not in claimed_positions
            and manhattan(core.position, current.position)
            <= self.config.offensive_patrol_radius
            and turn.tick - current.assigned_tick
            <= self.config.offensive_patrol_goal_ttl
        ):
            return current.position, "search outward for enemy units and Cores"

        # ``_resource_patrol_offsets`` sweeps a bounding box, so its corners
        # sit at twice the nominal radius in Manhattan terms.  Keeping only
        # the offsets inside the leash makes the route mean what the config
        # name says: the roaming squad searches out to
        # ``offensive_patrol_radius`` and can still be recalled from there.
        offsets = tuple(
            offset
            for offset in _resource_patrol_offsets(
                self.config.offensive_patrol_radius,
                COMBAT_PATROL_SPACING,
            )
            if abs(offset[0]) + abs(offset[1]) <= self.config.offensive_patrol_radius
        ) or _defensive_ring_offsets(max(1, self.config.offensive_patrol_radius))
        team_members = self._patrol_team_members(team, turn)
        member_index = next(
            index for index, item in enumerate(team_members) if item.id == unit.id
        )
        local_offsets = ((0, 0), (0, 1), (1, 0))
        local_dx, local_dy = local_offsets[member_index % len(local_offsets)]
        phase = turn.tick // self.config.offensive_patrol_goal_ttl
        team_count = self._patrol_team_count()
        if self._symmetric_posture():
            # Each team owns one quadrant.  The route still advances through
            # that quadrant over time, but never crosses into a neighbour's
            # sector, keeping the four patrol responsibilities independent.
            quadrant = {
                1: (1, -1),  # NE
                2: (1, 1),  # SE
                3: (-1, 1),  # SW
                4: (-1, -1),  # NW
            }[team]
            qx, qy = quadrant
            quadrant_offsets = tuple(
                offset
                for offset in offsets
                if (offset[0] * qx >= 0 and offset[1] * qy >= 0) and offset != (0, 0)
            )
            offsets = quadrant_offsets or offsets
            team_spacing = max(1, len(offsets) // max(1, len(team_members)))
            offset_index = (phase + member_index * team_spacing) % len(offsets)
        else:
            team_spacing = max(1, len(offsets) // team_count)
            offset_index = ((team - 1) * team_spacing + phase) % len(offsets)
        if current is not None and current.purpose == purpose:
            current_offset = (
                current.position[0] - core.position[0] - local_dx,
                current.position[1] - core.position[1] - local_dy,
            )
            if current_offset in offsets:
                offset_index = (offsets.index(current_offset) + team_spacing) % len(
                    offsets
                )

        center_offset = offsets[offset_index]
        # Keep the three members close while giving each a distinct cell.  The
        # team still shares one route and phase; the small local offsets
        # spread its sight coverage around the shared waypoint.
        patrol_position = (
            core.position[0] + center_offset[0] + local_dx,
            core.position[1] + center_offset[1] + local_dy,
        )
        if (
            manhattan(core.position, patrol_position)
            > self.config.offensive_patrol_radius
            or patrol_position in self.memory.obstacles
            or patrol_position in claimed_positions
        ):
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
            purpose=purpose,
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
            and line_of_fire(position, target.position, obstacles)
            # Range-three diagonals have Manhattan distance six, outside a
            # Ranger's local vision. Such a firing goal made it lose contact
            # on arrival and immediately chase back out of the firing cell.
            and manhattan(position, target.position) <= RANGER_VISION_RADIUS
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
        """Assign known resource cells to the nearest empty Worker.

        Every cell currently visible or remembered as having held a resource
        is a valid target, regardless of how far it sits from the Core.
        Enemy and unreachable cells stay excluded so a Worker does not walk
        into a known threat or a walled-off route.
        """

        self._resource_scout_id = None
        workers = {worker.id: worker for worker in turn.workers if worker.cargo == 0}
        self._unreachable_claims = {
            cell: seen_tick
            for cell, seen_tick in self._unreachable_claims.items()
            if turn.tick - seen_tick < UNREACHABLE_CLAIM_COOLDOWN
        }
        # A Worker-only exclusion disk sits around every remembered enemy Core
        # and every recently seen hostile fighter, and ``_move`` refuses to
        # route a Worker into one.  The pairing below only measured distance, so
        # a cell inside a disk stayed the nearest candidate for the Workers
        # beside it and every claim on it became an orbit around the rim - one
        # that never arrives, so it never records the absence that would rest
        # the cell.  Take those cells out of the contest instead, together with
        # the ones a route has just failed to reach.  A cell a Worker already
        # stands on stays in: arrival harvests it.
        unreachable = (
            self._worker_threat_exclusion_cells(turn) | set(self._unreachable_claims)
        ) - {worker.position for worker in workers.values()}
        hostile = {enemy.position for enemy in turn.visible_enemies}
        blocked_resources = self.memory.obstacles | set(turn.obstacle_cells)
        visible_resources = (
            set(turn.resource_cells) - hostile - unreachable - blocked_resources
        )
        remembered_resources = (
            set(self.memory.remembered_resource_cells(turn.tick))
            - hostile
            - unreachable
            - blocked_resources
            - visible_resources
        )
        resources = visible_resources | remembered_resources
        rechecks = (
            set(self.memory.resource_cells_worth_rechecking(turn.tick))
            - hostile
            - unreachable
            - blocked_resources
            - resources
        )
        if not self._preserves_resources():
            rechecks = set()
        elif not resources and not rechecks and self._unbounded_growth() and workers:
            # With no known site anywhere, keep one deterministic Worker as a
            # bounded resource scout.  The other Workers fall through to the
            # local patrol ring instead of walking in lockstep.
            scout = self._resource_scout(workers)
            if scout is not None:
                self._resource_scout_id = scout.id
                workers = {scout.id: scout}
            else:
                workers = {}
        stale_candidates = resources | rechecks
        if (
            self._unbounded_growth()
            and len(workers) > 1
            and stale_candidates
            and turn.tick % RESOURCE_SCOUT_INTERVAL == 0
            and (
                (
                    not visible_resources
                    and all(
                        turn.tick - self.memory.resource_cells.get(resource, turn.tick)
                        >= RESOURCE_RECHECK_FLOOR
                        for resource in stale_candidates
                    )
                )
                or len(self.memory.resource_cells) >= RESOURCE_MEMORY_LIMIT
            )
        ):
            # A durable resource map eventually contains enough old sites to
            # give every Worker a claim forever, which suppresses the scout
            # branch even when the live view has gone quiet.  Keep every held
            # claim above intact and leave one otherwise-unassigned Worker to
            # refresh the map.  A full map is also treated as stale enough for
            # this fallback: its 2,048 retained cells otherwise keep a recent
            # timestamp somewhere forever and suppress discovery indefinitely.
            # The twelve-Tick cadence keeps that refresh alive while the same
            # Worker rejoins known-site pairing between refreshes.  Keep one
            # refresh Worker even when another Worker currently sees a
            # resource; the fresh site remains first in the pairing pool, and
            # a Worker standing on it is never selected as the scout.
            scout = self._resource_scout(
                workers,
                avoid_positions=visible_resources,
            )
            if scout is not None:
                self._resource_scout_id = scout.id
                workers.pop(scout.id)
        assignments: dict[UUID, Position] = {}
        # Live view and durable memory are equally trustworthy: a remembered
        # site is as good a target as one still in view, so both share one
        # candidate pool and the nearest pairings win.  The outreach band is no
        # longer a separate priority tier; it is simply the natural result of
        # ordering every known site by approach (and loaded-return) cost.
        core_position = turn.core.position if turn.core is not None else None
        pools: list[set[Position]] = []
        known = visible_resources | remembered_resources
        if known:
            pools.append(known)
        if rechecks:
            pools.append(rechecks)
        self._hold_existing_claims(workers, pools, assignments)

        def assignment_key(
            worker: Worker,
            resource: Position,
        ) -> tuple[int, int, UUID, Position]:
            approach = manhattan(worker.position, resource)
            core_distance = (
                0 if core_position is None else manhattan(core_position, resource)
            )
            # Every known target costs an unloaded approach and a loaded
            # return to Core.  The return is short in the inner ring, but it
            # still decides close alternatives there; ignoring it made a
            # Worker walk to the farther side of the ring for a small approach
            # saving and then pay the whole distance again with cargo.  Keep
            # every known site eligible and let the same soft round-trip cost
            # cover local, outer-band, and remote targets.
            return (
                approach
                + (
                    REMOTE_RETURN_WEIGHT
                    if core_distance > self.config.resource_patrol_radius * 2
                    else LOCAL_RETURN_WEIGHT
                )
                * core_distance,
                approach,
                worker.id,
                resource,
            )

        for pool in pools:
            # Freshest evidence and local range come first; within one pool,
            # the nearest/shortest-round-trip Worker pairing still wins so the
            # fleet does not cross itself or overpay for a remote return.
            while workers and pool:
                _, _, worker_id, resource = min(
                    (
                        assignment_key(worker, resource)
                        for worker in workers.values()
                        for resource in pool
                    ),
                )
                assignments[worker_id] = resource
                workers.pop(worker_id)
                pool.remove(resource)
        self._uncross_claims(turn, assignments)
        self._prioritize_underfoot_resource_claims(
            turn,
            visible_resources,
            assignments,
        )
        return assignments

    def _prioritize_underfoot_resource_claims(
        self,
        turn: Turn,
        visible_resources: set[Position],
        assignments: dict[UUID, Position],
    ) -> None:
        """Swap a live underfoot resource with its current claimant.

        A Worker can pass over a visible resource while its retained claim
        points elsewhere.  When another Worker already owns the underfoot
        cell, exchanging the two claims harvests the confirmed opportunity
        without abandoning either cell.  If nobody owns that cell, keep the
        original claim: the hysteresis is specifically meant to prevent a
        Worker from dropping an approach that no other Worker will finish.
        """

        workers = {worker.id: worker for worker in turn.workers if worker.cargo == 0}
        for resource in sorted(visible_resources):
            underfoot = sorted(
                (worker for worker in workers.values() if worker.position == resource),
                key=lambda worker: worker.id.bytes,
            )
            if not underfoot:
                continue
            harvester = underfoot[0]
            current_claim = assignments.get(harvester.id)
            if current_claim == resource:
                continue
            owner = next(
                (
                    worker_id
                    for worker_id, target in assignments.items()
                    if worker_id != harvester.id and target == resource
                ),
                None,
            )
            if owner is None:
                held = self.memory.goal_for(str(harvester.id))
                if (
                    held is not None
                    and held.purpose == RESOURCE_CLAIM_PURPOSE
                    and current_claim == held.position
                ):
                    # The Worker is committed to an active claim it is still
                    # progressing toward; the hysteresis pins who walks to
                    # what, so a live cell underfoot must not steal that route.
                    continue
                # Nobody else owns the live cell underfoot, so the Worker
                # harvests the confirmed resource instead of walking away; its
                # prior approach falls back to the pool for another Worker.
                assignments[harvester.id] = resource
                continue
            if current_claim is None:
                assignments[harvester.id] = resource
                assignments.pop(owner)
            else:
                assignments[harvester.id], assignments[owner] = (
                    resource,
                    current_claim,
                )

    def _release_unreachable_claim(
        self,
        worker: Worker,
        cell: Position,
        context: _TurnContext,
    ) -> None:
        """Rest a claimed cell no route reaches and free the Worker."""

        self._unreachable_claims[cell] = context.turn.tick
        context.resource_assignments.pop(worker.id, None)
        self._claim_stalls.pop(worker.id, None)
        goal = self.memory.goal_for(str(worker.id))
        if goal is not None and goal.purpose == RESOURCE_CLAIM_PURPOSE:
            self.memory.clear_goal(str(worker.id))

    def _uncross_claims(
        self,
        turn: Turn,
        assignments: dict[UUID, Position],
    ) -> None:
        """Trade two claims when the Workers would otherwise walk past each other.

        The greedy pairing and the claim hold are both per-Worker decisions,
        so the fleet drifts into crossings: a Worker heading north-west passes
        one heading south-east and each is closer to the other's cell.
        Measured over Ticks 169752-170556, 66.4% of Ticks carried at least one
        such pair, and 4280 cells of travel sat in the longer half of them.

        A swap is not the reassignment ``_hold_existing_claims`` refuses.
        That guard protects the *cell*: an abandoned approach never confirms
        what is there, so nothing feeds back into the recheck machinery.  A
        swap keeps every claimed cell claimed and only trades who walks to it,
        so both approaches still terminate in an observation.
        """

        positions = {worker.id: worker.position for worker in turn.workers}
        for _ in range(len(assignments)):
            improvement: tuple[int, UUID, UUID] | None = None
            claimed = sorted(assignments)
            for index, first in enumerate(claimed):
                for second in claimed[index + 1 :]:
                    start, other = positions[first], positions[second]
                    goal, other_goal = assignments[first], assignments[second]
                    current = manhattan(start, goal) + manhattan(other, other_goal)
                    swapped = manhattan(start, other_goal) + manhattan(other, goal)
                    gain = current - swapped
                    if gain > 0 and (improvement is None or gain > improvement[0]):
                        improvement = (gain, first, second)
            if improvement is None:
                return
            _, first, second = improvement
            assignments[first], assignments[second] = (
                assignments[second],
                assignments[first],
            )

    def _hold_existing_claims(
        self,
        workers: dict[UUID, Worker],
        pools: list[set[Position]],
        assignments: dict[UUID, Position],
    ) -> None:
        """Keep a Worker on the cell it already claimed while it closes in.

        The greedy pass re-matches every candidate from scratch each Tick, so
        as the Workers walk the minimal pairing flips and two of them trade
        targets mid-route.  Measured over Ticks 167705-167742: 42 of 99 target
        changes happened with the old cell still more than one cell away, and
        22.9% of all Worker travel went into approaches that were reassigned
        before arrival.  One Worker dropped a cell 11 out for one 32 out.

        Holding a claim can cost distance on the Tick it is held, because the
        pairing it refuses really is the shorter one.  It buys something worth
        more at a 1.8% refill rate: an approach that terminates in an
        observation.  An abandoned approach never confirms its cell, so it
        feeds nothing back into the absence and recheck machinery that decides
        where the next Worker goes.

        The bound is only against *abandonment*.  When another Worker takes
        the released cell the approach is handed off rather than dropped, and
        ``_uncross_claims`` is free to trade the two claims afterwards.

        Arrival needs no special case.  A Worker reaching an empty cell
        records the absence before this runs, which takes the cell out of
        every pool for RESOURCE_RECHECK_FLOOR Ticks, so the observation itself
        releases the claim.  Only a Worker that cannot close the distance
        needs a bound, which is what the stall budget provides.
        """

        stalls = self._claim_stalls
        for worker in list(workers.values()):
            goal = self.memory.goal_for(str(worker.id))
            if goal is None or goal.purpose != RESOURCE_CLAIM_PURPOSE:
                continue
            pool = next((item for item in pools if goal.position in item), None)
            if pool is None:
                continue
            reference = goal.last_progress_position
            closing = reference is None or manhattan(
                worker.position, goal.position
            ) < manhattan(reference, goal.position)
            stalled = 0 if closing else stalls.get(worker.id, 0) + 1
            if stalled > CLAIM_STALL_BUDGET:
                continue
            stalls[worker.id] = stalled
            assignments[worker.id] = goal.position
            workers.pop(worker.id)
            pool.remove(goal.position)
        for worker_id in set(stalls) - set(assignments):
            # Drops both released claims and Workers that no longer exist.
            del stalls[worker_id]

    def _resource_scout(
        self,
        workers: dict[UUID, Worker],
        *,
        avoid_positions: set[Position] | frozenset[Position] = frozenset(),
    ) -> Worker | None:
        """Choose the stable unclaimed Worker reserved for map refreshes."""

        candidates = [
            worker
            for worker in workers.values()
            if (
                (goal := self.memory.goal_for(str(worker.id))) is None
                or goal.purpose != RESOURCE_CLAIM_PURPOSE
            )
            and worker.position not in avoid_positions
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda worker: worker.id.bytes)

    def _is_resource_scout(self, worker: Worker, turn: Turn) -> bool:
        """Return whether this Worker is the sole long-range economy scout."""

        return self._unbounded_growth() and worker.id == self._resource_scout_id

    def _has_local_resource_cells(self, turn: Turn) -> bool:
        """Return whether a visible, non-hostile resource is inside the loop."""

        if turn.core is None:
            return False
        enemy_positions = {enemy.position for enemy in turn.visible_enemies}
        local_radius = self.config.resource_patrol_radius * 2
        return any(
            resource not in enemy_positions
            and manhattan(turn.core.position, resource) <= local_radius
            for resource in turn.resource_cells
        )

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

        if self.config.expedition_mode and unit.id in self._expedition_member_ids(
            context.turn
        ):
            # Expeditions have no distance limit.
            return EXPEDITION_HORIZON
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

        offensive_ids = self._offensive_ids(turn)
        guards = sorted(
            (
                unit
                for unit in (*turn.rangers, *turn.vanguards)
                if unit.id not in offensive_ids and unit.id not in raid_ids
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

        if self.config.expedition_mode:
            # The bounded raid is superseded by the non-returning expedition
            # squads, which already seek and destroy every hostile they see.
            self._raid_ids = frozenset()
            self._raid_target = None
            self._raid_until_tick = -1
            return frozenset()

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
                or self._raid_recall_needed(turn, threat)
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
            or self._raid_recall_needed(turn, threat)
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

    def _raid_recall_needed(self, turn: Turn, threat: ThreatAssessment) -> bool:
        """Whether the Core itself needs the detachment back at home.

        ``ThreatAssessment.combat_pressure`` cannot answer this question.  It
        is true whenever any unit of ours took damage in the last few Ticks,
        anywhere on the map, so a detachment sent out to fight recalls itself
        the moment it trades its first shot - and the trigger that launches it
        is a burst of kills, which leaves a launch window one quiet Tick wide
        in the middle of a battle.  Replaying the engagement this rule comes
        from shows exactly that: the raid formed on the single Tick the fight
        fell quiet and dissolved on the next, while no enemy fighter was
        visible at all and none had been inside the alert ring for a hundred
        Ticks.

        Only evidence about the Core counts here - damage it has taken or is
        in line to take, hostiles closing on it or pursuing towards it, and a
        fleet already inside the alert ring.  The rest is the noise of a fight
        the detachment is supposed to be finishing.
        """

        return (
            threat.should_evacuate_core
            or bool(threat.near_core_enemy_ids)
            or self._emergency_combat_mode(turn)
        )

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
        perimeter_intercept_ids: frozenset[UUID] = frozenset(),
    ) -> frozenset[UUID]:
        """Recall an intercepted roaming pair for a bounded return window.

        The raid detachment is exempt: making contact is the point of the
        raid, so the generic intercept recall would cancel it on arrival.
        ``_update_raid`` owns that decision instead.  The squad converging on a
        lone fighter inside the defensive zone is exempt for the same reason:
        it was detached precisely because that contact has to be finished, and
        recalling the only unit in touch with the target is what let single
        intruders walk through a stationary army.
        """

        live_ids = {unit.id for unit in (*turn.vanguards, *turn.rangers)}
        self._squad_return_until = {
            unit_id: until
            for unit_id, until in self._squad_return_until.items()
            if unit_id in live_ids and until >= turn.tick
        }
        if self.config.expedition_mode:
            # Expedition members never return; the generic intercept recall
            # must not pull them back into the defensive zone.  The patrol
            # contingent is different: it is only a bounded scouting screen,
            # so a local fighter contact should send it home before it gets
            # trapped far outside the Core.  The old early return protected
            # the expedition but accidentally left those patrol members in
            # the normal chase branch, which cost two remote patrol deaths in
            # one hostile contact.
            self._squad_return_until = {
                unit_id: until
                for unit_id, until in self._squad_return_until.items()
                if unit_id not in self._expedition_member_ids(turn)
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
            patrol_ids = self._patrol_ids(turn)
            expedition_members = self._expedition_member_ids(turn)
            for unit in (*turn.vanguards, *turn.rangers):
                if (
                    unit.id not in patrol_ids
                    or unit.id in expedition_members
                    or unit.id in raid_ids
                    or unit.id in perimeter_intercept_ids
                ):
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
                if (
                    unit is not None
                    and manhattan(unit.position, turn.core.position) <= 3
                ):
                    self._squad_return_until.pop(unit_id, None)
            return frozenset(self._squad_return_until)
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
                if (
                    unit.id.int % 3 != 0
                    or unit.id in raid_ids
                    or unit.id in perimeter_intercept_ids
                ):
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

    def _resource_patrol_goal(
        self,
        worker: Worker,
        turn: Turn,
        context: _TurnContext,
    ) -> Position:
        """Assign a stable, statically reachable patrol point.

        A patrol point can remain a perfectly valid coordinate while every
        route to it is sealed by the remembered obstacle map.  Reusing that
        goal then turns a Worker into a stationary WAIT stream until its goal
        TTL expires.  Check static reachability before holding or creating a
        goal; ``_move`` also checks current enemy positions.
        """

        core = turn.core
        if core is None:
            return worker.position
        obstacles = self.memory.obstacles | set(turn.obstacle_cells)
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
            and current.position not in obstacles
            and current.position not in claimed_positions
            and not self._near_remembered_worker_danger(current.position, turn)
            and manhattan(core.position, current.position)
            <= self.config.resource_patrol_radius * 2
            and self._has_static_route(
                worker,
                current.position,
                context,
                allow_goal=True,
            )
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
                and candidate not in obstacles
                and candidate not in claimed_positions
                and not self._near_remembered_worker_danger(candidate, turn)
                and self._has_static_route(
                    worker,
                    candidate,
                    context,
                    allow_goal=True,
                )
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
            if (
                (
                    enemy.kind == "CORE"
                    and turn.tick - enemy.tick <= self.config.enemy_memory_ttl
                )
                or (
                    enemy.kind == "UNIT"
                    and enemy.unit_type
                    in {UnitType.VANGUARD.value, UnitType.RANGER.value}
                    and turn.tick - enemy.tick <= self.config.worker_threat_memory_ttl
                )
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

    def _core_can_spawn(self, context: _TurnContext) -> bool:
        core = context.turn.core
        growth_target = self._growth_population_target()
        if core is None:
            return False
        if growth_target is not None and context.turn.state.population >= growth_target:
            return False
        if (
            self.config.resource_target > 0
            and context.remaining_resources >= self.config.resource_target
        ):
            return False
        if context.core_service_action is not None:
            return False
        core_position = core.position
        # A spawn appears on the Core cell.  Account for current and planned
        # occupants, and wait one Tick after departures or inbound traffic so
        # resolution order cannot turn a valid-looking plan into a limit hit.
        if context.departure_counts.get(core_position, 0):
            return False
        if context.arrival_counts.get(core_position, 0):
            return False
        return self._predicted_friendly_occupancy(core_position, context) < 2

    def _choose_spawn(self, turn: Turn, resources: int) -> UnitType | None:
        growth_target = self._growth_population_target()
        if growth_target is not None and turn.state.population >= growth_target:
            return None
        if self.config.expedition_mode:
            return self._expedition_spawn(turn, resources)
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
        elif len(turn.rangers) < len(turn.vanguards):
            # Even Vanguard/Ranger expansion, and Workers deliberately stay
            # out of the ratio once ``target_workers`` is met.  Every unit
            # built raises the price of all later ones, so a Worker added at
            # this population tier is paid for twice: once in its own cost and
            # again in the inflation it puts on the combat units after it.
            # Income scales as the square root of the Worker count anyway,
            # because reaching more concurrent sites needs a wider radius.
            candidates = (UnitType.RANGER, UnitType.VANGUARD)
        else:
            candidates = (UnitType.VANGUARD, UnitType.RANGER)
        return self._affordable_spawn(
            turn,
            resources,
            candidates,
            strict_preference=not emergency,
        )

    def _expedition_spawn(self, turn: Turn, resources: int) -> UnitType | None:
        """Choose the next Unit for the active-offense formation.

        The formation fills 12 Workers, 16 Vanguards and 32 Rangers first,
        then spends surplus production on an alternating 1:1 Vanguard/Ranger
        stream whose units are staged at the ring edge.  Emergency combat
        decided by the existing threat logic still buys the type that best
        balances the local roster, and the early resource-guard sequence is
        preserved so a freshly respawned Core always establishes its first
        combat unit before widening the economy.
        """

        if self._emergency_combat_mode(turn):
            candidates = (
                (UnitType.RANGER, UnitType.VANGUARD)
                if len(turn.rangers) * 2 < len(turn.vanguards)
                else (UnitType.VANGUARD, UnitType.RANGER)
            )
            return self._affordable_spawn(
                turn, resources, candidates, strict_preference=False
            )
        if self._needs_resource_guard(turn):
            if not turn.vanguards:
                candidates = (UnitType.VANGUARD,)
            elif not turn.rangers:
                candidates = (UnitType.RANGER, UnitType.VANGUARD)
            else:
                candidates = (UnitType.VANGUARD, UnitType.RANGER)
            return self._affordable_spawn(
                turn, resources, candidates, strict_preference=True
            )
        vanguards = len(turn.vanguards)
        rangers = len(turn.rangers)
        formation_vanguards, formation_rangers, _, _ = self._formation_quota()
        if len(turn.workers) < self.config.target_workers:
            candidates = (UnitType.WORKER,)
        elif vanguards < formation_vanguards or rangers < formation_rangers:
            # Fill the formation gaps first, preferring the type whose
            # formation quota is furthest behind (targets a 1:2 V:R split).
            vanguard_short = max(0, formation_vanguards - vanguards)
            ranger_short = max(0, formation_rangers - rangers)
            if vanguard_short <= 0:
                candidates = (UnitType.RANGER,)
            elif ranger_short <= 0:
                candidates = (UnitType.VANGUARD,)
            elif ranger_short >= 2 * vanguard_short:
                candidates = (UnitType.RANGER, UnitType.VANGUARD)
            else:
                candidates = (UnitType.VANGUARD, UnitType.RANGER)
        else:
            # Formation full: surplus built 1:1, balancing the two quotas so
            # staged units can pair off into full expeditions.
            surplus_vanguards = vanguards - formation_vanguards
            surplus_rangers = rangers - formation_rangers
            candidates = (
                (UnitType.RANGER, UnitType.VANGUARD)
                if surplus_rangers < surplus_vanguards
                else (UnitType.VANGUARD, UnitType.RANGER)
            )
        return self._affordable_spawn(
            turn, resources, candidates, strict_preference=True
        )

    def _normal_growth_on_cooldown(self, context: _TurnContext) -> bool:
        """Throttle routine high-tier growth while leaving emergency spend free.

        The throttle is a payback gate with a post-growth stockpile floor:
        routine growth waits until the bank has re-earned what the previous
        spawn spent and the next Core can retain a healthy reserve.  There is
        deliberately no wall-clock cooldown beside it.  One used to sit here
        (675 Tick) and it could only ever fire while the bank was already
        pinned at capacity, because banking-tier production is affordable
        exactly at a full Core -- ``_spawn_safety_reserve`` clamps the 90%
        reserve to ``capacity - cost``, so ``cost + reserve`` is identically
        ``capacity``.  Waiting out a timer on a full bank buys nothing: the
        bank cannot grow, the reserve is already ~18x a full Core recovery,
        and enemy pressure bypasses this gate anyway.  It also costs real
        income, because a full Core refuses loaded Workers the single deposit
        cell and harvesting stalls (live delivery rate fell from 2.22 to 1.47
        per Worker-hour in the Tick 180699-181373 window).  The timer's stated
        win -- net resources going from -17 to +33 across Tick 178621-179973 --
        was the arithmetic of buying one Ranger fewer, not a healthier economy:
        income was 0.108 vs 0.116 per Tick across those two windows.  Cap
        population instead if roster growth needs a limit; that is what
        ``max_population`` is for.

        The payback target must stay inside the Core, or the throttle stops
        being a delay and becomes a deadlock: a full Core cannot bank one more
        resource, and because it also refuses loaded Workers the single deposit
        cell, the whole team fills up and jams the Core neighbourhood instead.
        """

        # ``context.emergency`` also covers a recent attack on a remote
        # combat unit so the fleet can coordinate.  That is not, by itself,
        # a reason to spend the high-tier stockpile: only a damaged Core or
        # an enemy in its local alert ring should bypass this reserve floor.
        if self._emergency_combat_mode(
            context.turn
        ) or not self._growth_slowdown_active(context.turn):
            return False
        next_ranger_cost = unit_cost(UnitType.RANGER, context.turn.state.population)
        next_unit = self._choose_spawn(context.turn, context.remaining_resources)
        next_unit_cost = (
            unit_cost(next_unit, context.turn.state.population)
            if next_unit is not None
            else next_ranger_cost
        )
        # A full Core refuses a loaded Worker's deposit cell, so the nominal
        # post-growth floor can become self-defeating: once every Worker has
        # cargo, no income can be banked to change the decision.  Spend one
        # affordable growth action to reopen the storage path, while keeping
        # the floor for a full but otherwise idle Core.
        if context.remaining_resources >= context.turn.resource_capacity and any(
            worker.cargo > 0 for worker in context.turn.workers
        ):
            return False
        # A high-cost unit can be affordable at a full current Core while
        # still dropping the newly enlarged Core below a healthy stockpile.
        # Price the actual preferred candidate so a cheaper Vanguard at the
        # edge of the tier is not rejected merely because a Ranger costs more.
        next_capacity = core_resource_capacity(context.turn.state.population + 1)
        minimum_post_growth_resources = (
            next_capacity * CORE_STOCKPILE_CAPACITY_PERCENT // 100
        )
        if context.remaining_resources - next_unit_cost < minimum_post_growth_resources:
            return True
        if self._needs_growth_marker_migration:
            self.memory.last_normal_growth_tick = context.turn.tick
            self.memory.last_normal_growth_resources = context.turn.resources
            self._needs_growth_marker_migration = False
            return True
        if self.memory.last_normal_growth_tick is None:
            return False
        if self.memory.last_normal_growth_resources is None:
            return False
        # Clamp the payback target to what the Core can physically hold.  The
        # baseline is a stored absolute balance, not a running income counter,
        # so any baseline above ``capacity - cost`` demands a balance the Core
        # can never reach and freezes production for good.  A restart writes
        # exactly such a baseline: the migration stamps the balance that
        # happens to be in the bank, not a post-production one.  Live Tick
        # 182024 stamped 201 against a 235 capacity and a 58-resource Ranger,
        # so the Core needed 259 to grow again and stopped producing forever.
        target = min(
            self.memory.last_normal_growth_resources + next_ranger_cost,
            context.turn.resource_capacity,
        )
        return context.turn.resources < target

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
        stockpile_target = self._stockpile_target(
            turn,
            population=(turn.state.population + 1)
            if production_cost is not None
            else None,
        )
        reserve = max(
            0,
            self.config.safety_reserve,
            missing_recovery,
            stockpile_target,
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
                return max(missing_recovery, stockpile_target)
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

    def _stockpile_target(
        self,
        turn: Turn,
        *,
        population: int | None = None,
    ) -> int:
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
            if self._growth_slowdown_active(turn, population=population)
            else CORE_STOCKPILE_CAPACITY_PERCENT
        )
        return max(CORE_CAPACITY_HIGH_RESERVE, capacity * percent // 100)

    def _growth_slowdown_active(
        self,
        turn: Turn,
        *,
        population: int | None = None,
    ) -> bool:
        """Return whether growth should yield to banking at this population.

        Unit prices rise 1.3x per five population while each Unit only adds
        five storage, so past the soft threshold every additional guard costs
        much more and adds very little to a Core that already holds a full
        defensive ring.  Enter the banking tier one Unit early: the next
        production action would otherwise cross the threshold, pay its old
        lower reserve, and leave the newly larger Core below its reserve for
        a long time.  Real enemy pressure lifts the slowdown so the bank can
        be spent on defenders at once.
        """

        threshold = self.config.growth_slowdown_population
        banking_threshold = max(1, threshold - 1) if threshold is not None else None
        current_population = turn.state.population if population is None else population
        return (
            banking_threshold is not None
            and self._unbounded_growth()
            and current_population >= banking_threshold
            and not self._emergency_combat_mode(turn)
        )

    def _unbounded_growth(self) -> bool:
        """Return whether the live strategy has no fixed population/stockpile goal."""

        return self.config.max_population is None and self.config.resource_target <= 0

    def _spawn_reason(self, unit_type: UnitType) -> str:
        if self.config.expedition_mode:
            return f"build active-offense {unit_type.value.lower()} roster"
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
    ) -> tuple[bool, bool, bytes]:
        at_core = worker.position == core_position
        # Process on-site deposits first for deterministic resource accounting.
        return not at_core, worker.cargo > 0, worker.id.bytes

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


def _assign_layered_ring_slots(
    plan: tuple[tuple[int, int], ...],
    cells_by_radius: dict[int, list[Position]],
    guards: tuple[Ranger | Vanguard, ...],
    visions: dict[UUID, int],
    outer_cells: tuple[Position, ...],
    core_position: Position,
) -> dict[UUID, Position]:
    """Seat guards on several concentric rings, outermost ring first.

    ``plan`` gives ``(radius, count)`` from the outside in.  The outer ring is
    filled first and with the longest-sighted guards, because it is the ring
    that has to see an approach coming; the inner rings are the depth that
    stops whatever crosses it.  Slots inside one ring are spread evenly in
    angular order so neighbours stay roughly ``DEFENSIVE_RING_SPACING`` apart,
    and each ring is rotated off the one outside it so an inner guard does not
    hide directly behind an outer one.
    """

    ordered = sorted(guards, key=lambda unit: (-visions[unit.id], unit.id.bytes))
    assignments: dict[UUID, Position] = {}
    cursor = 0
    for depth, (radius, count) in enumerate(plan):
        cells = cells_by_radius[radius]
        if not cells or count <= 0:
            continue
        picks: list[Position] = []
        if count >= len(cells):
            picks = list(cells)
        else:
            # Half-step rotation per ring keeps successive rings staggered.
            offset = (depth * len(cells)) // (2 * count) if count else 0
            for index in range(count):
                picks.append(
                    cells[(offset + (index * len(cells)) // count) % len(cells)]
                )
        if radius == plan[0][0]:
            # Anchoring the outer ring's cardinals keeps the four axes covered
            # even when sector rounding lands every centre on a diagonal.
            anchors = [
                position
                for dx, dy in ((0, -radius), (0, radius), (radius, 0), (-radius, 0))
                for position in ((core_position[0] + dx, core_position[1] + dy),)
                if position in cells
            ]
            index = {position: order for order, position in enumerate(cells)}
            chosen = set(picks)
            for anchor in anchors:
                if anchor in chosen:
                    continue
                replaceable = [
                    position for position in picks if position not in set(anchors)
                ]
                if not replaceable:
                    break
                nearest = min(
                    replaceable,
                    key=lambda position: min(
                        (index[position] - index[anchor]) % len(cells),
                        (index[anchor] - index[position]) % len(cells),
                    ),
                )
                picks[picks.index(nearest)] = anchor
                chosen = set(picks)
        for position in picks:
            if cursor >= len(ordered):
                break
            assignments[ordered[cursor].id] = position
            cursor += 1
    for unit in ordered[cursor:]:
        # No legal cell left anywhere: hold whatever ring cell is still free
        # rather than crowd the Core.
        spare = next(
            (
                position
                for position in outer_cells
                if position not in set(assignments.values())
            ),
            None,
        )
        if spare is None:
            break
        assignments[unit.id] = spare
    return assignments


def _repair_layered_outer_ring(
    assignments: dict[UUID, Position],
    outer_cells: tuple[Position, ...],
    outer_free: list[Position],
    visions: dict[UUID, int],
    obstacles: set[Position],
    core_position: Position,
    radius: int,
) -> dict[UUID, Position]:
    """Slide outer-ring guards until the outer ring has no blind cell.

    Even spacing can leave one ring cell unseen when obstacles thin the ring
    unevenly.  Only guards standing on the outer ring are moved, and only to
    other free outer-ring cells, so the inner rings keep their depth.  The
    four cardinals are pinned: they are the cells an approach along an axis
    crosses first.
    """

    outer_ids = [
        unit_id
        for unit_id, position in assignments.items()
        if manhattan(core_position, position) == radius
    ]
    if not outer_ids:
        return assignments
    anchors = {
        (core_position[0] + dx, core_position[1] + dy)
        for dx, dy in ((0, -radius), (0, radius), (radius, 0), (-radius, 0))
    }
    cache: dict[tuple[Position, int], set[Position]] = {}

    def seen(position: Position, vision: int) -> set[Position]:
        key = position, vision
        if key not in cache:
            cache[key] = {
                cell
                for cell in outer_cells
                if manhattan(position, cell) <= vision
                and _clear_manhattan_path(position, cell, obstacles)
            }
        return cache[key]

    def covered(layout: dict[UUID, Position]) -> int:
        cells: set[Position] = set()
        for unit_id, position in layout.items():
            cells.update(seen(position, visions[unit_id]))
        return len(cells)

    current = covered(assignments)
    while current < len(outer_cells):
        used = set(assignments.values())
        best: tuple[int, Position, UUID] | None = None
        for unit_id in outer_ids:
            if assignments[unit_id] in anchors:
                continue
            for candidate in outer_free:
                if candidate in used:
                    continue
                trial = dict(assignments)
                trial[unit_id] = candidate
                score = covered(trial)
                choice = (score, candidate, unit_id)
                if best is None or choice > best:
                    best = choice
        if best is None or best[0] <= current:
            break
        current, candidate, unit_id = best
        assignments[unit_id] = candidate
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


def _assign_fixed_defense_posts(
    guards: tuple[Ranger | Vanguard, ...],
    positions: tuple[Position, ...],
    previous: dict[str, Position],
) -> dict[UUID, Position]:
    """Keep surviving post owners, then fill vacancies with unassigned guards."""

    available = set(positions)
    assigned: dict[UUID, Position] = {}
    for unit in guards:
        post = previous.get(str(unit.id))
        if post is not None and post in available:
            assigned[unit.id] = post
            available.remove(post)
    # On migration from old memory, preserve guards already on valid posts
    # before assigning nearby recruits.  This only runs for unassigned UUIDs;
    # temporarily crossing another guard's post never transfers ownership.
    for unit in guards:
        if unit.id not in assigned and unit.position in available:
            assigned[unit.id] = unit.position
            available.remove(unit.position)
    for unit in guards:
        if unit.id in assigned or not available:
            continue
        post = min(
            (position for position in positions if position in available),
            key=lambda position: manhattan(unit.position, position),
        )
        assigned[unit.id] = post
        available.remove(post)
    return assigned


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
