"""Reusable combat policy primitives adapted to the Arena Hero rules.

The policy is deliberately independent from the unit-action controller.  It
keeps short-lived enemy motion memory, predicts when a closing fighter enters
attack range, scores Core escape lanes, and accounts for damage already queued
in the current Tick.  This separation makes the rules easy to replay without
submitting a plan to the server.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import ceil

from arena_hero import CoreView, Direction, Position, Turn, UnitType, UnitView

from .geometry import DIRECTIONS, add, line_of_fire, manhattan

CombatObject = CoreView | UnitView
COMBAT_UNIT_TYPES = frozenset({UnitType.VANGUARD, UnitType.RANGER})
CORE_EVADE_DISTANCE = 12
PREEMPTIVE_EVADE_HORIZON = 16
ACTIVE_ENEMY_ALERT_TICKS = 2
PURSUIT_MEMORY_TICKS = 2
PURSUIT_SCORE_THRESHOLD = 3
RECENT_ATTACK_MEMORY_TICKS = 6


class ThreatLevel(StrEnum):
    """Hierarchical threat levels used by the live controller."""

    NORMAL = "NORMAL"
    ALERT = "ALERT"
    PRE_EVADE = "PRE_EVADE"
    ENGAGED = "ENGAGED"
    BREAKOUT = "BREAKOUT"


@dataclass(frozen=True, slots=True)
class EnemyMotion:
    """Short-lived motion and pursuit evidence for one visible fighter."""

    object_id: str
    position: Position
    last_tick: int
    core_distance: int
    unit_type: UnitType
    pursuit_score: int = 0
    pursuit_ticks: int = 0
    active_until_tick: int = 0
    preemptive_until_tick: int = 0
    ticks_to_attack_range: int | None = None


@dataclass(frozen=True, slots=True)
class ThreatAssessment:
    """Evidence and resulting posture for one authoritative Turn."""

    level: ThreatLevel = ThreatLevel.NORMAL
    reason: str = "NONE"
    active_enemy_ids: frozenset[str] = frozenset()
    preemptive_enemy_ids: frozenset[str] = frozenset()
    pursuing_enemy_ids: frozenset[str] = frozenset()
    near_core_enemy_ids: frozenset[str] = frozenset()
    threatening_core_enemy_ids: frozenset[str] = frozenset()
    recent_attack: bool = False
    recent_core_attack: bool = False
    minimum_ticks_to_range: int | None = None
    projected_core_damage: int = 0

    @property
    def combat_pressure(self) -> bool:
        """Whether economy/raids should yield to combat preparation."""

        return self.level is not ThreatLevel.NORMAL or self.recent_attack

    @property
    def requires_coordination(self) -> bool:
        """Whether all local roles should switch to a combat formation."""

        return bool(
            self.level in {ThreatLevel.ENGAGED, ThreatLevel.BREAKOUT}
            or self.preemptive_enemy_ids
            or self.pursuing_enemy_ids
        )

    @property
    def should_evacuate_core(self) -> bool:
        """Whether starting Core movement is safer than waiting another Tick."""

        return bool(
            self.recent_core_attack
            or self.threatening_core_enemy_ids
            or self.preemptive_enemy_ids
            or self.pursuing_enemy_ids
        )


@dataclass(slots=True)
class CombatTargetLedger:
    """Same-Tick damage ledger that prevents avoidable overkill."""

    planned_damage: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def capacity(target: CombatObject) -> int:
        """Return the damage still required to destroy ``target``."""

        if isinstance(target, CoreView):
            return max(1, target.hp + target.shield)
        return max(1, target.hp)

    def remaining(self, target: CombatObject) -> int:
        """Return unallocated durability after this Tick's queued attacks."""

        return max(
            0, self.capacity(target) - self.planned_damage.get(str(target.id), 0)
        )

    def record(self, target: CombatObject, damage: int = 1) -> None:
        """Reserve expected damage for a target."""

        target_id = str(target.id)
        self.planned_damage[target_id] = self.planned_damage.get(target_id, 0) + damage

    def select(
        self,
        candidates: Sequence[CombatObject],
        focus: CombatObject | None,
        origin: Position,
        core_position: Position | None,
    ) -> CombatObject | None:
        """Choose an uncovered threat while retaining focus when possible."""

        if not candidates:
            return None
        viable = [
            candidate for candidate in candidates if self.remaining(candidate) > 0
        ]
        pool = viable or list(candidates)
        if focus is not None:
            focused = next(
                (candidate for candidate in pool if candidate.id == focus.id),
                None,
            )
            if focused is not None:
                return focused

        def selection_key(target: CombatObject) -> tuple[int, int, int, int, int, str]:
            return (
                _target_type_priority(target),
                _core_proximity_score(target, core_position),
                int(self.remaining(target) <= 1),
                -self.remaining(target),
                -manhattan(origin, target.position),
                str(target.id),
            )

        best = pool[0]
        best_key = selection_key(best)
        for candidate in pool[1:]:
            candidate_key = selection_key(candidate)
            if candidate_key > best_key:
                best = candidate
                best_key = candidate_key
        return best


class CombatPolicy:
    """Maintain enemy motion memory and evaluate the current threat posture."""

    def __init__(self) -> None:
        self.enemy_motion: dict[str, EnemyMotion] = {}
        self.recent_attack_until_tick = -1
        self.recent_core_attack_until_tick = -1

    def assess(
        self,
        turn: Turn,
        obstacles: set[Position] | frozenset[Position],
    ) -> ThreatAssessment:
        """Update motion memory and classify the current Turn."""

        core = turn.core
        visible = {
            str(enemy.id): enemy
            for enemy in turn.visible_enemies
            if isinstance(enemy, UnitView) and enemy.unit_type in COMBAT_UNIT_TYPES
        }
        prior = dict(self.enemy_motion)
        current_motion: dict[str, EnemyMotion] = {}
        for enemy_id, enemy in visible.items():
            previous = prior.get(enemy_id)
            core_distance = (
                manhattan(core.position, enemy.position) if core is not None else 2**31
            )
            motion = self._next_motion(
                enemy_id,
                enemy,
                turn.tick,
                core_distance,
                previous,
            )
            current_motion[enemy_id] = motion

        for enemy_id, motion in prior.items():
            if enemy_id in current_motion:
                continue
            hidden_ticks = turn.tick - motion.last_tick
            if hidden_ticks <= PURSUIT_MEMORY_TICKS or turn.tick <= max(
                motion.active_until_tick,
                motion.preemptive_until_tick,
            ):
                current_motion[enemy_id] = motion

        self.enemy_motion = current_motion
        self._observe_attack_events(turn, visible, prior)

        active_ids = frozenset(
            enemy_id
            for enemy_id, motion in current_motion.items()
            if turn.tick <= motion.active_until_tick
        )
        preemptive_ids = frozenset(
            enemy_id
            for enemy_id, motion in current_motion.items()
            if turn.tick <= motion.preemptive_until_tick
        )
        pursuing_ids = frozenset(
            enemy_id
            for enemy_id, motion in current_motion.items()
            if motion.pursuit_score >= PURSUIT_SCORE_THRESHOLD
            and (
                motion.core_distance <= CORE_EVADE_DISTANCE
                or motion.pursuit_score >= PURSUIT_SCORE_THRESHOLD
            )
        )
        near_core_ids = frozenset(
            enemy_id
            for enemy_id, enemy in visible.items()
            if core is not None
            and manhattan(core.position, enemy.position) <= CORE_EVADE_DISTANCE
        )
        threatening_ids = frozenset(
            enemy_id
            for enemy_id, enemy in visible.items()
            if core is not None and _can_attack_core(enemy, core.position, obstacles)
        )
        minimum_ticks = min(
            (
                motion.ticks_to_attack_range
                for motion in current_motion.values()
                if motion.ticks_to_attack_range is not None
            ),
            default=None,
        )
        recent_attack = turn.tick <= self.recent_attack_until_tick
        recent_core_attack = turn.tick <= self.recent_core_attack_until_tick
        projected_damage = (
            projected_core_damage(core.position, tuple(visible.values()), obstacles)
            if core is not None
            else 0
        )

        if recent_core_attack or threatening_ids:
            level, reason = ThreatLevel.ENGAGED, "CORE_ATTACK"
        elif preemptive_ids or pursuing_ids or near_core_ids:
            level, reason = ThreatLevel.PRE_EVADE, "TIME_TO_RANGE"
        elif recent_attack:
            level, reason = ThreatLevel.ENGAGED, "FLEET_ATTACK"
        elif active_ids:
            level, reason = ThreatLevel.ALERT, "HOSTILE_ACTIVITY"
        else:
            level, reason = ThreatLevel.NORMAL, "NONE"

        return ThreatAssessment(
            level=level,
            reason=reason,
            active_enemy_ids=active_ids,
            preemptive_enemy_ids=preemptive_ids,
            pursuing_enemy_ids=pursuing_ids,
            near_core_enemy_ids=near_core_ids,
            threatening_core_enemy_ids=threatening_ids,
            recent_attack=recent_attack,
            recent_core_attack=recent_core_attack,
            minimum_ticks_to_range=minimum_ticks,
            projected_core_damage=projected_damage,
        )

    def escape_direction(
        self,
        position: Position,
        enemies: Sequence[CombatObject],
        obstacles: set[Position] | frozenset[Position],
        blocked: set[Position] | frozenset[Position],
        *,
        beacon_position: Position | None = None,
        previous_direction: Direction | None = None,
    ) -> Direction | None:
        """Choose a multi-axis breakout lane using projected damage first."""

        if not enemies:
            return None
        enemy_positions = {enemy.position for enemy in enemies}
        current_damage = projected_core_damage(position, enemies, obstacles)
        candidates: list[
            tuple[tuple[int, tuple[int, ...], int, int, int], Direction]
        ] = []
        for index, direction in enumerate(DIRECTIONS):
            destination = add(position, direction)
            if destination in blocked or destination in obstacles:
                continue
            if destination in enemy_positions:
                continue
            damage = projected_core_damage(destination, enemies, obstacles)
            distances = tuple(
                sorted(manhattan(destination, enemy.position) for enemy in enemies)
            )
            beacon_distance = (
                manhattan(destination, beacon_position)
                if beacon_position is not None
                else 0
            )
            key = (
                damage,
                tuple(-distance for distance in distances),
                -beacon_distance,
                -int(direction is previous_direction),
                index,
            )
            candidates.append((key, direction))
        if not candidates:
            return None
        best_key, best_direction = min(candidates, key=lambda candidate: candidate[0])
        if current_damage > 0 and best_key[0] > current_damage:
            return None
        return best_direction

    def _next_motion(
        self,
        enemy_id: str,
        enemy: UnitView,
        tick: int,
        core_distance: int,
        previous: EnemyMotion | None,
    ) -> EnemyMotion:
        pursuit_score = 0
        pursuit_ticks = 0
        active_until = 0
        preemptive_until = 0
        ticks_to_range: int | None = None
        if previous is not None:
            gap = tick - previous.last_tick
            if gap <= PURSUIT_MEMORY_TICKS + 1 and enemy.position != previous.position:
                active_until = tick + ACTIVE_ENEMY_ALERT_TICKS
                closed_distance = previous.core_distance - core_distance
                if closed_distance > 0:
                    pursuit_score = min(
                        PURSUIT_SCORE_THRESHOLD + 1,
                        previous.pursuit_score + 2,
                    )
                    remaining_distance = max(
                        0,
                        core_distance - _attack_range(enemy.unit_type),
                    )
                    ticks_to_range = ceil(
                        remaining_distance * max(1, gap) / closed_distance
                    )
                    if ticks_to_range <= PREEMPTIVE_EVADE_HORIZON:
                        preemptive_until = tick + ACTIVE_ENEMY_ALERT_TICKS
                elif core_distance == previous.core_distance:
                    pursuit_score = min(
                        PURSUIT_SCORE_THRESHOLD + 1,
                        previous.pursuit_score + 1,
                    )
                else:
                    pursuit_score = max(0, previous.pursuit_score - 1)
                pursuit_ticks = previous.pursuit_ticks + int(pursuit_score > 0)
            elif enemy.position == previous.position:
                pursuit_score = 0
        return EnemyMotion(
            object_id=enemy_id,
            position=enemy.position,
            last_tick=tick,
            core_distance=core_distance,
            unit_type=enemy.unit_type,
            pursuit_score=pursuit_score,
            pursuit_ticks=pursuit_ticks,
            active_until_tick=active_until
            or (previous.active_until_tick if previous is not None else 0),
            preemptive_until_tick=preemptive_until
            or (previous.preemptive_until_tick if previous is not None else 0),
            ticks_to_attack_range=ticks_to_range,
        )

    def _observe_attack_events(
        self,
        turn: Turn,
        visible: dict[str, UnitView],
        prior: dict[str, EnemyMotion],
    ) -> None:
        respawned = any(
            _event_name(event, "event_type") == "CORE_RESPAWNED"
            for event in turn.events
        )
        attack_events = tuple(
            event
            for event in turn.events
            if not respawned
            and _event_name(event, "reason_code") == "ATTACK"
            and _event_name(event, "event_type") in {"CORE_DAMAGED", "UNIT_DAMAGED"}
        )
        if not attack_events:
            return
        expires = turn.tick + RECENT_ATTACK_MEMORY_TICKS - 1
        self.recent_attack_until_tick = max(self.recent_attack_until_tick, expires)
        if any(
            _event_name(event, "event_type") == "CORE_DAMAGED"
            for event in attack_events
        ):
            self.recent_core_attack_until_tick = max(
                self.recent_core_attack_until_tick,
                expires,
            )
        for event in attack_events:
            actor_id = getattr(event, "actor_id", None)
            if actor_id is None:
                continue
            enemy_id = str(actor_id)
            enemy = visible.get(enemy_id)
            motion = prior.get(enemy_id) or self.enemy_motion.get(enemy_id)
            if enemy is not None:
                self.enemy_motion[enemy_id] = EnemyMotion(
                    object_id=enemy_id,
                    position=enemy.position,
                    last_tick=turn.tick,
                    core_distance=motion.core_distance if motion else 0,
                    unit_type=enemy.unit_type,
                    pursuit_score=motion.pursuit_score if motion else 0,
                    pursuit_ticks=motion.pursuit_ticks if motion else 0,
                    active_until_tick=expires,
                    preemptive_until_tick=motion.preemptive_until_tick if motion else 0,
                    ticks_to_attack_range=motion.ticks_to_attack_range
                    if motion
                    else None,
                )
            elif motion is not None:
                self.enemy_motion[enemy_id] = EnemyMotion(
                    object_id=motion.object_id,
                    position=motion.position,
                    last_tick=motion.last_tick,
                    core_distance=motion.core_distance,
                    unit_type=motion.unit_type,
                    pursuit_score=motion.pursuit_score,
                    pursuit_ticks=motion.pursuit_ticks,
                    active_until_tick=expires,
                    preemptive_until_tick=motion.preemptive_until_tick,
                    ticks_to_attack_range=motion.ticks_to_attack_range,
                )


def projected_core_damage(
    core_position: Position,
    enemies: Sequence[CombatObject],
    obstacles: set[Position] | frozenset[Position],
) -> int:
    """Count attacks that are legal from the authoritative current cells."""

    return sum(_can_attack_core(enemy, core_position, obstacles) for enemy in enemies)


def _can_attack_core(
    enemy: CombatObject,
    core_position: Position,
    obstacles: set[Position] | frozenset[Position],
) -> bool:
    if isinstance(enemy, CoreView):
        return False
    if enemy.unit_type is UnitType.VANGUARD:
        return manhattan(enemy.position, core_position) == 1
    return enemy.unit_type is UnitType.RANGER and line_of_fire(
        enemy.position,
        core_position,
        obstacles,
    )


def _attack_range(unit_type: UnitType) -> int:
    return 3 if unit_type is UnitType.RANGER else 1


def _target_type_priority(target: CombatObject) -> int:
    if isinstance(target, CoreView):
        return 1
    return {
        UnitType.RANGER: 3,
        UnitType.VANGUARD: 2,
        UnitType.WORKER: 0,
    }[target.unit_type]


def _core_proximity_score(
    target: CombatObject,
    core_position: Position | None,
) -> int:
    if core_position is None:
        return 0
    return max(0, 12 - manhattan(target.position, core_position))


def _event_name(event: object, attribute: str) -> str:
    value = getattr(event, attribute, "")
    return str(getattr(value, "value", value))
