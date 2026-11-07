"""Regression tests for the imported hierarchical combat mechanisms."""

import pytest
from arena_hero import Direction, UnitType, UnitView

from arena_hero_bot.combat_policy import (
    CombatPolicy,
    CombatTargetLedger,
    ThreatLevel,
)
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy

from .factories import core, make_turn, object_id, unit


def _damage_event(target, *, kind="UNIT_DAMAGED"):
    return {
        "event_id": object_id(900),
        "tick": 100,
        "event_type": kind,
        "reason_code": "ATTACK",
        "target_id": object_id(target),
        "values": {"damage": 1},
    }


@pytest.mark.parametrize("kind", ["CORE_DAMAGED", "UNIT_DAMAGED"])
def test_enemy_damage_does_not_trigger_our_defense(kind):
    enemy = (
        core(90, controlled=False, position=(40, 40))
        if kind == "CORE_DAMAGED"
        else unit(90, "RANGER", controlled=False, position=(40, 40))
    )
    turn = make_turn(objects=[core(), enemy], events=[_damage_event(90, kind=kind)])

    assessment = CombatPolicy().assess(turn, set())

    assert assessment.level is ThreatLevel.NORMAL
    assert not assessment.recent_attack
    assert not assessment.recent_core_attack


def test_a_killed_friendly_still_triggers_defense_after_leaving_the_snapshot():
    policy = CombatPolicy()
    policy.assess(make_turn(tick=99, objects=[core(), unit(2, "RANGER")]), set())

    assessment = policy.assess(
        make_turn(objects=[core()], events=[_damage_event(2)]), set()
    )

    assert assessment.level is ThreatLevel.ENGAGED
    assert assessment.recent_attack
    assert not assessment.recent_core_attack


def test_an_enemy_respawn_does_not_hide_damage_to_our_core():
    turn = make_turn(
        objects=[core()],
        events=[
            _damage_event(1, kind="CORE_DAMAGED"),
            {
                "event_id": object_id(901),
                "tick": 100,
                "event_type": "CORE_RESPAWNED",
                "target_id": object_id(90),
            },
        ],
    )

    assert CombatPolicy().assess(turn, set()).recent_core_attack


def test_a_replacement_core_does_not_inherit_the_old_core_attack_alert():
    policy = CombatPolicy()
    policy.assess(
        make_turn(objects=[core()], events=[_damage_event(1, kind="CORE_DAMAGED")]),
        set(),
    )

    assessment = policy.assess(make_turn(tick=101, objects=[core(9)]), set())

    assert assessment.level is ThreatLevel.NORMAL
    assert not assessment.recent_core_attack


def test_damage_ledger_switches_after_lethal_same_tick_coverage() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 0)),
            unit(3, "RANGER", controlled=False, position=(0, 3), hp=1),
            unit(4, "VANGUARD", controlled=False, position=(3, 0)),
        ]
    )
    enemies = tuple(
        enemy
        for enemy in turn.visible_enemies
        if isinstance(enemy, UnitView)
        and enemy.unit_type in {UnitType.RANGER, UnitType.VANGUARD}
    )
    ledger = CombatTargetLedger()

    first = ledger.select(enemies, enemies[0], (0, 0), (0, 0))
    assert first is enemies[0]
    ledger.record(first)

    second = ledger.select(enemies, enemies[0], (0, 0), (0, 0))
    assert second is enemies[1]


def test_closing_enemy_triggers_preemptive_evasion_before_attack_range() -> None:
    policy = CombatPolicy()
    first = make_turn(
        tick=100,
        objects=[
            core(position=(0, 0)),
            unit(2, "RANGER", controlled=False, position=(0, 20)),
        ],
    )
    assert policy.assess(first, set()).level is ThreatLevel.NORMAL

    second = make_turn(
        tick=101,
        objects=[
            core(position=(0, 0)),
            unit(2, "RANGER", controlled=False, position=(0, 19)),
        ],
    )
    assessment = policy.assess(second, set())

    assert assessment.level is ThreatLevel.PRE_EVADE
    assert assessment.minimum_ticks_to_range == 16
    assert assessment.preemptive_enemy_ids


def test_current_ranger_attack_is_engaged_and_projects_core_damage() -> None:
    policy = CombatPolicy()
    turn = make_turn(
        objects=[
            core(position=(0, 0)),
            unit(2, "RANGER", controlled=False, position=(0, 3)),
        ]
    )

    assessment = policy.assess(turn, set())

    assert assessment.level is ThreatLevel.ENGAGED
    assert assessment.projected_core_damage == 1
    assert assessment.should_evacuate_core


def test_breakout_prefers_an_exit_outside_two_attack_axes() -> None:
    policy = CombatPolicy()
    enemies_turn = make_turn(
        objects=[
            core(position=(0, 0)),
            unit(2, "VANGUARD", controlled=False, position=(0, 1)),
            unit(3, "VANGUARD", controlled=False, position=(1, 0)),
        ]
    )
    enemies = tuple(enemies_turn.visible_enemies)

    direction = policy.escape_direction(
        (0, 0),
        enemies,
        set(),
        set(),
    )

    assert direction in {Direction.UP, Direction.LEFT}


def test_strategy_holds_core_on_predicted_contact() -> None:
    strategy = AggressiveStrategy(WorldMemory())
    first = make_turn(
        tick=100,
        objects=[
            core(position=(0, 0)),
            unit(2, "WORKER", position=(4, 4)),
            unit(3, "RANGER", controlled=False, position=(0, 20)),
        ],
    )
    strategy.decide(first)

    second = make_turn(
        tick=101,
        objects=[
            core(position=(0, 0)),
            unit(2, "WORKER", position=(4, 4)),
            unit(3, "RANGER", controlled=False, position=(0, 19)),
        ],
    )
    report = strategy.decide(second)

    assert report.threat_level == ThreatLevel.PRE_EVADE.value
    core_action = second.plan.core_action
    assert core_action is None or core_action.type != "START_MOVE"
