"""Regression tests for the imported hierarchical combat mechanisms."""

from arena_hero import Direction, UnitType, UnitView

from arena_hero_bot.combat_policy import (
    CombatPolicy,
    CombatTargetLedger,
    ThreatLevel,
)
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy

from .factories import core, make_turn, unit


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


def test_strategy_starts_core_evasion_on_predicted_contact() -> None:
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
    assert second.plan.core_action is not None
    assert second.plan.core_action.type == "START_MOVE"
