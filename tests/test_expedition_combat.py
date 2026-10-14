"""Local attack opportunities must survive squad pursuit targeting."""

from uuid import UUID

import pytest
from arena_hero import Direction

from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import (
    AggressiveStrategy,
    StrategyConfig,
    _ExpeditionPursuit,
    _TurnContext,
)

from .factories import core, make_turn, object_id, unit


def combat_context(turn):
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={item.position for item in turn.units},
        enemy_positions={item.position for item in turn.visible_enemies},
    )


def expedition_strategy(*members):
    strategy = AggressiveStrategy(
        WorldMemory(),
        StrategyConfig(target_workers=16, expedition_mode=True, max_population=None),
    )
    squad = frozenset(UUID(int=member) for member in members)
    strategy._expedition_squads = [(squad, (1, 0))]
    return strategy, squad


@pytest.mark.parametrize("pursued_position", [(9, 6), (11, 6)])
def test_ranger_fires_locally_when_pursued_enemy_cannot_be_shot(pursued_position):
    # The squad target is either off the firing ray or outside local vision.
    strategy, squad = expedition_strategy(2, 3)
    strategy._expedition_pursuits[squad] = _ExpeditionPursuit(
        target_id=object_id(90), position=pursued_position
    )
    turn = make_turn(
        objects=[
            core(),
            unit(2, "VANGUARD", position=(4, 5)),
            unit(3, "RANGER", position=(5, 5), hp=1),
            unit(90, "RANGER", controlled=False, position=pursued_position),
            unit(91, "VANGUARD", controlled=False, position=(7, 5)),
        ]
    )
    context = combat_context(turn)

    strategy._decide_ranger(turn.rangers[0], context, (), offensive=True)

    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type == "SHOOT"
    assert action.target_id == UUID(int=91)
    assert action.expected_cell == (7, 5)
    assert strategy._expedition_pursuits[squad].target_id == object_id(90)


@pytest.mark.parametrize("target_hp", [1, 2])
def test_rangers_keep_focus_until_damage_is_allocated_then_use_another_target(
    target_hp,
):
    strategy, squad = expedition_strategy(2, 3)
    strategy._expedition_pursuits[squad] = _ExpeditionPursuit(
        target_id=object_id(90), position=(7, 5)
    )
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(5, 5)),
            unit(3, "RANGER", position=(5, 5)),
            unit(90, "RANGER", controlled=False, position=(7, 5), hp=target_hp),
            unit(91, "RANGER", controlled=False, position=(5, 8)),
        ]
    )
    context = combat_context(turn)

    for ranger in turn.rangers:
        strategy._decide_ranger(ranger, context, (), offensive=True)

    target_ids = []
    for ranger in turn.rangers:
        action = turn.plan.unit_actions[ranger.id]
        assert action.type == "SHOOT"
        target_ids.append(action.target_id)
    assert target_ids == [UUID(int=90), UUID(int=91 if target_hp == 1 else 90)]


@pytest.mark.parametrize("blocker", ["rock", "friendly"])
def test_local_shot_fallback_still_checks_rocks_and_friendly_firing_lines(blocker):
    strategy, squad = expedition_strategy(2, 3)
    strategy._expedition_pursuits[squad] = _ExpeditionPursuit(
        target_id=object_id(90), position=(9, 6)
    )
    turn = make_turn(
        objects=[
            core(),
            unit(2, "VANGUARD", position=(6, 5) if blocker == "friendly" else (4, 5)),
            unit(3, "RANGER", position=(5, 5)),
            unit(90, "RANGER", controlled=False, position=(9, 6)),
            unit(91, "VANGUARD", controlled=False, position=(7, 5)),
        ],
        obstacles=[(6, 5)] if blocker == "rock" else [],
    )
    context = combat_context(turn)

    strategy._decide_ranger(turn.rangers[0], context, (), offensive=True)

    assert turn.plan.unit_actions[turn.rangers[0].id].type == "MOVE"


@pytest.mark.parametrize("pursued_position", [(9, 5), (12, 5)])
def test_vanguard_engages_nearby_enemy_before_chasing_a_distant_ranger(
    pursued_position,
):
    strategy, squad = expedition_strategy(2, 3)
    strategy._expedition_pursuits[squad] = _ExpeditionPursuit(
        target_id=object_id(90), position=pursued_position
    )
    turn = make_turn(
        objects=[
            core(),
            unit(2, "VANGUARD", position=(5, 5), hp=1),
            unit(3, "RANGER", position=(5, 7)),
            unit(90, "RANGER", controlled=False, position=pursued_position),
            unit(91, "VANGUARD", controlled=False, position=(7, 5)),
        ]
    )
    context = combat_context(turn)

    strategy._decide_vanguard(turn.vanguards[0], context, (), offensive=True)

    action = turn.plan.unit_actions[turn.vanguards[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.RIGHT
    assert context.report.decisions[0].target == (6, 5)
    assert strategy._expedition_pursuits[squad].target_id == object_id(90)


def test_vanguard_sweeps_local_enemy_before_starting_pursuit():
    strategy, _squad = expedition_strategy(2, 3)
    turn = make_turn(
        objects=[
            core(),
            unit(2, "VANGUARD", position=(5, 5)),
            unit(3, "RANGER", position=(5, 7)),
            unit(90, "RANGER", controlled=False, position=(8, 5)),
            unit(91, "VANGUARD", controlled=False, position=(6, 5)),
        ]
    )
    context = combat_context(turn)

    strategy._decide_vanguard(turn.vanguards[0], context, (), offensive=True)

    action = turn.plan.unit_actions[turn.vanguards[0].id]
    assert action.type == "SWEEP"
    assert action.direction is Direction.RIGHT
