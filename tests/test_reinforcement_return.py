"""Returning defenders may retrace their approach without circling into contact."""

from uuid import UUID

import pytest

from arena_hero_bot.geometry import add, manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig, _TurnContext

from .factories import object_id, unit
from .test_layered_defense import Encounter


def returning_encounter(*, rocks=()):
    encounter = Encounter(
        [unit(2, "RANGER")],
        [unit(50, "RANGER", controlled=False, position=(4, -1))],
        rocks=rocks,
    )
    encounter.core["position"] = [0, -32]
    memory = WorldMemory(
        unit_roles={object_id(2): "patrol-1"},
        unit_roles_initialized=True,
        core_home_position=(0, -32),
        position_history={object_id(2): [(-2, 0), (-1, 0), (0, 0), (0, -1), (0, 0)]},
    )
    return AggressiveStrategy(memory, StrategyConfig(campaign_mode=True)), encounter


def return_context(strategy, turn, goal=(0, -27)):
    strategy.memory.observe(turn)
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={unit.position for unit in turn.units},
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
        reinforcement_assignments={UUID(int=2): goal},
    )


@pytest.mark.parametrize("restart", [False, True])
def test_recalled_ranger_retraces_safe_route_before_a_closing_enemy(tmp_path, restart):
    strategy, encounter = returning_encounter()
    if restart:
        path = tmp_path / "memory.json"
        strategy.memory.save(path)
        strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)

    first = encounter.turn()
    context = return_context(strategy, first)
    assert strategy._move_to_reinforcement_post(first.rangers[0], context)
    action = first.plan.unit_actions[UUID(int=2)]
    assert action.type == "MOVE"
    assert add(first.rangers[0].position, action.direction) == (0, -1)
    encounter.settle(first.plan, {object_id(50): {"type": "MOVE", "direction": "DOWN"}})

    # The enemy now covers the sideways detour that the old recent-position
    # penalty preferred. Continuing home stays outside that firing line.
    second = encounter.turn()
    context = return_context(strategy, second)
    assert strategy._move_to_reinforcement_post(second.rangers[0], context)
    encounter.settle(
        second.plan,
        {
            object_id(50): {
                "type": "SHOOT",
                "expected_cell": [1, 0],
            }
        },
    )
    survivor = encounter.friends[object_id(2)]
    assert survivor["hp"] == 2
    assert manhattan(tuple(survivor["position"]), (0, -32)) == 30
    assert encounter.losses == 0


@pytest.mark.parametrize("blocker", ["rock", "fire", "enemy", "capacity", "hit"])
def test_recalled_ranger_still_checks_real_hazards_on_the_old_route(blocker):
    strategy, encounter = returning_encounter(
        rocks=[(0, -1)] if blocker == "rock" else ()
    )
    if blocker in {"fire", "enemy"}:
        foe = unit(
            51,
            "RANGER" if blocker == "fire" else "WORKER",
            controlled=False,
            position=(2, -1) if blocker == "fire" else (0, -1),
        )
        encounter.foes[foe["id"]] = foe
    if blocker == "capacity":
        for number in (3, 4):
            friend = unit(number, "WORKER", position=(0, -1))
            encounter.friends[friend["id"]] = friend
    if blocker == "hit":
        strategy.memory.incoming_damage[(0, -1)] = (99, 2)

    turn = encounter.turn()
    context = return_context(strategy, turn)
    strategy._move_to_reinforcement_post(turn.rangers[0], context)
    action = turn.plan.unit_actions[UUID(int=2)]
    if action.type == "MOVE":
        destination = add(turn.rangers[0].position, action.direction)
        assert destination != (0, -1)
        assert not strategy._ranger_attackers_at(
            destination, turn, strategy._rock_map(turn)
        )


def test_returning_ranger_does_not_erase_the_history_used_by_other_movement():
    strategy, encounter = returning_encounter()
    turn = encounter.turn()
    context = return_context(strategy, turn)
    recent = strategy.memory.recent_positions(object_id(2))
    assert strategy._move_to_reinforcement_post(turn.rangers[0], context)
    assert strategy.memory.recent_positions(object_id(2)) == recent


def test_a_returned_ranger_holds_its_stable_post():
    strategy, encounter = returning_encounter()
    turn = encounter.turn()
    context = return_context(strategy, turn, goal=(0, 0))
    assert strategy._move_to_reinforcement_post(turn.rangers[0], context)
    assert turn.plan.unit_actions[UUID(int=2)].type != "MOVE"
