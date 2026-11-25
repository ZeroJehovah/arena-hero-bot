"""Arrival and simultaneous retaliation belong to the same local trade."""

from uuid import UUID

import pytest

from arena_hero_bot.geometry import add, line_of_fire
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy

from .factories import core, object_id, unit
from .test_encounter_combat import _context, _decide, _strategy
from .test_layered_defense import Encounter


def _encounter(size, *, enemy_hp=2, core_position=(0, 10)):
    origins = ((3, 4), (0, 4), (-3, 4), (4, 0))
    goals = ((3, 3), (0, 3), (-3, 3), (3, 0))
    friends = [unit(i + 2, "RANGER", position=origins[i]) for i in range(size)]
    foes = [
        unit(i, "RANGER", controlled=False, position=(0, 0), hp=enemy_hp)
        for i in (30, 31)
    ]
    encounter = Encounter(friends, foes, resources=0)
    encounter.core = core(position=core_position)
    strategy = _strategy({friend["id"]: "defense" for friend in friends})
    strategy.memory.core_home_position = core_position
    return encounter, strategy, {UUID(int=i + 2): goals[i] for i in range(size)}


@pytest.mark.parametrize("restart", [False, True])
def test_three_entrants_do_not_pay_two_casualties_for_one_kill(tmp_path, restart):
    encounter, strategy, goals = _encounter(3)
    if restart:
        path = tmp_path / "memory.json"
        strategy.memory.save(path)
        strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    turn = encounter.turn()
    context = _context(strategy, turn)
    context.layered_assignments = goals

    strategy._prepare_layered_steps(context, turn.visible_enemies)
    for ranger in turn.rangers:
        _decide(strategy, ranger, context)

    for ranger in turn.rangers:
        action = turn.plan.unit_actions[ranger.id]
        if action.type == "MOVE":
            destination = add(ranger.position, action.direction)
            assert not line_of_fire((0, 0), destination, set())
    encounter.settle(
        turn.plan,
        {
            object_id(n): {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [3, 3],
            }
            for n in (30, 31)
        },
    )
    assert encounter.losses == 0


@pytest.mark.parametrize("enemy_hp,size", [(2, 4), (1, 3)])
def test_viable_shared_entry_still_completes_an_equal_or_better_trade(enemy_hp, size):
    encounter, strategy, goals = _encounter(size, enemy_hp=enemy_hp)
    for step in range(3):
        turn = encounter.turn()
        context = _context(strategy, turn)
        if step == 0:
            context.layered_assignments = goals
            strategy._prepare_layered_steps(context, turn.visible_enemies)
            assert len(turn.plan.unit_actions) == size
            assert all(a.type == "MOVE" for a in turn.plan.unit_actions.values())
        else:
            for ranger in turn.rangers:
                _decide(strategy, ranger, context)
        victim = min(encounter.friends)
        victim_position = encounter.friends[victim]["position"]
        if step == 0:
            victim_position = goals[UUID(victim)]
        encounter.settle(
            turn.plan,
            {
                key: {
                    "type": "SHOOT",
                    "target_id": victim,
                    "expected_cell": victim_position,
                }
                for key in encounter.foes
            },
        )
        if not encounter.foes:
            break
    assert encounter.kills == 2
    assert encounter.losses <= encounter.kills


def test_current_core_fire_still_forces_the_three_ranger_entry():
    encounter, strategy, _goals = _encounter(3, core_position=(0, -3))
    turn = encounter.turn()
    context = _context(strategy, turn)

    assert strategy._force_core_defense(context)

    assert len(turn.plan.unit_actions) == 3
    assert all(action.type == "MOVE" for action in turn.plan.unit_actions.values())
