"""Coordinated entry must not depend on an isolated survivor declining retreat."""

from dataclasses import replace
from uuid import UUID

import pytest
from arena_hero import UnitView

from arena_hero_bot.geometry import add
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy

from .factories import core, object_id, unit
from .test_layered_defense import Encounter, _context, _strategy


def _pair_entry(*, core_position=(0, 0), opponents=2, hp=2):
    friends = [
        unit(2, "RANGER", position=(4, -4), hp=hp),
        unit(3, "RANGER", position=(2, -1), hp=hp),
    ]
    foes = [
        unit(50, "RANGER", controlled=False, position=(0, -4), hp=1),
        *(
            [unit(51, "RANGER", controlled=False, position=(0, -4))]
            if opponents == 2
            else []
        ),
    ]
    encounter = Encounter(friends, foes, resources=0)
    encounter.core = core(position=core_position)
    strategy = _strategy({friend["id"]: "defense" for friend in friends})
    return strategy, encounter


@pytest.mark.parametrize("restart", [False, True])
def test_pair_does_not_enter_when_concentrated_fire_leaves_a_losing_solo_shooter(
    tmp_path, restart
):
    strategy, encounter = _pair_entry()
    for tick in range(100, 103):
        if restart:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
        turn = encounter.turn()
        context = _context(strategy, turn)
        context.layered_assignments = {UUID(int=2): (3, -4), UUID(int=3): (2, -2)}
        strategy._prepare_layered_steps(
            context,
            tuple(e for e in turn.visible_enemies if isinstance(e, UnitView)),
        )
        for ranger in turn.rangers:
            strategy._move_layered_ranger(ranger, context)
        assert not any(a.type == "MOVE" for a in turn.plan.unit_actions.values())
        assert not context.arrival_counts and not context.departure_counts
        encounter.settle(
            turn.plan,
            {
                object_id(n): {
                    "type": "SHOOT",
                    "target_id": object_id(2),
                    "expected_cell": [3, -4],
                }
                for n in (50, 51)
            },
        )
        assert encounter.tick == tick + 1
        assert encounter.losses == 0


@pytest.mark.parametrize("pressure", ["current", "recent", "projected"])
def test_core_pressure_keeps_the_emergency_shared_entry(pressure):
    strategy, encounter = _pair_entry(
        core_position=(0, -1) if pressure == "current" else (0, 0)
    )
    turn = encounter.turn()
    context = _context(strategy, turn)
    if pressure == "recent":
        context.threat = replace(context.threat, recent_core_attack=True)
    elif pressure == "projected":
        context.threat = replace(context.threat, projected_core_damage=1)
    context.layered_assignments = {UUID(int=2): (3, -4), UUID(int=3): (2, -2)}
    strategy._prepare_layered_steps(
        context,
        tuple(e for e in turn.visible_enemies if isinstance(e, UnitView)),
    )
    assert len(turn.plan.unit_actions) == 2
    assert all(a.type == "MOVE" for a in turn.plan.unit_actions.values())


def test_one_hostile_arrow_does_not_become_two_entry_losses(monkeypatch):
    strategy, encounter = _pair_entry(opponents=1)
    # Exercise entry eligibility independently of the optional preference
    # for covering a neighboring empty cell from the current position.
    monkeypatch.setattr(strategy, "_defensive_cover_cells", lambda *_args: ((), ()))
    turn = encounter.turn()
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (3, -4), UUID(int=3): (2, -2)}
    strategy._prepare_layered_steps(
        context,
        tuple(e for e in turn.visible_enemies if isinstance(e, UnitView)),
    )
    for ranger in turn.rangers:
        strategy._move_layered_ranger(ranger, context)
    assert all(a.type == "MOVE" for a in turn.plan.unit_actions.values())
    assert len(turn.plan.unit_actions) == 2
    for ranger in turn.rangers:
        action = turn.plan.unit_actions[ranger.id]
        assert (
            add(ranger.position, action.direction)
            == context.layered_assignments[ranger.id]
        )
    encounter.settle(
        turn.plan,
        {
            object_id(50): {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [3, -4],
            }
        },
    )
    assert encounter.losses == 0
    assert encounter.friends[object_id(2)]["hp"] == 1


def test_a_lone_survivor_can_still_complete_a_mutual_single_enemy_trade():
    strategy, encounter = _pair_entry(opponents=1, hp=1)
    turn = encounter.turn()
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (3, -4), UUID(int=3): (2, -2)}
    strategy._prepare_layered_steps(
        context,
        tuple(e for e in turn.visible_enemies if isinstance(e, UnitView)),
    )
    assert len(turn.plan.unit_actions) == 2
    assert all(a.type == "MOVE" for a in turn.plan.unit_actions.values())
    encounter.settle(
        turn.plan,
        {
            object_id(50): {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [3, -4],
            }
        },
    )
    reply = encounter.turn()
    context = _context(strategy, reply)
    strategy._prepare_defensive_trade_shots(context)
    survivor = reply.rangers[0]
    assert context.defensive_trade_shots[survivor.id].id == UUID(int=50)
