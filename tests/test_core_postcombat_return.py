"""Finished defensive retreats do not force a detour back to the same home."""

import pytest
from arena_hero import Direction

from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, object_id, unit
from .test_layered_defense import Encounter
from .test_shared_movement import context_for


def _returning_core():
    memory = WorldMemory(
        core_home_position=(15, 16),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=4,
        campaign_last_core_id=object_id(1),
        unit_roles_initialized=True,
        position_history={object_id(1): [(15, 16), (16, 16), (16, 15)]},
        layered_defense_core_id=object_id(1),
        layered_defense_anchor=(16, 15),
        layered_retreat_destination=(16, 15),
        layered_defense_last_contact_tick=91,
    )
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            campaign_mode=True,
            expedition_mode=True,
            target_workers=16,
            max_population=None,
        ),
    )
    encounter = Encounter(
        [unit(2, "WORKER", position=(20, 20))],
        [],
        resources=0,
        rocks=[(15, 15), (16, 13), (16, 14), (16, 17), (18, 15)],
    )
    encounter.core = core(position=(16, 15))
    return strategy, encounter


@pytest.mark.parametrize("restart", [False, True])
def test_quiet_retreat_returns_in_two_real_steps_without_changing_home(
    tmp_path, restart
):
    strategy, encounter = _returning_core()
    moves = []
    for tick in range(100, 108):
        if restart and tick in (100, 104):
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
        turn = encounter.turn()
        strategy.decide(turn)
        action = turn.plan.core_action
        if action is not None and action.type == "START_MOVE":
            moves.append(action.direction)
        if tick == 100:
            assert action is not None and action.type == "START_MOVE"
            assert action.direction is Direction.DOWN
        assert strategy.memory.core_home_position == (15, 16)
        encounter.settle(turn.plan, {})
    assert moves == [Direction.DOWN, Direction.LEFT]
    assert encounter.core["position"] == [15, 16]
    assert encounter.core["hp"] == encounter.core["shield"] == 5


@pytest.mark.parametrize("contact", ["recent", "visible"])
def test_combat_still_preserves_retreat_history_and_holds_the_old_home(contact):
    strategy, encounter = _returning_core()
    strategy.memory.layered_defense_last_contact_tick = 92
    if contact == "visible":
        encounter.foes[object_id(50)] = unit(
            50, "RANGER", position=(22, 18), controlled=False
        )
    before = list(strategy.memory.position_history[object_id(1)])
    turn = encounter.turn()
    strategy.decide(turn)
    assert strategy.memory.position_history[object_id(1)] == before
    assert strategy.memory.layered_retreat_destination == (16, 15)
    assert turn.plan.core_action is None or turn.plan.core_action.type == "WAIT"


@pytest.mark.parametrize("retreat", [None, (16, 16)])
def test_history_is_not_cleared_without_an_arrived_defensive_retreat(retreat):
    strategy, encounter = _returning_core()
    strategy.memory.layered_retreat_destination = retreat
    before = list(strategy.memory.position_history[object_id(1)])
    turn = encounter.turn()
    strategy._prepare_layered_defense(context_for(turn))
    assert strategy.memory.position_history[object_id(1)] == before


def test_active_core_move_keeps_its_history_even_after_contacts_expire():
    strategy, encounter = _returning_core()
    encounter.progress = 2
    encounter.direction = Direction.DOWN
    before = list(strategy.memory.position_history[object_id(1)])
    strategy._prepare_layered_defense(context_for(encounter.turn()))
    assert strategy.memory.position_history[object_id(1)] == before


def test_return_still_obeys_rocks_and_only_rebases_the_core_history():
    strategy, _ = _returning_core()
    strategy.memory.position_history[object_id(2)] = [(19, 20), (20, 20)]
    turn = make_turn(
        objects=[core(position=(16, 15)), unit(2, "WORKER", position=(20, 20))],
        obstacles=[(15, 15), (16, 14), (16, 16)],
    )
    strategy.memory.observe(turn)
    context = context_for(turn)
    strategy._prepare_layered_defense(context)
    assert strategy._core_home_direction(context) is Direction.RIGHT
    assert strategy.memory.position_history[object_id(2)] == [(19, 20), (20, 20)]
    assert strategy.memory.core_home_position == (15, 16)
