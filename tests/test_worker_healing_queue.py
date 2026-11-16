"""Wounded Workers wait safely for funding without stopping Core traffic."""

from uuid import UUID

import pytest

from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, object_id, unit
from .test_core_traffic import settled_positions
from .test_layered_defense import Encounter
from .test_shared_movement import context_for


def _strategy():
    return AggressiveStrategy(
        WorldMemory(core_home_position=(0, 0), core_layout_version=CORE_LAYOUT_VERSION),
        StrategyConfig(target_workers=16, max_population=None),
    )


@pytest.mark.parametrize("restart", [False, True])
def test_unfunded_patient_holds_then_heals_after_real_delivery(tmp_path, restart):
    strategy = _strategy()
    encounter = Encounter([unit(2, "WORKER", position=(2, 0), hp=1)], [], resources=0)
    for tick in range(100, 113):
        if restart and tick == 102:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
        if tick == 103:
            encounter.friends[object_id(3)] = unit(
                3, "WORKER", position=(3, 0), cargo=1
            )
        turn = encounter.turn()
        report = strategy.decide(turn)
        positions = settled_positions(turn)
        if tick < 103:
            assert positions[UUID(int=2)] == (2, 0)
            assert any(
                d.actor_id == object_id(2) and d.action == "WAIT"
                for d in report.decisions
            )
        encounter.settle(turn.plan, {})
        if encounter.heals:
            break
    assert encounter.deposits == encounter.heals == 1
    assert encounter.friends[object_id(2)]["hp"] == 2
    assert encounter.resources == 0


def test_waiting_patients_allow_a_real_capacity_swap_to_fund_healing():
    strategy = _strategy()
    encounter = Encounter(
        [
            unit(2, "WORKER", position=(1, 0), hp=1),
            unit(3, "WORKER", position=(1, 0), hp=1),
            unit(4, "WORKER", position=(2, 0), cargo=1),
        ],
        [],
        resources=0,
        rocks=[(x, y) for x in range(-1, 4) for y in (-1, 1)] + [(-1, 0)],
    )
    for _ in range(12):
        turn = encounter.turn()
        strategy.decide(turn)
        settled_positions(turn)
        encounter.settle(turn.plan, {})
        if encounter.heals:
            break
    assert encounter.deposits == encounter.heals == 1
    assert encounter.losses == 0


def test_patient_uses_safe_path_distance_for_the_waiting_circle():
    strategy = _strategy()
    turn = make_turn(
        objects=[core(), unit(2, "WORKER", position=(2, 0), hp=1)],
        obstacles=[(1, y) for y in range(-2, 3)],
    )
    strategy.memory.observe(turn)
    context = context_for(turn)
    worker = turn.workers[0]
    assert strategy._core_service_distances(worker, context)[worker.position] > 4
    strategy._decide_worker(worker, context)
    assert turn.plan.unit_actions[worker.id].type == "MOVE"


@pytest.mark.parametrize("hazard", ["visible", "remembered", "incoming"])
def test_unfunded_patient_does_not_wait_on_an_exposed_cell(hazard):
    strategy = _strategy()
    friends = [
        core(),
        unit(2, "WORKER", position=(1, 0), hp=1),
        unit(3, "RANGER", position=(1, 1)),
        unit(4, "RANGER", position=(2, 1)),
    ]
    foe = unit(50, "RANGER", controlled=False, position=(4, 0))
    if hazard == "remembered":
        strategy.memory.observe(make_turn(tick=99, objects=[*friends, foe]))
    elif hazard == "incoming":
        strategy.memory.incoming_damage[(1, 0)] = (99, 1)
    turn = make_turn(objects=[*friends, *([foe] if hazard == "visible" else [])])
    context = context_for(turn)
    strategy._decide_worker(turn.workers[0], context)
    action = turn.plan.unit_actions.get(turn.workers[0].id)
    assert action is not None and action.type == "MOVE"
    positions = settled_positions(turn)
    assert positions[turn.workers[0].id] != (1, 0)


def test_unfunded_patient_continues_approaching_a_moving_core():
    strategy = _strategy()
    turn = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                destination=[1, 0],
                move_progress=1,
                move_required_ticks=4,
            ),
            unit(2, "WORKER", position=(2, 2), hp=1),
        ]
    )
    strategy._decide_worker(turn.workers[0], context_for(turn))
    assert turn.plan.unit_actions[turn.workers[0].id].type == "MOVE"


def test_wounded_loaded_worker_delivers_before_waiting_for_healing():
    strategy = _strategy()
    encounter = Encounter(
        [unit(2, "WORKER", position=(2, 0), hp=1, cargo=1)], [], resources=0
    )
    for _ in range(6):
        turn = encounter.turn()
        strategy.decide(turn)
        settled_positions(turn)
        encounter.settle(turn.plan, {})
        if encounter.heals:
            break
    assert encounter.deposits == encounter.heals == 1
    assert encounter.friends[object_id(2)]["hp"] == 2
