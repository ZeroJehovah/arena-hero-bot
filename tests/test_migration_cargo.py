"""Cargo keeps making safe progress while the Core's service queue moves."""

import pytest

from arena_hero_bot.geometry import add, manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, unit
from .test_shared_movement import context_for


def moving_core():
    return core(
        state="MOVING",
        move_direction="RIGHT",
        move_progress=2,
        move_required_ticks=4,
        destination=[1, 0],
    )


def migration_context(turn):
    context = context_for(turn)
    context.core_migration_destination = (1, 0)
    context.core_migration_lane = ((1, 0), (2, 0), (3, 0), (4, 0))
    return context


def test_arrival_worker_waits_at_a_full_entrance_then_delivers_after_restart(tmp_path):
    strategy = AggressiveStrategy(WorldMemory())
    position = (2, 2)
    destinations = []
    for tick in range(100, 106):
        guards = [unit(10, "RANGER", position=(2, 1))]
        if tick < 102:
            guards.append(unit(11, "RANGER", position=(2, 1)))
        turn = make_turn(
            tick=tick,
            objects=[
                moving_core() if tick < 105 else core(position=(1, 0)),
                unit(2, "WORKER", position=position, cargo=1),
                *guards,
            ],
            obstacles=[(1, 2), (3, 1), (4, 2)],
        )
        strategy.memory.observe(turn)
        context = migration_context(turn)
        worker = turn.workers[0]
        context.core_arrival_worker_id = worker.id
        strategy._decide_worker(worker, context)
        action = turn.plan.unit_actions.get(worker.id)
        if tick < 102:
            assert action is None or action.type == "WAIT"
        elif tick < 105:
            assert action is not None and action.type == "MOVE"
            position = add(position, action.direction)
            destinations.append(position)
        else:
            assert position == (1, 0)
            assert action is not None and action.type == "DEPOSIT"
        if tick == 101:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path))
    assert destinations[0] == (2, 1)
    assert destinations[-1] == (1, 0)
    assert len(set(destinations)) == 3


def test_migration_cargo_takes_a_real_rock_detour_without_reversing():
    strategy = AggressiveStrategy(WorldMemory())
    position = (1, 2)
    visited = {position}
    for tick in range(100, 112):
        turn = make_turn(
            tick=tick,
            objects=[
                moving_core(),
                unit(3, "WORKER", position=position, cargo=1),
            ],
            obstacles=[(0, 2), (1, 1), (2, 1)],
        )
        strategy.memory.observe(turn)
        context = migration_context(turn)
        context.core_arrival_worker_id = turn.workers[0].id
        worker = turn.workers[0]
        strategy._decide_worker(worker, context)
        action = turn.plan.unit_actions.get(worker.id)
        assert action is not None and action.type == "MOVE"
        destination = add(position, action.direction)
        if tick == 100:
            assert destination == (2, 2)
            assert manhattan(destination, (1, 0)) > manhattan(position, (1, 0))
        assert destination not in visited
        visited.add(destination)
        position = destination
        if position == (1, 0):
            break
    assert position == (1, 0)


@pytest.mark.parametrize("moving", [False, True])
def test_arrival_cargo_can_transit_a_core_side_cell_without_waiting_there(moving):
    turn = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="UP",
                move_progress=2,
                move_required_ticks=4,
                destination=[0, -1],
            )
            if moving
            else core(),
            unit(2, "WORKER", position=(1, 1), cargo=1),
        ],
        obstacles=[(0, 1), (2, 1), (1, 2)],
    )
    strategy = AggressiveStrategy(
        WorldMemory(
            core_home_position=(0, -8), core_layout_version=CORE_LAYOUT_VERSION
        ),
        StrategyConfig(target_workers=16, expedition_mode=True),
    )
    strategy.memory.observe(turn)
    context = migration_context(turn)
    context.core_migration_destination = (0, -1)
    context.core_migration_lane = ((0, -1), (0, -2), (0, -3), (0, -4))
    worker = turn.workers[0]
    context.core_arrival_worker_id = worker.id
    # Exercise arrival routing on both a service Tick and a MOVING Tick.
    strategy._decide_worker(worker, context)
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert add(worker.position, action.direction) == (1, 0)
    assert context.report.decisions[-1].target == (0, -1)


def test_migration_arrival_selection_uses_walking_distance_around_rocks():
    turn = make_turn(
        objects=[
            moving_core(),
            unit(2, "WORKER", position=(1, 2), cargo=1),
            unit(3, "WORKER", position=(4, 0), cargo=1),
        ],
        obstacles=[(0, 1), (1, 1), (2, 1)],
    )
    strategy = AggressiveStrategy(WorldMemory())
    strategy.memory.observe(turn)
    context = migration_context(turn)
    assert strategy._select_core_arrival_worker(context) == turn.workers[1].id


@pytest.mark.parametrize(("near_id", "far_id"), [(20, 30), (30, 20)])
def test_migration_reserves_the_last_queue_slot_for_nearby_cargo(near_id, far_id):
    turn = make_turn(
        objects=[
            moving_core(),
            *(unit(2 + x, "WORKER", position=(x, 0), cargo=1) for x in (1, 2, 3)),
            unit(near_id, "WORKER", position=(4, 2), cargo=1),
            unit(far_id, "WORKER", position=(-10, 10), cargo=1),
        ],
    )
    strategy = AggressiveStrategy(WorldMemory())
    strategy.memory.observe(turn)
    context = migration_context(turn)
    context.core_arrival_worker_id = turn.workers[0].id
    for worker in sorted(
        turn.workers, key=lambda worker: strategy._worker_priority(worker, context)
    ):
        strategy._decide_worker(worker, context)
    near = next(worker for worker in turn.workers if worker.id.int == near_id)
    decision = next(d for d in context.report.decisions if d.actor_id == str(near.id))
    assert decision.action == "MOVE"
    assert decision.target == (4, 0)
    assert decision.reason == "queue carried resources on the Core forward route"


@pytest.mark.parametrize("position", [(12, 8), (160, 0)])
def test_distant_cargo_keeps_approaching_a_full_migration_queue(position):
    turn = make_turn(
        objects=[
            moving_core(),
            *(unit(2 + x, "WORKER", position=(x, 0), cargo=1) for x in range(1, 5)),
            unit(20, "WORKER", position=position, cargo=1),
        ],
    )
    strategy = AggressiveStrategy(WorldMemory())
    strategy.memory.observe(turn)
    context = migration_context(turn)
    worker = turn.workers[-1]
    strategy._decide_worker(worker, context)
    action = turn.plan.unit_actions.get(worker.id)
    assert action is not None and action.type == "MOVE"
    destination = add(position, action.direction)
    assert manhattan(destination, (4, 0)) < manhattan(position, (4, 0))
    assert destination not in context.core_migration_lane


def test_migration_waiters_share_an_off_route_cell_when_no_progress_is_safe():
    turn = make_turn(
        objects=[
            moving_core(),
            *(unit(2 + x, "WORKER", position=(x, 0), cargo=1) for x in range(1, 5)),
            unit(20, "WORKER", position=(4, 1), cargo=1),
            unit(21, "WORKER", position=(4, 1), cargo=1),
        ],
        obstacles=[(3, 1), (5, 1)],
    )
    strategy = AggressiveStrategy(WorldMemory())
    strategy.memory.observe(turn)
    context = migration_context(turn)
    for worker in turn.workers[-2:]:
        strategy._decide_worker(worker, context)
        action = turn.plan.unit_actions.get(worker.id)
        assert action is None or action.type == "WAIT"
