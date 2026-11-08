"""Cargo keeps making safe progress while the Core's service queue moves."""

from collections import Counter
from uuid import UUID

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


@pytest.mark.parametrize("moving", [False, True])
def test_queueing_cargo_routes_through_the_core_cell_in_full_planning(moving):
    turn = make_turn(
        objects=[
            moving_core() if moving else core(),
            unit(2, "WORKER", position=(1, 0), cargo=1),
            unit(3, "WORKER", position=(-2, 2), cargo=1),
        ],
        obstacles=[(-1, 1), (-1, 2), (-1, 3), (-3, 2)],
    )
    strategy = AggressiveStrategy(
        WorldMemory(
            core_home_position=(16, 0), core_layout_version=CORE_LAYOUT_VERSION
        ),
        StrategyConfig(target_workers=16, max_population=None, expedition_mode=True),
    )

    # Full planning installs standing blockers that a bare context omits.
    # The queue endpoint is vacant; its shortest route crosses the moving
    # Core's current cell, which must not block an earlier, empty step.
    report = strategy.decide(turn)
    action = turn.plan.unit_actions.get(UUID(int=3))
    assert action is not None and action.type == "MOVE"
    assert add((-2, 2), action.direction) == (-2, 1)
    decision = next(d for d in report.decisions if d.actor_id == str(UUID(int=3)))
    assert decision.target == (2, 0)


def test_queued_cargo_catches_moving_core_and_delivers_across_restart(tmp_path):
    memory = WorldMemory(
        core_home_position=(16, 0), core_layout_version=CORE_LAYOUT_VERSION
    )
    config = StrategyConfig(
        target_workers=16, max_population=None, expedition_mode=True
    )
    strategy = AggressiveStrategy(memory, config)
    core_position = (0, 0)
    destination = (1, 0)
    progress = 1
    positions = {2: (1, 0), 3: (-2, 2)}
    cargo = {2: 1, 3: 1}
    delivered = []
    for tick in range(100, 124):
        turn = make_turn(
            tick=tick,
            resources=len(delivered),
            objects=[
                core(
                    position=core_position,
                    state="MOVING" if progress else "NORMAL",
                    move_direction="RIGHT" if progress else None,
                    move_progress=progress if progress else None,
                    move_required_ticks=4 if progress else None,
                    destination=list(destination) if progress else None,
                ),
                *(
                    unit(i, "WORKER", position=positions[i], cargo=cargo[i])
                    for i in positions
                ),
            ],
            obstacles=[(-1, 1), (-1, 2), (-1, 3), (-3, 2)],
        )
        strategy.decide(turn)
        for ident in positions:
            action = turn.plan.unit_actions.get(UUID(int=ident))
            if tick == 100 and ident == 3:
                assert action is not None and action.type == "MOVE"
            if action is None:
                continue
            if action.type == "MOVE":
                positions[ident] = add(positions[ident], action.direction)
            elif action.type == "DEPOSIT":
                assert progress == 0 and positions[ident] == core_position
                assert turn.plan.core_action is None
                cargo[ident] = 0
                delivered.append(ident)
        if progress:
            progress += 1
            if progress == 4:
                core_position = destination
                progress = 0
        elif turn.plan.core_action and turn.plan.core_action.type == "START_MOVE":
            destination = add(core_position, turn.plan.core_action.direction)
            progress = 1
        assert max(Counter([core_position, *positions.values()]).values()) <= 2
        if tick == 102:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), config)
        if len(delivered) == 2:
            break
    assert delivered == [2, 3]


def test_cargo_on_moving_core_takes_arrival_slot_from_waiter_behind_it():
    turn = make_turn(
        objects=[
            moving_core(),
            unit(2, "WORKER", position=(-1, 0), cargo=1),
            unit(3, "WORKER", position=(0, 0), cargo=1),
        ],
        obstacles=[(-1, -1), (0, -1)],
    )
    memory = WorldMemory(
        core_home_position=(16, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_arrival_worker_id=str(UUID(int=2)),
        core_arrival_destination=(1, 0),
    )
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(target_workers=16, max_population=None, expedition_mode=True),
    )

    strategy.decide(turn)

    action = turn.plan.unit_actions.get(UUID(int=3))
    assert action is not None and action.type == "MOVE"
    assert add((0, 0), action.direction) == (1, 0)
    assert memory.core_arrival_worker_id == str(UUID(int=3))
    assert all(a.type != "DEPOSIT" for a in turn.plan.unit_actions.values())


@pytest.mark.parametrize("reservation", ["cargo", "healing"])
def test_blocked_cargo_on_moving_core_waits_without_booking_a_delivery(reservation):
    turn = make_turn(
        resources=1,
        objects=[
            moving_core(),
            unit(2, "WORKER", cargo=1),
            unit(3, "WORKER", position=(1, 0), cargo=1)
            if reservation == "cargo"
            else unit(3, "RANGER", position=(1, 0), hp=1),
        ],
    )
    strategy = AggressiveStrategy(WorldMemory())
    strategy.memory.observe(turn)
    context = migration_context(turn)
    if reservation == "cargo":
        context.core_arrival_worker_id = UUID(int=3)
    else:
        context.core_arrival_healer_id = UUID(int=3)
    worker = turn.workers[0]
    initial_resources = context.remaining_resources
    initial_space = context.remaining_resource_space

    strategy._decide_worker(worker, context)

    action = turn.plan.unit_actions.get(worker.id)
    assert action is None or action.type == "WAIT"
    assert context.report.decisions[-1].action == "WAIT"
    assert context.core_service_action is None
    assert context.remaining_resources == initial_resources
    assert context.remaining_resource_space == initial_space


def test_moving_core_overlap_clears_and_delivers_after_restart(tmp_path):
    config = StrategyConfig(
        target_workers=16, max_population=None, expedition_mode=True
    )
    strategy = AggressiveStrategy(
        WorldMemory(
            core_home_position=(16, 0),
            core_layout_version=CORE_LAYOUT_VERSION,
            core_arrival_worker_id=str(UUID(int=2)),
            core_arrival_destination=(1, 0),
        ),
        config,
    )
    core_position = (0, 0)
    destination = (1, 0)
    progress = 1
    positions = {2: (-1, 0), 3: (0, 0)}
    cargo = {2: 1, 3: 1}
    delivered = []
    for tick in range(100, 120):
        turn = make_turn(
            tick=tick,
            resources=len(delivered),
            objects=[
                core(
                    position=core_position,
                    state="MOVING" if progress else "NORMAL",
                    move_direction="RIGHT" if progress else None,
                    move_progress=progress if progress else None,
                    move_required_ticks=4 if progress else None,
                    destination=list(destination) if progress else None,
                ),
                *(
                    unit(i, "WORKER", position=positions[i], cargo=cargo[i])
                    for i in positions
                ),
            ],
        )
        strategy.decide(turn)
        for ident in positions:
            action = turn.plan.unit_actions.get(UUID(int=ident))
            if action is None:
                continue
            if action.type == "MOVE":
                positions[ident] = add(positions[ident], action.direction)
            elif action.type == "DEPOSIT":
                assert progress == 0 and positions[ident] == core_position
                assert turn.plan.core_action is None
                cargo[ident] = 0
                delivered.append(ident)
        if tick == 100:
            assert positions[3] == (1, 0)
        if progress:
            progress += 1
            if progress == 4:
                core_position = destination
                progress = 0
        elif turn.plan.core_action and turn.plan.core_action.type == "START_MOVE":
            destination = add(core_position, turn.plan.core_action.direction)
            progress = 1
        assert max(Counter([core_position, *positions.values()]).values()) <= 2
        if tick == 100:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), config)
        if len(delivered) == 2:
            break
    assert delivered == [3, 2]
