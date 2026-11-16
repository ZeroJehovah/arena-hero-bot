"""Workers honor confirmed incoming fire even without a visible attacker."""

from uuid import UUID

import pytest
from arena_hero import Direction

from arena_hero_bot.geometry import add
from arena_hero_bot.memory import INCOMING_DAMAGE_TTL, UnitGoal, WorldMemory
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    RESOURCE_CLAIM_PURPOSE,
    WORKER_PATROL_PURPOSE,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, object_id, unit
from .test_shared_movement import context_for


def strategy_for(memory):
    memory.core_layout_version = CORE_LAYOUT_VERSION
    return AggressiveStrategy(
        memory,
        StrategyConfig(expedition_mode=True, target_workers=16, max_population=None),
    )


@pytest.mark.parametrize("restart", [False, True])
def test_worker_patrol_routes_around_a_previous_tick_fatality(tmp_path, restart):
    origin, fatal_cell, goal = (8, 12), (9, 12), (10, 12)
    memory = WorldMemory(core_home_position=(16, 16))
    memory.observe(
        make_turn(
            tick=99,
            objects=[
                core(position=(16, 16)),
                unit(2, "WORKER", position=origin),
                unit(3, "WORKER", position=fatal_cell),
            ],
        )
    )
    memory.set_goal(object_id(2), UnitGoal(goal, 99, WORKER_PATROL_PURPOSE))
    strategy = strategy_for(memory)
    position = origin
    for tick in range(100, 104):
        if restart and tick == 101:
            path = tmp_path / "memory.json"
            memory.save(path)
            memory = WorldMemory.load(path)
            strategy = strategy_for(memory)
        turn = make_turn(
            tick=tick,
            objects=[core(position=(16, 16)), unit(2, "WORKER", position=position)],
            events=[
                {
                    "event_id": object_id(900),
                    "tick": 99,
                    "event_type": "UNIT_DAMAGED",
                    "reason_code": "ATTACK",
                    "target_id": object_id(3),
                    "position": list(fatal_cell),
                    "values": {"damage": 2, "hp": 0},
                }
            ]
            if tick == 100
            else [],
        )
        strategy.decide(turn)
        assert memory.incoming_damage[fatal_cell] == (99, 2)
        action = turn.plan.unit_actions[UUID(object_id(2))]
        assert action.type == "MOVE"
        position = add(position, action.direction)
        assert position != fatal_cell
        assert all(3 <= coordinate <= 28 for coordinate in position)
    assert position == goal


def test_worker_waits_when_fatal_fire_closes_the_only_patrol_passage():
    origin, fatal_cell, goal = (8, 12), (9, 12), (10, 12)
    rocks = {(9, y) for y in range(32)} - {fatal_cell}
    memory = WorldMemory(
        core_home_position=(16, 16),
        obstacles=rocks,
        incoming_damage={fatal_cell: (99, 2)},
    )
    memory.set_goal(object_id(2), UnitGoal(goal, 99, WORKER_PATROL_PURPOSE))
    turn = make_turn(
        objects=[core(position=(16, 16)), unit(2, "WORKER", position=origin)],
        obstacles=rocks,
    )
    context = context_for(turn)
    strategy_for(memory)._move_or_wait(
        turn.workers[0], goal, context, reason="refresh patrol coverage"
    )
    action = turn.plan.unit_actions.get(turn.workers[0].id)
    assert action is None or action.type == "WAIT"
    assert context.report.decisions[-1].action == "WAIT"


@pytest.mark.parametrize("hp,damage", [(2, 2), (1, 1)])
def test_service_goal_and_direct_traffic_cannot_override_fatal_fire(hp, damage):
    destination = (10, 12)
    memory = WorldMemory(incoming_damage={destination: (99, damage)})
    strategy = strategy_for(memory)
    turn = make_turn(
        objects=[
            core(position=destination),
            unit(2, "WORKER", position=(9, 12), hp=hp, cargo=1),
        ]
    )
    worker = turn.workers[0]
    context = context_for(turn)
    assert not strategy._has_static_route(worker, destination, context, allow_goal=True)
    assert not strategy._move(
        worker, destination, context, reason="deliver", allow_goal=True
    )
    assert not strategy._queue_move(
        worker, Direction.RIGHT, context, reason="deliver", target=destination
    )
    context.core_queue_head_id = worker.id
    assert not strategy._core_traffic_step_safe(
        worker, destination, context, target=destination
    )
    assert not turn.plan.unit_actions


def test_worker_can_leave_a_fatal_cell_without_excluding_its_neighbors():
    origin = (8, 12)
    memory = WorldMemory(incoming_damage={origin: (99, 2)})
    strategy = strategy_for(memory)
    turn = make_turn(
        objects=[core(position=(16, 16)), unit(2, "WORKER", position=origin)]
    )
    assert strategy._move(
        turn.workers[0], (9, 12), context_for(turn), reason="leave incoming fire"
    )
    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert add(origin, action.direction) == (9, 12)


def test_wounded_worker_routes_around_fire_with_shared_movement_maps():
    fatal_cell, goal = (9, 12), (10, 12)
    memory = WorldMemory(incoming_damage={fatal_cell: (99, 1)})
    strategy = strategy_for(memory)
    position = (8, 12)
    for tick in range(100, 104):
        turn = make_turn(
            tick=tick,
            objects=[
                core(position=(16, 16)),
                unit(2, "WORKER", position=position, hp=1),
            ],
        )
        context = context_for(turn)
        context.dynamic_blockers = set()
        context.worker_dynamic_blockers = set()
        assert strategy._move(
            turn.workers[0],
            goal,
            context,
            reason="approach service",
            avoid_recent=False,
        )
        action = turn.plan.unit_actions[turn.workers[0].id]
        assert action.type == "MOVE"
        position = add(position, action.direction)
        assert position != fatal_cell
    assert position == goal


def test_worker_releases_a_fatal_mine_and_reaches_another_resource():
    fatal_cell, resource = (9, 12), (8, 14)
    memory = WorldMemory(
        core_home_position=(16, 16), incoming_damage={fatal_cell: (99, 2)}
    )
    memory.set_goal(object_id(2), UnitGoal(fatal_cell, 99, RESOURCE_CLAIM_PURPOSE))
    strategy = strategy_for(memory)
    position = (8, 12)
    for tick in range(100, 103):
        turn = make_turn(
            tick=tick,
            objects=[core(position=(16, 16)), unit(2, "WORKER", position=position)],
            resource_cells=[fatal_cell, resource],
        )
        strategy.decide(turn)
        action = turn.plan.unit_actions[turn.workers[0].id]
        if tick < 102:
            assert action.type == "MOVE"
            position = add(position, action.direction)
            assert position != fatal_cell
        else:
            assert position == resource
            assert action.type == "HARVEST"


@pytest.mark.parametrize("damage,age", [(1, 1), (2, INCOMING_DAMAGE_TTL + 1)])
def test_nonlethal_or_expired_fire_does_not_close_worker_routes(damage, age):
    destination = (9, 12)
    memory = WorldMemory(incoming_damage={destination: (100 - age, damage)})
    turn = make_turn(
        objects=[core(position=(16, 16)), unit(2, "WORKER", position=(8, 12))]
    )
    memory.observe(turn)
    strategy = strategy_for(memory)
    assert strategy._move(
        turn.workers[0], destination, context_for(turn), reason="continue patrol"
    )
    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert add(turn.workers[0].position, action.direction) == destination


def test_friendly_cover_does_not_erase_fatal_fire_or_keep_its_patrol_goal():
    fatal_cell = (18, 18)
    memory = WorldMemory(
        core_home_position=(16, 16), incoming_damage={fatal_cell: (99, 2)}
    )
    memory.set_goal(object_id(2), UnitGoal(fatal_cell, 99, WORKER_PATROL_PURPOSE))
    turn = make_turn(
        objects=[
            core(position=(16, 16)),
            unit(2, "WORKER", position=(19, 18)),
            unit(3, "RANGER", position=(18, 17)),
        ]
    )
    memory.observe(turn)
    strategy = strategy_for(memory)
    context = context_for(turn)
    assert fatal_cell in strategy._worker_safe_defense_cells(turn)
    assert fatal_cell in strategy._worker_threat_exclusion_cells(turn)
    goal = strategy._worker_block_patrol_goal(turn.workers[0], context)
    assert goal != fatal_cell
