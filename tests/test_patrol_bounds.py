"""Coverage routes stay inside their assigned blocks, including after restart."""

from uuid import UUID

import pytest

from arena_hero_bot.exploration import visible_cells
from arena_hero_bot.geometry import add
from arena_hero_bot.memory import UnitGoal, WorldMemory
from arena_hero_bot.strategy import (
    CAMPAIGN_SCOUT_PURPOSE,
    CORE_LAYOUT_VERSION,
    PATROL_RING_PURPOSE,
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


@pytest.mark.parametrize("kind", ["WORKER", "RANGER"])
@pytest.mark.parametrize("restart", [False, True])
def test_patrol_takes_the_longer_interior_route_instead_of_leaving_the_block(
    tmp_path, kind, restart
):
    position, goal = (3, 7), (3, 9)
    rocks = {(x, 8) for x in range(9)}
    purpose = WORKER_PATROL_PURPOSE if kind == "WORKER" else PATROL_RING_PURPOSE
    memory = WorldMemory(core_home_position=(16, 16), obstacles=rocks)
    memory.set_goal(object_id(2), UnitGoal(goal, 100, purpose))
    strategy = strategy_for(memory)
    visited = {position}
    for tick in range(100, 114):
        if restart and tick == 105:
            path = tmp_path / "memory.json"
            memory.save(path)
            memory = WorldMemory.load(path)
            strategy = strategy_for(memory)
        turn = make_turn(
            tick=tick,
            objects=[core(position=(16, 16)), unit(2, kind, position=position)],
            obstacles=rocks,
        )
        memory.observe(turn)
        strategy._move_or_wait(
            turn.units[0], goal, context_for(turn), reason="refresh patrol coverage"
        )
        action = turn.plan.unit_actions[turn.units[0].id]
        assert action.type == "MOVE"
        position = add(position, action.direction)
        assert all(3 <= coordinate <= 28 for coordinate in position)
        assert position not in rocks
        assert position not in visited
        visited.add(position)
    assert position == goal


@pytest.mark.parametrize("kind", ["WORKER", "RANGER"])
@pytest.mark.parametrize("sealed", [False, True])
def test_patrol_can_use_a_needed_edge_passage_but_not_an_outside_passage(kind, sealed):
    position, goal = (15, 4), (17, 4)
    rocks = {(16, y) for y in range(0 if sealed else 2, 32)}
    purpose = WORKER_PATROL_PURPOSE if kind == "WORKER" else PATROL_RING_PURPOSE
    memory = WorldMemory(core_home_position=(16, 16), obstacles=rocks)
    memory.set_goal(object_id(2), UnitGoal(goal, 100, purpose))
    strategy = strategy_for(memory)
    for tick in range(100, 108):
        turn = make_turn(
            tick=tick,
            objects=[core(position=(20, 16)), unit(2, kind, position=position)],
            obstacles=rocks,
        )
        context = context_for(turn)
        strategy._move_or_wait(
            turn.units[0], goal, context, reason="refresh patrol coverage"
        )
        action = turn.plan.unit_actions.get(turn.units[0].id)
        if sealed:
            assert action is None
            assert context.report.decisions[-1].action == "WAIT"
        else:
            assert action is not None
            assert action.type == "MOVE"
            position = add(position, action.direction)
            assert all(0 <= coordinate <= 31 for coordinate in position)
    assert position == ((15, 4) if sealed else goal)


@pytest.mark.parametrize("kind", ["WORKER", "RANGER"])
def test_displaced_patrol_returns_and_does_not_leave_again(tmp_path, kind):
    position, goal = (-2, 10), (5, 10)
    purpose = WORKER_PATROL_PURPOSE if kind == "WORKER" else PATROL_RING_PURPOSE
    memory = WorldMemory(core_home_position=(16, 16))
    memory.set_goal(object_id(2), UnitGoal(goal, 100, purpose))
    for tick in range(100, 107):
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = strategy_for(memory)
        turn = make_turn(
            tick=tick,
            objects=[core(position=(16, 16)), unit(2, kind, position=position)],
        )
        strategy._move_or_wait(
            turn.units[0], goal, context_for(turn), reason="return to patrol coverage"
        )
        action = turn.plan.unit_actions[turn.units[0].id]
        assert action.type == "MOVE"
        following = add(position, action.direction)
        assert following == (position[0] + 1, position[1])
        position = following
    assert position == goal


def test_worker_replaces_old_edge_goal_with_an_interior_observation_post():
    unseen = (0, 12)
    memory = WorldMemory(
        core_home_position=(16, 16),
        worker_seen_ticks={
            (x, y): 100 for x in range(32) for y in range(32) if (x, y) != unseen
        },
    )
    memory.set_goal(object_id(2), UnitGoal(unseen, 100, WORKER_PATROL_PURPOSE))
    turn = make_turn(
        objects=[core(position=(16, 16)), unit(2, "WORKER", position=(5, 12))]
    )
    strategy_for(memory).decide(turn)
    goal = memory.goal_for(object_id(2))
    assert goal is not None
    assert all(3 <= coordinate <= 28 for coordinate in goal.position)
    assert unseen in visible_cells(goal.position, 3, set())
    action = turn.plan.unit_actions[UUID(object_id(2))]
    assert action.type == "MOVE"
    assert add((5, 12), action.direction) == (4, 12)


def test_worker_can_observe_a_corner_that_the_interior_cannot_see():
    unseen = (0, 0)
    memory = WorldMemory(
        core_home_position=(16, 16),
        worker_seen_ticks={
            (x, y): 100 for x in range(32) for y in range(32) if (x, y) != unseen
        },
    )
    turn = make_turn(
        objects=[core(position=(16, 16)), unit(2, "WORKER", position=(3, 3))]
    )
    strategy_for(memory).decide(turn)
    goal = memory.goal_for(object_id(2))
    assert goal is not None
    assert all(0 <= coordinate <= 31 for coordinate in goal.position)
    assert unseen in visible_cells(goal.position, 3, set())
    assert turn.plan.unit_actions[UUID(object_id(2))].type == "MOVE"


@pytest.mark.parametrize("restart", [False, True])
def test_worker_complete_plans_keep_a_rock_detour_inside(tmp_path, restart):
    worker = unit(2, "WORKER", position=(3, 7))
    rocks = {(x, 8) for x in range(9)}
    memory = WorldMemory(core_home_position=(16, 16), obstacles=rocks)
    memory.set_goal(worker["id"], UnitGoal((3, 9), 100, WORKER_PATROL_PURPOSE))
    strategy = strategy_for(memory)
    for tick in range(100, 114):
        if restart and tick == 105:
            path = tmp_path / "memory.json"
            memory.save(path)
            memory = WorldMemory.load(path)
            strategy = strategy_for(memory)
        turn = make_turn(
            tick=tick, objects=[core(position=(16, 16)), worker], obstacles=rocks
        )
        report = strategy.decide(turn)
        decision = next(d for d in report.decisions if d.actor_id == worker["id"])
        assert decision.target == (3, 9)
        action = turn.plan.unit_actions[UUID(worker["id"])]
        assert action.type == "MOVE"
        position = add(turn.workers[0].position, action.direction)
        assert all(3 <= coordinate <= 28 for coordinate in position)
        worker["position"] = list(position)
    assert worker["position"] == [3, 9]


def test_forward_scout_crosses_only_the_published_destination_block():
    target = (48, 16)
    memory = WorldMemory(core_home_position=(16, 16), campaign_scout_target=target)
    memory.set_goal(
        object_id(2), UnitGoal((35, 12), 100, f"{CAMPAIGN_SCOUT_PURPOSE}-48-16")
    )
    strategy = strategy_for(memory)
    position = (30, 12)
    for tick in range(100, 105):
        turn = make_turn(
            tick=tick,
            objects=[core(position=(16, 16)), unit(2, "RANGER", position=position)],
        )
        strategy._move_or_wait(
            turn.rangers[0], (35, 12), context_for(turn), reason="scout destination"
        )
        action = turn.plan.unit_actions[turn.rangers[0].id]
        assert action.type == "MOVE"
        position = add(position, action.direction)
    assert position == (35, 12)


def test_scout_observes_an_edge_gap_from_the_interior():
    target, unseen = (48, 16), (32, 12)
    memory = WorldMemory(
        core_home_position=(16, 16),
        campaign_scout_target=target,
        campaign_scout_started_tick=100,
        campaign_scout_seen_ticks={
            (x, y): 100 for x in range(32, 64) for y in range(32) if (x, y) != unseen
        },
    )
    # Block the nearest regular anchor so this exercises gap observation.
    rocks = {(36, 12), (36, 4), (36, 20)}
    turn = make_turn(
        objects=[core(position=(16, 16)), unit(2, "RANGER", position=(39, 12))],
        obstacles=rocks,
    )
    goal = strategy_for(memory)._campaign_scout_goal(
        turn.rangers[0], turn, context_for(turn)
    )
    assert 35 <= goal[0] <= 60 and 3 <= goal[1] <= 28
    assert unseen in visible_cells(goal, 5, rocks)
