"""Friendly occupancy is not terrain, including on the Core service cell."""

import pytest
from arena_hero import Direction, MoveAction, core_resource_capacity

from arena_hero_bot.geometry import add
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import (
    AggressiveStrategy,
    StrategyConfig,
    _MoveIntent,
    _TurnContext,
)

from .factories import core, make_turn, unit


def context_for(turn):
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={u.position for u in turn.units},
        enemy_positions={e.position for e in turn.visible_enemies},
        remaining_resources=turn.resources,
        remaining_resource_space=turn.resource_space,
    )


@pytest.mark.parametrize("kind", ["WORKER", "RANGER", "VANGUARD"])
def test_core_capacity_allows_one_inbound_and_rejects_the_next(kind):
    turn = make_turn(
        objects=[
            core(),
            unit(2, kind, position=(-1, 0)),
            unit(3, kind, position=(1, 0)),
        ]
    )
    strategy = AggressiveStrategy(WorldMemory())
    context = context_for(turn)
    first, second = turn.units
    assert strategy._move(
        first,
        (0, 0),
        context,
        reason="return to shared Core",
        intent=_MoveIntent.INBOUND,
    )
    assert not strategy._move(
        second,
        (0, 0),
        context,
        reason="return to shared Core",
        intent=_MoveIntent.INBOUND,
    )


def test_queue_slots_are_unique_and_stationary_units_hold_them():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(5, 0), cargo=1),
            unit(3, "WORKER", position=(6, 0), cargo=1),
        ]
    )
    strategy = AggressiveStrategy(WorldMemory())
    context = context_for(turn)
    first, second = turn.workers
    assert strategy._move_to_core_queue(first, context, reason="queue full cargo")
    assert strategy._move_to_core_queue(second, context, reason="queue full cargo")
    first_action = turn.plan.unit_actions[first.id]
    second_action = turn.plan.unit_actions[second.id]
    assert isinstance(first_action, MoveAction)
    assert isinstance(second_action, MoveAction)
    first_destination = add(first.position, first_action.direction)
    second_destination = add(second.position, second_action.direction)
    assert first_destination != second_destination
    assert len(context.standing_reserved) == 2


@pytest.mark.parametrize(
    "kinds", [("WORKER", "WORKER"), ("RANGER", "VANGUARD"), ("WORKER", "RANGER")]
)
def test_units_swap_positions_in_a_one_cell_corridor(kinds):
    turn = make_turn(
        objects=[
            core(position=(10, 0)),
            unit(2, kinds[0]),
            unit(3, kinds[1], position=(1, 0)),
        ],
        obstacles=[(0, -1), (0, 1), (1, -1), (1, 1)],
    )
    strategy = AggressiveStrategy(WorldMemory())
    context = context_for(turn)
    first, second = turn.units
    assert strategy._move(first, second.position, context, reason="cross corridor")
    assert strategy._move(second, first.position, context, reason="cross corridor")
    first_action = turn.plan.unit_actions[first.id]
    second_action = turn.plan.unit_actions[second.id]
    assert first_action.type == "MOVE"
    assert second_action.type == "MOVE"
    assert first_action.direction is Direction.RIGHT
    assert second_action.direction is Direction.LEFT


def test_worker_harvest_trip_swaps_with_cargo_return_in_complete_plan():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER"),
            unit(3, "WORKER", position=(1, 0), cargo=1),
        ],
        resource_cells=[(1, 0)],
    )
    AggressiveStrategy(WorldMemory()).decide(turn)
    outbound = turn.plan.unit_actions[turn.workers[0].id]
    inbound = turn.plan.unit_actions[turn.workers[1].id]
    assert outbound.type == "MOVE"
    assert inbound.type == "MOVE"
    assert outbound.direction is Direction.RIGHT
    assert inbound.direction is Direction.LEFT


def test_core_service_accepts_one_deposit_and_queues_the_next_unit():
    turn = make_turn(
        resources=2,
        objects=[
            core(),
            unit(2, "WORKER", cargo=1),
            unit(3, "WORKER", position=(1, 0), cargo=1),
        ],
    )
    report = AggressiveStrategy(WorldMemory()).decide(turn)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "DEPOSIT"
    assert turn.plan.unit_actions.get(turn.workers[1].id) is None
    assert [d.action for d in report.decisions if d.actor_kind == "WORKER"] == [
        "DEPOSIT",
        "WAIT",
    ]


def test_shared_core_deposits_respect_remaining_capacity():
    turn = make_turn(
        resources=9,
        objects=[core(), unit(2, "WORKER", cargo=1), unit(3, "WORKER", cargo=1)],
    )
    report = AggressiveStrategy(WorldMemory()).decide(turn)
    workers = [d for d in report.decisions if d.actor_kind == "WORKER"]
    assert sorted(d.action for d in workers) == ["DEPOSIT", "MOVE"]
    assert any("Core service cell" in d.reason for d in workers if d.action == "MOVE")


def test_returned_expedition_rangers_use_one_core_heal_service():
    turn = make_turn(
        resources=2,
        objects=[
            core(),
            unit(2, "RANGER", hp=1),
            unit(3, "RANGER", position=(1, 0), hp=1),
        ],
    )
    strategy = AggressiveStrategy(WorldMemory(), StrategyConfig(expedition_mode=True))
    context = context_for(turn)
    assert strategy._heal_if_critical(turn.rangers[0], maximum_hp=2, context=context)
    assert strategy._recover_if_critical(turn.rangers[1], maximum_hp=2, context=context)
    actions = [turn.plan.unit_actions.get(ranger.id) for ranger in turn.rangers]
    assert sum(action is not None and action.type == "HEAL" for action in actions) == 1
    assert sum(action is not None and action.type == "MOVE" for action in actions) == 0
    assert context.remaining_resources == 1


def test_full_core_ring_clears_an_approach_before_core_departure():
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    objects = [core(), unit(2, "WORKER", cargo=1)]
    number = 3
    for position in neighbors:
        for _ in range(2):
            objects.append(unit(number, "WORKER", position=position, cargo=1))
            number += 1
    population = len(objects) - 1
    turn = make_turn(
        resources=core_resource_capacity(population),
        objects=objects,
    )

    report = AggressiveStrategy(
        WorldMemory(), StrategyConfig(target_workers=16, max_population=None)
    ).decide(turn)

    core_worker = turn.workers[0]
    core_action = turn.plan.unit_actions[core_worker.id]
    assert core_action.type == "MOVE"
    core_destination = add(core_worker.position, core_action.direction)
    assert core_destination in neighbors
    assert any(
        item.action == "MOVE" and item.reason == "clear a full Core approach cell"
        for item in report.decisions
    )


@pytest.mark.parametrize("remembered", [False, True])
@pytest.mark.parametrize("kind", ["WORKER", "RANGER", "VANGUARD"])
def test_rock_remains_blocked_even_when_goal_is_allowed(kind, remembered):
    rocks = {(1, 0)}
    turn = make_turn(
        objects=[core(position=(10, 0)), unit(2, kind)],
        obstacles=() if remembered else rocks,
    )
    strategy = AggressiveStrategy(WorldMemory(obstacles=rocks if remembered else set()))
    context = context_for(turn)
    actor = turn.units[0]
    assert not strategy._has_static_route(actor, (1, 0), context, allow_goal=True)
    assert not strategy._move(
        actor, (1, 0), context, reason="invalid goal", allow_goal=True
    )
    assert not strategy._queue_move(
        actor, Direction.RIGHT, context, reason="invalid step", target=(2, 0)
    )
    assert not turn.plan.unit_actions
    assert strategy._move(
        actor, (2, 0), context, reason="route around rock", allow_goal=True
    )
    action = turn.plan.unit_actions[actor.id]
    assert action.type == "MOVE"
    assert add(actor.position, action.direction) not in rocks


def test_enemy_positions_still_constrain_a_route():
    turn = make_turn(
        objects=[
            core(position=(10, 0)),
            unit(2, "VANGUARD"),
            unit(3, "VANGUARD", controlled=False, position=(1, 0)),
        ]
    )
    strategy = AggressiveStrategy(WorldMemory())
    context = context_for(turn)
    actor = turn.units[0]
    assert strategy._move(actor, (2, 0), context, reason="avoid hostile cell")
    action = turn.plan.unit_actions[actor.id]
    assert action.type == "MOVE"
    assert add(actor.position, action.direction) != (1, 0)
