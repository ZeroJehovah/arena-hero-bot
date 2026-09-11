"""Friendly occupancy is not terrain, including on the Core service cell."""

import pytest
from arena_hero import Direction

from arena_hero_bot.geometry import add
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig, _TurnContext

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


@pytest.mark.parametrize("target_workers", [2, 16])
@pytest.mark.parametrize("kind", ["WORKER", "RANGER", "VANGUARD"])
def test_all_units_can_share_destination_with_stationary_friends(kind, target_workers):
    turn = make_turn(
        objects=[
            core(),
            unit(2, kind, position=(-1, 0)),
            unit(3, kind, position=(1, 0)),
            unit(4, kind, position=(0, -1)),
            unit(5, "RANGER"),
            unit(6, "VANGUARD"),
        ]
    )
    strategy = AggressiveStrategy(
        WorldMemory(), StrategyConfig(target_workers=target_workers)
    )
    context = context_for(turn)
    context.reserved.add((0, 0))
    for actor in turn.units[:3]:
        assert strategy._move(actor, (0, 0), context, reason="return to shared Core")
        action = turn.plan.unit_actions[actor.id]
        assert action.type == "MOVE"
        assert add(actor.position, action.direction) == (0, 0)
    assert all(actor.id not in turn.plan.unit_actions for actor in turn.units[3:])


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


def test_multiple_deposits_and_heals_share_core_without_eviction():
    turn = make_turn(
        resources=2,
        objects=[
            core(),
            unit(2, "WORKER", cargo=1),
            unit(3, "WORKER", cargo=1),
            unit(4, "RANGER", hp=1),
            unit(5, "VANGUARD", hp=3),
        ],
    )
    report = AggressiveStrategy(WorldMemory()).decide(turn)
    assert all(turn.plan.unit_actions[w.id].type == "DEPOSIT" for w in turn.workers)
    assert all(
        turn.plan.unit_actions[u.id].type == "HEAL"
        for u in (*turn.rangers, *turn.vanguards)
    )
    assert len([d for d in report.decisions if d.actor_kind != "CORE"]) == 4


def test_shared_core_deposits_respect_remaining_capacity():
    turn = make_turn(
        resources=9,
        objects=[core(), unit(2, "WORKER", cargo=1), unit(3, "WORKER", cargo=1)],
    )
    report = AggressiveStrategy(WorldMemory()).decide(turn)
    workers = [d for d in report.decisions if d.actor_kind == "WORKER"]
    assert sorted(d.action for d in workers) == ["DEPOSIT", "WAIT"]
    assert (
        next(d for d in workers if d.action == "WAIT").reason == "Core storage is full"
    )


def test_returned_expedition_rangers_heal_together_at_core():
    turn = make_turn(
        resources=2, objects=[core(), unit(2, "RANGER", hp=1), unit(3, "RANGER", hp=1)]
    )
    strategy = AggressiveStrategy(WorldMemory(), StrategyConfig(expedition_mode=True))
    context = context_for(turn)
    for ranger in turn.rangers:
        assert strategy._decide_expedition_ranger(ranger, context)
        assert turn.plan.unit_actions[ranger.id].type == "HEAL"
    assert context.remaining_resources == 0


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
