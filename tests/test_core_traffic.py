"""Same-Tick swaps, shared queues and complete Core production exits."""

from collections import Counter
from uuid import UUID

import pytest
from arena_hero import core_resource_capacity

from arena_hero_bot.geometry import add
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy
from arena_hero_bot.traffic import resolve_moves

from .factories import core, make_turn, unit
from .test_core_service import campaign_strategy
from .test_shared_movement import context_for


def settled_positions(turn):
    """Apply all emitted movements at once, independent of planner ledgers."""
    positions = {
        member.id: (
            add(member.position, action.direction)
            if (action := turn.plan.unit_actions.get(member.id)) is not None
            and action.type == "MOVE"
            else member.position
        )
        for member in turn.units
    }
    occupancy = Counter(positions.values())
    if turn.core is not None:
        occupancy[turn.core.position] += 1
    assert max(occupancy.values()) <= 2
    assert not set(positions.values()) & set(turn.obstacle_cells)
    return positions


@pytest.mark.parametrize("reverse", [False, True])
def test_full_cells_exchange_one_occupant_without_any_empty_cell(reverse):
    origins = {UUID(int=2): (0, 0), UUID(int=3): (1, 0)}
    requests = {actor: ((1 - origin[0], 0),) for actor, origin in origins.items()}
    if reverse:
        requests = dict(reversed(tuple(requests.items())))
    moves = resolve_moves({(0, 0): 2, (1, 0): 2}, origins, requests, requests)
    assert moves == {UUID(int=2): (1, 0), UUID(int=3): (0, 0)}


def test_closed_cycle_rotates_one_slot_per_full_cell():
    cells = ((0, 0), (1, 0), (1, 1), (0, 1))
    origins = {UUID(int=i + 2): cell for i, cell in enumerate(cells)}
    requests = {
        actor: (cells[(i + 1) % len(cells)],) for i, actor in enumerate(origins)
    }
    assert resolve_moves(dict.fromkeys(cells, 2), origins, requests, requests) == {
        actor: destinations[0] for actor, destinations in requests.items()
    }


@pytest.mark.parametrize("free_end", [False, True])
def test_long_exit_chain_is_atomic_and_bounded(free_end):
    origins = {UUID(int=i + 2): (i, 0) for i in range(7)}
    options = {actor: ((position[0] + 1, 0),) for actor, position in origins.items()}
    occupancy = dict.fromkeys([(i, 0) for i in range(8)], 2)
    if free_end:
        occupancy[(7, 0)] = 1
    moves = resolve_moves(occupancy, origins, {UUID(int=2): ((1, 0),)}, options)
    assert len(moves) == (7 if free_end else 0)
    assert not resolve_moves(
        occupancy, origins, {UUID(int=2): ((1, 0),)}, options, max_depth=2
    )


def test_two_competing_arrivals_cannot_overbook_a_shared_cell():
    origins = {UUID(int=2): (0, 0), UUID(int=3): (2, 0)}
    requests = dict.fromkeys(origins, ((1, 0),))
    moves = resolve_moves({(0, 0): 1, (1, 0): 1, (2, 0): 1}, origins, requests, {})
    assert moves == {UUID(int=2): (1, 0)}


def test_full_core_and_full_doorstep_swap_in_the_complete_planner():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER"),
            unit(3, "WORKER", position=(1, 0), cargo=1),
            unit(4, "WORKER", position=(1, 0)),
        ],
        resource_cells=[(2, 0)],
        obstacles=[(0, -1), (0, 1), (-1, 0), (1, -1), (1, 1)],
    )
    AggressiveStrategy(WorldMemory()).decide(turn)
    positions = settled_positions(turn)
    outbound, inbound, _ = turn.workers
    assert positions[outbound.id] == (1, 0)
    assert positions[inbound.id] == (0, 0)


def test_cargo_advances_onto_a_guard_and_does_not_disperse_while_front_is_busy():
    strategy = AggressiveStrategy(WorldMemory())
    position = (-3, 0)
    for tick in (100, 101):
        turn = make_turn(
            tick=tick,
            objects=[
                core(),
                unit(2, "WORKER", cargo=1),
                unit(3, "WORKER", position=(-1, 0), cargo=1),
                unit(4, "WORKER", position=position, cargo=1),
                unit(5, "RANGER", position=(-2, 0)),
            ],
            obstacles=[(x, y) for x in range(-5, 2) for y in (-1, 1)],
        )
        context = context_for(turn)
        context.core_service_action = "DEPOSIT"
        strategy._move_to_core_queue(turn.workers[-1], context, reason="queue cargo")
        positions = settled_positions(turn)
        position = positions[turn.workers[-1].id]
        assert position == ((-2, 0) if tick == 100 else (-1, 0))


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("cargo", [0, 1])
def test_production_clears_a_chain_of_fully_stacked_cells(blocked, cargo):
    strategy = campaign_strategy()
    roster = [unit(2, "WORKER")]
    for x in range(1, 6):
        roster.extend(
            [
                unit(10 + 2 * x, "WORKER", position=(x, 0), cargo=cargo),
                unit(11 + 2 * x, "WORKER", position=(x, 0), cargo=cargo),
            ]
        )
    corridor = {(x, 0) for x in range(7 if not blocked else 6)}
    turn = make_turn(
        resources=core_resource_capacity(len(roster)),
        objects=[core(), *roster],
        obstacles=[
            (x, y) for x in range(-1, 8) for y in (-1, 0, 1) if (x, y) not in corridor
        ],
    )
    context = context_for(turn)
    context.core_build_priority = True
    context.core_traffic_collecting = True
    strategy._prepare_core_departure(context)
    strategy._resolve_core_traffic(context)
    positions = settled_positions(turn)
    if blocked:
        assert positions[turn.workers[0].id] == (0, 0)
        assert not turn.plan.unit_actions
    else:
        assert positions[turn.workers[0].id] == (1, 0)
        assert Counter(positions.values())[(6, 0)] == 1
        strategy._decide_core(context)
        assert turn.plan.core_action is None  # spawn after clearance is observed


def test_injured_empty_worker_can_enter_for_a_paid_heal():
    turn = make_turn(
        resources=1,
        objects=[core(), unit(2, "WORKER", position=(1, 0), hp=1)],
    )
    AggressiveStrategy(WorldMemory()).decide(turn)
    assert settled_positions(turn)[turn.workers[0].id] == (0, 0)


def test_distant_cargo_keeps_closing_outside_the_local_traffic_radius():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", cargo=1),
            unit(3, "WORKER", position=(20, 0), cargo=1),
        ],
    )
    AggressiveStrategy(WorldMemory()).decide(turn)
    assert settled_positions(turn)[turn.workers[-1].id] == (19, 0)


@pytest.mark.parametrize("reason", ["combat shot", "heal", "harvest"])
def test_chain_never_overrides_committed_combat_or_service(reason):
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER"),
            unit(3, "RANGER", position=(1, 0)),
            unit(4, "WORKER", position=(1, 0)),
        ],
        obstacles=[(-1, 0), (0, 1), (0, -1), (1, 1), (1, -1)],
    )
    strategy = campaign_strategy()
    context = context_for(turn)
    context.core_build_priority = True
    context.core_traffic_collecting = True
    guard = turn.rangers[0]
    guard.shoot_cell((3, 0))
    if reason == "heal":
        turn.workers[-1].heal()
    else:
        turn.workers[-1].harvest()
    committed = dict(turn.plan.unit_actions)
    strategy._prepare_core_departure(context)
    strategy._resolve_core_traffic(context)
    assert turn.plan.unit_actions == committed
    assert settled_positions(turn)[turn.workers[0].id] == (0, 0)
