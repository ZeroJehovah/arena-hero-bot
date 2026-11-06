"""Production and cargo service remain live while a defended Core is parked."""

import pytest
from arena_hero import UnitType, core_resource_capacity, unit_cost

from arena_hero_bot.geometry import add
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import (
    CAMPAIGN_PHASE_STAGE4,
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
    _MoveIntent,
)

from .factories import core, make_turn, unit
from .test_shared_movement import context_for


def campaign_strategy():
    return AggressiveStrategy(
        WorldMemory(
            core_home_position=(0, 0),
            core_layout_version=CORE_LAYOUT_VERSION,
            campaign_phase=CAMPAIGN_PHASE_STAGE4,
            campaign_resting=False,
            unit_roles_initialized=True,
        ),
        StrategyConfig(
            target_workers=16,
            max_population=None,
            expedition_mode=True,
            campaign_mode=True,
        ),
    )


@pytest.mark.parametrize("posture", ["layers", "counterattack"])
@pytest.mark.parametrize(
    ("workers", "rangers", "full", "expected"),
    [
        (14, 16, True, UnitType.WORKER),
        (16, 16, True, UnitType.RANGER),
        (16, 28, True, UnitType.RANGER),
        (14, 16, False, UnitType.WORKER),
    ],
)
def test_parked_campaign_produces_the_next_slot_during_defensive_holds(
    posture, workers, rangers, full, expected
):
    strategy = campaign_strategy()
    roster = [unit(100 + n, "WORKER", position=(n, 10)) for n in range(workers)]
    guards = [unit(200 + n, "RANGER", position=(n, -4)) for n in range(rangers)]
    strategy.memory.unit_roles = {
        item["id"]: "defense" if n < 20 else "patrol-1" for n, item in enumerate(guards)
    }
    population = workers + rangers
    turn = make_turn(
        resources=(
            core_resource_capacity(population)
            if full
            else unit_cost(expected, population)
        ),
        objects=[core(), *roster, *guards],
    )
    context = context_for(turn)
    context.core_build_priority = strategy._campaign_build_priority(context)
    if posture == "layers":
        context.layered_assignments = {turn.rangers[0].id: (0, -4)}
    else:
        context.counterattack_ids = frozenset({turn.rangers[0].id})

    strategy._decide_core(context)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is expected


def test_full_parked_core_clears_cargo_then_builds_before_reopening_delivery():
    strategy = campaign_strategy()
    position = (0, 0)
    for tick in (100, 101):
        turn = make_turn(
            tick=tick,
            resources=core_resource_capacity(2),
            objects=[
                core(),
                unit(2, "WORKER", position=position, cargo=1),
                unit(3, "WORKER", position=(2, 0), cargo=1),
            ],
        )
        context = context_for(turn)
        context.core_build_priority = strategy._campaign_build_priority(context)
        strategy._prepare_core_departure(context)
        for worker in turn.workers:
            if not strategy._has_unit_action(worker, context):
                strategy._decide_worker(worker, context)
        strategy._decide_core(context)
        assert context.core_inbound_id is None
        if tick == 100:
            action = turn.plan.unit_actions[turn.workers[0].id]
            assert action.type == "MOVE"
            position = add(position, action.direction)
            assert position != (0, 0)
            assert turn.plan.core_action is None
        else:
            assert turn.plan.core_action is not None
            assert turn.plan.core_action.type == "SPAWN"
            assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_actual_layered_planning_does_not_suppress_full_bank_production():
    strategy = campaign_strategy()
    guards = [unit(20 + n, "RANGER", position=(n - 3, -3)) for n in range(8)]
    strategy.memory.unit_roles = {guard["id"]: "defense" for guard in guards}
    turn = make_turn(
        resources=core_resource_capacity(10),
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 2), cargo=1),
            unit(3, "WORKER", position=(1, 2), cargo=1),
            *guards,
            unit(90, "RANGER", controlled=False, position=(0, -7)),
        ],
    )
    assert turn.core is not None
    strategy.memory.campaign_last_core_id = str(turn.core.id)

    strategy.decide(turn)

    assert strategy.memory.layered_defense_posts
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_queue_head_uses_walkable_distance_around_rocks():
    turn = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 2), cargo=1),
            unit(3, "WORKER", position=(-1, 1), cargo=1),
        ],
        obstacles=[(0, 1)],
    )
    strategy = AggressiveStrategy(WorldMemory())

    assert strategy._select_stationary_core_queue_head(context_for(turn)) == (
        turn.workers[1].id
    )


def test_geometrically_close_waiter_leaves_a_rock_pocket_toward_the_queue():
    turn = make_turn(
        objects=[core(), unit(2, "WORKER", position=(0, 2), cargo=1)],
        obstacles=[(0, 1), (-1, 1), (1, 1), (-1, 2), (1, 2)],
    )
    strategy = AggressiveStrategy(WorldMemory())
    context = context_for(turn)
    worker = turn.workers[0]

    assert strategy._move_to_core_queue(worker, context, reason="queue cargo")

    action = turn.plan.unit_actions.get(worker.id)
    assert action is not None and action.type == "MOVE"
    assert add(worker.position, action.direction) == (0, 3)


def test_queue_compaction_never_steps_into_a_shorter_geometric_dead_end():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(-1, 2), cargo=1),
            unit(3, "WORKER", position=(-1, 0)),
            unit(4, "WORKER", position=(1, 0)),
            unit(5, "WORKER", position=(0, -1)),
        ],
        obstacles=[(0, 1)],
    )
    strategy = AggressiveStrategy(WorldMemory())
    context = context_for(turn)
    worker = turn.workers[0]

    assert strategy._move_to_core_queue(worker, context, reason="queue cargo")

    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert add(worker.position, action.direction) == (-1, 1)


def test_queue_cells_are_within_four_walkable_steps_of_core():
    turn = make_turn(
        objects=[core(), unit(2, "WORKER", position=(2, 2), cargo=1)],
        obstacles=[(0, 1), (-1, 1), (1, 1), (-1, 2), (1, 2)],
    )
    strategy = AggressiveStrategy(WorldMemory())

    positions = strategy._core_queue_positions(turn.workers[0], context_for(turn))

    assert (0, 2) not in positions
    assert (2, 0) in positions


def test_delivery_pause_does_not_send_a_worker_back_around_the_same_rock():
    strategy = AggressiveStrategy(WorldMemory())
    position = (0, 2)
    travelled = [position]
    delivered = False
    for tick in range(100, 107):
        busy = tick == 101
        turn = make_turn(
            tick=tick,
            objects=[
                core(),
                unit(2, "WORKER", position=position, cargo=1),
                unit(3, "WORKER", position=(-1, 0)),
                unit(4, "WORKER", position=(0, -1)),
                *(
                    [
                        unit(5, "WORKER", cargo=1),
                        unit(6, "WORKER", position=(-1, 1)),
                    ]
                    if busy
                    else []
                ),
            ],
            obstacles=[(0, 1), (1, 1), (1, 2)],
        )
        strategy.memory.observe(turn)
        context = context_for(turn)
        if busy:
            depositor = next(w for w in turn.workers if w.position == (0, 0))
            strategy._decide_worker(depositor, context)
            assert context.core_service_action == "DEPOSIT"
        context.core_queue_head_id = strategy._select_stationary_core_queue_head(
            context
        )
        worker = turn.workers[0]
        strategy._decide_worker(worker, context)
        action = turn.plan.unit_actions.get(worker.id)
        if action is not None and action.type == "MOVE":
            position = add(position, action.direction)
            assert position not in travelled
            travelled.append(position)
        elif action is not None and action.type == "DEPOSIT":
            delivered = True
            break
    assert delivered
    assert travelled == [(0, 2), (-1, 2), (-1, 1), (-1, 0), (0, 0)]


@pytest.mark.parametrize(
    ("hp", "shield", "expected"), [(2, 5, "HEAL"), (5, 4, "REPAIR_SHIELD")]
)
def test_build_priority_keeps_emergency_core_recovery(hp, shield, expected):
    strategy = campaign_strategy()
    turn = make_turn(
        resources=core_resource_capacity(1),
        objects=[
            core(hp=hp, shield=shield),
            unit(2, "WORKER", position=(0, 5)),
            unit(3, "RANGER", controlled=False, position=(0, -3)),
        ],
    )
    context = context_for(turn)
    context.core_build_priority = strategy._campaign_build_priority(context)

    strategy._decide_core(context)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == expected


def test_queue_slots_keep_combat_danger_exclusion():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(-3, 0), cargo=1),
            unit(3, "RANGER", controlled=False, position=(4, 3)),
        ]
    )
    strategy = AggressiveStrategy(WorldMemory())
    strategy.memory.observe(turn)

    positions = strategy._core_queue_positions(turn.workers[0], context_for(turn))

    assert (3, 0) not in positions
    assert (-1, 0) in positions


def test_shared_waiters_both_advance_into_the_open_forward_slot():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(-1, 2), cargo=1),
            unit(3, "WORKER", position=(-1, 2), cargo=1),
            unit(4, "RANGER", position=(-1, 0)),
            unit(5, "RANGER", position=(0, -1)),
            unit(6, "RANGER", position=(1, 0)),
        ],
        obstacles=[(0, 1)],
    )
    strategy = AggressiveStrategy(WorldMemory())
    context = context_for(turn)

    strategy._move_to_core_queue(turn.workers[0], context, reason="queue cargo")
    strategy._move_to_core_queue(turn.workers[1], context, reason="queue cargo")

    first = turn.plan.unit_actions[turn.workers[0].id]
    assert first.type == "MOVE"
    assert add(turn.workers[0].position, first.direction) == (-1, 1)
    second = turn.plan.unit_actions[turn.workers[1].id]
    assert second.type == "MOVE"
    assert add(turn.workers[1].position, second.direction) == (-1, 1)
    assert strategy._predicted_friendly_occupancy((-1, 1), context) == 2


def test_transit_through_a_queue_slot_does_not_evict_the_waiting_cargo():
    """An empty Worker may pass through without making the waiter step out/back."""

    strategy = AggressiveStrategy(WorldMemory())
    for tick, passing_position in ((100, (-2, 0)), (101, (-3, 0))):
        turn = make_turn(
            tick=tick,
            objects=[
                core(),
                unit(2, "WORKER", position=(-3, 0), cargo=1),
                unit(3, "WORKER", position=passing_position),
                unit(4, "RANGER", position=(-2, 0)),
                unit(5, "RANGER", position=(-1, 0)),
            ],
            obstacles=[(x, y) for x in range(-5, 2) for y in (-1, 1)],
        )
        context = context_for(turn)
        waiter, passing = turn.workers
        assert strategy._move(
            passing,
            (-5, 0),
            context,
            reason="return to mining",
            intent=_MoveIntent.TRANSIT,
            avoid_recent=False,
        )
        assert strategy._move_to_core_queue(waiter, context, reason="queue cargo")

        action = turn.plan.unit_actions[waiter.id]
        assert action.type == "MOVE"
        assert add(waiter.position, action.direction) == (-2, 0)
        assert strategy._predicted_friendly_occupancy((-2, 0), context) == 2


def test_distant_cargo_passes_through_rangers_to_reach_a_unique_queue_slot():
    """The path may share two guard cells; only the final waiting slot is unique."""

    strategy = AggressiveStrategy(WorldMemory())
    position = (-6, 0)
    destinations = []
    for tick in range(100, 106):
        turn = make_turn(
            tick=tick,
            objects=[
                core(),
                unit(2, "WORKER", position=position, cargo=1),
                unit(3, "RANGER", position=(-5, 0)),
                unit(4, "RANGER", position=(-4, 0)),
            ],
            obstacles=[(x, y) for x in range(-7, 2) for y in (-1, 1)],
        )
        context = context_for(turn)
        context.core_service_action = "DEPOSIT"
        worker = turn.workers[0]

        assert strategy._move_to_core_queue(worker, context, reason="queue cargo")

        action = turn.plan.unit_actions.get(worker.id)
        if position == (-1, 0):
            assert action is None
            break
        assert action is not None and action.type == "MOVE"
        position = add(position, action.direction)
        destinations.append(position)
        assert strategy._predicted_friendly_occupancy(position, context) <= 2
    assert destinations == [(-5, 0), (-4, 0), (-3, 0), (-2, 0), (-1, 0)]


@pytest.mark.parametrize("near_id, far_id", [(20, 10), (10, 20)])
@pytest.mark.parametrize("shared", [False, True])
def test_nearby_cargo_gets_the_open_queue_slot_before_distant_cargo(
    monkeypatch, near_id, far_id, shared
):
    strategy = AggressiveStrategy(WorldMemory())
    # Keep the guards at their posts while the complete planner arbitrates
    # the depositing Worker, the waiting head and the two approaching loads.
    monkeypatch.setattr(
        strategy,
        "_decide_ranger",
        lambda ranger, context, *args, **kwargs: strategy._record_wait(
            ranger, context, "hold guard post"
        ),
    )
    corridor = (
        {(x, 0) for x in range(-5, 1)} | {(-3, y) for y in range(1, 6)} | {(-4, -1)}
    )
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", cargo=1),
            unit(3, "WORKER", position=(-1, 0), cargo=1),
            unit(near_id, "WORKER", position=(-4, 0), cargo=1),
            unit(far_id, "WORKER", position=(-3, 5), cargo=1),
            unit(100, "RANGER", position=(-2, 0)),
            *([unit(101, "RANGER", position=(-4, 0))] if shared else []),
        ],
        obstacles=[
            (x, y) for x in range(-6, 2) for y in range(-2, 7) if (x, y) not in corridor
        ],
    )

    strategy.decide(turn)

    nearby = next(worker for worker in turn.workers if worker.id.int == near_id)
    action = turn.plan.unit_actions.get(nearby.id)
    assert action is not None and action.type == "MOVE"
    assert add(nearby.position, action.direction) == (-3, 0)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "DEPOSIT"


def test_cargo_reservations_use_walking_distance_before_geometric_distance():
    strategy = AggressiveStrategy(WorldMemory())
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", cargo=1),
            unit(3, "WORKER", position=(-1, 0), cargo=1),
            unit(10, "WORKER", position=(0, 2), cargo=1),
            unit(20, "WORKER", position=(-3, 0), cargo=1),
        ],
        obstacles=[(-1, 1), (0, 1), (1, 1), (1, 2)],
    )

    strategy.decide(turn)

    nearby = turn.workers[-1]
    action = turn.plan.unit_actions.get(nearby.id)
    assert action is not None and action.type == "MOVE"
    assert add(nearby.position, action.direction) == (-2, 0)


@pytest.mark.parametrize("head_id, empty_id", [(10, 20), (20, 10)])
@pytest.mark.parametrize("departing", [True, False])
def test_queue_head_fills_a_same_tick_worker_departure(head_id, empty_id, departing):
    strategy = AggressiveStrategy(WorldMemory())
    corridor = {(0, 0), (-1, 0), (-2, 0), (-3, 0), (-1, 1)}
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", cargo=1),
            unit(head_id, "WORKER", position=(-1, 1), cargo=1),
            unit(empty_id, "WORKER", position=(-1, 0)),
        ],
        resource_cells=[(-3, 0) if departing else (-1, 0)],
        obstacles=[
            (x, y) for x in range(-4, 2) for y in range(-1, 3) if (x, y) not in corridor
        ],
    )

    strategy.decide(turn)

    depositor, head, empty = turn.workers
    assert turn.plan.unit_actions[depositor.id].type == "DEPOSIT"
    empty_action = turn.plan.unit_actions[empty.id]
    head_action = turn.plan.unit_actions.get(head.id)
    if departing:
        assert empty_action.type == "MOVE"
        assert add(empty.position, empty_action.direction) == (-2, 0)
        assert head_action is not None and head_action.type == "MOVE"
        assert add(head.position, head_action.direction) == (-1, 0)
    else:
        assert empty_action.type == "HARVEST"
        assert head_action is not None and head_action.type == "MOVE"
        assert add(head.position, head_action.direction) == (-1, 0)


@pytest.mark.parametrize("blocked_exit", [False, True])
def test_production_clearance_shares_an_exit_without_pushing_cargo_back(blocked_exit):
    strategy = campaign_strategy()
    turn = make_turn(
        resources=core_resource_capacity(5),
        objects=[
            core(),
            unit(2, "WORKER"),
            unit(3, "WORKER", position=(-2, 0), cargo=1),
            unit(4, "RANGER", position=(-2, 0)),
            unit(5, "WORKER", position=(0, -2), cargo=1),
            unit(6, "WORKER", position=(0, -3), cargo=1),
        ],
        obstacles=[(0, 1), (1, 0), (-1, -1), *(([(0, -1)]) if blocked_exit else [])],
    )
    context = context_for(turn)
    context.core_build_priority = strategy._campaign_build_priority(context)
    assert context.core_build_priority

    strategy._prepare_core_departure(context)
    departing, cargo, *_ = turn.workers
    departure = turn.plan.unit_actions[departing.id]
    assert departure.type == "MOVE"
    assert add(departing.position, departure.direction) == (-1, 0)
    strategy._decide_worker(cargo, context)
    action = turn.plan.unit_actions[cargo.id]
    assert action.type == "MOVE"
    destination = add(cargo.position, action.direction)
    assert destination == (-1, 0)
    assert strategy._predicted_friendly_occupancy(destination, context) == 2


def test_unaffordable_emergency_ranger_does_not_reserve_the_delivery_window():
    strategy = campaign_strategy()
    turn = make_turn(
        resources=unit_cost(UnitType.WORKER, 30),
        objects=[
            core(),
            unit(2, "WORKER", cargo=1),
            *[unit(10 + n, "WORKER", position=(n, 10)) for n in range(13)],
            *[unit(100 + n, "RANGER", position=(n, -5)) for n in range(16)],
            unit(900, "RANGER", controlled=False, position=(7, 0)),
            unit(901, "RANGER", controlled=False, position=(7, 1)),
        ],
    )
    context = context_for(turn)
    assert strategy._choose_spawn(turn, turn.resources) is None

    context.core_build_priority = strategy._campaign_build_priority(context)

    assert not context.core_build_priority
    strategy._decide_worker(turn.workers[0], context)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "DEPOSIT"
