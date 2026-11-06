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


def test_shared_waiters_separate_into_the_open_forward_slot():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(-1, 2), cargo=1),
            unit(3, "WORKER", position=(-1, 2), cargo=1),
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
    assert turn.workers[1].id not in turn.plan.unit_actions
