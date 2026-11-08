"""Wounded escorts use travel time to reach the Core's arrival service."""

from collections import Counter
from uuid import UUID

import pytest

from arena_hero_bot.geometry import add
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, unit
from .test_migration_cargo import migration_context, moving_core


def strategy_for(memory=None):
    return AggressiveStrategy(
        memory
        or WorldMemory(
            core_home_position=(16, 0), core_layout_version=CORE_LAYOUT_VERSION
        ),
        StrategyConfig(target_workers=16, max_population=None, expedition_mode=True),
    )


@pytest.mark.parametrize(("ranger_id", "worker_id"), [(2, 3), (3, 2)])
def test_funded_patient_precedes_cargo_and_stays_until_arrival(ranger_id, worker_id):
    strategy = strategy_for()
    turn = make_turn(
        resources=1,
        objects=[
            moving_core(),
            unit(ranger_id, "RANGER", hp=1, position=(1, 1)),
            unit(worker_id, "WORKER", cargo=1, position=(1, -1)),
        ],
    )
    strategy.decide(turn)
    action = turn.plan.unit_actions[UUID(int=ranger_id)]
    assert action.type == "MOVE"
    assert add((1, 1), action.direction) == (1, 0)
    worker_action = turn.plan.unit_actions.get(UUID(int=worker_id))
    assert not (
        worker_action
        and worker_action.type == "MOVE"
        and add((1, -1), worker_action.direction) == (1, 0)
    )

    for tick, moving in [(101, True), (102, False)]:
        turn = make_turn(
            tick=tick,
            resources=1,
            objects=[
                moving_core() if moving else core(position=(1, 0)),
                unit(ranger_id, "RANGER", hp=1, position=(1, 0)),
            ],
        )
        strategy.decide(turn)
        action = turn.plan.unit_actions[UUID(int=ranger_id)]
        assert action.type == ("WAIT" if moving else "HEAL")
        assert turn.plan.core_action is None


def test_two_waiting_rangers_catch_up_heal_and_resume_across_restart(tmp_path):
    strategy = strategy_for()
    core_position = (0, 0)
    destination = (1, 0)
    progress = 1
    positions = {2: (-3, 0), 3: (-3, 0)}
    hp = {2: 1, 3: 1}
    healed = []
    departure_after_heal = False
    for tick in range(100, 124):
        turn = make_turn(
            tick=tick,
            resources=10 - len(healed),
            objects=[
                core(
                    position=core_position,
                    state="MOVING" if progress else "NORMAL",
                    move_direction="RIGHT" if progress else None,
                    move_progress=progress if progress else None,
                    move_required_ticks=4 if progress else None,
                    destination=list(destination) if progress else None,
                ),
                *(unit(i, "RANGER", position=positions[i], hp=hp[i]) for i in hp),
            ],
        )
        strategy.decide(turn)
        heals_this_tick = 0
        for ident in positions:
            action = turn.plan.unit_actions.get(UUID(int=ident))
            if tick == 100:
                assert action is not None and action.type == "MOVE"
            if action is None:
                continue
            if action.type == "MOVE":
                positions[ident] = add(positions[ident], action.direction)
            elif action.type == "HEAL":
                assert progress == 0 and positions[ident] == core_position
                assert turn.plan.core_action is None
                hp[ident] = 2
                healed.append(ident)
                heals_this_tick += 1
        assert heals_this_tick <= 1
        if progress:
            progress += 1
            if progress == 4:
                core_position = destination
                progress = 0
        elif turn.plan.core_action and turn.plan.core_action.type == "START_MOVE":
            destination = add(core_position, turn.plan.core_action.direction)
            progress = 1
            departure_after_heal = len(healed) == 2
        occupancy = Counter([core_position, *positions.values()])
        assert max(occupancy.values()) <= 2
        if tick == 104:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = strategy_for(WorldMemory.load(path))
        if departure_after_heal:
            break
    assert sorted(healed) == [2, 3]
    assert departure_after_heal


@pytest.mark.parametrize("blocked_by", ["funds", "enemy", "recent_damage", "rock"])
def test_arrival_healing_keeps_funding_and_safety_checks(blocked_by):
    turn = make_turn(
        resources=0 if blocked_by == "funds" else 1,
        objects=[
            moving_core(),
            unit(2, "RANGER", hp=1, position=(1, 1)),
            *(
                [unit(30, "RANGER", controlled=False, position=(1, -2))]
                if blocked_by == "enemy"
                else []
            ),
        ],
        obstacles=[(1, 0)] if blocked_by == "rock" else [],
    )
    strategy = strategy_for()
    if blocked_by == "recent_damage":
        strategy.memory.incoming_damage[(1, 0)] = (turn.tick, 1)
    context = migration_context(turn)
    strategy._prepare_migration_healing(context)
    assert context.core_arrival_healer_id is None


def test_existing_cargo_arrival_keeps_capacity_while_patient_approaches():
    turn = make_turn(
        resources=1,
        objects=[
            moving_core(),
            unit(2, "RANGER", hp=1, position=(-3, 0)),
            unit(3, "WORKER", cargo=1, position=(1, 0)),
        ],
    )
    strategy_for().decide(turn)
    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "MOVE"
    assert add((-3, 0), action.direction) != (1, 0)
    worker_action = turn.plan.unit_actions.get(UUID(int=3))
    assert worker_action is None or worker_action.type == "WAIT"


def test_arrival_build_priority_keeps_the_service_cell_for_production():
    turn = make_turn(
        resources=100,
        objects=[core(), unit(2, "RANGER", hp=1, position=(1, 0))],
    )
    context = migration_context(turn)
    context.core_build_priority = True
    strategy = strategy_for()
    strategy._prepare_migration_healing(context)
    assert not context.core_migration_healing_ids
    assert context.core_arrival_healer_id is None
    strategy._recover_if_critical(turn.rangers[0], maximum_hp=2, context=context)
    action = turn.plan.unit_actions.get(turn.rangers[0].id)
    assert action is None or action.type == "WAIT"


def test_cargo_approaches_temporarily_reserved_core_route_without_long_detour():
    turn = make_turn(
        objects=[
            moving_core(),
            unit(2, "WORKER", cargo=1, position=(1, 0)),
            unit(3, "WORKER", cargo=1, position=(-1, -2)),
        ],
        obstacles=[
            (-6, -6),
            (-6, -4),
            (-6, 4),
            (-5, -4),
            (-5, -1),
            (-5, 3),
            (-5, 4),
            (-4, -7),
            (-4, 6),
            (-3, -3),
            (-3, 4),
            (-3, 7),
            (-2, -7),
            (-2, -4),
            (-2, -1),
            (-2, 3),
            (-1, -6),
            (-1, -5),
            (-1, 2),
            (0, -7),
            (0, -2),
            (0, 1),
            (1, -3),
            (1, -1),
            (1, 3),
            (1, 5),
            (2, -4),
            (2, 1),
            (3, -7),
            (3, -4),
            (3, 6),
            (4, -6),
            (4, -2),
            (4, 5),
            (5, -3),
        ],
    )
    strategy = strategy_for()
    strategy.memory.observe(turn)
    context = migration_context(turn)
    context.core_arrival_worker_id = turn.workers[0].id
    worker = turn.workers[1]
    strategy._decide_worker(worker, context)
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert add(worker.position, action.direction) == (-1, -1)


def test_cargo_waits_at_a_reserved_arrival_instead_of_walking_around_it():
    turn = make_turn(
        objects=[
            moving_core(),
            unit(2, "WORKER", cargo=1, position=(1, 0)),
            unit(3, "WORKER", cargo=1, position=(1, 1)),
        ],
        obstacles=[(2, 1)],
    )
    strategy = strategy_for()
    strategy.memory.observe(turn)
    context = migration_context(turn)
    context.core_arrival_worker_id = turn.workers[0].id
    worker = turn.workers[1]
    strategy._decide_worker(worker, context)
    action = turn.plan.unit_actions.get(worker.id)
    assert action is None or action.type == "WAIT"
