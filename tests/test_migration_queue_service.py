"""Moving cargo queues compact without crossing or stealing production slots."""

from uuid import UUID

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
from .test_campaign import _roster
from .test_core_traffic import settled_positions


def queue_strategy():
    return AggressiveStrategy(
        WorldMemory(
            core_home_position=(16, 0), core_layout_version=CORE_LAYOUT_VERSION
        ),
        StrategyConfig(target_workers=16, max_population=None, expedition_mode=True),
    )


def travelling_core():
    return core(
        position=(16, 16),
        state="MOVING",
        move_direction="UP",
        move_progress=2,
        move_required_ticks=4,
        destination=[16, 15],
    )


@pytest.mark.parametrize("reverse_ids", [False, True])
def test_cargo_matches_waiting_slots_without_trading_approach_lanes(reverse_ids):
    near, far = (3, 2) if reverse_ids else (2, 3)
    turn = make_turn(
        objects=[
            travelling_core(),
            unit(4, "WORKER", position=(16, 15), cargo=1),
            unit(near, "WORKER", position=(19, 14), cargo=1),
            unit(far, "WORKER", position=(19, 13), cargo=1),
        ]
    )
    queue_strategy().decide(turn)
    positions = settled_positions(turn)
    assert positions[UUID(int=near)] == (18, 14)
    assert positions[UUID(int=far)] == (18, 13)


@pytest.mark.parametrize("distant_cargo", [False, True])
def test_on_route_cargo_compacts_into_the_empty_second_slot(distant_cargo):
    turn = make_turn(
        objects=[
            travelling_core(),
            unit(2, "WORKER", position=(16, 15), cargo=1),
            unit(3, "WORKER", position=(16, 13), cargo=1),
            unit(4, "WORKER", position=(16, 12), cargo=1),
            *([unit(5, "WORKER", position=(10, 14), cargo=1)] if distant_cargo else []),
        ]
    )
    queue_strategy().decide(turn)
    positions = settled_positions(turn)
    assert positions[UUID(int=3)] == (16, 14)
    assert positions[UUID(int=4)] == (16, 13)
    assert positions[UUID(int=2)] == (16, 15)


def test_queue_approaches_survive_restart_and_deliver_without_crossing(tmp_path):
    strategy = queue_strategy()
    positions = {2: (19, 13), 3: (19, 14), 4: (16, 15)}
    cargo = dict.fromkeys(positions, 1)
    core_position = (16, 16)
    destination = (16, 15)
    progress = 2
    delivered = []
    for tick in range(100, 125):
        turn = make_turn(
            tick=tick,
            resources=len(delivered),
            objects=[
                core(
                    position=core_position,
                    state="MOVING" if progress else "NORMAL",
                    move_direction="UP" if progress else None,
                    move_progress=progress if progress else None,
                    move_required_ticks=4 if progress else None,
                    destination=list(destination) if progress else None,
                ),
                *(
                    unit(i, "WORKER", position=p, cargo=cargo[i])
                    for i, p in positions.items()
                ),
            ],
        )
        strategy.decide(turn)
        settled = settled_positions(turn)
        for ident, position in positions.items():
            action = turn.plan.unit_actions.get(UUID(int=ident))
            if action and action.type == "DEPOSIT":
                assert not progress and position == core_position
                assert turn.plan.core_action is None
                cargo[ident] = 0
                delivered.append(ident)
        positions = {ident: settled[UUID(int=ident)] for ident in positions}
        if tick <= 102:
            assert positions[2] == (18 - (tick - 100), 13)
            assert positions[3] == (18 - (tick - 100), 14)
        if progress:
            progress += 1
            if progress == 4:
                core_position = destination
                progress = 0
        elif turn.plan.core_action and turn.plan.core_action.type == "START_MOVE":
            destination = add(core_position, turn.plan.core_action.direction)
            progress = 1
        if tick == 100:
            targets = strategy.memory.core_queue_targets.copy()
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
            assert strategy.memory.core_queue_targets == targets
        if len(delivered) == 3:
            break
    assert delivered == [4, 3, 2]


def production_strategy():
    workers, rangers, roles = _roster()
    memory = WorldMemory(
        core_home_position=(24, 16),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=str(UUID(int=1)),
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    return (
        AggressiveStrategy(memory, StrategyConfig(campaign_mode=True)),
        workers,
        rangers,
        core_resource_capacity(len(workers) + len(rangers)),
    )


def test_build_tick_does_not_admit_cargo_crossing_the_core_to_a_forward_slot():
    strategy, workers, rangers, resources = production_strategy()
    workers[0]["position"] = [16, 17]
    workers[0]["cargo"] = 1
    turn = make_turn(
        resources=resources,
        objects=[core(position=(16, 16)), *workers, *rangers],
        obstacles=[(15, 17), (17, 17)],
    )
    strategy.decide(turn)
    assert settled_positions(turn)[UUID(workers[0]["id"])] == (16, 17)
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"


def test_full_moving_core_clears_its_arrival_before_spawning_across_restart(tmp_path):
    strategy, workers, rangers, resources = production_strategy()
    workers[0]["position"] = [17, 16]
    workers[0]["cargo"] = 1
    turn = make_turn(
        resources=resources,
        objects=[
            core(
                position=(16, 16),
                state="MOVING",
                move_direction="RIGHT",
                move_progress=3,
                move_required_ticks=4,
                destination=[17, 16],
            ),
            *workers,
            *rangers,
        ],
    )
    strategy.decide(turn)
    positions = settled_positions(turn)
    assert (17, 16) not in positions.values()
    assert turn.plan.core_action is None
    for member in [*workers, *rangers]:
        member["position"] = list(positions[UUID(member["id"])])
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    restarted = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    arrived = make_turn(
        tick=101,
        resources=resources,
        objects=[core(position=(17, 16)), *workers, *rangers],
    )
    restarted.decide(arrived)
    assert (17, 16) not in settled_positions(arrived).values()
    assert arrived.plan.core_action is not None
    assert arrived.plan.core_action.type == "SPAWN"


def test_production_arrival_still_allows_a_wounded_escort_to_approach():
    strategy, workers, rangers, resources = production_strategy()
    rangers[0]["position"] = [14, 17]
    rangers[0]["hp"] = 1
    turn = make_turn(
        resources=resources,
        objects=[
            core(
                position=(16, 16),
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[17, 16],
            ),
            *workers,
            *rangers,
        ],
    )
    strategy.decide(turn)
    positions = settled_positions(turn)
    assert positions[UUID(rangers[0]["id"])] == (15, 17)
    assert (17, 16) not in positions.values()


def test_clearing_core_for_production_blocks_replacement_then_spawns(tmp_path):
    strategy, workers, rangers, resources = production_strategy()
    workers[0]["position"] = [16, 16]
    workers[1]["position"] = [15, 16]
    workers[1]["cargo"] = 1
    turn = make_turn(
        resources=resources,
        objects=[core(position=(16, 16)), *workers, *rangers],
    )
    report = strategy.decide(turn)
    assert (16, 16) not in settled_positions(turn).values()
    assert turn.plan.core_action is None
    assert any(d.actor_kind == "CORE" and d.action == "WAIT" for d in report.decisions)
    for member in [*workers, *rangers]:
        member["position"] = list(settled_positions(turn)[UUID(member["id"])])
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    cleared = make_turn(
        tick=101,
        resources=resources,
        objects=[core(position=(16, 16)), *workers, *rangers],
    )
    strategy.decide(cleared)
    assert (16, 16) not in settled_positions(cleared).values()
    assert cleared.plan.core_action is not None
    assert cleared.plan.core_action.type == "SPAWN"
    for member in [*workers, *rangers]:
        member["position"] = list(settled_positions(cleared)[UUID(member["id"])])
    resumed = make_turn(
        tick=102,
        resources=resources - unit_cost(UnitType.RANGER, len(workers) + len(rangers)),
        objects=[
            core(position=(16, 16)),
            *workers,
            *rangers,
            unit(2000, "RANGER", position=(16, 16)),
        ],
    )
    strategy.decide(resumed)
    settled_positions(resumed)
    assert resumed.plan.core_action is not None
    assert resumed.plan.core_action.type == "START_MOVE"
