"""Moving coverage areas and bounded mining, service, and combat exceptions."""

from uuid import UUID

import pytest

from arena_hero_bot.geometry import manhattan
from arena_hero_bot.memory import UnitGoal, WorldMemory
from arena_hero_bot.strategy import (
    CAMPAIGN_PHASE_STAGE2,
    CAMPAIGN_SCOUT_PURPOSE,
    CORE_LAYOUT_VERSION,
    PATROL_RING_PURPOSE,
    RESOURCE_CLAIM_PURPOSE,
    WORKER_PATROL_PURPOSE,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, object_id, unit
from .test_core_traffic import settled_positions
from .test_shared_movement import context_for


def strategy_for(memory, *, campaign=False):
    memory.core_layout_version = CORE_LAYOUT_VERSION
    return AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=16,
            max_population=None,
            expedition_mode=True,
            campaign_mode=campaign,
        ),
    )


@pytest.mark.parametrize("restart", [False, True])
def test_core_step_releases_a_distant_source_claim_without_forgetting_the_mine(
    tmp_path, restart
):
    source, target = (-48, -16), (-16, -16)
    old_mine, near_mine, far_target_mine = (-49, -10), (-36, -10), (-2, -10)
    memory = WorldMemory(
        core_home_position=target,
        core_migration_start_position=source,
        resource_cells={old_mine: 99, far_target_mine: 99},
    )
    strategy = strategy_for(memory)
    worker = unit(2, "WORKER", position=(-44, -10))
    first = make_turn(
        objects=[
            core(
                position=(-33, -16),
                state="MOVING",
                destination=[-32, -16],
                move_direction="RIGHT",
                move_progress=3,
                move_required_ticks=4,
            ),
            worker,
        ]
    )
    strategy.decide(first)
    claim = memory.goal_for(worker["id"])
    assert claim is not None and claim.position == old_mine
    worker["position"] = list(settled_positions(first)[UUID(worker["id"])])
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = strategy_for(memory)

    second = make_turn(
        tick=101,
        objects=[
            core(
                position=(-32, -16),
                state="MOVING",
                destination=[-31, -16],
                move_direction="RIGHT",
                move_progress=0,
                move_required_ticks=4,
            ),
            worker,
        ],
        resource_cells=[near_mine],
    )
    strategy.decide(second)

    claim = memory.goal_for(worker["id"])
    assert claim is not None and claim.position == near_mine
    assert old_mine in memory.resource_cells
    assert far_target_mine in memory.resource_cells
    following = settled_positions(second)[UUID(worker["id"])]
    assert -48 <= following[0] <= -17 and -32 <= following[1] <= -1
    assert following != second.workers[0].position
    assert memory.worker_sector_count == 32


@pytest.mark.parametrize("restart", [False, True])
def test_displaced_idle_worker_replaces_its_old_goal_and_returns(tmp_path, restart):
    memory = WorldMemory(
        core_home_position=(-16, -16),
        core_migration_start_position=(-48, -16),
    )
    memory.set_goal(object_id(2), UnitGoal((-59, -10), 99, WORKER_PATROL_PURPOSE))
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    turn = make_turn(
        objects=[
            core(
                position=(-24, -16),
                state="MOVING",
                destination=[-23, -16],
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
            ),
            unit(2, "WORKER", position=(-56, -10)),
        ]
    )
    strategy_for(memory).decide(turn)

    goal = memory.goal_for(object_id(2))
    assert goal is not None
    assert -40 <= goal.position[0] <= -9 and -32 <= goal.position[1] <= -1
    following = settled_positions(turn)[turn.workers[0].id]
    assert following[0] > turn.workers[0].position[0]


def test_cancelled_home_with_sealed_centers_does_not_restore_global_worker_search():
    memory = WorldMemory(
        campaign_last_core_id=object_id(1),
        campaign_phase=1,
    )
    turn = make_turn(
        objects=[core(), unit(2, "WORKER")],
        obstacles=[(15, 15), (15, 16), (16, 15), (16, 16)],
    )
    strategy_for(memory, campaign=True).decide(turn)

    assert memory.core_home_position is None
    goal = memory.goal_for(object_id(2))
    assert goal is not None and goal.purpose == WORKER_PATROL_PURPOSE
    assert 0 <= goal.position[0] <= 15 and 0 <= goal.position[1] <= 15
    following = settled_positions(turn)[turn.workers[0].id]
    assert 0 <= following[0] <= 15 and 0 <= following[1] <= 15


@pytest.mark.parametrize(
    ("position", "live", "cargo", "allowed"),
    [
        ((1, 12), True, 0, True),
        ((4, 12), True, 0, True),
        ((5, 12), True, 0, False),
        ((1, 12), False, 0, False),
        ((1, 12), True, 1, False),
    ],
)
def test_only_a_nearby_empty_worker_can_claim_a_live_exterior_mine(
    position, live, cargo, allowed
):
    mine = (-2, 12)
    memory = WorldMemory(core_home_position=(16, 16), resource_cells={mine: 99})
    strategy = strategy_for(memory)
    turn = make_turn(
        objects=[
            core(position=(16, 16)),
            unit(2, "WORKER", position=position, cargo=cargo),
        ],
        resource_cells=[mine] if live else [],
    )
    assignments = strategy._assign_resources(turn)

    assert (assignments.get(turn.workers[0].id) == mine) is allowed
    assert mine in memory.resource_cells
    if allowed:
        context = context_for(turn)
        context.resource_assignments = assignments
        strategy._decide_worker(turn.workers[0], context)
        assert settled_positions(turn)[turn.workers[0].id] == (position[0] - 1, 12)


def test_live_six_cell_claim_completes_its_eight_step_rock_detour():
    mine, position = (-2, 12), (4, 12)
    memory = WorldMemory(core_home_position=(16, 16))
    strategy = strategy_for(memory)
    for travelled, tick in enumerate(range(100, 109)):
        turn = make_turn(
            tick=tick,
            objects=[core(position=(16, 16)), unit(2, "WORKER", position=position)],
            resource_cells=[mine],
            obstacles=[(1, 12)],
        )
        memory.observe(turn)
        context = context_for(turn)
        context.resource_assignments = strategy._assign_resources(turn)
        assert context.resource_assignments == {turn.workers[0].id: mine}
        strategy._decide_worker(turn.workers[0], context)
        action = turn.plan.unit_actions[turn.workers[0].id]
        if action.type == "HARVEST":
            assert position == mine
            assert travelled == 8
            return
        assert action.type == "MOVE"
        position = settled_positions(turn)[turn.workers[0].id]
    pytest.fail("a bounded live mining detour was revoked before arrival")


@pytest.mark.parametrize("danger", ["rock_detour", "fighter", "too_far_from_core"])
def test_live_mining_exception_still_requires_a_short_safe_route_near_the_core(danger):
    mine, position = (-2, 12), (1, 12)
    rocks = {(0, y) for y in range(9, 16)} if danger == "rock_detour" else set()
    enemies = (
        [unit(900, "RANGER", controlled=False, position=(-2, 14))]
        if danger == "fighter"
        else []
    )
    if danger == "too_far_from_core":
        mine, position = (-60, 12), (-59, 12)
    memory = WorldMemory(core_home_position=(16, 16), obstacles=rocks)
    strategy = strategy_for(memory)
    turn = make_turn(
        objects=[
            core(position=(16, 16)),
            unit(2, "WORKER", position=position),
            *enemies,
        ],
        resource_cells=[mine],
        obstacles=rocks,
    )
    memory.observe(turn)

    assert strategy._assign_resources(turn) == {}


@pytest.mark.parametrize("sealed", [False, True])
def test_mining_can_take_a_small_edge_detour_but_cannot_search_arbitrarily_far(sealed):
    mine, position = (14, 14), (14, 12)
    rocks = {(x, 13) for x in range(10 if sealed else 14, 50 if sealed else 46)}
    memory = WorldMemory(
        core_home_position=(48, 16),
        core_migration_start_position=(16, 16),
        obstacles=rocks,
    )
    strategy = strategy_for(memory)
    visited = {position}
    for tick in range(100, 104):
        turn = make_turn(
            tick=tick,
            objects=[core(position=(30, 16)), unit(2, "WORKER", position=position)],
            resource_cells=[mine],
            obstacles=rocks,
        )
        context = context_for(turn)
        context.resource_assignments = {turn.workers[0].id: mine}
        moved = strategy._move_worker_resource(
            turn.workers[0], mine, context, reason="approach a mine"
        )
        assert moved is not sealed
        if sealed:
            assert not turn.plan.unit_actions
            assert not context.resource_assignments
            assert mine in strategy._unreachable_claims
            return
        position = settled_positions(turn)[turn.workers[0].id]
        assert 10 <= position[0] <= 49 and -4 <= position[1] <= 35
        visited.add(position)
    assert position == mine
    assert any(x < 14 for x, _y in visited)


def test_shorter_claim_swap_cannot_give_a_local_exception_to_a_distant_worker():
    memory = WorldMemory(core_home_position=(16, 16))
    strategy = strategy_for(memory)
    local_mine, ordinary_mine = (-2, 12), (0, 20)
    turn = make_turn(
        objects=[
            core(position=(16, 16)),
            unit(2, "WORKER", position=(0, 16)),
            unit(3, "WORKER", position=(5, 12)),
        ],
        resource_cells=[local_mine, ordinary_mine],
    )
    assignments = {turn.workers[0].id: local_mine, turn.workers[1].id: ordinary_mine}
    strategy._uncross_claims(turn, assignments)
    assert assignments == {
        turn.workers[0].id: local_mine,
        turn.workers[1].id: ordinary_mine,
    }


def test_underfoot_harvest_does_not_transfer_its_previous_exception_to_a_far_worker():
    underfoot, other_mine = (-1, 12), (-3, 12)
    memory = WorldMemory(core_home_position=(16, 16))
    memory.set_goal(object_id(2), UnitGoal(other_mine, 99, RESOURCE_CLAIM_PURPOSE))
    memory.set_goal(object_id(3), UnitGoal(underfoot, 99, RESOURCE_CLAIM_PURPOSE))
    strategy = strategy_for(memory)
    turn = make_turn(
        objects=[
            core(position=(16, 16)),
            unit(2, "WORKER", position=underfoot),
            unit(3, "WORKER", position=(4, 12)),
        ],
        resource_cells=[underfoot, other_mine],
    )
    assignments = strategy._assign_resources(turn)
    assert assignments == {turn.workers[0].id: underfoot}
    context = context_for(turn)
    context.resource_assignments = assignments
    strategy._decide_worker(turn.workers[0], context)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "HARVEST"


@pytest.mark.parametrize("restart", [False, True])
def test_ranger_source_route_cannot_cut_across_the_union_missing_corner(
    tmp_path, restart
):
    position, goal = (18, 5), (35, 4)
    rocks = {(x, 4) for x in range(18, 32)}
    memory = WorldMemory(
        core_home_position=(48, 16),
        core_migration_start_position=(16, 16),
        obstacles=rocks,
    )
    memory.set_goal(object_id(2), UnitGoal(goal, 100, PATROL_RING_PURPOSE))
    strategy = strategy_for(memory)
    for tick in range(100, 118):
        if restart and tick == 103:
            path = tmp_path / "memory.json"
            memory.save(path)
            memory = WorldMemory.load(path)
            strategy = strategy_for(memory)
        turn = make_turn(
            tick=tick,
            objects=[core(position=(38, 24)), unit(2, "RANGER", position=position)],
            obstacles=rocks,
        )
        strategy._move_or_wait(
            turn.rangers[0], goal, context_for(turn), reason="return to patrol coverage"
        )
        position = settled_positions(turn)[turn.rangers[0].id]
        assert 18 <= position[0] <= 63 and 0 <= position[1] <= 31
        assert position[0] >= 32 or position[1] >= 4
    assert position == goal


@pytest.mark.parametrize("scouting", [False, True])
def test_ranger_can_patrol_the_whole_destination_even_far_from_the_core(scouting):
    position, goal = (60, 3), (60, 27)
    memory = WorldMemory(
        core_home_position=(16, 16) if scouting else (48, 16),
        core_migration_start_position=None if scouting else (16, 16),
        campaign_scout_target=(48, 16) if scouting else None,
    )
    purpose = f"{CAMPAIGN_SCOUT_PURPOSE}-48-16" if scouting else PATROL_RING_PURPOSE
    memory.set_goal(object_id(2), UnitGoal(goal, 100, purpose))
    strategy = strategy_for(memory)
    for tick in range(100, 124):
        turn = make_turn(
            tick=tick,
            objects=[
                core(position=(16, 16) if scouting else (30, 16)),
                unit(2, "RANGER", position=position),
            ],
        )
        strategy._move_or_wait(
            turn.rangers[0], goal, context_for(turn), reason="cover the destination"
        )
        position = settled_positions(turn)[turn.rangers[0].id]
        assert position[0] == 60
    assert position == goal


@pytest.mark.parametrize("service", ["deposit", "heal"])
def test_real_core_services_can_route_outside_the_routine_block(service):
    position = (7, 0)
    memory = WorldMemory(core_home_position=(16, 16))
    strategy = strategy_for(memory)
    member = (
        unit(2, "WORKER", position=position, cargo=1)
        if service == "deposit"
        else unit(2, "RANGER", position=position, hp=1)
    )
    turn = make_turn(
        resources=1,
        objects=[core(position=(16, 16)), member],
        obstacles={(8, y) for y in range(32)},
    )
    context = context_for(turn)
    if service == "deposit":
        strategy._decide_worker(turn.workers[0], context)
    else:
        assert strategy._recover_if_critical(
            turn.rangers[0], maximum_hp=2, context=context
        )
    assert settled_positions(turn)[turn.units[0].id] == (7, -1)


def test_necessary_firing_position_may_leave_the_routine_block():
    memory = WorldMemory(core_home_position=(16, 12))
    strategy = strategy_for(memory)
    turn = make_turn(
        objects=[core(position=(16, 12)), unit(2, "RANGER", position=(0, 12))]
    )
    ranger = turn.rangers[0]
    assert strategy._move_within_leash(
        ranger,
        (-1, 12),
        context_for(turn),
        offensive=False,
        reason="take a needed firing angle",
    )
    assert settled_positions(turn)[ranger.id] == (-1, 12)


def test_core_hunter_can_cross_the_block_edge_while_other_rangers_guard_home():
    memory = WorldMemory(
        core_home_position=(16, 16),
        campaign_last_core_id=object_id(1),
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        unit_roles={
            object_id(2): "defense",
            object_id(3): "defense",
            object_id(4): "patrol-1",
            object_id(5): "patrol-1",
        },
        unit_roles_initialized=True,
    )
    strategy = strategy_for(memory, campaign=True)
    turn = make_turn(
        objects=[
            core(position=(16, 16)),
            unit(2, "RANGER", position=(0, 16)),
            unit(3, "RANGER", position=(16, 19)),
            unit(4, "RANGER", position=(16, 12)),
            unit(5, "RANGER", position=(20, 16)),
            core(900, controlled=False, position=(-4, 16)),
        ]
    )
    report = strategy.decide(turn)
    hunter = turn.rangers[0]
    assert settled_positions(turn)[hunter.id] == (-1, 16)
    hunting = [d for d in report.decisions if "enemy Core" in d.reason]
    assert {d.actor_id for d in hunting} == {str(hunter.id)}
    assert turn.core is not None
    assert (
        sum(manhattan(r.position, turn.core.position) <= 4 for r in turn.rangers[1:])
        >= 2
    )
