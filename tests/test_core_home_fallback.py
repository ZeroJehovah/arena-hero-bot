"""Migration stops at the closest reachable ground to a block's midpoint."""

from __future__ import annotations

import json

import pytest
from arena_hero import UnitType

from arena_hero_bot import migration_cli
from arena_hero_bot.geometry import add, next_step
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.migration import (
    campaign_stop_ring,
    cardinal_block_neighbors,
    load_migration_request,
    plan_next_resource_tier,
    plan_resource_cap,
    resource_block_index,
    resource_block_ring,
    write_migration_request,
)
from arena_hero_bot.strategy import (
    CAMPAIGN_PHASE_STAGE1,
    CAMPAIGN_PHASE_STAGE4,
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    _resource_block_center_cells,
    _resource_block_home_cells,
)

from .factories import core, make_turn, unit
from .test_campaign import _gathered_roster, _strategy


@pytest.fixture(autouse=True)
def require_complete_fallback_plans(monkeypatch):
    def deadline_reached(*_args):
        pytest.fail("fallback scenario exhausted planning time before completion")

    monkeypatch.setattr(AggressiveStrategy, "_wait_unplanned_units", deadline_reached)


def _closest_outer_cells(position, rings=1):
    """Eight nearest cells outside a centered square of permanent rocks."""
    x, y = _resource_block_center_cells(position)[0]
    return {
        (x - rings, y),
        (x - rings, y + 1),
        (x + 1 + rings, y),
        (x + 1 + rings, y + 1),
        (x, y - rings),
        (x + 1, y - rings),
        (x, y + 1 + rings),
        (x + 1, y + 1 + rings),
    }


def _plan(memory, origin, planner):
    if planner == "next-tier":
        return plan_next_resource_tier(memory, current_position=origin)
    stop_ring = campaign_stop_ring(6)
    assert stop_ring is not None
    return plan_resource_cap(
        memory,
        current_position=origin,
        target_quota=6,
        stop_ring=stop_ring,
    )


@pytest.mark.parametrize("origin", [(16, 527), (-16, 527), (16, -528), (-16, -528)])
@pytest.mark.parametrize("planner", ["cap", "next-tier"])
def test_inward_hop_uses_the_closest_reachable_cells_when_the_midpoint_is_rock(
    origin, planner
):
    current = resource_block_index(origin)
    inward = min(cardinal_block_neighbors(current), key=resource_block_ring)
    target_block_origin = inward[0] * 32, inward[1] * 32
    centers = set(_resource_block_center_cells(target_block_origin))
    memory = WorldMemory(core_home_position=origin, obstacles=centers)

    plan = _plan(memory, origin, planner)

    assert plan is not None
    assert plan.target_block == inward
    assert plan.target in _closest_outer_cells(target_block_origin)
    assert plan.target not in memory.obstacles
    assert (
        next_step(origin, plan.target, blocked=centers, require_path=True) is not None
    )


@pytest.mark.parametrize("planner", ["cap", "next-tier"])
def test_an_open_central_cell_is_preferred_to_the_outer_ring(planner):
    origin = (16, 527)
    centers = _resource_block_center_cells((0, 480))
    remaining = centers[0]
    memory = WorldMemory(
        core_home_position=origin,
        obstacles=set(centers[1:]),
        resource_cells={remaining: 100},
    )

    plan = _plan(memory, origin, planner)

    assert plan is not None
    # A resource may be harvested; it must not expand the permanent-rock fallback.
    assert plan.target == remaining


@pytest.mark.parametrize("planner", ["cap", "next-tier"])
@pytest.mark.parametrize("rings", [2, 5, 10])
def test_center_search_continues_beyond_one_ring(planner, rings):
    origin = (16, 527)
    obstacles = {
        (x, y)
        for x in range(16 - rings, 16 + rings)
        for y in range(496 - rings, 496 + rings)
    }
    memory = WorldMemory(core_home_position=origin, obstacles=obstacles)

    plan = _plan(memory, origin, planner)

    assert plan is not None
    assert plan.target in _closest_outer_cells((0, 480), rings)
    assert plan.target not in obstacles


@pytest.mark.parametrize("planner", ["cap", "next-tier"])
def test_unreachable_free_center_pocket_is_skipped_before_ranking_the_stop(planner):
    origin = (16, 527)
    obstacles = {
        (x, y)
        for x in range(14, 18)
        for y in range(494, 498)
        if x in (14, 17) or y in (494, 497)
    }
    memory = WorldMemory(core_home_position=origin, obstacles=obstacles)

    plan = _plan(memory, origin, planner)

    assert plan is not None
    assert plan.target in _closest_outer_cells((0, 480), 2)
    assert set(_resource_block_center_cells(plan.target)).isdisjoint(obstacles)
    assert next_step(origin, plan.target, blocked=obstacles, require_path=True)


def test_only_reachable_member_of_an_equal_distance_center_tie_is_retained():
    origin = (16, 527)
    obstacles = {(14, 495), (16, 495), (15, 494), (15, 496)}

    assert _resource_block_home_cells((0, 480), origin=origin, obstacles=obstacles) == (
        (16, 496),
    )


@pytest.mark.parametrize("planner", ["cap", "next-tier"])
def test_physical_center_distance_prefers_near_diagonals_to_farther_axis_cells(planner):
    origin = (16, 527)
    memory = WorldMemory(
        core_home_position=origin,
        obstacles=set(_resource_block_center_cells((0, 480)))
        | _closest_outer_cells((0, 480)),
    )

    plan = _plan(memory, origin, planner)

    assert plan is not None
    assert plan.target in {(14, 494), (17, 494), (14, 497), (17, 497)}


@pytest.mark.parametrize("planner", ["cap", "next-tier"])
def test_an_off_center_boundary_cell_can_be_the_only_reachable_center(
    planner, tmp_path
):
    origin = (16, 527)
    destination = (0, 511)
    memory = WorldMemory(
        core_home_position=origin,
        obstacles={
            (x, y) for x in range(32) for y in range(480, 512) if (x, y) != destination
        },
    )

    plan = _plan(memory, origin, planner)

    assert plan is not None
    assert plan.target == destination
    assert plan.target_block == (0, 15)
    path = tmp_path / "core-migration.json"
    written = write_migration_request(path, plan.target, memory=memory)
    assert load_migration_request(path, memory=memory) == written


@pytest.mark.parametrize("planner", ["cap", "next-tier"])
def test_a_wholly_unreachable_block_does_not_publish_a_lateral_hop(planner):
    origin = (16, 527)
    memory = WorldMemory(
        core_home_position=origin,
        obstacles={
            (x, y)
            for x in range(32)
            for y in range(480, 512)
            if x in (0, 31) or y in (480, 511)
        },
    )

    assert _plan(memory, origin, planner) is None


@pytest.mark.parametrize("target", [(14, 15), (17, 16), (-370, 431)])
def test_terrain_adjusted_center_request_round_trip_preserves_its_exact_target(
    target, tmp_path
):
    path = tmp_path / "core-migration.json"
    memory = WorldMemory(obstacles=set(_resource_block_center_cells(target)))

    request = write_migration_request(path, target, memory=memory)
    loaded = load_migration_request(path, memory=memory)

    assert loaded == request
    assert loaded.target == target
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("target", [(14, 14), (13, 15), (18, 16), (0, 0)])
def test_migration_requests_reject_reachable_cells_farther_from_the_midpoint(
    target, tmp_path
):
    memory = WorldMemory(obstacles=set(_resource_block_center_cells((0, 0))))
    with pytest.raises(ValueError, match="center cells"):
        write_migration_request(tmp_path / "core-migration.json", target, memory=memory)


def test_operator_migration_accepts_a_terrain_adjusted_center():
    memory = WorldMemory(
        core_home_position=(16, 527),
        obstacles=set(_resource_block_center_cells((0, 480))),
    )
    strategy = _strategy(memory)

    assert strategy.request_core_migration((14, 495)) is True
    assert memory.core_home_position == (14, 495)
    assert strategy.request_core_migration((14, 495)) is False


@pytest.mark.parametrize("command", ["request-next-tier", "request"])
def test_migration_cli_uses_terrain_for_requests_and_status(command, tmp_path, capsys):
    memory = WorldMemory(
        core_home_position=(16, 527),
        obstacles=set(_resource_block_center_cells((0, 480))),
    )
    memory.save(tmp_path / "memory.json")
    arguments = ["--data-dir", str(tmp_path), command]
    if command == "request":
        arguments.extend(["14", "495"])

    migration_cli.main(arguments)
    requested = json.loads(capsys.readouterr().out)
    migration_cli.main(["--data-dir", str(tmp_path), "status"])
    status = json.loads(capsys.readouterr().out)

    assert tuple(requested["target"]) in _closest_outer_cells((0, 480))
    assert status["requested_target"] == requested["target"]


@pytest.mark.parametrize("crossed_boundary", [False, True])
def test_new_rocks_repair_the_existing_hop_inside_its_target_block(
    crossed_boundary, tmp_path
):
    origin = (16, 527)
    position = (16, 510) if crossed_boundary else (16, 518)
    old_target = (16, 495)
    centers = set(_resource_block_center_cells(old_target))
    core_object = core(position=position)
    workers, rangers, roles = _gathered_roster(50, center=position)
    memory = WorldMemory(
        core_home_position=old_target,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_start_position=origin,
        core_migration_last_position=position,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=6,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    strategy = _strategy(memory)
    turn = make_turn(objects=[core_object, *workers, *rangers], obstacles=centers)

    strategy.decide(turn)

    target = memory.core_home_position
    assert target in _closest_outer_cells(old_target)
    assert memory.core_migration_start_position == origin
    assert memory.campaign_scout_target is None
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    core_object["position"] = list(add(position, turn.plan.core_action.direction))
    path = tmp_path / "memory.json"
    memory.save(path)
    resumed = _strategy(WorldMemory.load(path))
    resumed.decide(
        make_turn(
            tick=104, objects=[core_object, *workers, *rangers], obstacles=centers
        )
    )
    assert resumed.memory.core_home_position == target
    assert resumed.memory.core_migration_start_position == origin


@pytest.mark.parametrize("restart", [False, True])
def test_campaign_scout_publishes_and_preserves_an_outer_ring_hop(restart, tmp_path):
    origin = (16, 527)
    centers = set(_resource_block_center_cells((0, 480)))
    core_object = core(position=origin)
    workers, rangers, roles = _gathered_roster(50, center=origin)
    memory = WorldMemory(
        core_home_position=origin,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=origin,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=6,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    strategy = _strategy(memory)
    turn = make_turn(objects=[core_object, *workers, *rangers], obstacles=centers)

    strategy.decide(turn)

    target = memory.campaign_scout_target
    assert target in _closest_outer_cells((0, 480))
    assert memory.core_home_position == origin
    assert turn.plan.core_action is None
    memory.campaign_scout_seen_ticks.update(
        {(x, y): turn.tick for x in range(32) for y in range(480, 512)}
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = _strategy(memory)

    cleared = make_turn(
        tick=101, objects=[core_object, *workers, *rangers], obstacles=centers
    )
    strategy.decide(cleared)

    assert memory.core_home_position == target
    assert memory.campaign_scout_target is None
    assert memory.core_migration_start_position == origin
    assert cleared.plan.core_action is not None
    assert cleared.plan.core_action.type == "START_MOVE"
    core_object["position"] = list(add(origin, cleared.plan.core_action.direction))
    if restart:
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = _strategy(memory)
    moving = make_turn(
        tick=105, objects=[core_object, *workers, *rangers], obstacles=centers
    )
    strategy.decide(moving)
    assert memory.core_home_position == target
    assert memory.core_migration_start_position == origin
    assert moving.plan.core_action is not None
    assert moving.plan.core_action.type == "START_MOVE"


@pytest.mark.parametrize("restart", [False, True])
def test_outer_ring_arrival_serves_cargo_then_allows_the_next_scout(restart, tmp_path):
    home = (14, 495)
    centers = set(_resource_block_center_cells(home))
    core_object = core(position=home)
    workers, rangers, roles = _gathered_roster(50, center=home)
    workers[0]["position"] = list(home)
    workers[0]["cargo"] = 1
    memory = WorldMemory(
        core_home_position=home,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=home,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=6,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    strategy = _strategy(memory)
    turn = make_turn(objects=[core_object, *workers, *rangers], obstacles=centers)

    report = strategy.decide(turn)

    assert memory.core_home_position == home
    assert memory.campaign_resting is True
    assert memory.campaign_scout_target is None
    delivery = next(d for d in report.decisions if d.actor_id == workers[0]["id"])
    assert delivery.action == "DEPOSIT"
    activity = strategy._worker_activity_area(turn)
    assert activity is not None
    assert (activity.min_x, activity.max_x, activity.min_y, activity.max_y) == (
        0,
        31,
        480,
        511,
    )
    workers[0]["cargo"] = 0
    if restart:
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = _strategy(memory)
    serviced = make_turn(
        tick=101,
        resources=1,
        objects=[core_object, *workers, *rangers],
        obstacles=centers,
    )
    strategy.decide(serviced)
    assert memory.core_home_position == home
    assert memory.campaign_resting is False
    assert memory.campaign_scout_target is not None
    assert resource_block_index(memory.campaign_scout_target) == (0, 14)


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("ready", [False, True])
def test_scout_repairs_its_center_without_losing_coverage_or_restarting_the_budget(
    restart, ready, tmp_path
):
    origin, old_target = (16, 527), (16, 495)
    centers = set(_resource_block_center_cells(old_target))
    core_object = core(position=origin)
    workers, rangers, roles = _gathered_roster(50, center=origin)
    memory = WorldMemory(
        core_home_position=origin,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=origin,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=6,
        campaign_last_core_id=core_object["id"],
        campaign_scout_target=old_target,
        campaign_scout_started_tick=98,
        campaign_scout_last_gain_tick=99,
        campaign_scout_seen_ticks=(
            {(x, y): 99 for x in range(32) for y in range(480, 512)}
            if ready
            else {(0, 480): 99}
        ),
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    turn = make_turn(objects=[core_object, *workers, *rangers], obstacles=centers)

    _strategy(memory).decide(turn)

    if ready:
        assert memory.campaign_scout_target is None
        assert memory.core_home_position in _closest_outer_cells(old_target)
        assert memory.core_migration_start_position == origin
        assert turn.plan.core_action is not None
        assert turn.plan.core_action.type == "START_MOVE"
    else:
        assert memory.core_home_position == origin
        assert memory.campaign_scout_target in _closest_outer_cells(old_target)
        assert memory.campaign_scout_started_tick == 98
        assert memory.campaign_scout_seen_ticks[(0, 480)] == 99
        assert turn.plan.core_action is None


def test_a_new_core_with_rock_centers_walks_to_its_own_outer_ring():
    origin = (20, 20)
    centers = set(_resource_block_center_cells(origin))
    memory = WorldMemory()
    turn = make_turn(
        objects=[core(position=origin), unit(100, UnitType.WORKER, position=(24, 24))],
        obstacles=centers,
    )

    _strategy(memory).decide(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE1
    assert memory.core_home_position is not None
    assert memory.core_home_position in _closest_outer_cells(origin)
    assert resource_block_index(memory.core_home_position) == (0, 0)
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"


def test_scout_cancels_a_wholly_unreachable_block_and_keeps_the_current_home():
    origin, target = (16, 527), (16, 495)
    core_object = core(position=origin)
    workers, rangers, roles = _gathered_roster(50, center=origin)
    memory = WorldMemory(
        core_home_position=origin,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=origin,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=6,
        campaign_last_core_id=core_object["id"],
        campaign_scout_target=target,
        campaign_scout_started_tick=98,
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(
        objects=[core_object, *workers, *rangers],
        obstacles={
            (x, y)
            for x in range(32)
            for y in range(480, 512)
            if x in (0, 31) or y in (480, 511)
        },
    )

    _strategy(memory).decide(turn)

    assert memory.core_home_position == origin
    assert memory.campaign_scout_target is None
    assert memory.campaign_scout_started_tick == 0
    assert turn.plan.core_action is None
