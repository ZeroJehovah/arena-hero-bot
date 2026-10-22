"""Single-chunk Worker sightline patrol and chunk-edge combat coverage."""

from __future__ import annotations

import json
from uuid import UUID

from arena_hero import Direction

from arena_hero_bot.exploration import visible_cells
from arena_hero_bot.geometry import manhattan
from arena_hero_bot.memory import UnitGoal, WorldMemory
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    PATROL_RING_PURPOSE,
    RESOURCE_CLAIM_PURPOSE,
    WORKER_PATROL_PURPOSE,
    AggressiveStrategy,
    StrategyConfig,
    _resource_block_cells,
    _ring_slot_index,
    _single_chunk_outer_ring,
    _single_chunk_patrol_sectors,
    _single_chunk_resource_block,
    _worker_patrol_sector_index,
)

from .factories import core, make_turn, object_id, unit

HOME = (-400, 655)


def _strategy(memory: WorldMemory | None = None) -> AggressiveStrategy:
    memory = memory or WorldMemory(core_home_position=HOME)
    if memory.core_home_position is not None:
        memory.core_layout_version = CORE_LAYOUT_VERSION
    return AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=16,
            max_population=None,
            expedition_mode=True,
        ),
    )


def _workers() -> list[dict]:
    return [
        unit(
            300 + index,
            "WORKER",
            position=(HOME[0] - 8 + index % 8, HOME[1] - 8 + index // 8),
        )
        for index in range(16)
    ]


def test_patrol_sectors_tile_one_centered_chunk_into_sixteen_cells() -> None:
    chunk = _single_chunk_resource_block(HOME)
    sectors = _single_chunk_patrol_sectors(HOME)

    assert (chunk.min_x, chunk.max_x, chunk.min_y, chunk.max_y) == (
        -416,
        -385,
        640,
        671,
    )
    assert [sector.index for sector in sectors] == list(range(16))
    first = (sectors[0].index, sectors[0].min_x, sectors[0].max_x)
    assert first == (0, -416, -409)
    assert (sectors[5].index, sectors[5].min_x, sectors[5].min_y) == (
        5,
        -408,
        648,
    )
    assert (sectors[15].index, sectors[15].max_x, sectors[15].max_y) == (
        15,
        -385,
        671,
    )
    assert len({(sector.min_x, sector.min_y) for sector in sectors}) == 16


def test_home_selection_chooses_the_least_blocked_nearby_chunk_center() -> None:
    strategy = _strategy(WorldMemory())
    obstacles = {
        (chunk_x * 32 + 8, chunk_y * 32 + 8)
        for chunk_x in range(-1, 2)
        for chunk_y in range(-1, 2)
        if (chunk_x, chunk_y) != (-1, -1)
    }

    home = strategy._symmetric_home_candidate_from((0, 0), obstacles, set(), set())

    assert home in {(-17, -17), (-16, -17), (-17, -16), (-16, -16)}


def test_outer_ring_is_a_single_124_cell_clockwise_chunk_edge() -> None:
    route = _single_chunk_outer_ring(HOME, set())

    assert len(route) == 124
    assert len(set(route)) == 124
    assert route[0] == (-416, 671)
    assert route[31] == (-385, 671)
    assert route[32] == (-385, 670)
    assert route[62] == (-385, 640)
    assert route[63] == (-386, 640)
    assert route[-1] == (-416, 670)


def test_worker_sectors_are_one_each_and_survive_replacements() -> None:
    strategy = _strategy()
    workers = _workers()
    first = make_turn(objects=[core(position=HOME), *workers])

    strategy._assign_resources(first)

    sectors = strategy.memory.worker_sectors
    assert sorted(sectors.values()) == list(range(16))
    survivor_sectors = dict(sectors)
    dead_id = next(worker["id"] for worker in workers if sectors[worker["id"]] == 2)
    survivors = [worker for worker in workers if worker["id"] != dead_id]
    replacement = unit(999, "WORKER", position=(HOME[0] - 12, HOME[1] + 4))
    second = make_turn(
        tick=101,
        objects=[core(position=HOME), *survivors, replacement],
    )

    strategy._assign_resources(second)

    updated = strategy.memory.worker_sectors
    for worker in survivors:
        assert updated[worker["id"]] == survivor_sectors[worker["id"]]
    assert updated[replacement["id"]] == 2
    assert sorted(updated.values()) == list(range(16))


def test_legacy_worker_blocks_migrate_to_fresh_patrol_sectors(tmp_path) -> None:
    workers = _workers()
    memory = WorldMemory(core_home_position=HOME)
    memory.worker_sectors = {
        worker["id"]: index % 4 for index, worker in enumerate(workers)
    }
    path = tmp_path / "memory.json"
    memory.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["worker_blocks"] = raw.pop("worker_sectors")
    raw["worker_sector_layout_version"] = 0
    path.write_text(json.dumps(raw), encoding="utf-8")

    restored = WorldMemory.load(path)
    strategy = _strategy(restored)
    turn = make_turn(objects=[core(position=HOME), *workers])
    strategy._reconcile_worker_sectors(turn)

    assert restored.worker_sector_layout_version == 1
    assert sorted(restored.worker_sectors.values()) == list(range(16))


def test_workers_claim_any_resource_inside_the_chunk_but_none_outside() -> None:
    strategy = _strategy()
    workers = _workers()
    chunk = _single_chunk_resource_block(HOME)
    resources = [
        (chunk.min_x + 1, chunk.min_y + 1),
        (chunk.max_x - 1, chunk.min_y + 1),
        (chunk.min_x + 1, chunk.max_y - 1),
        (chunk.max_x - 1, chunk.max_y - 1),
        (100, 100),
    ]
    turn = make_turn(
        objects=[core(position=HOME), *workers],
        resource_cells=resources,
    )

    assignments = strategy._assign_resources(turn)

    assert len(assignments) == 4
    assert all(chunk.contains(resource) for resource in assignments.values())
    assert (100, 100) not in assignments.values()


def test_underfoot_resource_overrides_a_held_cross_cell_claim() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    worker = unit(301, "WORKER", position=(HOME[0] - 10, HOME[1] - 10))
    other = (HOME[0] - 11, HOME[1] - 10)
    memory.set_goal(
        worker["id"],
        UnitGoal(
            position=other,
            assigned_tick=100,
            purpose=RESOURCE_CLAIM_PURPOSE,
        ),
    )
    turn = make_turn(
        objects=[core(position=HOME), worker],
        resource_cells=[tuple(worker["position"]), other],
    )

    assignments = strategy._assign_resources(turn)

    assert assignments[UUID(worker["id"])] == tuple(worker["position"])


def test_underfoot_resource_in_another_patrol_sector_is_claimed() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    foreign_sector = _single_chunk_patrol_sectors(HOME)[15]
    worker = unit(
        301,
        "WORKER",
        position=(foreign_sector.min_x + 2, foreign_sector.min_y + 3),
    )
    memory.worker_sector_layout_version = 1
    memory.worker_sectors[worker["id"]] = 0
    turn = make_turn(
        objects=[core(position=HOME), worker],
        resource_cells=[tuple(worker["position"])],
    )

    assignments = strategy._assign_resources(turn)

    assert assignments == {UUID(worker["id"]): tuple(worker["position"])}
    assert memory.worker_sectors[worker["id"]] == 0


def test_worker_patrol_chooses_stale_sightline_and_persists_coverage(
    tmp_path,
) -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    worker = unit(301, "WORKER", position=(HOME[0] - 10, HOME[1] - 10))
    first = make_turn(tick=100, objects=[core(position=HOME), worker])
    strategy.decide(first)
    chunk = _single_chunk_resource_block(HOME)
    target = (chunk.max_x - 3, chunk.max_y - 3)
    target_view = visible_cells(target, 3, set()) & set(_resource_block_cells(chunk))
    for x in range(chunk.min_x, chunk.max_x + 1):
        for y in range(chunk.min_y, chunk.max_y + 1):
            memory.worker_seen_ticks[(x, y)] = 181
    for position in target_view:
        memory.worker_seen_ticks[position] = 5

    second = make_turn(tick=182, objects=[core(position=HOME), worker])
    strategy.decide(second)

    goal = memory.goal_for(worker["id"])
    assert goal is not None
    assert goal.purpose == WORKER_PATROL_PURPOSE
    assert goal.position in target_view
    assert memory.worker_seen_ticks[tuple(worker["position"])] == 182
    path = tmp_path / "memory.json"
    memory.save(path)
    restored = WorldMemory.load(path)
    assert restored.worker_sectors == memory.worker_sectors
    assert restored.worker_seen_ticks[target] == 5


def test_visible_resource_interrupts_worker_block_patrol() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    worker = unit(301, "WORKER", position=(HOME[0] - 12, HOME[1] - 12))
    peaceful = make_turn(objects=[core(position=HOME), worker])
    strategy.decide(peaceful)
    patrol_goal = memory.goal_for(worker["id"])
    assert patrol_goal is not None
    assert patrol_goal.purpose == WORKER_PATROL_PURPOSE

    resource = (HOME[0] - 13, HOME[1] - 12)
    interrupted = make_turn(
        tick=101,
        objects=[core(position=HOME), worker],
        resource_cells=[resource],
    )
    strategy.decide(interrupted)

    claim = memory.goal_for(worker["id"])
    assert claim is not None
    assert claim.purpose == RESOURCE_CLAIM_PURPOSE
    assert claim.position == resource


def test_empty_campaign_chunk_keeps_all_workers_refreshing_the_chunk() -> None:
    core_object = core(1, position=HOME)
    memory = WorldMemory(
        core_home_position=HOME,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=16,
            max_population=None,
            expedition_mode=True,
            campaign_mode=True,
        ),
    )
    turn = make_turn(
        objects=[
            core_object,
            unit(2, "WORKER", position=(HOME[0] - 8, HOME[1] - 8)),
            unit(3, "WORKER", position=(HOME[0] - 7, HOME[1] - 8)),
            unit(4, "WORKER", position=(HOME[0] - 6, HOME[1] - 8)),
        ],
    )

    report = strategy.decide(turn)

    scout = next(item for item in report.decisions if item.actor_id == object_id(2))
    patrol = [item for item in report.decisions if item.actor_id != object_id(2)]
    assert scout.action == "MOVE"
    assert scout.reason == (
        "refresh least-seen coverage in the resource chunk after the local "
        "resource pool is exhausted"
    )
    assert scout.target is not None
    assert _single_chunk_resource_block(HOME).contains(tuple(scout.target))
    assert all(
        item.reason == "patrol least-seen coverage in the resource chunk"
        for item in patrol
    )


def test_campaign_chunk_refresh_brings_a_remote_scout_back_home() -> None:
    core_object = core(1, position=HOME)
    memory = WorldMemory(
        core_home_position=HOME,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=16,
            max_population=None,
            expedition_mode=True,
            campaign_mode=True,
        ),
    )
    worker = unit(2, "WORKER", position=(HOME[0] + 700, HOME[1]))
    memory.set_goal(
        worker["id"],
        UnitGoal(
            position=(HOME[0] + 724, HOME[1]),
            assigned_tick=50,
            purpose="explore-center-v3",
            last_progress_position=(HOME[0] + 701, HOME[1]),
        ),
    )
    turn = make_turn(tick=100, objects=[core_object, worker])

    report = strategy.decide(turn)

    decision = next(item for item in report.decisions if item.actor_id == worker["id"])
    goal = memory.goal_for(worker["id"])
    assert decision.reason == (
        "refresh least-seen coverage in the resource chunk after the local "
        "resource pool is exhausted"
    )
    assert goal is not None
    assert _single_chunk_resource_block(HOME).contains(goal.position)


def test_worker_routes_out_of_a_remembered_core_exclusion_pocket() -> None:
    core_object = core(1, position=HOME)
    memory = WorldMemory(
        core_home_position=HOME,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=16,
            max_population=None,
            expedition_mode=True,
            campaign_mode=True,
            worker_threat_radius=6,
        ),
    )
    worker_position = (HOME[0] - 8, HOME[1])
    enemy_core = core(
        3,
        controlled=False,
        owner_username="rival",
        position=(HOME[0] - 15, HOME[1]),
    )
    observed = make_turn(
        tick=100,
        objects=[
            core_object,
            unit(2, "WORKER", position=worker_position),
            enemy_core,
        ],
    )
    strategy.decide(observed)

    pocket = make_turn(
        tick=101,
        objects=[
            core_object,
            unit(2, "WORKER", position=worker_position),
        ],
        obstacles=[
            (worker_position[0], worker_position[1] - 1),
            (worker_position[0], worker_position[1] + 1),
            (worker_position[0] + 1, worker_position[1]),
        ],
    )
    report = strategy.decide(pocket)

    action = pocket.plan.unit_actions[UUID(object_id(2))]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert next(
        item for item in report.decisions if item.actor_id == object_id(2)
    ).reason == (
        "refresh least-seen coverage in the resource chunk after the local "
        "resource pool is exhausted"
    )


def test_ring_slots_are_eight_flagship_phases_and_move_clockwise_together() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    vanguards = [
        unit(100 + number, "VANGUARD", position=(HOME[0] - 1, HOME[1] - 1))
        for number in range(12)
    ]
    rangers = [
        unit(200 + number, "RANGER", position=(HOME[0] - 1, HOME[1] - 1))
        for number in range(24)
    ]
    first = make_turn(
        tick=200,
        objects=[core(position=HOME), *vanguards, *rangers],
    )
    strategy._reconcile_unit_roles(first)
    slots = strategy._reconcile_patrol_ring_slots(first)

    assert len(slots) == 8
    assert sorted(slots.values()) == list(range(8))
    route = _single_chunk_outer_ring(HOME, set())
    indices = sorted(_ring_slot_index(slot, 8, len(route)) for slot in slots.values())
    gaps = [
        (right - left) % len(route)
        for left, right in zip(indices, indices[1:] + indices[:1], strict=True)
    ]
    assert min(gaps) == 15
    assert max(gaps) == 16

    patrol_ids = strategy._patrol_ids(first)
    targets = {
        UUID(member["id"]): route[
            _ring_slot_index(slots[UUID(member["id"])], 8, len(route))
        ]
        for member in (*rangers, *vanguards)
        if UUID(member["id"]) in patrol_ids
    }
    second_objects = [core(position=HOME)]
    for item in (*vanguards, *rangers):
        copied = dict(item)
        if UUID(item["id"]) in patrol_ids:
            copied["position"] = list(targets[UUID(item["id"])])
        second_objects.append(copied)
    second = make_turn(tick=201, objects=second_objects)

    goals = {
        member.id: strategy._combat_patrol_goal(member, second)[0]
        for member in (*second.rangers, *second.vanguards)
        if member.id in patrol_ids
    }

    assert memory.patrol_ring_phase == 1
    for member_id, goal in goals.items():
        stored = strategy.memory.goal_for(str(member_id))
        assert stored is not None
        assert stored.purpose == PATROL_RING_PURPOSE
        expected = route[
            (_ring_slot_index(slots[member_id], 8, len(route)) + 1) % len(route)
        ]
        assert goal == expected


def test_all_worker_patrol_targets_are_unique_and_cover_every_sector() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    workers = _workers()
    turn = make_turn(objects=[core(position=HOME), *workers])

    strategy.decide(turn)

    chunk = _single_chunk_resource_block(HOME)
    targets: dict[str, tuple[int, int]] = {}
    for worker in workers:
        goal = memory.goal_for(worker["id"])
        assert goal is not None
        targets[worker["id"]] = goal.position
    assert len(set(targets.values())) == len(targets)
    assert all(chunk.contains(target) for target in targets.values())
    assert sorted(
        _worker_patrol_sector_index(chunk, target) for target in targets.values()
    ) == list(range(16))
    for worker_id in targets:
        goal = memory.goal_for(worker_id)
        assert goal is not None
        assert goal.purpose == WORKER_PATROL_PURPOSE


def test_idle_workers_still_fill_distinct_sectors_when_four_are_returning() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    workers = _workers()
    for worker in workers[-4:]:
        worker["cargo"] = 1
    turn = make_turn(objects=[core(position=HOME), *workers])

    strategy.decide(turn)

    chunk = _single_chunk_resource_block(HOME)
    patrol_sectors = []
    for worker in workers[:12]:
        goal = memory.goal_for(worker["id"])
        assert goal is not None
        assert goal.purpose == WORKER_PATROL_PURPOSE
        patrol_sectors.append(_worker_patrol_sector_index(chunk, goal.position))
    assert len(set(patrol_sectors)) == 12


def test_ring_contact_alert_uses_clockwise_path_distance() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    vanguards = [
        unit(100 + number, "VANGUARD", position=(HOME[0] - 1, HOME[1] - 1))
        for number in range(12)
    ]
    rangers = [
        unit(200 + number, "RANGER", position=(HOME[0] - 1, HOME[1] - 1))
        for number in range(24)
    ]
    first = make_turn(objects=[core(position=HOME), *vanguards, *rangers])
    strategy._reconcile_unit_roles(first)
    patrol_ids = strategy._patrol_ids(first)
    route = _single_chunk_outer_ring(HOME, set())
    patrol_ranger = next(item for item in rangers if UUID(item["id"]) in patrol_ids)
    patrol_ranger = dict(patrol_ranger)
    patrol_ranger["position"] = list(route[0])
    second_objects = [core(position=HOME)]
    second_objects.extend(
        patrol_ranger if item["id"] == patrol_ranger["id"] else item
        for item in (*vanguards, *rangers)
    )
    second = make_turn(tick=101, objects=second_objects)
    member = next(unit for unit in second.rangers if unit.id in patrol_ids)

    assert strategy._patrol_ring_distance(route[0], route[10], second) == 10
    assert strategy._combat_target_is_local(member, route[10], second)
    assert not strategy._combat_target_is_local(member, route[20], second)


def test_first_ring_assignment_preserves_circular_order_with_minimal_travel() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    vanguards = [
        unit(100 + number, "VANGUARD", position=(HOME[0] - 1, HOME[1] - 1))
        for number in range(12)
    ]
    rangers = [
        unit(200 + number, "RANGER", position=(HOME[0] - 1, HOME[1] - 1))
        for number in range(24)
    ]
    first = make_turn(objects=[core(position=HOME), *vanguards, *rangers])
    strategy._reconcile_unit_roles(first)
    patrol_ids = sorted(strategy._patrol_ids(first), key=lambda value: value.bytes)
    route = _single_chunk_outer_ring(HOME, set())
    positions = {
        member_id: route[_ring_slot_index(slot, 8, len(route))]
        for slot, member_id in enumerate(reversed(patrol_ids))
    }
    second_objects = [core(position=HOME)]
    for item in (*vanguards, *rangers):
        copied = dict(item)
        member_id = UUID(item["id"])
        if member_id in positions:
            copied["position"] = list(positions[member_id])
        second_objects.append(copied)
    second = make_turn(tick=101, objects=second_objects)
    memory.patrol_ring_slots = {}
    memory.patrol_ring_slots_initialized = True
    memory.patrol_ring_layout_version = 0
    memory.patrol_ring_phase = 0

    slots = strategy._reconcile_patrol_ring_slots(second)

    total_distance = 0
    for member_id, slot in slots.items():
        target = route[_ring_slot_index(slot, 8, len(route))]
        member = next(item for item in second.rangers if item.id == member_id)
        total_distance += manhattan(member.position, target)
    assert total_distance == 0
    assert memory.patrol_ring_layout_version == 3


def test_patrol_goal_identities_are_durable_across_restart(tmp_path) -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    vanguards = [
        unit(100 + number, "VANGUARD", position=(HOME[0] - 1, HOME[1] - 1))
        for number in range(12)
    ]
    rangers = [
        unit(200 + number, "RANGER", position=(HOME[0] - 1, HOME[1] - 1))
        for number in range(24)
    ]
    turn = make_turn(objects=[core(position=HOME), *vanguards, *rangers])
    strategy._reconcile_unit_roles(turn)
    expected = dict(strategy._reconcile_patrol_ring_slots(turn))
    path = tmp_path / "memory.json"
    memory.save(path)

    restored = WorldMemory.load(path)
    restored_strategy = _strategy(restored)
    resumed = make_turn(tick=101, objects=[core(position=HOME), *vanguards, *rangers])
    actual = restored_strategy._reconcile_patrol_ring_slots(resumed)

    assert actual == expected
    assert len(actual) == 8
