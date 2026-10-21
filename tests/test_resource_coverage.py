"""Durable single-chunk Worker quadrants and chunk-edge patrol coverage."""

from __future__ import annotations

from uuid import UUID

from arena_hero_bot.geometry import manhattan
from arena_hero_bot.memory import UnitGoal, WorldMemory
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    PATROL_RING_PURPOSE,
    RESOURCE_CLAIM_PURPOSE,
    WORKER_BLOCK_PATROL_PURPOSE,
    AggressiveStrategy,
    StrategyConfig,
    _ring_slot_index,
    _single_chunk_outer_ring,
    _single_chunk_resource_blocks,
)

from .factories import core, make_turn, unit

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


def test_resource_quadrants_share_one_centered_chunk() -> None:
    blocks = _single_chunk_resource_blocks(HOME)

    assert [block.index for block in blocks] == [0, 1, 2, 3]
    assert {
        (block.min_x, block.max_x, block.min_y, block.max_y) for block in blocks
    } == {
        (-416, -401, 640, 655),
        (-400, -385, 640, 655),
        (-416, -401, 656, 671),
        (-400, -385, 656, 671),
    }
    assert sum(block.contains(HOME) for block in blocks) == 1


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


def test_worker_assignments_are_four_per_block_and_survive_replacements() -> None:
    strategy = _strategy()
    workers = _workers()
    first = make_turn(objects=[core(position=HOME), *workers])

    strategy._assign_resources(first)

    blocks = strategy.memory.worker_blocks
    assert sorted(blocks.values()) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    survivor_blocks = dict(blocks)
    dead_id = next(worker["id"] for worker in workers if blocks[worker["id"]] == 2)
    survivors = [worker for worker in workers if worker["id"] != dead_id]
    replacement = unit(999, "WORKER", position=(HOME[0] - 12, HOME[1] + 4))
    second = make_turn(
        tick=101,
        objects=[core(position=HOME), *survivors, replacement],
    )

    strategy._assign_resources(second)

    updated = strategy.memory.worker_blocks
    for worker in survivors:
        assert updated[worker["id"]] == survivor_blocks[worker["id"]]
    assert updated[replacement["id"]] == 2
    assert sorted(updated.values()) == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
    ]


def test_workers_never_claim_resources_outside_their_assigned_block() -> None:
    strategy = _strategy()
    workers = _workers()
    blocks = _single_chunk_resource_blocks(HOME)
    resources = [(block.min_x + 5, block.min_y + 5) for block in blocks] + [(100, 100)]
    turn = make_turn(
        objects=[core(position=HOME), *workers],
        resource_cells=resources,
    )

    assignments = strategy._assign_resources(turn)

    assert len(assignments) == 4
    for worker_id, resource in assignments.items():
        block = next(
            block
            for block in blocks
            if block.index == strategy.memory.worker_blocks[str(worker_id)]
        )
        assert block.contains(resource)
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


def test_underfoot_resource_in_another_block_is_not_claimed() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    foreign_block = next(
        block for block in _single_chunk_resource_blocks(HOME) if block.index == 1
    )
    worker = unit(
        301,
        "WORKER",
        position=(foreign_block.min_x + 7, foreign_block.min_y + 9),
    )
    memory.worker_blocks[worker["id"]] = 0
    turn = make_turn(
        objects=[core(position=HOME), worker],
        resource_cells=[tuple(worker["position"])],
    )

    assignments = strategy._assign_resources(turn)

    assert assignments == {}
    assert memory.worker_blocks[worker["id"]] == 0


def test_worker_patrol_chooses_oldest_reachable_cell_and_persists_visits(
    tmp_path,
) -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    worker = unit(301, "WORKER", position=(HOME[0] - 10, HOME[1] - 10))
    turn = make_turn(objects=[core(position=HOME), worker])
    strategy._assign_resources(turn)
    block = next(
        block
        for block in _single_chunk_resource_blocks(HOME)
        if block.index == memory.worker_blocks[worker["id"]]
    )
    target = (block.min_x + 7, block.min_y + 9)
    for x in range(block.min_x, block.max_x + 1):
        for y in range(block.min_y, block.max_y + 1):
            memory.worker_visit_ticks[(x, y)] = 100
    memory.worker_visit_ticks[target] = 5

    strategy.decide(turn)

    goal = memory.goal_for(worker["id"])
    assert goal is not None
    assert goal.purpose == f"{WORKER_BLOCK_PATROL_PURPOSE}-{block.index}"
    assert goal.position == target
    assert memory.worker_visit_ticks[tuple(worker["position"])] == 100
    path = tmp_path / "memory.json"
    memory.save(path)
    restored = WorldMemory.load(path)
    assert restored.worker_blocks == memory.worker_blocks
    assert restored.worker_visit_ticks[target] == 5


def test_visible_resource_interrupts_worker_block_patrol() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    worker = unit(301, "WORKER", position=(HOME[0] - 12, HOME[1] - 12))
    peaceful = make_turn(objects=[core(position=HOME), worker])
    strategy.decide(peaceful)
    patrol_goal = memory.goal_for(worker["id"])
    assert patrol_goal is not None
    assert patrol_goal.purpose.startswith(WORKER_BLOCK_PATROL_PURPOSE)

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


def test_all_worker_patrol_targets_are_unique_and_inside_their_blocks() -> None:
    memory = WorldMemory(core_home_position=HOME)
    strategy = _strategy(memory)
    workers = _workers()
    turn = make_turn(objects=[core(position=HOME), *workers])

    strategy.decide(turn)

    blocks = _single_chunk_resource_blocks(HOME)
    targets: dict[str, tuple[int, int]] = {}
    for worker in workers:
        goal = memory.goal_for(worker["id"])
        assert goal is not None
        targets[worker["id"]] = goal.position
    assert len(set(targets.values())) == len(targets)
    for worker_id, target in targets.items():
        block = next(
            block for block in blocks if block.index == memory.worker_blocks[worker_id]
        )
        assert block.contains(target)
        goal = memory.goal_for(worker_id)
        assert goal is not None
        assert goal.purpose.startswith(WORKER_BLOCK_PATROL_PURPOSE)


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
