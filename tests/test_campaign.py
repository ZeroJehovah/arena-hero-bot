"""Stage-gated campaign economy, formation, and migration behavior."""

from __future__ import annotations

from pathlib import Path

from arena_hero import UnitType, core_resource_capacity, unit_cost

from arena_hero_bot.geometry import add, manhattan, next_step
from arena_hero_bot.memory import ExpeditionSquad, WorldMemory
from arena_hero_bot.migration import (
    blocks_are_cardinally_adjacent,
    plan_resource_quota,
    resource_block_index,
    resource_block_quota,
    resource_block_ring,
)
from arena_hero_bot.strategy import (
    CAMPAIGN_PHASE_STAGE1,
    CAMPAIGN_PHASE_STAGE2,
    CAMPAIGN_PHASE_STAGE3,
    CAMPAIGN_PHASE_STAGE4,
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
    _uncovered_symmetric_cells,
)

from .factories import core, make_turn, object_id, unit


def _strategy(memory: WorldMemory) -> AggressiveStrategy:
    return AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=16,
            max_population=None,
            expedition_mode=True,
            campaign_mode=True,
        ),
    )


def _workers(count: int) -> list[dict[str, object]]:
    return [
        unit(100 + index, "WORKER", position=(20 + index, 20)) for index in range(count)
    ]


def _rangers(count: int, *, offset: int = 1000) -> list[dict[str, object]]:
    return [
        unit(offset + index, "RANGER", position=(40 + index, 20))
        for index in range(count)
    ]


def _max_roster() -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[str, str]
]:
    workers = _workers(16)
    rangers = _rangers(28)
    roles = {
        **{str(worker["id"]): "staged" for worker in workers},
        **{str(ranger["id"]): "defense" for ranger in rangers[:20]},
        **{
            str(ranger["id"]): f"patrol-{index // 2 + 1}"
            for index, ranger in enumerate(rangers[20:])
        },
    }
    return workers, rangers, roles


def test_campaign_replaces_a_rock_sealed_block_center() -> None:
    """An unreachable center cannot freeze the Core between two resources."""

    origin = (-1074, -17)
    sealed_center = (-1072, -17)
    core_object = core(9, position=origin)
    workers, rangers, roles = _max_roster()
    memory = WorldMemory(
        core_home_position=sealed_center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=origin,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(
        objects=[core_object, *workers, *rangers],
        obstacles=[
            (-1073, -17),
            (-1073, -16),
            (-1072, -16),
            (-1072, -18),
            (-1071, -17),
        ],
    )

    _strategy(memory).decide(turn)

    assert memory.core_home_position is not None
    assert memory.core_home_position != sealed_center
    assert blocks_are_cardinally_adjacent(
        resource_block_index(origin),
        resource_block_index(memory.core_home_position),
    )
    assert (
        next_step(
            origin,
            memory.core_home_position,
            blocked=memory.obstacles,
            require_path=True,
        )
        is not None
    )
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"


def test_campaign_repairs_a_blocked_adjacent_center_in_the_same_block() -> None:
    """A newly seen rock must not send an advancing Core back to its old block."""

    origin = (-1033, -15)
    blocked_center = (-1008, -16)
    core_object = core(9, position=origin)
    workers, rangers, roles = _max_roster()
    memory = WorldMemory(
        core_home_position=blocked_center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=origin,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(
        objects=[core_object, *workers, *rangers],
        obstacles=[blocked_center],
    )

    _strategy(memory).decide(turn)

    assert memory.core_home_position is not None
    assert memory.core_home_position != blocked_center
    assert resource_block_index(memory.core_home_position) == resource_block_index(
        blocked_center
    )
    assert memory.core_home_position not in turn.obstacle_cells
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"


def test_explicit_quota_plan_publishes_one_adjacent_hop() -> None:
    memory = WorldMemory(core_home_position=(-2033, -2929))

    tier3 = plan_resource_quota(memory, 3, current_position=(-2033, -2929))
    tier4 = plan_resource_quota(memory, 4, current_position=(-2033, -2929))

    assert tier3 is not None
    assert (tier3.goal_ring, tier3.goal_quota) == (34, 3)
    assert blocks_are_cardinally_adjacent(tier3.current_block, tier3.target_block)
    assert tier4 is not None
    assert (tier4.goal_ring, tier4.goal_quota) == (24, 4)
    assert blocks_are_cardinally_adjacent(tier4.current_block, tier4.target_block)


def test_new_core_resets_old_campaign_to_its_spawn_chunk() -> None:
    old_core_id = object_id(1)
    new_core_id = object_id(9)
    memory = WorldMemory(
        core_home_position=(64, 64),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=old_core_id,
        unit_roles={"old-ranger": "defense"},
        expedition_squads=[ExpeditionSquad(7, ("old-ranger",), (1, 0))],
    )
    turn = make_turn(
        tick=100,
        objects=[
            core(9, position=(-2033, -2929)),
            unit(2, "WORKER", position=(-2030, -2929)),
        ],
    )

    _strategy(memory).decide(turn)

    assert memory.campaign_last_core_id == new_core_id
    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE1
    assert memory.campaign_target_quota == 2
    assert memory.core_home_position is not None
    assert resource_block_index(memory.core_home_position) == (-64, -92)
    assert memory.unit_roles == {}
    assert memory.expedition_squads == []


def test_new_core_keeps_its_own_chunk_when_every_center_is_temporarily_busy() -> None:
    current_core = core(9, position=(-1740, -1900))
    centers = ((-1745, -1905), (-1744, -1905), (-1745, -1904), (-1744, -1904))
    memory = WorldMemory(
        core_home_position=(-2033, -2929),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=object_id(1),
    )
    turn = make_turn(
        objects=[
            current_core,
            *(
                unit(20 + index, "WORKER", position=cell)
                for index, cell in enumerate(centers)
            ),
        ]
    )

    _strategy(memory).decide(turn)

    assert memory.core_home_position in centers
    assert resource_block_index(memory.core_home_position) == resource_block_index(
        current_core["position"]
    )


def test_core_respawn_event_resets_even_with_replayed_uuid() -> None:
    current_core = core(9, position=(-1740, -1900))
    memory = WorldMemory(
        core_home_position=(-2033, -2929),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=current_core["id"],
    )
    turn = make_turn(
        tick=101,
        objects=[current_core, unit(2, "WORKER", position=(-1739, -1900))],
        events=[
            {
                "event_id": object_id(900),
                "tick": 101,
                "event_type": "CORE_RESPAWNED",
                "target_id": current_core["id"],
                "position": [-1740, -1900],
            }
        ],
    )

    _strategy(memory).decide(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE1
    assert memory.campaign_target_quota == 2
    assert memory.core_home_position is not None
    assert resource_block_index(memory.core_home_position) == resource_block_index(
        current_core["position"]
    )


def test_same_core_discards_a_distant_persisted_target() -> None:
    current_core = core(9, position=(-1740, -1900))
    memory = WorldMemory(
        core_home_position=(-2033, -2929),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=current_core["id"],
        core_migration_last_position=(-1740, -1900),
    )
    turn = make_turn(objects=[current_core])

    _strategy(memory).decide(turn)

    assert memory.core_home_position is not None
    assert resource_block_index(memory.core_home_position) == resource_block_index(
        current_core["position"]
    )


def test_non_cardinal_core_jump_discards_a_cross_block_target() -> None:
    current_core = core(9, position=(-2033, -2929))
    memory = WorldMemory(
        core_home_position=(-2001, -2929),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=2,
        campaign_last_core_id=current_core["id"],
        core_migration_last_position=(0, 0),
        core_migration_steps_since_pause=12,
    )
    turn = make_turn(objects=[current_core])

    _strategy(memory).decide(turn)

    assert memory.core_home_position is not None
    assert resource_block_index(memory.core_home_position) == resource_block_index(
        current_core["position"]
    )
    assert memory.core_migration_steps_since_pause == 0


def test_stage_two_publishes_only_one_adjacent_hop() -> None:
    current_core = core(9, position=(-2033, -2929))
    current_center = (-2033, -2929)
    memory = WorldMemory(
        core_home_position=current_center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=current_core["id"],
        core_migration_last_position=current_center,
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(objects=[current_core, *_workers(16), *rangers])

    _strategy(memory).decide(turn)

    assert memory.core_home_position is not None
    previous_block = resource_block_index(current_center)
    target_block = resource_block_index(memory.core_home_position)
    assert blocks_are_cardinally_adjacent(previous_block, target_block)
    assert resource_block_ring(target_block) < resource_block_ring(previous_block)
    assert memory.campaign_resting is False


def test_campaign_memory_round_trip_preserves_stage_and_rest(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = WorldMemory(
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=object_id(9),
        campaign_resting=True,
    )

    memory.save(path)
    restored = WorldMemory.load(path)

    assert restored.campaign_phase == CAMPAIGN_PHASE_STAGE2
    assert restored.campaign_target_quota == 3
    assert restored.campaign_last_core_id == object_id(9)
    assert restored.campaign_resting is True


def test_stage_one_spends_on_the_eighth_worker_without_a_reserve() -> None:
    core_object = core(position=(0, 0))
    memory = WorldMemory(
        core_home_position=(0, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    workers = _workers(7)
    worker_cost = unit_cost(UnitType.WORKER, 7)
    turn = make_turn(
        resources=worker_cost,
        objects=[core_object, *workers],
    )

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_stage_one_spends_low_capacity_resources_on_a_worker() -> None:
    core_object = core(position=(0, 0))
    memory = WorldMemory(
        core_home_position=(0, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
    )
    turn = make_turn(
        resources=10,
        objects=[core_object, unit(2, "WORKER", position=(1, 0))],
    )

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_stage_one_keeps_workers_first_until_twelve() -> None:
    core_object = core(position=(0, 0))
    memory = WorldMemory(
        core_home_position=(0, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
    )
    turn = make_turn(
        resources=unit_cost(UnitType.RANGER, 3),
        objects=[
            core_object,
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(2, 0)),
            unit(4, "WORKER", position=(3, 0)),
        ],
    )

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_stage_one_builds_a_defender_before_a_worker_under_core_threat() -> None:
    core_object = core(position=(0, 0), hp=4, shield=5)
    memory = WorldMemory(
        core_home_position=(0, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
    )
    turn = make_turn(
        resources=unit_cost(UnitType.RANGER, 5),
        objects=[
            core_object,
            *_workers(4),
            unit(900, "RANGER", controlled=False, position=(2, 0)),
        ],
    )

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.RANGER


def test_stage_two_keeps_workers_ahead_of_rangers_until_sixteen() -> None:
    center = (15, 15)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(7)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(resources=60, objects=[core_object, *_workers(12), *rangers])

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_persisted_rest_off_the_published_center_resumes_the_hop() -> None:
    """A rest flag can only hold a Core that stands on its target center."""

    core_object = core(position=(-1743, -1905))
    memory = WorldMemory(
        core_home_position=(-1744, -1904),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
        # Persisted by the retired step-count rule: the Core is one cell short
        # of the published block center and must keep walking to it.
        campaign_resting=True,
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(objects=[core_object, *_workers(12), *rangers])

    _strategy(memory).decide(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE2
    assert memory.campaign_resting is False
    assert memory.core_home_position == (-1744, -1904)
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"


def test_stage_two_finishes_workers_before_expanding_its_screen() -> None:
    core_object = core(position=(-400, 655))
    memory = WorldMemory(
        core_home_position=(-400, 655),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(resources=60, objects=[core_object, *_workers(12), *rangers])

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_stage_one_advances_but_waits_for_full_stage_two_before_interblock_move() -> (
    None
):
    core_object = core(position=(-2033, -2929))
    memory = WorldMemory(
        core_home_position=(-2033, -2929),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    turn = make_turn(objects=[core_object, *_workers(8)])

    _strategy(memory).decide(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE2
    assert memory.campaign_target_quota == 3
    assert memory.core_home_position is not None
    assert resource_block_index(memory.core_home_position) == resource_block_index(
        core_object["position"]
    )


def test_stage_one_still_walks_the_core_to_its_spawn_block_center() -> None:
    core_object = core(position=(10, 15))
    center = (15, 15)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    turn = make_turn(objects=[core_object])

    _strategy(memory).decide(turn)

    assert memory.core_home_position == center
    assert resource_block_index(center) == resource_block_index(core_object["position"])
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"


def test_stage_two_roster_advances_to_stage_three_and_asks_for_a_ranger() -> None:
    center = (15, 15)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(objects=[core_object, *_workers(16), *rangers])
    strategy = _strategy(memory)

    strategy._advance_campaign_phase(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE3
    assert memory.campaign_target_quota == 4
    assert strategy._campaign_spawn_candidates(turn) == (UnitType.RANGER,)


def test_completed_stage_three_reaches_max_and_targets_quota_eight() -> None:
    core_object = core(position=(-2033, -2929))
    memory = WorldMemory(
        core_home_position=(-2033, -2929),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    rangers = _rangers(28)
    for ranger in rangers[:20]:
        memory.unit_roles[str(ranger["id"])] = "defense"
    for index, ranger in enumerate(rangers[20:]):
        memory.unit_roles[str(ranger["id"])] = f"patrol-{index // 2 + 1}"
    turn = make_turn(objects=[core_object, *_workers(16), *rangers])
    strategy = _strategy(memory)

    strategy._reconcile_unit_roles(turn)
    assert strategy._campaign_defense_rangers(turn) == 20
    assert strategy._campaign_patrol_rangers(turn) == 8
    strategy._advance_campaign_phase(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE4
    assert memory.campaign_target_quota == 8
    assert strategy._formation_quota() == (0, 28, 0, 20)
    assert strategy._expedition_squad_quota() == (0, 4)
    assert memory.core_home_position is not None
    assert blocks_are_cardinally_adjacent(
        resource_block_index(core_object["position"]),
        resource_block_index(memory.core_home_position),
    )


def test_persisted_max_stage_repairs_the_old_quota_four_goal() -> None:
    center = (15, 15)
    core_object = core(position=center)
    workers, rangers, roles = _max_roster()
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(objects=[core_object, *workers, *rangers])

    _strategy(memory)._advance_campaign_phase(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE4
    assert memory.campaign_target_quota == 8


def test_campaign_defense_covers_radius_eight_and_compresses_to_four() -> None:
    center = (17, -23)
    memory = WorldMemory(campaign_phase=CAMPAIGN_PHASE_STAGE3)
    strategy = _strategy(memory)

    posts, radius = strategy._symmetric_defense_posts(center, set())

    assert len(posts) == 20
    assert len(set(posts)) == 20
    assert radius == 8
    assert max(manhattan(center, post) for post in posts) <= 8
    assert (
        _uncovered_symmetric_cells(
            center,
            posts,
            set(),
            radius,
            vanguard_slots=0,
        )
        == set()
    )

    memory.defense_compressed = True
    compressed, compressed_radius = strategy._symmetric_defense_posts(center, set())

    assert len(compressed) == 20
    assert compressed_radius == 4
    assert max(manhattan(center, post) for post in compressed) <= 4
    assert (
        _uncovered_symmetric_cells(
            center,
            compressed,
            set(),
            compressed_radius,
            vanguard_slots=0,
        )
        == set()
    )


def test_campaign_arrival_without_unit_service_starts_next_move_immediately() -> None:
    core_object = core(position=(14, 15))
    memory = WorldMemory(
        core_home_position=(15, 15),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(13, 15),
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
    )
    strategy = _strategy(memory)

    arrival = make_turn(tick=100, objects=[core_object])
    strategy.decide(arrival)
    assert arrival.plan.core_action is not None
    assert arrival.plan.core_action.type == "START_MOVE"
    assert arrival.plan.core_action.direction.value == "RIGHT"
    assert memory.core_migration_pause_until_tick == -1


def test_campaign_continues_to_a_valid_adjacent_center() -> None:
    """A valid published hop survives a stale rest flag."""

    core_object = core(position=(-1686, -1904))
    memory = WorldMemory(
        core_home_position=(-1649, -1904),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-1686, -1904),
        core_migration_pause_until_tick=-1,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
        campaign_resting=True,
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(tick=100, objects=[core_object, *_workers(16), *rangers])

    report = _strategy(memory).decide(turn)

    assert memory.campaign_resting is False
    assert memory.core_home_position == (-1649, -1904)
    assert resource_block_ring(
        resource_block_index(memory.core_home_position)
    ) < resource_block_ring(resource_block_index(core_object["position"]))
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"
    assert not any("hold the block center" in item.reason for item in report.decisions)


def test_campaign_mid_leg_cargo_holds_without_delaying_core() -> None:
    """Cargo in the shared next cell waits there instead of being evicted."""

    core_object = core(position=(-1686, -1904))
    memory = WorldMemory(
        core_home_position=(-1649, -1904),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-1686, -1904),
        core_migration_pause_until_tick=-1,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    loaded = unit(500, "WORKER", position=(-1685, -1904), cargo=1)
    turn = make_turn(tick=100, objects=[core_object, *_workers(16), loaded, *rangers])

    report = _strategy(memory).decide(turn)

    assert memory.campaign_resting is False
    assert memory.core_migration_pause_until_tick == -1
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"
    assert any(
        item.actor_id == str(loaded["id"])
        and item.action == "WAIT"
        and item.reason == "hold the next Core arrival slot for in-transit delivery"
        for item in report.decisions
    )
    assert not any(
        item.actor_id == str(loaded["id"])
        and item.reason == "yield the Core migration lane"
        for item in report.decisions
    )


def test_campaign_migration_builds_on_the_arrival_tick_then_resumes() -> None:
    """A formation gap spends the one service Tick before migration resumes."""

    core_object = core(position=(-945, -945))
    memory = WorldMemory(
        core_home_position=(-913, -945),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-945, -945),
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(12)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    population = 16 + len(rangers)
    turn = make_turn(
        tick=100,
        resources=unit_cost(UnitType.RANGER, population),
        objects=[core_object, *_workers(16), *rangers],
    )
    strategy = _strategy(memory)

    strategy.decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.RANGER
    assert memory.core_home_position == (-913, -945)

    resumed = make_turn(
        tick=101,
        objects=[core_object, *_workers(16), *rangers],
    )
    strategy.decide(resumed)

    assert resumed.plan.core_action is not None
    assert resumed.plan.core_action.type == "START_MOVE"


def test_campaign_migration_build_priority_blocks_deposit_that_tick() -> None:
    """A build-due service Tick clears the cell and holds for the spawn."""

    core_object = core(position=(-945, -945))
    memory = WorldMemory(
        core_home_position=(-913, -945),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-945, -945),
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(12)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    workers = _workers(16)
    loaded = workers[0]
    loaded["position"] = [-945, -945]
    loaded["cargo"] = 1
    population = 16 + len(rangers)
    turn = make_turn(
        tick=100,
        resources=unit_cost(UnitType.RANGER, population),
        objects=[core_object, *workers, *rangers],
    )

    report = _strategy(memory).decide(turn)

    worker = next(unit for unit in turn.workers if str(unit.id) == loaded["id"])
    assert turn.plan.unit_actions[worker.id].type == "MOVE"
    assert turn.plan.core_action is None
    assert any(
        item.actor_kind == "CORE"
        and item.action == "WAIT"
        and item.reason
        == "hold one service Tick for the Core cell to clear before production"
        for item in report.decisions
    )


def test_campaign_migration_spawns_after_the_cleared_service_tick() -> None:
    """The one-Tick hold after clearing the Core cell actually produces."""

    core_object = core(position=(-945, -945))
    memory = WorldMemory(
        core_home_position=(-913, -945),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-945, -945),
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(12)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    workers = _workers(16)
    loaded = workers[0]
    loaded["position"] = [-945, -945]
    loaded["cargo"] = 1
    population = 16 + len(rangers)
    resources = unit_cost(UnitType.RANGER, population)
    strategy = _strategy(memory)

    first = make_turn(
        tick=100,
        resources=resources,
        objects=[core_object, *workers, *rangers],
    )
    strategy.decide(first)

    loaded["position"] = [-944, -945]
    second = make_turn(
        tick=101,
        resources=resources,
        objects=[core_object, *workers, *rangers],
    )
    strategy.decide(second)

    assert second.plan.core_action is not None
    assert second.plan.core_action.type == "SPAWN"


def test_full_bank_mid_migration_does_not_skip_the_cleared_spawn_tick() -> None:
    """A full bank clears the staged Worker, holds, then spends on the Ranger."""

    core_object = core(position=(-945, -945))
    memory = WorldMemory(
        core_home_position=(-913, -945),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-945, -945),
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
    )
    workers, rangers, roles = _max_roster()
    memory.unit_roles = roles
    memory.unit_roles_initialized = True
    loaded = workers[0]
    loaded["position"] = [-945, -945]
    loaded["cargo"] = 1
    resources = core_resource_capacity(len(workers) + len(rangers))
    strategy = _strategy(memory)

    first = make_turn(
        tick=100,
        resources=resources,
        objects=[core_object, *workers, *rangers],
    )
    report = strategy.decide(first)

    assert first.plan.core_action is None
    assert any(
        item.actor_kind == "CORE"
        and item.action == "WAIT"
        and item.reason
        == "hold one service Tick for the Core cell to clear before production"
        for item in report.decisions
    )

    loaded["position"] = [-944, -945]
    second = make_turn(
        tick=101,
        resources=resources,
        objects=[core_object, *workers, *rangers],
    )
    strategy.decide(second)

    assert second.plan.core_action is not None
    assert second.plan.core_action.type == "SPAWN"
    assert second.plan.core_action.unit_type is UnitType.RANGER


def test_campaign_migration_build_priority_blocks_unit_heal_that_tick() -> None:
    """A build-due service Tick does not let an ordinary heal take priority."""

    core_object = core(position=(-945, -945))
    memory = WorldMemory(
        core_home_position=(-913, -945),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-945, -945),
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(12)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    workers = _workers(16)
    critical = workers[0]
    critical["position"] = [-945, -945]
    critical["hp"] = 1
    population = 16 + len(rangers)
    turn = make_turn(
        tick=100,
        resources=unit_cost(UnitType.RANGER, population),
        objects=[core_object, *workers, *rangers],
    )

    report = _strategy(memory).decide(turn)

    assert not any(
        decision.actor_id == str(critical["id"]) and decision.action == "HEAL"
        for decision in report.decisions
    )


def test_campaign_migration_build_priority_clears_a_parked_worker() -> None:
    """The next spawn needs the Core cell even when the occupant has no cargo."""

    core_object = core(position=(-945, -945))
    memory = WorldMemory(
        core_home_position=(-913, -945),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-945, -945),
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(12)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    workers = _workers(16)
    parked = workers[0]
    parked["position"] = [-945, -945]
    population = 16 + len(rangers)
    turn = make_turn(
        tick=100,
        resources=unit_cost(UnitType.RANGER, population),
        objects=[core_object, *workers, *rangers],
    )

    _strategy(memory).decide(turn)

    worker = next(unit for unit in turn.workers if str(unit.id) == parked["id"])
    assert turn.plan.unit_actions[worker.id].type == "MOVE"


def test_quota_two_block_departs_after_the_full_stage_two_formation() -> None:
    """Stage two grants quota-2 movement only after 16W+8R is intact."""

    center = (-945, -945)
    core_object = core(position=center)
    assert resource_block_quota(resource_block_index(center)) == 2
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(tick=100, objects=[core_object, *_workers(16), *rangers])

    _strategy(memory).decide(turn)

    assert memory.campaign_resting is False
    assert memory.core_home_position is not None
    assert memory.core_home_position != center
    assert blocks_are_cardinally_adjacent(
        resource_block_index(center),
        resource_block_index(memory.core_home_position),
    )
    assert resource_block_quota(resource_block_index(memory.core_home_position)) <= 2
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"


def test_campaign_rest_holds_the_center_until_nearby_cargo_lands() -> None:
    """The rest ends when the Core neighbourhood has nothing left to deliver."""

    center = (-945, -945)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    loaded = unit(500, "WORKER", position=(-943, -945), cargo=1)
    turn = make_turn(tick=100, objects=[core_object, *_workers(16), loaded, *rangers])

    report = _strategy(memory).decide(turn)

    assert memory.campaign_resting is True
    assert memory.core_home_position == center
    assert turn.plan.core_action is None
    assert any("hold the block center" in item.reason for item in report.decisions)

    delivered = make_turn(
        tick=101,
        objects=[
            core_object,
            *_workers(16),
            unit(500, "WORKER", position=(-944, -945)),
            *rangers,
        ],
    )
    _strategy(memory).decide(delivered)

    assert memory.campaign_resting is False
    assert memory.core_home_position is not None
    assert memory.core_home_position != center
    assert delivered.plan.core_action is not None
    assert delivered.plan.core_action.type == "START_MOVE"


def test_campaign_rest_clears_the_service_cell_for_the_next_delivery() -> None:
    """A resting Core keeps its cell free so the queue it waits on can land."""

    center = (-945, -945)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    parked = unit(600, "RANGER", position=center)
    loaded = unit(500, "WORKER", position=(-942, -945), cargo=1)
    turn = make_turn(
        tick=100,
        objects=[core_object, parked, *_workers(16), loaded, *rangers],
    )

    report = _strategy(memory).decide(turn)

    assert memory.campaign_resting is True
    parked_unit = next(unit for unit in turn.rangers if str(unit.id) == parked["id"])
    action = turn.plan.unit_actions[parked_unit.id]
    assert action.type == "MOVE"
    assert any(
        item.actor_id == parked["id"]
        and item.reason == "leave Core service cell for unique traffic"
        for item in report.decisions
    )


def test_campaign_rest_pipeline_admits_the_next_worker_on_departure() -> None:
    """A resting Core swaps one service Worker for the next in one Tick."""

    center = (-945, -945)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
        campaign_resting=True,
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    leaving = unit(600, "WORKER", position=center)
    entering = unit(601, "WORKER", position=(-944, -945), cargo=1)
    turn = make_turn(
        tick=100,
        resources=2,
        objects=[core_object, leaving, entering, *rangers],
    )

    AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=16,
            max_population=None,
            expedition_mode=True,
            campaign_mode=True,
        ),
    ).decide(turn)

    leaving_unit = next(unit for unit in turn.workers if str(unit.id) == leaving["id"])
    entering_unit = next(
        unit for unit in turn.workers if str(unit.id) == entering["id"]
    )
    leaving_action = turn.plan.unit_actions[leaving_unit.id]
    entering_action = turn.plan.unit_actions[entering_unit.id]
    assert leaving_action.type == "MOVE"
    assert add(leaving_unit.position, leaving_action.direction) != center
    assert entering_action.type == "MOVE"
    assert add(entering_unit.position, entering_action.direction) == center


def test_full_bank_center_spawns_before_the_next_campaign_hop() -> None:
    """A full bank must not carry loaded Workers past its center service tick."""

    center = (15, 15)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=(47, 15),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
    )
    loaded = unit(500, "WORKER", position=(16, 15), cargo=1)
    turn = make_turn(
        tick=100,
        resources=65,
        objects=[core_object, *_workers(12), loaded],
    )

    _strategy(memory).decide(turn)

    assert turn.resource_capacity == 65
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_pre_max_migration_spends_an_affordable_formation_gap() -> None:
    """A pre-max formation gap interrupts migration even without waiting cargo."""

    center = (-945, -945)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=(-913, -945),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(20)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(
        tick=100,
        resources=65,
        objects=[core_object, *_workers(16), *rangers],
    )

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.RANGER


def test_max_formation_migration_does_not_spawn_below_a_full_bank() -> None:
    """The max formation keeps its full-bank-only reserve while travelling."""

    center = (-945, -945)
    core_object = core(position=center)
    workers, rangers, roles = _max_roster()
    memory = WorldMemory(
        core_home_position=(-913, -945),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(
        tick=100,
        resources=core_resource_capacity(16 + len(rangers)) - 1,
        objects=[core_object, *workers, *rangers],
    )

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"


def test_campaign_center_holds_for_an_affordable_build_before_the_next_hop() -> None:
    """A center cannot publish a hop while an affordable formation slot is owed."""

    center = (-945, -945)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(18)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(
        tick=100,
        resources=unit_cost(UnitType.RANGER, 16 + len(rangers)),
        objects=[core_object, *_workers(16), *rangers],
    )
    strategy = _strategy(memory)

    strategy.decide(turn)

    assert memory.campaign_resting is True
    assert memory.core_home_position == center
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.RANGER

    depleted = make_turn(
        tick=101,
        objects=[core_object, *_workers(16), *rangers],
    )
    strategy.decide(depleted)

    assert memory.campaign_resting is False
    assert memory.core_home_position != center
    assert depleted.plan.core_action is not None
    assert depleted.plan.core_action.type == "START_MOVE"


def test_campaign_build_priority_preempts_beacon_at_the_center() -> None:
    """A build-due rest spends the Core action before picking up a Beacon."""

    center = (-945, -945)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(18)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(
        tick=100,
        resources=unit_cost(UnitType.RANGER, 16 + len(rangers)),
        objects=[core_object, *_workers(16), *rangers],
        beacon={"position": list(center)},
    )

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"


def test_quota_three_block_waits_for_the_stage_two_roster() -> None:
    """Quota 3 is reached here, so the Core holds until the screen is full."""

    center = (-529, -529)
    core_object = core(position=center)
    assert resource_block_quota(resource_block_index(center)) == 3
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(tick=100, objects=[core_object, *_workers(12), *rangers])

    _strategy(memory).decide(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE2
    assert memory.campaign_target_quota == 3
    assert memory.core_home_position == center
    assert turn.plan.core_action is None


def test_stage_three_moves_only_inside_quota_three_toward_quota_four() -> None:
    center = (-529, -529)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    rangers = _rangers(20)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(tick=100, objects=[core_object, *_workers(16), *rangers])

    _strategy(memory).decide(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE3
    assert memory.campaign_target_quota == 4
    assert memory.campaign_resting is False
    assert memory.core_home_position is not None
    assert memory.core_home_position != center
    assert blocks_are_cardinally_adjacent(
        resource_block_index(center),
        resource_block_index(memory.core_home_position),
    )
    assert resource_block_quota(resource_block_index(memory.core_home_position)) <= 3
    assert resource_block_ring(
        resource_block_index(memory.core_home_position)
    ) < resource_block_ring(resource_block_index(center))
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"


def test_quota_four_hop_waits_for_forward_scout_and_enemy_settle() -> None:
    center = (815, 15)
    core_object = core(position=center)
    workers, rangers, roles = _max_roster()
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(
        tick=100,
        objects=[core_object, *workers, *rangers],
    )
    strategy = _strategy(memory)

    strategy.decide(turn)

    scout_target = memory.campaign_scout_target
    assert scout_target is not None
    assert resource_block_quota(resource_block_index(scout_target)) == 4
    assert memory.core_home_position == center
    assert turn.plan.core_action is None

    block = resource_block_index(scout_target)
    for x in range(
        block[0] * 32,
        (block[0] + 1) * 32,
    ):
        for y in range(
            block[1] * 32,
            (block[1] + 1) * 32,
        ):
            if (x, y) not in memory.obstacles:
                memory.campaign_scout_seen_ticks[(x, y)] = turn.tick
    memory.campaign_scout_last_enemy_tick = turn.tick

    strategy._update_campaign_scout(turn)

    assert memory.core_home_position == center
    assert memory.campaign_scout_target == scout_target

    settled = make_turn(
        tick=108,
        objects=[core_object, *workers, *rangers],
    )
    strategy._update_campaign_scout(settled)

    assert memory.core_home_position == scout_target
    assert memory.campaign_scout_target is None


def test_campaign_reserve_starts_only_at_the_max_formation() -> None:
    core_object = core(position=(0, 0))
    pre_stage2 = WorldMemory(
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_last_core_id=core_object["id"],
    )
    post_stage2 = WorldMemory(
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_last_core_id=core_object["id"],
    )
    turn = make_turn(resources=100, objects=[core_object])

    assert (
        _strategy(pre_stage2)._spawn_safety_reserve(
            turn,
            production_cost=10,
            available_resources=100,
        )
        == 0
    )
    assert (
        _strategy(post_stage2)._spawn_safety_reserve(
            turn,
            production_cost=10,
            available_resources=100,
        )
        == 0
    )


def test_max_formation_builds_only_at_full_bank_and_keeps_spawning_rangers() -> None:
    center = (15, 15)
    core_object = core(position=center)
    workers, rangers, roles = _max_roster()
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    strategy = _strategy(memory)
    full = make_turn(
        resources=core_resource_capacity(16 + 28),
        objects=[core_object, *workers, *rangers],
    )

    strategy.decide(full)

    assert full.plan.core_action is not None
    assert full.plan.core_action.type == "SPAWN"
    assert full.plan.core_action.unit_type is UnitType.RANGER


def test_max_formation_assigns_every_extra_ranger_to_patrol() -> None:
    center = (15, 15)
    core_object = core(position=center)
    workers, rangers, roles = _max_roster()
    extra = unit(2000, "RANGER", position=(70, 20))
    extra_id = str(extra["id"])
    roles[extra_id] = "staged"
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(objects=[core_object, *workers, *rangers, extra])
    strategy = _strategy(memory)

    strategy._reconcile_unit_roles(turn)

    assert memory.unit_roles[extra_id].startswith("patrol-")
    assert strategy._campaign_spawn_candidates(turn) == (UnitType.RANGER,)


def test_forward_patrols_are_all_recalled_when_the_defense_ring_is_hit() -> None:
    center = (15, 15)
    core_object = core(position=center)
    workers, rangers, roles = _max_roster()
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    enemies = [
        unit(3000 + index, "VANGUARD", controlled=False, position=(20 + index, 15))
        for index in range(3)
    ]
    turn = make_turn(objects=[core_object, *workers, *rangers, *enemies])
    strategy = _strategy(memory)

    assert strategy._campaign_patrols_to_recall(turn, False) == frozenset()
    assert strategy._campaign_patrols_to_recall(turn, True) == strategy._patrol_ids(
        turn
    )
    report = strategy.decide(turn)

    for patrol_id in strategy._patrol_ids(turn):
        action = turn.plan.unit_actions.get(patrol_id)
        assert action is not None
        assert action.type == "MOVE"
    assert (
        sum(
            decision.reason == "recall forward patrol to defend the threatened Core"
            for decision in report.decisions
        )
        == 8
    )


def test_legacy_expedition_recall_keeps_fighting_before_it_returns() -> None:
    core_object = core(position=(0, 0))
    recalled = unit(2, "RANGER", position=(30, 0))
    enemy = unit(9, "RANGER", controlled=False, position=(34, 0))
    memory = WorldMemory(
        core_home_position=(0, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        unit_roles={str(recalled["id"]): "expedition-7"},
        expedition_squads=[ExpeditionSquad(7, (str(recalled["id"]),), (1, 0))],
    )
    turn = make_turn(objects=[core_object, recalled, enemy])

    report = _strategy(memory).decide(turn)

    ranger = next(unit for unit in turn.rangers if str(unit.id) == recalled["id"])
    assert turn.plan.unit_actions[ranger.id].type in {"MOVE", "SHOOT"}
    assert any(
        decision.actor_id == recalled["id"] and "expedition" in decision.reason
        for decision in report.decisions
    )
    assert memory.unit_roles[str(ranger.id)] == "expedition-7"


def test_returned_legacy_expedition_joins_the_patrol_roster() -> None:
    center = (15, 15)
    core_object = core(position=center)
    returned = unit(2000, "RANGER", position=(18, 15))
    rangers = _rangers(20)
    roles = {str(ranger["id"]): "defense" for ranger in rangers}
    roles[str(returned["id"])] = "expedition-7"
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
        expedition_squads=[ExpeditionSquad(7, (str(returned["id"]),), (1, 0))],
    )
    turn = make_turn(objects=[core_object, *rangers, returned])

    _strategy(memory).decide(turn)

    assert memory.unit_roles[str(returned["id"])].startswith("patrol-")
    assert memory.expedition_squads == []
