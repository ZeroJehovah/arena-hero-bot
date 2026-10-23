"""Stage-gated campaign economy, formation, and migration behavior."""

from __future__ import annotations

from pathlib import Path

from arena_hero import UnitType, unit_cost

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
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
    _optimized_symmetric_defense_posts,
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


def test_campaign_retargets_only_after_reaching_the_previous_hop() -> None:
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
    turn = make_turn(objects=[current_core, *_workers(12), *rangers])

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


def test_stage_one_spends_on_the_twelfth_worker_without_a_reserve() -> None:
    core_object = core(position=(0, 0))
    memory = WorldMemory(
        core_home_position=(0, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    workers = _workers(11)
    worker_cost = unit_cost(UnitType.WORKER, 11)
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


def test_stage_one_keeps_workers_first_under_core_threat() -> None:
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
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_stage_one_builds_ranger_after_twelve_workers() -> None:
    core_object = core(position=(0, 0))
    memory = WorldMemory(
        core_home_position=(0, 0),
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
    assert turn.plan.core_action.unit_type is UnitType.RANGER


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
    core_object = core(position=(0, 0))
    memory = WorldMemory(
        core_home_position=(0, 0),
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


def test_completed_stage_one_retargets_quota_three_immediately() -> None:
    core_object = core(position=(-2033, -2929))
    memory = WorldMemory(
        core_home_position=(-2033, -2929),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    rangers = _rangers(8)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    turn = make_turn(objects=[core_object, *_workers(12), *rangers])

    _strategy(memory).decide(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE2
    assert memory.campaign_target_quota == 3
    assert memory.core_home_position is not None
    assert blocks_are_cardinally_adjacent(
        resource_block_index(core_object["position"]),
        resource_block_index(memory.core_home_position),
    )


def test_completed_stage_two_switches_to_ranger_only_expedition_and_quota_four() -> (
    None
):
    core_object = core(position=(-2033, -2929))
    memory = WorldMemory(
        core_home_position=(-2033, -2929),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
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

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE3
    assert memory.campaign_target_quota == 4
    assert strategy._formation_quota() == (0, 28, 0, 20)
    assert strategy._expedition_squad_quota() == (0, 4)


def test_campaign_defense_optimizer_uses_twenty_ranger_posts() -> None:
    center = (17, -23)

    posts, radius = _optimized_symmetric_defense_posts(
        center,
        set(),
        vanguard_slots=0,
    )

    assert len(posts) == 20
    assert len(set(posts)) == 20
    assert radius >= 10
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


def test_campaign_arrival_reserves_exactly_one_service_tick() -> None:
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
    assert arrival.plan.core_action is None
    assert memory.core_migration_pause_until_tick == 101

    resumed = make_turn(tick=101, objects=[core_object])
    strategy.decide(resumed)
    assert resumed.plan.core_action is not None
    assert resumed.plan.core_action.type == "START_MOVE"


def test_campaign_never_rests_away_from_a_block_center() -> None:
    """A stale rest flag cannot park the Core between two centers."""

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
    turn = make_turn(tick=100, objects=[core_object, *_workers(12), *rangers])

    report = _strategy(memory).decide(turn)

    assert memory.campaign_resting is False
    assert memory.core_home_position == (-1649, -1904)
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"
    assert not any("hold the block center" in item.reason for item in report.decisions)


def test_campaign_mid_leg_cargo_only_earns_a_short_service_halt() -> None:
    """Cargo near a travelling Core waits one service Tick, not a roster rest."""

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
    turn = make_turn(tick=100, objects=[core_object, *_workers(12), loaded, *rangers])

    report = _strategy(memory).decide(turn)

    assert memory.campaign_resting is False
    assert memory.core_migration_pause_until_tick == 101
    assert turn.plan.core_action is None
    assert any(
        item.reason == "wait briefly for a loaded Worker to enter the service cell"
        for item in report.decisions
    )


def test_quota_two_block_departs_on_the_stage_one_roster() -> None:
    """A quota-2 block only owes the stage-1 economy before the next hop."""

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
    turn = make_turn(tick=100, objects=[core_object, *_workers(12), *rangers])

    _strategy(memory).decide(turn)

    assert memory.campaign_resting is False
    assert memory.core_home_position is not None
    assert memory.core_home_position != center
    assert blocks_are_cardinally_adjacent(
        resource_block_index(center),
        resource_block_index(memory.core_home_position),
    )
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
    turn = make_turn(tick=100, objects=[core_object, *_workers(12), loaded, *rangers])

    report = _strategy(memory).decide(turn)

    assert memory.campaign_resting is True
    assert memory.core_home_position == center
    assert turn.plan.core_action is None
    assert any("hold the block center" in item.reason for item in report.decisions)

    delivered = make_turn(
        tick=101,
        objects=[
            core_object,
            *_workers(12),
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
        objects=[core_object, parked, *_workers(12), loaded, *rangers],
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


def test_quota_three_block_publishes_the_quota_four_hop_with_a_stage_two_roster() -> (
    None
):
    center = (-529, -529)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    rangers = _rangers(28)
    for ranger in rangers[:20]:
        memory.unit_roles[str(ranger["id"])] = "defense"
    for index, ranger in enumerate(rangers[20:]):
        memory.unit_roles[str(ranger["id"])] = f"patrol-{index // 2 + 1}"
    turn = make_turn(tick=100, objects=[core_object, *_workers(16), *rangers])

    _strategy(memory).decide(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE3
    assert memory.campaign_target_quota == 4
    assert memory.campaign_resting is False
    assert memory.core_home_position is not None
    assert blocks_are_cardinally_adjacent(
        resource_block_index(center),
        resource_block_index(memory.core_home_position),
    )
    assert resource_block_ring(
        resource_block_index(memory.core_home_position)
    ) < resource_block_ring(resource_block_index(center))


def test_completed_stage_two_restores_normal_stockpile_reserve() -> None:
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
        > 0
    )
