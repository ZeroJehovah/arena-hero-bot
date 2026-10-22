"""Stage-gated campaign economy, formation, and migration behavior."""

from __future__ import annotations

from pathlib import Path

from arena_hero import UnitType, unit_cost

from arena_hero_bot.memory import ExpeditionSquad, WorldMemory
from arena_hero_bot.migration import (
    plan_resource_quota,
    resource_block_index,
    resource_block_quota,
)
from arena_hero_bot.strategy import (
    CAMPAIGN_LEG_CELLS,
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


def test_explicit_quota_plan_stops_at_outer_guaranteed_tier() -> None:
    memory = WorldMemory(core_home_position=(-2033, -2929))

    tier3 = plan_resource_quota(memory, 3, current_position=(-2033, -2929))
    tier4 = plan_resource_quota(memory, 4, current_position=(-2033, -2929))

    assert tier3 is not None
    assert (tier3.target_ring, tier3.target_quota) == (34, 3)
    assert tier4 is not None
    assert (tier4.target_ring, tier4.target_quota) == (24, 4)


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


def test_campaign_memory_round_trip_preserves_stage_and_leg(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = WorldMemory(
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=object_id(9),
        campaign_leg_steps=17,
        campaign_resting=True,
    )

    memory.save(path)
    restored = WorldMemory.load(path)

    assert restored.campaign_phase == CAMPAIGN_PHASE_STAGE2
    assert restored.campaign_target_quota == 3
    assert restored.campaign_last_core_id == object_id(9)
    assert restored.campaign_leg_steps == 17
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


def test_stage_one_guardless_core_expands_capacity_for_an_unfittable_guard() -> None:
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


def test_stage_one_guardless_core_builds_ranger_once_capacity_allows() -> None:
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
    assert turn.plan.core_action.unit_type is UnitType.RANGER


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
    assert resource_block_quota(resource_block_index(memory.core_home_position)) == 3


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
    core_object = core(position=(1, 0))
    memory = WorldMemory(
        core_home_position=(3, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(0, 0),
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


def test_campaign_enters_a_roster_rest_after_thirty_two_cells() -> None:
    core_object = core(position=(1, 0))
    memory = WorldMemory(
        core_home_position=(100, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(0, 0),
        campaign_phase=CAMPAIGN_PHASE_STAGE1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
        campaign_leg_steps=CAMPAIGN_LEG_CELLS - 1,
    )
    turn = make_turn(tick=100, objects=[core_object])

    report = _strategy(memory).decide(turn)

    assert memory.campaign_resting is True
    assert memory.campaign_leg_steps == 0
    assert turn.plan.core_action is None
    assert any(
        "finish the 32-cell campaign rest" in item.reason for item in report.decisions
    )


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
