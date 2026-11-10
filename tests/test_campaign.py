"""Stage-gated campaign economy, formation, and migration behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from arena_hero import UnitType, core_resource_capacity, unit_cost

from arena_hero_bot.exploration import visible_cells
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
    RANGER_VISION_RADIUS,
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


def test_campaign_alone_advances_and_builds_the_ranger_formation():
    memory = WorldMemory()
    turn = make_turn(
        resources=100,
        objects=[core(position=(16, 16)), *_workers(16), *_rangers(8)],
    )
    strategy = AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))

    strategy.decide(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE3
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.RANGER


@pytest.mark.parametrize("enemy_count", [1, 2])
@pytest.mark.parametrize("resources", [10, 12])
def test_campaign_local_contact_never_builds_worker_or_melee_fallback(
    enemy_count, resources
):
    memory = WorldMemory()
    strategy = _strategy(memory)
    objects = [core(position=(16, 16)), *_workers(3)]
    turn = make_turn(
        resources=resources,
        objects=[
            *objects,
            *[
                unit(900 + index, "RANGER", controlled=False, position=(16 + index, 10))
                for index in range(enemy_count)
            ],
        ],
    )

    strategy.decide(turn)

    action = turn.plan.core_action
    if resources < 12:
        assert action is None or action.type != "SPAWN"
    else:
        assert action is not None
        assert action.type == "SPAWN"
        assert action.unit_type is UnitType.RANGER

    calm = make_turn(tick=101, resources=10, objects=objects)
    strategy.decide(calm)
    assert calm.plan.core_action is not None
    assert calm.plan.core_action.type == "SPAWN"
    assert calm.plan.core_action.unit_type is UnitType.WORKER


@pytest.mark.parametrize("enemy_type", ["WORKER", "CORE"])
def test_campaign_noncombat_contacts_do_not_override_worker_production(enemy_type):
    enemy = (
        core(900, controlled=False, position=(16, 10))
        if enemy_type == "CORE"
        else unit(900, "WORKER", controlled=False, position=(16, 10))
    )
    turn = make_turn(
        resources=12, objects=[core(position=(16, 16)), *_workers(3), enemy]
    )
    _strategy(WorldMemory()).decide(turn)
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def _workers(count: int) -> list[dict[str, object]]:
    return [
        unit(100 + index, "WORKER", position=(20 + index, 20)) for index in range(count)
    ]


@pytest.mark.parametrize("remembered", [False, True])
@pytest.mark.parametrize("enemy_type", ["CORE", "WORKER"])
def test_noncombat_contacts_do_not_trigger_campaign_defense(enemy_type, remembered):
    friendlies = [core(position=(16, 16)), *_workers(3)]
    enemies = [
        core(900 + n, controlled=False, position=(18 + n, 16))
        if enemy_type == "CORE"
        else unit(900 + n, "WORKER", controlled=False, position=(18 + n, 16))
        for n in range(3)
    ]
    strategy = _strategy(WorldMemory())
    strategy.decide(make_turn(tick=99, objects=[*friendlies, *enemies]))
    turn = make_turn(
        objects=friendlies if remembered else [*friendlies, *enemies], resources=10
    )

    report = strategy.decide(turn)

    assert not strategy.memory.defense_compressed
    assert report.threat_level == "NORMAL"
    assert not strategy._emergency_combat_mode(turn)
    assert not strategy._core_assault_enemies(turn)
    assert not strategy._visible_combat_enemies(turn)
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def _rangers(count: int, *, offset: int = 1000) -> list[dict[str, object]]:
    return [
        unit(offset + index, "RANGER", position=(40 + index, 20))
        for index in range(count)
    ]


def _roster(
    count: int = 30,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, str]]:
    workers = _workers(16)
    rangers = _rangers(count)
    defense = (count + 1) // 2
    roles = {
        **{str(worker["id"]): "staged" for worker in workers},
        **{str(ranger["id"]): "defense" for ranger in rangers[:defense]},
        **{
            str(ranger["id"]): f"patrol-{index // 2 + 1}"
            for index, ranger in enumerate(rangers[defense:])
        },
    }
    return workers, rangers, roles


def test_campaign_replaces_a_rock_sealed_block_center() -> None:
    """An unreachable center cannot freeze the Core between two resources."""

    origin = (-1074, -17)
    sealed_center = (-1072, -17)
    core_object = core(9, position=origin)
    workers, rangers, roles = _roster()
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
    workers, rangers, roles = _roster()
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


@pytest.mark.parametrize("under_pressure", [False, True])
@pytest.mark.parametrize("fresh_core", [False, True])
def test_campaign_revalidates_distant_centers_before_planning(
    under_pressure, fresh_core
):
    origin = (-1104, -17) if not fresh_core else (-1086, -17)
    target = (-1072, -17) if not fresh_core else (-1073, -17)
    rocks = {
        (target[0] + dx, target[1] + dy)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
    }
    core_object = core(9, position=origin)
    workers, rangers, roles = _roster()
    memory = (
        WorldMemory()
        if fresh_core
        else WorldMemory(
            core_home_position=target,
            core_layout_version=CORE_LAYOUT_VERSION,
            core_migration_last_position=origin,
            core_migration_start_position=origin,
            campaign_phase=CAMPAIGN_PHASE_STAGE4,
            campaign_target_quota=8,
            campaign_last_core_id=core_object["id"],
            unit_roles=roles,
            unit_roles_initialized=True,
        )
    )
    turn = make_turn(
        objects=[
            core_object,
            *workers,
            *rangers,
            *(
                [
                    unit(
                        900,
                        "RANGER",
                        controlled=False,
                        position=(origin[0], origin[1] + 4),
                    )
                ]
                if under_pressure
                else []
            ),
        ],
        obstacles=rocks,
    )

    _strategy(memory).decide(turn)

    replacement = memory.core_home_position
    assert replacement is not None and replacement != target
    assert resource_block_index(replacement) == resource_block_index(target)
    assert next_step(origin, replacement, blocked=rocks, require_path=True) is not None
    if not fresh_core:
        assert memory.core_migration_start_position == origin


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


@pytest.mark.parametrize("layout_version", [0, CORE_LAYOUT_VERSION])
def test_new_core_releases_migration_when_all_spawn_centers_are_sealed(layout_version):
    origin = (-1740, -1900)
    centers = {(-1745, -1905), (-1744, -1905), (-1745, -1904), (-1744, -1904)}
    rocks = {
        (x + dx, y + dy)
        for x, y in centers
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
    } - centers
    memory = WorldMemory(
        core_home_position=(-2033, -2929),
        core_layout_version=layout_version,
        core_migration_start_position=(-2065, -2929),
        core_arrival_destination=(-2034, -2929),
        core_arrival_worker_id=object_id(2),
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_last_core_id=object_id(1),
    )
    turn = make_turn(objects=[core(9, position=origin)], obstacles=rocks)

    _strategy(memory).decide(turn)

    assert memory.core_home_position is None
    assert memory.core_layout_version == CORE_LAYOUT_VERSION
    assert memory.core_migration_start_position is None
    assert memory.core_arrival_destination is None
    assert memory.core_arrival_worker_id is None
    assert turn.plan.core_action is None or turn.plan.core_action.type != "START_MOVE"


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
    planned = plan_resource_quota(memory, 3, current_position=current_center)
    assert planned is not None
    memory.campaign_scout_rejections = {planned.target_block: 100}
    turn = make_turn(tick=100, objects=[current_core, *_workers(16), *rangers])

    _strategy(memory).decide(turn)

    assert memory.core_home_position is not None
    previous_block = resource_block_index(current_center)
    target_block = resource_block_index(memory.core_home_position)
    assert blocks_are_cardinally_adjacent(previous_block, target_block)
    assert resource_block_ring(target_block) < resource_block_ring(previous_block)
    assert target_block == planned.target_block
    assert not memory.campaign_scout_rejections
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


def test_completed_stage_three_reaches_max_and_targets_quota_four_boundary() -> None:
    core_object = core(position=(-2033, -2929))
    memory = WorldMemory(
        core_home_position=(-2033, -2929),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE3,
        campaign_target_quota=4,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
    )
    workers, rangers, roles = _roster(20)
    memory.unit_roles = roles
    turn = make_turn(objects=[core_object, *workers, *rangers])
    strategy = _strategy(memory)

    strategy._reconcile_unit_roles(turn)
    assert strategy._campaign_defense_rangers(turn) == 10
    assert strategy._campaign_patrol_rangers(turn) == 10
    strategy._advance_campaign_phase(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE4
    assert memory.campaign_target_quota == 4
    assert strategy._formation_quota() == (0, 20, 0, 10)
    assert strategy._expedition_squad_quota() == (0, 4)
    assert memory.core_home_position is not None
    assert blocks_are_cardinally_adjacent(
        resource_block_index(core_object["position"]),
        resource_block_index(memory.core_home_position),
    )


def test_persisted_max_stage_replaces_the_old_fixed_quota_goal() -> None:
    center = (15, 15)
    core_object = core(position=center)
    workers, rangers, roles = _roster()
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(objects=[core_object, *workers, *rangers])

    _strategy(memory)._advance_campaign_phase(turn)

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE4
    assert memory.campaign_target_quota == 4


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


def test_campaign_defense_hold_services_and_compacts_loaded_workers() -> None:
    """A held Core must accept cargo and collect even distant cargo."""

    core_object = core(position=(0, 0))
    memory = WorldMemory(
        core_home_position=(10, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(0, 0),
        core_migration_pause_until_tick=-1,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
    )
    loaded = unit(500, "WORKER", position=(1, 0), cargo=1)
    distant = unit(501, "WORKER", position=(20, 0), cargo=1)
    rangers = [
        unit(1000 + index, "RANGER", position=(-4, index - 2)) for index in range(5)
    ]
    enemy = unit(900, "RANGER", controlled=False, position=(0, 8))
    turn = make_turn(
        tick=100,
        objects=[core_object, *_workers(16), loaded, distant, *rangers, enemy],
    )

    report = _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "WAIT"
    assert any(
        item.actor_kind == "CORE"
        and item.reason == "hold Core behind the two defensive firing ranks"
        for item in report.decisions
    )
    decision = next(
        item for item in report.decisions if item.actor_id == str(loaded["id"])
    )
    assert decision.action == "MOVE"
    assert decision.target == (0, 0)
    assert decision.reason == "return carried resources to Core"
    assert not any(
        item.actor_id == str(loaded["id"])
        and item.reason == "hold the next Core arrival slot for in-transit delivery"
        for item in report.decisions
    )
    distant_decision = next(
        item for item in report.decisions if item.actor_id == str(distant["id"])
    )
    assert distant_decision.action == "MOVE"
    assert distant_decision.target is not None
    assert manhattan(distant_decision.target, core_object["position"]) < 20


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
    workers, rangers, roles = _roster()
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
        and item.reason == "leave Core service cell in simultaneous traffic"
        for item in report.decisions
    )


def test_stationary_campaign_core_clears_service_cell_for_queued_cargo() -> None:
    """Combat recall must not park a healthy guard on the only service slot."""

    center = (-945, -945)
    core_object = core(position=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
    )
    rangers = _rangers(20)
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
    parked = unit(600, "RANGER", position=center)
    loaded = unit(500, "WORKER", position=(-943, -945), cargo=1)
    turn = make_turn(
        tick=100,
        resources=10,
        objects=[core_object, parked, *_workers(10), loaded, *rangers],
    )

    report = _strategy(memory).decide(turn)

    assert memory.campaign_resting is False
    parked_unit = next(unit for unit in turn.rangers if str(unit.id) == parked["id"])
    action = turn.plan.unit_actions[parked_unit.id]
    assert action.type == "MOVE"
    assert any(
        item.actor_id == parked["id"]
        and item.reason == "leave Core service cell in simultaneous traffic"
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
    rangers = _rangers(19)
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
    workers, rangers, roles = _roster()
    # Keep the formation local so the spending assertion is independent of
    # thousands of cells of return-path search under coverage instrumentation.
    for index, actor in enumerate([*workers, *rangers]):
        actor["position"] = [center[0] + 2 + index % 8, center[1] + 2 + index // 8]
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


def test_twenty_rangers_reach_max_and_approach_quota_four() -> None:
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

    assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE4
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
    workers, rangers, roles = _roster()
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


def test_campaign_scout_ignores_cells_sealed_inside_rocks() -> None:
    """A sealed pocket must not stall every sweep into a timeout loop."""

    center = (815, 15)
    core_object = core(position=center)
    workers, rangers, roles = _roster()
    sealed = (799, 0)
    seal = [(800, 0), (799, 1), (798, 0), (799, -1)]
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
        obstacles=seal,
    )
    strategy = _strategy(memory)

    strategy.decide(turn)

    scout_target = memory.campaign_scout_target
    assert scout_target is not None
    assert sealed in memory.campaign_scout_unobservable
    assert memory.campaign_scout_unobservable_target == scout_target
    block = resource_block_index(scout_target)
    for x in range(block[0] * 32, (block[0] + 1) * 32):
        for y in range(block[1] * 32, (block[1] + 1) * 32):
            if (x, y) in memory.obstacles or (x, y) == sealed:
                continue
            memory.campaign_scout_seen_ticks[(x, y)] = turn.tick

    settled = make_turn(
        tick=108,
        objects=[core_object, *workers, *rangers],
        obstacles=seal,
    )
    strategy._update_campaign_scout(settled)

    assert memory.campaign_scout_target is None
    assert memory.core_home_position == scout_target
    assert block not in memory.campaign_scout_rejections


def test_campaign_scout_refreshes_sealed_cells_after_new_rocks() -> None:
    """A pocket enclosed after the sweep starts must stop blocking coverage."""

    center = (815, 15)
    core_object = core(position=center)
    workers, rangers, roles = _roster()
    sealed = (799, 0)
    seal = [(800, 0), (799, 1), (798, 0), (799, -1)]
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
    initial = make_turn(tick=100, objects=[core_object, *workers, *rangers])
    strategy = _strategy(memory)
    strategy.decide(initial)

    scout_target = memory.campaign_scout_target
    assert scout_target is not None
    assert sealed not in memory.campaign_scout_unobservable
    block = resource_block_index(scout_target)
    pending = {scout_target}
    for x in range(block[0] * 32, (block[0] + 1) * 32):
        for y in range(block[1] * 32, (block[1] + 1) * 32):
            if (x, y) != sealed and len(pending) < 8:
                pending.add((x, y))
    for x in range(block[0] * 32, (block[0] + 1) * 32):
        for y in range(block[1] * 32, (block[1] + 1) * 32):
            if (x, y) != sealed and (x, y) not in pending:
                memory.campaign_scout_seen_ticks[(x, y)] = initial.tick

    observed_rocks = make_turn(
        tick=108,
        objects=[core_object, *workers, *rangers],
        obstacles=seal,
    )
    memory.observe(observed_rocks)
    strategy._update_campaign_scout(observed_rocks)

    assert sealed in memory.campaign_scout_unobservable
    assert memory.campaign_scout_target == scout_target
    assert memory.core_home_position == center

    for position in pending:
        memory.campaign_scout_seen_ticks[position] = observed_rocks.tick
    completed = make_turn(
        tick=109,
        objects=[core_object, *workers, *rangers],
        obstacles=seal,
    )
    memory.observe(completed)
    strategy._update_campaign_scout(completed)

    assert memory.campaign_scout_target is None
    assert memory.core_home_position == scout_target
    assert block not in memory.campaign_scout_rejections


def test_campaign_scout_allows_one_unseen_cell_per_four_rangers() -> None:
    """Blind cells may hide at most half the live Ranger force."""

    center = (815, 15)
    target = (784, 15)
    core_object = core(position=center)
    workers, rangers, roles = _roster()
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        campaign_scout_target=target,
        campaign_scout_started_tick=100,
        campaign_scout_last_gain_tick=100,
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(tick=100, objects=[core_object, *workers, *rangers])
    block = resource_block_index(target)
    unseen: list[tuple[int, int]] = []
    for x in range(block[0] * 32, (block[0] + 1) * 32):
        for y in range(block[1] * 32, (block[1] + 1) * 32):
            if len(unseen) < 8:
                unseen.append((x, y))
            else:
                memory.campaign_scout_seen_ticks[(x, y)] = turn.tick

    strategy = _strategy(memory)
    strategy._update_campaign_scout(turn)

    assert len(rangers) // 4 == 7
    assert memory.campaign_scout_target == target
    assert memory.core_home_position == center

    memory.campaign_scout_seen_ticks[unseen[0]] = turn.tick
    strategy._update_campaign_scout(turn)

    assert memory.campaign_scout_target is None
    assert memory.core_home_position == target


def test_campaign_scout_fills_the_gaps_between_anchor_discs() -> None:
    """A staffed anchor grid must not stall a few percent short."""

    center = (-817, 16)
    target = (-785, 15)
    gap = (-793, 0)
    core_object = core(position=center)
    workers = [
        unit(100 + index, "WORKER", position=(-824 + index, -8 - index // 4))
        for index in range(16)
    ]
    defense = [
        unit(
            1000 + index,
            "RANGER",
            position=(center[0] + index % 5, center[1] - 4 - index // 5),
        )
        for index in range(20)
    ]
    anchors = [
        (-796 + 8 * column, 4 + 8 * row) for column in range(4) for row in range(4)
    ]
    patrol = [
        unit(2000 + index, "RANGER", position=anchors[index]) for index in range(16)
    ]
    rangers = [*defense, *patrol]
    roles = {
        **{str(worker["id"]): "staged" for worker in workers},
        **{str(ranger["id"]): "defense" for ranger in defense},
        **{
            str(ranger["id"]): f"patrol-{index // 2 + 1}"
            for index, ranger in enumerate(patrol)
        },
    }
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        campaign_scout_target=target,
        campaign_scout_started_tick=100,
        campaign_scout_last_gain_tick=100,
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(tick=100, objects=[core_object, *workers, *rangers])
    gap_strategy = _strategy(memory)
    block_cells = {(x, y) for x in range(-800, -768) for y in range(32)}
    gaps = gap_strategy._campaign_scout_gaps(turn, target, set(), block_cells)
    unseen = {gap}
    for cell in gaps:
        if len(unseen) >= 10:
            break
        if cell != gap:
            unseen.add(cell)
    for cell in block_cells - unseen:
        memory.campaign_scout_seen_ticks[cell] = 100

    report = _strategy(memory).decide(turn)

    decisions = {decision.actor_id: decision for decision in report.decisions}
    moves = [
        decisions[str(ranger["id"])]
        for ranger in patrol
        if decisions[str(ranger["id"])].action == "MOVE"
    ]
    assert any(decision.target == gap for decision in moves)
    assert memory.campaign_scout_target == target


@pytest.mark.parametrize("age", [0, 50])
def test_campaign_scout_prioritizes_unseen_gaps_over_aging_covered_cells(age):
    target = (-785, 15)
    unseen = (-793, 0)
    memory = WorldMemory(
        campaign_scout_target=target,
        campaign_scout_started_tick=100,
        campaign_scout_seen_ticks={
            (x, y): 150
            for x in range(-800, -768)
            for y in range(32)
            if (x, y) != unseen
        },
    )
    turn = make_turn(
        tick=150 + age,
        objects=[core(position=(-817, 16)), unit(2, "RANGER", position=(-796, 4))],
    )

    goal = _strategy(memory)._campaign_scout_goal(turn.rangers[0], turn)

    assert unseen in visible_cells(goal, RANGER_VISION_RADIUS, set())


@pytest.mark.parametrize("phase", [CAMPAIGN_PHASE_STAGE3, CAMPAIGN_PHASE_STAGE4])
def test_campaign_rock_repair_cannot_sidestep_or_skip_forward_scout(phase):
    origin = (-818, 16)
    sealed_centers = [(-817, 15), (-817, 16), (-816, 15), (-816, 16)]
    workers, rangers, roles = _roster()
    if phase == CAMPAIGN_PHASE_STAGE3:
        rangers = rangers[:20]
        roles = {
            key: role for key, role in roles.items() if not role.startswith("patrol-")
        }
    memory = WorldMemory(
        core_home_position=sealed_centers[0],
        campaign_phase=phase,
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(
        objects=[core(position=origin), *workers, *rangers],
        obstacles=sealed_centers,
    )
    strategy = _strategy(memory)
    memory.observe(turn)

    replacement = strategy._repair_unreachable_home_target(turn, origin)

    assert replacement is None
    assert memory.core_home_position is None
    assert memory.core_migration_start_position is None


@pytest.mark.parametrize("blocked_anchor", [False, True])
def test_scout_finishes_observable_blind_cells_beside_a_blocked_anchor(blocked_anchor):
    center, target, anchor = (815, 15), (783, 15), (772, 4)
    obstacles = {anchor} if blocked_anchor else set()
    unseen = {
        (anchor[0] + dx, anchor[1] + dy)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        if 0 < abs(dx) + abs(dy) <= 2
    }
    workers, rangers, roles = _roster()
    scout = rangers[-1]
    scout["position"] = [776, 8]
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_scout_target=target,
        campaign_scout_started_tick=100,
        campaign_scout_last_gain_tick=100,
        campaign_scout_seen_ticks={
            (x, y): 100
            for x in range(768, 800)
            for y in range(32)
            if (x, y) not in unseen
        },
        obstacles=obstacles,
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    strategy = _strategy(memory)
    assert len(unseen) == 12 > len(rangers) // 4

    for tick in range(101, 121):
        turn = make_turn(
            tick=tick,
            objects=[core(position=center), *workers, *rangers],
            obstacles=obstacles,
        )
        memory.observe(turn)
        strategy._update_campaign_scout(turn)
        if tick == 101:
            assert memory.campaign_scout_target == target
        if memory.campaign_scout_target is None:
            break
        ranger = next(actor for actor in turn.rangers if str(actor.id) == scout["id"])
        goal = strategy._campaign_scout_goal(ranger, turn)
        assert goal != ranger.position
        assert goal not in obstacles
        assert unseen & visible_cells(goal, RANGER_VISION_RADIUS, obstacles)
        step = next_step(ranger.position, goal, blocked=obstacles, require_path=True)
        assert step is not None
        # Observe only the actual one-cell move on the next Tick. Choosing a
        # waypoint must never itself satisfy the coverage gate.
        scout["position"] = list(add(ranger.position, step))

    assert memory.campaign_scout_target is None
    assert memory.core_home_position == target


def test_campaign_scout_reaches_a_far_corner_outside_the_combat_leash() -> None:
    """A full block sweep may cross the generic 60-cell patrol leash."""

    center = (-817, 16)
    target = (-785, 15)
    gap = (-769, 31)
    core_object = core(position=center)
    workers = [
        unit(100 + index, "WORKER", position=(-824 + index, -8 - index // 4))
        for index in range(16)
    ]
    defense = [
        unit(
            1000 + index,
            "RANGER",
            position=(center[0] + index % 5, center[1] + 4 + index // 5),
        )
        for index in range(20)
    ]
    anchors = [
        (-796 + 8 * column, 4 + 8 * row) for column in range(4) for row in range(4)
    ]
    patrol = [
        unit(2000 + index, "RANGER", position=anchors[index]) for index in range(16)
    ]
    obstacles = {
        (-769, 28),
        (-768, 28),
        (-772, 29),
        (-770, 30),
        (-766, 30),
        (-772, 31),
        (-771, 31),
    }
    rangers = [*defense, *patrol]
    roles = {
        **{str(worker["id"]): "staged" for worker in workers},
        **{str(ranger["id"]): "defense" for ranger in defense},
        **{
            str(ranger["id"]): f"patrol-{index // 2 + 1}"
            for index, ranger in enumerate(patrol)
        },
    }
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        campaign_scout_target=target,
        campaign_scout_started_tick=100,
        campaign_scout_last_gain_tick=100,
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(
        tick=100,
        objects=[core_object, *workers, *rangers],
        obstacles=sorted(obstacles),
    )
    gap_strategy = _strategy(memory)
    block_cells = {
        (x, y) for x in range(-800, -768) for y in range(32) if (x, y) not in obstacles
    }
    gaps = gap_strategy._campaign_scout_gaps(
        turn,
        target,
        obstacles,
        block_cells,
    )
    unseen = {gap}
    for cell in gaps:
        if len(unseen) >= 10:
            break
        if cell != gap:
            unseen.add(cell)
    for cell in block_cells - unseen:
        memory.campaign_scout_seen_ticks[cell] = 100

    report = _strategy(memory).decide(turn)

    decisions = {decision.actor_id: decision for decision in report.decisions}
    moves = [
        decisions[str(ranger["id"])]
        for ranger in patrol
        if decisions[str(ranger["id"])].action == "MOVE"
    ]
    assert any(decision.target == gap for decision in moves)
    assert memory.campaign_scout_target == target


def test_campaign_scouts_use_the_full_patrol_leash_before_cover_fire() -> None:
    center = (-817, -16)
    target = (-784, -17)
    core_object = core(position=center)
    workers = [
        unit(100 + index, "WORKER", position=(-824 + index, -8 - index // 4))
        for index in range(16)
    ]
    defense = [
        unit(
            1000 + index,
            "RANGER",
            position=(center[0] + index % 5, center[1] + 4 + index // 5),
        )
        for index in range(15)
    ]
    patrol = [
        unit(2000 + index, "RANGER", position=(-819 + index, -24))
        for index in range(15)
    ]
    rangers = [*defense, *patrol]
    roles = {
        **{str(worker["id"]): "staged" for worker in workers},
        **{str(ranger["id"]): "defense" for ranger in defense},
        **{
            str(ranger["id"]): f"patrol-{index // 2 + 1}"
            for index, ranger in enumerate(patrol)
        },
    }
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        campaign_scout_target=target,
        campaign_scout_started_tick=200,
        campaign_scout_last_gain_tick=200,
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    turn = make_turn(tick=200, objects=[core_object, *workers, *rangers])

    report = _strategy(memory).decide(turn)

    decisions = {decision.actor_id: decision for decision in report.decisions}
    moves = [
        decisions[str(ranger["id"])]
        for ranger in patrol
        if decisions[str(ranger["id"])].action == "MOVE"
    ]
    assert len(moves) == len(patrol)
    assert all(
        decision.target is not None
        and resource_block_index(decision.target) == resource_block_index(target)
        for decision in moves
    )


@pytest.mark.parametrize("casualty", ["worker", "defense", "patrol"])
@pytest.mark.parametrize("restart", [False, True])
def test_campaign_scout_stops_on_roster_loss_before_sweep_finishes(
    tmp_path: Path, casualty: str, restart: bool
) -> None:
    center = (815, 15)
    target = (784, 15)
    core_object = core(position=center)
    workers, rangers, roles = _roster()
    for index, worker in enumerate(workers):
        worker["position"] = [806 + index, 24]
    for index, ranger in enumerate(rangers[:15]):
        ranger["position"] = [811 + index % 5, 8 + index // 5]
    for index, ranger in enumerate(rangers[15:]):
        ranger["position"] = [784 + index, 22]
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_last_core_id=core_object["id"],
        campaign_scout_target=target,
        campaign_scout_started_tick=100,
        campaign_scout_last_gain_tick=100,
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    strategy = _strategy(memory)
    report = strategy.decide(
        make_turn(tick=100, objects=[core_object, *workers, *rangers])
    )
    assert memory.campaign_scout_target == target
    assert any("scout the next" in decision.reason for decision in report.decisions)
    lost = {
        "worker": workers[-1],
        "defense": rangers[0],
        "patrol": rangers[-1],
    }[casualty]
    survivors = [item for item in [*workers, *rangers] if item["id"] != lost["id"]]
    memory_path = tmp_path / "memory.json"
    if restart:
        memory.save(memory_path)
        memory = WorldMemory.load(memory_path)
        strategy = _strategy(memory)

    for tick in (101, 102):
        turn = make_turn(tick=tick, objects=[core_object, *survivors])
        report = strategy.decide(turn)

        assert memory.campaign_scout_target is None
        assert memory.core_home_position == center
        assert memory.campaign_phase == CAMPAIGN_PHASE_STAGE4
        assert memory.campaign_target_quota == 4
        assert (
            turn.plan.core_action is None or turn.plan.core_action.type != "START_MOVE"
        )
        assert not any(
            "scout the next" in decision.reason for decision in report.decisions
        )
        patrol_moves = [
            decision
            for decision in report.decisions
            if roles.get(decision.actor_id, "").startswith("patrol-")
            and decision.action == "MOVE"
        ]
        assert patrol_moves
        assert all(
            decision.target is not None
            and resource_block_index(decision.target) == resource_block_index(center)
            for decision in patrol_moves
        )
        changed = sum(
            memory.unit_roles[str(ranger["id"])] != roles[str(ranger["id"])]
            for ranger in rangers
            if ranger["id"] != lost["id"]
        )
        assert changed == (casualty == "defense")
        if restart:
            memory.save(memory_path)
            memory = WorldMemory.load(memory_path)
            strategy = _strategy(memory)

    strategy.decide(make_turn(tick=103, objects=[core_object, *workers, *rangers]))
    assert memory.campaign_scout_target is not None
    assert memory.campaign_scout_started_tick == 103
    assert memory.core_home_position == center


def test_campaign_scout_waits_instead_of_failing_when_coverage_stalls() -> None:
    target = (-784, -17)
    workers, rangers, roles = _roster()
    memory = WorldMemory(
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_scout_target=target,
        campaign_scout_started_tick=100,
        campaign_scout_last_gain_tick=100,
        unit_roles=roles,
    )

    _strategy(memory)._update_campaign_scout(
        make_turn(
            tick=1000,
            objects=[core(position=(-817, -16)), *workers, *rangers],
        )
    )

    assert memory.campaign_scout_target == target
    assert not memory.campaign_scout_rejections


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
    workers, rangers, roles = _roster(20)
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
        resources=core_resource_capacity(16 + len(rangers)),
        objects=[core_object, *workers, *rangers],
    )

    strategy.decide(full)

    assert full.plan.core_action is not None
    assert full.plan.core_action.type == "SPAWN"
    assert full.plan.core_action.unit_type is UnitType.RANGER


def test_odd_ranger_recruit_fills_defense_without_moving_existing_patrols() -> None:
    center = (15, 15)
    core_object = core(position=center)
    workers, rangers, roles = _roster()
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

    assert memory.unit_roles[extra_id] == "defense"
    assert strategy._campaign_spawn_candidates(turn) == (UnitType.RANGER,)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("restart", [False, True])
def test_incremental_patrol_recruits_fill_stable_pairs(
    tmp_path: Path, batch_size: int, restart: bool
) -> None:
    workers, rangers, roles = _roster()
    roles = {uid: role for uid, role in roles.items() if role != "staged"}
    memory = WorldMemory(
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        unit_roles=roles.copy(),
        unit_roles_initialized=True,
    )
    strategy = _strategy(memory)
    objects = [core(position=(15, 15)), *workers, *rangers]
    path = tmp_path / "memory.json"
    for offset in range(0, 6, batch_size):
        previous = memory.unit_roles.copy()
        objects.extend(
            unit(2000 - index, "RANGER", position=(20 + index, 10))
            for index in range(offset, offset + batch_size)
        )
        strategy._update_expeditions(make_turn(tick=100 + offset, objects=objects))
        assert all(memory.unit_roles[uid] == role for uid, role in previous.items())
        assert not memory.expedition_squads
        if restart:
            memory.save(path)
            memory = WorldMemory.load(path)
            strategy = _strategy(memory)

    patrol_sizes = [
        sum(role == team for role in memory.unit_roles.values())
        for team in ("patrol-8", "patrol-9")
    ]
    assert patrol_sizes == [2, 2]
    assert sum(role == "defense" for role in memory.unit_roles.values()) == 18
    assert all(memory.unit_roles[uid] == role for uid, role in roles.items())


def test_surplus_patrol_recruit_fills_a_casualty_without_reshuffling() -> None:
    workers, rangers, roles = _roster(20)
    roles = {uid: role for uid, role in roles.items() if role != "staged"}
    extra = [
        unit(2000 + index, "RANGER", position=(20 + index, 10)) for index in range(4)
    ]
    roles.update(
        {
            object_id(2000): "patrol-6",
            object_id(2001): "patrol-6",
            object_id(2002): "defense",
            object_id(2003): "defense",
        }
    )
    memory = WorldMemory(
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        unit_roles=roles.copy(),
        unit_roles_initialized=True,
    )
    strategy = _strategy(memory)
    turn = make_turn(
        objects=[
            core(position=(15, 15)),
            *workers,
            *rangers,
            *extra[1:],
            unit(1999, "RANGER", position=(21, 11)),
        ]
    )

    strategy._update_expeditions(turn)

    assert memory.unit_roles == {
        **{uid: role for uid, role in roles.items() if uid != object_id(2000)},
        object_id(1999): "patrol-6",
    }


def test_forward_patrols_are_all_recalled_when_the_defense_ring_is_hit() -> None:
    center = (15, 15)
    core_object = core(position=center)
    workers, rangers, roles = _roster()
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
            decision.reason
            == "reinforce the threatened side from a stable defensive post"
            for decision in report.decisions
        )
        == len(rangers) // 2
    )
    assert all(
        decision.target != center
        for decision in report.decisions
        if decision.actor_id in {str(i) for i in strategy._patrol_ids(turn)}
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


@pytest.mark.parametrize("unit_type", ["RANGER", "VANGUARD"])
def test_returned_legacy_expedition_joins_the_balanced_roster(
    tmp_path: Path, unit_type: str
) -> None:
    center = (15, 15)
    core_object = core(position=center)
    returned = unit(2000, unit_type, position=(18, 15))
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

    assert (
        memory.unit_roles[str(returned["id"])] == "defense"
        if unit_type == "RANGER"
        else memory.unit_roles[str(returned["id"])].startswith("patrol-")
    )
    assert memory.expedition_squads == []
    role = memory.unit_roles[str(returned["id"])]
    path = tmp_path / "memory.json"
    memory.save(path)
    resumed = _strategy(WorldMemory.load(path))
    resumed.decide(make_turn(tick=101, objects=[core_object, *rangers, returned]))
    assert resumed.memory.unit_roles[str(returned["id"])] == role
    assert resumed.memory.expedition_squads == []
    assert resumed._campaign_patrol_rangers(turn) == 10


CLEARANCE_HOME = (-497, 719)
CLEARANCE_NODES = [(-493, 719), (-487, 720)]
CLEARANCE_FIGHTER = (-490, 719)


def _clearance_rangers(count: int) -> list[dict[str, object]]:
    offsets = [(0, 3), (0, -3), (3, 0), (-3, 0), (5, 5), (-5, 5)]
    return [
        unit(
            100 + index,
            "RANGER",
            position=(CLEARANCE_HOME[0] + dx, CLEARANCE_HOME[1] + dy),
        )
        for index, (dx, dy) in enumerate(offsets[:count])
    ]


def _clearance_workers(count: int) -> list[dict[str, object]]:
    return [
        unit(
            300 + index,
            "WORKER",
            position=(
                CLEARANCE_HOME[0] - 8 + index % 8,
                CLEARANCE_HOME[1] - 8 + index // 8,
            ),
        )
        for index in range(count)
    ]


def _sealed_memory(
    core_object: dict[str, object],
    rangers: list[dict[str, object]],
) -> WorldMemory:
    return WorldMemory(
        core_home_position=CLEARANCE_HOME,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        campaign_target_quota=3,
        campaign_last_core_id=str(core_object["id"]),
        unit_roles={str(ranger["id"]): "defense" for ranger in rangers},
        unit_roles_initialized=True,
    )


def _sealed_fighters(count: int) -> list[dict[str, object]]:
    cells = [(-490, 719), (-489, 719), (-490, 720), (-489, 720)]
    return [
        unit(200 + index, "RANGER", controlled=False, position=cell)
        for index, cell in enumerate(cells[:count])
    ]


def test_sealed_block_sends_defenders_to_clear_the_fighters() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    rangers = _clearance_rangers(4)
    memory = _sealed_memory(core_object, rangers)
    turn = make_turn(
        objects=[core_object, *rangers, *_sealed_fighters(1)],
        resource_cells=CLEARANCE_NODES,
    )

    report = _strategy(memory).decide(turn)

    clearing = [
        decision
        for decision in report.decisions
        if "sealing this block" in decision.reason
    ]
    assert clearing
    assert len({decision.actor_id for decision in clearing}) < len(rangers)
    assert memory.core_home_position == CLEARANCE_HOME
    assert memory.campaign_emergency_target is None


def test_sealed_block_without_a_numeric_advantage_keeps_the_defenders_home() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    rangers = _clearance_rangers(4)
    memory = _sealed_memory(core_object, rangers)
    turn = make_turn(
        objects=[core_object, *rangers, *_sealed_fighters(4)],
        resource_cells=CLEARANCE_NODES,
    )

    report = _strategy(memory).decide(turn)

    assert not any(
        "sealing this block" in decision.reason for decision in report.decisions
    )
    assert memory.campaign_emergency_target is None
    assert memory.core_home_position == CLEARANCE_HOME


def test_clearance_counts_stacked_enemies_before_launching():
    core_object = core(position=CLEARANCE_HOME)
    rangers = _clearance_rangers(4)
    strategy = _strategy(_sealed_memory(core_object, rangers))
    turn = make_turn(
        objects=[
            core_object,
            *rangers,
            unit(200, "RANGER", controlled=False, position=CLEARANCE_FIGHTER),
            unit(201, "RANGER", controlled=False, position=CLEARANCE_FIGHTER),
            unit(202, "RANGER", controlled=False, position=(-490, 720)),
        ],
        resource_cells=CLEARANCE_NODES,
    )
    strategy.memory.observe(turn)
    threat = strategy._combat_policy.assess(turn, set())

    # Three available Rangers cannot claim an advantage over three enemies
    # merely because the enemies occupy only two distinct cells.
    assert not strategy._update_resource_clearance(turn, threat)
    assert not strategy._clearance_ids


def test_clearance_recalls_for_unseen_reinforcements_until_confirmed_destroyed():
    core_object = core(position=CLEARANCE_HOME)
    rangers = _clearance_rangers(4)
    strategy = _strategy(_sealed_memory(core_object, rangers))
    pair = [
        unit(200, "RANGER", controlled=False, position=CLEARANCE_FIGHTER),
        unit(201, "RANGER", controlled=False, position=CLEARANCE_FIGHTER),
    ]

    def update(tick, *, reinforcement=False, destroyed=False):
        turn = make_turn(
            tick=tick,
            objects=[
                core_object,
                *rangers,
                *pair,
                *(
                    [unit(202, "RANGER", controlled=False, position=(-490, 720))]
                    if reinforcement
                    else []
                ),
            ],
            resource_cells=CLEARANCE_NODES,
            events=(
                [
                    {
                        "event_id": object_id(900),
                        "tick": tick - 1,
                        "event_type": "DESTRUCTION_PARTICIPATION",
                        "reason_code": "UNIT",
                        "target_id": object_id(202),
                    }
                ]
                if destroyed
                else []
            ),
        )
        strategy.memory.observe(turn)
        threat = strategy._combat_policy.assess(turn, set())
        return strategy._update_resource_clearance(turn, threat)

    assert len(update(100)) == 3
    assert not update(101, reinforcement=True)
    assert not update(102)
    assert len(update(103, destroyed=True)) == 3


def test_sealed_block_with_combat_units_does_not_abandon_the_core() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    rangers = _clearance_rangers(1)
    memory = _sealed_memory(core_object, rangers)
    turn = make_turn(
        objects=[core_object, *rangers, *_sealed_fighters(1)],
        resource_cells=CLEARANCE_NODES,
    )

    _strategy(memory).decide(turn)

    assert memory.campaign_emergency_target is None
    assert memory.core_home_position == CLEARANCE_HOME


def test_sealed_block_without_combat_units_migrates_away_from_the_danger() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    workers = _clearance_workers(16)
    memory = _sealed_memory(core_object, [])
    turn = make_turn(
        objects=[core_object, *workers, *_sealed_fighters(1)],
        resource_cells=CLEARANCE_NODES,
    )

    _strategy(memory).decide(turn)

    target = memory.campaign_emergency_target
    assert target is not None
    origin_block = resource_block_index(CLEARANCE_HOME)
    target_block = resource_block_index(target)
    assert blocks_are_cardinally_adjacent(origin_block, target_block)
    assert resource_block_quota(target_block) <= resource_block_quota(origin_block)
    assert manhattan(target, CLEARANCE_FIGHTER) > manhattan(
        CLEARANCE_HOME, CLEARANCE_FIGHTER
    )
    assert memory.core_home_position == target
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"


@pytest.mark.parametrize("all_centers_sealed", [False, True])
def test_emergency_migration_repairs_its_own_target_or_releases_it(
    tmp_path, all_centers_sealed
):
    core_object = core(position=CLEARANCE_HOME)
    workers = _clearance_workers(16)
    enemies = _sealed_fighters(1)
    memory = _sealed_memory(core_object, [])
    strategy = _strategy(memory)
    first = make_turn(
        objects=[core_object, *workers, *enemies], resource_cells=CLEARANCE_NODES
    )
    strategy.decide(first)
    target = memory.campaign_emergency_target
    assert target is not None
    assert first.plan.core_action is not None
    assert first.plan.core_action.type == "START_MOVE"
    arrived = add(CLEARANCE_HOME, first.plan.core_action.direction)
    core_object["position"] = list(arrived)
    block = resource_block_index(target)
    centers = {
        (block[0] * 32 + x, block[1] * 32 + y) for x in (15, 16) for y in (15, 16)
    }
    rocks = (
        {
            (x + dx, y + dy)
            for x, y in centers
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
        }
        - centers
        if all_centers_sealed
        else {target}
    )
    # The old escape remains committed even after the fighters are destroyed;
    # only new permanent terrain can require replacement or cancellation.
    changed = make_turn(
        tick=104,
        objects=[core_object, *workers],
        obstacles=rocks,
        events=[
            {
                "event_id": object_id(3000 + i),
                "tick": 103,
                "event_type": "UNIT_DESTROYED",
                "target_id": enemy["id"],
            }
            for i, enemy in enumerate(enemies)
        ],
    )
    strategy.decide(changed)

    if all_centers_sealed:
        assert memory.campaign_emergency_target is None
        assert memory.campaign_emergency_started_tick == 0
        assert memory.core_home_position is None
        assert memory.core_migration_start_position is None
    else:
        replacement = memory.campaign_emergency_target
        assert replacement is not None and replacement != target
        assert replacement == memory.core_home_position
        assert resource_block_index(replacement) == block
        assert memory.campaign_emergency_started_tick == 100
        assert memory.core_migration_start_position == CLEARANCE_HOME
        assert (
            next_step(
                arrived,
                replacement,
                blocked=rocks,
                require_path=True,
            )
            is not None
        )

    path = tmp_path / "memory.json"
    memory.save(path)
    resumed = _strategy(WorldMemory.load(path))
    resumed.decide(
        make_turn(tick=105, objects=[core_object, *workers], obstacles=rocks)
    )
    assert resumed.memory.campaign_emergency_target == memory.campaign_emergency_target
    if not all_centers_sealed:
        assert resumed.memory.core_home_position == memory.core_home_position


def test_partially_discovered_block_does_not_trigger_a_clearance() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    rangers = _clearance_rangers(4)
    memory = _sealed_memory(core_object, rangers)
    turn = make_turn(
        objects=[core_object, *rangers, *_sealed_fighters(1)],
        resource_cells=[CLEARANCE_NODES[0]],
    )

    report = _strategy(memory).decide(turn)

    assert not any(
        "sealing this block" in decision.reason for decision in report.decisions
    )
    assert memory.campaign_emergency_target is None


def test_enemy_cores_do_not_seal_a_resource_block() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    workers = _clearance_workers(16)
    memory = _sealed_memory(core_object, [])
    turn = make_turn(
        objects=[
            core_object,
            *workers,
            core(
                7,
                controlled=False,
                owner_username="rival",
                position=CLEARANCE_NODES[0],
            ),
            core(
                8,
                controlled=False,
                owner_username="rival",
                position=CLEARANCE_NODES[1],
            ),
        ],
        resource_cells=[],
    )

    _strategy(memory).decide(turn)

    assert memory.campaign_emergency_target is None
    assert memory.core_home_position == CLEARANCE_HOME


CORE_HUNT_TARGET = (-485, 719)


def _rival_core(position: tuple[int, int] = CORE_HUNT_TARGET) -> dict[str, object]:
    return core(7, controlled=False, owner_username="rival", position=position)


def _core_hunt_reasons(report) -> list:
    return [
        decision for decision in report.decisions if "enemy Core" in decision.reason
    ]


def _hunter_memory(core_object, rangers):
    """Keep the designated hunter defensive in a balanced three-Ranger roster."""

    memory = _sealed_memory(core_object, rangers)
    memory.unit_roles[str(rangers[1]["id"])] = "patrol-1"
    return memory


def test_enemy_core_in_the_block_detaches_one_defender() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    rangers = _clearance_rangers(4)
    memory = _sealed_memory(core_object, rangers)
    turn = make_turn(objects=[core_object, *rangers, _rival_core()])

    report = _strategy(memory).decide(turn)

    hunting = _core_hunt_reasons(report)
    assert hunting
    assert len({decision.actor_id for decision in hunting}) == 1
    assert hunting[0].actor_id in {str(ranger["id"]) for ranger in rangers}
    assert turn.plan.core_action is None or turn.plan.core_action.type != "START_MOVE"
    assert memory.core_home_position == CLEARANCE_HOME
    assert memory.campaign_emergency_target is None


def test_detached_defender_shoots_the_enemy_core_in_range() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    rangers = [
        unit(101, "RANGER", position=(CLEARANCE_HOME[0], CLEARANCE_HOME[1] + 3)),
        unit(102, "RANGER", position=(CLEARANCE_HOME[0], CLEARANCE_HOME[1] - 3)),
        unit(103, "RANGER", position=(CORE_HUNT_TARGET[0] + 2, CORE_HUNT_TARGET[1])),
    ]
    memory = _sealed_memory(core_object, rangers)
    turn = make_turn(objects=[core_object, *rangers, _rival_core()])

    report = _strategy(memory).decide(turn)

    assert any(
        decision.action == "SHOOT" and decision.target == CORE_HUNT_TARGET
        for decision in report.decisions
    )


def test_core_hunter_ignores_bycatch_while_the_core_has_no_ray() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    hunter = unit(103, "RANGER", position=(-484, 721))
    rangers = [
        unit(101, "RANGER", position=(CLEARANCE_HOME[0], CLEARANCE_HOME[1] + 3)),
        unit(102, "RANGER", position=(CLEARANCE_HOME[0], CLEARANCE_HOME[1] - 3)),
        hunter,
    ]
    memory = _hunter_memory(core_object, rangers)
    # The Core is three cells away but on no legal eight-direction ray from
    # the hunter, while the enemy Worker sharing its cell does offer one.
    turn = make_turn(
        objects=[
            core_object,
            *rangers,
            _rival_core(),
            unit(300, "WORKER", controlled=False, position=CORE_HUNT_TARGET),
        ]
    )

    report = _strategy(memory).decide(turn)

    hunter_decisions = [
        decision
        for decision in report.decisions
        if decision.actor_id == str(hunter["id"])
    ]
    assert [decision.action for decision in hunter_decisions] == ["MOVE"]
    assert "enemy Core" in hunter_decisions[0].reason


def test_enemy_core_hunt_is_vetoed_by_a_nearby_fighter() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    rangers = _clearance_rangers(4)
    memory = _sealed_memory(core_object, rangers)
    turn = make_turn(
        objects=[
            core_object,
            *rangers,
            _rival_core(),
            unit(200, "RANGER", controlled=False, position=(-482, 719)),
        ]
    )

    report = _strategy(memory).decide(turn)

    assert not _core_hunt_reasons(report)


def test_enemy_core_hunt_keeps_a_pair_on_station() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    rangers = _clearance_rangers(2)
    memory = _sealed_memory(core_object, rangers)
    turn = make_turn(objects=[core_object, *rangers, _rival_core()])

    report = _strategy(memory).decide(turn)

    assert not _core_hunt_reasons(report)


def test_enemy_core_outside_the_hunt_radius_is_left_alone() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    rangers = _clearance_rangers(4)
    memory = _sealed_memory(core_object, rangers)
    turn = make_turn(
        objects=[core_object, *rangers, _rival_core((CLEARANCE_HOME[0] + 40, 719))]
    )

    report = _strategy(memory).decide(turn)

    assert not _core_hunt_reasons(report)


@pytest.mark.parametrize("enemy_position", [(-494, 719), (-394, 719)])
def test_enemy_core_hunt_aborts_when_a_fighter_appears(enemy_position) -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    rangers = _clearance_rangers(4)
    memory = _sealed_memory(core_object, rangers)
    strategy = _strategy(memory)
    first = make_turn(objects=[core_object, *rangers, _rival_core()], tick=100)

    assert _core_hunt_reasons(strategy.decide(first))

    second = make_turn(
        tick=101,
        objects=[
            core_object,
            *rangers,
            _rival_core(),
            unit(200, "RANGER", controlled=False, position=enemy_position),
            unit(201, "WORKER", position=(enemy_position[0] + 1, enemy_position[1])),
        ],
    )

    assert not _core_hunt_reasons(strategy.decide(second))


def test_core_hunt_hunter_closes_and_destroys_the_core() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    hunter = unit(
        103, "RANGER", position=(CORE_HUNT_TARGET[0] + 4, CORE_HUNT_TARGET[1])
    )
    rangers = [
        unit(101, "RANGER", position=(CLEARANCE_HOME[0], CLEARANCE_HOME[1] + 3)),
        unit(102, "RANGER", position=(CLEARANCE_HOME[0], CLEARANCE_HOME[1] - 3)),
        hunter,
    ]
    memory = _hunter_memory(core_object, rangers)
    strategy = _strategy(memory)

    approach = make_turn(objects=[core_object, *rangers, _rival_core()], tick=100)
    report = strategy.decide(approach)
    closing = [d for d in report.decisions if d.actor_id == str(hunter["id"])]
    assert closing and closing[0].action == "MOVE"
    firing_cell = closing[0].target
    assert firing_cell is not None
    assert manhattan(firing_cell, CORE_HUNT_TARGET) == 3

    hunter["position"] = list(firing_cell)
    in_range = make_turn(objects=[core_object, *rangers, _rival_core()], tick=101)
    report = strategy.decide(in_range)
    assert any(
        decision.actor_id == str(hunter["id"])
        and decision.action == "SHOOT"
        and decision.target == CORE_HUNT_TARGET
        for decision in report.decisions
    )


def test_core_hunt_ends_when_the_last_seen_cell_is_empty() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    hunter = unit(
        103, "RANGER", position=(CORE_HUNT_TARGET[0] + 4, CORE_HUNT_TARGET[1])
    )
    rangers = [
        unit(101, "RANGER", position=(CLEARANCE_HOME[0], CLEARANCE_HOME[1] + 3)),
        unit(102, "RANGER", position=(CLEARANCE_HOME[0], CLEARANCE_HOME[1] - 3)),
        hunter,
    ]
    memory = _hunter_memory(core_object, rangers)
    strategy = _strategy(memory)

    strategy.decide(make_turn(objects=[core_object, *rangers, _rival_core()], tick=100))
    hunter["position"] = [CORE_HUNT_TARGET[0] + 2, CORE_HUNT_TARGET[1]]

    report = strategy.decide(make_turn(objects=[core_object, *rangers], tick=101))

    assert not _core_hunt_reasons(report)
    assert memory.core_home_position == CLEARANCE_HOME


def test_core_hunt_does_not_restart_during_its_cooldown() -> None:
    core_object = core(1, position=CLEARANCE_HOME)
    hunter = unit(
        103, "RANGER", position=(CORE_HUNT_TARGET[0] + 4, CORE_HUNT_TARGET[1])
    )
    rangers = [
        unit(101, "RANGER", position=(CLEARANCE_HOME[0], CLEARANCE_HOME[1] + 3)),
        unit(102, "RANGER", position=(CLEARANCE_HOME[0], CLEARANCE_HOME[1] - 3)),
        hunter,
    ]
    memory = _hunter_memory(core_object, rangers)
    strategy = _strategy(memory)

    assert _core_hunt_reasons(
        strategy.decide(
            make_turn(objects=[core_object, *rangers, _rival_core()], tick=100)
        )
    )
    hunter["position"] = [CORE_HUNT_TARGET[0] + 2, CORE_HUNT_TARGET[1]]
    assert not _core_hunt_reasons(
        strategy.decide(make_turn(objects=[core_object, *rangers], tick=101))
    )

    report = strategy.decide(
        make_turn(objects=[core_object, *rangers, _rival_core()], tick=102)
    )

    assert not _core_hunt_reasons(report)
