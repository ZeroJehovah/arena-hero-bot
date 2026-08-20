"""Representative aggressive tactic scenarios."""

from itertools import pairwise

from arena_hero import Direction, SpawnAction, UnitType

from arena_hero_bot.geometry import manhattan
from arena_hero_bot.memory import UnitGoal, WorldMemory
from arena_hero_bot.strategy import (
    AggressiveStrategy,
    StrategyConfig,
    _resource_patrol_offsets,
)

from .factories import core, make_turn, object_id, unit


def decide(turn, memory=None, config=None):
    strategy = AggressiveStrategy(memory or WorldMemory(), config)
    return strategy.decide(turn)


def test_ranger_prioritizes_and_shoots_visible_core() -> None:
    turn = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 1)),
            unit(3, "WORKER", controlled=False, position=(1, 1)),
            core(
                4,
                controlled=False,
                owner_username="rival",
                position=(0, 3),
                hp=3,
                shield=0,
            ),
        ],
    )
    report = decide(turn)

    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type == "SHOOT"
    assert action.expected_cell == (0, 3)
    assert any(item.action == "SHOOT" for item in report.decisions)


def test_ranger_does_not_shoot_through_remembered_obstacle() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 0)),
            core(
                3,
                controlled=False,
                owner_username="rival",
                position=(0, 3),
            ),
        ],
        obstacles=[(0, 2)],
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.rangers[0].id].type == "MOVE"


def test_vanguard_sweeps_every_adjacent_hostile() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "VANGUARD", position=(1, 0)),
            unit(3, "WORKER", controlled=False, position=(2, 0)),
            unit(4, "RANGER", controlled=False, position=(2, 0)),
        ]
    )
    decide(turn)
    action = turn.plan.unit_actions[turn.vanguards[0].id]
    assert action.type == "SWEEP"
    assert action.direction is Direction.RIGHT


def test_worker_harvests_then_returns_cargo() -> None:
    harvesting = make_turn(
        objects=[core(), unit(2, "WORKER", position=(1, 0))],
        resource_cells=[(1, 0)],
    )
    decide(harvesting)
    assert harvesting.plan.unit_actions[harvesting.workers[0].id].type == "HARVEST"

    returning = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(1, 0), cargo=1)],
    )
    decide(returning)
    action = returning.plan.unit_actions[returning.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT


def test_worker_deposits_at_core_and_waits_when_storage_full() -> None:
    deposit = make_turn(
        resources=5,
        objects=[core(), unit(2, "WORKER", position=(0, 0), cargo=1)],
    )
    decide(deposit)
    assert deposit.plan.unit_actions[deposit.workers[0].id].type == "DEPOSIT"

    full = make_turn(
        resources=10,
        objects=[core(), unit(2, "WORKER", position=(0, 0), cargo=1)],
    )
    report = decide(full)
    assert full.workers[0].id not in full.plan.unit_actions
    assert any(item.reason == "Core storage is full" for item in report.decisions)


def test_only_one_worker_reserves_the_core_cell() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0), cargo=1),
            unit(3, "WORKER", position=(0, 1), cargo=1),
        ]
    )
    report = decide(turn)
    moving_home = [
        worker
        for worker in turn.workers
        if turn.plan.unit_actions.get(worker.id) is not None
        and turn.plan.unit_actions[worker.id].type == "MOVE"
    ]
    assert len(moving_home) == 1
    assert any(
        "Core cell is not currently reachable" in item.reason
        for item in report.decisions
    )


def test_empty_worker_vacates_core_before_returning_worker_moves() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0), cargo=1),
            unit(3, "WORKER", position=(0, 0)),
        ]
    )
    decide(turn)

    returning = turn.workers[0]
    action = turn.plan.unit_actions[returning.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT


def test_core_does_not_spawn_into_worker_reserved_cell() -> None:
    turn = make_turn(
        resources=10,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0), cargo=1),
            unit(3, "WORKER", position=(0, 0)),
        ],
    )
    decide(turn)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert turn.plan.core_action is None


def test_worker_retreats_from_nearby_visible_enemy() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(2, 0)),
            unit(3, "VANGUARD", controlled=False, position=(3, 0)),
        ]
    )
    report = decide(turn)

    worker = turn.workers[0]
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )


def test_worker_can_retreat_into_core_cell_when_threatened() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "VANGUARD", controlled=False, position=(2, 0)),
        ]
    )
    decide(turn)

    worker = turn.workers[0]
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT


def test_worker_retreat_prefers_distance_from_enemy_over_shortest_path() -> None:
    worker_id = str(object_id(2))
    memory = WorldMemory(
        obstacles={
            (-423, 824),
            (-422, 823),
            (-421, 822),
            (-418, 825),
            (-423, 828),
            (-421, 828),
            (-420, 828),
            (-418, 828),
            (-419, 830),
            (-422, 829),
            (-421, 830),
        },
        position_history={
            worker_id: [(-422, 824), (-421, 824), (-420, 824), (-419, 824)]
        },
    )
    turn = make_turn(
        objects=[
            core(position=(-521, 882)),
            unit(2, "WORKER", position=(-418, 824)),
            unit(3, "WORKER", controlled=False, position=(-415, 824)),
            unit(4, "RANGER", controlled=False, position=(-415, 824)),
        ]
    )
    decide(turn, memory=memory)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT


def test_cargo_worker_does_not_oscillate_around_enemy() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 2), cargo=1),
            unit(3, "RANGER", controlled=False, position=(-1, -3)),
        ],
    )
    strategy.decide(first)
    first_action = first.plan.unit_actions[first.workers[0].id]
    assert first_action.type == "MOVE"
    assert first_action.direction is Direction.DOWN

    second = make_turn(
        tick=101,
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 3), cargo=1),
            unit(3, "RANGER", controlled=False, position=(-1, -3)),
        ],
    )
    strategy.decide(second)
    second_action = second.plan.unit_actions[second.workers[0].id]
    assert second_action.type == "MOVE"
    assert second_action.direction is not Direction.UP

    third = make_turn(
        tick=102,
        objects=[
            core(),
            unit(2, "WORKER", position=(-1, 3), cargo=1),
            unit(3, "RANGER", controlled=False, position=(-1, -3)),
        ],
    )
    report = strategy.decide(third)
    action = third.plan.unit_actions[third.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is not Direction.RIGHT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )

    looping = make_turn(
        tick=103,
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 3), cargo=1),
            unit(3, "RANGER", controlled=False, position=(-1, -3)),
        ],
    )
    strategy.decide(looping)
    loop_action = looping.plan.unit_actions[looping.workers[0].id]
    assert loop_action.type == "MOVE"
    assert loop_action.direction is not Direction.LEFT


def test_worker_does_not_retreat_from_noncombat_enemy_worker() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(5, 0)),
            unit(3, "WORKER", controlled=False, position=(6, 0)),
        ]
    )
    report = decide(turn)

    assert not any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )


def test_worker_retreats_from_visible_enemy_core() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(5, 0)),
            core(
                3,
                controlled=False,
                owner_username="rival",
                position=(6, 0),
            ),
        ]
    )
    report = decide(turn)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )


def test_worker_retreats_from_recently_seen_enemy_core() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(5, 0)),
            core(
                3,
                controlled=False,
                owner_username="rival",
                position=(8, 0),
            ),
        ],
    )
    strategy.decide(first)

    second = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(5, 0))],
    )
    report = strategy.decide(second)

    action = second.plan.unit_actions[second.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )

    still_remembered = make_turn(
        tick=125,
        objects=[core(), unit(2, "WORKER", position=(5, 0))],
    )
    remembered_report = strategy.decide(still_remembered)
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in remembered_report.decisions
    )

    long_term = make_turn(
        tick=261,
        objects=[core(), unit(2, "WORKER", position=(5, 0))],
    )
    long_term_report = strategy.decide(long_term)
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in long_term_report.decisions
    )


def test_cargo_worker_retreats_from_visible_combat_enemy() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(2, 0), cargo=1),
            unit(3, "RANGER", controlled=False, position=(3, 0)),
        ]
    )
    report = decide(turn)

    worker = turn.workers[0]
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )


def test_worker_retreats_from_recently_seen_combat_enemy() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 1), cargo=1),
            unit(3, "VANGUARD", controlled=False, position=(8, 1)),
        ],
    )
    strategy.decide(first)

    second = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(5, 1), cargo=1)],
    )
    report = strategy.decide(second)

    action = second.plan.unit_actions[second.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is not Direction.RIGHT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )

    expired = make_turn(
        tick=125,
        objects=[core(), unit(2, "WORKER", position=(5, 1), cargo=1)],
    )
    expired_report = strategy.decide(expired)
    assert not any(
        item.reason == "retreat from visible enemy pressure"
        for item in expired_report.decisions
    )


def test_only_lowest_uuid_worker_harvests_contested_resource() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(1, 0)),
        ],
        resource_cells=[(1, 0)],
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "HARVEST"
    assert turn.plan.unit_actions.get(turn.workers[1].id) is None or (
        turn.plan.unit_actions[turn.workers[1].id].type != "HARVEST"
    )


def test_nearest_worker_is_assigned_visible_resource() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 5)),
            unit(3, "WORKER", position=(9, 0)),
        ],
        resource_cells=[(10, 0)],
    )
    report = decide(turn)

    near_worker = turn.workers[1]
    action = turn.plan.unit_actions[near_worker.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.RIGHT
    claims = [
        item
        for item in report.decisions
        if item.reason == "claim nearest unassigned visible resource"
    ]
    assert [(item.actor_id, item.target) for item in claims] == [
        (str(near_worker.id), (10, 0))
    ]


def test_worker_vacates_core_and_core_spawns_second_worker() -> None:
    turn = make_turn(
        resources=5,
        objects=[core(), unit(2, "WORKER", position=(0, 0))],
        resource_cells=[(1, 0)],
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "MOVE"
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_core_builds_aggressive_composition() -> None:
    vanguard_turn = make_turn(
        resources=10,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(-1, 0)),
        ],
    )
    decide(vanguard_turn)
    assert isinstance(vanguard_turn.plan.core_action, SpawnAction)
    assert vanguard_turn.plan.core_action.unit_type is UnitType.VANGUARD

    ranger_turn = make_turn(
        resources=12,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(-1, 0)),
            unit(4, "VANGUARD", position=(0, 1)),
        ],
    )
    decide(ranger_turn)
    assert isinstance(ranger_turn.plan.core_action, SpawnAction)
    assert ranger_turn.plan.core_action.unit_type is UnitType.RANGER


def test_core_prioritizes_survival_under_pressure() -> None:
    heal_turn = make_turn(resources=10, objects=[core(hp=2)])
    decide(heal_turn)
    assert heal_turn.plan.core_action is not None
    assert heal_turn.plan.core_action.type == "HEAL"

    shield_turn = make_turn(
        resources=10,
        objects=[
            core(shield=1),
            unit(2, "VANGUARD", controlled=False, position=(2, 0)),
        ],
    )
    decide(shield_turn)
    assert shield_turn.plan.core_action is not None
    assert shield_turn.plan.core_action.type == "REPAIR_SHIELD"


def test_same_tick_deposit_can_fund_core_healing() -> None:
    turn = make_turn(
        resources=0,
        objects=[core(hp=2), unit(2, "WORKER", position=(0, 0), cargo=3)],
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "DEPOSIT"
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "HEAL"


def test_critical_ranger_heals_at_stationary_core() -> None:
    turn = make_turn(
        resources=4,
        objects=[core(), unit(2, "RANGER", position=(0, 0), hp=1)],
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.rangers[0].id].type == "HEAL"


def test_critical_vanguard_returns_to_core_before_attacking() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "VANGUARD", position=(2, 0), hp=2),
            unit(3, "RANGER", controlled=False, position=(3, 0)),
        ],
    )

    report = decide(turn)

    action = turn.plan.unit_actions[turn.vanguards[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "return critical unit to Core for healing"
        for item in report.decisions
    )


def test_core_and_unit_pick_up_ground_beacon() -> None:
    unit_turn = make_turn(
        objects=[core(), unit(2, "VANGUARD", position=(1, 0))],
        beacon={"position": [1, 0], "status": "GROUND"},
    )
    decide(unit_turn)
    assert unit_turn.plan.unit_actions[unit_turn.vanguards[0].id].type == (
        "PICKUP_BEACON"
    )

    core_turn = make_turn(
        objects=[core()], beacon={"position": [0, 0], "status": "GROUND"}
    )
    decide(core_turn)
    assert core_turn.plan.core_action is not None
    assert core_turn.plan.core_action.type == "PICKUP_BEACON"


def test_unit_beacon_claim_prevents_duplicate_core_pickup() -> None:
    turn = make_turn(
        resources=20,
        objects=[core(), unit(2, "VANGUARD", position=(0, 0))],
        beacon={"position": [0, 0], "status": "GROUND"},
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.vanguards[0].id].type == "PICKUP_BEACON"
    assert turn.plan.core_action is None


def test_combat_unit_hunts_recent_enemy_after_losing_vision() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 1)),
            core(
                3,
                controlled=False,
                owner_username="rival",
                position=(5, 1),
            ),
        ],
    )
    strategy.decide(first)

    second = make_turn(
        tick=101,
        objects=[core(), unit(2, "RANGER", position=(1, 1))],
    )
    report = strategy.decide(second)
    action = second.plan.unit_actions[second.rangers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.RIGHT
    assert any("last seen core" in item.reason for item in report.decisions)


def test_respawning_state_submits_no_invented_actions() -> None:
    turn = make_turn(
        status="RESPAWNING",
        respawn_at_tick=101,
        objects=[],
        resources=0,
    )
    report = decide(turn)
    assert turn.plan.unit_actions == {}
    assert turn.plan.core_action is None
    assert report.decisions == []


def test_moving_core_does_not_queue_illegal_action() -> None:
    turn = make_turn(
        resources=20,
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            )
        ],
    )
    report = decide(turn)
    assert turn.plan.core_action is None
    assert report.decisions[0].reason == "Core migration is already progressing"


def test_idle_core_migrates_toward_resource_rich_center() -> None:
    turn = make_turn(
        objects=[
            core(position=(-10, 10)),
            unit(2, "WORKER", position=(-12, 12)),
            unit(3, "WORKER", position=(-13, 13)),
        ]
    )
    report = decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction in {Direction.UP, Direction.RIGHT}
    assert any(
        item.reason == "advance mobile base toward the resource-rich center"
        for item in report.decisions
    )


def test_core_migration_avoids_immediate_backtrack() -> None:
    core_id = str(object_id(1))
    memory = WorldMemory(
        obstacles={
            (-10, 518),
            (-8, 514),
            (-8, 517),
            (-8, 521),
            (-7, 521),
            (-6, 485),
            (-6, 519),
            (-6, 521),
            (-5, 484),
            (-5, 517),
            (-4, 488),
            (-2, 471),
            (-2, 485),
            (-2, 486),
            (-1, 469),
            (1, 469),
        },
        position_history={core_id: [(-9, 518)]},
    )
    turn = make_turn(
        tick=126628,
        resources=9,
        objects=[
            core(position=(-8, 518)),
            unit(2, "VANGUARD", position=(-5, 485)),
            unit(3, "WORKER", position=(0, 468)),
            unit(4, "WORKER", position=(-9, 514)),
        ],
        obstacles=[],
    )
    decide(turn, memory=memory)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction is Direction.RIGHT


def test_core_stays_receptive_while_worker_returns_cargo() -> None:
    turn = make_turn(
        objects=[
            core(position=(-10, 10)),
            unit(2, "WORKER", position=(-12, 12), cargo=1),
            unit(3, "WORKER", position=(-13, 13)),
        ]
    )
    decide(turn)

    assert turn.plan.core_action is None


def test_worker_stages_cargo_at_migrating_core_destination() -> None:
    turn = make_turn(
        objects=[
            core(
                position=(0, 0),
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(2, 0), cargo=1),
        ]
    )

    report = decide(turn)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "stage carried resources at migrating Core destination"
        for item in report.decisions
    )


def test_worker_never_deposits_while_core_is_migrating() -> None:
    turn = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=3,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(0, 0), cargo=1),
        ]
    )

    report = decide(turn)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.RIGHT
    assert action.type != "DEPOSIT"
    assert any(
        item.reason == "stage carried resources at migrating Core destination"
        for item in report.decisions
    )


def test_population_cap_stops_production() -> None:
    units = [unit(number, "RANGER", position=(number, 2)) for number in range(2, 14)]
    turn = make_turn(resources=100, objects=[core(), *units])
    decide(turn, config=StrategyConfig(max_population=12))
    assert turn.plan.core_action is None


def test_unbounded_population_keeps_expanding_past_legacy_cap() -> None:
    units = [unit(number, "RANGER", position=(number, 2)) for number in range(2, 22)]
    turn = make_turn(resources=100, objects=[core(), *units])

    decide(
        turn,
        config=StrategyConfig(
            target_workers=0,
            max_population=None,
            safety_reserve=10,
        ),
    )

    assert turn.state.population == 20
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_unbounded_population_preserves_emergency_resource_reserve() -> None:
    workers = [
        unit(2, "WORKER", position=(1, 0)),
        unit(3, "WORKER", position=(2, 0)),
    ]
    turn = make_turn(resources=14, objects=[core(), *workers])

    decide(
        turn,
        config=StrategyConfig(
            target_workers=12,
            max_population=None,
            safety_reserve=10,
        ),
    )

    assert turn.plan.core_action is None


def test_unbounded_population_clears_full_core_cell_for_expansion() -> None:
    units = [unit(number, "VANGUARD", position=(number, 2)) for number in range(3, 14)]
    units.extend(
        unit(number, "WORKER", position=(number, 3)) for number in range(14, 21)
    )
    units.append(unit(2, "WORKER", position=(0, 0), cargo=1))
    turn = make_turn(resources=95, objects=[core(), *units])

    decide(
        turn,
        config=StrategyConfig(
            target_workers=12,
            max_population=None,
            safety_reserve=10,
        ),
    )

    worker = next(item for item in turn.workers if str(item.id) == object_id(2))
    assert turn.plan.unit_actions[worker.id].type == "MOVE"
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"


def test_resource_goal_builds_one_capacity_unit_beyond_required_population() -> None:
    units = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 20)]
    turn = make_turn(resources=5, objects=[core(), *units])

    decide(
        turn,
        config=StrategyConfig(
            target_workers=20,
            max_population=20,
            resource_target=95,
        ),
    )

    assert turn.state.population == 18
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_resource_goal_stops_production_at_buffered_capacity() -> None:
    units = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 22)]
    turn = make_turn(resources=100, objects=[core(), *units])

    decide(
        turn,
        config=StrategyConfig(
            target_workers=20,
            max_population=20,
            resource_target=95,
        ),
    )

    assert turn.state.population == 20
    assert turn.plan.core_action is None


def test_resource_goal_reserves_for_one_guard_then_resumes_workers() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=20,
            max_population=20,
            resource_target=95,
        ),
    )
    workers = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 8)]
    nearby_enemy = core(
        20,
        controlled=False,
        owner_username="rival",
        position=(20, 0),
    )

    saving = make_turn(
        tick=100,
        resources=5,
        objects=[core(), *workers, nearby_enemy],
    )
    strategy.decide(saving)
    assert saving.plan.core_action is None

    guarded = make_turn(
        tick=101,
        resources=10,
        objects=[core(), *workers, nearby_enemy],
    )
    strategy.decide(guarded)
    assert guarded.plan.core_action is not None
    assert guarded.plan.core_action.type == "SPAWN"
    assert guarded.plan.core_action.unit_type is UnitType.VANGUARD

    growing = make_turn(
        tick=102,
        resources=5,
        objects=[
            core(),
            *workers,
            unit(8, "VANGUARD", position=(2, 3)),
            nearby_enemy,
        ],
    )
    strategy.decide(growing)
    assert growing.plan.core_action is not None
    assert growing.plan.core_action.type == "SPAWN"
    assert growing.plan.core_action.unit_type is UnitType.WORKER


def test_resource_goal_builds_first_guard_before_full_worker_target() -> None:
    workers = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 8)]
    turn = make_turn(
        resources=10,
        objects=[core(), *workers],
    )

    decide(
        turn,
        config=StrategyConfig(
            target_workers=12,
            max_population=20,
            resource_target=95,
        ),
    )

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_resource_goal_remembers_nearby_enemy_core_for_guard_reserve() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=20,
            max_population=20,
            resource_target=95,
        ),
    )
    workers = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 8)]
    observed = make_turn(
        tick=100,
        resources=5,
        objects=[
            core(),
            *workers,
            core(
                20,
                controlled=False,
                owner_username="rival",
                position=(20, 0),
            ),
        ],
    )
    strategy.decide(observed)

    hidden = make_turn(tick=261, resources=5, objects=[core(), *workers])
    strategy.decide(hidden)

    assert hidden.plan.core_action is None


def test_resource_goal_keeps_core_stationary_and_preserves_stockpile() -> None:
    turn = make_turn(
        resources=95,
        objects=[
            core(position=(100, 100), shield=4),
            unit(2, "WORKER", position=(101, 100)),
        ],
    )

    decide(turn, config=StrategyConfig(resource_target=95, max_population=20))

    assert turn.plan.core_action is None


def test_resource_goal_retargets_distant_worker_to_local_patrol() -> None:
    memory = WorldMemory()
    turn = make_turn(
        tick=160,
        objects=[
            core(position=(100, 100)),
            unit(2, "WORKER", position=(0, 0)),
        ],
    )

    decide(
        turn,
        memory=memory,
        config=StrategyConfig(resource_target=95, resource_patrol_radius=10),
    )

    goal = memory.goal_for(object_id(2))
    assert goal is not None
    assert goal.purpose == "resource-patrol-v3"
    assert 0 < manhattan(goal.position, (100, 100)) <= 20
    assert turn.plan.unit_actions[turn.workers[0].id].type == "MOVE"


def test_resource_patrol_rotates_immediately_after_reaching_goal() -> None:
    memory = WorldMemory()
    config = StrategyConfig(resource_target=95, resource_patrol_radius=10)
    strategy = AggressiveStrategy(memory, config)
    first = make_turn(
        tick=160,
        objects=[core(position=(100, 100)), unit(2, "WORKER", position=(90, 90))],
    )
    strategy.decide(first)
    first_goal = memory.goal_for(object_id(2))
    assert first_goal is not None

    arrived = make_turn(
        tick=161,
        objects=[
            core(position=(100, 100)),
            unit(2, "WORKER", position=first_goal.position),
        ],
    )
    strategy.decide(arrived)
    next_goal = memory.goal_for(object_id(2))

    assert next_goal is not None
    assert next_goal.position != first_goal.position
    assert arrived.plan.unit_actions[arrived.workers[0].id].type == "MOVE"


def test_resource_patrol_assigns_distinct_initial_sectors() -> None:
    memory = WorldMemory()
    turn = make_turn(
        tick=160,
        objects=[
            core(position=(100, 100)),
            unit(2, "WORKER", position=(100, 100)),
            unit(10, "WORKER", position=(101, 100)),
            unit(18, "WORKER", position=(99, 100)),
        ],
    )

    decide(
        turn,
        memory=memory,
        config=StrategyConfig(resource_target=95, resource_patrol_radius=10),
    )

    goals = set()
    for worker in turn.workers:
        goal = memory.goal_for(str(worker.id))
        assert goal is not None
        goals.add(goal.position)
    assert len(goals) == 3


def test_resource_patrol_splits_duplicate_existing_goals() -> None:
    memory = WorldMemory()
    shared_goal = UnitGoal((12, 12), 100, "resource-patrol-v3")
    memory.set_goal(object_id(2), shared_goal)
    memory.set_goal(object_id(3), shared_goal)
    turn = make_turn(
        tick=101,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(-1, 0)),
        ],
    )

    decide(
        turn,
        memory=memory,
        config=StrategyConfig(resource_target=95, resource_patrol_radius=18),
    )

    goals = [memory.goal_for(object_id(number)) for number in (2, 3)]
    assert all(goal is not None for goal in goals)
    assert len({goal.position for goal in goals if goal is not None}) == 2


def test_resource_patrol_replaces_goal_near_remembered_enemy_core() -> None:
    memory = WorldMemory()
    config = StrategyConfig(
        resource_target=95,
        resource_patrol_radius=18,
        worker_threat_radius=2,
    )
    strategy = AggressiveStrategy(memory, config)
    observed = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER"),
            core(3, controlled=False, owner_username="rival", position=(6, 0)),
        ],
    )
    strategy.decide(observed)
    memory.set_goal(
        object_id(2),
        UnitGoal((6, 0), observed.tick, "resource-patrol-v3"),
    )

    hidden = make_turn(
        tick=125,
        objects=[core(), unit(2, "WORKER", position=(-12, -18))],
    )
    strategy.decide(hidden)

    replacement = memory.goal_for(object_id(2))
    assert replacement is not None
    assert replacement.position != (6, 0)
    assert replacement.position != hidden.workers[0].position
    assert manhattan(replacement.position, (6, 0)) > 2


def test_worker_path_avoids_remembered_enemy_core_exclusion() -> None:
    memory = WorldMemory()
    config = StrategyConfig(
        resource_target=95,
        resource_patrol_radius=18,
        worker_threat_radius=2,
    )
    strategy = AggressiveStrategy(memory, config)
    observed = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(3, 0)),
            core(3, controlled=False, owner_username="rival", position=(6, 0)),
        ],
    )
    strategy.decide(observed)
    memory.set_goal(
        object_id(2),
        UnitGoal((10, 0), observed.tick, "resource-patrol-v3"),
    )

    hidden = make_turn(
        tick=125,
        objects=[core(), unit(2, "WORKER", position=(3, 0))],
    )
    strategy.decide(hidden)

    action = hidden.plan.unit_actions[hidden.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is not Direction.RIGHT


def test_worker_path_temporarily_avoids_recent_combat_enemy() -> None:
    memory = WorldMemory()
    config = StrategyConfig(
        resource_target=95,
        resource_patrol_radius=18,
        worker_threat_radius=2,
    )
    strategy = AggressiveStrategy(memory, config)
    observed = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(3, 0)),
            unit(3, "VANGUARD", controlled=False, position=(6, 0)),
        ],
    )
    strategy.decide(observed)
    memory.set_goal(
        object_id(2),
        UnitGoal((10, 0), observed.tick, "resource-patrol-v3"),
    )

    hidden = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(3, 0))],
    )
    strategy.decide(hidden)
    recent_action = hidden.plan.unit_actions[hidden.workers[0].id]
    assert recent_action.type == "MOVE"
    assert recent_action.direction is not Direction.RIGHT

    expired = make_turn(
        tick=125,
        objects=[core(), unit(2, "WORKER", position=(3, 0))],
    )
    strategy.decide(expired)
    expired_action = expired.plan.unit_actions[expired.workers[0].id]
    assert expired_action.type == "MOVE"
    assert expired_action.direction is Direction.RIGHT


def test_resource_patrol_route_covers_square_without_vision_gaps() -> None:
    radius = StrategyConfig().resource_patrol_radius
    offsets = _resource_patrol_offsets(radius=radius, spacing=6)

    assert radius == 30
    assert len(offsets) == 120
    assert len(set(offsets)) == len(offsets)
    assert all(max(abs(x), abs(y)) <= radius for x, y in offsets)
    assert all(manhattan(left, right) <= 12 for left, right in pairwise(offsets))


def test_strategy_avoids_recently_contested_resource_patrol_cell() -> None:
    memory = WorldMemory()
    config = StrategyConfig(resource_target=95, resource_patrol_radius=18)
    strategy = AggressiveStrategy(memory, config)
    first = make_turn(
        tick=160,
        objects=[core(position=(100, 100)), unit(2, "WORKER", position=(100, 100))],
    )
    strategy.decide(first)
    first_action = first.plan.unit_actions[first.workers[0].id]
    assert first_action.type == "MOVE"
    contested = (
        first.workers[0].position[0] + first_action.direction.delta[0],
        first.workers[0].position[1] + first_action.direction.delta[1],
    )
    memory.contested_positions[contested] = first.tick

    second = make_turn(
        tick=161,
        objects=[core(position=(100, 100)), unit(2, "WORKER", position=(100, 100))],
    )
    strategy.decide(second)
    second_action = second.plan.unit_actions[second.workers[0].id]

    assert second_action.type == "MOVE"
    assert second_action.direction != first_action.direction


def test_worker_retreats_from_recently_contested_area() -> None:
    memory = WorldMemory(contested_positions={(6, 0): 160})
    turn = make_turn(
        tick=161,
        objects=[core(position=(0, 0)), unit(2, "WORKER", position=(5, 0))],
    )

    report = decide(
        turn,
        memory=memory,
        config=StrategyConfig(resource_target=95),
    )

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any("retreat" in item.reason for item in report.decisions)


def test_exploration_goal_remains_stable_across_turns() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    goal = memory.goal_for(object_id(2))
    assert goal is not None

    second = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(99, 100))],
    )
    strategy.decide(second)
    assert memory.goal_for(object_id(2)) == goal


def test_exploration_goal_renews_after_progress_beyond_ttl() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    first_goal = memory.goal_for(object_id(2))
    assert first_goal is not None

    progressed = make_turn(
        tick=first.tick + strategy.config.exploration_goal_ttl + 1,
        objects=[core(), unit(2, "WORKER", position=(99, 100))],
    )
    strategy.decide(progressed)
    renewed = memory.goal_for(object_id(2))

    assert renewed is not None
    assert renewed.position == first_goal.position
    assert renewed.assigned_tick == progressed.tick
    assert renewed.last_progress_position == (99, 100)


def test_legacy_exploration_goal_initializes_progress_baseline() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    initial = memory.goal_for(object_id(2))
    assert initial is not None
    memory.set_goal(
        object_id(2),
        UnitGoal(initial.position, initial.assigned_tick, initial.purpose),
    )

    baseline = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(99, 100))],
    )
    strategy.decide(baseline)
    migrated = memory.goal_for(object_id(2))
    assert migrated is not None
    assert migrated.last_progress_position == (99, 100)

    progressed = make_turn(
        tick=baseline.tick + strategy.config.exploration_goal_ttl,
        objects=[core(), unit(2, "WORKER", position=(98, 100))],
    )
    strategy.decide(progressed)
    renewed = memory.goal_for(object_id(2))
    assert renewed is not None
    assert renewed.position == initial.position
    assert renewed.assigned_tick == progressed.tick


def test_exploration_goal_is_replaced_when_target_becomes_obstacle() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    initial = memory.goal_for(object_id(2))
    assert initial is not None

    blocked = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(99, 100))],
        obstacles=[initial.position],
    )
    strategy.decide(blocked)
    replacement = memory.goal_for(object_id(2))

    assert replacement is not None
    assert replacement.position != initial.position
    assert replacement.position not in memory.obstacles


def test_exploration_goal_expires_after_worker_stalls() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    first_goal = memory.goal_for(object_id(2))
    assert first_goal is not None

    stalled = make_turn(
        tick=first.tick + strategy.config.exploration_goal_ttl + 1,
        objects=[core(), unit(2, "WORKER", position=(100, 100))],
    )
    strategy.decide(stalled)
    replacement = memory.goal_for(object_id(2))

    assert replacement is not None
    assert replacement.position != first_goal.position
    assert replacement.assigned_tick == stalled.tick


def test_exploration_goal_uses_aggressive_default_stride() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    turn = make_turn(
        tick=0,
        objects=[core(), unit(2, "WORKER", position=(-100, 100))],
    )

    strategy.decide(turn)

    goal = memory.goal_for(object_id(2))
    assert goal is not None
    assert goal.position == (-76, 76)
    assert goal.purpose == "explore-center-v3"


def test_reaching_exploration_goal_immediately_advances_toward_center() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    first_goal = memory.goal_for(object_id(2))
    assert first_goal is not None

    reached = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=first_goal.position)],
    )
    strategy.decide(reached)
    next_goal = memory.goal_for(object_id(2))

    assert next_goal is not None
    assert next_goal.position != first_goal.position
    assert abs(next_goal.position[0]) < abs(first_goal.position[0])
    assert abs(next_goal.position[1]) < abs(first_goal.position[1])
    assert reached.plan.unit_actions[reached.workers[0].id].type == "MOVE"
