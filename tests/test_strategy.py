"""Representative aggressive tactic scenarios."""

from arena_hero import Direction, SpawnAction, UnitType

from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig

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


def test_population_cap_stops_production() -> None:
    units = [unit(number, "RANGER", position=(number, 2)) for number in range(2, 14)]
    turn = make_turn(resources=100, objects=[core(), *units])
    decide(turn, config=StrategyConfig(max_population=12))
    assert turn.plan.core_action is None


def test_exploration_goal_remains_stable_across_turns() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(1, 0))])
    strategy.decide(first)
    goal = memory.goal_for(object_id(2))
    assert goal is not None

    second = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(2, 0))],
    )
    strategy.decide(second)
    assert memory.goal_for(object_id(2)) == goal
