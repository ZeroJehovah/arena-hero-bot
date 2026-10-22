"""Long Core migration planning, staging, and economy continuity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena_hero_bot import migration_cli
from arena_hero_bot.geometry import manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.migration import (
    load_migration_request,
    next_resource_tier_ring,
    plan_next_resource_tier,
    resource_block_index,
    resource_block_quota,
    resource_block_ring,
    write_migration_request,
)
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    CORE_MIGRATION_PAUSE_MAX_TICKS,
    AggressiveStrategy,
    StrategyConfig,
    _resource_block_center_cells,
)

from .factories import core, make_turn, unit


def _strategy(memory: WorldMemory) -> AggressiveStrategy:
    return AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=16,
            max_population=None,
            expedition_mode=True,
        ),
    )


def _workers() -> list[dict[str, object]]:
    return [
        unit(100 + index, "WORKER", position=(20 + index, 20)) for index in range(16)
    ]


def test_resource_block_quota_boundary_and_next_tier() -> None:
    assert resource_block_index((-400, 655)) == (-13, 20)
    assert resource_block_ring((-13, 20)) == 32
    assert resource_block_quota((-13, 20)) == 3
    assert resource_block_quota((-12, 13)) == 4
    assert next_resource_tier_ring(32) == 24
    assert next_resource_tier_ring(0) is None


def test_next_tier_planner_selects_a_clean_center_block() -> None:
    memory = WorldMemory(core_home_position=(-400, 655))

    plan = plan_next_resource_tier(memory)

    assert plan is not None
    assert (plan.current_ring, plan.current_quota) == (32, 3)
    assert (plan.target_ring, plan.target_quota) == (24, 4)
    assert plan.target in _resource_block_center_cells(plan.target)
    assert resource_block_index(plan.target) == plan.target_block
    assert 254 <= plan.estimated_cells <= 320
    assert plan.blocked_defense_posts == 0


def test_migration_request_round_trip_is_atomic_and_private(tmp_path: Path) -> None:
    path = tmp_path / "control" / "core-migration.json"
    target = (-368, 432)

    written = write_migration_request(path, target, note="test")
    loaded = load_migration_request(path)

    assert loaded is not None
    assert loaded.target == target
    assert loaded.note == "test"
    assert loaded.requested_at == written.requested_at
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text()) == {
        "version": 1,
        "target": [-368, 432],
        "requested_at": written.requested_at,
        "note": "test",
    }
    with pytest.raises(ValueError, match="center cells"):
        write_migration_request(path, (0, 0))


def test_migration_request_rejects_bad_payloads(tmp_path: Path) -> None:
    path = tmp_path / "core-migration.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_migration_request(path)
    path.write_text('{"version": 9, "target": [-368, 432]}')
    with pytest.raises(ValueError, match="unsupported"):
        load_migration_request(path)
    path.write_text('{"version": 1, "target": [0, 0]}')
    with pytest.raises(ValueError, match="center cells"):
        load_migration_request(path)


def test_migration_schedule_survives_memory_round_trip(tmp_path: Path) -> None:
    memory = WorldMemory(
        core_home_position=(-368, 432),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-369, 432),
        core_migration_steps_since_pause=5,
        core_migration_pause_until_tick=900,
    )
    path = tmp_path / "memory.json"

    memory.save(path)
    restored = WorldMemory.load(path)

    assert restored.core_migration_last_position == (-369, 432)
    assert restored.core_migration_steps_since_pause == 5
    assert restored.core_migration_pause_until_tick == 900


def test_committed_migration_resets_identity_and_tuning_state() -> None:
    memory = WorldMemory(
        core_home_position=(-400, 655),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-399, 655),
        core_migration_steps_since_pause=7,
        core_migration_pause_until_tick=120,
        defense_posts={"unit": (1, 1)},
        defense_anchor=(-400, 655),
    )
    strategy = _strategy(memory)

    assert strategy.request_core_migration((-368, 432)) is True
    assert memory.core_home_position == (-368, 432)
    assert memory.core_migration_last_position is None
    assert memory.core_migration_steps_since_pause == 0
    assert memory.core_migration_pause_until_tick == -1
    assert memory.defense_posts == {}
    assert strategy.request_core_migration((-368, 432)) is False
    with pytest.raises(ValueError, match="center cells"):
        strategy.request_core_migration((0, 0))


def test_arrival_deposit_uses_one_tick_before_next_core_move() -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(1, 0),
        core_migration_steps_since_pause=7,
    )
    strategy = _strategy(memory)
    workers = _workers()
    workers[0] = unit(100, "WORKER", position=(0, 0), cargo=1)
    turn = make_turn(tick=100, objects=[core(), *workers])

    strategy.decide(turn)

    assert turn.plan.unit_actions[turn.workers[0].id].type == "DEPOSIT"
    assert turn.plan.core_action is None
    assert memory.core_migration_pause_until_tick == 101

    workers[0]["cargo"] = 0
    resumed = make_turn(tick=101, objects=[core(), *workers])
    strategy.decide(resumed)

    assert resumed.plan.core_action is not None
    assert resumed.plan.core_action.type == "START_MOVE"


def test_migration_does_not_pause_every_eight_steps() -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(1, 0),
        core_migration_steps_since_pause=7,
    )

    turn = make_turn(tick=100, objects=[core(), *_workers()])
    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert memory.core_migration_pause_until_tick == -1


def test_core_steps_into_a_cell_with_one_friendly_unit() -> None:
    memory = WorldMemory(
        core_home_position=(3, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(0, 0),
    )
    turn = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
        ],
    )

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"


def test_unstaged_loaded_worker_opens_a_short_arrival_wait() -> None:
    core_object = core()
    memory = WorldMemory(
        core_home_position=(3, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-1, 0),
        core_migration_steps_since_pause=4,
        position_history={core_object["id"]: [(0, 1), (0, 0)]},
    )
    turn = make_turn(
        tick=100,
        objects=[
            core_object,
            unit(2, "WORKER", position=(1, 0), cargo=1),
            unit(3, "WORKER", position=(1, 0)),
        ],
        obstacles=[(0, 1), (0, -1), (-1, 0), (1, 1), (1, -1), (2, 0)],
    )

    report = _strategy(memory).decide(turn)

    assert turn.plan.core_action is None
    assert memory.core_migration_pause_until_tick == 101
    assert any(
        item.reason == "wait briefly for a loaded Worker to enter the service cell"
        for item in report.decisions
    )


def test_critical_unit_blocking_core_step_opens_immediate_service_halt() -> None:
    core_object = core()
    memory = WorldMemory(
        core_home_position=(3, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-1, 0),
        core_migration_steps_since_pause=4,
        position_history={core_object["id"]: [(0, 1), (0, 0)]},
    )
    turn = make_turn(
        tick=100,
        objects=[
            core_object,
            unit(2, "RANGER", position=(1, 0), hp=1),
            unit(3, "WORKER", position=(1, 0)),
        ],
        obstacles=[(0, 1), (0, -1), (-1, 0), (1, 1), (1, -1), (2, 0)],
    )

    report = _strategy(memory).decide(turn)

    assert turn.plan.core_action is None
    assert memory.core_migration_pause_until_tick == (
        100 + CORE_MIGRATION_PAUSE_MAX_TICKS
    )
    assert any(
        item.reason == "hold for a critical unit blocking the Core approach"
        for item in report.decisions
    )


def test_migration_pause_clears_non_service_unit_from_core_cell() -> None:
    memory = WorldMemory(
        core_home_position=(3, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(0, 0),
        core_migration_steps_since_pause=5,
        core_migration_pause_until_tick=200,
    )
    ranger_object = unit(3, "RANGER", position=(0, 0))
    turn = make_turn(
        tick=100,
        objects=[
            core(),
            ranger_object,
            unit(2, "WORKER", position=(1, 0), cargo=1),
        ],
    )

    report = _strategy(memory).decide(turn)

    ranger = next(unit for unit in turn.rangers if str(unit.id) == ranger_object["id"])
    action = turn.plan.unit_actions.get(ranger.id)
    assert action is not None
    assert action.type == "MOVE"
    assert any(
        item.actor_id == ranger_object["id"]
        and item.reason == "leave Core service cell for unique traffic"
        for item in report.decisions
    )


def test_loaded_worker_deposits_during_the_arrival_tick() -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(1, 0),
        core_migration_steps_since_pause=7,
    )
    turn = make_turn(objects=[core(), unit(2, "WORKER", position=(0, 0), cargo=1)])

    _strategy(memory).decide(turn)

    assert turn.plan.unit_actions[turn.workers[0].id].type == "DEPOSIT"
    assert turn.plan.core_action is None
    assert memory.core_migration_pause_until_tick == turn.tick + 1


def test_loaded_worker_stages_at_moving_core_destination() -> None:
    moving = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(5, 0), cargo=1),
        ]
    )
    _strategy(WorldMemory()).decide(moving)
    action = moving.plan.unit_actions[moving.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction.value == "LEFT"

    near = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(2, 0), cargo=1),
        ]
    )
    report = _strategy(WorldMemory()).decide(near)
    action = near.plan.unit_actions[near.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction.value == "LEFT"
    assert any(
        item.reason == "stage carried resources at the Core arrival cell"
        for item in report.decisions
    )


def test_loaded_worker_routes_around_the_locked_core_approach() -> None:
    moving = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="UP",
                move_progress=2,
                move_required_ticks=4,
                destination=[0, -1],
            ),
            unit(2, "WORKER", position=(1, 0), cargo=1),
        ],
        obstacles=[(1, 1)],
    )

    report = _strategy(WorldMemory()).decide(moving)

    action = moving.plan.unit_actions[moving.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction.value == "UP"
    assert any(
        item.reason == "stage carried resources at the Core arrival cell"
        for item in report.decisions
    )


def test_economy_and_patrol_follow_the_current_core_chunk() -> None:
    memory = WorldMemory(
        core_home_position=(64, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
    )
    strategy = _strategy(memory)
    moving = make_turn(objects=[core(position=(40, 5))])

    chunk = strategy._resource_chunk(moving)
    route = strategy._patrol_ring_route(moving)

    assert chunk is not None
    assert chunk.contains((40, 5))
    assert not chunk.contains((64, 0))
    assert route
    assert all(manhattan(cell, (40, 5)) == 18 for cell in route)


def test_worker_harvests_in_current_chunk_during_long_migration() -> None:
    memory = WorldMemory(
        core_home_position=(40, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
    )
    turn = make_turn(
        objects=[core(), unit(2, "WORKER", position=(2, 0))],
        resource_cells=[(2, 0)],
    )

    _strategy(memory).decide(turn)

    assert turn.plan.unit_actions[turn.workers[0].id].type == "HARVEST"


def test_migration_cli_plans_requests_and_reports(tmp_path: Path, capsys) -> None:
    WorldMemory(core_home_position=(-400, 655)).save(tmp_path / "memory.json")

    migration_cli.main(["--data-dir", str(tmp_path), "plan"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["target_ring"] == 24
    assert plan["target_quota"] == 4

    migration_cli.main(["--data-dir", str(tmp_path), "request-next-tier"])
    request = json.loads(capsys.readouterr().out)
    assert (tmp_path / "core-migration.json").exists()

    migration_cli.main(["--data-dir", str(tmp_path), "status"])
    status = json.loads(capsys.readouterr().out)
    assert status["requested_target"] == request["target"]
    assert status["last_core_position"] == [-400, 655]
    assert status["remaining_estimated_cells"] > 0
