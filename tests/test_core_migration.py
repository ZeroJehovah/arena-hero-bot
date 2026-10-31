"""Long Core migration planning, staging, and economy continuity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from arena_hero import UnitType

from arena_hero_bot import migration_cli
from arena_hero_bot.geometry import manhattan, next_step
from arena_hero_bot.memory import UnitGoal, WorldMemory
from arena_hero_bot.migration import (
    blocks_are_cardinally_adjacent,
    load_migration_request,
    next_resource_tier_ring,
    plan_next_resource_tier,
    plan_resource_cap,
    resource_block_index,
    resource_block_quota,
    resource_block_ring,
    write_migration_request,
)
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    WORKER_PATROL_PURPOSE,
    AggressiveStrategy,
    StrategyConfig,
    _resource_block_center_cells,
    _TurnContext,
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
    assert (plan.target_ring, plan.target_quota) == (31, 3)
    assert (plan.goal_ring, plan.goal_quota) == (24, 4)
    assert plan.target in _resource_block_center_cells(plan.target)
    assert resource_block_index(plan.target) == plan.target_block
    assert blocks_are_cardinally_adjacent(
        plan.current_block,
        plan.target_block,
    )
    assert 30 <= plan.estimated_cells <= 34
    assert plan.blocked_defense_posts == 0


def test_quota_planner_skips_a_center_sealed_by_rocks() -> None:
    """A free center inside a rock cave is not a publishable migration hop."""

    origin = (-2033, -2929)
    memory = WorldMemory(core_home_position=origin)
    first = plan_resource_cap(
        memory,
        target_quota=8,
        stop_ring=8,
        current_position=origin,
    )
    assert first is not None
    x, y = first.target
    memory.obstacles.update({(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)})

    second = plan_resource_cap(
        memory,
        target_quota=8,
        stop_ring=8,
        current_position=origin,
    )

    assert second is not None
    assert second.target != first.target
    assert (
        next_step(
            origin,
            second.target,
            blocked=memory.obstacles,
            require_path=True,
        )
        is not None
    )


def test_lateral_detour_requires_a_reachable_inward_exit() -> None:
    """Do not cross a quota-equivalent block into another scouted dead end."""

    origin = (-817, -16)
    memory = WorldMemory(core_home_position=origin)

    blocked = plan_resource_cap(
        memory,
        target_quota=8,
        stop_ring=8,
        current_position=origin,
        excluded_blocks=frozenset({(-25, -1), (-25, 0)}),
        allow_lateral=True,
    )

    assert blocked is None

    detour = plan_resource_cap(
        memory,
        target_quota=8,
        stop_ring=8,
        current_position=origin,
        excluded_blocks=frozenset({(-25, -1)}),
        allow_lateral=True,
    )

    assert detour is not None
    assert detour.target_block == (-26, 0)


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
        core_migration_start_position=(-400, 432),
        core_migration_steps_since_pause=5,
        core_migration_pause_until_tick=900,
        campaign_scout_last_gain_tick=777,
        campaign_scout_unobservable={(-786, -7)},
        campaign_scout_unobservable_target=(-784, -17),
    )
    path = tmp_path / "memory.json"

    memory.save(path)
    restored = WorldMemory.load(path)

    assert restored.core_migration_last_position == (-369, 432)
    assert restored.core_migration_start_position == (-400, 432)
    assert restored.core_migration_steps_since_pause == 5
    assert restored.core_migration_pause_until_tick == 900
    assert restored.campaign_scout_last_gain_tick == 777
    assert restored.campaign_scout_unobservable == {(-786, -7)}
    assert restored.campaign_scout_unobservable_target == (-784, -17)


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

    assert strategy.request_core_migration((-368, 655)) is True
    assert memory.core_home_position == (-368, 655)
    assert memory.core_migration_last_position is None
    assert memory.core_migration_steps_since_pause == 0
    assert memory.core_migration_pause_until_tick == -1
    assert memory.defense_posts == {}
    assert strategy.request_core_migration((-368, 655)) is False
    with pytest.raises(ValueError, match="cardinal"):
        strategy.request_core_migration((-368, 432))
    with pytest.raises(ValueError, match="center cells"):
        strategy.request_core_migration((0, 0))


def test_arrival_deposit_uses_service_tick_then_resumes_next_tick() -> None:
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
    assert memory.core_migration_pause_until_tick == -1

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
        obstacles=[(1, -1), (1, 1)],
    )

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"


def test_loaded_worker_does_not_block_a_shared_core_step() -> None:
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
        obstacles=[(0, 1), (0, -1), (-1, 0), (1, 1), (1, -1)],
    )

    report = _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"
    assert memory.core_migration_pause_until_tick == -1
    assert not any(
        item.reason == "wait for friendlies to clear Core's next cell"
        for item in report.decisions
    )


def test_critical_units_blocking_core_step_do_not_trigger_an_extra_pause() -> None:
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
        obstacles=[(0, 1), (0, -1), (-1, 0), (1, 1), (1, -1)],
    )

    report = _strategy(memory).decide(turn)

    assert turn.plan.core_action is None
    assert memory.core_migration_pause_until_tick == -1
    assert any(
        item.reason == "wait for friendlies to clear Core's next cell"
        and item.target == (1, 0)
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


def test_loaded_worker_deposits_during_the_arrival_tick_and_core_holds_once() -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(1, 0),
        core_migration_steps_since_pause=7,
    )
    turn = make_turn(objects=[core(), unit(2, "WORKER", position=(0, 0), cargo=1)])

    report = _strategy(memory).decide(turn)

    assert turn.plan.unit_actions[turn.workers[0].id].type == "DEPOSIT"
    assert turn.plan.core_action is None
    assert memory.core_migration_pause_until_tick == -1
    assert any(
        item.reason == "hold one arrival Tick for a completed delivery or heal"
        for item in report.decisions
    )


def test_core_starts_next_step_on_arrival_when_no_unit_service_is_needed() -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(1, 0),
        core_migration_steps_since_pause=7,
    )
    turn = make_turn(objects=[core()])

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"
    assert memory.core_migration_pause_until_tick == -1


def test_core_migration_moves_before_nonurgent_production(monkeypatch) -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(1, 0),
    )
    turn = make_turn(resources=100, objects=[core()])
    strategy = _strategy(memory)
    monkeypatch.setattr(strategy, "_core_can_spawn", lambda context: True)
    monkeypatch.setattr(
        strategy,
        "_choose_spawn",
        lambda current_turn, resources: UnitType.RANGER,
    )

    strategy.decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"


def test_unit_heal_on_arrival_consumes_the_service_tick_before_core_moves() -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(1, 0),
    )
    turn = make_turn(
        resources=1,
        objects=[core(), unit(2, "RANGER", position=(0, 0), hp=1)],
    )

    report = _strategy(memory).decide(turn)

    ranger = turn.rangers[0]
    assert turn.plan.unit_actions[ranger.id].type == "HEAL"
    assert turn.plan.core_action is None
    assert any(
        item.reason == "hold one arrival Tick for a completed delivery or heal"
        for item in report.decisions
    )


def test_core_migration_route_ignores_friendly_arrival_reservations() -> None:
    core_position = (0, 0)
    target = (-1, 0)
    turn = make_turn(objects=[core(position=core_position)])
    strategy = _strategy(
        WorldMemory(
            core_home_position=target,
            core_layout_version=CORE_LAYOUT_VERSION,
        )
    )
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={core_position},
        enemy_positions=set(),
        reserved={target},
        friendly_occupancy={core_position: 1},
        arrival_counts={target: 1},
    )

    direction = strategy._core_home_direction(context)
    assert direction is not None
    assert direction.value == "LEFT"

    context.arrival_counts[target] = 2
    assert strategy._core_home_direction(context) is direction


def test_core_migration_makes_friendly_yield_its_next_cell() -> None:
    core_position = (0, 0)
    destination = (1, 0)
    worker_object = unit(2, "WORKER", position=destination)
    turn = make_turn(
        objects=[core(position=core_position), worker_object],
    )
    strategy = _strategy(
        WorldMemory(
            core_home_position=(3, 0),
            core_layout_version=CORE_LAYOUT_VERSION,
        )
    )

    report = strategy.decide(turn)

    worker = turn.workers[0]
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert action.direction.value in {"UP", "DOWN"}
    assert any(
        item.actor_id == worker_object["id"]
        and item.reason == "yield the Core migration lane"
        for item in report.decisions
    )
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"


def test_core_waits_on_route_when_friendly_cannot_yield_instead_of_detouring() -> None:
    core_position = (0, 0)
    destination = (1, 0)
    core_object = core(position=core_position)
    turn = make_turn(
        objects=[
            core_object,
            unit(2, "WORKER", position=destination),
            unit(3, "WORKER", position=destination),
        ],
        obstacles=[(1, -1), (1, 1)],
    )
    strategy = _strategy(
        WorldMemory(
            core_home_position=(3, 0),
            core_layout_version=CORE_LAYOUT_VERSION,
        )
    )

    report = strategy.decide(turn)

    assert turn.plan.core_action is None
    assert any(
        item.actor_id == core_object["id"]
        and item.action == "WAIT"
        and item.target == destination
        and item.reason == "wait for friendlies to clear Core's next cell"
        for item in report.decisions
    )


def test_full_loaded_core_destination_uses_the_vacated_core_cell() -> None:
    core_object = core()
    destination = (1, 0)
    memory = WorldMemory(
        core_home_position=(3, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(-1, 0),
    )
    turn = make_turn(
        objects=[
            core_object,
            unit(2, "WORKER", position=destination, cargo=1),
            unit(3, "WORKER", position=destination, cargo=1),
        ],
        obstacles=[(1, -1), (1, 1)],
    )

    report = _strategy(memory).decide(turn)

    moving_workers = [
        item
        for item in report.decisions
        if item.actor_kind == "WORKER" and item.action == "MOVE"
    ]
    assert len(moving_workers) == 1
    assert moving_workers[0].reason == (
        "clear the full Core migration destination through the vacated Core cell"
    )
    assert moving_workers[0].target == (0, 0)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "MOVE"
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"


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
        item.reason == "stage the next carried resources on the Core forward route"
        for item in report.decisions
    )


def test_only_one_loaded_worker_stages_at_moving_core_destination() -> None:
    turn = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(2, 0), cargo=1),
            unit(3, "WORKER", position=(0, 1), cargo=1),
        ]
    )

    report = _strategy(WorldMemory()).decide(turn)

    staging = next(worker for worker in turn.workers if worker.position == (2, 0))
    waiting = next(worker for worker in turn.workers if worker.position == (0, 1))
    staging_action = turn.plan.unit_actions[staging.id]
    assert staging_action.type == "MOVE"
    assert staging_action.direction.value == "LEFT"
    waiting_action = turn.plan.unit_actions.get(waiting.id)
    assert waiting_action is None or waiting_action.type in {"MOVE", "WAIT"}
    assert any(
        item.actor_id == str(waiting.id)
        and item.action == "MOVE"
        and item.reason == "queue carried resources on the Core forward route"
        and item.target != (1, 0)
        for item in report.decisions
    )


def test_loaded_workers_queue_on_the_core_forward_route() -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(0, 0),
    )
    turn = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(2, 0), cargo=1),
            unit(3, "WORKER", position=(0, 1), cargo=1),
            unit(4, "WORKER", position=(-1, 0), cargo=1),
            unit(5, "WORKER", position=(0, -1), cargo=1),
        ]
    )

    report = _strategy(memory).decide(turn)

    decisions = {
        item.actor_id: item for item in report.decisions if item.actor_kind == "WORKER"
    }
    staging = next(worker for worker in turn.workers if worker.position == (2, 0))
    assert decisions[str(staging.id)].target == (1, 0)
    assert decisions[str(staging.id)].reason == (
        "stage the next carried resources on the Core forward route"
    )
    queued = [
        decisions[str(worker.id)]
        for worker in turn.workers
        if worker.position != (2, 0)
    ]
    assert sorted(target for item in queued if (target := item.target) is not None) == [
        (2, 0),
        (3, 0),
        (4, 0),
    ]
    assert all(
        item.reason == "queue carried resources on the Core forward route"
        for item in queued
    )


def test_migrating_core_service_tick_queues_cargo_on_the_forward_route() -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(0, 0),
    )
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 0), cargo=1),
            unit(3, "WORKER", position=(0, 1), cargo=1),
            unit(4, "WORKER", position=(-1, 0), cargo=1),
        ]
    )

    report = _strategy(memory).decide(turn)

    depositor = next(worker for worker in turn.workers if worker.position == (0, 0))
    assert turn.plan.unit_actions[depositor.id].type == "DEPOSIT"
    decisions = {
        item.actor_id: item for item in report.decisions if item.actor_kind == "WORKER"
    }
    off_core = [
        decisions[str(worker.id)]
        for worker in turn.workers
        if worker.position != (0, 0)
    ]
    assert all(item.target != (0, 0) for item in off_core)
    assert all(item.target in {(1, 0), (2, 0), (3, 0), (4, 0)} for item in off_core)
    assert any(
        item.reason == "stage the next carried resources on the Core forward route"
        for item in off_core
    )


def test_core_arrival_worker_stays_committed_to_the_same_forward_cell() -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(0, 0),
    )
    first = make_turn(
        tick=100,
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(2, 0), cargo=1),
            unit(3, "WORKER", position=(0, 1), cargo=1),
        ],
    )
    first_report = _strategy(memory).decide(first)
    selected = next(worker for worker in first.workers if worker.position == (2, 0))
    assert memory.core_arrival_worker_id == str(selected.id)
    assert any(
        item.actor_id == str(selected.id)
        and item.reason == "stage the next carried resources on the Core forward route"
        for item in first_report.decisions
    )

    second = make_turn(
        tick=101,
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=3,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(3, 0), cargo=1),
            unit(3, "WORKER", position=(1, 1), cargo=1),
        ],
    )
    second_report = _strategy(memory).decide(second)

    assert memory.core_arrival_worker_id == str(selected.id)
    decisions = {
        item.actor_id: item
        for item in second_report.decisions
        if item.actor_kind == "WORKER"
    }
    assert decisions[str(selected.id)].target == (1, 0)
    assert decisions[str(selected.id)].reason == (
        "stage the next carried resources on the Core forward route"
    )
    closer = next(worker for worker in second.workers if worker.position == (1, 1))
    assert decisions[str(closer.id)].target != (1, 0)


def test_stacked_loaded_workers_separate_when_the_forward_queue_is_full() -> None:
    memory = WorldMemory(
        core_home_position=(8, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(0, 0),
    )
    turn = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(1, 0), cargo=1),
            unit(3, "WORKER", position=(2, 0), cargo=1),
            unit(4, "WORKER", position=(3, 0), cargo=1),
            unit(5, "WORKER", position=(4, 0), cargo=1),
            unit(6, "WORKER", position=(-1, -1), cargo=1),
            unit(7, "WORKER", position=(-1, -1), cargo=1),
        ]
    )

    report = _strategy(memory).decide(turn)

    decisions = {
        item.actor_id: item for item in report.decisions if item.actor_kind == "WORKER"
    }
    stacked = [worker for worker in turn.workers if worker.position == (-1, -1)]
    moved = [
        decisions[str(worker.id)]
        for worker in stacked
        if worker.id in turn.plan.unit_actions
    ]
    assert len(moved) == 1
    assert moved[0].reason == "separate from a shared Core queue waiting cell"
    assert moved[0].target not in {(1, 0), (2, 0), (3, 0), (4, 0), (0, 0)}
    assert moved[0].target is not None
    assert manhattan(moved[0].target, (0, 0)) >= 2


def test_loaded_worker_in_the_next_core_cell_holds_for_the_arrival_tick() -> None:
    turn = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(1, 0), cargo=1),
        ]
    )

    report = _strategy(WorldMemory()).decide(turn)

    worker = turn.workers[0]
    assert worker.id not in turn.plan.unit_actions or (
        turn.plan.unit_actions[worker.id].type == "WAIT"
    )
    assert any(
        item.actor_id == str(worker.id)
        and item.action == "WAIT"
        and item.reason == "hold the next Core arrival slot for in-transit delivery"
        for item in report.decisions
    )


def test_loaded_worker_already_on_the_next_core_cell_is_not_yielded() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0), cargo=1),
        ]
    )
    memory = WorldMemory(
        core_home_position=(2, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=(0, 0),
    )

    report = _strategy(memory).decide(turn)

    worker = turn.workers[0]
    assert any(
        item.actor_id == str(worker.id)
        and item.reason == "hold the next Core arrival slot for in-transit delivery"
        for item in report.decisions
    )
    assert not any(
        item.actor_id == str(worker.id)
        and item.reason == "yield the Core migration lane"
        for item in report.decisions
    )
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction.value == "RIGHT"


def test_loaded_worker_can_cross_clear_space_to_a_stationary_core() -> None:
    memory = WorldMemory(
        core_home_position=(0, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
    )
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 1), cargo=1),
        ],
        obstacles=[(0, 1), (1, 2)],
    )

    report = _strategy(memory).decide(turn)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert any(
        item.action == "MOVE"
        and item.reason == "return carried resources to Core"
        and item.target == (0, 0)
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
        item.reason == "stage the next carried resources on the Core forward route"
        for item in report.decisions
    )


def test_unloaded_worker_routes_out_of_the_locked_core_approach() -> None:
    core_object = core(position=(0, 0))
    worker_object = unit(2, "WORKER", position=(-1, -1))
    memory = WorldMemory(
        core_home_position=(1, 1),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=1,
        campaign_target_quota=2,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
        core_migration_last_position=(0, 0),
        core_migration_pause_until_tick=-1,
        campaign_resting=False,
    )
    memory.set_goal(
        worker_object["id"],
        UnitGoal(
            position=(7, 0),
            assigned_tick=90,
            purpose=WORKER_PATROL_PURPOSE,
            last_progress_position=(-1, -1),
        ),
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
        tick=100,
        objects=[core_object, worker_object],
    )

    report = strategy.decide(turn)

    worker = turn.workers[0]
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    core_view = turn.core
    assert core_view is not None
    destination = (
        worker.position[0] + action.direction.delta[0],
        worker.position[1] + action.direction.delta[1],
    )
    assert manhattan(destination, core_view.position) > manhattan(
        worker.position, core_view.position
    )
    assert not any(
        item.actor_id == worker_object["id"] and item.action == "WAIT"
        for item in report.decisions
    )


def test_campaign_rest_allows_an_adjacent_resource_harvest() -> None:
    """A Core holding its block center must not lock its four neighbours."""

    center = (-945, -945)
    core_object = core(position=center)
    worker_object = unit(2, "WORKER", position=(-944, -945))
    loaded_worker = unit(500, "WORKER", position=(-943, -945), cargo=1)
    rangers = [unit(200 + index, "RANGER") for index in range(8)]
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=2,
        campaign_target_quota=3,
        campaign_last_core_id=core_object["id"],
        unit_roles_initialized=True,
        core_migration_last_position=center,
        core_migration_pause_until_tick=-1,
    )
    for ranger in rangers:
        memory.unit_roles[str(ranger["id"])] = "defense"
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
        tick=100,
        objects=[
            core_object,
            worker_object,
            loaded_worker,
            *_workers(),
            *rangers,
        ],
        resource_cells=[(-944, -945)],
    )

    strategy.decide(turn)

    # A nearby delivery keeps the completed stage-two center in its service
    # stance.  The adjacent resource cell remains claimable for another Worker.
    assert memory.campaign_resting is True
    worker = next(unit for unit in turn.workers if str(unit.id) == worker_object["id"])
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "HARVEST"


def test_worker_harvests_resource_blocking_the_core_next_migration_step() -> None:
    """A migration approach cannot evict the Worker clearing its next cell."""

    core_object = core(position=(0, 0))
    worker_object = unit(2, "WORKER", position=(1, 0))
    memory = WorldMemory(
        core_home_position=(1, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
    )
    turn = make_turn(
        objects=[core_object, worker_object],
        obstacles=[(0, 1), (0, -1), (1, 1), (1, -1)],
        resource_cells=[(1, 0)],
    )

    report = _strategy(memory).decide(turn)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "HARVEST"
    assert turn.plan.core_action is None
    assert any(
        item.actor_id == worker_object["id"]
        and item.action == "HARVEST"
        and item.reason == "harvest current visible resource"
        for item in report.decisions
    )


def test_economy_expands_to_current_and_target_chunks_during_migration() -> None:
    memory = WorldMemory(
        core_home_position=(64, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
    )
    strategy = _strategy(memory)
    moving = make_turn(objects=[core(position=(40, 5))])

    chunk = strategy._resource_chunk(moving)
    route = strategy._patrol_ring_route(moving)
    sectors = strategy._resource_patrol_sectors(moving)

    assert chunk is not None
    assert chunk.contains((40, 5))
    assert chunk.contains((64, 0))
    assert len(sectors) == 32
    assert route
    assert all(manhattan(cell, (40, 5)) == 18 for cell in route)


def test_economy_keeps_the_source_chunk_after_crossing_into_the_target() -> None:
    memory = WorldMemory(
        core_home_position=(64, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_start_position=(32, 0),
    )
    strategy = _strategy(memory)
    crossing = make_turn(objects=[core(position=(70, 0))])

    chunk = strategy._resource_chunk(crossing)
    sectors = strategy._resource_patrol_sectors(crossing)

    assert chunk is not None
    assert chunk.contains((32, 0))
    assert chunk.contains((64, 0))
    assert len(sectors) == 32


def test_stale_far_migration_start_does_not_expand_the_economy_area() -> None:
    memory = WorldMemory(
        core_home_position=(64, 0),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_start_position=(4096, 4096),
    )
    strategy = _strategy(memory)
    turn = make_turn(objects=[core(position=(70, 0))])

    chunk = strategy._resource_chunk(turn)
    sectors = strategy._resource_patrol_sectors(turn)

    assert chunk is not None
    assert chunk.contains((64, 0))
    assert not chunk.contains((4096, 4096))
    assert len(sectors) == 16


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
    assert plan["target_ring"] == 31
    assert plan["target_quota"] == 3
    assert plan["goal_ring"] == 24
    assert plan["goal_quota"] == 4

    migration_cli.main(["--data-dir", str(tmp_path), "request-next-tier"])
    request = json.loads(capsys.readouterr().out)
    assert (tmp_path / "core-migration.json").exists()

    migration_cli.main(["--data-dir", str(tmp_path), "status"])
    status = json.loads(capsys.readouterr().out)
    assert status["requested_target"] == request["target"]
    assert status["last_core_position"] == [-400, 655]
    assert status["remaining_estimated_cells"] > 0
