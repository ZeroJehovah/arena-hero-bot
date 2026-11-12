"""Core targets interrupt patrol, retain identity and yield to real pressure."""

from __future__ import annotations

from uuid import UUID

import pytest
from arena_hero import Direction

from arena_hero_bot.exploration import visible_cells
from arena_hero_bot.geometry import line_of_fire, manhattan
from arena_hero_bot.memory import CoreHunt, WorldMemory
from arena_hero_bot.runtime import _identity_snapshot
from arena_hero_bot.strategy import (
    CAMPAIGN_PHASE_STAGE2,
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, object_id, unit
from .test_core_traffic import settled_positions

HOME = (16, 16)
TARGET = (7, 2)


@pytest.fixture(autouse=True)
def require_complete_hunt_plans(monkeypatch):
    def deadline_reached(*_args):
        pytest.fail("Core hunt scenario exhausted the planning deadline")

    monkeypatch.setattr(AggressiveStrategy, "_wait_unplanned_units", deadline_reached)


def _roster():
    return [
        unit(2, "RANGER", position=(7, 6)),
        unit(3, "RANGER", position=(16, 18)),
        unit(4, "RANGER", position=(18, 16)),
        unit(5, "RANGER", position=(16, 6)),
    ]


def _memory():
    return WorldMemory(
        core_home_position=HOME,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_last_core_id=object_id(1),
        campaign_phase=CAMPAIGN_PHASE_STAGE2,
        unit_roles={
            object_id(2): "patrol-1",
            object_id(3): "defense",
            object_id(4): "defense",
            object_id(5): "patrol-1",
        },
        unit_roles_initialized=True,
    )


def _strategy(memory):
    return AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))


def _rival(position=TARGET, **movement):
    return core(900, controlled=False, position=position, **movement)


def _hunt_actions(report):
    return [d for d in report.decisions if "enemy Core" in d.reason]


def _advance(roster, turn):
    positions = settled_positions(turn)
    for actor in roster:
        actor["position"] = list(positions[turn.unit(actor["id"]).id])


@pytest.mark.parametrize("restart", [False, True])
def test_patrol_discovery_takes_one_step_then_fires_at_the_core(tmp_path, restart):
    memory = _memory()
    roster = _roster()
    strategy = _strategy(memory)
    before_roles = dict(memory.unit_roles)
    first = make_turn(
        tick=100,
        objects=[
            core(position=HOME),
            *roster,
            _rival(),
            unit(901, "WORKER", controlled=False, position=TARGET),
        ],
    )

    report = strategy.decide(first)

    ranger = first.rangers[0]
    action = first.plan.unit_actions[ranger.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.UP
    assert settled_positions(first)[ranger.id] == (7, 5)
    assert any(d.actor_id == object_id(2) for d in _hunt_actions(report))
    assert memory.unit_roles == before_roles
    slot = memory.patrol_ring_slots[object_id(2)]
    _advance(roster, first)
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = _strategy(memory)
    second = make_turn(
        tick=101,
        objects=[
            core(position=HOME),
            *roster,
            _rival(),
            unit(901, "WORKER", controlled=False, position=TARGET),
        ],
    )

    strategy.decide(second)

    shot = second.plan.unit_actions[ranger.id]
    assert shot.type == "SHOOT"
    assert shot.target_id == UUID(object_id(900))
    assert shot.expected_cell == TARGET
    assert memory.unit_roles == before_roles
    assert memory.patrol_ring_slots[object_id(2)] == slot


def test_far_unrelated_fighter_does_not_cancel_a_patrol_core_target():
    memory = _memory()
    roster = _roster()
    turn = make_turn(
        objects=[
            core(position=HOME),
            *roster,
            _rival(),
            unit(950, "RANGER", controlled=False, position=(100, 100)),
            unit(951, "WORKER", position=(101, 100)),
        ]
    )

    report = _strategy(memory).decide(turn)

    assert any(d.actor_id == object_id(2) for d in _hunt_actions(report))
    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.UP


@pytest.mark.parametrize(
    "ranger_position,target_position,should_hunt",
    [((29, 0), (33, 0), True), ((100, 100), (100, 96), False)],
)
def test_border_discovery_requires_a_ranger_in_the_authorized_activity_area(
    ranger_position, target_position, should_hunt
):
    memory = _memory()
    roster = _roster()
    roster[0]["position"] = list(ranger_position)
    turn = make_turn(objects=[core(position=HOME), *roster, _rival(target_position)])

    report = _strategy(memory).decide(turn)

    assert bool(_hunt_actions(report)) is should_hunt
    assert (memory.core_hunt is not None) is should_hunt
    if should_hunt:
        assert memory.core_hunt.ranger_id == object_id(2)
        action = turn.plan.unit_actions[turn.rangers[0].id]
        assert action.type == "MOVE"
        assert action.direction is Direction.RIGHT


@pytest.mark.parametrize("restart", [False, True])
def test_core_fleeing_under_the_same_uuid_keeps_the_patrol_hunter(tmp_path, restart):
    memory = _memory()
    roster = _roster()
    roster[0]["position"] = [7, 5]
    strategy = _strategy(memory)
    first = make_turn(tick=100, objects=[core(position=HOME), *roster, _rival()])
    strategy.decide(first)
    hunter_id = first.rangers[0].id
    assert first.plan.unit_actions[hunter_id].type == "SHOOT"
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = _strategy(memory)
    fleeing = _rival((7, 1))
    second = make_turn(tick=104, objects=[core(position=HOME), *roster, fleeing])

    report = strategy.decide(second)

    assert any(d.actor_id == str(hunter_id) for d in _hunt_actions(report))
    assert second.plan.unit_actions[hunter_id].type == "MOVE"
    destination = settled_positions(second)[hunter_id]
    assert line_of_fire(destination, (7, 1), memory.obstacles)
    roster[0]["position"] = list(destination)
    third = make_turn(tick=105, objects=[core(position=HOME), *roster, fleeing])
    strategy.decide(third)
    shot = third.plan.unit_actions[hunter_id]
    assert shot.type == "SHOOT"
    assert shot.expected_cell == (7, 1)


def test_core_pressure_recalls_the_patrol_hunter():
    memory = _memory()
    roster = _roster()
    strategy = _strategy(memory)
    strategy.decide(
        make_turn(tick=100, objects=[core(position=HOME), *roster, _rival()])
    )
    danger = make_turn(
        tick=101,
        objects=[
            core(position=HOME),
            *roster,
            _rival(),
            unit(950, "RANGER", controlled=False, position=(16, 13)),
        ],
    )

    report = strategy.decide(danger)

    assert not _hunt_actions(report)
    assert report.projected_core_damage > 0


def test_hunt_approach_routes_around_a_rock_in_the_nearest_firing_cell():
    memory = _memory()
    roster = _roster()
    first = make_turn(
        objects=[core(position=HOME), *roster, _rival()], obstacles=[(7, 5)]
    )

    strategy = _strategy(memory)
    report = strategy.decide(first)

    hunt = _hunt_actions(report)
    assert hunt
    hunter = next(
        ranger for ranger in first.rangers if str(ranger.id) == hunt[0].actor_id
    )
    assert settled_positions(first)[hunter.id] not in first.obstacle_cells
    assert manhattan(settled_positions(first)[hunter.id], TARGET) <= (
        manhattan(hunter.position, TARGET) + 1
    )
    _advance(roster, first)
    for tick in range(101, 109):
        turn = make_turn(
            tick=tick,
            objects=[core(position=HOME), *roster, _rival()],
            obstacles=[(7, 5)],
        )
        strategy.decide(turn)
        action = turn.plan.unit_actions[hunter.id]
        if action.type == "SHOOT":
            assert action.target_id == UUID(object_id(900))
            assert action.expected_cell == TARGET
            break
        assert action.type == "MOVE"
        assert settled_positions(turn)[hunter.id] not in turn.obstacle_cells
        _advance(roster, turn)
    else:
        pytest.fail("hunter did not reach a firing line around the rock")


@pytest.mark.parametrize("restart", [False, True])
def test_nearer_patrol_takes_over_without_resetting_the_original_budget(
    tmp_path, restart
):
    memory = _memory()
    memory.core_hunt = CoreHunt(
        object_id(1), object_id(900), object_id(3), 80, 220, TARGET, 99
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    turn = make_turn(tick=100, objects=[core(position=HOME), *_roster(), _rival()])

    report = _strategy(memory).decide(turn)

    hunt = memory.core_hunt
    assert hunt is not None
    assert hunt.ranger_id == object_id(2)
    assert hunt.target_id == object_id(900)
    assert (hunt.started, hunt.deadline) == (80, 220)
    assert {d.actor_id for d in _hunt_actions(report)} == {object_id(2)}
    assert memory.unit_roles[object_id(2)] == "patrol-1"
    assert memory.unit_roles[object_id(3)] == "defense"


def test_handoff_cannot_count_the_departing_old_hunter_as_a_home_reserve():
    memory = _memory()
    roster = _roster()[:3]
    roster[1]["position"] = [7, 10]
    memory.core_hunt = CoreHunt(
        object_id(1), object_id(900), object_id(3), 80, 220, TARGET, 99
    )
    turn = make_turn(tick=100, objects=[core(position=HOME), *roster, _rival()])

    _strategy(memory).decide(turn)

    assert memory.core_hunt is not None
    assert memory.core_hunt.ranger_id == object_id(3)


def test_far_survivors_do_not_satisfy_the_home_cover_reserve():
    memory = _memory()
    roster = _roster()
    roster[1]["position"] = [100, 100]
    roster[2]["position"] = [100, 102]
    turn = make_turn(objects=[core(position=HOME), *roster, _rival()])

    _strategy(memory).decide(turn)

    assert memory.core_hunt is None


@pytest.mark.parametrize("hp,shield", [(4, 5), (5, 4)])
def test_damaged_home_core_takes_priority_over_a_legal_hunt_shot(hp, shield):
    memory = _memory()
    roster = _roster()
    roster[0]["position"] = [7, 5]
    strategy = _strategy(memory)
    strategy.decide(
        make_turn(tick=100, objects=[core(position=HOME), *roster, _rival()])
    )
    turn = make_turn(
        tick=101, objects=[core(position=HOME, hp=hp, shield=shield), *roster, _rival()]
    )

    strategy.decide(turn)

    assert memory.core_hunt is None
    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type != "SHOOT" or action.expected_cell != TARGET


@pytest.mark.parametrize("loss", ["wounded_hunter", "home_reserve"])
def test_a_wounded_hunter_or_loss_of_home_reserves_aborts_the_task(loss):
    memory = _memory()
    roster = _roster()
    strategy = _strategy(memory)
    strategy.decide(
        make_turn(tick=100, objects=[core(position=HOME), *roster, _rival()])
    )
    if loss == "wounded_hunter":
        roster[0]["hp"] = 1
    else:
        roster = roster[:2]
    turn = make_turn(tick=101, objects=[core(position=HOME), *roster, _rival()])

    strategy.decide(turn)

    assert memory.core_hunt is None
    assert memory.core_hunt_cooldown_until == 161


@pytest.mark.parametrize("progress,aim", [(1, (7, 2)), (2, (7, 2)), (3, (7, 1))])
@pytest.mark.parametrize("ranger_position", [(7, 3), (7, 4)])
def test_public_core_progress_selects_the_actual_post_movement_aim(
    progress, aim, ranger_position
):
    memory = _memory()
    roster = _roster()
    roster[0]["position"] = list(ranger_position)
    target = _rival(
        state="MOVING",
        move_direction="UP",
        move_progress=progress,
        move_required_ticks=4,
        destination=[7, 1],
    )
    turn = make_turn(objects=[core(position=HOME), *roster, target])

    _strategy(memory).decide(turn)

    shot = turn.plan.unit_actions[turn.rangers[0].id]
    assert shot.type == "SHOOT"
    assert shot.target_id == UUID(object_id(900))
    assert shot.expected_cell == aim


def test_core_arrival_out_of_range_requires_a_step_before_the_next_shot():
    memory = _memory()
    roster = _roster()
    roster[0]["position"] = [7, 5]
    target = _rival(
        state="MOVING",
        move_direction="UP",
        move_progress=3,
        move_required_ticks=4,
        destination=[7, 1],
    )
    turn = make_turn(objects=[core(position=HOME), *roster, target])

    _strategy(memory).decide(turn)

    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.UP
    assert settled_positions(turn)[turn.rangers[0].id] == (7, 4)


@pytest.mark.parametrize("restart", [False, True])
def test_a_visible_fleeing_core_can_leave_the_initial_acquisition_radius(
    tmp_path, restart
):
    memory = _memory()
    roster = _roster()
    roster[0]["position"] = [16, -12]
    strategy = _strategy(memory)
    first = make_turn(
        tick=100, objects=[core(position=HOME), *roster, _rival((16, -15))]
    )
    strategy.decide(first)
    assert memory.core_hunt is not None
    deadline = memory.core_hunt.deadline
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = _strategy(memory)
    second = make_turn(
        tick=112, objects=[core(position=HOME), *roster, _rival((16, -18))]
    )

    report = strategy.decide(second)

    assert memory.core_hunt is not None
    assert memory.core_hunt.deadline == deadline
    assert memory.core_hunt.target_id == object_id(900)
    assert any(d.actor_id == object_id(2) for d in _hunt_actions(report))
    action = second.plan.unit_actions[second.rangers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.UP


@pytest.mark.parametrize("restart", [False, True])
def test_empty_old_cell_does_not_erase_a_core_with_a_reachable_hidden_destination(
    tmp_path, restart
):
    memory = _memory()
    roster = _roster()
    roster[0]["position"] = [7, 7]
    memory.core_hunt = CoreHunt(
        object_id(1),
        object_id(900),
        object_id(2),
        80,
        220,
        TARGET,
        100,
        direction=(0, -1),
        destination=(7, 1),
        arrival_tick=101,
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    strategy = _strategy(memory)
    turn = make_turn(tick=101, objects=[core(position=HOME), *roster])
    vision = visible_cells(turn.rangers[0].position, 5, memory.obstacles)
    assert TARGET in vision
    assert (7, 1) not in vision

    report = strategy.decide(turn)

    assert memory.core_hunt is not None
    assert memory.core_hunt.target_id == object_id(900)
    assert (memory.core_hunt.started, memory.core_hunt.deadline) == (80, 220)
    assert any("search the fleeing enemy Core" in d.reason for d in report.decisions)
    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.UP


@pytest.mark.parametrize("tick", [141, 241])
def test_loss_or_absolute_deadline_survives_restart_and_starts_cooldown(tmp_path, tick):
    memory = _memory()
    memory.core_hunt = CoreHunt(
        object_id(1), object_id(900), object_id(2), 100, 240, TARGET, 100
    )
    path = tmp_path / "memory.json"
    memory.save(path)
    loaded = WorldMemory.load(path)
    turn = make_turn(tick=tick, objects=[core(position=HOME), *_roster(), _rival()])
    if tick == 141:
        turn = make_turn(tick=tick, objects=[core(position=HOME), *_roster()])

    _strategy(loaded).decide(turn)

    assert loaded.core_hunt is None
    assert loaded.core_hunt_cooldown_until == tick + 60
    loaded.save(path)
    assert WorldMemory.load(path).core_hunt_cooldown_until == tick + 60


def test_an_expired_hunt_yields_a_legal_core_shot_to_the_actual_return():
    memory = _memory()
    roster = _roster()
    roster[0]["position"] = [7, 5]
    memory.core_hunt = CoreHunt(
        object_id(1), object_id(900), object_id(2), 100, 240, TARGET, 240
    )
    turn = make_turn(tick=241, objects=[core(position=HOME), *roster, _rival()])
    assert line_of_fire(turn.rangers[0].position, TARGET, memory.obstacles)

    report = _strategy(memory).decide(turn)

    assert memory.core_hunt is None
    assert object_id(2) in memory.core_hunt_returns
    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type == "MOVE"
    assert manhattan(settled_positions(turn)[turn.rangers[0].id], HOME) < 20
    assert any(
        d.actor_id == object_id(2) and "return from Core pursuit" in d.reason
        for d in report.decisions
    )


def test_destruction_does_not_transfer_the_hunt_to_a_respawned_uuid():
    memory = _memory()
    memory.core_hunt = CoreHunt(
        object_id(1), object_id(900), object_id(2), 100, 240, TARGET, 100
    )
    turn = make_turn(
        tick=101,
        objects=[
            core(position=HOME),
            *_roster(),
            core(902, controlled=False, position=(27, 2)),
        ],
        events=[
            {
                "event_id": object_id(990),
                "tick": 101,
                "event_type": "CORE_DESTROYED",
                "target_id": object_id(900),
                "position": list(TARGET),
            }
        ],
    )

    _strategy(memory).decide(turn)

    assert memory.core_hunt is None
    assert memory.core_hunt_cooldown_until == 161


def test_detached_patrol_and_returning_hunter_do_not_hold_the_shared_ring_phase():
    memory = _memory()
    memory.core_hunt = CoreHunt(
        object_id(1), object_id(900), object_id(2), 100, 240, TARGET, 100
    )
    strategy = _strategy(memory)
    roster = _roster()
    turn = make_turn(objects=[core(position=HOME), *roster])
    route = strategy._patrol_ring_route(turn)
    roster[3]["position"] = list(route[len(route) // 2])
    turn = make_turn(objects=[core(position=HOME), *roster])
    slots = {turn.rangers[0].id: 0, turn.rangers[3].id: 4}

    strategy._advance_patrol_ring_phase(turn, route, slots, 8)

    assert memory.patrol_ring_phase == 1
    strategy._abort_core_hunt(turn)
    assert turn.rangers[0].id in strategy._patrol_ring_exempt_ids(turn)
    roster[0]["position"] = [16, 10]
    arrived = make_turn(tick=101, objects=[core(position=HOME), *roster])
    assert strategy._core_hunt_return_ids(arrived) == frozenset()
    assert turn.rangers[0].id not in strategy._patrol_ring_exempt_ids(arrived)


def test_hunt_telemetry_contains_the_target_budget_and_pending_returns():
    memory = _memory()
    memory.core_hunt = CoreHunt(
        object_id(1), object_id(900), object_id(2), 100, 240, TARGET, 100
    )
    memory.core_hunt_returns[object_id(3)] = 230

    snapshot = _identity_snapshot(memory)

    assert snapshot["core_hunt"] == memory.core_hunt.to_dict()
    assert snapshot["core_hunt_returns"] == {object_id(3): 230}
