"""Beacon pickup, shield capacity, and formation-gated campaign direction."""

from __future__ import annotations

import pytest
from arena_hero import UnitType, core_resource_capacity

from arena_hero_bot.geometry import manhattan, next_step
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.migration import (
    blocks_are_cardinally_adjacent,
    plan_resource_cap,
    resource_block_index,
    resource_block_quota,
    resource_block_ring,
)
from arena_hero_bot.runtime import _identity_snapshot
from arena_hero_bot.strategy import CORE_LAYOUT_VERSION, AggressiveStrategy

from .factories import core, make_turn, object_id, unit
from .test_campaign import _gathered_roster, _strategy
from .test_core_service import context_for


def ready_campaign(center=(15, 15), rangers=60):
    workers, fighters, roles = _gathered_roster(rangers, center=center)
    memory = WorldMemory(
        core_home_position=center,
        core_layout_version=CORE_LAYOUT_VERSION,
        core_migration_last_position=center,
        campaign_phase=4,
        campaign_last_core_id=object_id(1),
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    return memory, workers, fighters


@pytest.mark.parametrize("role", ["defense", "patrol"])
def test_campaign_rangers_pick_up_a_ground_beacon_before_idle_movement(role):
    memory, workers, rangers = ready_campaign(rangers=20)
    picker = rangers[0] if role == "defense" else rangers[-1]
    turn = make_turn(
        objects=[core(position=(15, 15)), *workers, *rangers],
        beacon={"position": picker["position"], "status": "GROUND"},
    )

    _strategy(memory).decide(turn)

    action = turn.plan.unit_actions[
        next(r.id for r in turn.rangers if str(r.id) == picker["id"])
    ]
    assert action.type == "PICKUP_BEACON"


def test_campaign_legacy_local_vanguard_can_pick_up_a_ground_beacon():
    memory, workers, rangers = ready_campaign(rangers=20)
    vanguard = unit(3000, "VANGUARD", position=(12, 11))
    memory.unit_roles[str(vanguard["id"])] = "patrol-20"
    turn = make_turn(
        objects=[core(position=(15, 15)), *workers, *rangers, vanguard],
        beacon={"position": [12, 11], "status": "GROUND"},
    )

    _strategy(memory).decide(turn)

    assert turn.plan.unit_actions[turn.vanguards[0].id].type == "PICKUP_BEACON"


def test_empty_worker_picks_up_before_harvesting_the_same_cell():
    memory, workers, rangers = ready_campaign(rangers=20)
    picker = workers[0]
    position = tuple(picker["position"])
    turn = make_turn(
        objects=[core(position=(15, 15)), *workers, *rangers],
        resource_cells=[position],
        beacon={"position": list(position), "status": "GROUND"},
    )

    _strategy(memory).decide(turn)

    assert turn.plan.unit_actions[turn.workers[0].id].type == "PICKUP_BEACON"


def test_core_picks_up_on_its_route_before_starting_an_ordinary_move():
    memory, workers, rangers = ready_campaign(rangers=20)
    turn = make_turn(
        objects=[core(position=(16, 15)), *workers, *rangers],
        beacon={"position": [16, 15], "status": "GROUND"},
    )

    _strategy(memory).decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "PICKUP_BEACON"


def test_shared_ground_beacon_produces_only_one_pickup_plan():
    memory, workers, rangers = ready_campaign(rangers=20)
    rangers[0]["position"] = [12, 10]
    rangers[1]["position"] = [12, 10]
    turn = make_turn(
        objects=[core(position=(15, 15)), *workers, *rangers],
        beacon={"position": [12, 10], "status": "GROUND"},
    )

    _strategy(memory).decide(turn)

    assert sum(a.type == "PICKUP_BEACON" for a in turn.plan.unit_actions.values()) == 1
    assert (
        turn.plan.core_action is None or turn.plan.core_action.type != "PICKUP_BEACON"
    )


def test_ranger_keeps_a_current_enemy_shot_before_beacon_pickup():
    memory, workers, rangers = ready_campaign(rangers=20)
    rangers[0]["position"] = [15, 9]
    turn = make_turn(
        objects=[
            core(position=(15, 15)),
            *workers,
            *rangers,
            unit(3000, "WORKER", controlled=False, position=(15, 7)),
        ],
        beacon={"position": [15, 9], "status": "GROUND"},
    )

    _strategy(memory).decide(turn)

    assert turn.plan.unit_actions[turn.rangers[0].id].type == "SHOOT"


def test_pickup_does_not_pin_a_ranger_on_a_lethal_current_firing_line():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(10, 0)),
            unit(3, "RANGER", controlled=False, position=(12, 0)),
            unit(4, "RANGER", controlled=False, position=(13, 0)),
        ],
        beacon={"position": [10, 0], "status": "GROUND"},
    )
    strategy = _strategy(WorldMemory())
    strategy.memory.observe(turn)

    assert not strategy._pickup_beacon(turn.rangers[0], context_for(turn))
    assert not turn.plan.unit_actions


@pytest.mark.parametrize("service", ["deposit", "heal"])
def test_actual_core_service_keeps_priority_over_ground_beacon(service):
    memory, workers, rangers = ready_campaign(rangers=20)
    if service == "deposit":
        workers[0]["position"] = [15, 15]
        workers[0]["cargo"] = 1
    else:
        rangers[0]["position"] = [15, 15]
        rangers[0]["hp"] = 1
    turn = make_turn(
        resources=1,
        objects=[core(position=(15, 15)), *workers, *rangers],
        beacon={"position": [15, 15], "status": "GROUND"},
    )

    _strategy(memory).decide(turn)

    assert any(a.type == service.upper() for a in turn.plan.unit_actions.values())
    assert all(a.type != "PICKUP_BEACON" for a in turn.plan.unit_actions.values())
    assert (
        turn.plan.core_action is None or turn.plan.core_action.type != "PICKUP_BEACON"
    )


def test_wounded_local_vanguard_heals_before_pickup_during_an_assault():
    turn = make_turn(
        resources=1,
        objects=[core(), unit(2, "VANGUARD", hp=3)],
        beacon={"position": [0, 0], "status": "GROUND"},
    )
    memory = WorldMemory(unit_roles={object_id(2): "patrol-1"})
    strategy = _strategy(memory)
    strategy.memory.observe(turn)
    context = context_for(turn)
    context.combat_assault = True

    strategy._decide_vanguard(turn.vanguards[0], context, ())

    assert turn.plan.unit_actions[turn.vanguards[0].id].type == "HEAL"
    assert not context.beacon_claimed


@pytest.mark.parametrize("carrier_kind", ["CORE", "WORKER", "RANGER", "VANGUARD"])
@pytest.mark.parametrize("friendly", [False, True])
def test_shield_capacity_depends_on_the_current_carrier_owner(carrier_kind, friendly):
    carrier = (
        core(2, controlled=friendly, position=(10, 0))
        if carrier_kind == "CORE"
        else unit(2, carrier_kind, controlled=friendly, position=(10, 0))
    )
    objects = [carrier] if carrier_kind == "CORE" and friendly else [core(), carrier]
    turn = make_turn(
        objects=objects,
        beacon={"position": [10, 0], "status": "CARRIED", "carrier_id": object_id(2)},
    )

    assert AggressiveStrategy._core_shield_limit(turn) == (10 if friendly else 5)


@pytest.mark.parametrize("carrier_kind", ["CORE", "WORKER", "RANGER"])
def test_campaign_fills_beacon_shields_through_ten_across_ticks_and_restart(
    tmp_path, carrier_kind
):
    memory, workers, rangers = ready_campaign(rangers=20)
    carrier = (
        core(position=(15, 15))
        if carrier_kind == "CORE"
        else (workers[0] if carrier_kind == "WORKER" else rangers[0])
    )
    path = tmp_path / "memory.json"
    strategy = _strategy(memory)
    for shield in range(5, 11):
        turn = make_turn(
            tick=100 + shield,
            resources=10,
            objects=[core(position=(15, 15), shield=shield), *workers, *rangers],
            beacon={
                "position": carrier["position"],
                "status": "CARRIED",
                "carrier_id": carrier["id"],
            },
        )
        strategy.decide(turn)
        action = turn.plan.core_action
        if shield < 10:
            assert action is not None and action.type == "REPAIR_SHIELD"
        else:
            assert action is None or action.type != "REPAIR_SHIELD"
        if shield == 7:
            strategy.memory.save(path)
            strategy = _strategy(WorldMemory.load(path))


@pytest.mark.parametrize("status", [None, "GROUND", "CARRIED"])
def test_enemy_or_unseen_beacon_never_funds_bonus_shields(status):
    beacon = {"position": [10, 0], "status": status}
    objects = [core()]
    if status == "CARRIED":
        beacon["carrier_id"] = object_id(2)
        objects.append(unit(2, "RANGER", controlled=False, position=(10, 0)))
    turn = make_turn(resources=1, objects=objects, beacon=beacon)

    _strategy(WorldMemory())._decide_core(context_for(turn))

    assert (
        turn.plan.core_action is None or turn.plan.core_action.type != "REPAIR_SHIELD"
    )


def test_losing_beacon_stops_bonus_repair_and_never_uses_persisted_ownership():
    strategy = _strategy(WorldMemory())
    held = make_turn(
        resources=1,
        objects=[core()],
        beacon={"position": [0, 0], "status": "CARRIED", "carrier_id": object_id(1)},
    )
    strategy._decide_core(context_for(held))
    assert held.plan.core_action is not None
    assert held.plan.core_action.type == "REPAIR_SHIELD"
    lost = make_turn(
        tick=101,
        resources=1,
        objects=[core()],
        beacon={"position": [10, 0], "status": "GROUND"},
    )

    strategy._decide_core(context_for(lost))

    assert (
        lost.plan.core_action is None or lost.plan.core_action.type != "REPAIR_SHIELD"
    )


@pytest.mark.parametrize(
    "reason", ["moving", "unfunded", "production", "noncombat-contact"]
)
def test_beacon_shield_repair_respects_core_action_constraints(reason):
    memory, workers, rangers = ready_campaign(rangers=20)
    core_object = core(position=(15, 15), shield=7)
    if reason == "moving":
        core_object.update(
            {
                "state": "MOVING",
                "move_direction": "RIGHT",
                "move_progress": 2,
                "move_required_ticks": 4,
                "destination": [16, 15],
            }
        )
    turn = make_turn(
        resources=(
            core_resource_capacity(36)
            if reason in {"production", "noncombat-contact"}
            else 0
            if reason == "unfunded"
            else 10
        ),
        objects=[
            core_object,
            *workers,
            *rangers,
            *(
                [unit(3000, "WORKER", controlled=False, position=(15, 12))]
                if reason == "noncombat-contact"
                else []
            ),
        ],
        beacon={"position": [15, 15], "status": "CARRIED", "carrier_id": object_id(1)},
    )

    _strategy(memory).decide(turn)

    action = turn.plan.core_action
    assert action is None or action.type != "REPAIR_SHIELD"
    if reason in {"production", "noncombat-contact"}:
        assert action is not None and action.type == "SPAWN"
        assert action.unit_type is UnitType.RANGER


@pytest.mark.parametrize("rangers", [59, 60, 62])
@pytest.mark.parametrize("workers", [15, 16])
@pytest.mark.parametrize("beacon_x", [335, 367])
def test_beacon_goal_uses_live_roster_and_exact_quota_seven_boundary(
    rangers, workers, beacon_x
):
    _, builders, fighters = ready_campaign(rangers=rangers)
    turn = make_turn(
        objects=[core(position=(15, 15)), *builders[:workers], *fighters],
        beacon={"position": [beacon_x, 15]},
    )
    goal = _strategy(WorldMemory())._campaign_beacon_goal(turn)

    eligible = rangers >= 60 and workers == 16 and beacon_x == 335
    assert goal == ((beacon_x, 15) if eligible else None)
    assert resource_block_quota(resource_block_index((beacon_x, 15))) == (
        7 if beacon_x == 335 else 6
    )


@pytest.mark.parametrize(
    ("origin", "beacon", "next_block"),
    [
        ((15, 15), (111, 15), (1, 0)),
        ((-17, 15), (15, 15), (0, 0)),
        ((79, 15), (79, 175), (2, 1)),
        ((15, 367), (143, 15), (0, 10)),
        ((-369, -17), (-145, -17), (-11, -1)),
    ],
)
def test_beacon_planner_reduces_block_distance_and_preserves_the_quota_region(
    origin, beacon, next_block
):
    plan = plan_resource_cap(
        WorldMemory(),
        target_quota=16,
        stop_ring=0,
        current_position=origin,
        beacon_position=beacon,
    )

    assert plan is not None
    assert plan.target_block == next_block
    assert blocks_are_cardinally_adjacent(plan.current_block, plan.target_block)
    assert manhattan(plan.target_block, resource_block_index(beacon)) == (
        manhattan(plan.current_block, resource_block_index(beacon)) - 1
    )
    if plan.current_quota >= 7:
        assert plan.target_quota >= 7
    else:
        assert plan.target_ring < plan.current_ring
    assert next_step(origin, plan.target, blocked=set(), require_path=True) is not None


@pytest.mark.parametrize("beacon", [(15, 15), (367, 15)])
def test_beacon_planner_does_not_publish_a_same_block_or_low_quota_goal(beacon):
    assert (
        plan_resource_cap(
            WorldMemory(),
            target_quota=16,
            stop_ring=0,
            current_position=(15, 15),
            beacon_position=beacon,
        )
        is None
    )


def test_blocked_beacon_step_never_chooses_an_unrelated_detour():
    memory = WorldMemory(obstacles={(x, y) for x in range(32, 64) for y in range(32)})

    assert (
        plan_resource_cap(
            memory,
            target_quota=16,
            stop_ring=0,
            current_position=(15, 15),
            beacon_position=(111, 15),
        )
        is None
    )


def test_beacon_scout_keeps_its_budget_on_restart_then_releases_the_adjacent_hop(
    tmp_path,
):
    center, beacon = (15, 15), (111, 15)
    memory, workers, rangers = ready_campaign(center)
    strategy = _strategy(memory)
    turn = make_turn(
        objects=[core(position=center), *workers, *rangers],
        beacon={"position": list(beacon)},
    )
    strategy.decide(turn)
    target = memory.campaign_scout_target
    assert target is not None and resource_block_index(target) == (1, 0)
    assert memory.core_home_position == center
    assert memory.campaign_beacon_target == beacon
    assert turn.plan.core_action is None or turn.plan.core_action.type != "START_MOVE"
    assert len(turn.plan.unit_actions) == 76
    original_started = memory.campaign_scout_started_tick
    path = tmp_path / "memory.json"
    memory.save(path)
    memory = WorldMemory.load(path)
    strategy = _strategy(memory)
    # A changed public destination does not renew an in-flight scout's budget.
    changed_beacon = (111, 47)
    changed = make_turn(
        tick=101,
        objects=[core(position=center), *workers, *rangers],
        beacon={"position": list(changed_beacon)},
    )
    strategy.decide(changed)
    assert memory.campaign_scout_target == target
    assert memory.campaign_scout_started_tick == original_started
    assert memory.campaign_beacon_target == changed_beacon
    assert _identity_snapshot(memory)["campaign"]["beacon_target"] == list(
        changed_beacon
    )
    block = resource_block_index(target)
    memory.campaign_scout_seen_ticks = {
        (x, y): 101
        for x in range(block[0] * 32, (block[0] + 1) * 32)
        for y in range(block[1] * 32, (block[1] + 1) * 32)
    }
    ready = make_turn(
        tick=102,
        objects=[core(position=center), *workers, *rangers],
        beacon={"position": list(changed_beacon)},
    )
    strategy.decide(ready)
    assert memory.campaign_scout_target is None
    assert memory.core_home_position == target
    assert memory.core_migration_start_position == center
    assert ready.plan.core_action is not None
    assert ready.plan.core_action.type == "START_MOVE"
    assert ready.plan.core_action.direction.value == "RIGHT"


@pytest.mark.parametrize("hold", ["straggler", "cargo", "production"])
def test_beacon_approach_keeps_the_existing_rest_gate(hold):
    memory, workers, rangers = ready_campaign()
    if hold == "straggler":
        rangers[-1]["position"] = [32, 15]
    elif hold == "cargo":
        workers[0]["position"] = [15, 16]
        workers[0]["cargo"] = 1
    turn = make_turn(
        resources=core_resource_capacity(76) if hold == "production" else 0,
        objects=[core(position=(15, 15)), *workers, *rangers],
        beacon={"position": [111, 15]},
    )

    _strategy(memory).decide(turn)

    assert memory.campaign_resting
    assert memory.core_home_position == (15, 15)
    assert memory.campaign_scout_target is None
    assert turn.plan.core_action is None or turn.plan.core_action.type != "START_MOVE"


def test_beacon_block_is_the_stop_and_low_quota_beacon_restores_center_direction():
    center = (111, 15)
    memory, workers, rangers = ready_campaign(center)
    strategy = _strategy(memory)
    same_block = make_turn(
        objects=[core(position=center), *workers, *rangers],
        beacon={"position": [97, 1]},
    )
    strategy.decide(same_block)
    assert memory.campaign_beacon_target == (97, 1)
    assert memory.core_home_position == center
    assert memory.campaign_scout_target is None
    assert (
        same_block.plan.core_action is None
        or same_block.plan.core_action.type != "START_MOVE"
    )
    moved = make_turn(
        tick=101,
        objects=[core(position=center), *workers, *rangers],
        beacon={"position": [367, 15]},
    )
    strategy.decide(moved)
    assert memory.campaign_beacon_target is None
    target = memory.campaign_scout_target
    assert target is not None
    assert resource_block_ring(resource_block_index(target)) == 2


def test_respawn_discards_the_old_beacon_goal_and_rebuilds_locally():
    memory, _, _ = ready_campaign()
    memory.campaign_beacon_target = (111, 15)
    memory.campaign_scout_target = (47, 15)
    reborn = make_turn(
        objects=[core(9, position=(415, 415)), unit(10, "WORKER", position=(415, 415))],
        beacon={"position": [111, 15]},
    )

    _strategy(memory).decide(reborn)

    assert memory.campaign_phase == 1
    assert memory.campaign_beacon_target is None
    assert memory.campaign_scout_target is None
    assert memory.core_home_position is not None
    assert resource_block_index(memory.core_home_position) == (12, 12)
