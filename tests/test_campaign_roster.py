"""Dynamic Ranger identities and roster-based migration boundaries."""

from collections import Counter

import pytest
from arena_hero import UnitType, core_resource_capacity, unit_cost

from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.migration import (
    campaign_stop_ring,
    plan_resource_cap,
    resource_block_index,
    resource_block_quota,
)
from arena_hero_bot.strategy import CORE_LAYOUT_VERSION

from .factories import core, make_turn, object_id, unit
from .test_campaign import _gathered_roster, _rangers, _roster, _strategy, _workers


@pytest.mark.parametrize("phase", [1, 2, 3, 4])
def test_births_and_casualties_minimize_role_changes_across_restart(tmp_path, phase):
    memory = WorldMemory(campaign_phase=phase, unit_roles_initialized=True)
    strategy = _strategy(memory)
    roster = []
    path = tmp_path / "memory.json"

    def reconcile():
        nonlocal memory, strategy
        previous = memory.unit_roles.copy()
        turn = make_turn(objects=[core(position=(15, 15)), *roster])
        strategy._reconcile_unit_roles(turn)
        roles = memory.unit_roles
        defense = sum(role == "defense" for role in roles.values())
        assert defense == (len(roster) + 1) // 2
        assert all(
            role == "defense" or role.startswith("patrol-") for role in roles.values()
        )
        assert (
            max(
                Counter(role for role in roles.values() if role != "defense").values(),
                default=0,
            )
            <= 2
        )
        survivors = set(previous) & set(roles)
        old_defense = sum(previous[uid] == "defense" for uid in survivors)
        new_count = len(roles.keys() - previous.keys())
        required_changes = max(
            0, old_defense - defense, defense - old_defense - new_count
        )
        assert sum(roles[uid] != previous[uid] for uid in survivors) == required_changes
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = _strategy(memory)
        strategy._reconcile_unit_roles(turn)
        assert memory.unit_roles == roles

    # Decreasing UUIDs must not displace surviving identities.
    for index in range(41):
        roster.append(unit(2000 - index, "RANGER", position=(20 + index, 20)))
        reconcile()
    # Exercise losses on both sides at odd/even totals, including zero.
    while roster:
        preferred = "defense" if len(roster) % 3 else "patrol-"
        victim = next(
            (
                actor
                for actor in roster
                if memory.unit_roles[str(actor["id"])].startswith(preferred)
            ),
            roster[0],
        )
        roster.remove(victim)
        reconcile()


def test_legacy_twenty_eight_roster_changes_only_six_defenders():
    rangers = _rangers(28)
    roles = {
        **{str(actor["id"]): "defense" for actor in rangers[:20]},
        **{
            str(actor["id"]): f"patrol-{i // 2 + 1}"
            for i, actor in enumerate(rangers[20:])
        },
    }
    memory = WorldMemory(unit_roles=roles.copy(), unit_roles_initialized=True)
    memory.defense_posts = {
        str(actor["id"]): (i, 0) for i, actor in enumerate(rangers[:20])
    }
    memory.patrol_ring_slots = {
        str(actor["id"]): i for i, actor in enumerate(rangers[20:])
    }
    previous_slots = memory.patrol_ring_slots.copy()
    strategy = _strategy(memory)
    turn = make_turn(objects=[core(position=(15, 15)), *rangers])

    strategy._reconcile_unit_roles(turn)

    assert sum(role == "defense" for role in memory.unit_roles.values()) == 14
    assert sum(roles[uid] != role for uid, role in memory.unit_roles.items()) == 6
    assert memory.patrol_ring_slots == previous_slots
    assert all(memory.unit_roles[uid] == "defense" for uid in memory.defense_posts)
    previous = memory.unit_roles.copy()
    strategy._reconcile_unit_roles(turn)
    assert memory.unit_roles == previous


def test_simultaneous_birth_and_defender_loss_uses_recruit_first():
    rangers = _rangers(6)
    strategy = _strategy(WorldMemory())
    strategy._reconcile_unit_roles(make_turn(objects=[core(), *rangers]))
    previous = strategy.memory.unit_roles.copy()
    lost = next(uid for uid, role in previous.items() if role == "defense")
    survivors = [actor for actor in rangers if str(actor["id"]) != lost]
    strategy._reconcile_unit_roles(
        make_turn(objects=[core(), *survivors, unit(2, "RANGER", position=(60, 0))])
    )
    assert strategy.memory.unit_roles == {
        **{uid: role for uid, role in previous.items() if uid != lost},
        object_id(2): "defense",
    }


@pytest.mark.parametrize(
    ("workers", "rangers", "cap"),
    [
        (15, 80, None),
        (16, 7, None),
        (16, 8, 2),
        (16, 19, 2),
        (16, 20, 3),
        (16, 29, 3),
        (16, 30, 4),
        (16, 39, 4),
        (16, 40, 5),
        (16, 49, 5),
        (16, 50, 6),
        (16, 59, 6),
        (15, 60, None),
        (16, 60, 16),
        (16, 62, 16),
        (16, 69, 16),
        (16, 70, 16),
        (16, 80, 16),
        (16, 150, 16),
    ],
)
def test_quota_gate_uses_total_rangers_not_roles(workers, rangers, cap):
    actors = _rangers(rangers)
    memory = WorldMemory(
        campaign_phase=4, unit_roles={str(actor["id"]): "patrol-1" for actor in actors}
    )
    strategy = _strategy(memory)
    turn = make_turn(objects=[core(), *_workers(workers), *actors])
    assert strategy._campaign_movement_cap(turn) == cap
    memory.unit_roles = {str(actor["id"]): "defense" for actor in actors}
    assert strategy._campaign_movement_cap(turn) == cap


@pytest.mark.parametrize("cap", range(2, 18))
def test_low_quotas_reach_the_frontier_and_high_quotas_stop_on_entry(cap):
    stop = campaign_stop_ring(cap)
    assert stop is not None
    assert resource_block_quota((stop, 0)) <= cap
    if stop and cap < 4:
        assert resource_block_quota((stop - 1, 0)) > cap
    if cap >= 4:
        assert resource_block_quota((stop + 1, 0)) < resource_block_quota((stop, 0))
    memory = WorldMemory()
    assert (
        plan_resource_cap(
            memory,
            target_quota=cap,
            stop_ring=stop,
            current_position=(stop * 32 + 15, 15),
        )
        is None
    )
    hop = plan_resource_cap(
        memory,
        target_quota=cap,
        stop_ring=stop,
        current_position=((stop + 1) * 32 + 15, 15),
    )
    assert hop is not None
    assert hop.target_ring == stop
    assert hop.target_quota <= cap


@pytest.mark.parametrize(
    ("ranger_count", "phase", "frontier"),
    [(19, 3, 4), (20, 4, 4), (30, 4, 4), (59, 4, 6), (60, 4, 16), (80, 4, 16)],
)
def test_full_decision_advances_stage_and_replaces_legacy_quota_goal(
    ranger_count, phase, frontier
):
    center = (15, 15)
    memory = WorldMemory(
        core_home_position=center,
        campaign_last_core_id=object_id(1),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=3,
        campaign_target_quota=8,
    )
    strategy = _strategy(memory)
    turn = make_turn(
        objects=[core(position=center), *_workers(16), *_rangers(ranger_count)]
    )
    strategy.decide(turn)
    assert memory.campaign_phase == phase
    assert memory.campaign_target_quota == frontier
    assert strategy._campaign_defense_rangers(turn) == (ranger_count + 1) // 2
    assert strategy._campaign_patrol_rangers(turn) == ranger_count // 2
    assert len(memory.defense_posts) == (ranger_count + 1) // 2
    assert len(set(memory.defense_posts.values())) == len(memory.defense_posts)


@pytest.mark.parametrize("count", [19, 20])
@pytest.mark.parametrize("full_bank", [False, True])
def test_twentieth_ranger_switches_to_full_bank_growth(count, full_bank):
    population = 16 + count
    turn = make_turn(
        objects=[core(position=(15, 15)), *_workers(16), *_rangers(count)],
        resources=core_resource_capacity(population)
        if full_bank
        else unit_cost(UnitType.RANGER, population),
    )
    strategy = _strategy(WorldMemory())
    strategy.decide(turn)
    if count < 20 or full_bank:
        assert turn.plan.core_action is not None
        assert turn.plan.core_action.type == "SPAWN"
        assert turn.plan.core_action.unit_type is UnitType.RANGER
    else:
        assert turn.plan.core_action is None or turn.plan.core_action.type != "SPAWN"


@pytest.mark.parametrize("ring", range(11))
@pytest.mark.parametrize("ranger_count", [59, 60])
def test_sixty_rangers_authorize_every_high_quota_without_a_next_tier(
    ring, ranger_count
):
    center = (ring * 32 + 15, 15)
    workers, rangers, roles = _gathered_roster(ranger_count, center=center)
    strategy = _strategy(WorldMemory(campaign_phase=4, unit_roles=roles))
    turn = make_turn(objects=[core(position=center), *workers, *rangers])
    target_block = (ring - 1, 0)

    assert resource_block_quota(resource_block_index(center)) >= 7
    assert resource_block_quota(target_block) >= 7
    assert strategy._campaign_cross_block_allowed(turn, target_block) == (
        ranger_count == 60
    )


@pytest.mark.parametrize("ring", [10, 9, 8, 5, 2, 1])
@pytest.mark.parametrize("restart", [False, True])
def test_sixty_rangers_scout_before_releasing_a_high_quota_hop(tmp_path, ring, restart):
    center = (ring * 32 + 15, 15)
    workers, rangers, roles = _gathered_roster(60, center=center)
    memory = WorldMemory(
        campaign_phase=4,
        campaign_target_quota=7,
        campaign_last_core_id=object_id(1),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_home_position=center,
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    strategy = _strategy(memory)
    turn = make_turn(
        tick=100,
        objects=[core(position=center), *workers, *rangers],
        beacon={"position": [10000, 10000]},
    )

    report = strategy.decide(turn)

    target = memory.campaign_scout_target
    assert not any("planning time limit" in item.reason for item in report.decisions)
    assert target is not None
    assert resource_block_index(target) == (ring - 1, 0)
    assert memory.campaign_target_quota == 16
    assert memory.core_home_position == center
    assert turn.plan.core_action is None or turn.plan.core_action.type != "START_MOVE"
    for actor in rangers:
        uid = str(actor["id"])
        assert memory.unit_roles[uid] == roles[uid]

    memory.campaign_scout_seen_ticks = {
        (x, y): 100 for x in range((ring - 1) * 32, ring * 32) for y in range(32)
    }
    if restart:
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = _strategy(memory)
    ready = make_turn(
        tick=101,
        objects=[core(position=center), *workers, *rangers],
        beacon={"position": [10000, 10000]},
    )

    report = strategy.decide(ready)

    assert not any("planning time limit" in item.reason for item in report.decisions)
    assert memory.campaign_scout_target is None
    assert memory.core_home_position == target
    assert memory.core_migration_start_position == center
    assert ready.plan.core_action is not None
    assert ready.plan.core_action.type == "START_MOVE"
    assert ready.plan.core_action.direction.value == "LEFT"


@pytest.mark.parametrize("loss", ["ranger", "worker"])
@pytest.mark.parametrize("restart", [False, True])
def test_full_migration_permission_is_revoked_with_its_scout_after_a_loss(
    tmp_path, loss, restart
):
    center, target = (335, 15), (303, 15)
    workers, rangers, roles = _gathered_roster(60, center=center)
    memory = WorldMemory(
        campaign_phase=4,
        campaign_target_quota=16,
        campaign_last_core_id=object_id(1),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_home_position=center,
        campaign_scout_target=target,
        campaign_scout_started_tick=90,
        campaign_scout_seen_ticks={target: 99},
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    if loss == "ranger":
        rangers.pop()
    else:
        workers.pop()
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    strategy = _strategy(memory)
    turn = make_turn(tick=100, objects=[core(position=center), *workers, *rangers])

    report = strategy.decide(turn)

    assert not any("planning time limit" in item.reason for item in report.decisions)
    assert memory.campaign_phase == 4
    assert memory.campaign_target_quota == (6 if loss == "ranger" else 16)
    assert memory.campaign_scout_target is None
    assert memory.campaign_scout_started_tick == 0
    assert not memory.campaign_scout_seen_ticks
    assert memory.core_home_position == center
    assert turn.plan.core_action is None or turn.plan.core_action.type != "START_MOVE"


@pytest.mark.parametrize(
    ("count", "may_travel"), [(30, False), (39, False), (40, True)]
)
def test_quota_four_waits_for_forty_rangers_before_another_hop(count, may_travel):
    center = (783, 15)
    workers, rangers, roles = _roster(count)
    for index, actor in enumerate([*workers, *rangers]):
        actor["position"] = [center[0] + 2 + index % 6, center[1] + 2 + index // 6]
    memory = WorldMemory(
        campaign_phase=4,
        campaign_target_quota=8,
        unit_roles=roles,
        campaign_last_core_id=object_id(1),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_home_position=center,
    )
    strategy = _strategy(memory)
    strategy.decide(make_turn(objects=[core(position=center), *workers, *rangers]))
    assert (memory.campaign_scout_target is not None) == may_travel
    assert memory.core_home_position == center
    if may_travel:
        assert memory.campaign_scout_target is not None
        assert (
            resource_block_quota(resource_block_index(memory.campaign_scout_target))
            == 4
        )
        assert memory.campaign_target_quota == 5


_BATTLE_ROUTES = [
    ((783, 15), (751, 15), 40),
    ((815, 15), (783, 15), 30),
    ((591, 15), (559, 15), 40),
    ((335, 15), (303, 15), 60),
    ((47, 15), (15, 15), 60),
]


def _battle_migration_scenario(origin, destination, ranger_count, crossed):
    position = (origin[0] - (16 if crossed else 3), origin[1])
    workers, rangers, roles = _roster(ranger_count)
    for index, actor in enumerate([*workers, *rangers]):
        actor["position"] = [position[0] + 2 + index % 7, position[1] + 2 + index // 7]
    memory = WorldMemory(
        campaign_phase=4,
        campaign_target_quota=ranger_count // 10 + 1,
        unit_roles=roles,
        campaign_last_core_id=object_id(1),
        core_layout_version=CORE_LAYOUT_VERSION,
        core_home_position=destination,
        core_migration_start_position=origin,
        core_migration_last_position=position,
    )
    enemies = [
        unit(
            3000 + i,
            "RANGER",
            controlled=False,
            position=(position[0] + i + 2, position[1] - 4),
        )
        for i in range(3)
    ]
    return _strategy(memory), position, workers, rangers, enemies


def _assert_recovered_journey(memory, origin, destination, position, crossed, loss):
    assert not memory.defense_compressed
    assert memory.campaign_phase == 4
    if loss == "none":
        assert memory.core_home_position == destination
        assert memory.core_migration_start_position == origin
    else:
        assert memory.core_home_position is not None
        assert resource_block_index(memory.core_home_position) == resource_block_index(
            position
        )
        if crossed:
            assert memory.core_home_position == destination
        assert memory.core_migration_start_position is None
        assert memory.campaign_scout_target is None
        assert memory.campaign_emergency_target is None


@pytest.mark.parametrize("crossed", [False, True])
@pytest.mark.parametrize("loss", ["none", "ranger", "worker"])
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize(("origin", "destination", "ranger_count"), _BATTLE_ROUTES)
def test_battle_recovery_rechecks_every_roster_and_route_combination(
    tmp_path, crossed, loss, restart, origin, destination, ranger_count
):
    # All roster/route recovery transitions run directly. The full decisions below
    # separately cover pressure detection, call ordering and Core moves.
    strategy, position, workers, rangers, enemies = _battle_migration_scenario(
        origin, destination, ranger_count, crossed
    )
    memory = strategy.memory
    battle = make_turn(
        tick=100, objects=[core(position=position), *workers, *rangers, *enemies]
    )
    strategy._update_campaign_defense_posture(battle, under_pressure=True)
    assert memory.defense_compressed
    assert memory.core_home_position == destination
    if loss == "ranger":
        rangers.pop()
    elif loss == "worker":
        workers.pop()
    # Pending tasks must actually be cleared, not merely remain None.
    memory.campaign_scout_target = destination
    memory.campaign_scout_started_tick = 90
    memory.campaign_scout_seen_ticks = {destination: 99}
    memory.campaign_emergency_target = destination
    memory.campaign_emergency_started_tick = 90
    memory.core_arrival_worker_id = str(workers[0]["id"])
    memory.core_arrival_destination = (position[0] - 1, position[1])
    memory.core_queue_targets = {str(workers[0]["id"]): destination}

    for tick in (101, 108, 109):
        turn = make_turn(
            tick=tick, objects=[core(position=position), *workers, *rangers]
        )
        strategy._update_campaign_defense_posture(turn, under_pressure=False)
        if tick < 109:
            assert memory.defense_compressed
            assert memory.defense_calm_since_tick == 101
            assert memory.core_home_position == destination
            assert memory.core_migration_start_position == origin
        if tick == 101 and restart:
            path = tmp_path / "memory.json"
            memory.save(path)
            memory = WorldMemory.load(path)
            strategy = _strategy(memory)

    _assert_recovered_journey(memory, origin, destination, position, crossed, loss)
    if loss != "none":
        assert memory.campaign_scout_started_tick == 0
        assert not memory.campaign_scout_seen_ticks
        assert memory.campaign_emergency_started_tick == 0
        assert memory.core_arrival_worker_id is None
        assert memory.core_arrival_destination is None
        assert not memory.core_queue_targets
    else:
        assert memory.core_arrival_worker_id == str(workers[0]["id"])
        assert memory.core_queue_targets


@pytest.mark.parametrize(
    ("route", "crossed", "loss", "restart"),
    [
        (0, False, "none", False),
        (0, True, "ranger", True),
        (1, False, "ranger", False),
        (1, True, "worker", False),
        (2, False, "worker", True),
        (2, True, "none", True),
        (3, False, "none", True),
        (3, True, "ranger", False),
        (4, True, "none", True),
        (4, True, "worker", False),
    ],
)
def test_major_battle_refits_in_current_block_when_journey_is_no_longer_authorized(
    tmp_path, route, crossed, loss, restart
):
    origin, destination, ranger_count = _BATTLE_ROUTES[route]
    strategy, position, workers, rangers, enemies = _battle_migration_scenario(
        origin, destination, ranger_count, crossed
    )
    memory = strategy.memory
    path = tmp_path / "memory.json"
    battle = make_turn(
        tick=100, objects=[core(position=position), *workers, *rangers, *enemies]
    )
    report = strategy.decide(battle)
    assert not any("planning time limit" in item.reason for item in report.decisions)
    assert memory.defense_compressed
    assert memory.core_home_position == destination
    if loss == "ranger":
        rangers.pop()
    elif loss == "worker":
        workers.pop()
    if restart:
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = _strategy(memory)

    for tick in (101, 109):
        turn = make_turn(
            tick=tick,
            objects=[core(position=position), *workers, *rangers],
            events=[
                {
                    "event_id": object_id(4000 + index),
                    "tick": 100,
                    "event_type": "UNIT_DESTROYED",
                    "target_id": enemy["id"],
                }
                for index, enemy in enumerate(enemies)
            ]
            if tick == 101
            else [],
        )
        report = strategy.decide(turn)
        assert not any(
            "planning time limit" in item.reason for item in report.decisions
        )
        if tick == 101:
            assert memory.defense_compressed
            assert memory.defense_calm_since_tick == 101
            # The roster gate may already revoke an illegal hop; the defense
            # posture itself still waits for the full calm interval.
            if restart:
                memory.save(path)
                memory = WorldMemory.load(path)
                strategy = _strategy(memory)

    _assert_recovered_journey(memory, origin, destination, position, crossed, loss)
    if loss != "none":
        assert turn.plan.core_action is not None
        assert turn.plan.core_action.type == "START_MOVE"


def test_stale_same_quota_four_scout_is_revoked_after_restart():
    workers, rangers, roles = _roster(30)
    memory = WorldMemory(
        campaign_phase=4,
        unit_roles=roles,
        campaign_scout_target=(751, 15),
        campaign_scout_started_tick=90,
        core_home_position=(783, 15),
    )
    strategy = _strategy(memory)
    turn = make_turn(objects=[core(position=(783, 15)), *workers, *rangers])
    strategy._update_campaign_scout(turn)
    assert memory.campaign_scout_target is None
    assert memory.core_home_position == (783, 15)
