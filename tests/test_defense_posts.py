"""Fixed defensive posts must survive combat, roster changes, and restarts."""

from uuid import UUID

import pytest

from arena_hero_bot.geometry import add, manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig

from .factories import core, make_turn, object_id, unit


def _strategy(memory=None):
    return AggressiveStrategy(
        memory or WorldMemory(),
        StrategyConfig(target_workers=16, max_population=None, expedition_mode=True),
    )


def _stationed_units(center=(0, 0)):
    vanguards, rangers = AggressiveStrategy._symmetric_defense_offsets()
    return [
        unit(
            start + index,
            kind,
            position=(center[0] + dx, center[1] + dy),
        )
        for kind, start, offsets in (
            ("VANGUARD", 100, vanguards),
            ("RANGER", 200, rangers),
        )
        for index, (dx, dy) in enumerate(offsets)
    ]


@pytest.mark.parametrize("center", [(0, 0), (17, -23)])
def test_initial_posts_restore_symmetry_without_moving_stationed_guards(center):
    guards = _stationed_units(center)
    # Existing positions, rather than UUID order, identify the initial owners.
    guards[0]["id"], guards[3]["id"] = guards[3]["id"], guards[0]["id"]
    expected = {guard["id"]: tuple(guard["position"]) for guard in guards}
    for index in (1, 2, 3, 16, 18):
        guards[index]["position"][0] -= 1
    strategy = _strategy()
    turn = make_turn(objects=[core(position=center), *guards])

    report = strategy.decide(turn)

    assert strategy.memory.defense_anchor == center
    assert strategy.memory.defense_posts == expected
    assert sum(item.action == "MOVE" for item in report.decisions) == 5
    assert sum(item.action == "WAIT" for item in report.decisions) == 15
    for kind, count in (("VANGUARD", 4), ("RANGER", 16)):
        offsets = {
            (post[0] - center[0], post[1] - center[1])
            for guard in guards
            if guard["unit_type"] == kind
            for post in (strategy.memory.defense_posts[guard["id"]],)
        }
        assert len(offsets) == count
        assert offsets == {(-x, y) for x, y in offsets}
        assert offsets == {(x, -y) for x, y in offsets}
        assert offsets == {(-x, -y) for x, y in offsets}


def test_defender_returns_to_original_post_after_combat_and_restart(tmp_path):
    strategy = _strategy()
    guards = _stationed_units()
    strategy.decide(make_turn(objects=[core(), *guards]))
    expected = dict(strategy.memory.defense_posts)
    ranger_id = object_id(200)
    guards[4]["position"] = [1, -8]
    battle = make_turn(
        tick=101,
        objects=[
            core(),
            *guards,
            unit(900, "WORKER", controlled=False, position=(2, -8)),
        ],
    )

    strategy.decide(battle)

    assert battle.plan.unit_actions[UUID(ranger_id)].type == "SHOOT"
    assert strategy.memory.defense_posts == expected
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    restarted = _strategy(WorldMemory.load(path))
    peaceful = make_turn(
        tick=102,
        objects=[core(), *guards],
        events=[
            {
                "event_id": object_id(901),
                "tick": 102,
                "event_type": "UNIT_DESTROYED",
                "target_id": object_id(900),
            }
        ],
    )

    restarted.decide(peaceful)

    assert restarted.memory.defense_posts == expected
    returning = peaceful.plan.unit_actions[UUID(ranger_id)]
    assert returning.type == "MOVE"
    assert add((1, -8), returning.direction) == expected[ranger_id]
    guards[4]["position"] = list(expected[ranger_id])
    arrived = make_turn(tick=103, objects=[core(), *guards])
    restarted.decide(arrived)
    assert arrived.plan.unit_actions[UUID(ranger_id)].type == "WAIT"
    assert restarted.memory.defense_posts == expected


def test_crossing_another_defenders_post_does_not_transfer_ownership():
    strategy = _strategy()
    guards = _stationed_units()
    strategy.decide(make_turn(objects=[core(), *guards]))
    expected = dict(strategy.memory.defense_posts)
    guards[0]["position"], guards[1]["position"] = (
        guards[1]["position"],
        guards[0]["position"],
    )
    displaced = make_turn(tick=101, objects=[core(), *guards])

    strategy.decide(displaced)

    assert strategy.memory.defense_posts == expected
    for guard in guards[:2]:
        position = tuple(guard["position"])
        action = displaced.plan.unit_actions[UUID(guard["id"])]
        assert action.type == "MOVE"
        assert manhattan(add(position, action.direction), expected[guard["id"]]) < (
            manhattan(position, expected[guard["id"]])
        )


def test_casualty_and_lower_uuid_replacement_only_change_the_vacant_post():
    strategy = _strategy()
    guards = _stationed_units()
    strategy.decide(make_turn(objects=[core(), *guards]))
    expected = dict(strategy.memory.defense_posts)
    vacancy = expected.pop(object_id(101))
    survivors = [guard for guard in guards if guard["id"] != object_id(101)]

    strategy.decide(make_turn(tick=101, objects=[core(), *survivors]))

    assert strategy.memory.defense_posts == expected
    assert len(strategy.memory.defense_posts) == 19
    replacement = unit(2, "VANGUARD")
    extra_ranger = unit(3, "RANGER")
    strategy.decide(
        make_turn(tick=102, objects=[core(), *survivors, replacement, extra_ranger])
    )

    assert strategy.memory.defense_posts == {**expected, object_id(2): vacancy}
    assert strategy.memory.unit_roles[object_id(3)] == "patrol-1"


def test_healing_preserves_post_and_recovered_defender_returns_to_it():
    strategy = _strategy()
    guards = _stationed_units()
    strategy.decide(make_turn(objects=[core(), *guards]))
    expected = dict(strategy.memory.defense_posts)
    guards[0]["hp"] = 3
    guards[0]["position"] = [0, 0]
    healing = make_turn(tick=101, resources=20, objects=[core(), *guards])

    strategy.decide(healing)

    assert healing.plan.unit_actions[UUID(object_id(100))].type == "HEAL"
    assert strategy.memory.defense_posts == expected
    guards[0]["hp"] = 4
    recovered = make_turn(tick=102, objects=[core(), *guards])
    strategy.decide(recovered)
    returning = recovered.plan.unit_actions[UUID(object_id(100))]
    assert returning.type == "MOVE"
    assert manhattan(add((0, 0), returning.direction), expected[object_id(100)]) < 6
    assert strategy.memory.defense_posts == expected


def test_rock_does_not_move_individual_posts_or_allow_an_illegal_return():
    strategy = _strategy()
    guards = _stationed_units()
    strategy.decide(make_turn(objects=[core(), *guards]))
    expected = dict(strategy.memory.defense_posts)
    guards[4]["position"] = [1, -8]
    blocked = make_turn(tick=101, objects=[core(), *guards], obstacles=[(0, -8)])

    report = strategy.decide(blocked)

    assert strategy.memory.defense_posts == expected
    assert blocked.plan.unit_actions[UUID(object_id(200))].type == "WAIT"
    assert any(
        item.actor_id == object_id(200) and "post is obstructed" in item.reason
        for item in report.decisions
    )


def test_core_movement_translates_posts_without_reassigning_units():
    strategy = _strategy()
    guards = _stationed_units()
    strategy.decide(make_turn(objects=[core(), *guards]))
    expected = dict(strategy.memory.defense_posts)
    center = (5, -2)

    strategy.decide(make_turn(tick=101, objects=[core(position=center), *guards]))

    assert strategy.memory.defense_anchor == center
    assert strategy.memory.defense_posts == {
        unit_id: (post[0] + center[0], post[1] + center[1])
        for unit_id, post in expected.items()
    }


def test_posts_are_assigned_even_when_all_defenders_are_recovering():
    strategy = _strategy()
    guards = _stationed_units()
    expected = {guard["id"]: tuple(guard["position"]) for guard in guards}
    for guard in guards:
        guard["hp"] = 1
    recovering = make_turn(objects=[core(), *guards])

    strategy.decide(recovering)

    assert strategy.memory.defense_posts == expected
    assert all(
        action.type == "MOVE" for action in recovering.plan.unit_actions.values()
    )


def test_losing_core_and_defenders_clears_obsolete_posts():
    strategy = _strategy()
    strategy.decide(make_turn(objects=[core(), *_stationed_units()]))

    strategy.decide(make_turn(tick=101, status="RESPAWNING", respawn_at_tick=110))

    assert strategy.memory.defense_posts == {}
    assert strategy.memory.defense_anchor is None
    assert strategy._defensive_layout is None


@pytest.mark.parametrize("persisted_posts", [False, True])
def test_reduced_formation_keeps_cardinal_owners_and_releases_eight_vanguards(
    tmp_path, persisted_posts
):
    center = (17, -23)
    old_offsets = (
        (0, -6),
        (4, -4),
        (6, 0),
        (4, 4),
        (0, 6),
        (-4, 4),
        (-6, 0),
        (-4, -4),
    )
    guards = [
        unit(100 + index, "VANGUARD", position=(center[0] + dx, center[1] + dy))
        for index, (dx, dy) in enumerate(old_offsets)
    ]
    guards.extend(
        guard for guard in _stationed_units(center) if guard["unit_type"] == "RANGER"
    )
    posts = {guard["id"]: tuple(guard["position"]) for guard in guards}
    roles = dict.fromkeys(posts, "defense")
    cardinal_ids = {object_id(index) for index in (100, 102, 104, 106)}
    released_ids = {object_id(index) for index in (101, 103, 105, 107)}
    patrols = []
    for team in range(1, 5):
        for index, kind in enumerate(("VANGUARD", "RANGER", "RANGER")):
            member = unit(
                300 + team * 3 + index, kind, position=(center[0] + 20, center[1])
            )
            patrols.append(member)
            roles[member["id"]] = f"patrol-{team}"
            if kind == "VANGUARD":
                released_ids.add(member["id"])
    roles[object_id(400)] = roles[object_id(401)] = "staged"
    original_roles = dict(roles)
    memory = WorldMemory(
        core_home_position=center,
        defense_anchor=center if persisted_posts else None,
        defense_posts=posts.copy() if persisted_posts else {},
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    if persisted_posts:
        # The cardinal owner is healing while a released diagonal owner is
        # crossing its post. Temporary positions must not swap their roles.
        guards[0]["hp"] = 3
        guards[0]["position"] = list(center)
        guards[1]["position"] = list(posts[object_id(100)])
    objects = [
        core(position=center),
        *guards,
        *patrols,
        unit(400, "VANGUARD", position=(center[0] + 13, center[1])),
        unit(401, "RANGER", position=(center[0] + 12, center[1] + 1)),
    ]
    strategy = _strategy(memory)
    expected_posts = {
        uid: post for uid, post in posts.items() if uid not in released_ids
    }
    expected_roles = {
        uid: "staged" if uid in released_ids else role
        for uid, role in original_roles.items()
    }

    migrated = make_turn(resources=10000, objects=objects)
    strategy.decide(migrated)

    assert strategy.memory.defense_posts == expected_posts
    assert strategy.memory.unit_roles == expected_roles
    assert cardinal_ids <= set(strategy.memory.defense_posts)
    assert len(released_ids) == 8
    assert strategy._staged_ids == {
        UUID(uid) for uid in released_ids | {object_id(400), object_id(401)}
    }
    assert strategy.memory.expedition_squads == []
    if persisted_posts:
        assert migrated.plan.unit_actions[UUID(object_id(100))].type == "HEAL"
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    resumed = _strategy(WorldMemory.load(path))

    resumed.decide(make_turn(tick=101, resources=10000, objects=objects))

    assert resumed.memory.unit_roles == expected_roles
    assert resumed.memory.defense_posts == expected_posts
    assert resumed._staged_ids == strategy._staged_ids
