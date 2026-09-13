"""Expeditions must replace blocked exploration waypoints without splitting."""

from collections import Counter
from dataclasses import replace

import pytest

import arena_hero_bot.strategy as strategy_module
from arena_hero_bot.geometry import add, adjacent_positions, manhattan, next_step
from arena_hero_bot.memory import ExpeditionSquad, WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig

from .factories import core, make_turn, object_id, unit


def _strategy(bearing=(1, 0), obstacles=()):
    memory = WorldMemory(
        obstacles=set(obstacles),
        core_home_position=(-30, -30),
        expedition_squads=[
            ExpeditionSquad(
                serial=1,
                members=tuple(object_id(number) for number in range(2, 6)),
                bearing=bearing,
            )
        ],
    )
    return AggressiveStrategy(
        memory,
        StrategyConfig(target_workers=16, max_population=None, expedition_mode=True),
    )


def _members(bearing=(1, 0)):
    return [
        unit(number, kind, position=(distance * bearing[0], distance * bearing[1]))
        for number, kind, distance in (
            (2, "VANGUARD", 0),
            (3, "VANGUARD", 1),
            (4, "RANGER", 2),
            (5, "RANGER", -1),
        )
    ]


@pytest.mark.parametrize("bearing", [(1, 0), (-1, 0), (1, -1)])
@pytest.mark.parametrize("remembered", [False, True])
def test_whole_squad_moves_toward_one_clear_goal_when_waypoint_is_rock(
    bearing, remembered
):
    rock = (6 * bearing[0], 6 * bearing[1])
    strategy = _strategy(bearing, obstacles=[rock] if remembered else ())
    turn = make_turn(
        objects=[core(position=(-30, -30)), *_members(bearing)],
        obstacles=() if remembered else [rock],
    )

    report = strategy.decide(turn)

    decisions = [item for item in report.decisions if item.actor_kind != "CORE"]
    assert len(decisions) == 4
    assert all(item.action == "MOVE" for item in decisions)
    goals = {item.target for item in decisions}
    assert len(goals) == 1
    goal = goals.pop()
    assert goal is not None and goal != rock
    route = strategy.memory.expedition_squads[0].exploration_route
    assert route.expected_gain > 0
    assert goal == route.waypoint
    assert rock not in route.path
    for member in turn.units:
        action = turn.plan.unit_actions[member.id]
        assert action.type == "MOVE"
        assert add(member.position, action.direction) != rock


def test_replacement_goal_avoids_an_empty_cell_sealed_inside_rocks():
    rocks = {(6, 0), (8, 0), (7, -1), (7, 1)}
    strategy = _strategy(obstacles=rocks)
    turn = make_turn(objects=[core(position=(-30, -30)), *_members()])

    goal = strategy._expedition_goal(turn.rangers[0], turn)

    assert goal not in rocks | {(7, 0)}
    assert next_step((2, 0), goal, blocked=rocks, require_path=True) is not None


def test_expedition_keeps_advancing_after_routing_around_a_blocked_waypoint():
    rocks = {(6, 0)}
    strategy = _strategy(obstacles=rocks)
    members = _members()
    moves = Counter()
    new_ground = 0
    for tick in range(100, 116):
        turn = make_turn(
            tick=tick,
            objects=[core(position=(-30, -30)), *members],
            obstacles=rocks,
        )
        strategy.decide(turn)
        if tick > 100:
            new_ground += strategy.memory.exploration.new_cells
        positions = {}
        for member in turn.units:
            action = turn.plan.unit_actions.get(member.id)
            if action is not None and action.type == "MOVE":
                moves[str(member.id)] += 1
            positions[str(member.id)] = (
                add(member.position, action.direction)
                if action is not None and action.type == "MOVE"
                else member.position
            )
        assert not (set(positions.values()) & rocks)
        assert max(Counter(positions.values()).values()) <= 2
        connected = {next(iter(positions.values()))}
        while True:
            expanded = {
                position
                for position in positions.values()
                if any(manhattan(position, neighbor) <= 4 for neighbor in connected)
            }
            if expanded == connected:
                break
            connected = expanded
        assert connected == set(positions.values())
        for member in members:
            member["position"] = list(positions[member["id"]])

    assert all(moves[member["id"]] >= 8 for member in members)
    assert new_ground > 60


def test_trapped_expedition_member_still_waits_instead_of_entering_rock():
    front = (2, 0)
    rocks = {*adjacent_positions(front), (6, 0)}
    strategy = _strategy(obstacles=rocks)
    turn = make_turn(
        objects=[core(position=(-30, -30)), unit(4, "RANGER", position=front)]
    )

    report = strategy.decide(turn)

    assert strategy._expedition_goal(turn.rangers[0], turn) == front
    assert (
        next(item for item in report.decisions if item.actor_id == object_id(4)).action
        == "WAIT"
    )
    assert not turn.plan.unit_actions


def test_one_shared_route_and_heading_turn_out_of_a_known_corridor():
    strategy = _strategy()
    strategy.memory.exploration.mark(
        {(x, y) for x in range(-40, 41) for y in range(-5, 6)}, 99
    )
    turn = make_turn(objects=[core(position=(-30, -30)), *_members()])

    report = strategy.decide(turn)

    route = strategy.memory.expedition_squads[0].exploration_route
    assert route.goal is not None and abs(route.goal[1]) > 5
    assert strategy.memory.expedition_squads[0].bearing[1] != 0
    decisions = [item for item in report.decisions if item.actor_kind != "CORE"]
    assert all(item.action == "MOVE" for item in decisions)
    assert {item.target for item in decisions} == {route.waypoint}


def test_new_route_waits_for_detached_members_to_regroup():
    strategy = _strategy()
    members = _members()
    members[2]["position"] = [30, 0]
    turn = make_turn(objects=[core(position=(-30, -30)), *members])

    report = strategy.decide(turn)

    assert not strategy.memory.expedition_squads[0].exploration_route.path
    front = next(item for item in report.decisions if item.actor_id == object_id(4))
    assert front.action == "WAIT"
    assert "laggards" in front.reason
    assert any(
        "close up" in item.reason and item.action == "MOVE" for item in report.decisions
    )


def test_exploration_goal_and_map_survive_a_restart(tmp_path):
    strategy = _strategy()
    turn = make_turn(objects=[core(position=(-30, -30)), *_members()])
    strategy.decide(turn)
    original = strategy.memory.expedition_squads[0]
    path = tmp_path / "memory.json"
    strategy.memory.save(path)

    loaded = WorldMemory.load(path)
    assert loaded.exploration.tiles == strategy.memory.exploration.tiles
    assert loaded.expedition_squads[0] == original
    resumed = AggressiveStrategy(loaded, strategy.config)
    resumed.decide(
        make_turn(tick=101, objects=[core(position=(-30, -30)), *_members()])
    )
    route = loaded.expedition_squads[0].exploration_route
    assert route.goal == original.exploration_route.goal
    assert route.assigned_tick == original.exploration_route.assigned_tick
    assert loaded.exploration.new_cells == 0


def test_a_casualty_preserves_survivors_squad_identity_and_route():
    template = _strategy()
    template.memory.expedition_squads = [
        replace(template.memory.expedition_squads[0], serial=47)
    ]
    strategy = AggressiveStrategy(template.memory, template.config)
    strategy.decide(make_turn(objects=[core(position=(-30, -30)), *_members()]))
    original = strategy.memory.expedition_squads[0]

    strategy._update_expeditions(
        make_turn(tick=101, objects=[core(position=(-30, -30)), *_members()[1:]])
    )

    survivors = strategy.memory.expedition_squads[0]
    assert survivors.serial == 47
    assert set(survivors.members) == set(original.members) - {object_id(2)}
    assert survivors.exploration_route == original.exploration_route
    assert all(
        strategy.memory.unit_roles[mid] == "expedition-47" for mid in survivors.members
    )


def test_coordinator_reserves_different_routes_for_nearby_squads():
    template = _strategy()
    template.memory.expedition_squads.append(
        ExpeditionSquad(2, tuple(object_id(number) for number in range(6, 10)), (1, 0))
    )
    strategy = AggressiveStrategy(template.memory, template.config)
    second = [
        unit(number, kind, position=(x, 3))
        for number, kind, x in (
            (6, "VANGUARD", 0),
            (7, "VANGUARD", 1),
            (8, "RANGER", 2),
            (9, "RANGER", -1),
        )
    ]
    turn = make_turn(objects=[core(position=(-30, -30)), *_members(), *second])

    report = strategy.decide(turn)

    first_route, second_route = [
        s.exploration_route for s in strategy.memory.expedition_squads
    ]
    assert first_route.goal is not None and second_route.goal is not None
    assert manhattan(first_route.goal, second_route.goal) >= 8
    for squad in strategy.memory.expedition_squads:
        targets = {
            item.target for item in report.decisions if item.actor_id in squad.members
        }
        assert targets == {squad.exploration_route.waypoint}


@pytest.mark.parametrize("front_number", [2, 4])
def test_quiet_pursuit_holds_detached_front_and_rejoins_before_chasing(front_number):
    template = _strategy()
    template.memory.expedition_squads = [
        replace(
            template.memory.expedition_squads[0],
            pursuit_target_id=object_id(90),
            pursuit_position=(100, 0),
            pursuit_direction=(1, 0),
        )
    ]
    strategy = AggressiveStrategy(template.memory, template.config)
    members = _members()
    for member in members:
        if member["id"] == object_id(front_number):
            member["position"] = [30, 0]

    report = strategy.decide(make_turn(objects=[core(position=(-30, -30)), *members]))

    decisions = [item for item in report.decisions if item.actor_kind != "CORE"]
    front = next(item for item in decisions if item.actor_id == object_id(front_number))
    assert front.action == "WAIT" and "laggards" in front.reason
    for item in decisions:
        if item.actor_id != object_id(front_number):
            assert item.action == "MOVE"
            assert "close up" in item.reason
    assert not any("pursuit after lost sight" in item.reason for item in decisions)


def test_long_rejoin_requires_a_complete_route_with_a_bounded_larger_budget(
    monkeypatch,
):
    strategy = _strategy()
    turn = make_turn(
        objects=[
            core(position=(-30, -30)),
            unit(2, "VANGUARD", position=(400, 0)),
            unit(4, "RANGER", position=(0, 0)),
        ]
    )
    calls = []
    original = strategy_module.next_step

    def recorded(origin, goal, **kwargs):
        if origin == (0, 0) and goal[0] >= 399:
            calls.append((kwargs.get("max_expansions"), kwargs.get("require_path")))
        return original(origin, goal, **kwargs)

    monkeypatch.setattr(strategy_module, "next_step", recorded)
    report = strategy.decide(turn)

    ranger = next(item for item in report.decisions if item.actor_id == object_id(4))
    assert ranger.action == "MOVE" and "close up" in ranger.reason
    assert calls and all(budget == 8192 and required for budget, required in calls)


def test_trapped_rejoining_member_waits_without_greedy_orbit():
    rocks = set(adjacent_positions((0, 0)))
    strategy = _strategy(obstacles=rocks)
    turn = make_turn(
        objects=[
            core(position=(-30, -30)),
            unit(2, "VANGUARD", position=(20, 0)),
            unit(4, "RANGER", position=(0, 0)),
        ],
        obstacles=rocks,
    )

    report = strategy.decide(turn)

    ranger = next(item for item in report.decisions if item.actor_id == object_id(4))
    assert ranger.action == "WAIT"
    assert ranger.reason == "no safe route to rejoin the expedition"
    assert object_id(4) not in {str(mid) for mid in turn.plan.unit_actions}


def test_rejoining_member_does_not_choose_an_enemy_occupied_forward_cell():
    strategy = _strategy()
    enemy_cell = (11, 0)
    turn = make_turn(
        objects=[
            core(position=(-30, -30)),
            unit(2, "VANGUARD", position=(10, 0)),
            unit(4, "RANGER", position=(0, 0)),
            core(90, controlled=False, position=enemy_cell),
        ]
    )

    goal = strategy._expedition_close_cell(turn.rangers[0], (10, 0), turn)

    assert goal != enemy_cell
    assert goal in adjacent_positions((10, 0))
    assert next_step((0, 0), goal, blocked={enemy_cell}, require_path=True) is not None
