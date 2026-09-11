"""Expeditions must replace blocked exploration waypoints without splitting."""

from collections import Counter

import pytest

from arena_hero_bot.geometry import add, adjacent_positions, next_step
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
    front = (2 * bearing[0], 2 * bearing[1])
    assert goal[0] * bearing[0] + goal[1] * bearing[1] > (
        front[0] * bearing[0] + front[1] * bearing[1]
    )
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
    for tick in range(100, 116):
        turn = make_turn(
            tick=tick,
            objects=[core(position=(-30, -30)), *members],
            obstacles=rocks,
        )
        strategy.decide(turn)
        positions = {}
        for member in turn.units:
            action = turn.plan.unit_actions.get(member.id)
            positions[str(member.id)] = (
                add(member.position, action.direction)
                if action is not None and action.type == "MOVE"
                else member.position
            )
        assert not (set(positions.values()) & rocks)
        assert max(Counter(positions.values()).values()) <= 2
        for member in members:
            member["position"] = list(positions[member["id"]])

    assert all(member["position"][0] > 6 for member in members)


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
