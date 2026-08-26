"""Geometry and pathfinding behavior."""

from arena_hero import Direction

from arena_hero_bot.geometry import (
    add,
    adjacent_positions,
    direction_between,
    firing_positions,
    line_of_fire,
    manhattan,
    next_step,
)


def test_basic_geometry() -> None:
    assert add((2, 3), Direction.LEFT) == (1, 3)
    assert manhattan((0, 0), (-2, 3)) == 5
    assert direction_between((0, 0), (0, 1)) is Direction.DOWN
    assert direction_between((0, 0), (1, 1)) is None
    assert set(adjacent_positions((0, 0))) == {(0, -1), (1, 0), (0, 1), (-1, 0)}


def test_ranger_line_of_fire_geometry_and_obstacles() -> None:
    assert line_of_fire((0, 0), (3, 3), set())
    assert line_of_fire((0, 0), (0, -2), set())
    assert not line_of_fire((0, 0), (2, 1), set())
    assert not line_of_fire((0, 0), (4, 0), set())
    assert not line_of_fire((0, 0), (0, 0), set())
    assert not line_of_fire((0, 0), (3, 3), {(2, 2)})
    assert line_of_fire((0, 0), (3, 3), {(2, 1)})
    assert len(firing_positions((10, 10))) == 24


def test_pathfinder_routes_around_obstacle() -> None:
    direction = next_step((0, 0), (2, 0), blocked={(1, 0)})
    assert direction in {Direction.UP, Direction.DOWN}


def test_pathfinder_penalizes_recent_backtracking() -> None:
    direction = next_step(
        (0, 0),
        (2, 0),
        blocked={(1, 0)},
        recent=((0, -1),),
    )
    assert direction is Direction.DOWN


def test_pathfinder_respects_blocked_goal_unless_allowed() -> None:
    assert next_step((0, 0), (1, 0), blocked={(1, 0)}) is not Direction.RIGHT
    assert (
        next_step((0, 0), (1, 0), blocked={(1, 0)}, allow_goal=True) is Direction.RIGHT
    )


def test_pathfinder_returns_none_when_surrounded() -> None:
    blocked = {(0, -1), (1, 0), (0, 1), (-1, 0)}
    assert next_step((0, 0), (5, 0), blocked=blocked) is None


def test_pathfinder_distinguishes_no_route_from_a_greedy_step() -> None:
    walled = {(2, 0), (4, 0), (3, -1), (3, 1)}
    assert next_step((0, 0), (3, 0), blocked=set(walled)) is Direction.RIGHT
    assert next_step((0, 0), (3, 0), blocked=set(walled), require_path=True) is None
    assert (
        next_step((0, 0), (3, 0), blocked=set(), require_path=True) is Direction.RIGHT
    )
