"""Exploration values actual new sight, travel cost, and cooperative routes."""

from dataclasses import replace
from itertools import pairwise

import pytest

from arena_hero_bot import exploration
from arena_hero_bot.exploration import (
    ExpeditionNavigator,
    ExplorationMap,
    ExplorationRoute,
    visible_cells,
)
from arena_hero_bot.geometry import manhattan


def _planner(seen=(), rocks=(), origin=(0, 0)):
    atlas = ExplorationMap()
    atlas.mark(set(seen), 99)
    atlas.observe([(origin, 5)], set(rocks), 100)
    return ExpeditionNavigator(atlas, set(rocks), set(rocks), 5)


def _plan(navigator, origin=(0, 0), bearing=(1, 0), reserved=()):
    return navigator.plan(origin, bearing, 100, ExplorationRoute(), set(reserved))


@pytest.mark.parametrize("rock", [(1, 0), (0, 1)])
def test_vision_corner_is_blocked_by_either_side(rock):
    view = visible_cells((0, 0), 5, {rock})
    assert rock in view
    assert (1, 1) not in view
    assert (2, 2) not in view
    assert (-1, -1) in view


def test_vision_reveals_rock_but_not_ground_behind_it():
    view = visible_cells((0, 0), 5, {(2, 0)})
    assert {(0, 0), (1, 0), (2, 0), (0, 5)} <= view
    assert not ({(3, 0), (5, 0), (3, 3), (0, 6)} & view)


def test_observation_unions_allies_without_marking_occluded_empty_ground():
    atlas = ExplorationMap()
    atlas.observe([((0, 0), 3), ((0, 0), 5)], {(1, 0)}, 100)
    expected = visible_cells((0, 0), 5, {(1, 0)})
    assert atlas.new_cells == len(expected) - 1
    assert atlas.seen((1, 0)) and atlas.seen((0, 5))
    assert not atlas.seen((2, 0))
    atlas.observe([((0, 0), 5)], {(1, 0)}, 101)
    assert atlas.new_cells == 0
    atlas.observe([((2, 0), 5)], {(1, 0)}, 102)
    assert atlas.new_cells > 0 and atlas.seen((2, 0))


def test_atlas_handles_negative_tile_edges_and_retains_recent_ground(monkeypatch):
    monkeypatch.setattr(exploration, "MAX_SEEN_TILES", 2)
    atlas = ExplorationMap()
    atlas.mark({(-1, -1)}, 1)
    atlas.mark({(0, 0)}, 2)
    atlas.mark({(-1, -1)}, 3)
    atlas.mark({(16, 16)}, 4)
    assert atlas.seen((-1, -1)) and atlas.seen((16, 16))
    assert not atlas.seen((-1, 0)) and not atlas.seen((0, 0))
    assert len(atlas.tiles) == 2


def test_turns_sideways_out_of_a_known_corridor():
    navigator = _planner({(x, y) for x in range(-40, 41) for y in range(-5, 6)})
    route = _plan(navigator)
    assert route.goal is not None and abs(route.goal[1]) > 5
    assert route.waypoint is not None and route.waypoint[0] == 0
    assert route.expected_gain > 50


def test_nearby_unknown_beats_farther_progress_along_original_heading():
    navigator = _planner({(x, y) for x in range(-6, 25) for y in range(-40, 41)})
    route = _plan(navigator)
    assert route.goal is not None and route.goal[0] < -6
    assert len(route.path) - 1 < 24


def test_leaves_a_small_unvisited_hole_when_a_broad_frontier_is_better():
    hole = (8, 1)
    known = {(x, y) for x in range(-12, 41) for y in range(-40, 41)} - {hole}
    navigator = _planner(known)
    route = _plan(navigator)
    assert route.goal is not None and route.goal[0] < -12
    assert not navigator.exploration.seen(hole)
    assert hole not in navigator.reservation(route)


def test_distance_from_world_origin_does_not_change_exploration_policy():
    known = {(x, y) for x in range(-6, 25) for y in range(-40, 41)}
    near = _plan(_planner(known))
    shift = (20000, -30000)
    far = _plan(
        _planner({(x + shift[0], y + shift[1]) for x, y in known}, origin=shift),
        origin=shift,
    )
    assert near.path == tuple((x - shift[0], y - shift[1]) for x, y in far.path)
    assert near.expected_gain == pytest.approx(far.expected_gain)


def test_squads_choose_different_frontiers_without_blocking_each_other():
    navigator = _planner()
    first = _plan(navigator)
    reserved = navigator.reservation(first)
    second = _plan(navigator, reserved=reserved)
    assert first.goal is not None and second.goal is not None
    assert manhattan(first.goal, second.goal) >= 12
    assert len(navigator.reservation(second) - reserved) >= 50
    assert first.path[0] == second.path[0]  # Shared transit is still legal.


def test_planning_a_route_does_not_count_it_as_explored():
    navigator = _planner()
    before = dict(navigator.exploration.tiles)
    route = _plan(navigator)
    assert route.goal is not None
    assert not navigator.exploration.seen(route.goal)
    assert before == navigator.exploration.tiles


def test_small_map_changes_keep_a_useful_shared_goal():
    navigator = _planner()
    route = _plan(navigator)
    assert route.goal is not None
    origin = route.path[1]
    navigator.exploration.observe([(origin, 5)], set(), 101)
    navigator.exploration.mark({(-8, 2)}, 101)
    next_route = navigator.plan(origin, (1, 0), 101, route, set())
    assert next_route.goal == route.goal
    assert next_route.assigned_tick == route.assigned_tick
    assert next_route.progress_tick == 101


def test_stalled_route_changes_approach_instead_of_extending_the_same_goal():
    navigator = _planner()
    first = route = _plan(navigator)
    for tick in range(101, 111):
        route = navigator.plan((0, 0), (1, 0), tick, route, set())
    assert route.path and route.path[1] != first.path[1]
    assert (first.path[1], 150) in route.cooldowns
    expired = navigator.plan(
        (0, 0), (1, 0), 151, ExplorationRoute(cooldowns=route.cooldowns), set()
    )
    assert expired.path[1] == first.path[1]
    assert not expired.cooldowns


def test_a_new_wall_invalidates_the_route_before_the_regular_recheck():
    navigator = _planner()
    route = _plan(navigator)
    rocks = {route.path[3]}
    updated = ExpeditionNavigator(navigator.exploration, rocks, rocks, 5)
    replacement = updated.plan((0, 0), (1, 0), 101, route, set())
    assert replacement.path and not (set(replacement.path) & rocks)
    assert replacement.path != route.path
    assert all(manhattan(a, b) == 1 for a, b in pairwise(replacement.path))


def test_known_basin_transit_seeks_unknown_ground_without_local_coverage_work():
    known = {(x, y) for x in range(-50, 51) for y in range(-50, 51)}
    navigator = _planner(known)
    route = _plan(navigator)
    assert route.goal is not None and route.goal[0] > 0
    assert route.expected_gain == 0  # Honest transit, not invented information.
    assert len(route.path) > 20


def test_fully_observed_closed_pocket_has_no_unreachable_exploration_goal():
    rocks = {
        (x, y) for x in range(-3, 4) for y in range(-3, 4) if max(abs(x), abs(y)) == 3
    }
    known = {(x, y) for x in range(-5, 6) for y in range(-5, 6)}
    route = _plan(_planner(known, rocks))
    assert route.goal is None and route.waypoint is None


def test_detour_cost_can_outweigh_unknown_ground_behind_a_wall():
    rocks = {(3, y) for y in range(-24, 25)}
    known = {(x, y) for x in range(-10, 4) for y in range(-40, 41)}
    navigator = _planner(known, rocks)
    route = _plan(navigator)
    assert route.goal is not None and route.goal[0] < -10
    assert not (set(route.path) & rocks)


def test_observed_route_yields_to_a_new_frontier_at_recheck():
    navigator = _planner()
    previous = _plan(navigator)
    navigator.exploration.mark(navigator.reservation(previous), 105)
    updated = navigator.plan(
        (0, 0), (1, 0), 106, replace(previous, progress_tick=105), set()
    )
    assert updated.goal != previous.goal
    assert updated.expected_gain > 0
