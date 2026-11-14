"""Actual stationary guards spread out without claiming unstaffed coverage."""

from uuid import UUID

import pytest

from arena_hero_bot import defense_layout
from arena_hero_bot.exploration import visible_cells
from arena_hero_bot.geometry import manhattan
from arena_hero_bot.memory import UnitGoal, WorldMemory
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    WORKER_PATROL_PURPOSE,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, unit
from .test_core_traffic import settled_positions
from .test_shared_movement import context_for

HOME = (15, 15)
BOUNDS = (0, 31, 0, 31)


def strategy_for(guards, **fields):
    return AggressiveStrategy(
        WorldMemory(
            core_home_position=HOME,
            core_layout_version=CORE_LAYOUT_VERSION,
            campaign_phase=4,
            campaign_last_core_id=core()["id"],
            unit_roles={g["id"]: "defense" for g in guards},
            unit_roles_initialized=True,
            **fields,
        ),
        StrategyConfig(target_workers=16, campaign_mode=True, expedition_mode=True),
    )


def assert_covered(center, posts, rocks, radius):
    seen = set(visible_cells(center, 5, rocks))
    for post in posts:
        seen.update(visible_cells(post, 5, rocks))
    required = {
        (center[0] + dx, center[1] + dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if abs(dx) + abs(dy) <= radius
    } - rocks
    assert required <= seen


@pytest.mark.parametrize("count", [1, 4, 8, 10, 16, 25, 40])
@pytest.mark.parametrize("rocky", [False, True])
def test_counted_posts_cover_every_nonrock_cell(count, rocky):
    rocks = {(17, 16), (18, 16), (13, 13), (11, 18)} if rocky else set()
    posts, radius = defense_layout.plan_posts(HOME, count, tuple(sorted(rocks)), BOUNDS)
    assert len(posts) == count
    assert len(set(posts)) == count
    assert not set(posts) & rocks
    assert all(0 <= x <= 31 and 0 <= y <= 31 for x, y in posts)
    assert max(manhattan(HOME, p) for p in posts) <= radius <= 12
    assert_covered(HOME, posts, rocks, radius)
    if count >= 16 and not rocky:
        assert radius == 12
    if count == 25:
        assert (
            min(manhattan(a, b) for i, a in enumerate(posts) for b in posts[i + 1 :])
            >= 3
        )


def test_sealed_nonrock_cell_remains_a_gap_but_rocks_are_not_gaps():
    hole = (23, 15)
    rocks = {(22, 15), (24, 15), (23, 14), (23, 16)}
    posts, radius = defense_layout.plan_posts(HOME, 25, tuple(sorted(rocks)), BOUNDS)
    assert radius < manhattan(HOME, hole)
    assert hole not in posts
    assert_covered(HOME, posts, rocks, radius)
    # A known rock at that same location removes the unreachable blind cell.
    rocks.add(hole)
    posts, radius = defense_layout.plan_posts(HOME, 25, tuple(sorted(rocks)), BOUNDS)
    assert radius >= manhattan(HOME, hole)
    assert_covered(HOME, posts, rocks, radius)


def test_no_legal_roster_is_never_reported_as_a_gap_free_template():
    rocks = tuple((15 + dx, 15 + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    assert defense_layout.plan_posts(HOME, 4, rocks, BOUNDS) == ((), 0)


def test_posts_remain_owned_across_restart_new_recruit_and_casualty(tmp_path):
    guards = [
        unit(100 + n, "RANGER", position=(12 + n % 5, 12 + n // 5)) for n in range(25)
    ]
    strategy = strategy_for(guards)
    turn = make_turn(objects=[core(position=HOME), *guards])
    posts = strategy._stationary_defense_assignments(turn.rangers, context_for(turn))
    assert strategy.memory.defense_spread_version == defense_layout.LAYOUT_VERSION
    assert (
        strategy.memory.defense_visible_radius < strategy.memory.defense_coverage_radius
    )
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    resumed = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    assert (
        resumed._stationary_defense_assignments(turn.rangers, context_for(turn))
        == posts
    )
    newcomer = unit(99, "RANGER", position=HOME)
    resumed.memory.unit_roles[newcomer["id"]] = "defense"
    grown = make_turn(tick=101, objects=[core(position=HOME), *guards, newcomer])
    expanded = resumed._stationary_defense_assignments(
        grown.rangers, context_for(grown)
    )
    assert all(expanded[uid] == p for uid, p in posts.items())
    casualty = make_turn(tick=102, objects=[core(position=HOME), *guards])
    assert (
        resumed._stationary_defense_assignments(casualty.rangers, context_for(casualty))
        == posts
    )


def test_newly_confirmed_rock_repairs_posts_and_their_actual_coverage():
    guards = [
        unit(100 + n, "RANGER", position=(12 + n % 5, 12 + n // 5)) for n in range(25)
    ]
    strategy = strategy_for(guards)
    initial = make_turn(objects=[core(position=HOME), *guards])
    posts = strategy._stationary_defense_assignments(
        initial.rangers, context_for(initial)
    )
    rock = max(posts.values(), key=lambda p: manhattan(HOME, p))
    strategy.memory.obstacles.add(rock)
    turn = make_turn(tick=101, objects=[core(position=HOME), *guards])
    repaired = strategy._stationary_defense_assignments(turn.rangers, context_for(turn))
    assert len(repaired) == len(guards)
    assert rock not in repaired.values()
    assert_covered(
        HOME, tuple(repaired.values()), {rock}, strategy.memory.defense_coverage_radius
    )
    assert (
        strategy.memory.defense_visible_radius < strategy.memory.defense_coverage_radius
    )


@pytest.mark.parametrize("contact", ["current", "remembered", "new_home"])
def test_first_spread_waits_for_current_and_recent_combat_to_clear(contact):
    guards = [
        unit(100 + n, "RANGER", position=(12 + n % 5, 12 + n // 5)) for n in range(20)
    ]
    strategy = strategy_for(guards)
    enemy = unit(500, "RANGER", position=(15, 6), controlled=False)
    if contact == "new_home":
        initial = make_turn(objects=[core(position=HOME), *guards])
        strategy._stationary_defense_assignments(initial.rangers, context_for(initial))
        strategy._commit_symmetric_home(HOME)
    elif contact == "remembered":
        strategy.memory.observe(
            make_turn(tick=99, objects=[core(position=HOME), *guards, enemy])
        )
    turn = make_turn(
        objects=[
            core(position=HOME),
            *guards,
            *([] if contact == "remembered" else [enemy]),
        ]
    )
    posts = strategy._defensive_perimeter_assignments(context_for(turn))
    assert strategy.memory.defense_spread_version == 0
    assert max(manhattan(HOME, p) for p in posts.values()) <= 8


@pytest.mark.parametrize("mode", ["moving", "pending", "scouting", "compressed"])
def test_migration_and_compressed_postures_keep_their_existing_rules(mode):
    guards = [
        unit(100 + n, "RANGER", position=(12 + n % 5, 12 + n // 5)) for n in range(20)
    ]
    strategy = strategy_for(guards)
    obj = core(position=HOME)
    if mode == "moving":
        obj = core(
            position=HOME,
            state="MOVING",
            move_direction="RIGHT",
            move_progress=1,
            move_required_ticks=4,
            destination=[16, 15],
        )
    elif mode == "pending":
        strategy.memory.core_home_position = (47, 15)
    elif mode == "scouting":
        strategy.memory.campaign_scout_target = (47, 15)
    else:
        strategy.memory.defense_compressed = True
    turn = make_turn(objects=[obj, *guards])
    posts = strategy._defensive_perimeter_assignments(context_for(turn))
    assert strategy.memory.defense_spread_version == 0
    assert max(manhattan(HOME, p) for p in posts.values()) <= (
        4 if mode == "compressed" else 8
    )


@pytest.mark.parametrize("restart", [False, True])
def test_full_decisions_spread_then_hold_without_opening_interior_gaps(
    tmp_path, restart
):
    strategy = strategy_for([])
    compact, _radius = strategy._symmetric_defense_posts(HOME, set())
    guards = [unit(100 + n, "RANGER", position=p) for n, p in enumerate(compact)]
    patrols = [unit(200 + n, "RANGER", position=(3 + n, 3)) for n in range(20)]
    workers = [unit(300 + n, "WORKER", position=(3 + n, 27)) for n in range(16)]
    roster = [*guards, *patrols, *workers]
    strategy.memory.unit_roles = {
        **{g["id"]: "defense" for g in guards},
        **{p["id"]: f"patrol-{n // 2 + 1}" for n, p in enumerate(patrols)},
    }
    strategy.memory.defense_posts = {g["id"]: tuple(g["position"]) for g in guards}
    strategy.memory.defense_anchor = HOME
    previous_radius = 8
    final_posts = None
    for tick in range(100, 125):
        if restart and tick == 103:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
        turn = make_turn(tick=tick, objects=[core(position=HOME), *roster])
        report = strategy.decide(turn)
        assert not any("deadline" in d.reason for d in report.decisions)
        positions = settled_positions(turn)
        actual_guards = tuple(positions[UUID(g["id"])] for g in guards)
        assert_covered(HOME, actual_guards, set(), previous_radius)
        previous_radius = strategy.memory.defense_established_radius
        if final_posts is None:
            final_posts = dict(strategy.memory.defense_posts)
        assert strategy.memory.defense_posts == final_posts
        for member in roster:
            member["position"] = list(positions[UUID(member["id"])])
    assert previous_radius == 12
    assert final_posts is not None
    assert all(tuple(g["position"]) == final_posts[g["id"]] for g in guards)


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("remembered", [False, True])
def test_worker_replaces_a_confirmed_rock_patrol_goal(tmp_path, restart, remembered):
    worker = unit(300, "WORKER", position=(5, 5))
    rock = (7, 5)
    strategy = strategy_for([])
    strategy.memory.set_goal(
        worker["id"], UnitGoal(rock, 90, WORKER_PATROL_PURPOSE, (4, 5))
    )
    if remembered:
        strategy.memory.obstacles.add(rock)
    if restart:
        path = tmp_path / "memory.json"
        strategy.memory.save(path)
        strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    turn = make_turn(
        objects=[core(position=HOME), worker], obstacles=[] if remembered else [rock]
    )
    strategy.decide(turn)
    goal = strategy.memory.goal_for(worker["id"])
    assert goal is not None and goal.position != rock
    assert goal.position not in strategy.memory.obstacles
    assert settled_positions(turn)[UUID(worker["id"])] != rock


def test_invalid_other_worker_goal_does_not_reserve_vision():
    worker = unit(300, "WORKER", position=(5, 5))
    other = unit(301, "WORKER", position=(6, 5))
    rock = (8, 5)
    targets = []
    for obsolete in (False, True):
        strategy = strategy_for([])
        strategy.memory.obstacles.add(rock)
        if obsolete:
            strategy.memory.set_goal(
                other["id"], UnitGoal(rock, 90, WORKER_PATROL_PURPOSE, (6, 5))
            )
        turn = make_turn(objects=[core(position=HOME), worker, other])
        targets.append(
            strategy._worker_block_patrol_goal(turn.workers[0], context_for(turn))
        )
    assert targets[0] == targets[1]
