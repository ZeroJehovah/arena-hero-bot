"""Shared planning data must follow real roster and terrain changes."""

from uuid import UUID

import pytest
from arena_hero import Direction

from arena_hero_bot.exploration import ExplorationRoute
from arena_hero_bot.geometry import line_of_fire, next_step
from arena_hero_bot.memory import ExpeditionSquad, WorldMemory
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, object_id, unit

HOME = (15, 15)


def _roster():
    return [
        unit(2, "RANGER", position=(15, 16)),
        unit(3, "RANGER", position=(18, 15)),
        unit(4, "RANGER", position=(18, 18)),
        unit(5, "RANGER", position=(19, 18)),
    ]


def _strategy():
    return AggressiveStrategy(
        WorldMemory(
            core_home_position=HOME,
            core_layout_version=CORE_LAYOUT_VERSION,
            campaign_last_core_id=object_id(1),
            unit_roles={
                object_id(2): "defense",
                object_id(3): "defense",
                object_id(4): "patrol-1",
                object_id(5): "patrol-1",
            },
            unit_roles_initialized=True,
        ),
        StrategyConfig(campaign_mode=True, expedition_mode=True),
    )


def test_unobserved_rock_overlay_does_not_change_durable_terrain():
    memory = WorldMemory(obstacles={(8, 8)})
    strategy = AggressiveStrategy(memory)
    turn = make_turn(objects=[core()], obstacles=[(1, 0)])

    rocks = strategy._rock_map(turn)

    assert rocks == {(8, 8), (1, 0)}
    assert memory.obstacles == {(8, 8)}
    assert not line_of_fire((0, 0), (2, 0), rocks)
    assert next_step((0, 0), (2, 0), blocked=rocks) is not Direction.RIGHT


def test_observing_rocks_after_an_earlier_query_refreshes_the_same_turn():
    memory = WorldMemory(obstacles={(8, 8)})
    strategy = AggressiveStrategy(memory)
    turn = make_turn(objects=[core()], obstacles=[(1, 0)])
    strategy._rock_map(turn)

    memory.observe(turn)
    memory.obstacles.add((0, 1))
    rocks = strategy._rock_map(turn)

    assert rocks == {(8, 8), (1, 0), (0, 1)}
    assert not line_of_fire((0, 0), (0, 2), rocks)


def test_replacing_the_durable_map_with_equal_size_drops_old_rocks():
    strategy = AggressiveStrategy(WorldMemory(obstacles={(1, 0)}))
    turn = make_turn(objects=[core()])
    assert not line_of_fire((0, 0), (2, 0), strategy._rock_map(turn))

    strategy.memory.obstacles = {(0, 1)}
    rocks = strategy._rock_map(turn)

    assert line_of_fire((0, 0), (2, 0), rocks)
    assert not line_of_fire((0, 0), (0, 2), rocks)


def test_different_observations_in_the_same_tick_do_not_share_a_rock_overlay():
    strategy = AggressiveStrategy(WorldMemory())
    first = make_turn(tick=100, objects=[core()], obstacles=[(1, 0)])
    second = make_turn(tick=100, objects=[core()], obstacles=[(0, 1)])
    strategy._rock_map(first)

    rocks = strategy._rock_map(second)

    assert rocks == {(0, 1)}
    assert line_of_fire((0, 0), (2, 0), rocks)
    assert strategy.memory.obstacles == set()


@pytest.mark.parametrize("enemy_type", ["RANGER", "VANGUARD", "WORKER"])
def test_complete_planning_keeps_resources_enemies_and_reservations_out_of_rocks(
    enemy_type,
):
    strategy = _strategy()
    strategy.memory.obstacles.add((15, 14))
    turn = make_turn(
        resources=0,
        objects=[
            core(position=HOME),
            *_roster(),
            unit(10, "WORKER", position=HOME, cargo=1),
            unit(900, enemy_type, controlled=False, position=(22, 15)),
        ],
        resource_cells=[(16, 16)],
        obstacles=[(14, 15)],
    )

    report = strategy.decide(turn)

    assert strategy.memory.obstacles == {(15, 14), (14, 15)}
    assert all("deadline" not in decision.reason for decision in report.decisions)
    assert set(turn.plan.unit_actions) == {member.id for member in turn.units}


def test_patrol_query_follows_role_transfers_within_the_same_turn():
    strategy = _strategy()
    turn = make_turn(objects=[core(position=HOME), *_roster()])
    assert strategy._patrol_ids(turn) == {UUID(object_id(4)), UUID(object_id(5))}
    original = strategy.memory.unit_roles.copy()

    strategy.memory.unit_roles[object_id(2)] = "patrol-1"
    strategy.memory.unit_roles[object_id(4)] = "defense"

    assert strategy._patrol_ids(turn) == {UUID(object_id(2)), UUID(object_id(5))}
    assert strategy.memory.unit_roles[object_id(3)] == original[object_id(3)]
    assert strategy.memory.unit_roles[object_id(5)] == original[object_id(5)]


def test_patrol_query_reinitializes_roles_after_a_legacy_reset():
    strategy = _strategy()
    turn = make_turn(objects=[core(position=HOME), *_roster()])
    previous = strategy._patrol_ids(turn)

    strategy.memory.unit_roles_initialized = False

    assert strategy._patrol_ids(turn) == previous
    assert strategy.memory.unit_roles_initialized


def test_replacement_unit_in_the_same_tick_cannot_inherit_a_dead_patrol_uuid():
    strategy = _strategy()
    first = make_turn(tick=100, objects=[core(position=HOME), *_roster()])
    strategy._patrol_ids(first)
    second = make_turn(
        tick=100,
        objects=[
            core(position=HOME),
            *[obj for obj in _roster() if obj["id"] != object_id(4)],
            unit(6, "RANGER", position=(18, 18)),
        ],
    )

    assert strategy._patrol_ids(second) == {UUID(object_id(5)), UUID(object_id(6))}
    assert object_id(4) not in strategy.memory.unit_roles
    assert strategy.memory.unit_roles[object_id(2)] == "defense"
    assert strategy.memory.unit_roles[object_id(3)] == "defense"


def test_patrol_query_rechecks_squad_ownership_before_roles_have_been_saved():
    strategy = _strategy()
    turn = make_turn(objects=[core(position=HOME), *_roster()])
    strategy._patrol_ids(turn)
    members = frozenset(UUID(object_id(number)) for number in (4, 5))
    strategy._expedition_squads = [(members, (1, 0))]

    patrol = strategy._patrol_ids(turn)

    assert patrol == {UUID(object_id(3))}
    assert patrol.isdisjoint(members)
    assert all(
        strategy.memory.unit_roles[str(mid)] == "expedition-1" for mid in members
    )


def test_core_replacement_cannot_reuse_previous_patrol_members():
    strategy = _strategy()
    first = make_turn(tick=100, objects=[core(position=HOME), *_roster()])
    strategy._patrol_ids(first)
    new_ids = {UUID(object_id(number)) for number in range(12, 16)}
    second = make_turn(
        tick=100,
        objects=[
            core(99, position=(47, 47)),
            *[
                unit(number, "RANGER", position=(47 + number - 12, 48))
                for number in range(12, 16)
            ],
        ],
    )

    strategy.decide(second)

    assert strategy.memory.campaign_last_core_id == object_id(99)
    assert set(strategy.memory.unit_roles) == {str(uid) for uid in new_ids}
    assert len(strategy._patrol_ids(second)) == 2
    assert strategy._patrol_ids(second) <= new_ids


def test_empty_expedition_navigation_still_observes_and_clears_persisted_squads():
    memory = WorldMemory(
        expedition_squads=[
            ExpeditionSquad(serial=1, members=(object_id(2),), bearing=(1, 0))
        ]
    )
    strategy = AggressiveStrategy(memory, StrategyConfig(expedition_mode=True))
    strategy._expedition_squads.clear()
    turn = make_turn(objects=[core(), unit(2, "RANGER", position=(1, 0))])

    strategy._refresh_expedition_navigation(turn)

    assert memory.exploration.last_tick == turn.tick
    assert memory.exploration.seen((1, 1))
    assert memory.expedition_squads == []


def test_new_squad_in_the_same_turn_receives_navigation_after_an_empty_query():
    strategy = AggressiveStrategy(WorldMemory(), StrategyConfig(expedition_mode=True))
    turn = make_turn(
        objects=[
            core(position=(-30, -30)),
            *[
                unit(number, "RANGER", position=(number - 2, 0))
                for number in range(2, 6)
            ],
        ]
    )
    strategy._refresh_expedition_navigation(turn)
    members = frozenset(member.id for member in turn.rangers)
    strategy._expedition_squads = [(members, (1, 0))]
    strategy._expedition_routes[members] = ExplorationRoute()

    strategy._refresh_expedition_navigation(turn)

    route = strategy.memory.expedition_squads[0].exploration_route
    assert route.goal is not None
    assert route.waypoint is not None
    assert route.expected_gain > 0


def test_empty_membership_query_does_not_hide_later_or_surviving_members():
    strategy = AggressiveStrategy(WorldMemory(), StrategyConfig(expedition_mode=True))
    first = make_turn(objects=[core(), unit(2, "RANGER")])
    assert not strategy._expedition_member_ids(first)
    members = frozenset(UUID(object_id(number)) for number in (2, 3))
    strategy._expedition_squads = [(members, (1, 0))]

    assert strategy._expedition_member_ids(first) == {UUID(object_id(2))}
    second = make_turn(tick=101, objects=[core(), unit(3, "RANGER")])
    assert strategy._expedition_member_ids(second) == {UUID(object_id(3))}


@pytest.mark.parametrize("enemy_type", ["RANGER", "VANGUARD", "WORKER"])
def test_no_enemy_core_does_not_start_or_change_a_hunt(enemy_type, monkeypatch):
    strategy = _strategy()
    turn = make_turn(
        objects=[
            core(position=HOME),
            *_roster(),
            unit(900, enemy_type, controlled=False, position=(22, 15)),
        ]
    )
    original_roles = strategy.memory.unit_roles.copy()

    def unused_reserve_search(*_args):
        pytest.fail("no enemy Core needs a hunter reserve search")

    monkeypatch.setattr(strategy, "_core_hunt_has_reserves", unused_reserve_search)

    assert strategy._core_hunt_candidate(turn) is None
    assert strategy.memory.core_hunt is None
    assert strategy.memory.unit_roles == original_roles
