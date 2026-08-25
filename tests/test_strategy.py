"""Representative aggressive tactic scenarios."""

from itertools import pairwise

from arena_hero import Direction, SpawnAction, UnitType, unit_cost

from arena_hero_bot.geometry import manhattan
from arena_hero_bot.memory import UnitGoal, WorldMemory
from arena_hero_bot.strategy import (
    AggressiveStrategy,
    StrategyConfig,
    _clear_manhattan_path,
    _resource_patrol_offsets,
)

from .factories import core, make_turn, object_id, unit


def decide(turn, memory=None, config=None):
    strategy = AggressiveStrategy(memory or WorldMemory(), config)
    return strategy.decide(turn)


def test_ranger_prioritizes_and_shoots_visible_core() -> None:
    turn = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 1)),
            unit(3, "WORKER", controlled=False, position=(1, 1)),
            core(
                4,
                controlled=False,
                owner_username="rival",
                position=(0, 3),
                hp=3,
                shield=0,
            ),
        ],
    )
    report = decide(turn)

    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type == "SHOOT"
    assert action.expected_cell == (0, 3)
    assert any(item.action == "SHOOT" for item in report.decisions)


def test_ranger_does_not_shoot_through_remembered_obstacle() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 0)),
            core(
                3,
                controlled=False,
                owner_username="rival",
                position=(0, 3),
            ),
        ],
        obstacles=[(0, 2)],
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.rangers[0].id].type == "MOVE"


def test_ranger_tracks_target_and_leads_one_cell() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(
        tick=100,
        objects=[
            core(position=(100, 100)),
            unit(2, "RANGER", position=(0, 1)),
            unit(3, "VANGUARD", controlled=False, position=(0, 6)),
        ],
    )
    strategy.decide(first)

    second = make_turn(
        tick=101,
        objects=[
            core(position=(100, 100)),
            unit(2, "RANGER", position=(0, 1)),
            unit(3, "VANGUARD", controlled=False, position=(0, 5)),
        ],
    )
    strategy.decide(second)

    third = make_turn(
        tick=102,
        objects=[
            core(position=(100, 100)),
            unit(2, "RANGER", position=(0, 1)),
            unit(3, "VANGUARD", controlled=False, position=(0, 4)),
        ],
    )
    strategy.decide(third)

    action = third.plan.unit_actions[third.rangers[0].id]
    assert action.type == "SHOOT"
    assert action.target_id == third.visible_enemies[0].id
    assert action.expected_cell == (0, 3)


def test_emergency_focuses_fire_and_bursts_combat_production() -> None:
    turn = make_turn(
        resources=90,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 1)),
            unit(3, "RANGER", position=(0, 2)),
            unit(4, "VANGUARD", position=(-2, 1)),
            unit(5, "VANGUARD", position=(2, 1)),
            unit(6, "VANGUARD", controlled=False, position=(0, 4), hp=1),
            unit(7, "RANGER", controlled=False, position=(1, 4)),
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(target_workers=0, max_population=None),
    )

    ranger_actions = [
        turn.plan.unit_actions[unit.id]
        for unit in turn.rangers
        if unit.id in turn.plan.unit_actions
    ]
    shoot_actions = [action for action in ranger_actions if action.type == "SHOOT"]
    assert len(shoot_actions) == len(turn.rangers)
    assert len({action.target_id for action in shoot_actions}) == 1
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert any(item.action == "SPAWN" for item in report.decisions)


def test_vanguard_sweeps_every_adjacent_hostile() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "VANGUARD", position=(1, 0)),
            unit(3, "WORKER", controlled=False, position=(2, 0)),
            unit(4, "RANGER", controlled=False, position=(2, 0)),
        ]
    )
    decide(turn)
    action = turn.plan.unit_actions[turn.vanguards[0].id]
    assert action.type == "SWEEP"
    assert action.direction is Direction.RIGHT


def test_vanguards_intercept_worker_near_core_on_distinct_escape_cells() -> None:
    turn = make_turn(
        objects=[
            core(position=(0, 0)),
            unit(2, "VANGUARD", position=(-3, 0)),
            unit(3, "VANGUARD", position=(3, 0)),
            unit(4, "WORKER", controlled=False, position=(1, 0)),
        ]
    )

    report = decide(turn)

    destinations = []
    for vanguard in turn.vanguards:
        action = turn.plan.unit_actions[vanguard.id]
        assert action.type == "MOVE"
        decision = next(
            item for item in report.decisions if item.actor_id == str(vanguard.id)
        )
        destinations.append(decision.target)
    assert len(destinations) == len(set(destinations))
    assert all(
        destination in {(0, 0), (2, 0), (1, -1), (1, 1)} for destination in destinations
    )


def test_vanguards_spread_across_focus_target_escape_routes() -> None:
    turn = make_turn(
        objects=[
            core(position=(100, 100)),
            unit(2, "VANGUARD", position=(-3, 0)),
            unit(3, "VANGUARD", position=(3, 0)),
            unit(4, "VANGUARD", position=(0, -3)),
            unit(5, "VANGUARD", position=(0, 3)),
            unit(6, "RANGER", controlled=False, position=(0, 0)),
            unit(7, "VANGUARD", controlled=False, position=(0, 5)),
        ],
    )

    decide(turn)

    goals = []
    for vanguard in turn.vanguards:
        action = turn.plan.unit_actions[vanguard.id]
        if action.type == "MOVE":
            dx, dy = action.direction.delta
            goals.append((vanguard.position[0] + dx, vanguard.position[1] + dy))
    assert len(goals) == len(set(goals))


def test_outmatched_core_repairs_instead_of_an_unwinnable_migration() -> None:
    """A Core cannot outrun pursuit it is already in contact with.

    Migration costs four Ticks per cell where Units cover one cell per Tick,
    and a ``MOVING`` Core can neither ``HEAL`` nor ``REPAIR_SHIELD``.  With a
    single guard against five adjacent hostiles the retreat cannot be
    screened, so holding the cell and restoring the shield strictly dominates.
    """

    turn = make_turn(
        resources=10,
        objects=[
            core(position=(0, 0), shield=4),
            unit(2, "VANGUARD", position=(0, 5)),
            unit(3, "RANGER", controlled=False, position=(0, 1)),
            unit(4, "RANGER", controlled=False, position=(1, 1)),
            unit(5, "RANGER", controlled=False, position=(-1, 1)),
            unit(6, "VANGUARD", controlled=False, position=(1, 0)),
            unit(7, "VANGUARD", controlled=False, position=(-1, 0)),
        ],
    )

    decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "REPAIR_SHIELD"


def test_guardless_low_capacity_core_evacuates_before_first_hit() -> None:
    turn = make_turn(
        resources=5,
        objects=[
            core(),
            unit(2, "WORKER", position=(2, 0)),
            unit(3, "RANGER", controlled=False, position=(5, 0)),
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(target_workers=12, max_population=None),
    )

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert any(
        "evacuate Core from overwhelming enemy assault" in item.reason
        or "before the screen breaks" in item.reason
        for item in report.decisions
    )


def test_unprotected_core_evacuates_on_nearby_pre_evade_enemy() -> None:
    turn = make_turn(
        resources=5,
        objects=[
            core(),
            unit(2, "WORKER", position=(2, 0)),
            unit(3, "VANGUARD", position=(20, 0)),
            unit(4, "RANGER", controlled=False, position=(0, 9)),
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(target_workers=12, max_population=None),
    )

    assert report.threat_level == "PRE_EVADE"
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"


def test_core_escape_lane_is_reserved_before_combat_moves() -> None:
    turn = make_turn(
        resources=20,
        objects=[
            core(),
            unit(2, "VANGUARD", position=(0, 4)),
            unit(3, "VANGUARD", position=(4, 0)),
            unit(4, "VANGUARD", position=(-4, 0)),
            unit(5, "WORKER", position=(0, 6)),
            unit(6, "VANGUARD", controlled=False, position=(0, 2)),
            unit(7, "RANGER", controlled=False, position=(1, 2)),
            unit(8, "VANGUARD", controlled=False, position=(-1, 1)),
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(target_workers=0, max_population=None),
    )

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.core is not None
    escape_cell = (
        turn.core.position[0] + turn.plan.core_action.direction.delta[0],
        turn.core.position[1] + turn.plan.core_action.direction.delta[1],
    )
    unit_destinations = {
        (
            unit_view.position[0] + action.direction.delta[0],
            unit_view.position[1] + action.direction.delta[1],
        )
        for unit_view in turn.units
        if (action := turn.plan.unit_actions.get(unit_view.id)) is not None
        and action.type == "MOVE"
    }
    assert escape_cell not in unit_destinations
    assert any("before the screen breaks" in item.reason for item in report.decisions)


def test_assault_limits_vanguard_strike_team_and_assigns_unique_screen_slots() -> None:
    turn = make_turn(
        resources=20,
        objects=[
            core(),
            *[
                unit(number, "VANGUARD", position=(10, number))
                for number in range(2, 8)
            ],
            unit(20, "RANGER", controlled=False, position=(0, 5)),
            unit(21, "RANGER", controlled=False, position=(1, 5)),
            unit(22, "VANGUARD", controlled=False, position=(-1, 5)),
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(target_workers=0, max_population=None),
    )

    screen = [
        item
        for item in report.decisions
        if item.reason == "hold the Core screen while the strike team attacks"
    ]
    assert len(screen) == 3
    assert len({item.target for item in screen}) == len(screen)


def test_ranger_fires_at_close_focus_target_during_core_assault() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 0)),
            unit(3, "VANGUARD", controlled=False, position=(0, 2)),
            unit(4, "RANGER", controlled=False, position=(0, 2)),
            unit(5, "VANGUARD", controlled=False, position=(1, 2)),
        ],
    )

    decide(turn)

    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type == "SHOOT"
    assert action.expected_cell == (0, 2)


def test_ranger_uses_supporting_fire_when_focus_lane_is_not_legal() -> None:
    turn = make_turn(
        objects=[
            core(position=(100, 100)),
            unit(2, "RANGER", position=(0, 0)),
            unit(3, "RANGER", position=(2, 0)),
            unit(4, "RANGER", controlled=False, position=(0, 3)),
            unit(5, "VANGUARD", controlled=False, position=(2, 3)),
        ],
    )

    report = decide(turn)

    first = turn.plan.unit_actions[turn.rangers[0].id]
    second = turn.plan.unit_actions[turn.rangers[1].id]
    assert first.type == "SHOOT"
    assert second.type == "SHOOT"
    assert first.target_id != second.target_id
    assert any("supporting fire" in item.reason for item in report.decisions)


def test_high_pressure_forms_a_screen_before_core_escape_is_needed() -> None:
    turn = make_turn(
        objects=[
            core(),
            *[
                unit(number, "VANGUARD", position=(8, number - 2))
                for number in range(2, 8)
            ],
            *[
                unit(number, "VANGUARD", controlled=False, position=position)
                for number, position in enumerate(
                    ((-2, 10), (-1, 10), (0, 10), (1, 10), (2, 10), (3, 10)),
                    start=20,
                )
            ],
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(target_workers=0, max_population=None),
    )

    screen = [
        item
        for item in report.decisions
        if item.reason == "hold the Core screen while the strike team attacks"
    ]
    assert len(screen) == 3
    assert turn.plan.core_action is None


def test_worker_harvests_then_returns_cargo() -> None:
    harvesting = make_turn(
        objects=[core(), unit(2, "WORKER", position=(1, 0))],
        resource_cells=[(1, 0)],
    )
    decide(harvesting)
    assert harvesting.plan.unit_actions[harvesting.workers[0].id].type == "HARVEST"

    returning = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(1, 0), cargo=1)],
    )
    decide(returning)
    action = returning.plan.unit_actions[returning.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT


def test_worker_keeps_recent_resource_goal_across_missing_observation() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)

    first = make_turn(
        tick=100,
        objects=[core(position=(10, 10)), unit(2, "WORKER", position=(0, 0))],
        resource_cells=[(0, 2)],
    )
    strategy.decide(first)
    assert first.plan.unit_actions[first.workers[0].id].type == "MOVE"

    second = make_turn(
        tick=101,
        objects=[core(position=(10, 10)), unit(2, "WORKER", position=(0, 1))],
    )
    report = strategy.decide(second)
    action = second.plan.unit_actions[second.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.DOWN
    assert any(
        item.reason == "continue toward recently seen resource"
        for item in report.decisions
    )

    third = make_turn(
        tick=102,
        objects=[core(position=(10, 10)), unit(2, "WORKER", position=(0, 2))],
        resource_cells=[(0, 2)],
    )
    strategy.decide(third)
    assert third.plan.unit_actions[third.workers[0].id].type == "HARVEST"


def test_worker_deposits_at_core_and_waits_when_storage_full() -> None:
    deposit = make_turn(
        resources=5,
        objects=[core(), unit(2, "WORKER", position=(0, 0), cargo=1)],
    )
    decide(deposit)
    assert deposit.plan.unit_actions[deposit.workers[0].id].type == "DEPOSIT"

    full = make_turn(
        resources=10,
        objects=[core(), unit(2, "WORKER", position=(0, 0), cargo=1)],
    )
    report = decide(full)
    assert full.workers[0].id not in full.plan.unit_actions
    assert any(item.reason == "Core storage is full" for item in report.decisions)


def test_only_one_worker_reserves_the_core_cell() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0), cargo=1),
            unit(3, "WORKER", position=(0, 1), cargo=1),
        ]
    )
    report = decide(turn)
    moving_home = [
        worker
        for worker in turn.workers
        if turn.plan.unit_actions.get(worker.id) is not None
        and turn.plan.unit_actions[worker.id].type == "MOVE"
    ]
    assert len(moving_home) == 1
    assert any(
        "Core cell is not currently reachable" in item.reason
        for item in report.decisions
    )


def test_empty_worker_vacates_core_before_returning_worker_moves() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0), cargo=1),
            unit(3, "WORKER", position=(0, 0)),
        ]
    )
    decide(turn)

    returning = turn.workers[0]
    action = turn.plan.unit_actions[returning.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT


def test_core_does_not_spawn_into_worker_reserved_cell() -> None:
    turn = make_turn(
        resources=10,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0), cargo=1),
            unit(3, "WORKER", position=(0, 0)),
        ],
    )
    decide(turn)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert turn.plan.core_action is None


def test_worker_retreats_from_nearby_visible_enemy() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(2, 0)),
            unit(3, "VANGUARD", controlled=False, position=(3, 0)),
        ]
    )
    report = decide(turn)

    worker = turn.workers[0]
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )


def test_critical_worker_returns_to_core_before_retreating_from_threat() -> None:
    turn = make_turn(
        objects=[
            core(position=(1, 0)),
            unit(2, "WORKER", position=(0, 0), hp=1),
            unit(3, "VANGUARD", controlled=False, position=(0, 5)),
        ]
    )

    report = decide(turn)

    worker_decision = next(
        item for item in report.decisions if item.actor_id == str(turn.workers[0].id)
    )
    assert worker_decision.reason == "return critical unit to Core for healing"


def test_worker_can_retreat_into_core_cell_when_threatened() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "VANGUARD", controlled=False, position=(2, 0)),
        ]
    )
    decide(turn)

    worker = turn.workers[0]
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT


def test_worker_retreat_prefers_distance_from_enemy_over_shortest_path() -> None:
    worker_id = str(object_id(2))
    memory = WorldMemory(
        obstacles={
            (-423, 824),
            (-422, 823),
            (-421, 822),
            (-418, 825),
            (-423, 828),
            (-421, 828),
            (-420, 828),
            (-418, 828),
            (-419, 830),
            (-422, 829),
            (-421, 830),
        },
        position_history={
            worker_id: [(-422, 824), (-421, 824), (-420, 824), (-419, 824)]
        },
    )
    turn = make_turn(
        objects=[
            core(position=(-521, 882)),
            unit(2, "WORKER", position=(-418, 824)),
            unit(3, "WORKER", controlled=False, position=(-415, 824)),
            unit(4, "RANGER", controlled=False, position=(-415, 824)),
        ]
    )
    decide(turn, memory=memory)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT


def test_cargo_worker_does_not_oscillate_around_enemy() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 2), cargo=1),
            unit(3, "RANGER", controlled=False, position=(-1, -3)),
        ],
    )
    strategy.decide(first)
    first_action = first.plan.unit_actions[first.workers[0].id]
    assert first_action.type == "MOVE"
    assert first_action.direction is Direction.DOWN

    second = make_turn(
        tick=101,
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 3), cargo=1),
            unit(3, "RANGER", controlled=False, position=(-1, -3)),
        ],
    )
    strategy.decide(second)
    second_action = second.plan.unit_actions[second.workers[0].id]
    assert second_action.type == "MOVE"
    assert second_action.direction is not Direction.UP

    third = make_turn(
        tick=102,
        objects=[
            core(),
            unit(2, "WORKER", position=(-1, 3), cargo=1),
            unit(3, "RANGER", controlled=False, position=(-1, -3)),
        ],
    )
    report = strategy.decide(third)
    action = third.plan.unit_actions[third.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is not Direction.RIGHT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )

    looping = make_turn(
        tick=103,
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 3), cargo=1),
            unit(3, "RANGER", controlled=False, position=(-1, -3)),
        ],
    )
    strategy.decide(looping)
    loop_action = looping.plan.unit_actions[looping.workers[0].id]
    assert loop_action.type == "MOVE"
    assert loop_action.direction is not Direction.LEFT


def test_worker_does_not_retreat_from_noncombat_enemy_worker() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(5, 0)),
            unit(3, "WORKER", controlled=False, position=(6, 0)),
        ]
    )
    report = decide(turn)

    assert not any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )


def test_worker_retreats_from_visible_enemy_core() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(5, 0)),
            core(
                3,
                controlled=False,
                owner_username="rival",
                position=(6, 0),
            ),
        ]
    )
    report = decide(turn)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )


def test_worker_retreats_from_recently_seen_enemy_core() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(5, 0)),
            core(
                3,
                controlled=False,
                owner_username="rival",
                position=(8, 0),
            ),
        ],
    )
    strategy.decide(first)

    second = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(5, 0))],
    )
    report = strategy.decide(second)

    action = second.plan.unit_actions[second.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )

    still_remembered = make_turn(
        tick=125,
        objects=[core(), unit(2, "WORKER", position=(5, 0))],
    )
    remembered_report = strategy.decide(still_remembered)
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in remembered_report.decisions
    )

    long_term = make_turn(
        tick=261,
        objects=[core(), unit(2, "WORKER", position=(5, 0))],
    )
    long_term_report = strategy.decide(long_term)
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in long_term_report.decisions
    )


def test_cargo_worker_retreats_from_visible_combat_enemy() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(2, 0), cargo=1),
            unit(3, "RANGER", controlled=False, position=(3, 0)),
        ]
    )
    report = decide(turn)

    worker = turn.workers[0]
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )


def test_cargo_worker_returns_through_contested_noncombat_area() -> None:
    memory = WorldMemory(contested_positions={(1, 0): 100})
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(2, 0), cargo=1),
            unit(3, "WORKER", controlled=False, position=(1, 0)),
        ]
    )

    report = decide(turn, memory=memory)

    worker = turn.workers[0]
    action = turn.plan.unit_actions[worker.id]
    assert action.type == "MOVE"
    assert any(
        item.reason == "return carried resources to Core" for item in report.decisions
    )
    assert not any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )


def test_worker_retreats_from_recently_seen_combat_enemy() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 1), cargo=1),
            unit(3, "VANGUARD", controlled=False, position=(8, 1)),
        ],
    )
    strategy.decide(first)

    second = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(5, 1), cargo=1)],
    )
    report = strategy.decide(second)

    action = second.plan.unit_actions[second.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is not Direction.RIGHT
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in report.decisions
    )

    expired = make_turn(
        tick=125,
        objects=[core(), unit(2, "WORKER", position=(5, 1), cargo=1)],
    )
    expired_report = strategy.decide(expired)
    assert not any(
        item.reason == "retreat from visible enemy pressure"
        for item in expired_report.decisions
    )


def test_worker_harvests_immediately_after_enemy_destruction_event() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "VANGUARD", controlled=False, position=(2, 0)),
        ],
        resource_cells=[(1, 0)],
    )
    first_report = strategy.decide(first)
    assert any(
        item.reason == "retreat from visible enemy pressure"
        for item in first_report.decisions
    )

    second = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(1, 0))],
        resource_cells=[(1, 0)],
        events=[
            {
                "event_id": object_id(99),
                "tick": 101,
                "event_type": "DESTRUCTION_PARTICIPATION",
                "reason_code": "UNIT",
                "target_id": object_id(3),
                "position": [2, 0],
            }
        ],
    )

    report = strategy.decide(second)

    assert second.plan.unit_actions[second.workers[0].id].type == "HARVEST"
    assert any(item.action == "HARVEST" for item in report.decisions)


def test_only_lowest_uuid_worker_harvests_contested_resource() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(1, 0)),
        ],
        resource_cells=[(1, 0)],
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "HARVEST"
    assert turn.plan.unit_actions.get(turn.workers[1].id) is None or (
        turn.plan.unit_actions[turn.workers[1].id].type != "HARVEST"
    )


def test_nearest_worker_is_assigned_visible_resource() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 5)),
            unit(3, "WORKER", position=(9, 0)),
        ],
        resource_cells=[(10, 0)],
    )
    report = decide(turn)

    near_worker = turn.workers[1]
    action = turn.plan.unit_actions[near_worker.id]
    assert action.type == "MOVE"
    assert action.direction is Direction.RIGHT
    claims = [
        item
        for item in report.decisions
        if item.reason == "claim nearest unassigned visible resource"
    ]
    assert [(item.actor_id, item.target) for item in claims] == [
        (str(near_worker.id), (10, 0))
    ]


def test_worker_vacates_core_and_core_spawns_second_worker() -> None:
    turn = make_turn(
        resources=5,
        objects=[core(), unit(2, "WORKER", position=(0, 0))],
        resource_cells=[(1, 0)],
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "MOVE"
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_core_builds_aggressive_composition() -> None:
    vanguard_turn = make_turn(
        resources=10,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(-1, 0)),
        ],
    )
    decide(vanguard_turn)
    assert isinstance(vanguard_turn.plan.core_action, SpawnAction)
    assert vanguard_turn.plan.core_action.unit_type is UnitType.VANGUARD

    ranger_turn = make_turn(
        resources=12,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(-1, 0)),
            unit(4, "VANGUARD", position=(0, 1)),
        ],
    )
    decide(ranger_turn)
    assert isinstance(ranger_turn.plan.core_action, SpawnAction)
    assert ranger_turn.plan.core_action.unit_type is UnitType.RANGER


def test_core_prioritizes_survival_under_pressure() -> None:
    heal_turn = make_turn(resources=10, objects=[core(hp=2)])
    decide(heal_turn)
    assert heal_turn.plan.core_action is not None
    assert heal_turn.plan.core_action.type == "HEAL"

    shield_turn = make_turn(
        resources=10,
        objects=[
            core(shield=1),
            unit(2, "VANGUARD", controlled=False, position=(2, 0)),
        ],
    )
    decide(shield_turn)
    assert shield_turn.plan.core_action is not None
    assert shield_turn.plan.core_action.type == "REPAIR_SHIELD"

    early_shield_turn = make_turn(
        resources=10,
        objects=[
            core(shield=4),
            unit(2, "VANGUARD", controlled=False, position=(2, 0)),
        ],
    )
    early_report = decide(early_shield_turn)
    assert early_shield_turn.plan.core_action is not None
    assert early_shield_turn.plan.core_action.type == "REPAIR_SHIELD"
    assert any(
        item.reason == "repair damaged shield before the next enemy strike"
        for item in early_report.decisions
    )


def test_same_tick_deposit_can_fund_core_healing() -> None:
    turn = make_turn(
        resources=0,
        objects=[core(hp=2), unit(2, "WORKER", position=(0, 0), cargo=3)],
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.workers[0].id].type == "DEPOSIT"
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "HEAL"


def test_critical_ranger_heals_at_stationary_core() -> None:
    turn = make_turn(
        resources=4,
        objects=[core(), unit(2, "RANGER", position=(0, 0), hp=1)],
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.rangers[0].id].type == "HEAL"


def test_critical_ranger_takes_legal_shot_before_withdrawing() -> None:
    turn = make_turn(
        resources=4,
        objects=[
            core(position=(100, 100)),
            unit(2, "RANGER", position=(0, 0), hp=1),
            unit(3, "VANGUARD", controlled=False, position=(0, 3)),
        ],
    )

    decide(turn)

    assert turn.plan.unit_actions[turn.rangers[0].id].type == "SHOOT"


def test_ranger_disengages_to_max_range_against_enemy_unit() -> None:
    turn = make_turn(
        objects=[
            core(position=(100, 100)),
            unit(2, "RANGER", position=(0, 1)),
            unit(3, "RANGER", controlled=False, position=(0, 3)),
        ],
    )

    report = decide(turn)

    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type == "MOVE"
    assert any("max-range firing line" in item.reason for item in report.decisions)


def test_critical_vanguard_fights_before_withdrawing_during_core_assault() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "VANGUARD", position=(2, 0), hp=2),
            unit(3, "RANGER", controlled=False, position=(3, 0)),
        ],
    )

    report = decide(turn)

    action = turn.plan.unit_actions[turn.vanguards[0].id]
    assert action.type == "SWEEP"
    assert any(
        item.action == "SWEEP" and "adjacent hostile" in item.reason
        for item in report.decisions
    )


def test_core_and_unit_pick_up_ground_beacon() -> None:
    unit_turn = make_turn(
        objects=[core(), unit(2, "VANGUARD", position=(1, 0))],
        beacon={"position": [1, 0], "status": "GROUND"},
    )
    decide(unit_turn)
    assert unit_turn.plan.unit_actions[unit_turn.vanguards[0].id].type == (
        "PICKUP_BEACON"
    )

    core_turn = make_turn(
        objects=[core()], beacon={"position": [0, 0], "status": "GROUND"}
    )
    decide(core_turn)
    assert core_turn.plan.core_action is not None
    assert core_turn.plan.core_action.type == "PICKUP_BEACON"


def test_unit_beacon_claim_prevents_duplicate_core_pickup() -> None:
    turn = make_turn(
        resources=20,
        objects=[core(), unit(2, "VANGUARD", position=(0, 0))],
        beacon={"position": [0, 0], "status": "GROUND"},
    )
    decide(turn)
    assert turn.plan.unit_actions[turn.vanguards[0].id].type == "PICKUP_BEACON"
    assert turn.plan.core_action is None


def test_combat_unit_hunts_recent_enemy_after_losing_vision() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 1)),
            core(
                3,
                controlled=False,
                owner_username="rival",
                position=(5, 1),
            ),
        ],
    )
    strategy.decide(first)

    second = make_turn(
        tick=101,
        objects=[core(), unit(2, "RANGER", position=(1, 1))],
    )
    report = strategy.decide(second)
    action = second.plan.unit_actions[second.rangers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.RIGHT
    assert any("last seen core" in item.reason for item in report.decisions)


def _remembered_intruder_report(intruder_position, vanguard_positions):
    """Sight one enemy Worker, then replay the same Tick without vision."""

    strategy = AggressiveStrategy(
        WorldMemory(),
        StrategyConfig(target_workers=0, max_population=None),
    )
    guards = [
        unit(10 + index, "VANGUARD", position=position)
        for index, position in enumerate(vanguard_positions)
    ]
    strategy.decide(
        make_turn(
            tick=100,
            objects=[
                core(),
                *guards,
                unit(20, "WORKER", controlled=False, position=intruder_position),
            ],
            resource_cells=[intruder_position],
        )
    )
    recovered = make_turn(
        tick=101,
        objects=[core(), *guards, unit(3, "WORKER", position=(0, 3))],
        resource_cells=[intruder_position],
    )
    return recovered, strategy.decide(recovered)


def test_only_the_escort_hunts_a_remembered_intruder() -> None:
    # A thief inside the economy zone has to be run down even while the cell
    # is dark, but the rest of the roster keeps its perimeter slot instead of
    # stampeding after a single Worker.
    positions = ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2))
    recovered, report = _remembered_intruder_report((1, 1), positions)

    guards = {str(vanguard.id) for vanguard in recovered.vanguards}
    reasons = [item for item in report.decisions if item.actor_id in guards]
    hunting = [item for item in reasons if item.reason == "hunt last seen unit"]
    holding = [item for item in reasons if "perimeter" in item.reason]
    assert 0 < len(hunting) <= 4
    assert holding


def test_a_remembered_worker_outside_the_economy_zone_is_ignored() -> None:
    _, report = _remembered_intruder_report((0, 40), ((0, 1),))

    assert not any("hunt last seen unit" in item.reason for item in report.decisions)


def test_respawning_state_submits_no_invented_actions() -> None:
    turn = make_turn(
        status="RESPAWNING",
        respawn_at_tick=101,
        objects=[],
        resources=0,
    )
    report = decide(turn)
    assert turn.plan.unit_actions == {}
    assert turn.plan.core_action is None
    assert report.decisions == []


def test_moving_core_does_not_queue_illegal_action() -> None:
    turn = make_turn(
        resources=20,
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            )
        ],
    )
    report = decide(turn)
    assert turn.plan.core_action is None
    assert report.decisions[0].reason == "Core migration is already progressing"


def test_idle_core_migrates_toward_resource_rich_center() -> None:
    turn = make_turn(
        objects=[
            core(position=(-10, 10)),
            unit(2, "WORKER", position=(-12, 12)),
            unit(3, "WORKER", position=(-13, 13)),
        ]
    )
    report = decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction in {Direction.UP, Direction.RIGHT}
    assert any(
        item.reason == "advance mobile base toward the resource-rich center"
        for item in report.decisions
    )


def test_core_migration_avoids_immediate_backtrack() -> None:
    core_id = str(object_id(1))
    memory = WorldMemory(
        obstacles={
            (-10, 518),
            (-8, 514),
            (-8, 517),
            (-8, 521),
            (-7, 521),
            (-6, 485),
            (-6, 519),
            (-6, 521),
            (-5, 484),
            (-5, 517),
            (-4, 488),
            (-2, 471),
            (-2, 485),
            (-2, 486),
            (-1, 469),
            (1, 469),
        },
        position_history={core_id: [(-9, 518)]},
    )
    turn = make_turn(
        tick=126628,
        resources=9,
        objects=[
            core(position=(-8, 518)),
            unit(2, "VANGUARD", position=(-5, 485)),
            unit(3, "WORKER", position=(0, 468)),
            unit(4, "WORKER", position=(-9, 514)),
        ],
        obstacles=[],
    )
    decide(turn, memory=memory)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"
    assert turn.plan.core_action.direction is Direction.RIGHT


def test_core_stays_receptive_while_worker_returns_cargo() -> None:
    turn = make_turn(
        objects=[
            core(position=(-10, 10)),
            unit(2, "WORKER", position=(-12, 12), cargo=1),
            unit(3, "WORKER", position=(-13, 13)),
        ]
    )
    decide(turn)

    assert turn.plan.core_action is None


def test_worker_stages_cargo_at_migrating_core_destination() -> None:
    turn = make_turn(
        objects=[
            core(
                position=(0, 0),
                state="MOVING",
                move_direction="RIGHT",
                move_progress=2,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(2, 0), cargo=1),
        ]
    )

    report = decide(turn)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any(
        item.reason == "stage carried resources at migrating Core destination"
        for item in report.decisions
    )


def test_worker_never_deposits_while_core_is_migrating() -> None:
    turn = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                move_progress=3,
                move_required_ticks=4,
                destination=[1, 0],
            ),
            unit(2, "WORKER", position=(0, 0), cargo=1),
        ]
    )

    report = decide(turn)

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.RIGHT
    assert action.type != "DEPOSIT"
    assert any(
        item.reason == "stage carried resources at migrating Core destination"
        for item in report.decisions
    )


def test_population_cap_stops_production() -> None:
    units = [unit(number, "RANGER", position=(number, 2)) for number in range(2, 14)]
    turn = make_turn(resources=100, objects=[core(), *units])
    decide(turn, config=StrategyConfig(max_population=12))
    assert turn.plan.core_action is None


def test_unbounded_population_keeps_expanding_past_legacy_cap() -> None:
    units = [unit(number, "RANGER", position=(number, 2)) for number in range(2, 22)]
    turn = make_turn(resources=120, objects=[core(), *units])

    decide(
        turn,
        config=StrategyConfig(
            target_workers=0,
            max_population=None,
            safety_reserve=10,
        ),
    )

    assert turn.state.population == 20
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_unbounded_population_preserves_emergency_resource_reserve() -> None:
    workers = [
        unit(2, "WORKER", position=(1, 0)),
        unit(3, "WORKER", position=(2, 0)),
    ]
    turn = make_turn(resources=14, objects=[core(), *workers])

    decide(
        turn,
        config=StrategyConfig(
            target_workers=12,
            max_population=None,
            safety_reserve=10,
        ),
    )

    assert turn.plan.core_action is None


def test_unbounded_growth_spends_healthy_low_capacity_surplus() -> None:
    turn = make_turn(
        resources=13,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(2, 0)),
            unit(4, "VANGUARD", position=(3, 0)),
        ],
    )

    decide(
        turn,
        config=StrategyConfig(target_workers=12, max_population=None),
    )

    assert turn.resource_capacity == 15
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_unbounded_growth_builds_ranged_guard_before_more_workers() -> None:
    turn = make_turn(
        resources=12,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(2, 0)),
            unit(4, "WORKER", position=(3, 0)),
            unit(5, "WORKER", position=(4, 0)),
            unit(6, "VANGUARD", position=(5, 0)),
        ],
    )

    decide(
        turn,
        config=StrategyConfig(target_workers=12, max_population=None),
    )

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.RANGER


def test_unbounded_growth_keeps_reserve_for_damaged_low_capacity_core() -> None:
    turn = make_turn(
        resources=13,
        objects=[
            core(hp=4),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(2, 0)),
            unit(4, "VANGUARD", position=(3, 0)),
        ],
    )

    decide(
        turn,
        config=StrategyConfig(target_workers=12, max_population=None),
    )

    assert turn.plan.core_action is None


def test_pre_evade_moves_core_when_guards_match_the_closing_group() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(5, 5)),
            unit(5, "VANGUARD", position=(0, -3)),
            unit(6, "VANGUARD", position=(-3, 0)),
            unit(3, "VANGUARD", controlled=False, position=(0, 3)),
            unit(4, "VANGUARD", controlled=False, position=(1, 2)),
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(target_workers=0, max_population=None),
    )

    assert report.threat_level == "PRE_EVADE"
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"


def test_pre_evade_holds_an_unscreened_core_inside_breakaway_range() -> None:
    turn = make_turn(
        resources=5,
        objects=[
            core(shield=4),
            unit(2, "WORKER", position=(5, 5)),
            unit(3, "VANGUARD", controlled=False, position=(0, 3)),
            unit(4, "VANGUARD", controlled=False, position=(1, 2)),
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(target_workers=0, max_population=None),
    )

    assert report.threat_level == "PRE_EVADE"
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "REPAIR_SHIELD"


def test_unbounded_growth_sends_a_minority_of_combat_units_on_patrol() -> None:
    combat_units = [
        unit(number, "VANGUARD" if number % 2 else "RANGER", position=(number, 1))
        for number in range(2, 10)
    ]
    turn = make_turn(resources=15, objects=[core(), *combat_units])

    report = decide(
        turn,
        config=StrategyConfig(
            target_workers=0,
            max_population=None,
            resource_target=0,
        ),
    )

    offensive = [
        item
        for item in report.decisions
        if item.reason == "search outward for enemy units and Cores"
    ]
    defensive = [
        item
        for item in report.decisions
        if item.reason == "hold a defensive perimeter around the resource Core"
    ]
    assert len(offensive) == 3
    assert len(defensive) == 5
    assert all(item.target != (0, 0) for item in offensive)


def test_unbounded_growth_uses_capacity_stockpile_tiers() -> None:
    strategy = AggressiveStrategy(
        WorldMemory(),
        StrategyConfig(target_workers=0, max_population=None),
    )

    def reserve_for_population(population: int) -> int:
        units = [
            unit(number, "WORKER", position=(number, 2))
            for number in range(2, population + 2)
        ]
        turn = make_turn(objects=[core(), *units])
        return strategy._spawn_safety_reserve(turn)

    assert reserve_for_population(8) == 10  # capacity 40: fast expansion
    assert reserve_for_population(10) == 50  # capacity 50
    assert reserve_for_population(18) == 50  # capacity 90
    assert reserve_for_population(19) == 95  # capacity 95
    assert reserve_for_population(20) == 95  # capacity 100: previous tier
    assert reserve_for_population(21) == 100  # capacity 105: floor still wins
    assert reserve_for_population(29) == 101  # capacity 145: 70%
    assert reserve_for_population(30) == 105  # capacity 150: 70%
    assert reserve_for_population(31) == 108  # capacity 155: 70%
    assert reserve_for_population(39) == 136  # capacity 195: 70%
    # At the growth slowdown threshold the target jumps to the banking tier so
    # income fills the emergency reserve instead of an ever pricier roster.
    assert reserve_for_population(40) == 180  # capacity 200: 90%
    assert reserve_for_population(41) == 184  # capacity 205: 90%


def test_unbounded_growth_crosses_a_stockpile_boundary_without_deadlocking() -> None:
    workers = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 12)]
    turn = make_turn(resources=50, objects=[core(), *workers])

    decide(
        turn,
        config=StrategyConfig(target_workers=12, max_population=None),
    )

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_unbounded_growth_builds_first_guard_at_minimum_respawn_capacity() -> None:
    turn = make_turn(
        resources=10,
        objects=[core(), unit(2, "WORKER", position=(1, 0))],
    )

    decide(
        turn,
        config=StrategyConfig(target_workers=12, max_population=None),
    )

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_unbounded_growth_establishes_guard_before_second_worker() -> None:
    waiting = make_turn(
        resources=5,
        objects=[core(), unit(2, "WORKER", position=(1, 0))],
    )

    decide(
        waiting,
        config=StrategyConfig(target_workers=12, max_population=None),
    )

    assert waiting.plan.core_action is None

    guarded = make_turn(
        resources=10,
        objects=[core(), unit(2, "WORKER", position=(1, 0))],
    )

    decide(
        guarded,
        config=StrategyConfig(target_workers=12, max_population=None),
    )

    assert guarded.plan.core_action is not None
    assert guarded.plan.core_action.type == "SPAWN"
    assert guarded.plan.core_action.unit_type is UnitType.VANGUARD


def test_unbounded_growth_keeps_offensive_minority_when_core_reserve_is_low() -> None:
    combat_units = [
        unit(number, "VANGUARD" if number % 2 else "RANGER", position=(number, 1))
        for number in range(2, 10)
    ]
    turn = make_turn(resources=2, objects=[core(), *combat_units])

    report = decide(
        turn,
        config=StrategyConfig(
            target_workers=0,
            max_population=None,
            resource_target=0,
        ),
    )

    assert (
        sum(
            item.reason == "search outward for enemy units and Cores"
            for item in report.decisions
        )
        == 3
    )
    assert (
        sum(
            item.reason == "hold a defensive perimeter around the resource Core"
            for item in report.decisions
        )
        == 5
    )


def test_unbounded_growth_pauses_remote_patrol_below_large_core_reserve() -> None:
    combat_units = [
        unit(number, "VANGUARD" if number % 2 else "RANGER", position=(number, 1))
        for number in range(2, 10)
    ]
    workers = [unit(number, "WORKER", position=(number, 2)) for number in range(20, 33)]
    turn = make_turn(resources=76, objects=[core(), *combat_units, *workers])

    report = decide(
        turn,
        config=StrategyConfig(
            target_workers=0,
            max_population=None,
            resource_target=0,
        ),
    )

    assert not any(
        item.reason == "search outward for enemy units and Cores"
        for item in report.decisions
    )
    assert sum(
        item.reason == "hold a defensive perimeter around the resource Core"
        for item in report.decisions
    ) == len(combat_units)


def test_defensive_guards_hold_when_patrol_group_sees_a_distant_enemy() -> None:
    combat_units = [
        unit(number, "VANGUARD" if number % 2 else "RANGER", position=(number, 1))
        for number in range(2, 10)
    ]
    turn = make_turn(
        resources=2,
        objects=[
            core(),
            *combat_units,
            unit(20, "WORKER", controlled=False, position=(30, 0)),
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(
            target_workers=0,
            max_population=None,
            resource_target=0,
        ),
    )

    defensive = [
        item
        for item in report.decisions
        if item.reason == "hold a defensive perimeter around the resource Core"
    ]
    assert len(defensive) == 5
    assert all(item.action == "MOVE" for item in defensive)


def test_dense_core_guard_uses_unique_outer_slots_and_frees_worker_route() -> None:
    guard_positions = [
        (1, -1),
        (1, 0),
        (1, 1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 2),
        (2, 0),
        (-2, 0),
        (2, 2),
    ]
    guards = [
        unit(number, "VANGUARD", position=position)
        for number, position in enumerate(guard_positions, start=2)
    ]
    worker = unit(30, "WORKER", position=(0, 1))
    turn = make_turn(resources=14, objects=[core(), *guards, worker])

    report = decide(
        turn,
        config=StrategyConfig(
            target_workers=0,
            max_population=None,
            resource_target=0,
            offensive_min_combat_units=12,
        ),
    )

    defensive = [
        item
        for item in report.decisions
        if item.actor_kind == "VANGUARD"
        and item.reason == "hold a defensive perimeter around the resource Core"
    ]
    assert len(defensive) == len(guards)
    assert len({item.target for item in defensive}) == len(defensive)
    assert all(
        max(abs(item.target[0]), abs(item.target[1])) >= 3
        for item in defensive
        if item.target is not None
    )

    worker_decision = next(
        item for item in report.decisions if item.actor_id == str(turn.workers[0].id)
    )
    assert worker_decision.action == "MOVE"
    assert worker_decision.reason == "patrol near the stationary Core for resources"


def test_defensive_ring_radius_scales_with_guard_count_and_vision() -> None:
    small = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "VANGUARD", position=(10, 10)),
            unit(3, "VANGUARD", position=(11, 10)),
        ],
    )
    small_report = decide(
        small,
        config=StrategyConfig(target_workers=0, max_population=None),
    )
    small_targets = [
        item.target
        for item in small_report.decisions
        if item.reason == "hold a defensive perimeter around the resource Core"
    ]

    large = make_turn(
        resources=0,
        objects=[
            core(),
            *[
                unit(number, "VANGUARD", position=(number, 10))
                for number in range(2, 7)
            ],
        ],
    )
    large_report = decide(
        large,
        config=StrategyConfig(target_workers=0, max_population=None),
    )
    large_targets = [
        item.target
        for item in large_report.decisions
        if item.reason == "hold a defensive perimeter around the resource Core"
    ]

    assert max(manhattan((0, 0), target) for target in small_targets) == 2
    assert max(manhattan((0, 0), target) for target in large_targets) == 5
    assert len(set(large_targets)) == len(large_targets)


def test_defensive_guard_waits_explicitly_after_reaching_ring_slot() -> None:
    config = StrategyConfig(target_workers=0, max_population=None)
    initial = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "VANGUARD", position=(10, 10)),
            unit(3, "VANGUARD", position=(11, 10)),
        ],
    )
    initial_report = decide(initial, config=config)
    slots = {
        item.actor_id: item.target
        for item in initial_report.decisions
        if item.reason == "hold a defensive perimeter around the resource Core"
    }
    arrived = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "VANGUARD", position=slots[str(object_id(2))]),
            unit(3, "VANGUARD", position=slots[str(object_id(3))]),
        ],
    )

    report = decide(arrived, config=config)

    assert all(
        arrived.plan.unit_actions[guard.id].type == "WAIT"
        for guard in arrived.vanguards
    )

    assert all(
        item.action == "WAIT"
        and item.reason == "hold a defensive perimeter around the resource Core"
        for item in report.decisions
    )


def test_defensive_layout_stays_fixed_across_ticks_and_waits_after_arrival() -> None:
    config = StrategyConfig(target_workers=0, max_population=None)
    strategy = AggressiveStrategy(WorldMemory(), config)
    first = make_turn(
        tick=100,
        resources=0,
        objects=[
            core(),
            unit(2, "VANGUARD", position=(10, 10)),
            unit(3, "VANGUARD", position=(11, 10)),
            unit(4, "VANGUARD", position=(12, 10)),
            unit(5, "VANGUARD", position=(13, 10)),
        ],
    )
    first_report = strategy.decide(first)
    first_slots = {
        item.actor_id: item.target
        for item in first_report.decisions
        if (
            item.reason == "hold a defensive perimeter around the resource Core"
            and item.target is not None
        )
    }

    arrived = make_turn(
        tick=101,
        resources=0,
        obstacles=[(30, 30)],
        objects=[
            core(),
            *[
                unit(number, "VANGUARD", position=first_slots[str(object_id(number))])
                for number in range(2, 6)
            ],
        ],
    )
    second_report = strategy.decide(arrived)
    second_slots = {
        item.actor_id: item.target
        for item in second_report.decisions
        if item.reason == "hold a defensive perimeter around the resource Core"
    }

    assert second_slots == first_slots
    assert all(item.action == "WAIT" for item in second_report.decisions)
    assert all(
        arrived.plan.unit_actions[guard.id].type == "WAIT"
        for guard in arrived.vanguards
    )

    settled = make_turn(
        tick=102,
        resources=0,
        obstacles=[(31, 31)],
        objects=[
            core(),
            *[
                unit(number, "VANGUARD", position=first_slots[str(object_id(number))])
                for number in range(2, 6)
            ],
        ],
    )
    settled_report = strategy.decide(settled)
    assert all(item.action == "WAIT" for item in settled_report.decisions)
    assert {
        item.actor_id: item.target
        for item in settled_report.decisions
        if item.reason == "hold a defensive perimeter around the resource Core"
    } == first_slots


def test_defensive_layout_keeps_vertical_cardinal_anchors() -> None:
    turn = make_turn(
        resources=0,
        objects=[
            core(),
            *[
                unit(number, "VANGUARD", position=(number, 10))
                for number in range(2, 7)
            ],
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(target_workers=0, max_population=None),
    )
    targets = {
        item.target
        for item in report.decisions
        if item.reason == "hold a defensive perimeter around the resource Core"
    }

    assert (0, -5) in targets
    assert (0, 5) in targets


def test_defensive_ring_ignores_obstacle_cells_and_keeps_unique_slots() -> None:
    obstacle = (0, -4)
    turn = make_turn(
        resources=0,
        obstacles=[obstacle],
        objects=[
            core(),
            *[
                unit(number, "VANGUARD", position=(number, 10))
                for number in range(2, 6)
            ],
        ],
    )

    report = decide(
        turn,
        config=StrategyConfig(target_workers=0, max_population=None),
    )
    targets = [
        item.target
        for item in report.decisions
        if item.reason == "hold a defensive perimeter around the resource Core"
    ]

    assert obstacle not in targets
    assert len(set(targets)) == len(targets)
    assert all(manhattan((0, 0), target) == 4 for target in targets)


def test_clear_manhattan_path_allows_detour_but_rejects_complete_wall() -> None:
    assert _clear_manhattan_path((0, 0), (2, 1), {(1, 0)})
    assert not _clear_manhattan_path((0, 0), (2, 0), {(1, -1), (1, 0), (1, 1)})


def test_offensive_patrol_pursues_a_recently_seen_enemy_core() -> None:
    combat_units = [
        unit(number, "VANGUARD" if number % 2 else "RANGER", position=(number, 1))
        for number in range(2, 10)
    ]
    memory = WorldMemory()
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(target_workers=0, max_population=None, resource_target=0),
    )
    observed = make_turn(
        tick=100,
        resources=15,
        objects=[
            core(),
            *combat_units,
            core(20, controlled=False, owner_username="rival", position=(10, 0)),
        ],
    )
    strategy.decide(observed)

    hidden = make_turn(
        tick=101,
        resources=15,
        objects=[core(), *combat_units],
    )
    report = strategy.decide(hidden)

    assert any(item.reason == "hunt last seen core" for item in report.decisions)


def test_offensive_patrol_drops_stale_enemy_unit_memory() -> None:
    combat_units = [
        unit(number, "VANGUARD" if number % 2 else "RANGER", position=(number, 1))
        for number in range(2, 10)
    ]
    memory = WorldMemory()
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(target_workers=0, max_population=None, resource_target=0),
    )
    observed = make_turn(
        tick=100,
        resources=15,
        objects=[
            core(),
            *combat_units,
            unit(20, "WORKER", controlled=False, position=(10, 0)),
        ],
    )
    strategy.decide(observed)

    hidden = make_turn(
        tick=133,
        resources=15,
        objects=[core(), *combat_units],
    )
    report = strategy.decide(hidden)

    assert not any(item.reason == "hunt last seen unit" for item in report.decisions)
    assert any(
        item.reason == "search outward for enemy units and Cores"
        for item in report.decisions
    )


def test_unbounded_population_clears_full_core_cell_for_expansion() -> None:
    units = [unit(number, "VANGUARD", position=(number, 2)) for number in range(3, 14)]
    units.extend(
        unit(number, "WORKER", position=(number, 3)) for number in range(14, 21)
    )
    units.append(unit(2, "WORKER", position=(0, 0), cargo=1))
    turn = make_turn(resources=130, objects=[core(), *units])

    decide(
        turn,
        config=StrategyConfig(
            target_workers=12,
            max_population=None,
            safety_reserve=10,
        ),
    )

    worker = next(item for item in turn.workers if str(item.id) == object_id(2))
    assert turn.plan.unit_actions[worker.id].type == "MOVE"
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"


def test_resource_goal_builds_one_capacity_unit_beyond_required_population() -> None:
    units = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 20)]
    turn = make_turn(resources=5, objects=[core(), *units])

    decide(
        turn,
        config=StrategyConfig(
            target_workers=20,
            max_population=20,
            resource_target=95,
        ),
    )

    assert turn.state.population == 18
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_resource_goal_stops_production_at_buffered_capacity() -> None:
    units = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 22)]
    turn = make_turn(resources=100, objects=[core(), *units])

    decide(
        turn,
        config=StrategyConfig(
            target_workers=20,
            max_population=20,
            resource_target=95,
        ),
    )

    assert turn.state.population == 20
    assert turn.plan.core_action is None


def test_resource_goal_reserves_for_one_guard_then_resumes_workers() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=20,
            max_population=20,
            resource_target=95,
        ),
    )
    workers = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 8)]
    nearby_enemy = core(
        20,
        controlled=False,
        owner_username="rival",
        position=(20, 0),
    )

    saving = make_turn(
        tick=100,
        resources=5,
        objects=[core(), *workers, nearby_enemy],
    )
    strategy.decide(saving)
    assert saving.plan.core_action is None

    guarded = make_turn(
        tick=101,
        resources=10,
        objects=[core(), *workers, nearby_enemy],
    )
    strategy.decide(guarded)
    assert guarded.plan.core_action is not None
    assert guarded.plan.core_action.type == "SPAWN"
    assert guarded.plan.core_action.unit_type is UnitType.VANGUARD

    growing = make_turn(
        tick=102,
        resources=5,
        objects=[
            core(),
            *workers,
            unit(8, "VANGUARD", position=(2, 3)),
            nearby_enemy,
        ],
    )
    strategy.decide(growing)
    assert growing.plan.core_action is not None
    assert growing.plan.core_action.type == "SPAWN"
    assert growing.plan.core_action.unit_type is UnitType.WORKER


def test_resource_goal_builds_first_guard_before_full_worker_target() -> None:
    workers = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 8)]
    turn = make_turn(
        resources=10,
        objects=[core(), *workers],
    )

    decide(
        turn,
        config=StrategyConfig(
            target_workers=12,
            max_population=20,
            resource_target=95,
        ),
    )

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_resource_goal_remembers_nearby_enemy_core_for_guard_reserve() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=20,
            max_population=20,
            resource_target=95,
        ),
    )
    workers = [unit(number, "WORKER", position=(number, 2)) for number in range(2, 8)]
    observed = make_turn(
        tick=100,
        resources=5,
        objects=[
            core(),
            *workers,
            core(
                20,
                controlled=False,
                owner_username="rival",
                position=(20, 0),
            ),
        ],
    )
    strategy.decide(observed)

    hidden = make_turn(tick=261, resources=5, objects=[core(), *workers])
    strategy.decide(hidden)

    assert hidden.plan.core_action is None


def test_resource_guard_ignores_stale_enemy_core_after_respawn() -> None:
    memory = WorldMemory()
    config = StrategyConfig(
        target_workers=12,
        max_population=None,
        enemy_memory_ttl=160,
    )
    strategy = AggressiveStrategy(memory, config)
    observed = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            core(20, controlled=False, owner_username="rival", position=(20, 0)),
        ],
    )
    strategy.decide(observed)

    recovered = make_turn(
        tick=261,
        resources=10,
        objects=[core(), unit(2, "WORKER", position=(1, 0))],
    )
    strategy.decide(recovered)

    assert recovered.plan.core_action is not None
    assert recovered.plan.core_action.type == "SPAWN"
    assert recovered.plan.core_action.unit_type is UnitType.VANGUARD


def test_resource_guard_keeps_recent_enemy_core_priority() -> None:
    memory = WorldMemory()
    config = StrategyConfig(
        target_workers=12,
        max_population=None,
        enemy_memory_ttl=160,
    )
    strategy = AggressiveStrategy(memory, config)
    observed = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            core(20, controlled=False, owner_username="rival", position=(20, 0)),
        ],
    )
    strategy.decide(observed)

    guarded = make_turn(
        tick=101,
        resources=10,
        objects=[core(), unit(2, "WORKER", position=(1, 0))],
    )
    strategy.decide(guarded)

    assert guarded.plan.core_action is not None
    assert guarded.plan.core_action.type == "SPAWN"
    assert guarded.plan.core_action.unit_type is UnitType.VANGUARD


def test_resource_goal_keeps_core_stationary_and_preserves_stockpile() -> None:
    turn = make_turn(
        resources=95,
        objects=[
            core(position=(100, 100), shield=4),
            unit(2, "WORKER", position=(101, 100)),
        ],
    )

    decide(turn, config=StrategyConfig(resource_target=95, max_population=20))

    assert turn.plan.core_action is None


def test_resource_goal_retargets_distant_worker_to_local_patrol() -> None:
    memory = WorldMemory()
    turn = make_turn(
        tick=160,
        objects=[
            core(position=(100, 100)),
            unit(2, "WORKER", position=(0, 0)),
        ],
    )

    decide(
        turn,
        memory=memory,
        config=StrategyConfig(resource_target=95, resource_patrol_radius=10),
    )

    goal = memory.goal_for(object_id(2))
    assert goal is not None
    assert goal.purpose == "resource-patrol-v3"
    assert 0 < manhattan(goal.position, (100, 100)) <= 20
    assert turn.plan.unit_actions[turn.workers[0].id].type == "MOVE"


def test_resource_patrol_rotates_immediately_after_reaching_goal() -> None:
    memory = WorldMemory()
    config = StrategyConfig(resource_target=95, resource_patrol_radius=10)
    strategy = AggressiveStrategy(memory, config)
    first = make_turn(
        tick=160,
        objects=[core(position=(100, 100)), unit(2, "WORKER", position=(90, 90))],
    )
    strategy.decide(first)
    first_goal = memory.goal_for(object_id(2))
    assert first_goal is not None

    arrived = make_turn(
        tick=161,
        objects=[
            core(position=(100, 100)),
            unit(2, "WORKER", position=first_goal.position),
        ],
    )
    strategy.decide(arrived)
    next_goal = memory.goal_for(object_id(2))

    assert next_goal is not None
    assert next_goal.position != first_goal.position
    assert arrived.plan.unit_actions[arrived.workers[0].id].type == "MOVE"


def test_resource_patrol_assigns_distinct_initial_sectors() -> None:
    memory = WorldMemory()
    turn = make_turn(
        tick=160,
        objects=[
            core(position=(100, 100)),
            unit(2, "WORKER", position=(100, 100)),
            unit(10, "WORKER", position=(101, 100)),
            unit(18, "WORKER", position=(99, 100)),
        ],
    )

    decide(
        turn,
        memory=memory,
        config=StrategyConfig(resource_target=95, resource_patrol_radius=10),
    )

    goals = set()
    for worker in turn.workers:
        goal = memory.goal_for(str(worker.id))
        assert goal is not None
        goals.add(goal.position)
    assert len(goals) == 3


def test_resource_patrol_splits_duplicate_existing_goals() -> None:
    memory = WorldMemory()
    shared_goal = UnitGoal((12, 12), 100, "resource-patrol-v3")
    memory.set_goal(object_id(2), shared_goal)
    memory.set_goal(object_id(3), shared_goal)
    turn = make_turn(
        tick=101,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            unit(3, "WORKER", position=(-1, 0)),
        ],
    )

    decide(
        turn,
        memory=memory,
        config=StrategyConfig(resource_target=95, resource_patrol_radius=18),
    )

    goals = [memory.goal_for(object_id(number)) for number in (2, 3)]
    assert all(goal is not None for goal in goals)
    assert len({goal.position for goal in goals if goal is not None}) == 2


def test_resource_patrol_replaces_goal_near_remembered_enemy_core() -> None:
    memory = WorldMemory()
    config = StrategyConfig(
        resource_target=95,
        resource_patrol_radius=18,
        worker_threat_radius=2,
    )
    strategy = AggressiveStrategy(memory, config)
    observed = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER"),
            core(3, controlled=False, owner_username="rival", position=(6, 0)),
        ],
    )
    strategy.decide(observed)
    memory.set_goal(
        object_id(2),
        UnitGoal((6, 0), observed.tick, "resource-patrol-v3"),
    )

    hidden = make_turn(
        tick=125,
        objects=[core(), unit(2, "WORKER", position=(-12, -18))],
    )
    strategy.decide(hidden)

    replacement = memory.goal_for(object_id(2))
    assert replacement is not None
    assert replacement.position != (6, 0)
    assert replacement.position != hidden.workers[0].position
    assert manhattan(replacement.position, (6, 0)) > 2


def test_worker_path_avoids_remembered_enemy_core_exclusion() -> None:
    memory = WorldMemory()
    config = StrategyConfig(
        resource_target=95,
        resource_patrol_radius=18,
        worker_threat_radius=2,
    )
    strategy = AggressiveStrategy(memory, config)
    observed = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(3, 0)),
            core(3, controlled=False, owner_username="rival", position=(6, 0)),
        ],
    )
    strategy.decide(observed)
    memory.set_goal(
        object_id(2),
        UnitGoal((10, 0), observed.tick, "resource-patrol-v3"),
    )

    hidden = make_turn(
        tick=125,
        objects=[core(), unit(2, "WORKER", position=(3, 0))],
    )
    strategy.decide(hidden)

    action = hidden.plan.unit_actions[hidden.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is not Direction.RIGHT


def test_worker_path_temporarily_avoids_recent_combat_enemy() -> None:
    memory = WorldMemory()
    config = StrategyConfig(
        resource_target=95,
        resource_patrol_radius=18,
        worker_threat_radius=2,
    )
    strategy = AggressiveStrategy(memory, config)
    observed = make_turn(
        tick=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(3, 0)),
            unit(3, "VANGUARD", controlled=False, position=(6, 0)),
        ],
    )
    strategy.decide(observed)
    memory.set_goal(
        object_id(2),
        UnitGoal((10, 0), observed.tick, "resource-patrol-v3"),
    )

    hidden = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(3, 0))],
    )
    strategy.decide(hidden)
    recent_action = hidden.plan.unit_actions[hidden.workers[0].id]
    assert recent_action.type == "MOVE"
    assert recent_action.direction is not Direction.RIGHT

    expired = make_turn(
        tick=125,
        objects=[core(), unit(2, "WORKER", position=(3, 0))],
    )
    strategy.decide(expired)
    expired_action = expired.plan.unit_actions[expired.workers[0].id]
    assert expired_action.type == "MOVE"
    assert expired_action.direction is Direction.RIGHT


def test_resource_patrol_route_covers_square_without_vision_gaps() -> None:
    radius = StrategyConfig().resource_patrol_radius
    offsets = _resource_patrol_offsets(radius=radius, spacing=6)

    assert radius == 14
    assert len(offsets) == 36
    assert len(set(offsets)) == len(offsets)
    assert all(max(abs(x), abs(y)) <= radius for x, y in offsets)
    assert all(manhattan(left, right) <= 6 for left, right in pairwise(offsets))
    # The furthest patrol point is one 28-cell leg from the Core rather than
    # the 60-cell leg the old radius produced, so a sweep step is a round trip
    # a Worker can finish instead of a walk that expires mid-transit.
    assert max(abs(x) + abs(y) for x, y in offsets) == 28


def test_strategy_avoids_recently_contested_resource_patrol_cell() -> None:
    memory = WorldMemory()
    config = StrategyConfig(resource_target=95, resource_patrol_radius=18)
    strategy = AggressiveStrategy(memory, config)
    first = make_turn(
        tick=160,
        objects=[core(position=(100, 100)), unit(2, "WORKER", position=(100, 100))],
    )
    strategy.decide(first)
    first_action = first.plan.unit_actions[first.workers[0].id]
    assert first_action.type == "MOVE"
    contested = (
        first.workers[0].position[0] + first_action.direction.delta[0],
        first.workers[0].position[1] + first_action.direction.delta[1],
    )
    memory.contested_positions[contested] = first.tick

    second = make_turn(
        tick=161,
        objects=[core(position=(100, 100)), unit(2, "WORKER", position=(100, 100))],
    )
    strategy.decide(second)
    second_action = second.plan.unit_actions[second.workers[0].id]

    assert second_action.type == "MOVE"
    assert second_action.direction != first_action.direction


def test_worker_retreats_from_recently_contested_area() -> None:
    memory = WorldMemory(contested_positions={(6, 0): 160})
    turn = make_turn(
        tick=161,
        objects=[core(position=(0, 0)), unit(2, "WORKER", position=(5, 0))],
    )

    report = decide(
        turn,
        memory=memory,
        config=StrategyConfig(resource_target=95),
    )

    action = turn.plan.unit_actions[turn.workers[0].id]
    assert action.type == "MOVE"
    assert action.direction is Direction.LEFT
    assert any("retreat" in item.reason for item in report.decisions)


def test_exploration_goal_remains_stable_across_turns() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    goal = memory.goal_for(object_id(2))
    assert goal is not None

    second = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(99, 100))],
    )
    strategy.decide(second)
    assert memory.goal_for(object_id(2)) == goal


def test_exploration_goal_renews_after_progress_beyond_ttl() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    first_goal = memory.goal_for(object_id(2))
    assert first_goal is not None

    progressed = make_turn(
        tick=first.tick + strategy.config.exploration_goal_ttl + 1,
        objects=[core(), unit(2, "WORKER", position=(99, 100))],
    )
    strategy.decide(progressed)
    renewed = memory.goal_for(object_id(2))

    assert renewed is not None
    assert renewed.position == first_goal.position
    assert renewed.assigned_tick == progressed.tick
    assert renewed.last_progress_position == (99, 100)


def test_legacy_exploration_goal_initializes_progress_baseline() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    initial = memory.goal_for(object_id(2))
    assert initial is not None
    memory.set_goal(
        object_id(2),
        UnitGoal(initial.position, initial.assigned_tick, initial.purpose),
    )

    baseline = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(99, 100))],
    )
    strategy.decide(baseline)
    migrated = memory.goal_for(object_id(2))
    assert migrated is not None
    assert migrated.last_progress_position == (99, 100)

    progressed = make_turn(
        tick=baseline.tick + strategy.config.exploration_goal_ttl,
        objects=[core(), unit(2, "WORKER", position=(98, 100))],
    )
    strategy.decide(progressed)
    renewed = memory.goal_for(object_id(2))
    assert renewed is not None
    assert renewed.position == initial.position
    assert renewed.assigned_tick == progressed.tick


def test_exploration_goal_is_replaced_when_target_becomes_obstacle() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    initial = memory.goal_for(object_id(2))
    assert initial is not None

    blocked = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=(99, 100))],
        obstacles=[initial.position],
    )
    strategy.decide(blocked)
    replacement = memory.goal_for(object_id(2))

    assert replacement is not None
    assert replacement.position != initial.position
    assert replacement.position not in memory.obstacles


def test_exploration_goal_expires_after_worker_stalls() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    first_goal = memory.goal_for(object_id(2))
    assert first_goal is not None

    stalled = make_turn(
        tick=first.tick + strategy.config.exploration_goal_ttl + 1,
        objects=[core(), unit(2, "WORKER", position=(100, 100))],
    )
    strategy.decide(stalled)
    replacement = memory.goal_for(object_id(2))

    assert replacement is not None
    assert replacement.position != first_goal.position
    assert replacement.assigned_tick == stalled.tick


def test_exploration_goal_uses_aggressive_default_stride() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    turn = make_turn(
        tick=0,
        objects=[core(), unit(2, "WORKER", position=(-100, 100))],
    )

    strategy.decide(turn)

    goal = memory.goal_for(object_id(2))
    assert goal is not None
    assert goal.position == (-76, 76)
    assert goal.purpose == "explore-center-v3"


def test_reaching_exploration_goal_immediately_advances_toward_center() -> None:
    memory = WorldMemory()
    strategy = AggressiveStrategy(memory)
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(100, 100))])
    strategy.decide(first)
    first_goal = memory.goal_for(object_id(2))
    assert first_goal is not None

    reached = make_turn(
        tick=101,
        objects=[core(), unit(2, "WORKER", position=first_goal.position)],
    )
    strategy.decide(reached)
    next_goal = memory.goal_for(object_id(2))

    assert next_goal is not None
    assert next_goal.position != first_goal.position
    assert abs(next_goal.position[0]) < abs(first_goal.position[0])
    assert abs(next_goal.position[1]) < abs(first_goal.position[1])
    assert reached.plan.unit_actions[reached.workers[0].id].type == "MOVE"


def test_core_saves_for_the_ranger_the_composition_policy_asks_for() -> None:
    """Falling through to the cheaper Vanguard silently inverted the policy.

    A Ranger costs more than a Vanguard, so taking the first affordable
    candidate meant the melee fallback always won the race and the roster
    drifted to almost pure Vanguards while the policy asked for twice as many
    Rangers.  With enough resources for a Vanguard but not a Ranger the Core
    now waits instead of spending.
    """

    roster = [
        unit(number, "VANGUARD", position=(number, 3)) for number in range(2, 6)
    ] + [unit(number, "WORKER", position=(number, 4)) for number in range(6, 8)]
    vanguard_cost = unit_cost(UnitType.VANGUARD, len(roster))
    ranger_cost = unit_cost(UnitType.RANGER, len(roster))
    assert vanguard_cost < ranger_cost

    turn = make_turn(resources=vanguard_cost, objects=[core(), *roster])
    decide(turn, config=StrategyConfig(target_workers=2, max_population=None))
    assert turn.plan.core_action is None

    funded = make_turn(resources=ranger_cost, objects=[core(), *roster])
    decide(funded, config=StrategyConfig(target_workers=2, max_population=None))
    assert isinstance(funded.plan.core_action, SpawnAction)
    assert funded.plan.core_action.unit_type is UnitType.RANGER


def test_unaffordable_preference_still_falls_back_at_low_capacity() -> None:
    """Saving must never deadlock a Core that can never hold the price."""

    turn = make_turn(
        resources=10,
        objects=[core(), unit(2, "WORKER", position=(1, 0))],
    )

    decide(turn, config=StrategyConfig(target_workers=12, max_population=None))

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    # Capacity 10 cannot hold a Ranger, so the Vanguard fallback is correct.
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_growth_slowdown_banks_income_instead_of_buying_a_pricier_roster() -> None:
    roster = [
        unit(number, "VANGUARD", position=(number % 9, number // 9))
        for number in range(2, 42)
    ]
    assert len(roster) == 40
    config = StrategyConfig(target_workers=0, max_population=None)
    strategy = AggressiveStrategy(WorldMemory(), config)

    below_target = make_turn(resources=170, objects=[core(), *roster])
    assert below_target.resource_capacity == 200
    strategy.decide(below_target)
    assert below_target.plan.core_action is None

    strategy = AggressiveStrategy(WorldMemory(), config)
    banked = make_turn(resources=200, objects=[core(), *roster])
    strategy.decide(banked)
    assert banked.plan.core_action is not None
    assert banked.plan.core_action.type == "SPAWN"


def test_growth_slowdown_lifts_under_real_enemy_pressure() -> None:
    """Pressure must unlock the bank the slowdown spent hours filling."""

    roster = [
        unit(number, "VANGUARD", position=(number % 7 + 1, number // 7))
        for number in range(2, 42)
    ]
    turn = make_turn(
        resources=170,
        objects=[
            core(),
            *roster,
            unit(90, "RANGER", controlled=False, position=(0, 12)),
            unit(91, "RANGER", controlled=False, position=(1, 12)),
        ],
    )

    decide(turn, config=StrategyConfig(target_workers=0, max_population=None))

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"


def test_resource_patrol_goal_is_renewed_while_the_worker_closes_in() -> None:
    """A far patrol point outlives the goal TTL in travel Ticks.

    Rotating to the next sweep point mid-transit meant Workers walked between
    targets they never reached, so the goal is renewed while progress lasts.
    """

    memory = WorldMemory()
    config = StrategyConfig(resource_target=95, resource_patrol_radius=14)
    strategy = AggressiveStrategy(memory, config)
    first = make_turn(
        tick=160,
        objects=[core(position=(100, 100)), unit(2, "WORKER", position=(100, 100))],
    )
    strategy.decide(first)
    goal = memory.goal_for(object_id(2))
    assert goal is not None

    closer = (
        goal.position[0] + (1 if goal.position[0] < 100 else -1),
        goal.position[1],
    )
    progressed = make_turn(
        tick=first.tick + config.exploration_goal_ttl + 1,
        objects=[core(position=(100, 100)), unit(2, "WORKER", position=closer)],
    )
    strategy.decide(progressed)
    renewed = memory.goal_for(object_id(2))

    assert renewed is not None
    assert renewed.position == goal.position
    assert renewed.assigned_tick == progressed.tick


def test_loaded_worker_stages_next_to_a_busy_core_instead_of_idling() -> None:
    turn = make_turn(
        objects=[
            core(position=(0, 0)),
            unit(2, "WORKER", position=(0, 1), cargo=1),
            unit(3, "WORKER", position=(0, 4), cargo=1),
        ],
    )

    report = decide(turn, config=StrategyConfig(target_workers=12, max_population=None))

    far = next(worker for worker in turn.workers if worker.position == (0, 4))
    staged = turn.plan.unit_actions[far.id]
    assert staged.type == "MOVE"
    assert staged.direction is Direction.UP
    assert any(
        item.actor_id == str(far.id)
        and item.reason == "stage carried resources next to the busy Core"
        for item in report.decisions
    )


def test_distant_worker_is_not_dragged_into_the_core_assault_zone() -> None:
    config = StrategyConfig(target_workers=12, max_population=None)
    turn = make_turn(
        objects=[
            core(position=(0, 0), shield=4),
            unit(2, "WORKER", position=(0, 30)),
            unit(3, "VANGUARD", controlled=False, position=(0, 1)),
            unit(4, "VANGUARD", controlled=False, position=(1, 0)),
            unit(5, "RANGER", controlled=False, position=(-1, 0)),
        ],
    )

    report = decide(turn, config=config)

    distant = next(item for item in report.decisions if item.actor_id == object_id(2))
    assert "assault zone" not in distant.reason
    assert distant.reason == "patrol near the stationary Core for resources"


def _hunt_sequence(enemy_path, *, tick0=100, vanguards=((0, 0),)):
    """Replay an intruder track through one persistent strategy instance."""

    strategy = AggressiveStrategy(WorldMemory())
    turns = []
    for offset, enemy_position in enumerate(enemy_path):
        objects: list = [core(position=(-3, 0))]
        objects += [
            unit(10 + index, "VANGUARD", position=position)
            for index, position in enumerate(vanguards)
        ]
        if enemy_position is not None:
            objects.append(unit(9, "WORKER", controlled=False, position=enemy_position))
        turn = make_turn(tick=tick0 + offset, resources=0, objects=objects)
        report = strategy.decide(turn)
        turns.append((turn, report))
    return turns


def test_vanguard_sweeps_the_cell_an_intruder_walks_into() -> None:
    # The thief is diagonal to the guard right now, so the old adjacency-only
    # sweep could never fire; its drift puts it in the swept cell after
    # movement, which is the snapshot combat actually resolves on.
    turns = _hunt_sequence([(1, 2), (1, 1)])

    turn, report = turns[-1]
    action = turn.plan.unit_actions[turn.vanguards[0].id]
    assert action.type == "SWEEP"
    assert action.direction is Direction.RIGHT
    assert any(
        item.action == "SWEEP" and item.target == (1, 0) for item in report.decisions
    )


def test_vanguard_skips_a_sweep_the_intruder_walks_out_of() -> None:
    # Adjacent, but drifting on out of the cell: swinging at it is a
    # guaranteed miss, so cutting the runner off is worth strictly more.
    turns = _hunt_sequence([(1, 1), (1, 0)])

    turn, _ = turns[-1]
    action = turn.plan.unit_actions[turn.vanguards[0].id]
    assert action.type == "MOVE"


def test_vanguard_aims_ahead_of_a_fleeing_intruder() -> None:
    turns = _hunt_sequence([(4, 0), (5, 0)])

    _, report = turns[-1]
    decision = report.decisions[0]
    assert decision.action == "MOVE"
    # A goal behind or level with the runner means a permanent two-cell gap.
    assert decision.target is not None and decision.target[0] > 5


def test_vanguard_escort_keeps_hunting_an_intruder_through_a_vision_gap() -> None:
    turns = _hunt_sequence([(4, 0), (5, 0), None, None])

    for turn, report in turns[2:]:
        action = turn.plan.unit_actions[turn.vanguards[0].id]
        assert action.type == "MOVE"
        assert action.direction is Direction.RIGHT
        assert any(
            item.reason == "hunt last seen unit" and item.target == (5, 0)
            for item in report.decisions
        )


def test_intruder_hunt_expires_once_the_sighting_goes_stale() -> None:
    config = StrategyConfig(intruder_hunt_ttl=1)
    strategy = AggressiveStrategy(WorldMemory(), config)
    for tick, enemy_position in ((100, (4, 0)), (101, (5, 0)), (110, None)):
        objects: list = [core(position=(-3, 0)), unit(10, "VANGUARD", position=(0, 0))]
        if enemy_position is not None:
            objects.append(unit(9, "WORKER", controlled=False, position=enemy_position))
        turn = make_turn(tick=tick, resources=0, objects=objects)
        report = strategy.decide(turn)

    assert all(item.reason != "hunt last seen unit" for item in report.decisions)


def test_ranger_shoots_an_intruder_without_joining_the_chase() -> None:
    strategy = AggressiveStrategy(WorldMemory())
    turn = make_turn(
        objects=[
            core(position=(0, 0)),
            unit(2, "RANGER", position=(0, 1)),
            unit(3, "VANGUARD", position=(1, 0)),
            unit(9, "WORKER", controlled=False, position=(0, 4)),
        ],
    )

    report = strategy.decide(turn)

    action = turn.plan.unit_actions[turn.rangers[0].id]
    assert action.type == "SHOOT"
    # A Worker cannot shoot back, so backing off to max range would only give
    # the thief a free Tick.
    assert not any("disengage" in item.reason for item in report.decisions)


def test_defensive_perimeter_stops_growing_with_the_army() -> None:
    # The count-and-vision radius alone put a 29-guard fleet 30 cells out and
    # left the whole interior unguarded.
    guards = [unit(10 + index, "VANGUARD", position=(0, 40)) for index in range(29)]
    turn = make_turn(
        objects=[core(position=(0, 0)), *guards],
        resources=0,
    )
    strategy = AggressiveStrategy(
        WorldMemory(), StrategyConfig(target_workers=0, max_population=None)
    )
    strategy.decide(turn)

    layout = strategy._defensive_layout
    assert layout is not None
    assert layout.radius <= StrategyConfig().defensive_perimeter_max_radius
    assert all(
        manhattan((0, 0), slot) <= StrategyConfig().defensive_perimeter_max_radius
        for slot in layout.assignments.values()
    )
