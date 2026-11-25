"""Current Core attack geometry alone permits sacrificing defensive Rangers."""

from collections import Counter
from uuid import UUID

import pytest
from arena_hero import Direction

from arena_hero_bot.combat_policy import ThreatAssessment, ThreatLevel
from arena_hero_bot.geometry import add, line_of_fire
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig, _TurnContext

from .factories import core, make_turn, object_id, unit
from .test_layered_defense import Encounter


def _strategy():
    return AggressiveStrategy(
        WorldMemory(core_home_position=(0, 0)),
        StrategyConfig(target_workers=0, max_population=None, campaign_mode=True),
    )


def _context(strategy, turn):
    strategy.memory.observe(turn)
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={actor.position for actor in turn.units},
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
        remaining_resources=turn.resources,
        remaining_resource_space=turn.resource_space,
    )


@pytest.mark.parametrize("role", ["defense", "patrol-1"])
@pytest.mark.parametrize("hp", [1, 2])
def test_current_core_fire_forces_lethal_entry_then_current_cell_fire(role, hp):
    encounter = Encounter(
        [unit(2, "RANGER", position=(0, -2), hp=hp)],
        [unit(n, "RANGER", controlled=False, position=(-3, 0)) for n in (30, 31)],
        resources=0,
        rocks=[(-3, -1), (-3, -3)],
    )
    strategy = _strategy()
    strategy.memory.unit_roles = {object_id(2): role}
    turn = encounter.turn()
    context = _context(strategy, turn)
    for cell in ((0, -3), (-1, -2)):
        strategy.memory.incoming_damage[cell] = (turn.tick - 1, 2)

    strategy._decide_ranger(turn.rangers[0], context, ())

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "MOVE"
    destination = add((0, -2), action.direction)
    assert line_of_fire(destination, (-3, 0), encounter.rocks)
    assert len(strategy._visible_attackers_at(destination, turn, encounter.rocks)) == 2
    assert "current Core range" in context.report.decisions[-1].reason
    encounter.settle(
        turn.plan,
        {
            object_id(n): {
                "type": "SHOOT",
                "target_id": object_id(1),
                "expected_cell": [0, 0],
            }
            for n in (30, 31)
        },
    )
    assert encounter.core["shield"] == 3
    reply = encounter.turn()
    reply_context = _context(strategy, reply)
    strategy._decide_ranger(reply.rangers[0], reply_context, ())
    shot = reply.plan.unit_actions[UUID(int=2)]
    assert shot.type == "SHOOT" and shot.expected_cell == (-3, 0)
    encounter.settle(reply.plan, {})
    assert min(enemy["hp"] for enemy in encounter.foes.values()) == 1


def test_doomed_shooter_targets_core_attacker_before_an_unrelated_easy_kill():
    turn = make_turn(
        resources=100,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 2), hp=1),
            unit(30, "RANGER", controlled=False, position=(0, -1)),
            unit(31, "RANGER", controlled=False, position=(3, 2), hp=1),
            unit(32, "VANGUARD", controlled=False, position=(0, 3)),
        ],
    )
    strategy = _strategy()
    context = _context(strategy, turn)
    context.core_hunt_ids = frozenset({UUID(int=2)})
    context.flank_ids = frozenset({UUID(int=2)})
    strategy.memory.incoming_damage[(0, 2)] = (turn.tick - 1, 3)

    strategy._decide_ranger(turn.rangers[0], context, ())

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "SHOOT"
    assert action.target_id == UUID(int=30)
    assert action.expected_cell == (0, -1)
    assert context.damage_ledger.planned_damage == {object_id(30): 1}


def test_direct_core_volley_allocates_separate_stacked_target_uuids():
    encounter = Encounter(
        [
            unit(n, "RANGER", position=cell)
            for n, cell in ((2, (-1, -3)), (3, (1, -3)), (4, (2, -3)), (5, (0, -2)))
        ],
        [unit(n, "RANGER", controlled=False, position=(0, -3)) for n in (30, 31)],
        resources=0,
    )
    strategy = _strategy()
    turn = encounter.turn()
    context = _context(strategy, turn)

    assert strategy._force_core_defense(context)

    shots = list(turn.plan.unit_actions.values())
    assert all(a.type == "SHOOT" and a.expected_cell == (0, -3) for a in shots)
    assert Counter(a.target_id for a in shots) == {UUID(int=30): 2, UUID(int=31): 2}
    encounter.settle(turn.plan, {})
    assert encounter.kills == 2


def test_forced_shot_accepts_a_loss_without_a_same_tick_kill():
    encounter = Encounter(
        [unit(2, "RANGER", position=(0, 2), hp=1)],
        [unit(30, "RANGER", controlled=False, position=(0, -1))],
        resources=0,
    )
    strategy = _strategy()
    turn = encounter.turn()
    strategy._force_core_defense(_context(strategy, turn))
    encounter.settle(
        turn.plan,
        {
            object_id(30): {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [0, 2],
            }
        },
    )

    assert (encounter.losses, encounter.kills) == (1, 0)
    assert encounter.foes[object_id(30)]["hp"] == 1


def test_shootable_bystander_does_not_prevent_moving_toward_core_attacker():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, -2)),
            unit(30, "RANGER", controlled=False, position=(-3, 0)),
            unit(31, "WORKER", controlled=False, position=(0, -5), hp=1),
        ]
    )
    strategy = _strategy()
    context = _context(strategy, turn)
    strategy._decide_ranger(turn.rangers[0], context, ())

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "MOVE"
    assert line_of_fire(add((0, -2), action.direction), (-3, 0), set())


@pytest.mark.parametrize(
    "kind, position, rocks",
    [
        ("RANGER", (0, 4), []),
        ("RANGER", (3, 1), []),
        ("RANGER", (0, 3), [(0, 1)]),
        ("VANGUARD", (1, 1), []),
        ("WORKER", (0, 1), []),
    ],
)
def test_recent_or_predicted_pressure_never_grants_forced_entry(kind, position, rocks):
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(2, 2), hp=1),
            unit(30, kind, controlled=False, position=position),
        ],
        obstacles=rocks,
    )
    strategy = _strategy()
    context = _context(strategy, turn)
    context.threat = ThreatAssessment(
        level=ThreatLevel.ENGAGED,
        recent_core_attack=True,
        projected_core_damage=5,
        minimum_ticks_to_range=1,
    )

    assert not strategy._force_core_defense(context)
    assert not turn.plan.unit_actions
    assert not context.report.decisions


def test_remembered_attacker_and_enemy_core_do_not_activate_exception():
    strategy = _strategy()
    previous = make_turn(
        tick=99,
        objects=[core(), unit(30, "RANGER", controlled=False, position=(0, 2))],
    )
    strategy.memory.observe(previous)
    turn = make_turn(
        objects=[
            core(),
            core(50, controlled=False, position=(0, 2)),
            unit(2, "RANGER", position=(0, 1), hp=1),
        ],
    )
    context = _context(strategy, turn)

    assert not strategy._force_core_defense(context)
    assert not turn.plan.unit_actions


def test_force_permission_ends_on_actual_core_escape_even_with_recent_damage():
    strategy = _strategy()
    for tick, core_cell, expected in ((100, (0, 0), True), (101, (1, 0), False)):
        turn = make_turn(
            tick=tick,
            objects=[
                core(position=core_cell),
                unit(2, "RANGER", position=(0, -2), hp=1),
                unit(30, "RANGER", controlled=False, position=(-3, 0)),
            ],
        )
        context = _context(strategy, turn)
        context.threat = ThreatAssessment(
            level=ThreatLevel.ENGAGED, recent_core_attack=True
        )
        assert strategy._force_core_defense(context) is expected
        if not expected:
            assert not turn.plan.unit_actions
            strategy.memory.unit_roles[object_id(2)] = "defense"
            assert not strategy._queue_move(
                turn.rangers[0],
                Direction.UP,
                context,
                reason="attempt expired forced entry",
                target=(0, -3),
                force_core_attack=True,
            )


def test_moving_core_uses_actual_position_not_safe_destination():
    turn = make_turn(
        objects=[
            core(
                state="MOVING",
                move_direction="RIGHT",
                destination=[1, 0],
                move_progress=3,
                move_required_ticks=4,
            ),
            unit(2, "RANGER", position=(1, -1), hp=1),
            unit(30, "VANGUARD", controlled=False, position=(0, -1)),
        ]
    )
    strategy = _strategy()
    context = _context(strategy, turn)
    assert strategy._force_core_defense(context)
    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "SHOOT" and action.target_id == UUID(int=30)


@pytest.mark.parametrize(
    "obstruction",
    ["one_friend", "two_friends", "rock", "enemy", "arrival", "departure"],
)
def test_forced_entry_obeys_real_capacity_and_terrain(obstruction):
    rocks = [(1, -5), (2, -4), (1, -3)]
    objects = [
        core(),
        unit(2, "RANGER", position=(1, -4)),
        unit(30, "RANGER", controlled=False, position=(0, -2)),
    ]
    if obstruction in {"one_friend", "two_friends", "arrival", "departure"}:
        objects.append(unit(3, "WORKER", position=(0, -4)))
    if obstruction in {"two_friends", "departure"}:
        objects.append(unit(4, "WORKER", position=(0, -4)))
    if obstruction == "rock":
        rocks.append((0, -4))
    if obstruction == "enemy":
        objects.append(unit(31, "WORKER", controlled=False, position=(0, -4)))
    turn = make_turn(objects=objects, obstacles=rocks)
    strategy = _strategy()
    context = _context(strategy, turn)
    if obstruction == "arrival":
        context.arrival_counts[(0, -4)] = 1
    if obstruction == "departure":
        assert strategy._queue_move(
            turn.workers[0],
            Direction.LEFT,
            context,
            reason="actual departure",
            target=(-1, -4),
        )

    strategy._force_core_defense(context)

    action = turn.plan.unit_actions[UUID(int=2)]
    if obstruction in {"one_friend", "departure"}:
        assert action.type == "MOVE" and action.direction == Direction.LEFT
        assert strategy._predicted_friendly_occupancy((0, -4), context) == 2
    else:
        assert action.type == "WAIT"


def test_forced_response_follows_a_multistep_rock_detour_then_fires():
    strategy = _strategy()
    position = (6, -2)
    rocks = {(5, -2), (4, -3)}
    positions = [position]
    for tick in range(100, 112):
        turn = make_turn(
            tick=tick,
            objects=[
                core(),
                unit(2, "RANGER", position=position, hp=1),
                unit(30, "RANGER", controlled=False, position=(0, -2)),
            ],
            obstacles=rocks,
        )
        context = _context(strategy, turn)
        strategy._force_core_defense(context)
        action = turn.plan.unit_actions[UUID(int=2)]
        if action.type == "SHOOT":
            assert action.expected_cell == (0, -2)
            assert line_of_fire(position, (0, -2), rocks)
            break
        assert action.type == "MOVE"
        position = add(position, action.direction)
        assert position not in rocks
        assert position not in positions
        positions.append(position)
    else:
        pytest.fail("forced defender never reached a firing position")


@pytest.mark.parametrize("occupied", [False, True])
def test_only_firing_route_can_use_core_slot_but_never_a_third_entity(occupied):
    objects = [
        core(),
        unit(2, "RANGER", position=(0, 1)),
        unit(30, "RANGER", controlled=False, position=(0, -3)),
    ]
    if occupied:
        objects.append(unit(3, "WORKER", position=(0, 0), cargo=1))
    turn = make_turn(objects=objects, obstacles=[(-1, 1), (1, 1), (0, 2)])
    strategy = _strategy()
    strategy.memory.unit_roles[object_id(2)] = "defense"
    context = _context(strategy, turn)
    context.core_build_priority = True
    context.core_traffic_collecting = True
    context.core_migration_lane = ((0, 0),)
    context.counterattack_reserved = frozenset({(0, 0)})

    strategy._force_core_defense(context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == ("WAIT" if occupied else "MOVE")
    if not occupied:
        assert action.type == "MOVE" and action.direction is Direction.UP
        assert strategy._predicted_friendly_occupancy((0, 0), context) == 2


@pytest.mark.parametrize("occupied", [False, True])
def test_forced_route_reserves_one_slot_for_actual_core_arrival(occupied):
    objects = [
        core(
            state="MOVING",
            move_direction="RIGHT",
            destination=[1, 0],
            move_progress=3,
            move_required_ticks=4,
        ),
        unit(2, "RANGER", position=(1, 1)),
        unit(30, "RANGER", controlled=False, position=(3, 0)),
    ]
    if occupied:
        objects.append(unit(3, "WORKER", position=(1, 0), cargo=1))
    turn = make_turn(objects=objects, obstacles=[(0, 1), (2, 1), (1, 2)])
    strategy = _strategy()
    context = _context(strategy, turn)
    context.core_migration_lane = ((1, 0), (2, 0))

    strategy._force_core_defense(context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == ("WAIT" if occupied else "MOVE")
    if not occupied:
        assert action.type == "MOVE" and action.direction is Direction.UP
        assert strategy._predicted_friendly_occupancy((1, 0), context) == 1


def test_full_planner_prioritizes_forced_shot_over_healing_and_core_clearance():
    turn = make_turn(
        resources=100,
        objects=[
            core(),
            unit(2, "RANGER", hp=1),
            unit(3, "WORKER", position=(1, 0), cargo=1),
            unit(30, "RANGER", controlled=False, position=(0, -3)),
        ],
    )
    strategy = _strategy()

    strategy.decide(turn)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "SHOOT" and action.target_id == UUID(int=30)


def test_full_planner_preserves_forced_moves_through_defense_and_service():
    turn = make_turn(
        objects=[
            core(),
            *[
                unit(n, "RANGER", position=p, hp=1)
                for n, p in ((2, (1, 2)), (3, (-1, 2)), (4, (2, 1)), (5, (-2, 1)))
            ],
            unit(30, "RANGER", controlled=False, position=(0, -3)),
        ],
    )
    strategy = _strategy()

    report = strategy.decide(turn)

    assert all(a.type == "MOVE" for a in turn.plan.unit_actions.values())
    for ranger in turn.rangers:
        decisions = [d for d in report.decisions if d.actor_id == str(ranger.id)]
        assert len(decisions) == 1
        assert "current Core range" in decisions[0].reason
