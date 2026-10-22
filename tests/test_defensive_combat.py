"""Defensive combat fixes drawn from the 2026-10-21 Core-loss replay."""

from uuid import UUID

import pytest
from arena_hero import MoveAction, UnitView

from arena_hero_bot.combat_policy import ThreatAssessment, ThreatLevel
from arena_hero_bot.geometry import add, line_of_fire, manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import (
    DEFENSE_ROLE,
    AggressiveStrategy,
    StrategyConfig,
    _TurnContext,
)

from .factories import core, make_turn, object_id, unit


def _strategy(memory: WorldMemory | None = None) -> AggressiveStrategy:
    return AggressiveStrategy(
        memory or WorldMemory(),
        StrategyConfig(target_workers=0, max_population=None),
    )


def _decisions_by_number(report: DecisionReport) -> dict[int, str]:
    lookup = {item.actor_id: item.reason for item in report.decisions}
    return {
        number: lookup[object_id(number)]
        for number in range(1, 200)
        if object_id(number) in lookup
    }


def test_a_volley_hedges_across_the_targets_possible_next_cells() -> None:
    # Two shooters see the same legal cells: the predicted lead (3, 4) and
    # the cell the target would step into on its way to our Core (1, 4).
    # Once the first shooter covers the cell that can kill the target, the
    # second must not stack on the same prediction: live logs show every
    # shooter of a volley aiming at one cell, so a wrong prediction wasted
    # the whole volley even when an alternative cell was available.
    memory = WorldMemory()
    rangers = [
        unit(2, "RANGER", position=(0, 4)),
        unit(5, "RANGER", position=(4, 4)),
    ]
    for ranger in rangers:
        memory.unit_roles[ranger["id"]] = DEFENSE_ROLE
    enemy_number = 30
    enemy_id = object_id(enemy_number)
    memory.enemy_position_history[enemy_id] = [
        (98, (0, 4)),
        (99, (1, 4)),
        (100, (2, 4)),
    ]
    turn = make_turn(
        objects=[
            core(),
            *rangers,
            unit(enemy_number, "RANGER", controlled=False, position=(2, 4), hp=1),
        ],
    )
    strategy = _strategy(memory)

    report = strategy.decide(turn)

    aims = {}
    for number in (2, 5):
        action = turn.plan.unit_actions[UUID(int=number)]
        assert action.type == "SHOOT"
        aims[number] = action.expected_cell
    assert aims[2] == (3, 4)
    assert aims[5] == (1, 4)
    assert all(
        line_of_fire(position, aim, memory.obstacles)
        for position, aim in (
            (rangers[0]["position"], aims[2]),
            (rangers[1]["position"], aims[5]),
        )
    )
    reasons = _decisions_by_number(report)
    assert "covering a possible approach cell" in reasons[5]


def test_a_far_side_defender_closes_on_a_massed_assault() -> None:
    # The Core-loss replay at Tick 475541 left two Rangers seventeen cells
    # away holding the ring: every range-one-to-three cell around the enemy
    # formation was already covered, so no firing position existed and the
    # static post was the idle fallback.  All-hands defence must at least
    # walk them towards the fight until their next step would enter fire.
    turn = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 17)),
            *[
                unit(number, "RANGER", controlled=False, position=position)
                for number, position in (
                    (30, (0, 3)),
                    (31, (1, 3)),
                    (32, (-1, 3)),
                    (33, (0, 4)),
                    (34, (1, 4)),
                    (35, (-1, 4)),
                )
            ],
        ],
    )
    strategy = _strategy()
    report = strategy.decide(turn)

    reasons = _decisions_by_number(report)
    assert "close" in reasons[2]
    assert "perimeter" not in reasons[2]
    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "MOVE"
    destination = add((0, 17), action.direction)
    assert manhattan(destination, (0, 3)) < manhattan((0, 17), (0, 3))
    assert all(
        not strategy._enemy_can_attack_position(enemy, destination, set())
        for enemy in turn.visible_enemies
        if isinstance(enemy, UnitView)
    )


def test_a_far_side_defender_holds_when_no_massed_assault_exists() -> None:
    # Two contacts keep the far side on station; only a real mass release it.
    turn = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 17)),
            unit(30, "RANGER", controlled=False, position=(0, 3)),
            unit(31, "RANGER", controlled=False, position=(1, 3)),
        ],
    )
    strategy = _strategy()

    report = strategy.decide(turn)

    reasons = _decisions_by_number(report)
    assert reasons[2] != "close on the massed assault"
    assert reasons[2] == "hold a defensive perimeter around the resource Core" or (
        "hunt" in reasons[2]
    )


def test_a_wounded_ranger_fires_a_safe_shot_instead_of_withdrawing() -> None:
    # A one-HP Ranger used to run home for healing as soon as the threat
    # level rose, even against a Vanguard two cells away that could not hit
    # it back.  The firing branch already rejects a shot that would take
    # survivable return fire, so a legal safe shot must win.
    turn = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "RANGER", position=(5, 0), hp=1),
            unit(30, "VANGUARD", controlled=False, position=(8, 0)),
            unit(31, "RANGER", controlled=False, position=(0, 2)),
        ],
    )
    strategy = _strategy()
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={item.position for item in turn.units},
        enemy_positions={item.position for item in turn.visible_enemies},
        threat=ThreatAssessment(
            level=ThreatLevel.ENGAGED,
            reason="CORE_ATTACK",
        ),
    )

    ranger = turn.rangers[0]
    strategy._decide_ranger(ranger, context, ())

    decision = next(
        item for item in context.report.decisions if item.actor_id == str(ranger.id)
    )
    assert decision.action == "SHOOT"
    assert turn.plan.unit_actions[ranger.id].type == "SHOOT"


def test_wounded_ranger_still_withdraws_when_the_target_can_shoot_back() -> None:
    turn = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "RANGER", position=(5, 0), hp=1),
            unit(30, "RANGER", controlled=False, position=(8, 0)),
            unit(31, "RANGER", controlled=False, position=(0, 2)),
        ],
    )
    strategy = _strategy()
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={item.position for item in turn.units},
        enemy_positions={item.position for item in turn.visible_enemies},
        threat=ThreatAssessment(
            level=ThreatLevel.ENGAGED,
            reason="CORE_ATTACK",
        ),
    )

    ranger = turn.rangers[0]
    strategy._decide_ranger(ranger, context, ())

    decision = next(
        item for item in context.report.decisions if item.actor_id == str(ranger.id)
    )
    assert decision.action != "SHOOT"
    assert "healing" in decision.reason


def test_massed_assault_never_walks_a_defender_into_fire() -> None:
    # Closing the distance must stop at the hostile coverage edge even when
    # the anchor sits at the far side of a firing line.
    turn = make_turn(
        resources=0,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 8)),
            *[
                unit(number, "RANGER", controlled=False, position=position)
                for number, position in (
                    (30, (0, 3)),
                    (31, (1, 3)),
                    (32, (-1, 3)),
                )
            ],
        ],
    )
    strategy = _strategy()
    strategy.decide(turn)

    action = turn.plan.unit_actions[UUID(int=2)]
    ranger = next(item for item in turn.units if item.id == UUID(int=2))
    if action.type == "MOVE":
        destination = add(ranger.position, action.direction)
        assert all(
            not strategy._enemy_can_attack_position(enemy, destination, set())
            for enemy in turn.visible_enemies
            if isinstance(enemy, UnitView)
        )
        assert manhattan(destination, (0, 3)) <= manhattan(ranger.position, (0, 3))


def test_massed_assault_fallback_closes_when_no_firing_cell_exists() -> None:
    # The approach helper can find no legal firing cell at all when the
    # hostile formation covers every range-one-to-three cell around the
    # target.  The fallback must still walk the defender in rather than let
    # it hold a static post on the far side of the ring.
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 16)),
            *[
                unit(number, "RANGER", controlled=False, position=position)
                for number, position in (
                    (30, (0, 3)),
                    (31, (1, 3)),
                    (32, (-1, 3)),
                    (33, (0, 2)),
                    (34, (1, 2)),
                    (35, (-1, 2)),
                )
            ],
        ],
    )
    strategy = _strategy()
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={item.position for item in turn.units},
        enemy_positions={item.position for item in turn.visible_enemies},
        all_hands_defense=True,
    )
    ranger = turn.rangers[0]

    moved = strategy._close_on_massed_assault(
        ranger,
        tuple(turn.visible_enemies),
        context,
    )

    assert moved
    action = turn.plan.unit_actions[ranger.id]
    assert action.type == "MOVE"
    destination = add(ranger.position, action.direction)
    assert manhattan(destination, (0, 3)) < manhattan(ranger.position, (0, 3))
    assert all(
        not strategy._enemy_can_attack_position(enemy, destination, set())
        for enemy in turn.visible_enemies
        if isinstance(enemy, UnitView)
    )


def test_massed_assault_fallback_is_inert_without_the_shared_picture() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 16)),
            *[
                unit(number, "RANGER", controlled=False, position=position)
                for number, position in (
                    (30, (0, 3)),
                    (31, (1, 3)),
                    (32, (-1, 3)),
                )
            ],
        ],
    )
    strategy = _strategy()
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={item.position for item in turn.units},
        enemy_positions={item.position for item in turn.visible_enemies},
    )

    assert not strategy._close_on_massed_assault(
        turn.rangers[0],
        tuple(turn.visible_enemies),
        context,
    )


def test_killing_one_of_two_shooters_does_not_cancel_its_return_fire() -> None:
    # Both attacks resolve even when the one-HP target dies in this volley.
    turn = make_turn(
        objects=[
            core(position=(10, 10)),
            unit(2, "RANGER"),
            unit(30, "RANGER", controlled=False, position=(0, 2), hp=1),
            unit(31, "RANGER", controlled=False, position=(0, 3)),
        ],
    )
    strategy = _strategy()

    report = strategy.decide(turn)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "MOVE"
    destination = add((0, 0), action.direction)
    assert len(strategy._visible_attackers_at(destination, turn, set())) < 2
    assert "lethal multi-source" in _decisions_by_number(report)[2]


def test_straight_moving_shooters_are_counted_before_they_enter_range() -> None:
    memory = WorldMemory()
    for number in (30, 31):
        memory.enemy_position_history[object_id(number)] = [
            (98, (0, 6)),
            (99, (0, 5)),
            (100, (0, 4)),
        ]
    turn = make_turn(
        objects=[
            core(position=(10, 10)),
            unit(2, "RANGER"),
            unit(30, "RANGER", controlled=False, position=(0, 4)),
            unit(31, "RANGER", controlled=False, position=(0, 4)),
        ],
    )
    strategy = _strategy(memory)

    attackers = strategy._visible_attackers_at((0, 0), turn, set())

    assert len(attackers) == 2
    assert {str(enemy.id) for enemy in attackers} == {
        object_id(30),
        object_id(31),
    }
    assert not strategy._visible_attackers_at((0, 0), turn, {(0, 3)})


def test_defender_routes_around_an_unsafe_first_step() -> None:
    # Going right is the shortest path but exposes a fresh enemy ray. Going
    # down reaches the same overwatch goal safely instead of returning WAIT.
    turn = make_turn(
        objects=[
            core(position=(0, 10)),
            unit(2, "RANGER"),
            unit(30, "RANGER", controlled=False, position=(1, 2)),
        ],
    )
    strategy = _strategy()
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={item.position for item in turn.units},
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
    )

    assert strategy._move(
        turn.rangers[0],
        (3, 0),
        context,
        reason="join the defensive firing line",
        avoid_new_threats=True,
    )
    action = turn.plan.unit_actions[UUID(int=2)]
    assert isinstance(action, MoveAction)
    destination = add((0, 0), action.direction)
    assert destination == (0, -1)
    assert not strategy._visible_attackers_at(destination, turn, set())


def test_retreat_prefers_a_safe_ray_over_distance_to_core() -> None:
    # The old non-closing distance rule selected (1, 0), still exposed to
    # both stacked shooters. (0, -1) leaves their range in one step.
    turn = make_turn(
        objects=[
            core(position=(10, 0)),
            unit(2, "RANGER"),
            unit(30, "RANGER", controlled=False, position=(0, 3)),
            unit(31, "RANGER", controlled=False, position=(0, 3)),
        ],
    )
    strategy = _strategy()
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={item.position for item in turn.units},
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
    )

    assert strategy._move_ranger_toward_core(
        turn.rangers[0], context, reason="leave lethal fire"
    )
    action = turn.plan.unit_actions[UUID(int=2)]
    assert isinstance(action, MoveAction)
    destination = add((0, 0), action.direction)
    assert not strategy._visible_attackers_at(destination, turn, set())


def test_defender_fires_when_every_escape_is_equally_lethal() -> None:
    turn = make_turn(
        objects=[
            core(position=(10, 10)),
            unit(2, "RANGER"),
            unit(30, "RANGER", controlled=False, position=(1, 1), hp=1),
            unit(31, "RANGER", controlled=False, position=(2, 2)),
        ],
        obstacles=[(-1, 0), (1, 0), (0, -1), (0, 1)],
    )
    strategy = _strategy()

    strategy.decide(turn)

    assert turn.plan.unit_actions[UUID(int=2)].type == "SHOOT"


@pytest.mark.parametrize("end", [(2, 4), (1, 4), (2, 3), (2, 5)])
def test_defensive_aim_covers_stops_reversals_and_turns(end) -> None:
    memory = WorldMemory()
    memory.unit_roles[object_id(2)] = DEFENSE_ROLE
    memory.enemy_position_history[object_id(30)] = [
        (98, (0, 4)),
        (99, (1, 4)),
        (100, (2, 4)),
    ]
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(end[0] - 2, end[1] - 2)),
            unit(30, "RANGER", controlled=False, position=(2, 4)),
        ],
    )
    strategy = _strategy(memory)

    cells = strategy._ranger_shot_cells(
        turn.rangers[0], turn.visible_enemies[0], turn, set()
    )

    assert end in cells
    assert all(manhattan(cell, (2, 4)) <= 1 for cell in cells)
    assert end not in strategy._ranger_shot_cells(
        turn.rangers[0], turn.visible_enemies[0], turn, {end}
    )
    assert end not in strategy._ranger_shot_cells(
        turn.rangers[0], turn.visible_enemies[0], turn, set(), {end}
    )


def test_flickering_diagonal_drift_cannot_aim_two_steps_ahead() -> None:
    memory = WorldMemory()
    memory.unit_roles[object_id(2)] = DEFENSE_ROLE
    memory.enemy_position_history[object_id(30)] = [
        (98, (0, 2)),
        (100, (2, 4)),
    ]
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 2)),
            unit(30, "RANGER", controlled=False, position=(2, 4)),
        ],
    )
    strategy = _strategy(memory)

    cells = strategy._ranger_shot_cells(
        turn.rangers[0], turn.visible_enemies[0], turn, set()
    )

    assert cells
    assert (3, 5) not in cells
    assert all(manhattan(cell, (2, 4)) <= 1 for cell in cells)


def test_massed_defense_keeps_a_survivable_shot_instead_of_waiting_for_a_volley():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 6)),
            unit(30, "RANGER", controlled=False, position=(0, 9)),
            unit(31, "RANGER", controlled=False, position=(8, 0)),
            unit(32, "RANGER", controlled=False, position=(9, 0)),
        ],
    )
    strategy = _strategy()

    strategy.decide(turn)

    assert turn.plan.unit_actions[UUID(int=2)].type == "SHOOT"


def test_single_cell_shooter_counts_toward_the_next_shooters_aim_quota():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER"),
            unit(30, "RANGER", controlled=False, position=(2, 0), hp=1),
        ],
    )
    strategy = _strategy()
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={item.position for item in turn.units},
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
    )
    target = turn.visible_enemies[0]

    assert strategy._volley_aim_cell(context, target, ((2, 0),)) == (2, 0)
    assert strategy._volley_aim_cell(context, target, ((2, 0), (1, 0))) == (1, 0)


def test_alternate_aim_receives_enough_arrows_to_kill_a_full_health_target():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER"),
            unit(30, "RANGER", controlled=False, position=(2, 0)),
        ],
    )
    strategy = _strategy()
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={item.position for item in turn.units},
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
    )
    target = turn.visible_enemies[0]

    aims = [
        strategy._volley_aim_cell(context, target, ((2, 0), (1, 0))) for _ in range(4)
    ]

    assert aims == [(2, 0), (2, 0), (1, 0), (1, 0)]
