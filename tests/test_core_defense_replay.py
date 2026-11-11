"""Regression coverage from the 2026-10-27 Core-defense collapse."""

from uuid import UUID

from arena_hero import MoveAction, ShootAction, UnitType

from arena_hero_bot.combat_policy import ThreatAssessment, ThreatLevel
from arena_hero_bot.geometry import add, manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import (
    DEFENSE_ROLE,
    WORKER_EMERGENCY_STANDOFF_RADIUS,
    AggressiveStrategy,
    StrategyConfig,
    _TurnContext,
)

from .factories import core, make_turn, object_id, unit
from .test_layered_defense import Encounter


def _strategy(
    memory: WorldMemory | None = None,
    *,
    campaign_mode: bool = False,
    expedition_mode: bool = False,
) -> AggressiveStrategy:
    return AggressiveStrategy(
        memory or WorldMemory(),
        StrategyConfig(
            target_workers=0,
            max_population=None,
            campaign_mode=campaign_mode,
            expedition_mode=expedition_mode,
        ),
    )


def _combat_context(
    strategy: AggressiveStrategy,
    turn,
    posts: dict[int, tuple[int, int]] | None = None,
    *,
    projected_core_damage: int = 0,
) -> _TurnContext:
    posts = posts or {}
    assignments = {
        ranger.id: posts.get(ranger.id.int, ranger.position) for ranger in turn.rangers
    }
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={item.position for item in (*turn.units, *turn.visible_enemies)},
        enemy_positions={item.position for item in turn.visible_enemies},
        threat=ThreatAssessment(
            level=ThreatLevel.ENGAGED,
            reason="CORE_ATTACK",
            projected_core_damage=projected_core_damage,
        ),
        defensive_assignments=assignments,
        combat_assault=True,
        all_hands_defense=True,
    )


def test_core_heals_before_projected_damage_can_become_lethal() -> None:
    turn = make_turn(
        resources=20,
        objects=[
            core(hp=3, shield=1),
            unit(30, "RANGER", controlled=False, position=(3, 0)),
            unit(31, "RANGER", controlled=False, position=(-3, 0)),
            unit(32, "RANGER", controlled=False, position=(0, 3)),
        ],
    )

    report = _strategy().decide(turn)

    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "HEAL"
    assert any(
        item.reason == "heal before the projected strike becomes lethal"
        for item in report.decisions
    )


def test_core_assault_prefers_a_current_enemy_cell_over_a_one_cell_lead() -> None:
    memory = WorldMemory(
        unit_roles={object_id(2): DEFENSE_ROLE},
        unit_roles_initialized=True,
    )
    memory.enemy_position_history[object_id(30)] = [
        (98, (1, -1)),
        (99, (1, 0)),
        (100, (1, 1)),
    ]
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 1)),
            unit(30, "RANGER", controlled=False, position=(1, 1)),
        ]
    )
    strategy = _strategy(memory)
    memory.observe(turn)
    context = _combat_context(strategy, turn, projected_core_damage=1)

    strategy._decide_ranger(
        turn.rangers[0],
        context,
        strategy.memory.recent_enemies(turn.tick, 32),
    )

    action = turn.plan.unit_actions[UUID(int=2)]
    assert isinstance(action, ShootAction)
    assert action.target_id == UUID(int=30)
    assert action.expected_cell == (1, 1)


def test_defender_repositions_before_firing_only_a_predicted_cell() -> None:
    memory = WorldMemory(
        unit_roles={object_id(2): DEFENSE_ROLE, object_id(3): DEFENSE_ROLE},
        unit_roles_initialized=True,
    )
    memory.enemy_position_history[object_id(30)] = [
        (98, (1, 5)),
        (99, (1, 4)),
        (100, (1, 3)),
    ]
    encounter = Encounter(
        [unit(2, "RANGER", position=(0, 1)), unit(3, "RANGER", position=(2, 2))],
        [unit(30, "RANGER", controlled=False, position=(1, 3))],
        resources=0,
    )
    turn = encounter.turn()
    strategy = _strategy(memory)
    memory.observe(turn)
    context = _combat_context(strategy, turn)

    # Moving cannot fire: the ally must really take the first arrow off the
    # target, rather than leave this defender in a losing 1HP-v-2HP reply.
    strategy._decide_ranger(
        turn.rangers[1], context, strategy.memory.recent_enemies(turn.tick, 32)
    )
    ally_action = turn.plan.unit_actions[UUID(int=3)]
    assert isinstance(ally_action, ShootAction)
    assert ally_action.target_id == UUID(int=30)
    strategy._decide_ranger(
        turn.rangers[0],
        context,
        strategy.memory.recent_enemies(turn.tick, 32),
    )

    action = turn.plan.unit_actions[UUID(int=2)]
    assert isinstance(action, MoveAction)
    assert add((0, 1), action.direction) == (1, 1)
    assert "clear firing lane" in context.report.decisions[-1].reason

    encounter.settle(
        turn.plan,
        {
            object_id(30): {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [1, 1],
            }
        },
    )
    assert encounter.friends[object_id(2)]["hp"] == 1
    assert encounter.foes[object_id(30)]["hp"] == 1
    reply = encounter.turn()
    memory.observe(reply)
    reply_context = _combat_context(strategy, reply)
    for ranger in reply.rangers:
        strategy._decide_ranger(
            ranger, reply_context, memory.recent_enemies(reply.tick, 32)
        )
    encounter.settle(
        reply.plan,
        {
            object_id(30): {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [1, 1],
            }
        },
    )
    assert (encounter.kills, encounter.losses) == (1, 0)


def test_overlapping_defenders_keep_their_direct_shots_during_combat() -> None:
    memory = WorldMemory(
        unit_roles={object_id(2): DEFENSE_ROLE, object_id(3): DEFENSE_ROLE},
        unit_roles_initialized=True,
    )
    memory.enemy_position_history[object_id(30)] = [
        (98, (1, -1)),
        (99, (1, 0)),
        (100, (1, 1)),
    ]
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 1)),
            unit(3, "RANGER", position=(0, 1)),
            unit(30, "RANGER", controlled=False, position=(1, 1)),
        ]
    )
    strategy = _strategy(memory)
    memory.observe(turn)
    context = _combat_context(strategy, turn)

    for ranger in turn.rangers:
        strategy._decide_ranger(
            ranger,
            context,
            strategy.memory.recent_enemies(turn.tick, 32),
        )

    actions = [turn.plan.unit_actions[UUID(int=number)] for number in (2, 3)]
    assert all(isinstance(action, ShootAction) for action in actions)
    assert all(
        isinstance(action, ShootAction) and action.expected_cell == (1, 1)
        for action in actions
    )


def test_workers_in_an_assault_retreat_to_the_outer_standoff_ring() -> None:
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(2, 0)),
            unit(30, "RANGER", controlled=False, position=(3, 0)),
        ]
    )
    strategy = _strategy()
    context = _combat_context(strategy, turn)

    goal = strategy._emergency_worker_goal(turn.workers[0], context)

    assert manhattan((0, 0), goal) == WORKER_EMERGENCY_STANDOFF_RADIUS
    assert manhattan(goal, (3, 0)) > manhattan((2, 0), (3, 0))


def test_local_assault_never_spends_the_core_action_on_a_staged_worker() -> None:
    turn = make_turn(
        resources=100,
        objects=[
            core(),
            unit(2, "WORKER", position=(0, 2)),
            unit(30, "VANGUARD", controlled=False, position=(5, 0)),
            unit(31, "VANGUARD", controlled=False, position=(5, 1)),
            unit(32, "VANGUARD", controlled=False, position=(5, 2)),
        ],
    )
    strategy = _strategy(campaign_mode=True, expedition_mode=True)

    assert strategy._choose_spawn(turn, 100) is UnitType.RANGER
