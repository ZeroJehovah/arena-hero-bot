"""Bounded defensive response to an isolated wounded fighter."""

from __future__ import annotations

import pytest
from arena_hero import MoveAction

from arena_hero_bot.geometry import line_of_fire, manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import (
    DEFENSE_ROLE,
    DEFENSIVE_POST_SUPPORT_DISTANCE,
    AggressiveStrategy,
    StrategyConfig,
    _TurnContext,
)

from .factories import core, make_turn, object_id, unit

RANGER_STARTS = (
    (2, (0, -4)),
    (3, (1, -4)),
    (4, (-1, -4)),
    (5, (0, 3)),
    (6, (2, 3)),
    (7, (-2, 3)),
)


def _make_turn(tick, ranger_starts, *, target_hp=2, ranger_hp=2):
    return make_turn(
        tick=tick,
        objects=[
            core(),
            *[
                unit(number, "RANGER", position=position, hp=ranger_hp)
                for number, position in ranger_starts
            ],
            unit(50, "RANGER", controlled=False, position=(0, -9), hp=target_hp),
        ],
    )


def _context_for(strategy, turn, positions):
    assignments = {
        next(ranger.id for ranger in turn.rangers if ranger.id.int == number): position
        for number, position in positions.items()
    }
    threat = strategy._combat_policy.assess(turn, set(turn.obstacle_cells))
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={actor.position for actor in turn.units},
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
        defensive_assignments=assignments,
        layered_assignments=assignments,
        threat=threat,
    )


def _scenario(
    *,
    target_hp: int = 1,
    previous_target: tuple[int, int] = (0, -9),
    ranger_starts=RANGER_STARTS,
    ranger_hp: int = 2,
):
    positions = dict(ranger_starts)
    memory = WorldMemory(
        unit_roles={object_id(number): DEFENSE_ROLE for number, _ in ranger_starts},
        unit_roles_initialized=True,
        defense_posts={
            object_id(number): position for number, position in ranger_starts
        },
        defense_anchor=(0, 0),
        core_home_position=(0, 0),
    )
    memory.enemy_position_history[object_id(50)] = [
        (98, (0, -9)),
        (99, previous_target),
    ]
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=0,
            max_population=None,
            campaign_mode=True,
        ),
    )
    turn = _make_turn(
        100,
        ranger_starts,
        target_hp=target_hp,
        ranger_hp=ranger_hp,
    )
    strategy.memory.observe(turn)
    context = _context_for(strategy, turn, positions)
    return strategy, turn, context


def test_isolated_wounded_fighter_draws_one_short_firing_lane_response() -> None:
    strategy, turn, context = _scenario()

    strategy._prepare_local_fireline(context)

    assert len(context.local_fireline_orders) == 1
    ranger_id, (target, goal) = next(iter(context.local_fireline_orders.items()))
    ranger = next(unit_view for unit_view in turn.rangers if unit_view.id == ranger_id)
    assert ranger.id.int == 2
    assert target.id.int == 50
    assert goal == (0, -6)
    assert manhattan(ranger.position, goal) <= 2
    assert manhattan(ranger.position, goal) <= DEFENSIVE_POST_SUPPORT_DISTANCE
    assert line_of_fire(goal, target.position, set())

    strategy._decide_ranger(
        ranger,
        context,
        strategy.memory.recent_enemies(turn.tick, 32),
    )

    action = turn.plan.unit_actions[ranger.id]
    assert isinstance(action, MoveAction)
    assert action.direction.delta == (0, -1)
    assert context.report.decisions[-1].reason.startswith(
        "shift to a bounded local firing lane"
    )
    assert set(context.local_fireline_orders) == {ranger.id}


def test_moving_or_too_healthy_target_does_not_trigger_the_response() -> None:
    moving_strategy, _turn, moving_context = _scenario(previous_target=(0, -8))
    moving_strategy._prepare_local_fireline(moving_context)
    assert moving_context.local_fireline_orders == {}

    healthy_strategy, _turn, healthy_context = _scenario(target_hp=3)
    healthy_strategy._prepare_local_fireline(healthy_context)
    assert healthy_context.local_fireline_orders == {}


def test_pair_does_not_wait_with_the_leading_ranger_in_lethal_delay() -> None:
    starts = (
        (2, (0, -6)),
        (3, (0, -4)),
        (4, (2, -4)),
        (5, (-2, -4)),
        (6, (0, 3)),
        (7, (2, 3)),
    )
    positions = dict(starts)
    memory = WorldMemory(
        unit_roles={object_id(number): DEFENSE_ROLE for number, _ in starts},
        unit_roles_initialized=True,
        defense_posts={object_id(number): position for number, position in starts},
        defense_anchor=(0, 0),
        core_home_position=(0, 0),
    )
    memory.enemy_position_history[object_id(50)] = [
        (98, (0, -9)),
        (99, (0, -9)),
    ]
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            target_workers=0,
            max_population=None,
            campaign_mode=True,
        ),
    )
    turn = _make_turn(
        100,
        starts,
        target_hp=2,
        ranger_hp=2,
    )
    strategy.memory.observe(turn)
    context = _context_for(strategy, turn, positions)

    strategy._prepare_local_fireline(context)

    # The leading Ranger already has a ray, but every partner is at least two
    # steps away.  Waiting would take two hostile shots before the pair fires.
    assert context.local_fireline_orders == {}
    leading = next(ranger for ranger in turn.rangers if ranger.id.int == 2)
    strategy._decide_ranger(
        leading,
        context,
        strategy.memory.recent_enemies(turn.tick, 32),
    )
    assert turn.plan.unit_actions[leading.id].type == "SHOOT"


def test_two_hp_target_commits_a_pair_before_either_ranger_fires() -> None:
    strategy, turn, context = _scenario(
        target_hp=2,
        ranger_starts=((2, (0, -6)), (3, (0, -5)), *RANGER_STARTS[2:]),
    )

    strategy._prepare_local_fireline(context)

    assert len(context.local_fireline_orders) == 2
    leading = next(ranger for ranger in turn.rangers if ranger.id.int == 2)
    assert manhattan(leading.position, (0, -9)) == 3

    # The first ranger reaches the shared firing cell before its partner.  It
    # must hold instead of starting a one-arrow exchange that cannot kill a
    # two-HP Ranger and would leave it exposed alone.
    strategy._decide_ranger(
        leading,
        context,
        strategy.memory.recent_enemies(turn.tick, 32),
    )

    assert turn.plan.unit_actions[leading.id].type == "WAIT"
    assert context.report.decisions[-1].reason == (
        "hold for the bounded local firing pair to line up"
    )

    one_shifted = ((2, (0, -6)), (3, (0, -5)), *RANGER_STARTS[2:])
    turn = _make_turn(101, one_shifted, target_hp=2)
    strategy.memory.observe(turn)
    context = _context_for(strategy, turn, dict(one_shifted))
    strategy._prepare_local_fireline(context)
    trailing = next(ranger for ranger in turn.rangers if ranger.id.int == 3)
    strategy._decide_ranger(
        trailing,
        context,
        strategy.memory.recent_enemies(turn.tick, 32),
    )
    assert turn.plan.unit_actions[trailing.id].type == "MOVE"

    aligned = ((2, (0, -6)), (3, (0, -6)), *RANGER_STARTS[2:])
    turn = _make_turn(102, aligned, target_hp=2)
    strategy.memory.observe(turn)
    context = _context_for(strategy, turn, dict(aligned))
    strategy._prepare_local_fireline(context)
    assert context.local_fireline_orders == {}
    leading = next(ranger for ranger in turn.rangers if ranger.id.int == 2)
    strategy._decide_ranger(
        leading,
        context,
        strategy.memory.recent_enemies(turn.tick, 32),
    )
    assert turn.plan.unit_actions[leading.id].type == "SHOOT"


@pytest.mark.parametrize("partner, viable", [((0, -4), False), ((1, -6), True)])
def test_pair_accounts_for_fire_on_the_first_arrival_tick(partner, viable):
    strategy, turn, context = _scenario(
        target_hp=2,
        ranger_starts=((2, (0, -5)), (3, partner), *RANGER_STARTS[2:]),
    )

    strategy._prepare_local_fireline(context)

    # A one-step arrival takes a shot on ingress. Waiting for a two-step
    # partner costs another shot before the pair can shoot, exhausting HP2.
    # Simultaneous one-step arrivals survive until their shared volley.
    assert bool(context.local_fireline_orders) is viable
    if viable:
        assert len(context.local_fireline_orders) == 2
        for actor_id, (_target, goal) in context.local_fireline_orders.items():
            ranger = next(actor for actor in turn.rangers if actor.id == actor_id)
            assert manhattan(ranger.position, goal) == 1
            assert ranger.hp - 1 > 0
