"""Combat volleys, stable reinforcements and sheltered economy activity."""

from uuid import UUID

import pytest

from arena_hero_bot.combat_policy import ThreatAssessment, ThreatLevel
from arena_hero_bot.geometry import add, adjacent_positions, manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig, _TurnContext

from .factories import core, make_turn, object_id, unit


def _context(turn, *, roles=None, campaign=False):
    memory = WorldMemory(unit_roles=roles or {}, unit_roles_initialized=True)
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(target_workers=0, max_population=None, campaign_mode=campaign),
    )
    memory.observe(turn)
    occupancy = {}
    for actor in (*turn.units, turn.core):
        occupancy[actor.position] = occupancy.get(actor.position, 0) + 1
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied=set(occupancy),
        friendly_occupancy=occupancy,
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
        threat=ThreatAssessment(level=ThreatLevel.ENGAGED, recent_core_attack=True),
        combat_assault=True,
        assault_enemies=tuple(turn.visible_enemies),
        remaining_resources=turn.resources,
        remaining_resource_space=turn.resource_space,
        defensive_assignments={
            r.id: r.position
            for r in turn.rangers
            if roles and roles.get(str(r.id)) == "defense"
        },
    )
    return strategy, context


def _ranger_action(strategy, context, ranger):
    strategy._decide_ranger(
        ranger, context, strategy.memory.recent_enemies(context.turn.tick, 32)
    )
    return context.turn.plan.unit_actions[ranger.id]


def test_recalled_patrol_shoots_before_returning_to_its_post():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 5)),
            unit(30, "RANGER", controlled=False, position=(0, 8)),
        ]
    )
    strategy, context = _context(turn, roles={object_id(2): "patrol-1"}, campaign=True)
    context.campaign_patrol_recall_ids = frozenset({UUID(int=2)})

    assert _ranger_action(strategy, context, turn.rangers[0]).type == "SHOOT"


@pytest.mark.parametrize("enemy_hp, expected", [(1, "SHOOT"), (2, "MOVE")])
def test_last_hp_ranger_accepts_mutual_kill_but_avoids_one_sided_death(
    enemy_hp, expected
):
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 5), hp=1),
            unit(30, "RANGER", controlled=False, position=(0, 8), hp=enemy_hp),
        ]
    )
    strategy, context = _context(turn, roles={object_id(2): "defense"})

    assert _ranger_action(strategy, context, turn.rangers[0]).type == expected


def test_three_full_rangers_volley_a_stacked_pair_instead_of_all_withdrawing():
    # Each hostile has only one arrow. Two hostile arrows can kill at most
    # one full-health defender, even though both can aim at all three.
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(-5, -5)),
            unit(3, "RANGER", position=(0, -3)),
            unit(4, "RANGER", position=(-2, -2)),
            unit(30, "RANGER", controlled=False, position=(-2, -5)),
            unit(31, "RANGER", controlled=False, position=(-2, -5), hp=1),
        ]
    )
    strategy, context = _context(
        turn, roles={object_id(n): "defense" for n in (2, 3, 4)}
    )

    actions = [_ranger_action(strategy, context, ranger) for ranger in turn.rangers]

    assert all(action.type == "SHOOT" for action in actions)
    assert all(action.expected_cell == (-2, -5) for action in actions)
    assert all(action.target_id == UUID(int=31) for action in actions)


def test_existing_direct_shots_win_over_friendly_stack_spreading():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 1)),
            unit(3, "RANGER", position=(0, 1)),
            unit(30, "RANGER", controlled=False, position=(1, 1)),
        ]
    )
    strategy, context = _context(turn, roles={object_id(n): "defense" for n in (2, 3)})

    assert all(
        _ranger_action(strategy, context, r).type == "SHOOT" for r in turn.rangers
    )


def test_enemy_core_in_the_target_cell_prevents_a_false_guaranteed_kill():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 5), hp=1),
            core(29, controlled=False, position=(0, 8), hp=1),
            unit(30, "RANGER", controlled=False, position=(0, 8), hp=1),
        ]
    )
    strategy, context = _context(turn, roles={object_id(2): "defense"})

    strategy._prepare_defensive_trade_shots(context)

    assert context.defensive_trade_shots == {}


def test_healthy_ranger_leaves_core_without_a_service_request():
    turn = make_turn(
        objects=[core(), unit(2, "RANGER"), unit(3, "RANGER", position=(1, 1))]
    )
    strategy, context = _context(turn, roles={object_id(2): "patrol-1"})

    strategy._prepare_core_departure(context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "MOVE"
    assert add((0, 0), action.direction) != (0, 0)


def test_wounded_defender_waits_in_a_safe_unique_healing_queue_slot():
    turn = make_turn(
        resources=5,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 1), hp=1),
            unit(3, "RANGER", hp=1),
        ],
    )
    strategy, context = _context(
        turn, roles={object_id(2): "defense", object_id(3): "defense"}
    )

    action = _ranger_action(strategy, context, turn.rangers[0])

    assert action.type == "WAIT"
    assert "queue" in context.report.decisions[-1].reason


@pytest.mark.parametrize("funds,service", [(0, None), (5, "DEPOSIT"), (5, "HEAL")])
def test_wounded_ranger_stays_outside_core_until_a_funded_heal_is_available(
    funds, service
):
    turn = make_turn(
        resources=funds, objects=[core(), unit(2, "RANGER", position=(0, 1), hp=1)]
    )
    strategy, context = _context(turn, roles={object_id(2): "defense"})
    context.core_service_action = service

    action = _ranger_action(strategy, context, turn.rangers[0])

    assert action.type == "WAIT"


def test_reinforcements_favor_north_and_east_and_hold_after_arrival(tmp_path):
    roles = {object_id(n): "patrol-1" for n in range(2, 26)}
    enemies = [
        unit(30, "RANGER", controlled=False, position=(0, -10)),
        unit(31, "RANGER", controlled=False, position=(10, 0)),
    ]
    turn = make_turn(
        objects=[
            core(),
            *[unit(n, "RANGER", position=(-3, 4)) for n in range(2, 26)],
            *enemies,
        ]
    )
    strategy, context = _context(turn, roles=roles, campaign=True)
    strategy._assign_reinforcement_posts(context)
    posts = dict(context.reinforcement_assignments)

    assert len(set(posts.values())) == len(posts) == 24
    assert sum((x > 0 if abs(x) > abs(y) else y < 0) for x, y in posts.values()) >= 18
    assert all(
        not strategy._visible_attackers_at(p, turn, set()) for p in posts.values()
    )
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    restarted = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    for tick in (101, 102, 103):
        arrived = make_turn(
            tick=tick,
            objects=[
                core(),
                *[unit(n, "RANGER", position=posts[UUID(int=n)]) for n in range(2, 26)],
                *enemies,
            ],
        )
        _, next_context = _context(arrived, roles=roles, campaign=True)
        restarted.memory.observe(arrived)
        next_context.campaign_patrol_recall_ids = frozenset(posts)
        for ranger in arrived.rangers:
            assert _ranger_action(restarted, next_context, ranger).type != "MOVE"
        assert next_context.reinforcement_assignments == posts


@pytest.mark.parametrize(
    "position,cargo,hp,expected",
    [
        ((0, 2), 0, 2, "HARVEST"),
        ((0, 2), 1, 2, "MOVE"),
        ((0, 2), 0, 1, "MOVE"),
        ((0, 0), 1, 2, "DEPOSIT"),
        ((0, 0), 0, 1, "HEAL"),
    ],
)
def test_sheltered_workers_operate_during_a_core_assault(position, cargo, hp, expected):
    turn = make_turn(
        resources=5,
        objects=[
            core(),
            unit(2, "WORKER", position=position, cargo=cargo, hp=hp),
            unit(3, "RANGER", position=(-1, 1)),
            unit(30, "RANGER", controlled=False, position=(4, -1)),
        ],
        resource_cells=[(0, 2)] if not cargo and hp == 2 else [],
    )
    strategy, context = _context(turn)
    if turn.resource_cells:
        context.resource_assignments[UUID(int=2)] = (0, 2)

    strategy._decide_worker(turn.workers[0], context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == expected
    if action.type == "MOVE":
        destination = add(position, action.direction)
        assert manhattan(destination, (0, 0)) < manhattan(position, (0, 0))
        assert not strategy._visible_attackers_at(destination, turn, set())


def test_worker_does_not_harvest_inside_a_real_enemy_ray_even_with_cover():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "WORKER", position=(1, -1)),
            unit(3, "RANGER", position=(0, -1)),
            unit(30, "RANGER", controlled=False, position=(4, -1)),
        ],
        resource_cells=[(1, -1)],
    )
    strategy, context = _context(turn)
    context.resource_assignments[UUID(int=2)] = (1, -1)

    strategy._decide_worker(turn.workers[0], context)

    assert turn.plan.unit_actions[UUID(int=2)].type != "HARVEST"
    assert (1, -1) in strategy._worker_combat_threat_exclusion_cells(turn)


def test_visible_enemy_adjacent_cells_outrank_old_contacts_and_frontier():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 6)),
            unit(30, "RANGER", controlled=False, position=(1, 9)),
        ]
    )
    strategy, context = _context(turn, roles={object_id(2): "defense"})
    old = make_turn(
        tick=99,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 6)),
            unit(31, "RANGER", controlled=False, position=(-3, 6)),
        ],
    )
    other_memory = WorldMemory()
    other_memory.observe(old)

    assert strategy._defensive_cover_fire(
        turn.rangers[0],
        context,
        other_memory.recent_enemies(turn.tick, 32),
        frontier=True,
    )

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "SHOOT"
    assert action.expected_cell in adjacent_positions((1, 9))
    assert "visible enemy" in context.report.decisions[-1].reason


def test_stationary_visible_fighter_still_has_priority_over_the_frontier():
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 6)),
            unit(30, "RANGER", controlled=False, position=(1, 9)),
        ]
    )
    strategy, context = _context(turn, roles={object_id(2): "defense"})
    strategy.memory.enemy_position_history[object_id(30)] = [
        (98, (1, 9)),
        (99, (1, 9)),
        (100, (1, 9)),
    ]

    strategy._hold_defensive_ranger(turn.rangers[0], context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "SHOOT"
    assert action.expected_cell in adjacent_positions((1, 9))
