"""Holding fire, layered withdrawals, and bounded patrol engagements."""

from uuid import UUID

import pytest
from arena_hero import MoveAction, ShootAction

from arena_hero_bot.geometry import add, line_of_fire, manhattan
from arena_hero_bot.memory import PatrolContact, WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import (
    DEFENSIVE_POST_SUPPORT_DISTANCE,
    PATROL_PURSUIT_DISTANCE,
    PATROL_PURSUIT_TICKS,
    AggressiveStrategy,
    StrategyConfig,
    _TurnContext,
)

from .factories import core, make_turn, object_id, unit


def _scenario(rangers, enemies=(), *, tick=100, obstacles=(), roles=None):
    roles = roles or {number: "defense" for number, _, _ in rangers}
    memory = WorldMemory(
        unit_roles={object_id(number): role for number, role in roles.items()},
        unit_roles_initialized=True,
        defense_posts={
            object_id(number): position
            for number, position, _ in rangers
            if roles[number] == "defense"
        },
        defense_anchor=(0, 0),
    )
    strategy = AggressiveStrategy(
        memory, StrategyConfig(target_workers=0, max_population=None)
    )
    turn = make_turn(
        tick=tick,
        objects=[
            core(),
            *[
                unit(number, "RANGER", position=position, hp=hp)
                for number, position, hp in rangers
            ],
            *[
                unit(number, kind, controlled=False, position=position, hp=hp)
                for number, kind, position, hp in enemies
            ],
        ],
        obstacles=obstacles,
    )
    memory.observe(turn)
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=tick),
        occupied={item.position for item in turn.units},
        enemy_positions={item.position for item in turn.visible_enemies},
        defensive_assignments={
            ranger.id: ranger.position
            for ranger in turn.rangers
            if roles[ranger.id.int] == "defense"
        },
    )
    return strategy, turn, context


def _decide(strategy, ranger, context):
    strategy._decide_ranger(
        ranger, context, strategy.memory.recent_enemies(context.turn.tick, 32)
    )
    return context.turn.plan.unit_actions[ranger.id]


def test_defender_pre_fires_the_cell_an_enemy_can_enter_from_range_four():
    strategy, turn, context = _scenario([(2, (0, 6), 2)], [(30, "RANGER", (0, 10), 2)])

    action = _decide(strategy, turn.rangers[0], context)

    assert isinstance(action, ShootAction)
    assert action.expected_cell == (0, 9)
    assert not strategy._visible_attackers_at((0, 6), turn, set())


def test_full_health_defender_keeps_a_survivable_shot_without_waiting_for_help():
    strategy, turn, context = _scenario([(2, (0, 6), 2)], [(30, "RANGER", (0, 9), 2)])

    assert _decide(strategy, turn.rangers[0], context).type == "SHOOT"


def test_defender_does_not_follow_a_contact_outside_its_fire():
    strategy, turn, context = _scenario([(2, (0, 6), 2)], [(30, "RANGER", (0, 12), 2)])
    before = dict(strategy.memory.defense_posts)

    action = _decide(strategy, turn.rangers[0], context)

    assert isinstance(action, ShootAction)
    assert action.target_id is None
    assert strategy.memory.defense_posts == before
    assert not any("hunt" in decision.reason for decision in context.report.decisions)


def test_front_rank_withdrawal_opens_a_covering_shot_for_the_next_rank():
    strategy, turn, context = _scenario(
        [(2, (0, 6), 1), (3, (0, 3), 2)],
        [(30, "VANGUARD", (0, 7), 4)],
    )

    front = _decide(strategy, turn.rangers[0], context)
    rear = _decide(strategy, turn.rangers[1], context)

    assert isinstance(front, MoveAction)
    destination = add((0, 6), front.direction)
    assert not strategy._visible_attackers_at(destination, turn, set())
    assert destination in {(-1, 6), (1, 6)}
    assert isinstance(rear, ShootAction)
    assert rear.expected_cell == (0, 6)
    assert rear.target_id == UUID(int=30)


def test_firing_order_plans_lethal_withdrawals_before_rear_rank_shots():
    memory = WorldMemory(unit_roles={object_id(2): "defense", object_id(3): "defense"})
    strategy = AggressiveStrategy(
        memory, StrategyConfig(target_workers=0, max_population=None)
    )
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 6), hp=1),
            unit(3, "RANGER", position=(0, 3)),
            unit(30, "VANGUARD", controlled=False, position=(0, 7)),
        ]
    )

    strategy.decide(turn)

    assert turn.plan.unit_actions[UUID(int=2)].type == "MOVE"
    rear = turn.plan.unit_actions[UUID(int=3)]
    assert isinstance(rear, ShootAction)
    assert rear.expected_cell == (0, 6)


def test_defender_can_shift_inside_its_layer_to_cover_a_threatened_neighbor():
    strategy, turn, context = _scenario(
        [(2, (-4, 4), 2), (3, (0, 5), 2)],
        [(30, "RANGER", (0, 8), 2)],
    )
    post = turn.rangers[0].position

    action = _decide(strategy, turn.rangers[0], context)

    assert isinstance(action, MoveAction)
    decision = context.report.decisions[-1]
    assert "cover threatened allies" in decision.reason
    assert manhattan(post, decision.target) <= DEFENSIVE_POST_SUPPORT_DISTANCE
    assert manhattan((0, 0), decision.target) <= manhattan((0, 0), post)
    assert not strategy._visible_attackers_at(add(post, action.direction), turn, set())


def test_idle_defender_rotates_free_cell_shots_without_leaving_its_post():
    aims = set()
    for tick in range(100, 108):
        strategy, turn, context = _scenario([(2, (0, 6), 2)], tick=tick)
        action = _decide(strategy, turn.rangers[0], context)
        assert isinstance(action, ShootAction)
        assert action.target_id is None
        assert line_of_fire((0, 6), action.expected_cell, set())
        assert manhattan((0, 0), action.expected_cell) > 6
        assert strategy._aim_cell_has_enemy_opportunity(action.expected_cell, turn)
        aims.add(action.expected_cell)
    assert len(aims) > 1


def test_defender_waits_when_every_firing_ray_is_a_rock():
    strategy, turn, context = _scenario(
        [(2, (0, 6), 2)],
        obstacles=[(dx, 6 + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy],
    )

    assert _decide(strategy, turn.rangers[0], context).type == "WAIT"
    assert context.report.decisions[-1].action == "WAIT"


def test_fully_observed_empty_cell_is_not_a_valid_aim_cell():
    strategy, turn, _context = _scenario([(2, (0, 6), 2)])

    assert not strategy._aim_cell_has_enemy_opportunity((0, 9), turn)
    assert not strategy._aim_cell_has_enemy_opportunity((1, 9), turn)
    assert strategy._aim_cell_has_enemy_opportunity((3, 9), turn)


def test_visible_neighbour_keeps_an_empty_approach_cell_valid():
    strategy, turn, _context = _scenario(
        [(2, (0, 6), 2)], [(30, "VANGUARD", (0, 10), 4)]
    )

    assert strategy._aim_cell_has_enemy_opportunity((0, 9), turn)


def test_recent_hidden_contact_is_not_shot_when_its_neighbourhood_is_visible():
    strategy, _turn, context = _scenario([(2, (0, 6), 2)], [(30, "RANGER", (0, 9), 2)])
    dark = make_turn(tick=101, objects=[core(), unit(2, "RANGER", position=(0, 6))])
    strategy.memory.observe(dark)
    context = _TurnContext(
        turn=dark,
        report=DecisionReport(tick=101),
        occupied={(0, 6)},
        enemy_positions=set(),
        defensive_assignments={UUID(int=2): (0, 6)},
    )

    assert not strategy._defensive_cover_fire(
        dark.rangers[0],
        context,
        strategy.memory.recent_enemies(dark.tick, 32),
    )


def test_recent_hidden_contact_is_covered_when_a_neighbour_remains_unseen():
    strategy, _turn, context = _scenario(
        [(2, (0, 6), 2)],
        [(30, "RANGER", (0, 9), 2)],
        obstacles=[(1, 8)],
    )
    dark = make_turn(tick=101, objects=[core(), unit(2, "RANGER", position=(0, 6))])
    strategy.memory.observe(dark)
    context = _TurnContext(
        turn=dark,
        report=DecisionReport(tick=101),
        occupied={(0, 6)},
        enemy_positions=set(),
        defensive_assignments={UUID(int=2): (0, 6)},
    )

    assert strategy._defensive_cover_fire(
        dark.rangers[0],
        context,
        strategy.memory.recent_enemies(dark.tick, 32),
    )
    shot = dark.plan.unit_actions[dark.rangers[0].id]

    assert isinstance(shot, ShootAction)
    assert shot.target_id is None
    assert manhattan(shot.expected_cell, (0, 9)) <= 2
    assert "without pursuit" in context.report.decisions[-1].reason


def test_stationary_target_does_not_get_an_adjacent_prediction_shot():
    memory = WorldMemory(
        unit_roles={object_id(2): "defense"},
        unit_roles_initialized=True,
        defense_posts={object_id(2): (0, 6)},
        defense_anchor=(0, 0),
    )
    strategy = AggressiveStrategy(
        memory, StrategyConfig(target_workers=0, max_population=None)
    )
    for tick in (98, 99, 100):
        turn = make_turn(
            tick=tick,
            objects=[
                core(),
                unit(2, "RANGER", position=(0, 6)),
                unit(30, "WORKER", controlled=False, position=(1, 8)),
            ],
        )
        memory.observe(turn)
    context = _TurnContext(
        turn=turn,
        report=DecisionReport(tick=100),
        occupied={item.position for item in turn.units},
        enemy_positions={item.position for item in turn.visible_enemies},
        defensive_assignments={UUID(int=2): (0, 6)},
    )

    target = turn.visible_enemies[0]
    assert (
        strategy._ranger_shot_cells(
            turn.rangers[0], target, turn, set(), {item.position for item in turn.units}
        )
        == ()
    )
    assert not strategy._defensive_cover_fire(
        turn.rangers[0],
        context,
        memory.recent_enemies(turn.tick, 32),
    )


def test_moving_target_still_uses_its_observed_lead_cell():
    memory = WorldMemory(
        unit_roles={object_id(2): "defense"},
        unit_roles_initialized=True,
        defense_posts={object_id(2): (0, 6)},
        defense_anchor=(0, 0),
    )
    strategy = AggressiveStrategy(
        memory, StrategyConfig(target_workers=0, max_population=None)
    )
    for tick, position in ((98, (1, 10)), (99, (1, 9)), (100, (1, 8))):
        turn = make_turn(
            tick=tick,
            objects=[
                core(),
                unit(2, "RANGER", position=(0, 6)),
                unit(30, "WORKER", controlled=False, position=position),
            ],
        )
        memory.observe(turn)

    cells = strategy._ranger_shot_cells(
        turn.rangers[0],
        turn.visible_enemies[0],
        turn,
        set(),
        {item.position for item in turn.units},
    )

    assert cells[0] == (1, 7)


def _patrol_case(*, enemy_count=1, supporter=(0, 11), supporter_hp=2, tick=100):
    return _scenario(
        [(2, (0, 8), 2), (3, supporter, supporter_hp)],
        [(30 + index, "RANGER", (0, 13), 2) for index in range(enemy_count)],
        roles={2: "patrol-1", 3: "patrol-1"},
        tick=tick,
    )


def test_patrol_pursues_only_with_a_present_healthy_supporting_shooter():
    strategy, turn, context = _patrol_case()

    action = _decide(strategy, turn.rangers[0], context)

    assert isinstance(action, MoveAction)
    assert add((0, 8), action.direction) == (0, 9)
    assert "local force advantage" in context.report.decisions[-1].reason


@pytest.mark.parametrize(
    "conditions",
    [
        {"enemy_count": 2},
        {"supporter": (20, 20)},
        {"supporter_hp": 1},
    ],
)
def test_patrol_does_not_count_remote_wounded_or_outnumbered_help(conditions):
    strategy, turn, context = _patrol_case(**conditions)

    assert _decide(strategy, turn.rangers[0], context).type != "MOVE"


def test_patrol_pursuit_distance_time_and_restart_do_not_renew_on_contact(tmp_path):
    strategy, turn, context = _patrol_case()
    patrol = turn.rangers[0]
    strategy._update_patrol_contact(patrol, tuple(turn.visible_enemies), context)
    contact = strategy.memory.patrol_contacts["patrol-1"]
    assert strategy._patrol_can_pursue(patrol, turn.visible_enemies[0], (0, 9), context)
    assert not strategy._patrol_can_pursue(
        patrol, turn.visible_enemies[0], (0, 8 + PATROL_PURSUIT_DISTANCE + 1), context
    )
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    loaded = WorldMemory.load(path)
    assert loaded.patrol_contacts["patrol-1"] == contact
    late = make_turn(
        tick=100 + PATROL_PURSUIT_TICKS,
        objects=[
            core(),
            unit(2, "RANGER", position=(0, 8)),
            unit(3, "RANGER", position=(0, 11)),
            unit(30, "RANGER", controlled=False, position=(0, 13)),
        ],
    )
    restarted = AggressiveStrategy(loaded, strategy.config)
    loaded.observe(late)
    context = _TurnContext(
        turn=late,
        report=DecisionReport(tick=late.tick),
        occupied={r.position for r in late.units},
        enemy_positions={e.position for e in late.visible_enemies},
    )
    action = _decide(restarted, late.rangers[0], context)
    assert action.type != "MOVE"
    assert loaded.patrol_contacts["patrol-1"].started_tick == 100


def test_pursuit_never_enters_a_lethal_cell_even_with_a_local_advantage():
    strategy, turn, context = _patrol_case(enemy_count=2)
    patrol = turn.rangers[0]
    strategy._update_patrol_contact(patrol, tuple(turn.visible_enemies), context)

    assert not strategy._patrol_can_pursue(
        patrol, turn.visible_enemies[0], (0, 10), context
    )


def test_real_volley_count_excludes_blocked_and_doomed_rangers():
    strategy, turn, context = _scenario(
        [(2, (0, 6), 2), (3, (0, 7), 1), (5, (3, 6), 2)],
        [(30, "RANGER", (0, 9), 2)],
    )

    assert strategy._ranger_volley_count(turn.visible_enemies[0], context) == 1


def test_front_and_rear_wait_instead_of_targeting_a_friendly_reserved_endpoint():
    strategy, turn, context = _scenario(
        [(2, (0, 6), 2), (3, (0, 9), 2)], [(30, "RANGER", (0, 10), 2)]
    )

    action = _decide(strategy, turn.rangers[0], context)

    assert action.type == "WAIT"


def test_new_core_clears_persisted_patrol_contact():
    memory = WorldMemory(
        campaign_last_core_id=object_id(1),
        campaign_phase=3,
        patrol_contacts={"patrol-1": PatrolContact((0, 8), 100, 106, object_id(30))},
    )
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(
            expedition_mode=True,
            campaign_mode=True,
            target_workers=16,
            max_population=None,
        ),
    )

    strategy._observe_campaign_state(make_turn(objects=[core(9, position=(15, 15))]))

    assert memory.patrol_contacts == {}


def test_a_less_exposed_but_still_lethal_escape_does_not_forfeit_the_shot():
    strategy, turn, context = _scenario(
        [(2, (0, 6), 2)],
        [
            (30, "RANGER", (1, 7), 1),
            (31, "RANGER", (1, 7), 2),
            (32, "RANGER", (2, 8), 2),
        ],
        obstacles=[(-1, 6), (0, 5), (0, 7)],
    )
    rocks = set(turn.obstacle_cells)
    assert len(strategy._visible_attackers_at((0, 6), turn, rocks)) == 3
    assert len(strategy._visible_attackers_at((1, 6), turn, rocks)) == 2

    assert _decide(strategy, turn.rangers[0], context).type == "SHOOT"


def test_healing_recovery_cannot_fall_back_to_a_queue_through_lethal_fire():
    strategy, turn, context = _scenario(
        [(2, (0, 6), 1)],
        [(30, "RANGER", (1, 9), 2)],
        obstacles=[(-1, 6), (0, 5), (0, 7)],
    )

    action = _decide(strategy, turn.rangers[0], context)

    assert action.type != "MOVE"


def test_queued_waiting_patrol_mate_does_not_count_as_support():
    strategy, turn, context = _patrol_case()
    turn.rangers[1].wait()
    strategy._update_patrol_contact(
        turn.rangers[0], tuple(turn.visible_enemies), context
    )

    assert not strategy._patrol_can_pursue(
        turn.rangers[0], turn.visible_enemies[0], (0, 9), context
    )


def test_friendly_arrival_is_part_of_the_shooting_obstruction_picture():
    strategy, turn, context = _scenario(
        [(2, (0, 6), 2), (3, (1, 8), 2)], [(30, "RANGER", (0, 9), 2)]
    )
    context.arrival_counts[(0, 8)] = 1
    context.departure_counts[(1, 8)] = 1

    positions = strategy._predicted_friendly_positions(context)

    assert (0, 8) in positions
    assert (1, 8) not in positions
    assert (0, 9) not in strategy._ranger_shot_cells(
        turn.rangers[0], turn.visible_enemies[0], turn, set(), positions
    )


def test_defensive_arrows_on_alternate_cells_do_not_pretend_the_focus_is_dead():
    strategy, turn, context = _scenario(
        [(2, (0, 6), 2)],
        [(30, "RANGER", (0, 9), 2), (31, "VANGUARD", (3, 6), 4)],
    )
    focus = turn.visible_enemies[0]
    context.focus_target = focus
    context.damage_ledger.record(focus, 2)
    context.volley_aims[str(focus.id)] = [(0, 9), (0, 8)]

    action = _decide(strategy, turn.rangers[0], context)

    assert isinstance(action, ShootAction)
    assert action.target_id == focus.id
