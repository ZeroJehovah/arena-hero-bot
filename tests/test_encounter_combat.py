"""Encounter fire, safe arrivals, and continuity under partial visibility."""

from uuid import UUID

import pytest
from arena_hero import Direction, MoveAction, UnitView

from arena_hero_bot.geometry import add, line_of_fire, manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig, _TurnContext

from .factories import core, make_turn, object_id, unit
from .test_layered_defense import Encounter


def _strategy(roles, *, memory=None):
    memory = memory or WorldMemory(core_home_position=(0, 0))
    memory.unit_roles = roles
    memory.unit_roles_initialized = True
    return AggressiveStrategy(
        memory, StrategyConfig(target_workers=0, max_population=None)
    )


def _context(strategy, turn, *, posts=None):
    strategy.memory.observe(turn)
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={r.position for r in turn.units},
        enemy_positions={e.position for e in turn.visible_enemies},
        defensive_assignments=posts or {r.id: r.position for r in turn.rangers},
        all_hands_defense=True,
        remaining_resources=turn.resources,
        remaining_resource_space=turn.resource_space,
    )


def _decide(strategy, ranger, context):
    strategy._decide_ranger(
        ranger, context, strategy.memory.recent_enemies(context.turn.tick, 32)
    )
    return context.turn.plan.unit_actions[ranger.id]


@pytest.mark.parametrize("role", ["defense", "patrol-1"])
@pytest.mark.parametrize("idle_vanguard", [False, True])
def test_an_unrelated_kill_does_not_commit_a_lone_ranger_against_two(
    role, idle_vanguard
):
    friends = [
        unit(2, "RANGER", position=(-6, -3)),
        unit(3, "RANGER", position=(0, -3)),
        unit(4, "RANGER", position=(-1, -4)),
    ]
    foes = [
        unit(30, "RANGER", controlled=False, position=(-6, -5)),
        unit(31, "RANGER", controlled=False, position=(-6, -6)),
        unit(32, "RANGER", controlled=False, position=(0, -5)),
    ]
    if idle_vanguard:
        # Both duels can hit this Vanguard, but it cannot sweep either one.
        # No killing volley targets it, so it shares no hostile attack budget.
        foes.append(unit(33, "VANGUARD", controlled=False, position=(-3, -3)))
    encounter = Encounter(friends, foes, resources=0)
    strategy = _strategy({friend["id"]: role for friend in friends})
    turn = encounter.turn()
    context = _context(strategy, turn)

    for ranger in turn.rangers:
        _decide(strategy, ranger, context)

    assert turn.plan.unit_actions[UUID(int=2)].type == "MOVE"
    assert all(turn.plan.unit_actions[UUID(int=n)].type == "SHOOT" for n in (3, 4))
    encounter.settle(
        turn.plan,
        {
            foe["id"]: {
                "type": "SHOOT",
                "target_id": object_id(2 if n < 2 else 3),
                "expected_cell": friends[0 if n < 2 else 1]["position"],
            }
            for n, foe in enumerate(foes)
            if foe["unit_type"] == "RANGER"
        },
    )
    assert (encounter.kills, encounter.losses) == (1, 0)


@pytest.mark.parametrize("role", ["defense", "patrol-1"])
def test_a_healthy_ranger_kites_an_adjacent_vanguard_that_the_volley_cannot_kill(role):
    friends = [unit(2, "RANGER", position=(0, -4))]
    foes = [unit(30, "VANGUARD", controlled=False, position=(0, -5))]
    encounter = Encounter(friends, foes, resources=0)
    strategy = _strategy({object_id(2): role})
    turn = encounter.turn()
    context = _context(strategy, turn)

    action = _decide(strategy, turn.rangers[0], context)

    assert isinstance(action, MoveAction)
    assert manhattan(add((0, -4), action.direction), (0, -5)) > 1
    encounter.settle(turn.plan, {object_id(30): {"type": "SWEEP", "direction": "DOWN"}})
    assert encounter.losses == 0
    assert encounter.friends[object_id(2)]["hp"] == 2
    reply = encounter.turn()
    _decide(strategy, reply.rangers[0], _context(strategy, reply))
    assert reply.plan.unit_actions[UUID(int=2)].type == "SHOOT"
    encounter.settle(reply.plan, {})
    assert encounter.foes[object_id(30)]["hp"] == 3


def test_a_ready_volley_keeps_its_killing_arrows_against_an_adjacent_vanguard():
    friends = [
        unit(2, "RANGER", position=(0, -4)),
        unit(3, "RANGER", position=(1, -4)),
    ]
    foes = [unit(30, "VANGUARD", controlled=False, position=(0, -5), hp=2)]
    encounter = Encounter(friends, foes, resources=0)
    strategy = _strategy({friend["id"]: "defense" for friend in friends})
    turn = encounter.turn()
    context = _context(strategy, turn)
    for ranger in turn.rangers:
        assert _decide(strategy, ranger, context).type == "SHOOT"
    encounter.settle(turn.plan, {object_id(30): {"type": "SWEEP", "direction": "DOWN"}})
    assert (encounter.kills, encounter.losses) == (1, 0)
    # The killed Vanguard's simultaneous sweep still lands on this shooter.
    assert encounter.friends[object_id(2)]["hp"] == 1
    assert encounter.friends[object_id(3)]["hp"] == 2


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("age", [1, 12, 32])
def test_a_hidden_pair_is_not_forgotten_when_a_noncombat_enemy_is_visible(
    tmp_path, restart, age
):
    friends = [unit(2, "RANGER", position=(0, -4))]
    foes = [
        unit(30, "RANGER", controlled=False, position=(-3, -7)),
        unit(31, "RANGER", controlled=False, position=(-3, -7)),
    ]
    memory = WorldMemory(core_home_position=(0, 0))
    memory.observe(
        make_turn(
            tick=100 - age, objects=[core(), unit(2, "RANGER", position=(0, -5)), *foes]
        )
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    strategy = _strategy({object_id(2): "defense"}, memory=memory)
    turn = make_turn(
        objects=[
            core(),
            *friends,
            unit(40, "WORKER", controlled=False, position=(4, 0)),
        ]
    )

    action = _decide(strategy, turn.rangers[0], _context(strategy, turn))

    assert isinstance(action, MoveAction)
    encounter = Encounter(friends, foes, resources=0)
    encounter.settle(
        turn.plan,
        {
            foe["id"]: {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [0, -4],
            }
            for foe in foes
        },
    )
    assert encounter.losses == 0


@pytest.mark.parametrize("restart", [False, True])
def test_a_post_step_does_not_enter_two_recently_unseen_rays(tmp_path, restart):
    friends = [unit(2, "RANGER", position=(1, -4))]
    foes = [
        unit(30, "RANGER", controlled=False, position=(-3, -7)),
        unit(31, "RANGER", controlled=False, position=(-3, -7)),
    ]
    memory = WorldMemory(core_home_position=(0, 0))
    memory.observe(
        make_turn(tick=98, objects=[core(), unit(2, "RANGER", position=(0, -5)), *foes])
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    strategy = _strategy({object_id(2): "defense"}, memory=memory)
    turn = make_turn(objects=[core(), *friends])
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (0, -4)}
    strategy._prepare_layered_steps(context, ())
    _decide(strategy, turn.rangers[0], context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type != "MOVE" or add((1, -4), action.direction) != (0, -4)
    encounter = Encounter(friends, foes, resources=0)
    encounter.settle(
        turn.plan,
        {
            foe["id"]: {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [0, -4],
            }
            for foe in foes
        },
    )
    assert encounter.losses == 0


def test_a_lone_post_move_does_not_pay_an_arrow_before_an_unwinnable_duel():
    strategy = _strategy({object_id(2): "defense"})
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(1, -4)),
            unit(30, "RANGER", controlled=False, position=(0, -7)),
        ]
    )
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (0, -4)}
    enemies = tuple(e for e in turn.visible_enemies if isinstance(e, UnitView))
    strategy._prepare_layered_steps(context, enemies)
    _decide(strategy, turn.rangers[0], context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type != "MOVE" or add((1, -4), action.direction) != (0, -4)


def test_a_covered_adjacent_firing_step_precedes_an_unrelated_layered_post():
    friends = [unit(2, "RANGER", position=(0, -4)), unit(3, "RANGER", position=(2, -4))]
    foes = [unit(30, "RANGER", controlled=False, position=(2, -7))]
    encounter = Encounter(friends, foes, resources=0)
    strategy = _strategy({friend["id"]: "defense" for friend in friends})
    turn = encounter.turn()
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (0, -3)}
    context.layered_steps = {UUID(int=2): Direction.DOWN}

    action = _decide(strategy, turn.rangers[0], context)

    assert isinstance(action, MoveAction)
    destination = add((0, -4), action.direction)
    assert line_of_fire(destination, (2, -7), set())
    assert manhattan(destination, (0, -4)) == 1
    assert _decide(strategy, turn.rangers[1], context).type == "SHOOT"
    encounter.settle(
        turn.plan,
        {
            object_id(30): {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": list(destination),
            }
        },
    )
    assert encounter.losses == 0
    reply = encounter.turn()
    reply_context = _context(strategy, reply)
    for ranger in reply.rangers:
        _decide(strategy, ranger, reply_context)
    encounter.settle(reply.plan, {})
    assert (encounter.kills, encounter.losses) == (1, 0)


@pytest.mark.parametrize("restart", [False, True])
def test_recent_contact_fire_holds_a_safe_front_position_before_post_reassignment(
    tmp_path, restart
):
    memory = WorldMemory(core_home_position=(0, 5))
    memory.observe(
        make_turn(
            tick=99,
            objects=[
                core(position=(0, 5)),
                unit(2, "RANGER"),
                unit(30, "RANGER", controlled=False, position=(3, -3)),
            ],
        )
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    strategy = _strategy({object_id(2): "defense"}, memory=memory)
    turn = make_turn(objects=[core(position=(0, 5)), unit(2, "RANGER")])
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (-1, 0)}
    context.layered_steps = {UUID(int=2): Direction.LEFT}

    action = _decide(strategy, turn.rangers[0], context)

    assert action.type == "SHOOT"
    assert action.expected_cell == (3, -3)
    assert action.target_id is None


def test_outer_cover_prefers_recent_enemy_bearing_after_exact_contact_expires():
    memory = WorldMemory(core_home_position=(0, 5))
    memory.observe(
        make_turn(
            tick=95,
            objects=[
                core(position=(0, 5)),
                unit(2, "RANGER"),
                unit(30, "RANGER", controlled=False, position=(5, -5)),
            ],
        )
    )
    strategy = _strategy({object_id(2): "defense"}, memory=memory)
    turn = make_turn(objects=[core(position=(0, 5)), unit(2, "RANGER")])
    context = _context(strategy, turn)

    strategy._hold_defensive_ranger(turn.rangers[0], context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "SHOOT"
    assert action.expected_cell == (3, -3)


@pytest.mark.parametrize("age", [5, 32, 33])
def test_recent_approach_outranks_unrelated_unknown_corners_until_expiry(age):
    memory = WorldMemory(core_home_position=(0, 1))
    memory.observe(
        make_turn(
            tick=100 - age,
            objects=[
                core(position=(0, 1)),
                unit(2, "RANGER", position=(0, -5)),
                unit(30, "RANGER", controlled=False, position=(0, -6)),
            ],
        )
    )
    strategy = _strategy({object_id(2): "defense"}, memory=memory)
    turn = make_turn(
        objects=[core(position=(0, 1)), unit(2, "RANGER")],
        obstacles=[(-1, -1), (1, -1)],
    )
    context = _context(strategy, turn)

    strategy._hold_defensive_ranger(turn.rangers[0], context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "SHOOT"
    if age <= 32:
        assert action.expected_cell == (0, -3)
    else:
        assert action.expected_cell in {(-3, 3), (3, 3)}


@pytest.mark.parametrize("fresh", [False, True])
def test_old_damage_does_not_displace_healthy_core_fire_but_fresh_unknown_fire_does(
    fresh,
):
    strategy = _strategy({object_id(2): "defense"})
    strategy.memory.observe(
        make_turn(
            tick=99 if fresh else 87,
            objects=[core(position=(0, 5)), unit(3, "RANGER")],
        )
    )
    turn = make_turn(
        objects=[
            core(position=(0, 5)),
            unit(2, "RANGER"),
            core(50, controlled=False, position=(3, -3)),
            unit(30, "RANGER", controlled=False, position=(3, -4)),
        ],
        events=[
            {
                "event_id": object_id(900),
                "tick": 99 if fresh else 87,
                "event_type": "UNIT_DAMAGED",
                "reason_code": "ATTACK",
                "target_id": object_id(3),
                "position": [0, 0],
                "values": {"damage": 2, "hp": 0},
            }
        ],
    )
    context = _context(strategy, turn)

    action = _decide(strategy, turn.rangers[0], context)

    assert action.type == ("MOVE" if fresh else "SHOOT")
    if not fresh:
        assert action.target_id == UUID(int=50)
        assert action.expected_cell == (3, -3)


def test_a_safe_core_shot_inside_the_leash_is_kept_despite_a_distant_old_post():
    strategy = _strategy({object_id(2): "defense"})
    turn = make_turn(
        objects=[
            core(position=(0, 5)),
            unit(2, "RANGER"),
            core(50, controlled=False, position=(3, -3)),
        ]
    )
    context = _context(strategy, turn, posts={UUID(int=2): (-4, 5)})

    action = _decide(strategy, turn.rangers[0], context)

    assert action.type == "SHOOT"
    assert action.target_id == UUID(int=50)


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("role", ["defense", "patrol-1"])
@pytest.mark.parametrize("harmless_worker", [False, True])
def test_just_seen_core_outside_diagonal_vision_keeps_safe_cover_before_post_moves(
    tmp_path, restart, role, harmless_worker
):
    memory = WorldMemory(core_home_position=(0, 5))
    memory.observe(
        make_turn(
            tick=99,
            objects=[
                core(position=(0, 5)),
                unit(2, "RANGER"),
                core(50, controlled=False, position=(3, -3)),
            ],
        )
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    strategy = _strategy({object_id(2): role}, memory=memory)
    turn = make_turn(
        objects=[
            core(position=(0, 5)),
            unit(2, "RANGER"),
            *(
                [unit(40, "WORKER", controlled=False, position=(3, 0))]
                if harmless_worker
                else []
            ),
        ]
    )
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (0, 1)}

    strategy._prepare_layered_steps(context, ())
    action = _decide(strategy, turn.rangers[0], context)

    assert action.type == "SHOOT"
    assert action.expected_cell == (3, -3)
    assert action.target_id is None
    assert context.damage_ledger.planned_damage == {}


@pytest.mark.parametrize("cleared", ["expired", "destroyed", "visible-empty"])
def test_recent_core_cover_ends_after_expiry_destruction_or_observed_empty_cell(
    cleared,
):
    memory = WorldMemory(core_home_position=(0, 5))
    cell = (3, -3) if cleared != "visible-empty" else (3, 0)
    memory.observe(
        make_turn(
            tick=97 if cleared == "expired" else 99,
            objects=[
                core(position=(0, 5)),
                unit(2, "RANGER"),
                core(50, controlled=False, position=cell),
            ],
        )
    )
    strategy = _strategy({object_id(2): "defense"}, memory=memory)
    events = (
        [
            {
                "event_id": object_id(900),
                "tick": 99,
                "event_type": "DESTRUCTION_PARTICIPATION",
                "reason_code": "CORE",
                "target_id": object_id(50),
                "position": list(cell),
            }
        ]
        if cleared == "destroyed"
        else []
    )
    turn = make_turn(objects=[core(position=(0, 5)), unit(2, "RANGER")], events=events)
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (0, 1)}

    strategy._prepare_layered_steps(context, ())
    action = _decide(strategy, turn.rangers[0], context)

    assert action.type == "MOVE"
    assert add((0, 0), action.direction) == (0, 1)


@pytest.mark.parametrize("current_fighter", [False, True])
def test_core_contact_shooters_keep_the_ray_unless_a_fighter_is_currently_shootable(
    current_fighter,
):
    memory = WorldMemory(core_home_position=(0, 5))
    memory.observe(
        make_turn(
            tick=99,
            objects=[
                core(position=(0, 5)),
                unit(2, "RANGER"),
                core(50, controlled=False, position=(3, -3)),
            ],
        )
    )
    strategy = _strategy(
        {object_id(2): "defense", object_id(3): "defense"}, memory=memory
    )
    turn = make_turn(
        objects=[
            core(position=(0, 5)),
            unit(2, "RANGER"),
            unit(3, "RANGER"),
            unit(
                30,
                "RANGER",
                controlled=False,
                position=(3, 3) if current_fighter else (3, -2),
                hp=1,
            ),
        ]
    )
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (0, 1), UUID(int=3): (1, 1)}
    enemies = tuple(
        enemy for enemy in turn.visible_enemies if isinstance(enemy, UnitView)
    )
    strategy._prepare_layered_steps(context, enemies)

    action = _decide(strategy, turn.rangers[0], context)

    assert action.type == "SHOOT"
    assert action.expected_cell == ((3, 3) if current_fighter else (3, -3))
    assert action.target_id == (UUID(int=30) if current_fighter else None)
    if not current_fighter:
        second = _decide(strategy, turn.rangers[1], context)
        assert second.type == "SHOOT"
        assert second.expected_cell == (3, -3)


@pytest.mark.parametrize("block_first_exit", [False, True])
def test_wounded_isolated_ranger_tries_other_safe_exits_before_a_losing_shot(
    block_first_exit,
):
    memory = WorldMemory(core_home_position=(0, 5))
    for tick, cell in [(98, (0, -5)), (99, (0, -4))]:
        memory.observe(
            make_turn(
                tick=tick,
                objects=[
                    core(position=(0, 5)),
                    unit(2, "RANGER", hp=1),
                    unit(30, "RANGER", controlled=False, position=cell, hp=1),
                    unit(31, "RANGER", controlled=False, position=cell),
                ],
            )
        )
    strategy = _strategy({object_id(2): "defense"}, memory=memory)
    friends = [unit(2, "RANGER", hp=1)]
    foes = [
        unit(30, "RANGER", controlled=False, position=(0, -3), hp=1),
        unit(31, "RANGER", controlled=False, position=(0, -3)),
    ]
    turn = make_turn(resources=20, objects=[core(position=(0, 5)), *friends, *foes])
    context = _context(strategy, turn)
    context.core_migration_healing_ids = frozenset({UUID(int=2)})
    if block_first_exit:
        context.counterattack_reserved = frozenset({(-1, 0)})

    action = _decide(strategy, turn.rangers[0], context)

    assert isinstance(action, MoveAction)
    destination = add((0, 0), action.direction)
    assert destination == ((1, 0) if block_first_exit else (-1, 0))
    assert not strategy._future_attackers_at(destination, turn, set())
    encounter = Encounter(friends, foes, resources=20)
    encounter.settle(
        turn.plan,
        {
            foe["id"]: {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [0, 0],
            }
            for foe in foes
        },
    )
    assert encounter.losses == 0


@pytest.mark.parametrize("cleared", ["expired", "destroyed", "visible-empty"])
@pytest.mark.parametrize("restart", [False, True])
def test_post_routes_reopen_after_a_contact_is_cleared(tmp_path, cleared, restart):
    memory = WorldMemory(core_home_position=(0, 0))
    enemy_cell = (0, -6) if cleared == "visible-empty" else (-3, -7)
    old_tick = 67 if cleared == "expired" else 99
    memory.observe(
        make_turn(
            tick=old_tick,
            objects=[
                core(),
                unit(2, "RANGER", position=(0, -5)),
                unit(30, "RANGER", controlled=False, position=enemy_cell),
                unit(31, "RANGER", controlled=False, position=enemy_cell),
            ],
        )
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    position = (1, -4) if cleared != "visible-empty" else (0, -4)
    goal = (0, -4) if cleared != "visible-empty" else (0, -5)
    events = []
    if cleared == "destroyed":
        events = [
            {
                "event_id": object_id(900 + n),
                "tick": 99,
                "event_type": "DESTRUCTION_PARTICIPATION",
                "reason_code": "UNIT",
                "target_id": object_id(n),
                "position": list(enemy_cell),
            }
            for n in (30, 31)
        ]
    strategy = _strategy({object_id(2): "defense"}, memory=memory)
    turn = make_turn(
        objects=[core(), unit(2, "RANGER", position=position)], events=events
    )
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): goal}

    strategy._prepare_layered_steps(context, ())
    strategy._move_layered_ranger(turn.rangers[0], context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert isinstance(action, MoveAction)
    assert add(position, action.direction) == goal


@pytest.mark.parametrize("ally_action", ["WAIT", "MOVE"])
def test_a_waiting_or_moving_ally_does_not_cover_an_unsupported_ingress(ally_action):
    strategy = _strategy({object_id(n): "defense" for n in (2, 3)})
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(1, -4)),
            unit(3, "RANGER", position=(2, -5)),
            unit(30, "RANGER", controlled=False, position=(0, -7)),
        ]
    )
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (0, -4)}
    if ally_action == "WAIT":
        turn.rangers[1].wait()
    else:
        turn.rangers[1].move(Direction.RIGHT)

    strategy._prepare_layered_steps(
        context,
        tuple(enemy for enemy in turn.visible_enemies if isinstance(enemy, UnitView)),
    )
    _decide(strategy, turn.rangers[0], context)

    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type != "MOVE" or add((1, -4), action.direction) != (0, -4)


def test_atomic_post_ingress_does_not_take_an_unrelated_contact_shooter_backwards():
    friends = [
        unit(2, "RANGER", position=(0, -4)),
        unit(3, "RANGER", position=(-6, -4)),
        unit(4, "RANGER", position=(-7, -5)),
        unit(9, "WORKER", position=(4, -6)),
    ]
    hidden = unit(50, "RANGER", controlled=False, position=(-3, -6))
    foes = [
        unit(30, "RANGER", controlled=False, position=(-5, -7), hp=1),
        unit(31, "RANGER", controlled=False, position=(4, -7), hp=1),
    ]
    memory = WorldMemory(core_home_position=(0, 0))
    memory.observe(
        make_turn(
            tick=99, objects=[core(), unit(8, "RANGER", position=(-3, -5)), hidden]
        )
    )
    strategy = _strategy({object_id(n): "defense" for n in (2, 3, 4)}, memory=memory)
    turn = make_turn(objects=[core(), *friends, *foes], obstacles=[(-1, -5), (-4, -6)])
    context = _context(strategy, turn)
    context.layered_assignments = {UUID(int=2): (0, -3), UUID(int=3): (-5, -4)}
    enemies = tuple(
        enemy for enemy in turn.visible_enemies if isinstance(enemy, UnitView)
    )
    assert (-3, -6) not in strategy._fleet_visible_cells(turn)

    strategy._prepare_layered_steps(context, enemies)

    assert UUID(int=2) not in turn.plan.unit_actions
    assert _decide(strategy, turn.rangers[0], context).type == "SHOOT"
    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "SHOOT"
    assert action.expected_cell == (3, -7)
    _decide(strategy, turn.rangers[2], context)
    encounter = Encounter(
        friends, [*foes, hidden], resources=0, rocks=[(-1, -5), (-4, -6)]
    )
    encounter.settle(
        turn.plan,
        {
            object_id(30): {
                "type": "SHOOT",
                "target_id": object_id(3),
                "expected_cell": [-5, -4],
            },
            object_id(31): {"type": "MOVE", "direction": "LEFT"},
            object_id(50): {
                "type": "SHOOT",
                "target_id": object_id(3),
                "expected_cell": [-5, -4],
            },
        },
    )
    assert (encounter.kills, encounter.losses) == (2, 1)
