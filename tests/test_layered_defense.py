"""Layer continuity, real combat settlement, economy and four-Tick retreat."""

from collections import Counter, defaultdict
from copy import deepcopy
from uuid import UUID

import pytest
from arena_hero import Direction, MoveAction, StartMoveAction, UnitView

from arena_hero_bot.defense import (
    FRONT_DEPTH,
    REAR_DEPTH,
    casualty_bound,
    layered_posts,
    maximum_core_damage,
    precision_volley,
    projection,
    threatened_sides,
)
from arena_hero_bot.geometry import add, line_of_fire
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig, _TurnContext

from .factories import core, make_turn, object_id, unit


def _strategy(roles=None):
    return AggressiveStrategy(
        WorldMemory(
            unit_roles=roles or {},
            unit_roles_initialized=True,
            core_home_position=(0, 0),
        ),
        StrategyConfig(target_workers=0, max_population=None, campaign_mode=True),
    )


def _context(strategy, turn):
    strategy.memory.observe(turn)
    threat = strategy._combat_policy.assess(turn, set(turn.obstacle_cells))
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={actor.position for actor in turn.units},
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
        remaining_resources=turn.resources,
        remaining_resource_space=turn.resource_space,
        combat_assault=True,
        all_hands_defense=True,
        assault_enemies=tuple(
            e for e in turn.visible_enemies if isinstance(e, UnitView)
        ),
        threat=threat,
    )


def test_north_and_east_receive_two_ranks_before_rear_watch():
    turn = make_turn(
        objects=[
            core(),
            *[
                unit(n, "RANGER", position=(n % 4 - 2, n // 4 + 2))
                for n in range(2, 22)
            ],
            unit(50, "RANGER", controlled=False, position=(0, -6)),
            unit(51, "RANGER", controlled=False, position=(6, 0)),
        ],
        obstacles=[(0, -4), (4, 0)],
        resource_cells=[(1, -4), (4, 1)],
    )
    foes = tuple(e for e in turn.visible_enemies if isinstance(e, UnitView))
    unavailable = set(turn.resource_cells) | {enemy.position for enemy in foes}
    posts = layered_posts(
        (0, 0), turn.rangers, foes, {}, unavailable, turn.obstacle_cells
    )

    assert len(set(posts.values())) == len(posts)
    assert not set(posts.values()) & (unavailable | set(turn.obstacle_cells))
    for side in (Direction.UP, Direction.RIGHT):
        for depth in (FRONT_DEPTH, REAR_DEPTH):
            assert (
                sum(projection((0, 0), cell, side) == depth for cell in posts.values())
                >= 2
            )
    assert sum(cell[1] > 0 and cell[0] < 0 for cell in posts.values()) <= 1


def test_stationed_ranks_keep_uuid_posts_with_unchanged_contact():
    turn = make_turn(
        objects=[
            core(),
            *[unit(n, "RANGER", position=(n, 2)) for n in range(2, 12)],
            unit(50, "RANGER", controlled=False, position=(0, -6)),
        ]
    )
    posts = layered_posts(
        (0, 0),
        turn.rangers,
        tuple(e for e in turn.visible_enemies if isinstance(e, UnitView)),
        {},
        set(),
        set(),
    )
    arrived = make_turn(
        objects=[
            core(),
            *[
                unit(actor_id.int, "RANGER", position=position)
                for actor_id, position in posts.items()
            ],
            unit(50, "RANGER", controlled=False, position=(0, -6)),
        ]
    )
    again = layered_posts(
        (0, 0),
        arrived.rangers,
        tuple(
            enemy for enemy in arrived.visible_enemies if isinstance(enemy, UnitView)
        ),
        {str(k): v for k, v in posts.items()},
        set(),
        set(),
    )
    assert again == posts


def test_large_army_receives_stable_reserves_on_the_threatened_side():
    turn = make_turn(
        objects=[
            core(),
            *[
                unit(n, "RANGER", position=(n % 8 - 4, n // 8 + 1))
                for n in range(2, 57)
            ],
            unit(90, "RANGER", controlled=False, position=(0, -6)),
        ]
    )
    foes = tuple(e for e in turn.visible_enemies if isinstance(e, UnitView))
    posts = layered_posts((0, 0), turn.rangers, foes, {}, set(), set())

    assert len(posts) == 55
    assert len(set(posts.values())) == 55
    assert sum(cell[1] < 0 for cell in posts.values()) >= 52
    assert (0, 0) not in posts.values()


def test_diagonal_intruder_threatens_both_adjoining_sides():
    turn = make_turn(
        objects=[core(), unit(50, "RANGER", controlled=False, position=(5, -5))]
    )
    assert set(
        threatened_sides(
            (0, 0), tuple(e for e in turn.visible_enemies if isinstance(e, UnitView))
        )
    ) == {
        Direction.UP,
        Direction.RIGHT,
    }


def test_shared_arrows_and_precision_kills_for_stacked_pair():
    turn = make_turn(
        objects=[
            core(),
            unit(50, "RANGER", controlled=False, position=(0, -6)),
            unit(51, "RANGER", controlled=False, position=(0, -6), hp=1),
        ]
    )
    positions = {UUID(int=2): (0, -4), UUID(int=3): (2, -4), UUID(int=4): (0, -3)}
    assert (
        casualty_bound(
            {k: (v, 2) for k, v in positions.items()},
            tuple(e for e in turn.visible_enemies if isinstance(e, UnitView)),
            set(),
        )
        == 1
    )
    volley = precision_volley(
        positions,
        tuple(e for e in turn.visible_enemies if isinstance(e, UnitView)),
        (0, 0),
        set(),
    )
    assert Counter(target.id.int for target in volley.values()) == {50: 2, 51: 1}


@pytest.mark.parametrize(
    "route, expected", [(((0, 0),) * 4, 3), (((0, 0),) * 3 + ((0, 1),), 2)]
)
def test_enemy_cannot_move_and_shoot_during_core_withdrawal(route, expected):
    turn = make_turn(
        objects=[core(), unit(50, "RANGER", controlled=False, position=(0, -4))]
    )
    assert (
        maximum_core_damage(
            route,
            tuple(e for e in turn.visible_enemies if isinstance(e, UnitView)),
            set(),
        )
        == expected
    )


def test_rock_only_shot_ray_crosses_units_and_core():
    strategy = _strategy()
    assert strategy._line_of_fire_clear((0, -2), (0, 1), set(), {(0, -1), (0, 0)})
    assert not strategy._line_of_fire_clear((0, -2), (0, 1), {(0, -1)}, {(0, 0)})


def test_next_tick_threat_is_not_current_damage():
    strategy = _strategy()
    turn = make_turn(
        objects=[core(), unit(50, "RANGER", controlled=False, position=(0, -4))]
    )
    strategy.memory.enemy_position_history[object_id(50)] = [
        (98, (0, -6)),
        (99, (0, -5)),
        (100, (0, -4)),
    ]
    assert strategy._visible_attackers_at((0, 0), turn, set()) == ()
    assert len(strategy._future_attackers_at((0, 0), turn, set())) == 1


def test_replacements_avoid_a_recent_lethal_post_and_transit_cell(tmp_path):
    strategy = _strategy()
    strategy.memory.observe(
        make_turn(tick=99, objects=[core(), unit(9, "RANGER", position=(-3, -4))])
    )
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(-2, -4)),
            unit(3, "RANGER", position=(1, 0)),
            unit(4, "RANGER", position=(0, 1)),
            unit(5, "RANGER", position=(1, 1)),
            unit(50, "RANGER", controlled=False, position=(-8, -2)),
        ],
        events=[
            {
                "event_id": object_id(900),
                "tick": 99,
                "event_type": "UNIT_DAMAGED",
                "reason_code": "ATTACK",
                "target_id": object_id(9),
                "position": [-3, -4],
                "values": {"damage": 2, "hp": 0},
            }
        ],
    )
    strategy.memory.observe(turn)
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    context = _context(strategy, turn)
    strategy._prepare_layered_defense(context)
    assert (-3, -4) not in context.layered_assignments.values()

    # The evidence also forbids passing through the cell towards another
    # post; it is temporary tactical danger, not a permanent rock.
    context.layered_assignments = {UUID(int=2): (-4, -4)}
    context.layered_steps.clear()
    strategy._prepare_layered_steps(
        context,
        tuple(enemy for enemy in turn.visible_enemies if isinstance(enemy, UnitView)),
    )
    strategy._move_layered_ranger(turn.rangers[0], context)
    action = turn.plan.unit_actions[UUID(int=2)]
    assert isinstance(action, MoveAction)
    assert add((-2, -4), action.direction) != (-3, -4)
    assert (-3, -4) not in strategy.memory.obstacles


def test_recent_incoming_fire_keeps_a_current_winning_volley():
    friends = [
        unit(2, "RANGER", position=(0, -4)),
        unit(3, "RANGER", position=(2, -4)),
        unit(4, "RANGER", position=(0, -3)),
        unit(5, "RANGER", position=(2, 0)),
    ]
    foes = [
        unit(50, "RANGER", controlled=False, position=(0, -6)),
        unit(51, "RANGER", controlled=False, position=(0, -6), hp=1),
    ]
    encounter = Encounter(friends, foes)
    strategy = _strategy({o["id"]: "defense" for o in friends})
    strategy.memory.incoming_damage[(0, -4)] = (99, 2)
    plan, _ = _plan_encounter(strategy, encounter)
    assert plan.unit_actions[UUID(int=2)].type == "SHOOT"
    encounter.settle(
        plan,
        {
            foe["id"]: {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [0, -4],
            }
            for foe in foes
        },
    )
    assert encounter.kills == 2
    assert encounter.losses == 1


def test_retreat_state_survives_restart_and_resets_with_core(tmp_path):
    strategy = _strategy()
    memory = strategy.memory
    memory.layered_defense_core_id = object_id(1)
    memory.layered_defense_anchor = (0, 1)
    memory.layered_defense_posts = {object_id(2): (0, -3)}
    memory.layered_defense_ready_sides = {"UP"}
    memory.layered_retreat_destination = (0, 1)
    path = tmp_path / "memory.json"
    memory.save(path)
    loaded = WorldMemory.load(path)
    assert loaded.layered_defense_posts == memory.layered_defense_posts
    assert loaded.layered_retreat_destination == (0, 1)
    assert loaded.layered_defense_ready_sides == {"UP"}
    strategy = AggressiveStrategy(loaded, strategy.config)
    turn = make_turn(objects=[core(99, position=(20, 20))])
    strategy._prepare_layered_defense(_context(strategy, turn))
    assert not loaded.layered_defense_posts
    assert loaded.layered_retreat_destination is None


def test_core_does_not_start_a_fatal_four_tick_move():
    strategy = _strategy()
    turn = make_turn(
        objects=[
            core(hp=2, shield=0),
            unit(50, "RANGER", controlled=False, position=(0, -3)),
        ]
    )
    context = _context(strategy, turn)
    assert (
        strategy._layered_core_retreat(
            context,
            tuple(e for e in turn.visible_enemies if isinstance(e, UnitView)),
            set(),
        )
        is None
    )


class Encounter:
    """Small independent movement/combat/healing settlement for regression."""

    def __init__(self, friends, foes, *, resources=1):
        self.friends = {o["id"]: deepcopy(o) for o in friends}
        self.foes = {o["id"]: deepcopy(o) for o in foes}
        self.core = core()
        self.resources = resources
        self.progress = 0
        self.direction = None
        self.tick = 100
        self.kills = 0
        self.losses = 0
        self.deposits = 0
        self.heals = 0

    def turn(self):
        c = deepcopy(self.core)
        if self.progress:
            assert self.direction is not None
            c.update(
                state="MOVING",
                move_direction=self.direction.value,
                destination=list(add(tuple(c["position"]), self.direction)),
                move_progress=self.progress,
                move_required_ticks=4,
            )
        return make_turn(
            tick=self.tick,
            resources=self.resources,
            objects=[c, *self.friends.values(), *self.foes.values()],
        )

    def settle(self, plan, enemy_actions):
        if self.progress:
            assert self.direction is not None
            self.progress += 1
            if self.progress == 4:
                self.core["position"] = list(
                    add(tuple(self.core["position"]), self.direction)
                )
                self.progress = 0
        actions = {str(k): v.model_dump() for k, v in plan.unit_actions.items()}
        for fleet, commands in ((self.friends, actions), (self.foes, enemy_actions)):
            for actor_id, action in commands.items():
                if actor_id not in fleet:
                    continue
                actor = fleet[actor_id]
                if action["type"] == "MOVE":
                    actor["position"] = list(
                        add(tuple(actor["position"]), Direction(action["direction"]))
                    )
                elif (
                    action["type"] == "DEPOSIT"
                    and actor["position"] == self.core["position"]
                    and not self.progress
                ):
                    self.resources += actor.get("cargo", 0)
                    actor["cargo"] = 0
                    self.deposits += 1
        occupancy = Counter(tuple(o["position"]) for o in self.friends.values())
        occupancy[tuple(self.core["position"])] += 1
        assert max(occupancy.values()) <= 2
        damage = defaultdict(int)
        for fleet, victims, commands in (
            (self.friends, self.foes, actions),
            (self.foes, {**self.friends, self.core["id"]: self.core}, enemy_actions),
        ):
            for actor_id, action in commands.items():
                if actor_id not in fleet or action["type"] != "SHOOT":
                    continue
                origin = tuple(fleet[actor_id]["position"])
                cell = tuple(action["expected_cell"])
                if not line_of_fire(origin, cell, set()):
                    continue
                candidates = [
                    o for o in victims.values() if tuple(o["position"]) == cell
                ]
                target_id = action.get("target_id")
                if target_id:
                    candidates = [o for o in candidates if o["id"] == str(target_id)]
                if candidates:
                    target = min(candidates, key=lambda o: (o["hp"], o["id"]))
                    damage[target["id"]] += 1
        for fleet, count in ((self.friends, "losses"), (self.foes, "kills")):
            for actor_id in list(fleet):
                fleet[actor_id]["hp"] -= damage[actor_id]
                if fleet[actor_id]["hp"] <= 0:
                    del fleet[actor_id]
                    setattr(self, count, getattr(self, count) + 1)
        hit = damage[self.core["id"]]
        shield_hit = min(self.core["shield"], hit)
        self.core["shield"] -= shield_hit
        self.core["hp"] -= hit - shield_hit
        assert self.core["hp"] > 0
        for actor_id, action in actions.items():
            actor = self.friends.get(actor_id)
            if (
                actor
                and action["type"] == "HEAL"
                and not self.progress
                and actor["position"] == self.core["position"]
            ):
                amount = min(2 - actor["hp"], self.resources)
                actor["hp"] += amount
                self.resources -= amount
                self.heals += int(amount > 0)
        action = plan.core_action
        if action and action.type == "START_MOVE":
            self.direction = action.direction
            self.progress = 1
        self.tick += 1


def _plan_encounter(strategy, encounter):
    turn = encounter.turn()
    context = _context(strategy, turn)
    strategy._prepare_layered_defense(context)
    context.core_queue_head_id = strategy._select_stationary_core_queue_head(context)
    strategy._prepare_core_departure(context)
    for ranger in turn.rangers:
        if not strategy._has_unit_action(ranger, context):
            strategy._decide_ranger(
                ranger, context, strategy.memory.recent_enemies(turn.tick, 32)
            )
    for worker in turn.workers:
        if not strategy._has_unit_action(worker, context):
            strategy._decide_worker(worker, context)
    strategy._decide_core(context)
    return turn.plan, context


def test_front_rank_loss_costs_enemy_units_then_core_retreats_and_rebuilds_depth():
    friends = [
        unit(2, "RANGER", position=(0, -4)),
        unit(3, "RANGER", position=(2, -4)),
        unit(4, "RANGER", position=(0, -3)),
        unit(5, "RANGER", position=(3, -3)),
        unit(6, "RANGER", position=(0, 3)),
        unit(7, "RANGER", position=(2, 3)),
    ]
    foes = [
        unit(50, "RANGER", controlled=False, position=(0, -6)),
        unit(51, "RANGER", controlled=False, position=(0, -6), hp=1),
    ]
    encounter = Encounter(friends, foes)
    strategy = _strategy({o["id"]: "defense" for o in friends})
    memory = strategy.memory
    memory.layered_defense_core_id = object_id(1)
    memory.layered_defense_anchor = (0, 0)
    memory.layered_defense_posts = {o["id"]: tuple(o["position"]) for o in friends[:4]}
    plan, _context_value = _plan_encounter(strategy, encounter)
    opponent = {
        o["id"]: {"type": "SHOOT", "target_id": object_id(2), "expected_cell": [0, -4]}
        for o in foes
    }
    encounter.settle(plan, opponent)
    assert encounter.kills == 2
    assert encounter.losses == 1
    assert encounter.core["hp"] == 5 and encounter.core["shield"] == 5
    encounter.foes = {o["id"]: deepcopy(o) for o in foes}
    plan, context = _plan_encounter(strategy, encounter)
    assert plan.core_action.type == "START_MOVE"
    assert plan.core_action.direction is Direction.DOWN
    assert context.layered_assignments[UUID(int=4)] == (0, -3)
    assert context.layered_assignments[UUID(int=5)] == (3, -3)
    assert all(position[1] != -4 for position in context.layered_assignments.values())
    encounter.settle(plan, {})
    encounter.foes = {
        object_id(99): unit(99, "RANGER", controlled=False, position=(0, -8))
    }
    for _ in range(12):
        plan, _context_value = _plan_encounter(strategy, encounter)
        encounter.settle(plan, {})
    assert encounter.core["position"] == [0, 1]
    actual = [tuple(o["position"]) for o in encounter.friends.values()]
    for depth in (FRONT_DEPTH, REAR_DEPTH):
        assert (
            sum(projection((0, 1), cell, Direction.UP) == depth for cell in actual) >= 2
        )


def test_sheltered_worker_deposits_and_wounded_worker_actually_heals():
    friends = [
        unit(2, "RANGER", position=(-1, 1)),
        unit(3, "WORKER", position=(0, 0), cargo=1),
        unit(4, "WORKER", position=(0, 1), hp=1),
    ]
    encounter = Encounter(
        friends, [unit(50, "RANGER", controlled=False, position=(0, -5))], resources=1
    )
    strategy = _strategy({object_id(2): "defense"})
    for _ in range(8):
        plan, _context_value = _plan_encounter(strategy, encounter)
        encounter.settle(plan, {})
    assert encounter.deposits >= 1
    assert encounter.heals >= 1
    assert encounter.friends[object_id(4)]["hp"] == 2


def test_critical_ranger_does_not_churn_while_healing_slot_is_occupied():
    strategy = _strategy({object_id(2): "defense", object_id(3): "defense"})
    for tick in range(100, 108):
        turn = make_turn(
            tick=tick,
            resources=1,
            objects=[
                core(),
                unit(2, "RANGER", position=(0, 1), hp=1),
                unit(3, "RANGER", hp=1),
            ],
        )
        context = _context(strategy, turn)
        strategy._decide_ranger(turn.rangers[0], context, ())
        assert turn.plan.unit_actions[UUID(int=2)].type == "WAIT"


def test_core_shooter_is_not_evicted_for_workers_that_will_be_evacuated():
    strategy = _strategy({object_id(2): "defense"})
    turn = make_turn(
        resources=1,
        objects=[
            core(),
            unit(2, "RANGER"),
            unit(3, "WORKER", position=(-3, 1), cargo=1),
            unit(4, "WORKER", position=(-3, 1), cargo=1),
            unit(50, "RANGER", controlled=False, position=(0, -4)),
        ],
    )
    context = _context(strategy, turn)
    strategy._prepare_core_departure(context)
    assert not strategy._has_unit_action(turn.rangers[0], context)
    strategy._decide_ranger(turn.rangers[0], context, ())
    assert turn.plan.unit_actions[UUID(int=2)].type == "SHOOT"


def test_crowded_rear_cell_is_cleared_for_core_retreat_without_moving_shooters():
    friends = [
        unit(2, "RANGER", position=(0, -3)),
        unit(3, "RANGER", position=(3, -3)),
        unit(4, "RANGER", position=(0, 1)),
        unit(5, "RANGER", position=(0, 1)),
    ]
    strategy = _strategy({o["id"]: "defense" for o in friends})
    memory = strategy.memory
    memory.layered_defense_core_id = object_id(1)
    memory.layered_defense_anchor = (0, 0)
    memory.layered_defense_posts = {o["id"]: tuple(o["position"]) for o in friends[:2]}
    memory.layered_defense_ready_sides = {"UP"}
    turn = make_turn(
        objects=[
            core(),
            *friends,
            unit(50, "RANGER", controlled=False, position=(0, -6)),
        ]
    )
    context = _context(strategy, turn)
    strategy._prepare_layered_defense(context)
    assert context.core_escape_direction is Direction.DOWN
    assert strategy._predicted_friendly_occupancy((0, 1), context) == 1
    assert not strategy._has_unit_action(turn.rangers[0], context)
    assert not strategy._has_unit_action(turn.rangers[1], context)


def test_full_controller_reforms_northern_front_instead_of_rear_wandering():
    positions = [(x, y) for y in (1, 2, 3, 4) for x in (-2, -1, 0, 1)]
    turn = make_turn(
        resources=1,
        objects=[
            core(),
            *[
                unit(n + 2, "RANGER", position=position)
                for n, position in enumerate(positions)
            ],
            unit(50, "RANGER", controlled=False, position=(0, -3)),
            unit(51, "RANGER", controlled=False, position=(0, -3), hp=1),
        ],
    )
    strategy = _strategy({object_id(n + 2): "defense" for n in range(len(positions))})
    report = strategy.decide(turn)
    assert isinstance(turn.plan.core_action, StartMoveAction)
    assert turn.plan.core_action.direction is Direction.DOWN
    assert any(
        d.action == "MOVE"
        and "two defensive firing ranks" in d.reason
        and d.target[1] < 0
        for d in report.decisions
    )


def test_core_retreat_clears_a_two_link_crowded_departure_chain():
    friends = [
        unit(2, "RANGER", position=(0, -3)),
        unit(3, "RANGER", position=(3, -3)),
        unit(4, "RANGER", position=(0, 1)),
        unit(5, "RANGER", position=(0, 1)),
        unit(6, "RANGER", position=(0, 2)),
        unit(7, "RANGER", position=(0, 2)),
    ]
    strategy = _strategy({o["id"]: "defense" for o in friends})
    memory = strategy.memory
    memory.layered_defense_core_id = object_id(1)
    memory.layered_defense_anchor = (0, 0)
    memory.layered_defense_posts = {o["id"]: tuple(o["position"]) for o in friends[:2]}
    memory.layered_defense_ready_sides = {"UP"}
    turn = make_turn(
        objects=[
            core(),
            *friends,
            unit(50, "RANGER", controlled=False, position=(0, -6)),
        ],
        obstacles=[(-1, 1), (1, 1), (-1, 2), (1, 2)],
    )
    context = _context(strategy, turn)
    strategy._prepare_layered_defense(context)

    assert context.core_escape_direction is Direction.DOWN
    assert strategy._predicted_friendly_occupancy((0, 1), context) == 1
    assert strategy._predicted_friendly_occupancy((0, 2), context) == 2
    assert strategy._predicted_friendly_occupancy((0, 3), context) == 1
    assert len(turn.plan.unit_actions) == 2
    assert all(
        isinstance(action, MoveAction) and action.direction is Direction.DOWN
        for action in turn.plan.unit_actions.values()
    )


def test_surplus_volley_fire_uses_an_unfinished_visible_target():
    strategy = _strategy({object_id(2): "defense", object_id(3): "defense"})
    turn = make_turn(
        objects=[
            core(),
            unit(2, "RANGER", position=(0, -3)),
            unit(3, "RANGER", position=(0, -4)),
            unit(50, "RANGER", controlled=False, position=(0, -6), hp=1),
            unit(51, "RANGER", controlled=False, position=(2, -6)),
        ]
    )
    context = _context(strategy, turn)
    strategy._prepare_defensive_trade_shots(context)

    assert context.defensive_trade_shots[UUID(int=2)].id == UUID(int=50)
    assert context.defensive_trade_shots[UUID(int=3)].id == UUID(int=51)
