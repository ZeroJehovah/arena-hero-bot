"""Real settlement and failure boundaries for bounded local clearance."""

from copy import deepcopy

import pytest

from arena_hero_bot.counterattack import AssaultSlot
from arena_hero_bot.geometry import add, line_of_fire
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.skirmish import LOST_TICKS, MAX_TICKS, Skirmish
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig, _TurnContext

from .factories import core, make_turn, object_id, unit
from .test_layered_defense import Encounter


def _setup(*, target_hp=2, opponents=2, count=6):
    positions = [(-1, -4), (1, -4), (-1, -5), (1, -5), (-4, 2), (4, 2)]
    friends = [
        unit(2 + i, "RANGER", position=p) for i, p in enumerate(positions[:count])
    ]
    foes = [
        unit(50 + i, "RANGER", position=(0, -7), controlled=False, hp=target_hp)
        for i in range(opponents)
    ]
    memory = WorldMemory(
        unit_roles={o["id"]: "defense" for o in friends},
        unit_roles_initialized=True,
        core_home_position=(0, 0),
    )
    memory.observe(make_turn(tick=90, objects=[core(), *friends, *foes]))
    memory.observe(make_turn(tick=98, objects=[core(), *friends, *foes]))
    memory.observe(make_turn(tick=99, objects=[core(), *friends, *foes]))
    strategy = AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))
    return strategy, Encounter(friends, foes, resources=0)


def _context(strategy, turn):
    strategy.memory.observe(turn)
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={u.position for u in turn.units},
        enemy_positions={e.position for e in turn.visible_enemies},
        threat=strategy._combat_policy.assess(turn, strategy.memory.obstacles),
    )


def _hostile_volley(encounter, plan, *, spread=False):
    positions = {
        uid: tuple(actor["position"]) for uid, actor in encounter.friends.items()
    }
    for uid, action in plan.unit_actions.items():
        if action.type == "MOVE":
            positions[str(uid)] = add(positions[str(uid)], action.direction)
    orders = {}
    for index, (uid, foe) in enumerate(encounter.foes.items()):
        victims = [
            key
            for key, position in positions.items()
            if line_of_fire(tuple(foe["position"]), position, set())
        ]
        if victims:
            victims.sort(key=lambda k: (encounter.friends[k]["hp"], k))
            target = victims[index % len(victims)] if spread else victims[0]
            orders[uid] = {
                "type": "SHOOT",
                "target_id": target,
                "expected_cell": list(positions[target]),
            }
    return orders


@pytest.mark.parametrize("spread", [False, True])
def test_four_rangers_survive_ingress_then_clear_two_full_targets(tmp_path, spread):
    strategy, encounter = _setup()
    initial_ids = None
    start = None
    saw_ingress_loss = False
    for offset in range(12):
        turn = encounter.turn()
        context = _context(strategy, turn)
        strategy._prepare_skirmish(context)
        operation = strategy.memory.skirmish
        if operation is not None:
            if initial_ids is None:
                initial_ids = set(operation.slots)
                start = operation.started
                assert len(initial_ids) == 4
            assert set(operation.slots) <= initial_ids
            assert operation.started == start
        before = encounter.losses
        encounter.settle(
            turn.plan, _hostile_volley(encounter, turn.plan, spread=spread)
        )
        if operation is not None and operation.entered == turn.tick:
            saw_ingress_loss = encounter.losses > before
        if offset == 1:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
        if not encounter.foes:
            break
    assert saw_ingress_loss is not spread
    assert encounter.kills == 2
    assert encounter.losses <= 2
    assert encounter.core["hp"] == encounter.core["shield"] == 5


def test_full_decide_assigns_clearance_before_layered_posts_and_cover():
    strategy, encounter = _setup(target_hp=1, opponents=1)
    turn = encounter.turn()

    report = strategy.decide(turn)

    operation = strategy.memory.skirmish
    assert operation is not None
    assert len(operation.slots) == 2
    assert all(
        "local clearance" in d.reason
        for d in report.decisions
        if d.actor_id in operation.slots
    )


@pytest.mark.parametrize("count", [2, 3])
def test_clearance_never_spends_the_last_two_healthy_guards(count):
    strategy, encounter = _setup(target_hp=1, opponents=1, count=count)
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    assert strategy.memory.skirmish is None
    assert not turn.plan.unit_actions


@pytest.mark.parametrize("cause", ["shield", "core_id", "moving", "timeout", "death"])
def test_clearance_cancels_without_replacing_members_or_resetting_time(cause):
    strategy, encounter = _setup(target_hp=1, opponents=1)
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    operation = strategy.memory.skirmish
    assert operation is not None
    if cause == "shield":
        encounter.core["shield"] = 4
    elif cause == "core_id":
        encounter.core["id"] = object_id(999)
    elif cause == "moving":
        encounter.core["position"] = [1, 0]
    elif cause == "timeout":
        encounter.tick += MAX_TICKS
    else:
        del encounter.friends[next(iter(operation.slots))]
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    assert strategy.memory.skirmish is None
    assert strategy.memory.skirmish_cooldown_until > turn.tick
    assert not turn.plan.unit_actions


def test_unseen_target_allows_safe_assembly_but_never_blind_ingress(tmp_path):
    strategy, encounter = _setup()
    for number, position in zip(
        range(2, 6), [(-4, -3), (-4, -2), (4, -3), (4, -2)], strict=True
    ):
        encounter.friends[object_id(number)]["position"] = list(position)
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    operation = strategy.memory.skirmish
    assert operation is not None
    assert operation.entered is None
    started = operation.started
    slots = deepcopy(operation.slots)
    for uid, slot in slots.items():
        encounter.friends[uid]["position"] = list(slot.stage)
    encounter.foes.clear()
    encounter.tick += 1
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    assert strategy.memory.skirmish is not None
    assert strategy.memory.skirmish.started == started
    assert all(action.type == "WAIT" for action in turn.plan.unit_actions.values())
    encounter.tick = started + LOST_TICKS + 1
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    assert strategy.memory.skirmish is None


def test_ingress_rolls_back_all_actions_when_one_member_cannot_enter(monkeypatch):
    strategy, encounter = _setup(target_hp=1, opponents=1)
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    operation = strategy.memory.skirmish
    assert operation is not None
    for uid, slot in operation.slots.items():
        encounter.friends[uid]["position"] = list(slot.stage)
    turn = encounter.turn()
    context = _context(strategy, turn)
    members = tuple(r for r in turn.rangers if str(r.id) in operation.slots)
    original = strategy._queue_move
    calls = 0

    def deny_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        return calls != 2 and original(*args, **kwargs)

    monkeypatch.setattr(strategy, "_queue_move", deny_second)
    assert not strategy._skirmish_enter(context, members, operation)
    assert not turn.plan.unit_actions
    assert not context.arrival_counts and not context.departure_counts
    assert not context.report.decisions


def test_non_target_hidden_shooter_blocks_a_clearance_entry():
    strategy, encounter = _setup(target_hp=1, opponents=1)
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    operation = strategy.memory.skirmish
    assert operation is not None
    stage = next(iter(operation.slots.values())).stage
    hidden = unit(80, "RANGER", controlled=False, position=(stage[0], stage[1] + 2))
    # A remembered combatant remains a real safety constraint after visibility
    # disappears, even though the selected target is still visible.
    strategy.memory.observe(
        make_turn(
            tick=101,
            objects=[
                encounter.core,
                *encounter.friends.values(),
                *encounter.foes.values(),
                hidden,
            ],
        )
    )
    encounter.tick = 102
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    assert strategy.memory.skirmish is None


def test_shooting_uses_actual_arrivals_and_never_sends_a_late_replacement():
    strategy, encounter = _setup(target_hp=1, opponents=1)
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    operation = strategy.memory.skirmish
    assert operation is not None
    operation.entered = turn.tick
    # Neither mover actually arrived. A nominal planned pair is not firepower.
    encounter.tick += 1
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    assert strategy.memory.skirmish is None
    assert not turn.plan.unit_actions


def test_skirmish_serialization_preserves_identity_slots_and_deadline():
    value = Skirmish(
        object_id(1),
        (0, 0),
        (object_id(50),),
        (0, -7),
        100,
        102,
        {object_id(2): AssaultSlot((1, -4), (0, -4))},
        {object_id(2): (1, -3)},
        103,
    )
    assert Skirmish.from_dict(value.to_dict()) == value


def test_surplus_wounded_member_escapes_while_healthy_member_finishes():
    strategy, encounter = _setup(target_hp=1, opponents=1)
    for _ in range(4):
        turn = encounter.turn()
        strategy.decide(turn)
        encounter.settle(turn.plan, _hostile_volley(encounter, turn.plan))
        if not encounter.foes:
            break
    assert encounter.kills == 1
    assert encounter.losses == 0


def test_assembly_yields_to_a_new_current_shot():
    strategy, encounter = _setup(target_hp=1, opponents=1)
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    operation = strategy.memory.skirmish
    assert operation is not None
    operation.entered = None
    member_id = next(iter(operation.slots))
    encounter.friends[member_id]["position"] = [0, -4]
    encounter.tick += 1
    turn = encounter.turn()
    context = _context(strategy, turn)
    strategy._prepare_skirmish(context)
    assert strategy.memory.skirmish is None
    member = next(r for r in turn.rangers if str(r.id) == member_id)
    strategy._decide_ranger(member, context, ())
    assert turn.plan.unit_actions[member.id].type == "SHOOT"


def test_two_full_guards_remain_when_core_has_affordable_production():
    strategy, encounter = _setup(target_hp=1, opponents=1)
    encounter.resources = 150
    turn = encounter.turn()
    strategy.decide(turn)
    assert strategy.memory.skirmish is not None
    assert turn.plan.core_action.type == "SPAWN"


@pytest.mark.parametrize("failure", ["extra_enemy", "vanguard", "blocked_entry"])
def test_new_entry_constraints_abort_or_hold_before_any_member_enters(failure):
    strategy, encounter = _setup()
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    operation = strategy.memory.skirmish
    assert operation is not None
    operation.entered = None
    cell = next(iter(operation.slots.values())).firing
    if failure == "blocked_entry":
        encounter.friends[object_id(90)] = unit(90, "WORKER", position=cell)
        encounter.friends[object_id(91)] = unit(91, "WORKER", position=cell)
    else:
        encounter.foes[object_id(90)] = unit(
            90,
            "RANGER" if failure == "extra_enemy" else "VANGUARD",
            position=(cell[0], cell[1] + 1),
            controlled=False,
        )
    encounter.tick += 1
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    assert not any(action.type == "MOVE" for action in turn.plan.unit_actions.values())
    assert strategy.memory.skirmish is None or strategy.memory.skirmish.entered is None


@pytest.mark.parametrize("target_position", [(-9, 0), (-8, -1)])
def test_reappearing_wounded_target_is_cleared_with_the_original_group_budget(
    tmp_path, target_position
):
    friends = [
        unit(2, "RANGER", position=(-6, 1)),
        unit(3, "RANGER", position=(-6, 1)),
        unit(4, "RANGER", position=(-3, 3)),
        unit(5, "RANGER", position=(3, 3)),
    ]
    foe = unit(50, "RANGER", controlled=False, position=target_position, hp=1)
    encounter = Encounter(friends, [foe], resources=0)
    encounter.tick = 107
    memory = WorldMemory(
        unit_roles={actor["id"]: "defense" for actor in friends},
        unit_roles_initialized=True,
        core_home_position=(0, 0),
    )
    memory.skirmish = Skirmish(
        object_id(1),
        (0, 0),
        (object_id(50),),
        (-9, -1),
        100,
        100,
        {object_id(n): AssaultSlot((-6, 1), (-7, 1)) for n in (2, 3)},
        {object_id(2): (-4, 0), object_id(3): (-2, 2)},
    )
    strategy = AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))
    for offset in range(5):
        turn = encounter.turn()
        strategy.decide(turn)
        operation = strategy.memory.skirmish
        if operation is not None:
            assert operation.started == 100
            assert set(operation.slots) == {object_id(2), object_id(3)}
        encounter.settle(turn.plan, _hostile_volley(encounter, turn.plan))
        if not encounter.foes:
            break
        if offset == 0:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    assert encounter.kills == 1
    assert encounter.losses == 0
    assert encounter.tick <= 100 + MAX_TICKS
