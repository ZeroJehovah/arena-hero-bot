"""Entrenched-force response: real paired ingress, bounded state and volleys."""

from collections import Counter
from copy import deepcopy
from uuid import UUID

import pytest
from arena_hero import Direction, Turn, UnitView

from arena_hero_bot import counterattack as counter
from arena_hero_bot.geometry import add, line_of_fire
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.runtime import _identity_snapshot

from .factories import core, make_turn, object_id, unit
from .test_layered_defense import Encounter, _context, _strategy


def _friends(count=12):
    positions = [
        (2, -5),
        (3, -5),
        (4, -5),
        (2, -4),
        (3, -4),
        (4, -4),
        (-2, 0),
        (-1, 0),
        (-2, 1),
        (-1, 1),
        (1, 0),
        (1, 1),
    ]
    return [unit(n + 2, "RANGER", position=p) for n, p in enumerate(positions[:count])]


def _foes(count=2, position=(0, -6)):
    return [
        unit(50 + n, "RANGER", controlled=False, position=position)
        for n in range(count)
    ]


def _views(objects):
    return tuple(UnitView.model_validate(obj) for obj in objects)


def _warm(strategy, turn, ticks=counter.STANDOFF_TICKS):
    for tick in range(turn.tick - ticks, turn.tick):
        strategy.memory.counterattack.observe(
            Turn(tick=tick, state=turn.state, submitter=turn._submitter)
        )


def _start(*, friends=None, foes=None, c=None, ticks=counter.STANDOFF_TICKS):
    turn = make_turn(
        objects=[c or core(), *(friends or _friends()), *(foes or _foes())]
    )
    strategy = _strategy()
    _warm(strategy, turn, ticks)
    context = _context(strategy, turn)
    strategy._prepare_counterattack(context)
    return strategy, turn, context


def _at_stage(strategy, *, tick=101, extra=(), change=None):
    operation = strategy.memory.counterattack.operation
    assert operation is not None
    friends = _friends()
    for actor in friends:
        if actor["id"] in operation.slots:
            actor["position"] = list(operation.slots[actor["id"]].stage)
    if change:
        change(friends)
    turn = make_turn(tick=tick, objects=[core(), *friends, *_foes(), *extra])
    return turn, _context(strategy, turn)


def test_same_ray_is_legal_but_every_ingress_cell_contains_two_rangers():
    turn = make_turn(objects=[core(position=(8, 0)), *_foes(1, (0, 0))])
    targets = tuple(e for e in turn.visible_enemies if isinstance(e, UnitView))
    positions = {
        UUID(int=i): (p, 2) for i, p in enumerate([(2, 0)] * 2 + [(3, 0)] * 2, 2)
    }
    assert counter.entry_viable(positions, targets, targets, set())
    # A at (2, 1) and B at (3, 1) can enter these two cells on one Tick.
    assert not line_of_fire((2, 1), (0, 0), set())
    assert not line_of_fire((3, 1), (0, 0), set())
    positions[UUID(int=2)] = ((0, 2), 2)
    assert not counter.entry_viable(positions, targets, targets, set())


def test_two_enemy_occupants_need_more_arrival_survivors_than_one():
    targets = _views(_foes(2, (0, 0)))
    positions = {
        UUID(int=i): (p, 2) for i, p in enumerate([(2, 0)] * 2 + [(3, 0)] * 2, 2)
    }
    assert not counter.entry_viable(positions, targets, targets, set())
    positions.update({UUID(int=6): ((0, 3), 2), UUID(int=7): ((0, 3), 2)})
    assert counter.entry_viable(positions, targets, targets, set())


def test_other_current_rays_and_future_reinforcements_are_counted():
    target = _views(_foes(1, (0, 0)))
    positions = {
        UUID(int=i): (p, 2) for i, p in enumerate([(2, 0)] * 2 + [(3, 0)] * 2, 2)
    }
    side = _views([unit(80, "RANGER", controlled=False, position=(3, 3))])
    assert not counter.entry_viable(positions, target, (*target, *side), set())
    future = _views(
        [unit(n, "RANGER", controlled=False, position=(5, 1)) for n in (80, 81)]
    )
    assert all(
        not counter.can_attack(e, cell, set())
        for e in future
        for cell, _ in positions.values()
    )
    assert (
        counter.reachable_opposition(
            tuple(p for p, _ in positions.values()), (*target, *future), set()
        )
        == 3
    )
    assert not counter.entry_viable(positions, target, (*target, *future), set())


def test_contact_and_fixed_operation_survive_restart_without_refreshing_budgets(
    tmp_path,
):
    strategy, turn, _ = _start()
    operation = deepcopy(strategy.memory.counterattack.operation)
    assert operation is not None
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    loaded = WorldMemory.load(path)
    assert loaded.counterattack.operation == operation
    before = loaded.counterattack.contacts[object_id(50)].since
    loaded.observe(
        Turn(tick=turn.tick + 1, state=turn.state, submitter=turn._submitter)
    )
    assert loaded.counterattack.contacts[object_id(50)].since == before
    assert loaded.counterattack.operation is not None
    assert loaded.counterattack.operation.started == operation.started
    snapshot = _identity_snapshot(loaded)
    loaded.counterattack.finish(102, "done")
    assert snapshot["counterattack"]["operation"] is not None
    loaded.save(path)
    assert WorldMemory.load(path).counterattack.cooldown_until == 134


def test_observation_gaps_displacement_and_core_identity_reset_contact():
    state = counter.CounterattackState()
    state.observe(make_turn(tick=10, objects=[core(), *_foes()]))
    state.observe(make_turn(tick=17, objects=[core(), *_foes()]))
    assert state.contacts[object_id(50)].since == 10
    state.observe(make_turn(tick=26, objects=[core(), *_foes()]))
    assert state.contacts[object_id(50)].since == 26
    state.observe(make_turn(tick=27, objects=[core(), *_foes(position=(3, -6))]))
    assert state.contacts[object_id(50)].since == 27
    state.observe(make_turn(tick=28, objects=[core(9), *_foes(position=(3, -6))]))
    assert state.contacts[object_id(50)].since == 28
    state.observe(
        make_turn(tick=29, objects=[core(9, position=(1, 0)), *_foes(position=(3, -6))])
    )
    assert state.contacts[object_id(50)].since == 29
    state.observe(make_turn(tick=30, objects=[]))
    assert not state.contacts and state.operation is None


@pytest.mark.parametrize(
    "case",
    ["short", "ratio", "reserve", "too_many", "hp", "shield", "moving", "direct_core"],
)
def test_admission_rejects_short_or_unsafe_or_not_overwhelming_force(case):
    friends, foes, c, ticks = _friends(), _foes(), core(), counter.STANDOFF_TICKS
    if case == "short":
        ticks -= 1
    elif case == "ratio":
        friends = _friends(5)
    elif case == "reserve":
        friends, foes = _friends(7), _foes(1)
    elif case == "too_many":
        foes += [
            unit(n, "RANGER", controlled=False, position=(8, n - 53))
            for n in range(52, 55)
        ]
    elif case in {"hp", "shield"}:
        c[case] = 4
    elif case == "moving":
        c.update(
            state="MOVING",
            move_direction="UP",
            destination=[0, -1],
            move_progress=1,
            move_required_ticks=4,
        )
    elif case == "direct_core":
        foes = _foes(position=(0, -3))
    strategy, _, context = _start(friends=friends, foes=foes, c=c, ticks=ticks)
    assert strategy.memory.counterattack.operation is None
    assert not context.counterattack_ids


def test_formation_assembles_safely_and_only_enters_when_every_member_is_ready():
    strategy, turn, context = _start()
    operation = strategy.memory.counterattack.operation
    assert operation is not None and operation.entered is None
    assert len(operation.slots) == 6
    assert counter.paired_cells(tuple(s.firing for s in operation.slots.values()))
    assert len({slot.stage for slot in operation.slots.values()}) == 6
    assert len(turn.rangers) - len(operation.slots) >= 4
    for member in turn.rangers:
        action = turn.plan.unit_actions.get(member.id)
        if action and action.type == "MOVE":
            destination = add(member.position, action.direction)
            assert all(
                not counter.can_attack(e, destination, set())
                for e in turn.visible_enemies
            )
    arrived, context = _at_stage(strategy)
    strategy._prepare_counterattack(context)
    assert operation.entered == arrived.tick
    destinations = Counter()
    for member in arrived.rangers:
        if str(member.id) in operation.slots:
            action = arrived.plan.unit_actions[member.id]
            assert action.type == "MOVE"
            destinations[add(member.position, action.direction)] += 1
    assert len(destinations) == 3 and set(destinations.values()) == {2}


def test_blocked_ingress_waits_as_one_group_and_does_not_renew_deadline():
    strategy, _, _ = _start()
    operation = strategy.memory.counterattack.operation
    firing = next(iter(operation.slots.values())).firing
    turn, context = _at_stage(strategy, extra=[unit(99, "WORKER", position=firing)])
    strategy._prepare_counterattack(context)
    assert operation.entered is None
    assert all(turn.plan.unit_actions[UUID(k)].type == "WAIT" for k in operation.slots)
    later, context = _at_stage(
        strategy, tick=operation.started + counter.ASSEMBLY_TICKS
    )
    strategy._prepare_counterattack(context)
    assert strategy.memory.counterattack.operation is None
    assert (
        strategy.memory.counterattack.cooldown_until
        == later.tick + counter.COOLDOWN_TICKS
    )


def test_partial_queue_failure_rolls_back_every_ingress_action_and_ledger(monkeypatch):
    strategy, _, _ = _start()
    turn, context = _at_stage(strategy)
    original = strategy._queue_move
    calls = 0

    def fail_third(*args, **kwargs):
        nonlocal calls
        calls += 1
        return False if calls == 3 else original(*args, **kwargs)

    monkeypatch.setattr(strategy, "_queue_move", fail_third)
    before_targets = dict(strategy.memory.pending_move_targets)
    strategy._prepare_counterattack(context)
    assert calls == 3
    assert all(action.type == "WAIT" for action in turn.plan.unit_actions.values())
    assert not context.arrival_counts and not context.departure_counts
    assert strategy.memory.pending_move_targets == before_targets
    assert all(d.action == "WAIT" for d in context.report.decisions)


@pytest.mark.parametrize(
    "case",
    ["wounded", "missing", "target_moved", "invisible", "reinforced", "lethal_unknown"],
)
def test_assembly_cancels_without_replacement_or_chasing(case):
    strategy, _, _ = _start()
    operation = strategy.memory.counterattack.operation
    member_id = next(iter(operation.slots))
    turn, _ = _at_stage(strategy)
    objects = [o.model_dump(mode="json") for o in turn.state.objects]
    if case == "wounded":
        next(o for o in objects if o.get("id") == member_id)["hp"] = 1
    elif case == "missing":
        objects = [o for o in objects if o.get("id") != member_id]
    elif case == "target_moved":
        next(o for o in objects if o.get("id") == object_id(50))["position"] = [1, -6]
    elif case == "invisible":
        objects = [o for o in objects if o.get("id") != object_id(50)]
    elif case == "reinforced":
        objects.extend(
            unit(n, "RANGER", controlled=False, position=(6, n - 82))
            for n in (80, 81, 82)
        )
    else:
        strategy.memory.incoming_damage[next(iter(operation.slots.values())).stage] = (
            100,
            2,
        )
    turn = make_turn(tick=101, objects=objects)
    context = _context(strategy, turn)
    strategy._prepare_counterattack(context)
    assert strategy.memory.counterattack.operation is None
    assert not context.counterattack_ids


def test_unseen_enemy_counts_until_destroyed_confirmation():
    strategy, _, _ = _start()
    invisible = make_turn(tick=101, objects=[core(), *_friends()])
    strategy.memory.observe(invisible)
    assert len(strategy._counterattack_enemies(invisible)) == 2
    event = {
        "event_id": object_id(900),
        "tick": 101,
        "event_type": "DESTRUCTION_PARTICIPATION",
        "target_id": object_id(50),
        "reason_code": "UNIT",
    }
    confirmed = make_turn(tick=102, objects=[core(), *_friends()], events=[event])
    strategy.memory.observe(confirmed)
    assert {e.id for e in strategy._counterattack_enemies(confirmed)} == {UUID(int=51)}
    assert object_id(50) not in strategy.memory.counterattack.contacts


def test_real_settlement_enters_in_pairs_then_kills_both_overlapped_targets(tmp_path):
    encounter = Encounter(_friends(), _foes(), resources=0)
    strategy = _strategy({o["id"]: "defense" for o in _friends()})
    _warm(strategy, encounter.turn())
    ingress_tick = None
    for _ in range(16):
        turn = encounter.turn()
        context = _context(strategy, turn)
        strategy._prepare_counterattack(context)
        operation = strategy.memory.counterattack.operation
        assert operation is not None
        enemy_actions = {}
        if operation.entered is not None:
            ingress_tick = ingress_tick or turn.tick
            # Opponents concentrate both arrows on one actual occupied stack.
            first_id = next(k for k in operation.slots if k in encounter.friends)
            target_cell = operation.slots[first_id].firing
            enemy_actions = {
                k: {
                    "type": "SHOOT",
                    "target_id": first_id,
                    "expected_cell": list(target_cell),
                }
                for k in encounter.foes
            }
        encounter.settle(turn.plan, enemy_actions)
        if ingress_tick is not None and encounter.tick == ingress_tick + 1:
            assert encounter.losses == 1 and encounter.kills == 0
            assert all(
                tuple(o["position"]) == operation.slots[k].firing
                for k, o in encounter.friends.items()
                if k in operation.slots
            )
            path = tmp_path / "restart.json"
            strategy.memory.save(path)
            strategy.memory = WorldMemory.load(path)
        if encounter.kills:
            break
    assert ingress_tick is not None
    assert encounter.kills == 2 and encounter.losses == 2
    assert encounter.core["hp"] == encounter.core["shield"] == 5


@pytest.mark.parametrize("failed_count", [1, 3])
def test_failed_actual_arrival_never_gets_a_late_solo_ingress(failed_count):
    strategy, _, _ = _start()
    turn, context = _at_stage(strategy)
    strategy._prepare_counterattack(context)
    operation = strategy.memory.counterattack.operation
    friends = _friends()
    for actor in friends:
        if actor["id"] in operation.slots:
            actor["position"] = list(operation.slots[actor["id"]].firing)
    failed = set(list(operation.slots)[:failed_count])
    for key in failed:
        next(o for o in friends if o["id"] == key)["position"] = list(
            operation.slots[key].stage
        )
    turn = make_turn(tick=102, objects=[core(), *friends, *_foes()])
    context = _context(strategy, turn)
    strategy._prepare_counterattack(context)
    assert not any(UUID(key) in turn.plan.unit_actions for key in failed)
    if failed_count == 3:
        assert strategy.memory.counterattack.operation is None
        assert not turn.plan.unit_actions
    else:
        assert strategy.memory.counterattack.operation is not None
        assert not failed & operation.slots.keys()
        assert len(turn.plan.unit_actions) == 5
        assert all(action.type == "SHOOT" for action in turn.plan.unit_actions.values())
        encounter = Encounter(friends, _foes(), resources=0)
        encounter.settle(turn.plan, {})
        assert encounter.kills == 2


def test_full_planner_preserves_counterattack_and_stationary_worker_service():
    friends = _friends()
    strategy = _strategy({o["id"]: "defense" for o in friends})
    turn = make_turn(objects=[core(), *friends, unit(40, "WORKER", cargo=1), *_foes()])
    _warm(strategy, turn)
    report = strategy.decide(turn)
    operation = strategy.memory.counterattack.operation
    assert operation is not None
    assert turn.plan.unit_actions[UUID(int=40)].type == "DEPOSIT"
    assert not turn.plan.core_action or turn.plan.core_action.type != "START_MOVE"
    for key in operation.slots:
        decisions = [d for d in report.decisions if d.actor_id == key]
        assert len(decisions) == 1 and "counterattack" in decisions[0].reason
    assert not set(strategy.memory.layered_defense_posts.values()) & {
        slot.stage for slot in operation.slots.values()
    }


def test_committed_volley_can_trade_two_losses_for_one_kill():
    friends = [
        unit(n, "RANGER", position=p, hp=1)
        for n, p in enumerate([(2, 0)] * 2 + [(3, 0)] * 2, 2)
    ]
    friends += [
        unit(n, "RANGER", position=p)
        for n, p in enumerate([(7, 1), (7, 2), (8, 2), (9, 2)], 10)
    ]
    foes = [
        unit(50, "RANGER", controlled=False, position=(0, 0)),
        unit(51, "RANGER", controlled=False, position=(3, 3)),
    ]
    encounter = Encounter(friends, foes, resources=0)
    encounter.core = core(position=(8, 0))
    strategy = _strategy()
    turn = encounter.turn()
    context = _context(strategy, turn)
    # Actual wounded arrivals still outnumber a target plus one reinforcing
    # enemy. Do not apply the ordinary defense's kills >= losses veto.
    strategy.memory.counterattack.operation = counter.Counterattack(
        (object_id(50),),
        (0, 0),
        99,
        {
            o["id"]: counter.AssaultSlot((o["position"][0], 1), tuple(o["position"]))
            for o in friends[:4]
        },
        entered=99,
    )
    strategy._prepare_counterattack(context)
    assert len(turn.plan.unit_actions) == 4
    assert all(action.type == "SHOOT" for action in turn.plan.unit_actions.values())
    encounter.settle(
        turn.plan,
        {
            object_id(50): {
                "type": "SHOOT",
                "target_id": object_id(2),
                "expected_cell": [2, 0],
            },
            object_id(51): {
                "type": "SHOOT",
                "target_id": object_id(4),
                "expected_cell": [3, 0],
            },
        },
    )
    assert encounter.kills == 1 and encounter.losses == 2


@pytest.mark.parametrize(
    "case", ["core_pressure", "fire_deadline", "cooldown", "guard_losses"]
)
def test_bounds_and_core_priority_apply_to_persistent_operations(case, tmp_path):
    strategy, _, _ = _start()
    turn, context = _at_stage(strategy)
    strategy._prepare_counterattack(context)
    operation = strategy.memory.counterattack.operation
    assert operation is not None and operation.entered == 101
    friends = _friends()
    for actor in friends:
        if actor["id"] in operation.slots:
            actor["position"] = list(operation.slots[actor["id"]].firing)
    tick, c = 102, core()
    if case == "core_pressure":
        c["shield"] = 4
    elif case == "fire_deadline":
        tick = operation.entered + counter.FIRE_TICKS + 1
    elif case == "cooldown":
        strategy.memory.counterattack.finish(101, "completed")
        friends = _friends()
    else:
        operation.entered = None
        for actor in friends:
            if actor["id"] in operation.slots:
                actor["position"] = list(operation.slots[actor["id"]].stage)
            else:
                actor["hp"] = 1
    path = tmp_path / "restart.json"
    strategy.memory.save(path)
    strategy.memory = WorldMemory.load(path)
    turn = make_turn(tick=tick, objects=[c, *friends, *_foes()])
    context = _context(strategy, turn)
    strategy._prepare_counterattack(context)
    assert strategy.memory.counterattack.operation is None
    assert not turn.plan.unit_actions


def test_other_units_cannot_consume_reserved_staging_or_ingress_capacity():
    strategy, _, _ = _start()
    turn, context = _at_stage(strategy)
    strategy._prepare_counterattack(context)
    operation = strategy.memory.counterattack.operation
    assert operation is not None
    outsider = next(r for r in turn.rangers if str(r.id) not in operation.slots)
    reserved = add(outsider.position, Direction.UP)
    context.counterattack_reserved |= {reserved}
    assert not strategy._queue_move(
        outsider, Direction.UP, context, reason="ordinary post", target=reserved
    )


def test_foreign_respawn_and_duplicate_tick_do_not_reset_residence():
    strategy, turn, _ = _start()
    state = strategy.memory.counterattack
    before = deepcopy(state.contacts)
    state.observe(turn)
    assert state.contacts == before
    turn = make_turn(
        tick=101,
        objects=[core(), *_foes()],
        events=[
            {
                "event_id": object_id(900),
                "tick": 100,
                "event_type": "CORE_RESPAWNED",
                "target_id": object_id(99),
            }
        ],
    )
    state.observe(turn)
    assert state.contacts[object_id(50)].since == before[object_id(50)].since
