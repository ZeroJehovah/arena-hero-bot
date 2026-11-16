"""Safe staging cover must preserve the complete clearance entrance."""

from uuid import UUID

import pytest

from arena_hero_bot.counterattack import AssaultSlot
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.skirmish import Skirmish
from arena_hero_bot.strategy import AggressiveStrategy

from .factories import object_id, unit
from .test_skirmish import _context, _hostile_volley, _setup


def _assembling(*, ready=False):
    strategy, encounter = _setup()
    slots = {
        object_id(2): AssaultSlot((-1, -4), (0, -4)),
        object_id(3): AssaultSlot((1, -4), (0, -4)),
        object_id(4): AssaultSlot((-1, -5), (-1, -6)),
        object_id(5): AssaultSlot((1, -5), (1, -6)),
    }
    if not ready:
        encounter.friends[object_id(5)]["position"] = [2, -3]
    strategy.memory.skirmish = Skirmish(
        object_id(1),
        (0, 0),
        (object_id(50), object_id(51)),
        (0, -7),
        encounter.tick,
        encounter.tick,
        slots,
        {uid: tuple(encounter.friends[uid]["position"]) for uid in slots},
    )
    return strategy, encounter


@pytest.mark.parametrize("restart", [False, True])
def test_staging_cover_then_complete_entry_and_real_volley(tmp_path, restart):
    strategy, encounter = _assembling()
    initial = strategy.memory.skirmish
    assert initial is not None
    members = set(initial.slots)
    origins = initial.origins.copy()
    entered = False
    for offset in range(10):
        turn = encounter.turn()
        context = _context(strategy, turn)
        strategy._prepare_skirmish(context)
        operation = strategy.memory.skirmish
        assert operation is not None
        assert operation.started == initial.started
        assert operation.origins == origins
        assert set(operation.slots) <= members
        if offset == 0:
            shot = turn.plan.unit_actions[UUID(int=2)]
            assert shot.type == "SHOOT" and shot.expected_cell == (-1, -7)
            assert shot.target_id is None
            assert turn.plan.unit_actions[UUID(int=5)].type == "MOVE"
            assert operation.entered is None
        if operation.entered == turn.tick:
            entered = True
            assert set(operation.slots) == members
            assert all(
                turn.plan.unit_actions[UUID(uid)].type == "MOVE" for uid in members
            )
        encounter.settle(turn.plan, _hostile_volley(encounter, turn.plan))
        if restart and offset == 0:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
        if not encounter.foes:
            break
    assert entered and encounter.kills == 2
    assert encounter.losses <= 2


def test_staging_cover_hits_a_target_moving_into_its_neighbor():
    strategy, encounter = _assembling()
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    shot = turn.plan.unit_actions[UUID(int=2)]
    assert shot.type == "SHOOT" and shot.expected_cell == (-1, -7)
    encounter.settle(turn.plan, {object_id(50): {"type": "MOVE", "direction": "LEFT"}})
    assert encounter.losses == 0
    assert (
        object_id(50) not in encounter.foes or encounter.foes[object_id(50)]["hp"] < 2
    )
    assert strategy.memory.skirmish is not None
    assert strategy.memory.skirmish.entered is None


@pytest.mark.parametrize("obstruction", ["rock", "ally"])
def test_blocked_fixed_neighbor_waits_without_firing_at_unrelated_cells(obstruction):
    strategy, encounter = _assembling()
    if obstruction == "rock":
        encounter.rocks |= {(-1, -7)}
    else:
        guard = unit(80, "RANGER", position=(-1, -7))
        encounter.friends[guard["id"]] = guard
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    assert strategy.memory.skirmish is not None
    assert turn.plan.unit_actions[UUID(int=2)].type == "WAIT"
    assert turn.plan.unit_actions[UUID(int=5)].type == "MOVE"


def test_failed_joint_entrance_covers_from_stages_before_retry(monkeypatch):
    strategy, encounter = _assembling(ready=True)
    original = strategy._queue_move
    calls = 0

    def deny_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        return calls != 2 and original(*args, **kwargs)

    monkeypatch.setattr(strategy, "_queue_move", deny_second)
    turn = encounter.turn()
    context = _context(strategy, turn)
    strategy._prepare_skirmish(context)
    operation = strategy.memory.skirmish
    assert operation is not None and operation.entered is None
    assert all(action.type == "SHOOT" for action in turn.plan.unit_actions.values())
    assert not context.arrival_counts and not context.departure_counts
    encounter.settle(turn.plan, {})
    monkeypatch.setattr(strategy, "_queue_move", original)
    turn = encounter.turn()
    strategy._prepare_skirmish(_context(strategy, turn))
    assert strategy.memory.skirmish is operation
    assert operation.entered == turn.tick and operation.started == 100
    assert all(action.type == "MOVE" for action in turn.plan.unit_actions.values())
    encounter.settle(turn.plan, _hostile_volley(encounter, turn.plan))
