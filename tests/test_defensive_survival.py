"""Survival decisions with separate firefights and partial enemy visibility."""

from uuid import UUID

import pytest
from arena_hero import Direction, StartMoveAction, UnitView

from arena_hero_bot.geometry import add
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig, _TurnContext

from .factories import core, make_turn, object_id, unit
from .test_layered_defense import Encounter


def _strategy(memory=None, *, campaign=False):
    return AggressiveStrategy(
        memory or WorldMemory(core_home_position=(0, 0)),
        StrategyConfig(target_workers=0, max_population=None, campaign_mode=campaign),
    )


def _context(strategy, turn):
    strategy.memory.observe(turn)
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={u.position for u in turn.units},
        enemy_positions={e.position for e in turn.visible_enemies},
        threat=strategy._combat_policy.assess(turn, strategy.memory.obstacles),
    )


@pytest.mark.parametrize("safe_exit", [False, True])
def test_separate_lone_shooters_do_not_commit_each_other_to_equal_trades(safe_exit):
    friends = [unit(2, "RANGER", position=(-5, 0)), unit(3, "RANGER", position=(5, 0))]
    foes = [
        unit(30, "RANGER", controlled=False, position=(-5, -2), hp=1),
        unit(31, "RANGER", controlled=False, position=(-5, -3)),
        unit(32, "RANGER", controlled=False, position=(5, -2), hp=1),
        unit(33, "RANGER", controlled=False, position=(5, -3)),
    ]
    encounter = Encounter(friends, foes, resources=0)
    encounter.core = core(position=(0, 6))
    turn = make_turn(
        objects=[encounter.core, *friends, *foes],
        obstacles=(
            []
            if safe_exit
            else [
                position
                for x in (-5, 5)
                for position in ((x - 1, 0), (x + 1, 0), (x, 1))
            ]
        ),
    )

    _strategy().decide(turn)

    assert all(
        turn.plan.unit_actions[UUID(int=n)].type == ("MOVE" if safe_exit else "SHOOT")
        for n in (2, 3)
    )
    encounter.settle(
        turn.plan,
        {
            foe["id"]: {
                "type": "SHOOT",
                "target_id": friends[i // 2]["id"],
                "expected_cell": friends[i // 2]["position"],
            }
            for i, foe in enumerate(foes)
        },
    )
    assert encounter.losses == (0 if safe_exit else 2)
    assert encounter.kills == (0 if safe_exit else 2)
    assert all(friend["hp"] == 2 for friend in encounter.friends.values())


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("branch", ["layered", "mass_escape"])
def test_core_withdrawal_counts_dark_contacts_alongside_visible_ones(
    tmp_path, restart, branch
):
    memory = WorldMemory(core_home_position=(0, 0))
    memory.observe(
        make_turn(
            tick=98,
            objects=[
                core(position=(0, -1)),
                unit(30, "RANGER", controlled=False, position=(-2, -1)),
                unit(31, "RANGER", controlled=False, position=(-2, -1)),
            ],
        )
    )
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
    strategy = _strategy(memory, campaign=True)
    turn = make_turn(
        objects=[
            core(shield=2),
            unit(32, "RANGER", controlled=False, position=(3, 0)),
            unit(33, "RANGER", controlled=False, position=(3, 0)),
        ]
    )
    if branch == "layered":
        strategy.decide(turn)
        assert not isinstance(turn.plan.core_action, StartMoveAction)
    else:
        context = _context(strategy, turn)
        assert strategy._core_escape_direction(context, turn.visible_enemies) is None


@pytest.mark.parametrize("reason", ["expired", "destroyed", "noncombat"])
def test_inactive_contacts_do_not_prevent_a_survivable_core_withdrawal(reason):
    memory = WorldMemory(core_home_position=(0, 0))
    memory.observe(
        make_turn(
            tick=91 if reason == "expired" else 98,
            objects=[
                core(),
                *[
                    unit(
                        n,
                        "WORKER" if reason == "noncombat" else "RANGER",
                        controlled=False,
                        position=(-2, -1),
                    )
                    for n in (30, 31)
                ],
            ],
        )
    )
    turn = make_turn(
        objects=[
            core(shield=2),
            unit(32, "RANGER", controlled=False, position=(3, 0)),
            unit(33, "RANGER", controlled=False, position=(3, 0)),
        ],
        events=(
            [
                {
                    "event_id": object_id(900 + n),
                    "tick": 99,
                    "event_type": "DESTRUCTION_PARTICIPATION",
                    "target_id": object_id(n),
                }
                for n in (30, 31)
            ]
            if reason == "destroyed"
            else []
        ),
    )

    _strategy(memory, campaign=True).decide(turn)

    assert isinstance(turn.plan.core_action, StartMoveAction)


@pytest.mark.parametrize("blocker", [None, "capacity", "shot", "danger"])
def test_retreat_uses_only_a_successful_safe_formation_departure(blocker):
    friends = [
        unit(2, "RANGER", position=(0, -4)),
        unit(3, "RANGER", position=(0, -3)),
        unit(4, "RANGER", position=(0, -3)),
    ]
    if blocker == "capacity":
        friends.extend(unit(n, "RANGER", position=(0, -2)) for n in (5, 6))
    foes = [unit(n, "RANGER", controlled=False, position=(3, -4)) for n in (30, 31)]
    if blocker in {"shot", "danger"}:
        foes.append(
            unit(
                32,
                "RANGER",
                controlled=False,
                position=(3, -3 if blocker == "shot" else -2),
            )
        )
    turn = make_turn(
        objects=[core(), *friends, *foes],
        obstacles=[(-1, -4), (0, -5)],
    )
    strategy = _strategy(
        WorldMemory(unit_roles={friend["id"]: "defense" for friend in friends})
    )
    context = _context(strategy, turn)
    context.layered_steps[UUID(int=3)] = Direction.DOWN
    context.layered_assignments[UUID(int=3)] = (0, -2)

    strategy._decide_ranger(turn.rangers[0], context, ())

    if blocker:
        assert turn.plan.unit_actions[UUID(int=2)].type == "SHOOT"
        assert UUID(int=3) not in turn.plan.unit_actions
        assert not context.departure_counts
        return
    assert turn.plan.unit_actions[UUID(int=2)].type == "MOVE"
    assert turn.plan.unit_actions[UUID(int=3)].type == "MOVE"
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
    assert encounter.friends[object_id(2)]["position"] == [0, -3]
    assert encounter.friends[object_id(3)]["position"] == [0, -2]
    assert strategy._predicted_friendly_occupancy((0, -3), context) == 2


def test_visible_contact_replaces_its_remembered_position_for_retreat():
    memory = WorldMemory(core_home_position=(0, 0))
    memory.observe(
        make_turn(
            tick=99,
            objects=[core(), unit(30, "RANGER", controlled=False, position=(-2, -1))],
        )
    )
    strategy = _strategy(memory, campaign=True)
    turn = make_turn(
        objects=[
            core(hp=5, shield=0),
            unit(30, "RANGER", controlled=False, position=(3, 0)),
        ]
    )
    context = _context(strategy, turn)
    enemies = tuple(e for e in turn.visible_enemies if isinstance(e, UnitView))

    direction = strategy._layered_core_retreat(context, enemies, set())

    assert direction is not None
    assert add((0, 0), direction) != (3, 0)
