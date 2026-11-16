"""Forward clearance must produce actual hits without renewing patrol pursuit."""

import pytest
from arena_hero import Direction, UnitView

from arena_hero_bot.counterattack import AssaultSlot
from arena_hero_bot.geometry import add, manhattan
from arena_hero_bot.memory import PatrolContact, WorldMemory
from arena_hero_bot.skirmish import COOLDOWN_TICKS, MAX_TICKS, entry_viable
from arena_hero_bot.strategy import (
    CAMPAIGN_PHASE_STAGE4,
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, object_id, unit
from .test_campaign import _gathered_roster
from .test_layered_defense import Encounter
from .test_skirmish import _context as _skirmish_context

HOME = (-17, 303)
SCOUT = (-17, 271)
TARGET = (-31, 273)
ROCKS = (
    (-30, 271),
    (-30, 272),
    (-30, 274),
    (-30, 276),
    (-29, 273),
    (-33, 275),
    (-27, 275),
    (-26, 271),
    (-29, 269),
    (-28, 268),
    (-28, 272),
)


def _scout_case():
    workers, rangers, roles = _gathered_roster(60, center=HOME)
    positions = (
        (-32, 275),
        (-29, 272),
        (-29, 271),
        (-29, 270),
        (-28, 271),
        (-27, 271),
        (-28, 270),
        (-27, 270),
    )
    for ranger, position in zip(rangers[30:38], positions, strict=True):
        ranger["position"] = list(position)
    enemy = unit(9000, "VANGUARD", controlled=False, position=TARGET)
    c = core(position=HOME)
    memory = WorldMemory(
        core_home_position=HOME,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=16,
        campaign_last_core_id=c["id"],
        campaign_scout_target=SCOUT,
        campaign_scout_started_tick=90,
        unit_roles=roles.copy(),
        unit_roles_initialized=True,
    )
    for tick in (90, 98, 99):
        memory.observe(
            make_turn(
                tick=tick, objects=[c, *workers, *rangers, enemy], obstacles=ROCKS
            )
        )
    for ranger, origin in zip(rangers[30:38], positions, strict=True):
        memory.patrol_contacts[roles[str(ranger["id"])]] = PatrolContact(
            origin=origin,
            started_tick=20,
            last_contact_tick=99,
            target_id=enemy["id"],
        )
    config = StrategyConfig(campaign_mode=True, max_population=None)
    strategy = AggressiveStrategy(memory, config)
    encounter = Encounter([*workers, *rangers], [enemy], resources=0, rocks=ROCKS)
    encounter.core = c
    return strategy, encounter


@pytest.mark.parametrize("restart", [False, True])
def test_forward_scout_clears_stationary_full_vanguard_after_pursuit_expired(
    tmp_path, restart
):
    strategy, encounter = _scout_case()
    identities = strategy.memory.unit_roles.copy()
    scout_started = strategy.memory.campaign_scout_started_tick
    task_started = None
    member_ids = None
    entered = False
    for offset in range(12):
        turn = encounter.turn()
        strategy.decide(turn)
        task = strategy.memory.skirmish
        assert task is not None, "forward contact must get a bounded clearance task"
        assert task.scout_target == SCOUT
        if member_ids is None:
            member_ids = set(task.slots)
            task_started = task.started
            assert len(member_ids) == 2
            assert all(identities[uid].startswith("patrol-") for uid in member_ids)
        assert set(task.slots) == member_ids
        assert task.started == task_started
        assert strategy.memory.campaign_scout_started_tick == scout_started
        assert all(manhattan(slot.firing, TARGET) >= 3 for slot in task.slots.values())
        assert all(
            c.started_tick == 20 for c in strategy.memory.patrol_contacts.values()
        )
        if task.entered == turn.tick:
            entered = True
            assert all(
                turn.plan.unit_actions[r.id].type == "MOVE"
                for r in turn.rangers
                if str(r.id) in member_ids
            )
        positions = {
            uid: tuple(friend["position"]) for uid, friend in encounter.friends.items()
        }
        for uid, action in turn.plan.unit_actions.items():
            if action.type == "MOVE":
                positions[str(uid)] = add(positions[str(uid)], action.direction)
        direction = max(
            Direction,
            key=lambda value: sum(
                position == add(TARGET, value) for position in positions.values()
            ),
        )
        encounter.settle(
            turn.plan,
            {object_id(9000): {"type": "SWEEP", "direction": direction.value}},
        )
        for uid in member_ids:
            position = encounter.friends[uid]["position"]
            assert tuple(v // 32 for v in position) == (-1, 8)
            assert manhattan(tuple(position), task.origins[uid]) <= 6
        if restart and offset == 1:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
        if not encounter.foes:
            break
    assert entered
    assert encounter.kills == 1
    assert encounter.losses == 0
    assert encounter.core["position"] == list(HOME)
    assert encounter.core["hp"] == encounter.core["shield"] == 5
    assert {
        uid: role
        for uid, role in strategy.memory.unit_roles.items()
        if role != "staged"
    } == {uid: role for uid, role in identities.items() if role != "staged"}


@pytest.mark.parametrize("cause", ["scout_changed", "target_moved", "timeout"])
def test_scout_clearance_cancels_after_restart_without_renewing_budget(
    tmp_path, monkeypatch, cause
):
    strategy, encounter = _scout_case()
    turn = encounter.turn()
    strategy.decide(turn)
    task = strategy.memory.skirmish
    assert task is not None
    started = task.started
    roles = {strategy.memory.unit_roles[uid] for uid in task.slots}
    encounter.settle(turn.plan, {})
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    assert strategy.memory.skirmish is not None
    assert strategy.memory.skirmish.started == started
    assert strategy.memory.skirmish.scout_target == SCOUT
    if cause == "scout_changed":
        strategy.memory.campaign_scout_target = None
    elif cause == "target_moved":
        encounter.foes[object_id(9000)]["position"] = [-31, 272]
    else:
        monkeypatch.setattr(strategy, "_skirmish_enter", lambda *_args: False)
        while encounter.tick < started + MAX_TICKS:
            turn = encounter.turn()
            strategy._prepare_skirmish(_skirmish_context(strategy, turn))
            encounter.settle(turn.plan, {})

    strategy.decide(encounter.turn())

    assert strategy.memory.skirmish is None
    assert strategy.memory.skirmish_cooldown_until == encounter.tick + COOLDOWN_TICKS
    assert all(
        strategy.memory.patrol_contacts[role].started_tick == 20 for role in roles
    )


@pytest.mark.parametrize(
    "cause", ["home_pressure", "no_healthy_guards", "small_patrol"]
)
def test_scout_clearance_preserves_home_defense_and_patrol_reserves(cause):
    strategy, encounter = _scout_case()
    if cause == "home_pressure":
        enemy = unit(9001, "RANGER", controlled=False, position=(-17, 300))
        encounter.foes[enemy["id"]] = enemy
    elif cause == "no_healthy_guards":
        for friend in encounter.friends.values():
            if friend["unit_type"] == "RANGER" and friend["position"][1] >= 288:
                friend["hp"] = 1
    else:
        moved = 0
        for friend in encounter.friends.values():
            if (
                friend["unit_type"] == "RANGER"
                and friend["position"][1] < 288
                and moved < 5
            ):
                friend["position"] = [-6 + moved, 310]
                moved += 1

    strategy.decide(encounter.turn())

    assert strategy.memory.skirmish is None


@pytest.mark.parametrize("distance, permitted", [(2, False), (3, True)])
def test_two_volley_vanguard_entry_requires_no_current_or_one_step_sweep(
    distance, permitted
):
    turn = make_turn(
        objects=[
            core(position=(-10, -10)),
            unit(2, "RANGER", position=(-1, -distance)),
            unit(3, "RANGER", position=(-1, -distance)),
            unit(9000, "VANGUARD", controlled=False, position=(0, 0)),
        ]
    )
    targets = tuple(
        enemy for enemy in turn.visible_enemies if isinstance(enemy, UnitView)
    )
    slots = {
        str(member.id): AssaultSlot(member.position, (0, -distance))
        for member in turn.rangers
    }
    assert not entry_viable(turn.rangers, slots, targets, targets, set())
    assert (
        entry_viable(
            turn.rangers,
            slots,
            targets,
            targets,
            set(),
            allow_vanguard_two_volley=True,
        )
        is permitted
    )
