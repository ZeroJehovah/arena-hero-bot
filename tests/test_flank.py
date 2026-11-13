"""Fog geometry, actual observation, precise arrows and bounded state."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from time import monotonic
from uuid import UUID

import pytest
from arena_hero import UnitView

from arena_hero_bot import flank
from arena_hero_bot.exploration import can_see, visible_cells
from arena_hero_bot.geometry import adjacent_positions, firing_positions, line_of_fire
from arena_hero_bot.memory import WorldMemory

from .factories import core, make_turn, object_id, unit
from .flank_scenarios import seed_contact, standoff


def _fighters(turn):
    return tuple(enemy for enemy in turn.visible_enemies if isinstance(enemy, UnitView))


@pytest.mark.parametrize("offset", [(0, 0), (31, -32), (-100, 100)])
def test_blind_shooting_geometry_in_every_direction(offset):
    positions = firing_positions(offset)
    assert len(set(positions)) == 24
    hidden = [cell for cell in positions if not can_see(offset, cell, 5, set())]
    assert set(hidden) == {
        (offset[0] + dx, offset[1] + dy) for dx in (-3, 3) for dy in (-3, 3)
    }
    for cell in positions:
        assert line_of_fire(cell, offset, set())
        assert line_of_fire(offset, cell, set())
    assert not line_of_fire(offset, (offset[0] + 2, offset[1] + 1), set())
    for cell in hidden:
        assert all(
            not line_of_fire(cell, neighbor, set())
            for neighbor in adjacent_positions(offset)
        )


@pytest.mark.parametrize("rock", [(1, 0), (0, 1), (1, 1)])
def test_corner_vision_and_shooting_have_different_rock_rules(rock):
    rocks = {rock}
    assert not can_see((0, 0), (2, 2), 5, rocks)
    assert ((2, 2) in visible_cells((0, 0), 5, rocks)) is False
    assert line_of_fire((0, 0), (2, 2), rocks) is (rock != (1, 1))


@pytest.mark.parametrize(
    "kind,radius", [("WORKER", 3), ("VANGUARD", 4), ("RANGER", 5), ("CORE", 5)]
)
def test_each_known_enemy_observer_disqualifies_a_blind_cell(kind, radius):
    enemy = unit(50, "RANGER", controlled=False)
    watcher = (
        core(51, controlled=False, position=(3, 2))
        if kind == "CORE"
        else unit(51, kind, controlled=False, position=(3, 2))
    )
    turn = make_turn(objects=[core(position=(0, 10)), enemy, watcher])
    target, observer = turn.visible_enemies
    assert isinstance(target, UnitView)
    assert flank.observer_radius(observer) == radius
    assert (
        flank.cell_rejection((3, 3), (target,), (target,), (target,), set(), set())
        is None
    )
    assert (
        flank.cell_rejection(
            (3, 3), (target,), (target, observer), (target,), set(), set()
        )
        == "known_observer"
    )


def test_front_neighbor_prefire_is_geometry_not_an_extra_arrival_damage_charge():
    turn = make_turn(
        objects=[
            core(position=(0, 10)),
            unit(2, "RANGER", position=(2, 3)),
            unit(50, "RANGER", controlled=False),
        ]
    )
    target = turn.visible_enemies[0]
    assert isinstance(target, UnitView)
    assert (3, 3) in flank.prefire_cells((target,), turn.rangers, set())
    safe = make_turn(
        objects=[
            core(position=(0, 10)),
            unit(2, "RANGER", position=(0, 4)),
            unit(50, "RANGER", controlled=False),
        ]
    )
    assert (3, 3) not in flank.prefire_cells((target,), safe.rangers, set())
    assert (3, 3) in flank.prefire_cells(
        (target,), safe.rangers, set(), {UUID(int=2): (2, 3)}
    )


def test_actual_continuous_eight_tick_contact_does_not_own_any_fighters():
    strategy, encounter = standoff()
    for tick in range(93, 101):
        encounter.tick = tick
        turn = encounter.turn()
        strategy.memory.observe(turn)
        strategy._observe_flank_contacts(turn)
        contact = strategy.memory.flank.contacts[
            flank.group_key(flank.Kind.SKIRMISH, turn.visible_enemies)
        ]
        assert contact.ticks == tick - 92
        assert not strategy.memory.flank.operation
        assert not turn.plan.unit_actions
    strategy._observe_flank_contacts(turn)
    assert contact.ticks == 8


@pytest.mark.parametrize(
    "change", ["position", "uuid", "gap", "lost", "front", "group", "core"]
)
def test_new_contact_resets_on_each_real_identity_or_observation_change(change):
    strategy, encounter = standoff()
    seed_contact(strategy, encounter)
    if change == "position":
        encounter.foes[object_id(50)]["position"][0] += 1
    elif change == "uuid":
        encounter.foes[object_id(50)]["id"] = object_id(52)
    elif change == "gap":
        encounter.tick += 1
    elif change == "lost":
        encounter.foes[object_id(50)]["position"] = [20, -7]
    elif change == "front":
        encounter.friends[object_id(20)]["position"] = [20, 20]
    elif change == "group":
        encounter.foes.pop(object_id(51))
    else:
        encounter.core["id"] = object_id(99)
    turn = encounter.turn()
    strategy.memory.observe(turn)
    strategy._observe_flank_contacts(turn)
    targets = tuple(turn.visible_enemies)
    if targets:
        contact = strategy.memory.flank.contacts.get(
            flank.group_key(flank.Kind.SKIRMISH, targets)
        )
        assert contact is None or contact.ticks == 1
    assert strategy.memory.flank.operation is None


def test_actual_front_rotation_continues_contact_but_worker_sight_does_not():
    strategy, encounter = standoff()
    seed_contact(strategy, encounter)
    encounter.friends[object_id(20)]["position"] = [10, 10]
    encounter.friends[object_id(21)]["position"] = [0, -3]
    turn = encounter.turn()
    strategy.memory.observe(turn)
    strategy._observe_flank_contacts(turn)
    contact = strategy.memory.flank.contacts[
        flank.group_key(flank.Kind.SKIRMISH, turn.visible_enemies)
    ]
    assert contact.ticks == 8
    encounter.tick += 1
    encounter.friends[object_id(21)]["position"] = [10, 11]
    encounter.friends[object_id(80)] = unit(80, "WORKER", position=(0, -4))
    turn = encounter.turn()
    assert len(turn.visible_enemies) == 2
    strategy.memory.observe(turn)
    strategy._observe_flank_contacts(turn)
    assert (
        flank.group_key(flank.Kind.SKIRMISH, turn.visible_enemies)
        not in strategy.memory.flank.contacts
    )


@pytest.mark.parametrize(
    "count,expected,rounds,losses",
    [
        (2, True, 2, 0),
        (2, False, 3, 2),
        (3, False, 2, 1),
        (4, False, 1, 1),
        (4, True, 1, 0),
    ],
)
def test_exchange_uses_zero_entry_damage_and_simultaneous_uuid_budgets(
    count, expected, rounds, losses
):
    turn = make_turn(
        objects=[
            core(),
            *[unit(50 + index, "RANGER", controlled=False) for index in range(2)],
        ]
    )
    cells = [(3, 3), (3, 3), (-3, 3), (-3, 3)]
    positions = {UUID(int=2 + index): cell for index, cell in enumerate(cells[:count])}
    result = flank.exchange(
        positions,
        dict.fromkeys(positions, 2),
        _fighters(turn),
        _fighters(turn),
        set(),
        3,
        first_on_front=expected,
    )
    assert result.viable
    assert (result.kills, result.rounds, result.losses) == (2, rounds, losses)
    if count == 2:
        allocation = flank.allocate_fire(
            positions, dict.fromkeys(positions, 2), _fighters(turn), set()
        )
        assert len({target.id for target in allocation.values()}) == 1
    if count == 4:
        allocation = flank.allocate_fire(
            positions, dict.fromkeys(positions, 2), _fighters(turn), set()
        )
        assert sorted(
            Counter(target.id for target in allocation.values()).values()
        ) == [2, 2]


def test_opening_kill_cannot_be_made_from_missing_or_wrong_ray_shooters():
    turn = make_turn(
        objects=[
            core(),
            unit(50, "RANGER", controlled=False),
            unit(51, "RANGER", controlled=False, position=(10, 10)),
        ]
    )
    positions = {UUID(int=2): (3, 3), UUID(int=3): (13, 13)}
    volley = flank.allocate_fire(
        positions, dict.fromkeys(positions, 2), _fighters(turn), set()
    )
    assert all(
        count < 2 for count in Counter(target.id for target in volley.values()).values()
    )
    assert flank.exchange(
        positions,
        dict.fromkeys(positions, 2),
        _fighters(turn),
        _fighters(turn),
        set(),
        3,
    ).viable


def test_fire_allocation_accounts_for_committed_front_arrows_and_actual_hp():
    turn = make_turn(
        objects=[
            core(),
            unit(50, "RANGER", controlled=False),
            unit(51, "RANGER", controlled=False, hp=1),
        ]
    )
    positions = {UUID(int=key): (3, 3) for key in (2, 3)}
    targets = _fighters(turn)
    allocation = flank.allocate_fire(
        positions,
        {UUID(int=2): 1, UUID(int=3): 2},
        targets,
        set(),
        {str(targets[0].id): 2},
    )
    assert set(allocation) == {UUID(int=3)}
    assert allocation[UUID(int=3)].id == targets[1].id


@pytest.mark.parametrize("kind", ["VANGUARD", "WORKER"])
def test_non_ranger_units_do_not_supply_diagonal_retaliation_arrows(kind):
    turn = make_turn(
        objects=[
            core(position=(0, 10)),
            unit(50, "RANGER", controlled=False),
            unit(51, kind, controlled=False, position=(6, 6)),
        ]
    )
    target, other = _fighters(turn)
    positions = {UUID(int=2): (3, 3), UUID(int=3): (3, 3)}
    result = flank.exchange(
        positions, dict.fromkeys(positions, 2), (target,), (target, other), set(), 3
    )
    assert result.viable and result.kills == 1 and result.rounds == 1
    assert result.losses == 0


def test_a_real_vanguard_sweep_is_returned_to_the_mothers_melee_policy():
    turn = make_turn(
        objects=[
            core(position=(0, 10)),
            unit(50, "RANGER", controlled=False, position=(4, 3)),
            unit(51, "VANGUARD", controlled=False, position=(3, 4)),
        ]
    )
    target, vanguard = _fighters(turn)
    positions = {UUID(int=2): (3, 3), UUID(int=3): (4, 4)}
    result = flank.exchange(
        positions, dict.fromkeys(positions, 2), (target,), (target, vanguard), set(), 3
    )
    assert not result.viable and result.reason == "vanguard_risk"


@pytest.mark.parametrize("phase", list(flank.Phase))
def test_each_active_phase_restores_fixed_identity_and_original_budget(tmp_path, phase):
    strategy, encounter = standoff()
    seed_contact(strategy, encounter)
    strategy.decide(encounter.turn())
    operation = strategy.memory.flank.operation
    assert operation is not None
    operation.phase = phase
    if phase is not flank.Phase.ASSEMBLE:
        operation.entry_tick = 100
    if phase is flank.Phase.FINISH:
        operation.first_shot = 101
    before = deepcopy(operation.to_dict())
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    loaded = WorldMemory.load(path)
    assert loaded.flank.operation is not None and loaded.skirmish is not None
    assert loaded.flank.operation.to_dict() == before
    assert loaded.skirmish.started == operation.parent.started
    assert not hasattr(loaded.flank, "entry_permit")


def test_old_or_invalid_saved_flank_state_never_creates_a_new_task(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text('{"schema_version": 1, "last_tick": 100}')
    assert WorldMemory.load(path).flank.operation is None
    assert flank.State.from_dict({"operation": {"phase": "invalid"}}).operation is None


@pytest.mark.parametrize(
    "fault", ["roster", "budget", "route", "target", "entry", "shot"]
)
def test_saved_mission_cannot_restore_invalid_rosters_routes_or_phase_budgets(fault):
    strategy, encounter = standoff()
    seed_contact(strategy, encounter)
    strategy.decide(encounter.turn())
    operation = strategy.memory.flank.operation
    assert operation is not None
    raw = deepcopy(operation.to_dict())
    if fault == "roster":
        raw["parent"]["party_sizes"] = [3]
    elif fault == "budget":
        raw["parent"]["deadline"] += 1
    elif fault == "route":
        raw["routes"][raw["member_ids"][0]] = [(100, 100)]
    elif fault == "target":
        raw["target_ids"] *= 2
    elif fault == "entry":
        raw["entry_tick"] = None
    else:
        raw["phase"] = "FINISH"
        raw["first_shot"] = raw["parent"]["deadline"]
    assert flank.State.from_dict({"operation": raw}).operation is None


def test_contact_churn_and_repeated_loads_keep_a_bounded_recoverable_state():
    import json

    state = flank.State(core_id=object_id(1))
    for tick in range(100, 350):
        turn = make_turn(
            tick=tick,
            objects=[
                core(position=(0, 10)),
                unit(2, "RANGER", position=(0, 4)),
                unit(1000 + tick, "RANGER", controlled=False),
            ],
        )
        state.observe(
            turn, [(flank.Kind.SKIRMISH, _fighters(turn))], turn.rangers, set()
        )
        state = flank.State.from_dict(json.loads(json.dumps(state.to_dict())))
        assert len(state.contacts) <= flank.CONTACT_LIMIT
        assert state.operation is None
    state.observe(make_turn(tick=400, objects=[core()]), [], [], set())
    assert not state.contacts
    assert flank.State.from_dict({"contacts": []}).operation is None


def test_large_army_and_map_keep_search_bounded_and_preserve_original_actions():
    strategy, encounter = standoff(side_count=4)
    for index in range(90):
        actor = unit(1000 + index, "RANGER", position=(5 + index % 10, 5 + index // 10))
        encounter.friends[actor["id"]] = actor
    strategy.memory.obstacles.update((1000 + index, 1000) for index in range(20000))
    seed_contact(strategy, encounter)
    turn = encounter.turn()
    from arena_hero_bot.models import DecisionReport
    from arena_hero_bot.strategy import _TurnContext

    context = _TurnContext(
        turn=turn,
        report=DecisionReport(turn.tick),
        occupied={unit.position for unit in turn.units},
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
    )
    turn.rangers[-1].wait()
    original = deepcopy(turn.plan.model_dump())
    parent = flank.Parent(flank.Kind.SKIRMISH, "stress", 100, 112, 109, 3, 6, (2, 4, 6))
    strategy._choose_flank(context, parent, turn.rangers, _fighters(turn))
    budget = strategy._flank_search
    assert budget is not None
    assert budget.groups <= flank.GROUP_LIMIT
    assert budget.extensions <= flank.EXTENSION_LIMIT
    assert budget.paths <= flank.PATH_LIMIT
    assert turn.plan.model_dump() == original


def test_contact_and_prefire_caches_are_bounded_and_aged():
    state = flank.State(core_id=object_id(1))
    state.contacts = {
        str(index): flank.Contact(flank.Kind.SKIRMISH, 90, 99, ())
        for index in range(200)
    }
    state.prefire_cells = {(index, 0): 99 for index in range(200)}
    state.observe(make_turn(objects=[core()]), [], [], set())
    assert len(state.contacts) == len(state.prefire_cells) == 128
    state.observe(make_turn(tick=133, objects=[core()]), [], [], set())
    assert not state.contacts and not state.prefire_cells


@pytest.mark.parametrize("invalid", ["tick", "member", "cell", "task", "phase"])
def test_arrival_permit_is_exact_and_cannot_survive_a_tick(invalid):
    strategy, encounter = standoff()
    seed_contact(strategy, encounter)
    strategy.decide(encounter.turn())
    operation = strategy.memory.flank.operation
    operation.phase = flank.Phase.ASSEMBLE
    member = UUID(operation.member_ids[0])
    cell = operation.slots[str(member)].firing
    permit = flank.EntryPermit(
        100,
        operation.parent.task_id,
        {UUID(key): slot.firing for key, slot in operation.slots.items()},
    )
    assert permit.allows(100, operation, member, cell)
    if invalid == "tick":
        assert not permit.allows(101, operation, member, cell)
    elif invalid == "member":
        assert not permit.allows(100, operation, UUID(int=999), cell)
    elif invalid == "cell":
        assert not permit.allows(100, operation, member, (99, 99))
    elif invalid == "task":
        assert not replace(permit, task_id="ordinary-move").allows(
            100, operation, member, cell
        )
    else:
        operation.phase = flank.Phase.FINISH
        assert not permit.allows(100, operation, member, cell)


def test_search_budget_is_bounded_without_mutating_the_original_plan():
    strategy, encounter = standoff()
    seed_contact(strategy, encounter)
    turn = encounter.turn()
    strategy.memory.observe(turn)
    strategy._observe_flank_contacts(turn)
    budget = flank.SearchBudget(monotonic() - 1)
    parent = flank.Parent(flank.Kind.SKIRMISH, "test", 100, 112, 109, 3, 6, (2, 4, 6))
    before = turn.plan.model_dump()
    assert (
        flank.choose(
            turn,
            strategy.memory.flank,
            parent,
            turn.rangers,
            turn.rangers,
            turn.visible_enemies,
            turn.visible_enemies,
            turn.visible_enemies,
            set(),
            set(),
            budget,
            allowed=lambda _ranger, _cell: True,
            known=lambda _cell: True,
        )
        is None
    )
    assert turn.plan.model_dump() == before
    assert budget.reasons["search_budget_exhausted"] == 1


def test_hidden_destination_does_not_certify_an_unknown_assembly_route():
    strategy, encounter = standoff()
    encounter.friends[object_id(3)]["position"] = [5, -4]
    seed_contact(strategy, encounter)
    turn = encounter.turn()
    strategy.memory.observe(turn)
    strategy._observe_flank_contacts(turn)
    candidates = tuple(
        ranger for ranger in turn.rangers if ranger.id in {UUID(int=2), UUID(int=3)}
    )
    parent = flank.Parent(
        flank.Kind.SKIRMISH, "unknown-route", 100, 112, 109, 3, 6, (2,)
    )
    before = turn.plan.model_dump()
    allowed = {(3, -3), (5, -4), (4, -4), (3, -4)}
    proposal = flank.choose(
        turn,
        strategy.memory.flank,
        parent,
        candidates,
        turn.rangers,
        _fighters(turn),
        _fighters(turn),
        turn.visible_enemies,
        set(),
        set(),
        flank.SearchBudget(monotonic() + flank.SEARCH_SECONDS),
        allowed=lambda _ranger, cell: cell in allowed,
        known=lambda cell: cell != (4, -4),
    )
    assert proposal is None
    assert turn.plan.model_dump() == before


def test_enemy_fog_view_does_not_expose_hidden_flank_coordinates():
    _strategy, encounter = standoff()
    view = encounter.view(enemy=True)
    assert {str(enemy.id) for enemy in view.visible_enemies} == {object_id(20)}
    assert object_id(2) not in view.state.model_dump_json()
    response = encounter.response(view, mode="visible_front")
    assert all(action["expected_cell"] == [0, -4] for action in response.values())
