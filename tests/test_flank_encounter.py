"""Complete local missions, independent fog and atomic failure injection."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from time import monotonic
from uuid import UUID

import pytest
from arena_hero import Direction, UnitView

from arena_hero_bot import flank
from arena_hero_bot.counterattack import paired_cells
from arena_hero_bot.memory import EnemySighting, WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.runtime import _identity_snapshot, _turn_record
from arena_hero_bot.strategy import AggressiveStrategy, _TurnContext

from .factories import core, make_turn, object_id, unit
from .flank_scenarios import FogEncounter, seed_contact, standoff


def _run_comparison(enabled):
    strategy, encounter = standoff()
    strategy.config = replace(strategy.config, blind_flank=enabled)
    encounter.tick = 93
    worker = unit(80, "WORKER", position=(0, 0), cargo=1)
    encounter.friends[worker["id"]] = worker
    encounter.friendly_ids.add(worker["id"])
    for _ in range(12):
        turn = encounter.turn()
        report = strategy.decide(turn)
        assert not any(
            "planning deadline" in decision.reason for decision in report.decisions
        )
        orders = encounter.response(encounter.view(enemy=True), mode="infer_after_hit")
        encounter.settle(turn.plan, orders)
    return encounter, strategy


def test_same_initial_state_recomputes_fog_responses_and_preserves_economic_service():
    baseline, _ = _run_comparison(False)
    changed, strategy = _run_comparison(True)
    assert (
        baseline.trace[0]["plan"]["unit_actions"]
        == changed.trace[0]["plan"]["unit_actions"]
    )
    assert baseline.kills == 0 and changed.kills == 2
    assert baseline.losses == 1 and changed.losses == 0
    assert baseline.deposits == changed.deposits == 1
    assert baseline.spawns == changed.spawns == 0
    assert baseline.healing_cost == changed.healing_cost == 0
    assert baseline.resources == changed.resources == 1
    assert strategy.memory.flank.last_result is not None
    assert strategy.memory.flank.last_result["flank_damage"] == 1
    # The second policy receives the new hit events and changes its orders;
    # these enemy actions are freshly computed, never replayed from baseline.
    assert baseline.trace[9]["enemy"] != changed.trace[9]["enemy"]


def test_different_route_lengths_wait_at_safe_stages_before_a_joint_entry():
    strategy, encounter = standoff()
    encounter.friends[object_id(3)]["position"] = [4, -3]
    seed_contact(strategy, encounter)
    start = encounter.tick
    saw_assembly = False
    for _ in range(5):
        turn = encounter.turn()
        strategy.decide(turn)
        operation = strategy.memory.flank.operation
        assert operation is not None and operation.parent.started == start
        assert operation.parent.deadline == start + 12
        actions = {
            key: turn.plan.unit_actions[UUID(key)] for key in operation.member_ids
        }
        if operation.phase is flank.Phase.ASSEMBLE:
            saw_assembly = True
            assert sorted(action.type for action in actions.values()) == [
                "MOVE",
                "WAIT",
            ]
            assert all(
                encounter.friends[key]["hp"] == 2 for key in operation.member_ids
            )
        if operation.phase is flank.Phase.ENTER:
            assert all(action.type == "MOVE" for action in actions.values())
        if operation.first_shot is not None:
            assert saw_assembly
            assert all(action.type == "SHOOT" for action in actions.values())
            break
        encounter.settle(turn.plan, encounter.response(encounter.view(enemy=True)))
    else:
        pytest.fail("bounded assembly never reached its complete opening")


@pytest.mark.parametrize("service", ["spawn", "deposit", "heal"])
@pytest.mark.parametrize("stable_count", [0, 7])
def test_observation_and_qualified_entry_preserve_real_core_or_worker_services(
    service, stable_count
):
    from arena_hero import core_resource_capacity

    strategy, encounter = standoff()
    if service == "spawn":
        encounter.resources = core_resource_capacity(len(encounter.friends))
    else:
        worker = unit(
            80,
            "WORKER",
            position=(0, 0),
            cargo=int(service == "deposit"),
            hp=1 if service == "heal" else 2,
        )
        encounter.friends[worker["id"]] = worker
        encounter.friendly_ids.add(worker["id"])
        encounter.resources = int(service == "heal")
    seed_contact(strategy, encounter, count=stable_count)
    turn = encounter.turn()
    strategy.decide(turn)
    if stable_count:
        assert strategy.memory.flank.operation is not None
        assert strategy.memory.flank.operation.phase is flank.Phase.ENTER
    else:
        assert strategy.memory.flank.operation is None
    if service == "spawn":
        assert (
            turn.plan.core_action is not None and turn.plan.core_action.type == "SPAWN"
        )
    else:
        assert turn.plan.unit_actions[UUID(int=80)].type == (
            "DEPOSIT" if service == "deposit" else "HEAL"
        )
    encounter.settle(turn.plan, encounter.response(encounter.view(enemy=True)))
    strategy.decide(encounter.turn())
    operation = strategy.memory.flank.operation
    if stable_count:
        assert operation is not None
        assert operation.spawns == int(service == "spawn")
        assert operation.deposits == int(service == "deposit")
        assert operation.healing_cost == int(service == "heal")
    else:
        assert operation is None
    assert encounter.spawns == int(service == "spawn")
    if service == "heal":
        assert encounter.friends[object_id(80)]["hp"] == 2 and encounter.resources == 0


def test_incomplete_protection_roster_keeps_its_original_fixed_size_rule():
    strategy, encounter = standoff(side_count=4, home=(16, 16))
    del encounter.friends[object_id(5)]
    members = {object_id(key) for key in (2, 3, 4)}
    strategy.memory.unit_roles = {
        key: "patrol-1" if key in members else "defense" for key in encounter.friends
    }
    seed_contact(strategy, encounter)
    report = strategy.decide(encounter.turn())
    assert strategy.memory.worker_area_protection is None
    assert (
        strategy.memory.flank.operation is None
        or strategy.memory.flank.operation.parent.kind is not flank.Kind.PROTECTION
    )
    assert len(report.decisions) >= len(encounter.friends)


def test_expired_patrol_contact_is_not_renewed_by_eight_tick_flank_observation():
    from arena_hero_bot.memory import PatrolContact

    strategy, encounter = standoff()
    strategy.memory.campaign_last_core_id = object_id(1)
    strategy.memory.campaign_phase = 1
    strategy.memory.skirmish_cooldown_until = 200
    strategy.memory.patrol_contacts["patrol-1"] = PatrolContact(
        (-3, 2), 90, 99, object_id(50)
    )
    seed_contact(strategy, encounter)
    turn = encounter.turn()
    strategy.decide(turn)
    assert strategy.memory.flank.operation is None
    assert strategy.memory.patrol_contacts["patrol-1"].started_tick == 90
    patrol = next(
        ranger
        for ranger in turn.rangers
        if strategy.memory.unit_roles[str(ranger.id)] == "patrol-1"
    )
    assert not strategy._patrol_can_pursue(
        patrol, turn.visible_enemies[0], (3, -4), _context(strategy, turn)
    )


def test_parent_slot_change_cannot_reuse_an_old_qualified_permit():
    from arena_hero_bot.counterattack import AssaultSlot

    strategy, encounter, _ = _launched()
    operation = strategy.memory.flank.operation
    assert operation is not None and strategy.memory.skirmish is not None
    operation.phase = flank.Phase.ASSEMBLE
    turn = encounter.turn()
    context = _context(strategy, turn)
    strategy._reserve_flank(context, flank.Kind.SKIRMISH)
    member = next(
        ranger for ranger in turn.rangers if str(ranger.id) in operation.member_ids
    )
    destinations = {UUID(key): slot.firing for key, slot in operation.slots.items()}
    context.flank_entry_permit = flank.EntryPermit(
        turn.tick, operation.parent.task_id, destinations
    )
    assert strategy._flank_entry_allowed(member.id, destinations[member.id], context)
    strategy.memory.skirmish.slots[str(member.id)] = AssaultSlot(
        (99, 99), destinations[member.id]
    )
    assert not strategy._flank_entry_allowed(
        member.id, destinations[member.id], context
    )
    assert not strategy._queue_move(
        member,
        Direction.UP,
        context,
        reason="invalid parent must use ordinary ingress",
        target=destinations[member.id],
    )


def _context(strategy, turn):
    return _TurnContext(
        turn=turn,
        report=DecisionReport(tick=turn.tick),
        occupied={actor.position for actor in turn.units},
        enemy_positions={enemy.position for enemy in turn.visible_enemies},
        threat=strategy._combat_policy.assess(turn, strategy.memory.obstacles),
    )


def _launched():
    strategy, encounter = standoff()
    seed_contact(strategy, encounter)
    turn = encounter.turn()
    strategy.decide(turn)
    assert strategy.memory.flank.operation is not None
    return strategy, encounter, turn


@pytest.mark.parametrize(
    "response,losses,rounds", [("infer_after_hit", 0, 2), ("precise_pressure", 2, 3)]
)
def test_full_observation_entry_opening_finish_and_confirmed_results(
    response, losses, rounds, tmp_path
):
    strategy, encounter = standoff()
    encounter.tick = 93
    members = {object_id(2), object_id(3)}
    identity = None
    saw_partial_kill = False
    for _ in range(15):
        turn = encounter.turn()
        report = strategy.decide(turn)
        operation = strategy.memory.flank.operation
        enemy_view = encounter.view(enemy=True)
        orders = encounter.response(enemy_view, mode=response, member_ids=members)
        if turn.tick < 100:
            assert operation is None
            assert all(
                not (decision.actor_id in members and "flank" in decision.reason)
                for decision in report.decisions
            )
        if operation is not None:
            if identity is None:
                identity = (
                    operation.parent.task_id,
                    operation.parent.started,
                    operation.parent.deadline,
                    operation.member_ids,
                )
                assert operation.stable_ticks == 8
                assert turn.tick == 100
            assert (
                operation.parent.task_id,
                operation.parent.started,
                operation.parent.deadline,
                operation.member_ids,
            ) == identity
            if operation.phase is flank.Phase.ENTER:
                assert all(
                    turn.plan.unit_actions[UUID(key)].type == "MOVE" for key in members
                )
                assert all(encounter.friends[key]["hp"] == 2 for key in members)
                orders = encounter.response(enemy_view, mode="visible_front")
            if operation.phase is flank.Phase.OPENING:
                assert Counter(
                    str(turn.plan.unit_actions[UUID(key)].target_id) for key in members
                ) == {object_id(50): 2}
            if operation.phase is flank.Phase.FINISH:
                saw_partial_kill = True
                assert operation.remaining_ids == (object_id(51),)
                assert operation.confirmed_destroyed == {object_id(50)}
                assert operation.first_shot == 101
                assert turn.tick - operation.first_shot < 3
        encounter.settle(turn.plan, orders)
        if turn.tick == 101:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
        if not encounter.foes and strategy.memory.flank.operation is None:
            break
    assert saw_partial_kill
    assert encounter.kills == 2 and encounter.losses == losses
    result = strategy.memory.flank.last_result
    assert result is not None
    assert result["reason"] == "confirmed_targets_destroyed"
    assert result["first_shot"] == 101
    assert result["ended"] - result["first_shot"] == rounds
    assert len(result["confirmed_destroyed"]) == 2
    assert len(result["flank_losses"]) == losses
    assert result["front_losses"] == result["other_losses"] == []
    assert result["actual_hits"] == 4
    assert encounter.core["hp"] == encounter.core["shield"] == 5
    assert strategy.memory.skirmish is None
    assert set(result["members"]) == members


@pytest.mark.parametrize(
    "count,mode,spread,expected_kills,losses,rounds",
    [
        (2, "infer_after_hit", False, 2, 0, 2),
        (2, "infer_after_hit", True, 2, 1, 2),
        (2, "precise_pressure", False, 2, 2, 3),
        (3, "precise_pressure", False, 2, 1, 2),
        (4, "precise_pressure", False, 2, 1, 1),
        (4, "infer_after_hit", False, 2, 0, 1),
    ],
)
def test_s01_through_s06_real_simultaneous_settlement(
    count, mode, spread, expected_kills, losses, rounds
):
    cells = [(3, -4), (3, -4), (-3, -4), (-3, -4)]
    friends = [
        unit(index + 2, "RANGER", position=cell)
        for index, cell in enumerate(cells[:count])
    ]
    friends.append(unit(20, "RANGER", position=(0, -3)))
    foes = [unit(key, "RANGER", controlled=False, position=(0, -7)) for key in (50, 51)]
    encounter = FogEncounter(friends, foes)
    member_ids = {actor["id"] for actor in friends if actor["id"] != object_id(20)}
    for _round_index in range(4):
        turn = encounter.turn()
        positions = {
            ranger.id: ranger.position
            for ranger in turn.rangers
            if str(ranger.id) in member_ids
        }
        hp = {ranger.id: ranger.hp for ranger in turn.rangers if ranger.id in positions}
        targets = tuple(
            enemy for enemy in turn.visible_enemies if isinstance(enemy, UnitView)
        )
        allocation = flank.allocate_fire(positions, hp, targets, encounter.rocks)
        if spread:
            allocation = {
                ranger.id: targets[index % len(targets)]
                for index, ranger in enumerate(turn.rangers)
                if ranger.id in positions
            }
        for ranger in turn.rangers:
            if ranger.id in allocation:
                target = allocation[ranger.id]
                ranger.shoot(target, expected_cell=target.position)
        orders = encounter.response(
            encounter.view(enemy=True), mode=mode, member_ids=member_ids
        )
        encounter.settle(turn.plan, orders)
        if not encounter.foes:
            break
    assert encounter.kills == expected_kills
    assert encounter.losses == losses
    assert len(encounter.trace) == rounds
    if count == 4:
        assert sorted(
            Counter(
                action["target_id"]
                for action in encounter.trace[0]["plan"]["unit_actions"].values()
            ).values()
        ) == [2, 2]


def test_cell_shots_overkill_the_same_stacked_uuid_instead_of_splitting_damage():
    friends = [
        unit(index + 2, "RANGER", position=cell)
        for index, cell in enumerate([(3, -4), (3, -4), (-3, -4), (-3, -4)])
    ]
    friends.append(unit(20, "RANGER", position=(0, -3)))
    encounter = FogEncounter(
        friends,
        [unit(key, "RANGER", controlled=False, position=(0, -7)) for key in (50, 51)],
    )
    turn = encounter.turn()
    for ranger in turn.rangers:
        if ranger.id != UUID(int=20):
            ranger.shoot_cell((0, -7))
    encounter.settle(turn.plan, {})
    assert encounter.kills == 1
    assert encounter.foes[object_id(51)]["hp"] == 2


@pytest.mark.parametrize(
    "cause", ["moved", "observer", "rock", "core", "budget", "uuid", "known_observer"]
)
def test_active_assembly_is_invalidated_without_recruiting_or_refreshing(cause):
    strategy, encounter = standoff()
    # One actor needs a further safe assembly step.
    encounter.friends[object_id(3)]["position"] = [5, -4]
    seed_contact(strategy, encounter)
    first = encounter.turn()
    strategy.decide(first)
    operation = strategy.memory.flank.operation
    assert operation is not None and operation.phase is flank.Phase.ASSEMBLE
    original_ids = operation.member_ids
    encounter.settle(first.plan, encounter.response(encounter.view(enemy=True)))
    if cause == "moved":
        encounter.foes[object_id(50)]["position"][0] += 1
    elif cause == "observer":
        encounter.friends[object_id(20)]["position"] = [20, 20]
    elif cause == "rock":
        encounter.rocks |= {operation.slots[original_ids[0]].firing}
    elif cause == "core":
        encounter.core["shield"] = 4
    elif cause == "budget":
        encounter.tick = operation.parent.deadline
    elif cause == "uuid":
        encounter.foes[object_id(50)]["id"] = object_id(55)
    else:
        encounter.foes[object_id(70)] = unit(
            70,
            "WORKER",
            controlled=False,
            position=operation.slots[original_ids[0]].firing,
        )
    strategy.decide(encounter.turn())
    assert strategy.memory.flank.operation is None
    assert strategy.memory.flank.last_result["first_shot"] is None
    assert tuple(strategy.memory.flank.last_result["members"]) == original_ids


def test_partial_server_arrival_does_not_supplement_or_claim_an_opening():
    strategy, encounter, turn = _launched()
    operation = strategy.memory.flank.operation
    missing = UUID(operation.member_ids[0])
    submitted = turn.plan
    partial = submitted.model_copy(
        update={
            "unit_actions": {
                key: action
                for key, action in submitted.unit_actions.items()
                if key != missing
            }
        }
    )
    encounter.settle(partial, encounter.response(encounter.view(enemy=True)))
    report = strategy.decide(encounter.turn())
    assert strategy.memory.flank.operation is None
    result = strategy.memory.flank.last_result
    assert result["reason"] == "arrival_incomplete" and result["first_shot"] is None
    assert len(result["actual_arrivals"]) == 1
    assert all(
        "blind flank entry" not in decision.reason for decision in report.decisions
    )


def test_actual_arrival_damage_replaces_the_zero_damage_forecast():
    strategy, encounter, turn = _launched()
    members = strategy.memory.flank.operation.member_ids
    # This is deliberately an unexpected exact blind shot on the MOVE Tick.
    orders = {
        object_id(50): {
            "type": "SHOOT",
            "target_id": members[0],
            "expected_cell": [3, -4],
        }
    }
    encounter.settle(turn.plan, orders)
    strategy.decide(encounter.turn())
    operation = strategy.memory.flank.operation
    if operation is not None:
        assert operation.actual_hp[members[0]] == 1
        assert operation.flank_damage == 1
    else:
        assert strategy.memory.flank.last_result["actual_hp"][members[0]] == 1
        assert strategy.memory.flank.last_result["flank_damage"] == 1


def test_an_unexecuted_opening_does_not_count_a_kill_or_refresh_the_fire_budget():
    strategy, encounter, entry = _launched()
    operation = strategy.memory.flank.operation
    identity = (operation.parent.task_id, operation.parent.deadline)
    encounter.settle(entry.plan, encounter.response(encounter.view(enemy=True)))
    opening = encounter.turn()
    strategy.decide(opening)
    assert operation.first_shot == opening.tick
    partial = opening.plan.model_copy(
        update={
            "unit_actions": {
                key: action
                for key, action in opening.plan.unit_actions.items()
                if str(key) not in operation.member_ids
            }
        }
    )
    encounter.settle(partial, encounter.response(encounter.view(enemy=True)))
    strategy.decide(encounter.turn())
    result = strategy.memory.flank.last_result
    assert strategy.memory.flank.operation is None
    assert (result["parent_id"], result["deadline"]) == identity
    assert result["first_shot"] == opening.tick
    assert result["reason"] == "exchange_unfavorable"
    assert result["confirmed_destroyed"] == [] and result["actual_hits"] == 0
    assert all(enemy["hp"] == 2 for enemy in encounter.foes.values())
    assert encounter.kills == 0


def test_finish_can_use_current_worker_observation_after_front_mutual_sight_ends():
    strategy, encounter, turn = _launched()
    encounter.settle(turn.plan, encounter.response(encounter.view(enemy=True)))
    opening = encounter.turn()
    strategy.decide(opening)
    encounter.settle(opening.plan, encounter.response(encounter.view(enemy=True)))
    encounter.friends[object_id(20)]["position"] = [20, 20]
    encounter.friends[object_id(80)] = unit(80, "WORKER", position=(0, -5))
    finish = encounter.turn()
    report = strategy.decide(finish)
    assert strategy.memory.flank.operation is not None
    assert strategy.memory.flank.operation.phase is flank.Phase.FINISH
    assert any(
        "finish original flank" in decision.reason for decision in report.decisions
    )
    assert all(
        action.type == "SHOOT"
        for key, action in finish.plan.unit_actions.items()
        if str(key) in {object_id(2), object_id(3)}
    )


@pytest.mark.parametrize(
    "gate", ["rock", "enemy", "service", "capacity", "unknown_damage"]
)
def test_exact_entry_permit_does_not_bypass_physical_or_service_gates(gate):
    strategy, encounter, _turn = _launched()
    operation = strategy.memory.flank.operation
    operation.phase = flank.Phase.ASSEMBLE
    turn = encounter.turn()
    context = _context(strategy, turn)
    strategy._reserve_flank(context, flank.Kind.SKIRMISH)
    members = tuple(
        ranger for ranger in turn.rangers if str(ranger.id) in operation.member_ids
    )
    destinations = {
        member.id: operation.slots[str(member.id)].firing for member in members
    }
    context.flank_entry_permit = flank.EntryPermit(
        turn.tick, operation.parent.task_id, destinations
    )
    cell = next(iter(destinations.values()))
    if gate == "rock":
        strategy.memory.obstacles.add(cell)
    elif gate == "enemy":
        context.enemy_positions.add(cell)
    elif gate == "service":
        context.core_migration_lane = (cell,)
    elif gate == "capacity":
        context.friendly_occupancy[cell] = 1
    else:
        strategy.memory.incoming_damage[cell] = (turn.tick - 1, 2)
    before = deepcopy(turn.plan.model_dump())
    assert not strategy._flank_atomic_moves(context, members, destinations)
    assert turn.plan.model_dump() == before
    assert not context.arrival_counts and not context.departure_counts


def test_last_move_failure_restores_all_action_capacity_fire_and_member_ledgers(
    monkeypatch,
):
    strategy, encounter, _turn = _launched()
    operation = strategy.memory.flank.operation
    operation.phase = flank.Phase.ASSEMBLE
    turn = encounter.turn()
    context = _context(strategy, turn)
    strategy._reserve_flank(context, flank.Kind.SKIRMISH)
    members = tuple(
        ranger for ranger in turn.rangers if str(ranger.id) in operation.member_ids
    )
    destinations = {
        member.id: operation.slots[str(member.id)].firing for member in members
    }
    context.flank_entry_permit = flank.EntryPermit(
        turn.tick, operation.parent.task_id, destinations
    )
    old_move = strategy._queue_move
    context.volley_aims["existing"] = [(1, 1)]
    context.report.flank["existing"] = {"value": 1}
    ledger_names = (
        "arrival_counts",
        "departure_counts",
        "reserved",
        "standing_reserved",
        "planned_ids",
        "core_traffic_moves",
        "defensive_fire_cells",
        "defensive_firing_lanes",
        "volley_aims",
        "counterattack_ids",
        "counterattack_reserved",
        "flank_ids",
        "flank_front_ids",
    )
    before = {name: deepcopy(getattr(context, name)) for name in ledger_names}
    before_plan = deepcopy(turn.plan.model_dump())
    before_report = deepcopy(context.report.to_dict())
    before_damage = context.damage_ledger.planned_damage.copy()
    old_targets = strategy.memory.pending_move_targets.copy()

    def fail_last(member, direction, candidate, **kwargs):
        if member.id == members[-1].id:
            candidate.damage_ledger.planned_damage[object_id(99)] = 9
            candidate.volley_aims["injected"] = [(99, 99)]
            candidate.volley_aims["existing"].append((99, 99))
            candidate.report.flank["existing"]["value"] = 99
            candidate.flank_ids = frozenset()
            candidate.defensive_fire_cells[(99, 99)] = 2
            candidate.defensive_firing_lanes.add((99, 99))
            return False
        return old_move(member, direction, candidate, **kwargs)

    monkeypatch.setattr(strategy, "_queue_move", fail_last)
    assert not strategy._flank_atomic_moves(context, members, destinations)
    assert turn.plan.model_dump() == before_plan
    assert strategy.memory.pending_move_targets == old_targets
    for name in ledger_names:
        assert getattr(context, name) == before[name]
    assert context.report.to_dict() == before_report
    assert context.damage_ledger.planned_damage == before_damage


def test_entry_can_use_a_real_safe_same_tick_departure_but_not_a_promise():
    strategy, encounter, _turn = _launched()
    operation = strategy.memory.flank.operation
    operation.phase = flank.Phase.ASSEMBLE
    encounter.friends[object_id(80)] = unit(80, "RANGER", position=(3, -4))
    turn = encounter.turn()
    context = _context(strategy, turn)
    strategy._reserve_flank(context, flank.Kind.SKIRMISH)
    members = tuple(
        ranger for ranger in turn.rangers if str(ranger.id) in operation.member_ids
    )
    destinations = {
        member.id: operation.slots[str(member.id)].firing for member in members
    }
    context.flank_entry_permit = flank.EntryPermit(
        turn.tick, operation.parent.task_id, destinations
    )
    assert not strategy._flank_atomic_moves(context, members, destinations)
    departing = next(ranger for ranger in turn.rangers if ranger.id.int == 80)

    assert strategy._queue_move(
        departing,
        Direction.LEFT,
        context,
        reason="safe selected departure",
        target=(2, -4),
    )
    # Only the already selected safe departure releases this capacity.
    assert strategy._flank_atomic_moves(context, members, destinations)
    encounter.settle(turn.plan, {})
    assert (
        sum(actor["position"] == [3, -4] for actor in encounter.friends.values()) == 2
    )


def test_unrelated_kills_do_not_subsidize_a_losing_local_fire_group():
    turn = make_turn(
        objects=[
            core(position=(20, 20)),
            unit(50, "RANGER", controlled=False, hp=1),
            unit(51, "RANGER", controlled=False, hp=1, position=(10, 10)),
            unit(52, "RANGER", controlled=False, position=(3, 0)),
        ]
    )
    positions = {UUID(int=2): (3, 3), UUID(int=3): (0, 3), UUID(int=4): (13, 13)}
    hp = {UUID(int=2): 1, UUID(int=3): 1, UUID(int=4): 2}
    enemies = tuple(
        enemy for enemy in turn.visible_enemies if isinstance(enemy, UnitView)
    )
    result = flank.exchange(positions, hp, enemies[:2], enemies, set(), 3)
    assert not result.viable


@pytest.mark.parametrize("memory_position,empty", [((6, -4), True), ((0, -12), False)])
def test_recent_fire_sources_respect_current_empty_vision_and_uuid_updates(
    memory_position, empty
):
    strategy, encounter = standoff()
    seed_contact(strategy, encounter)
    strategy.memory.enemies[object_id(60)] = EnemySighting(
        object_id(60), "UNIT", memory_position, 2, 99, "RANGER"
    )
    turn = encounter.turn()
    strategy.memory.observe(turn)
    observers = strategy._flank_observers(turn)
    assert any(observer.id.int == 60 for observer in observers) is not empty
    if not empty:
        assert (
            flank.cell_rejection(
                (3, -10),
                turn.visible_enemies,
                observers,
                strategy._ranger_threats(turn),
                set(),
                set(),
            )
            == "known_observer"
        )


def test_long_counterattack_keeps_long_contact_gate_pairs_and_full_volley():
    strategy, encounter = standoff(side_count=4)
    for index, position in enumerate([(0, 6), (-6, 0), (6, 0), (0, 8)]):
        actor = unit(80 + index, "RANGER", position=position)
        encounter.friends[actor["id"]] = actor
        strategy.memory.unit_roles[actor["id"]] = "patrol-1"
    seed_contact(strategy, encounter)
    for contact in strategy.memory.counterattack.contacts.values():
        contact.since = 36
    turn = encounter.turn()
    strategy.decide(turn)
    operation = strategy.memory.flank.operation
    assert operation is not None and operation.parent.kind is flank.Kind.COUNTERATTACK
    assert len(operation.member_ids) == 4
    assert paired_cells(tuple(slot.firing for slot in operation.slots.values()))
    assert operation.parent.max_steps == 8 and operation.parent.fire_ticks == 4
    encounter.settle(turn.plan, encounter.response(encounter.view(enemy=True)))
    opening = encounter.turn()
    strategy.decide(opening)
    targets = Counter(
        str(action.target_id)
        for key, action in opening.plan.unit_actions.items()
        if str(key) in operation.member_ids
    )
    assert sorted(targets.values()) == [2, 2]
    encounter.settle(opening.plan, encounter.response(encounter.view(enemy=True)))
    strategy.decide(encounter.turn())
    assert encounter.kills == 2
    assert strategy.memory.flank.last_result["reason"] == "confirmed_targets_destroyed"


def test_worker_protection_preserves_four_member_full_hp_compilation_and_atomic_entry():
    strategy, encounter = standoff(side_count=4, home=(16, 16))
    members = {object_id(key) for key in (2, 3, 4, 5)}
    strategy.memory.unit_roles = {
        key: "patrol-1" if key in members else "defense" for key in encounter.friends
    }
    seed_contact(strategy, encounter)
    turn = encounter.turn()
    strategy.decide(turn)
    operation = strategy.memory.flank.operation
    assert operation is not None and operation.parent.kind is flank.Kind.PROTECTION
    assert set(operation.member_ids) == members
    assert operation.parent.deadline == 132
    assert all(turn.plan.unit_actions[UUID(key)].type == "MOVE" for key in members)
    encounter.settle(turn.plan, encounter.response(encounter.view(enemy=True)))
    opening = encounter.turn()
    strategy.decide(opening)
    assert sorted(
        Counter(
            str(opening.plan.unit_actions[UUID(key)].target_id) for key in members
        ).values()
    ) == [2, 2]
    encounter.settle(opening.plan, encounter.response(encounter.view(enemy=True)))
    strategy.decide(encounter.turn())
    assert encounter.kills == 2 and encounter.losses == 0
    assert strategy.memory.worker_area_protection is None
    assert not strategy.memory.worker_area_protection_cooldowns


def test_short_mission_compares_complete_two_and_four_member_combinations():
    strategy, encounter = standoff(side_count=4)
    seed_contact(strategy, encounter)
    turn = encounter.turn()
    strategy.decide(turn)
    operation = strategy.memory.flank.operation
    assert operation is not None and operation.parent.kind is flank.Kind.SKIRMISH
    assert len(operation.member_ids) == 4
    assert all(
        turn.plan.unit_actions[UUID(key)].type == "MOVE" for key in operation.member_ids
    )
    encounter.settle(turn.plan, encounter.response(encounter.view(enemy=True)))
    opening = encounter.turn()
    strategy.decide(opening)
    assert sorted(
        Counter(
            str(opening.plan.unit_actions[UUID(key)].target_id)
            for key in operation.member_ids
        ).values()
    ) == [2, 2]
    encounter.settle(opening.plan, encounter.response(encounter.view(enemy=True)))
    strategy.decide(encounter.turn())
    assert encounter.kills == 2 and encounter.losses == 0
    assert strategy.memory.flank.operation is None


def test_a_distant_vanguard_does_not_invent_ranged_fire_against_a_legal_flank():
    strategy, encounter = standoff()
    strategy.memory.campaign_last_core_id = object_id(1)
    strategy.memory.campaign_phase = 1
    encounter.friends[object_id(3)]["position"] = [3, -3]
    enemy = unit(70, "VANGUARD", controlled=False, position=(6, -7))
    encounter.foes[enemy["id"]] = enemy
    encounter.hostile_ids.add(enemy["id"])
    seed_contact(strategy, encounter)
    strategy.memory.enemies[enemy["id"]] = EnemySighting(
        enemy["id"], "UNIT", (6, -7), 4, encounter.tick - 1, "VANGUARD"
    )
    assert all(other.id.int != 70 for other in encounter.turn().visible_enemies)
    strategy.decide(encounter.turn())
    operation = strategy.memory.flank.operation
    assert operation is not None and operation.parent.kind is flank.Kind.SKIRMISH
    assert operation.phase is flank.Phase.ENTER
    assert set(operation.target_ids) == {object_id(50), object_id(51)}
    assert all(
        abs(slot.firing[0] - 6) + abs(slot.firing[1] + 7) > 2
        for slot in operation.slots.values()
    )


def test_one_extra_assembly_tick_can_win_a_complete_opening_volley():
    strategy, encounter = standoff(side_count=4)
    encounter.friends[object_id(4)]["position"] = [-3, -2]
    encounter.friends[object_id(5)]["position"] = [-4, -3]
    seed_contact(strategy, encounter)
    started = encounter.tick
    turn = encounter.turn()
    strategy.decide(turn)
    operation = strategy.memory.flank.operation
    assert operation is not None and operation.parent.kind is flank.Kind.SKIRMISH
    assert len(operation.member_ids) == 4
    assert operation.phase is flank.Phase.ASSEMBLE
    assert sorted(
        turn.plan.unit_actions[UUID(key)].type for key in operation.member_ids
    ) == ["MOVE", "MOVE", "WAIT", "WAIT"]
    for _ in range(5):
        encounter.settle(turn.plan, encounter.response(encounter.view(enemy=True)))
        turn = encounter.turn()
        strategy.decide(turn)
        if operation.first_shot is not None:
            assert operation.first_shot == started + 2
            assert sorted(
                Counter(
                    str(turn.plan.unit_actions[UUID(key)].target_id)
                    for key in operation.member_ids
                ).values()
            ) == [2, 2]
            encounter.settle(turn.plan, encounter.response(encounter.view(enemy=True)))
            strategy.decide(encounter.turn())
            break
    else:
        pytest.fail("complete first volley never followed the safe assembly")
    assert encounter.kills == 2 and encounter.losses == 0
    assert strategy.memory.flank.last_result["first_shot"] == started + 2


def test_a_one_hp_target_is_not_delayed_by_an_earlier_failed_search():
    strategy, encounter = standoff(opponents=1, target_hp=1)
    strategy.config = replace(strategy.config, blind_flank=False)
    strategy.memory.skirmish_retry_after = encounter.tick + 4
    seed_contact(strategy, encounter)
    strategy.decide(encounter.turn())
    assert strategy.memory.skirmish is not None
    assert strategy.memory.skirmish.started == encounter.tick


def test_worker_destruction_is_recorded_without_a_separate_damage_event():
    strategy, encounter = standoff()
    worker = unit(80, "WORKER")
    encounter.friends[worker["id"]] = worker
    encounter.friendly_ids.add(worker["id"])
    seed_contact(strategy, encounter)
    strategy.decide(encounter.turn())
    operation = strategy.memory.flank.operation
    assert operation is not None
    del encounter.friends[worker["id"]]
    encounter.tick += 1
    encounter.events = [
        {
            "event_id": object_id(90001),
            "tick": encounter.tick - 1,
            "event_type": "UNIT_DESTROYED",
            "actor_id": object_id(50),
            "target_id": worker["id"],
            "position": worker["position"],
        }
    ]
    for _ in range(2):
        strategy._observe_flank_contacts(encounter.turn())
    assert operation.worker_losses == operation.other_losses == {worker["id"]}
    assert not operation.front_losses and not operation.flank_losses


def test_legal_local_fireline_flank_survives_restart_without_expanding_permissions(
    tmp_path,
):
    guards = [
        (2, (-2, 1)),
        (3, (0, 1)),
        (4, (2, 1)),
        (20, (0, -5)),
        (21, (0, 3)),
        (30, (3, -5)),
        (31, (-3, -5)),
    ]
    guards = [(key, (cell[0] + 16, cell[1] + 16)) for key, cell in guards]
    patrols = [(40 + i, (i - 3, 5)) for i in range(7)]
    patrols = [(key, (cell[0] + 16, cell[1] + 16)) for key, cell in patrols]
    friends = [unit(key, "RANGER", position=cell) for key, cell in guards + patrols]
    encounter = FogEncounter(
        friends, [unit(50, "RANGER", controlled=False, position=(16, 7))], home=(16, 16)
    )
    memory = WorldMemory(
        core_home_position=(16, 16),
        core_layout_version=3,
        campaign_last_core_id=object_id(1),
        campaign_phase=1,
        unit_roles={object_id(key): "defense" for key, _ in guards}
        | {object_id(key): "patrol-1" for key, _ in patrols},
        unit_roles_initialized=True,
        defense_anchor=(16, 16),
        defense_posts={object_id(key): cell for key, cell in guards},
    )
    from arena_hero_bot.strategy import StrategyConfig

    strategy = AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))
    strategy.memory.skirmish_cooldown_until = 200
    strategy.memory.worker_area_protection_cooldowns[object_id(50)] = 200
    seed_contact(strategy, encounter)
    turn = encounter.turn()
    strategy.decide(turn)
    operation = strategy.memory.flank.operation
    assert operation is not None and operation.parent.kind is flank.Kind.FIRELINE
    assert operation.parent.max_steps == 3 and operation.parent.deadline == 108
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    restored = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    assert restored.memory.flank.operation is not None
    assert restored._flank_parent_valid(restored.memory.flank.operation)
    encounter.settle(turn.plan, encounter.response(encounter.view(enemy=True)))
    for _ in range(3):
        opening = encounter.turn()
        restored.decide(opening)
        if operation.first_shot is not None or (
            restored.memory.flank.operation is not None
            and restored.memory.flank.operation.first_shot is not None
        ):
            break
        encounter.settle(opening.plan, encounter.response(encounter.view(enemy=True)))
    assert (
        sum(
            action.type == "SHOOT" and action.target_id == UUID(int=50)
            for action in opening.plan.unit_actions.values()
        )
        == 2
    )


def test_disabled_search_and_expired_parent_preserve_normal_shots_and_services():
    strategy, encounter = standoff()
    strategy.config = replace(strategy.config, blind_flank=False)
    seed_contact(strategy, encounter)
    report = strategy.decide(encounter.turn())
    assert strategy.memory.flank.operation is None
    assert any(decision.action == "SHOOT" for decision in report.decisions)
    assert len(report.decisions) >= len(encounter.friends)
    strategy, encounter, _turn = _launched()
    operation = strategy.memory.flank.operation
    strategy.memory.skirmish = None
    encounter.tick += 1
    strategy.decide(encounter.turn())
    assert strategy.memory.flank.operation is None
    assert strategy.memory.flank.last_result["parent_id"] == operation.parent.task_id
    assert strategy.memory.flank.last_result["reason"] == "orphaned_parent"


def test_budget_exhaustion_falls_back_without_partial_group_actions(monkeypatch):
    strategy, encounter = standoff()
    seed_contact(strategy, encounter)
    old_choose = flank.choose

    def expired(*args, **kwargs):
        args[10].deadline = monotonic() - 1
        return old_choose(*args, **kwargs)

    monkeypatch.setattr(flank, "choose", expired)
    report = strategy.decide(encounter.turn())
    assert strategy.memory.flank.operation is None
    assert report.flank["search"]["rejections"]["search_budget_exhausted"] >= 1
    assert len(report.decisions) >= len(encounter.friends)
    assert all(
        "blind flank entry" not in decision.reason for decision in report.decisions
    )


@pytest.mark.parametrize(
    "source,manual", [("AGENT", False), ("PLAYER", True), (None, None)]
)
def test_telemetry_tracks_actual_results_separately_from_submission_source(
    source, manual
):
    strategy, _encounter, turn = _launched()
    report = DecisionReport(tick=turn.tick)
    record = _turn_record(
        turn=turn,
        observed_at="2026-01-01T00:00:00Z",
        report=report,
        planning_seconds=0.01,
        planning_error=None,
        submission={"accepted": True, "source": source},
        observe_only=False,
        identity_snapshot=_identity_snapshot(strategy.memory),
    )
    assert record["control"]["manual_override"] is manual
    summary = record["identity"]["flank"]["operation"]
    assert summary["confirmed_destroyed"] == []
    assert summary["actual_arrivals"] == []
    assert summary["opening_allocation"]
