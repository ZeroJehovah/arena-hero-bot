"""Actual losses, affordable replenishment, and bounded strategic recovery."""

from uuid import UUID

import pytest
from arena_hero import UnitType

from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.migration import resource_block_index, resource_block_quota
from arena_hero_bot.recovery import Recovery
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, object_id, unit
from .test_encounter_combat import _context
from .test_layered_defense import Encounter


def _strategy(rangers=30, workers=16, *, resources=0, position=(16, 16)):
    memory = WorldMemory(
        campaign_last_core_id=object_id(1),
        campaign_phase=4,
        core_home_position=position,
        core_layout_version=CORE_LAYOUT_VERSION,
        unit_roles_initialized=True,
    )
    objects = [core(position=position)]
    for n in range(workers + rangers):
        x, y = position
        objects.append(
            unit(
                100 + n,
                "WORKER" if n < workers else "RANGER",
                position=(x - 6 + n % 12, y + 2 + n // 12),
            )
        )
        if n >= workers:
            memory.unit_roles[object_id(100 + n)] = "defense" if n % 2 else "patrol-1"
    memory.recovery = Recovery(
        core_id=object_id(1), reserve=60, danger=(16, 4), danger_tick=190
    )
    memory.recovery.samples = [
        (tick, 0, 4, int(tick % 20 == 0)) for tick in range(72, 200)
    ]
    turn = make_turn(tick=200, resources=resources, objects=objects)
    strategy = AggressiveStrategy(
        memory,
        StrategyConfig(campaign_mode=True, expedition_mode=True, max_population=None),
    )
    memory.exploration.mark(
        {(x, y) for x in range(-32, 64) for y in range(-32, 64)}, 199
    )
    return strategy, turn, objects


@pytest.mark.parametrize("restart", [False, True])
def test_war_reserve_builds_before_full_storage_then_restores_surplus_gate(
    tmp_path, restart
):
    strategy, turn, _ = _strategy(rangers=30, resources=150)
    if restart:
        path = tmp_path / "memory.json"
        strategy.memory.save(path)
        strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    context = _context(strategy, turn)
    context.core_build_priority = strategy._campaign_build_priority(context)
    assert not strategy._normal_growth_on_cooldown(context)
    assert strategy._campaign_priority_spawn(context)
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.core_action.unit_type is UnitType.RANGER
    complete, full_turn, _ = _strategy(rangers=60, resources=100)
    assert complete._normal_growth_on_cooldown(_context(complete, full_turn))


@pytest.mark.parametrize(
    "workers,rangers,expected",
    [(7, 20, "WORKER"), (8, 20, "RANGER"), (12, 45, "WORKER"), (16, 45, "RANGER")],
)
def test_wartime_replacements_keep_an_economic_floor_and_share_deficits(
    workers, rangers, expected
):
    strategy, turn, _ = _strategy(workers=workers, rangers=rangers)
    assert strategy._campaign_spawn_candidates(turn) == (UnitType(expected),)


def test_loss_ledger_uses_actual_own_settlements_and_survives_restart(tmp_path):
    state = Recovery()
    friends = [core(), unit(2, "RANGER"), unit(3, "WORKER", cargo=1)]
    state.observe(make_turn(tick=100, objects=friends))
    events = [
        {
            "event_id": object_id(900),
            "tick": 100,
            "event_type": "UNIT_DAMAGED",
            "target_id": object_id(2),
            "values": {"damage": 2, "hp": 0},
        },
        {
            "event_id": object_id(901),
            "tick": 100,
            "event_type": "UNIT_DESTROYED",
            "target_id": object_id(2),
        },
        {
            "event_id": object_id(902),
            "tick": 100,
            "event_type": "UNIT_DESTROYED",
            "target_id": object_id(50),
        },
        {
            "event_id": object_id(903),
            "tick": 100,
            "event_type": "DEPOSIT_SUCCEEDED",
            "actor_id": object_id(3),
            "values": {"amount": 1},
        },
    ]
    turn = make_turn(tick=101, objects=[friends[0], friends[2]], events=events)
    state.observe(turn)
    assert state.summary()["ranger_losses"] == 1
    assert state.summary()["deposits"] == 1
    assert state.summary()["replacement_cost"] > 0
    memory = WorldMemory(recovery=state)
    path = tmp_path / "memory.json"
    memory.save(path)
    restored = WorldMemory.load(path).recovery
    restored.observe(turn)
    assert restored.to_dict() == state.to_dict()
    restored.observe(make_turn(tick=103, objects=[friends[0]]))
    assert restored.summary()["ranger_losses"] == 0
    restored.holding = True
    restored.observe(make_turn(tick=104, objects=[core(9)]))
    assert not restored.holding and restored.reserve == 20


@pytest.mark.parametrize(
    "gate", ["open", "rich", "roster", "outside", "unknown", "rock", "damage"]
)
def test_recovery_hop_requires_evidence_and_an_observed_safe_complete_route(gate):
    strategy, turn, objects = _strategy(
        resources=1000 if gate == "rich" else 0, rangers=55 if gate == "roster" else 30
    )
    if gate == "outside":
        objects.append(unit(500, "WORKER", position=(40, 16)))
        turn = make_turn(tick=200, objects=objects)
    if gate == "unknown":
        strategy.memory.exploration.tiles.clear()
    if gate == "rock":
        strategy.memory.obstacles.update({(15, 16), (17, 16), (16, 15), (16, 17)})
    if gate == "damage":
        strategy.memory.incoming_damage.update(
            dict.fromkeys(((15, 16), (17, 16), (16, 15), (16, 17)), (199, 1))
        )
    assert strategy._continue_campaign_recovery(turn) is (gate == "open")
    state = strategy.memory.recovery
    if gate == "open":
        assert turn.core is not None
        assert state.holding and state.target == strategy.memory.core_home_position
        assert len(state.route) <= 65 and state.route[0] == turn.core.position
        assert resource_block_quota(resource_block_index(state.target)) <= 16
        context = _context(strategy, turn)
        assert strategy._core_home_direction(context) is not None
        assert {resource_block_index(cell) for cell in state.route} == {
            resource_block_index(turn.core.position),
            resource_block_index(state.target),
        }
    else:
        assert state.target is None


def test_recovery_does_not_restart_its_hop_or_allow_an_unsafe_next_step(tmp_path):
    strategy, turn, objects = _strategy()
    assert strategy._continue_campaign_recovery(turn)
    target = strategy.memory.recovery.target
    next_cell = strategy.memory.recovery.route[1]
    path = tmp_path / "memory.json"
    strategy.memory.save(path)
    strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
    strategy.memory.obstacles.add(next_cell)
    assert strategy._core_home_direction(_context(strategy, turn)) is None
    assert strategy._continue_campaign_recovery(make_turn(tick=201, objects=objects))
    assert strategy.memory.recovery.target == target
    assert strategy.memory.campaign_emergency_started_tick == 200
    arrived = make_turn(tick=330, objects=[core(position=target)])
    assert strategy._continue_campaign_recovery(arrived)
    assert strategy.memory.recovery.target is None and strategy.memory.recovery.holding
    assert strategy._continue_campaign_recovery(
        make_turn(tick=800, objects=[core(position=target)])
    )
    assert strategy.memory.core_home_position == target


def test_direct_fire_blocks_strategic_step_and_keeps_forced_retaliation():
    strategy, turn, objects = _strategy()
    assert strategy._continue_campaign_recovery(turn)
    contact = make_turn(
        tick=201,
        objects=[*objects, unit(900, "RANGER", controlled=False, position=(16, 13))],
    )
    context = _context(strategy, contact)
    assert strategy._core_home_direction(context) is None
    assert strategy._force_core_defense(context)
    assert any(
        action.type in {"MOVE", "SHOOT"}
        for key, action in contact.plan.unit_actions.items()
        if key != UUID(int=1)
    )


@pytest.mark.parametrize("restart", [False, True])
def test_strategic_hop_moves_with_the_real_four_tick_clock_and_keeps_identity(
    tmp_path, restart
):
    strategy, turn, objects = _strategy()
    assert strategy._continue_campaign_recovery(turn)
    target = strategy.memory.recovery.target
    strategy.memory.recovery.last_tick = 199
    strategy.memory.recovery.roster = {
        str(unit.id): unit.view.unit_type.value for unit in turn.units
    }
    encounter = Encounter(objects[1:], [], resources=0)
    encounter.core = objects[0]
    encounter.tick = 200
    for tick in range(200, 209):
        if restart and tick == 204:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
        current = encounter.turn()
        strategy.decide(current)
        assert strategy.memory.recovery.target == target
        assert strategy.memory.campaign_emergency_started_tick == 200
        if tick in (200, 204, 208):
            assert current.plan.core_action is not None
            assert current.plan.core_action.type == "START_MOVE"
        else:
            assert current.plan.core_action is None
        encounter.settle(current.plan, {})
    assert encounter.core["position"] == [16, 18]


@pytest.mark.parametrize("unsafe", [False, True])
def test_layered_posts_only_release_a_strategic_move_with_zero_projected_damage(unsafe):
    strategy, turn, objects = _strategy()
    assert strategy._continue_campaign_recovery(turn)
    contact = make_turn(
        tick=201,
        objects=[
            *objects,
            unit(
                900,
                "RANGER",
                controlled=False,
                position=(16, 13) if unsafe else (25, 17),
            ),
        ],
    )
    context = _context(strategy, contact)
    context.layered_assignments = {contact.rangers[0].id: contact.rangers[0].position}
    strategy._prepare_core_migration_step(context)
    strategy._decide_core(context)
    assert contact.plan.core_action is not None
    assert (contact.plan.core_action.type == "START_MOVE") is (not unsafe)


@pytest.mark.parametrize("gate", ["losses", "observations", "income", "valid"])
def test_attrition_warning_requires_a_real_sustained_funding_deficit(gate):
    strategy, turn, _ = _strategy()
    state = strategy.memory.recovery
    if gate == "losses":
        state.samples = [(t, 0, 4, 0) for t in range(72, 200)]
    if gate == "observations":
        state.samples = state.samples[-127:]
    if gate == "income":
        state.samples = [(t, 4, 4, int(t % 20 == 0)) for t in range(72, 200)]
    strategy._continue_campaign_recovery(turn)
    assert state.underfunded is (gate == "valid")
    assert state.summary()["underfunded"] is (gate == "valid")


def test_lower_quota_restores_the_local_reserve_and_prevents_permanent_recovery():
    strategy, turn, _ = _strategy(rangers=20, position=(16, -2000), resources=50)
    strategy.memory.recovery.holding = True
    strategy.memory.recovery.retry_after = 100
    strategy.memory.recovery.danger_tick = 100
    assert strategy._campaign_max_formation_complete(turn)
    assert strategy._continue_campaign_recovery(turn)
    assert not strategy.memory.recovery.holding
    assert not strategy.memory.recovery.samples


def test_blocked_recovery_expires_at_its_original_deadline():
    strategy, turn, objects = _strategy()
    assert strategy._continue_campaign_recovery(turn)
    assert strategy._continue_campaign_recovery(make_turn(tick=713, objects=objects))
    assert strategy.memory.recovery.target is None
    assert strategy.memory.recovery.holding
    assert strategy.memory.campaign_emergency_target is None
    assert resource_block_index(strategy.memory.core_home_position) == (0, 0)
