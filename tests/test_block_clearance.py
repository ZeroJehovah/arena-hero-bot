"""Actual local vision and confirmed enemy Core destruction gate each hop."""

from __future__ import annotations

import pytest

from arena_hero_bot.block_clearance import BlockClearance, Survey, block_at, cells_in
from arena_hero_bot.exploration import visible_cells
from arena_hero_bot.memory import EnemySighting, WorldMemory
from arena_hero_bot.runtime import _identity_snapshot
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, object_id
from .test_campaign import _gathered_roster
from .test_core_traffic import settled_positions

HOME = (-945, -945)


@pytest.fixture(autouse=True)
def complete_plans(monkeypatch):
    def timed_out(*_args):
        pytest.fail("local survey exhausted the planning budget")

    monkeypatch.setattr(AggressiveStrategy, "_wait_unplanned_units", timed_out)


def scenario(home=HOME, rangers=8, *, seen=False):
    workers, fighters, roles = _gathered_roster(rangers, center=home)
    block = block_at(home)
    memory = WorldMemory(
        core_home_position=home,
        core_migration_last_position=home,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_last_core_id=object_id(1),
        campaign_phase=3 if rangers < 20 else 4,
        unit_roles=roles,
        unit_roles_initialized=True,
        block_clearance=BlockClearance(
            core_id=object_id(1),
            current=block,
            blocks={
                block: Survey(seen=dict.fromkeys(cells_in(block), 1) if seen else {})
            },
        ),
    )
    return memory, [*workers, *fighters]


def strategy(memory):
    return AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))


def destruction(number, tick, event_type="CORE_DESTROYED"):
    return {
        "event_id": object_id(10000 + tick + number),
        "tick": tick,
        "event_type": event_type,
        "target_id": object_id(number),
    }


@pytest.mark.parametrize("home,rangers", [(HOME, 8), ((815, 15), 30)])
@pytest.mark.parametrize("restart", [False, True])
def test_last_unseen_cell_holds_hop_and_scout_until_real_observation(
    home, rangers, restart, tmp_path
):
    memory, roster = scenario(home, rangers, seen=True)
    block = block_at(home)
    gap = (block[0] * 32, block[1] * 32)
    memory.block_clearance.blocks[block].seen.pop(gap)
    bot = strategy(memory)
    first = make_turn(objects=[core(position=home), *roster])
    report = bot.decide(first)
    assert memory.core_home_position == home
    assert memory.campaign_scout_target is None
    assert memory.campaign_resting
    assert first.plan.core_action is None
    assert any(
        d.action == "MOVE" and "current block" in d.reason for d in report.decisions
    )
    assert gap not in memory.block_clearance.blocks[block].seen
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
        bot = strategy(memory)
    # An actual scout sighting completes the survey, not its prior plan.
    roster[-1]["position"] = [gap[0] + 2, gap[1] + 2]
    second = make_turn(tick=101, objects=[core(position=home), *roster])
    bot.decide(second)
    assert gap in memory.block_clearance.blocks[block].seen
    if rangers == 8:
        assert memory.core_home_position != home
        assert second.plan.core_action is not None
        assert second.plan.core_action.type == "START_MOVE"
    else:
        assert memory.campaign_scout_target is not None
        assert memory.core_home_position == home


@pytest.mark.parametrize("event_type", ["CORE_DESTROYED", "DESTRUCTION_PARTICIPATION"])
def test_all_core_uuids_require_confirmation_after_loss_timeout_and_restart(
    event_type, tmp_path
):
    memory, roster = scenario(seen=True)
    bot = strategy(memory)
    first = make_turn(
        objects=[
            core(position=HOME),
            *roster,
            core(900, controlled=False, position=(-954, -954)),
            core(901, controlled=False, position=(-935, -935)),
        ]
    )
    bot.decide(first)
    survey = memory.block_clearance.blocks[block_at(HOME)]
    assert set(survey.cores) == {object_id(900), object_id(901)}
    for tick in (101, 1000):
        memory.core_hunt = None  # Task cancellation must not complete clearance.
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
        bot = strategy(memory)
        turn = make_turn(tick=tick, objects=[core(position=HOME), *roster])
        bot.decide(turn)
        assert memory.core_home_position == HOME
        assert turn.plan.core_action is None
    first_kill = make_turn(
        tick=1001,
        objects=[core(position=HOME), *roster],
        events=[destruction(900, 1000, event_type)],
    )
    bot.decide(first_kill)
    assert memory.core_home_position == HOME
    assert set(memory.block_clearance.blocks[block_at(HOME)].cores) == {object_id(901)}
    last_kill = make_turn(
        tick=1002,
        objects=[core(position=HOME), *roster],
        events=[destruction(901, 1001, event_type)],
    )
    bot.decide(last_kill)
    assert memory.core_home_position != HOME
    assert last_kill.plan.core_action is not None
    assert last_kill.plan.core_action.type == "START_MOVE"


def test_moving_core_stays_pending_in_discovery_block_and_new_uuid_is_separate():
    state = BlockClearance()
    state.observe(
        make_turn(
            objects=[core(position=HOME), core(900, controlled=False, position=HOME)]
        ),
        (),
        (),
    )
    state.observe(
        make_turn(
            tick=101,
            objects=[
                core(position=HOME),
                core(900, controlled=False, position=(-910, -945)),
                core(901, controlled=False, position=(-950, -950)),
            ],
            events=[destruction(901, 100, "SHOT_HIT")],
        ),
        (),
        (),
    )
    assert state.blocks[block_at(HOME)].cores == {
        object_id(900): (-910, -945),
        object_id(901): (-950, -950),
    }
    state.observe(
        make_turn(
            tick=102, objects=[core(position=HOME)], events=[destruction(900, 101)]
        ),
        (),
        (),
    )
    assert set(state.blocks[block_at(HOME)].cores) == {object_id(901)}


@pytest.mark.parametrize("same_uuid", [False, True])
@pytest.mark.parametrize("old_save", [False, True])
def test_own_respawn_discards_old_clearance_even_in_same_block(same_uuid, old_save):
    memory, _roster = scenario(seen=True)
    memory.block_clearance.blocks[block_at(HOME)].cores[object_id(900)] = HOME
    memory.enemies[object_id(900)] = EnemySighting(
        object_id(900), "CORE", HOME, 5, 99, shield=5
    )
    if old_save:
        memory.block_clearance = BlockClearance()
    replacement = 1 if same_uuid else 2
    turn = make_turn(
        objects=[core(replacement, position=HOME)],
        events=[
            {
                "event_id": object_id(9001),
                "tick": 99,
                "event_type": "CORE_RESPAWNED",
                "target_id": object_id(replacement),
            }
        ],
    )
    strategy(memory).decide(turn)
    survey = memory.block_clearance.blocks[block_at(HOME)]
    assert not survey.cores
    assert len(survey.seen) < 1024
    assert not strategy(memory)._campaign_block_cleared(turn)


def test_old_save_restores_known_unconfirmed_cores_without_inventing_coverage():
    memory, roster = scenario()
    memory.block_clearance = BlockClearance()
    memory.enemies[object_id(900)] = EnemySighting(
        object_id(900), "CORE", (-950, -950), 5, 99, shield=5
    )
    turn = make_turn(objects=[core(position=HOME), *roster])
    strategy(memory).decide(turn)
    survey = memory.block_clearance.blocks[block_at(HOME)]
    assert object_id(900) in survey.cores
    assert len(survey.seen) < 1024


def test_rocks_and_truly_sealed_pocket_are_excluded_but_observable_gap_is_not():
    memory, roster = scenario(seen=True)
    pocket = (-958, -958)
    rocks = {
        (pocket[0] + dx, pocket[1] + dy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
    }
    gap = (-929, -929)
    survey = memory.block_clearance.blocks[block_at(HOME)]
    for cell in {*rocks, pocket, gap}:
        survey.seen.pop(cell)
    turn = make_turn(objects=[core(position=HOME), *roster], obstacles=rocks)
    bot = strategy(memory)
    bot.decide(turn)
    assert bot._campaign_block_gaps(turn) == {gap}
    assert memory.core_home_position == HOME
    survey.seen[gap] = 100
    assert bot._campaign_block_cleared(turn)
    # A seen Core inside a newly sealed pocket is still an unresolved target.
    survey.cores[object_id(900)] = pocket
    assert not bot._campaign_block_cleared(turn)


def test_production_and_harvest_continue_while_local_survey_holds_migration():
    memory, roster = scenario()
    mine = tuple(roster[0]["position"])
    turn = make_turn(
        resources=100, objects=[core(position=HOME), *roster], resource_cells=[mine]
    )
    strategy(memory).decide(turn)
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "SPAWN"
    assert turn.plan.unit_actions[turn.workers[0].id].type == "HARVEST"
    assert memory.core_home_position == HOME
    assert memory.campaign_resting


def test_arrival_counts_real_travel_vision_and_does_not_reset_at_center():
    memory, roster = scenario()
    bot = strategy(memory)
    approach = (HOME[0] - 1, HOME[1])
    first = make_turn(objects=[core(position=approach), *roster])
    bot.decide(first)
    observed = set(memory.block_clearance.blocks[block_at(HOME)].seen)
    assert observed
    second = make_turn(tick=101, objects=[core(position=HOME), *roster])
    bot.decide(second)
    assert observed <= memory.block_clearance.blocks[block_at(HOME)].seen.keys()
    assert memory.core_home_position == HOME


def test_active_forward_scout_cannot_release_hop_over_new_local_core():
    memory, roster = scenario((815, 15), 30, seen=True)
    bot = strategy(memory)
    bot.decide(make_turn(objects=[core(position=(815, 15)), *roster]))
    target = memory.campaign_scout_target
    assert target is not None
    memory.campaign_scout_seen_ticks = dict.fromkeys(cells_in(block_at(target)), 100)
    blocked = make_turn(
        tick=101,
        objects=[
            core(position=(815, 15)),
            *roster,
            core(900, controlled=False, position=(825, 25)),
        ],
    )
    bot.decide(blocked)
    assert memory.core_home_position == (815, 15)
    assert memory.campaign_scout_target == target
    assert blocked.plan.core_action is None


def test_local_patrol_completes_real_sweep_without_freezing_economy(tmp_path):
    memory, roster = scenario()
    bot = strategy(memory)
    observed: set[tuple[int, int]] = set()
    for tick in range(100, 250):
        if tick == 115:
            path = tmp_path / "memory.json"
            memory.save(path)
            memory = WorldMemory.load(path)
            bot = strategy(memory)
        turn = make_turn(tick=tick, objects=[core(position=HOME), *roster])
        for position, radius in memory._vision_observers(turn):
            observed.update(
                visible_cells(position, radius, set()) & cells_in(block_at(HOME))
            )
        bot.decide(turn)
        if memory.core_home_position != HOME:
            assert observed == cells_in(block_at(HOME))
            assert turn.plan.core_action is not None
            assert turn.plan.core_action.type == "START_MOVE"
            snapshot = _identity_snapshot(memory)["campaign"]["block_clearance"]
            assert snapshot["observed_cells"] == 1024
            assert not snapshot["pending_cores"]
            return
        assert observed != cells_in(block_at(HOME)) or not bot._campaign_rest_complete(
            turn
        )
        positions = {
            str(key): list(value) for key, value in settled_positions(turn).items()
        }
        for actor in roster:
            actor["position"] = positions[actor["id"]]
    pytest.fail("patrol never completed the empty open block")


def test_clearance_state_only_keeps_active_journey_blocks():
    state = BlockClearance()
    for index in range(12):
        home = (index * 32 + 15, 15)
        target = (home[0] + 32, home[1])
        state.observe(
            make_turn(tick=100 + index, objects=[core(position=home)]),
            (target,),
            (home,),
        )
        assert set(state.blocks) == {block_at(home), block_at(target)}
    state.observe(make_turn(tick=112), (), ())
    assert not state.blocks


def test_adjacent_enemy_does_not_block_current_clearance():
    memory, roster = scenario(seen=True)
    turn = make_turn(
        objects=[
            core(position=HOME),
            *roster,
            core(900, controlled=False, position=(-928, -945)),
        ]
    )
    strategy(memory).decide(turn)
    assert memory.core_home_position != HOME
    assert turn.plan.core_action is not None
    assert turn.plan.core_action.type == "START_MOVE"


def test_next_center_does_not_inherit_previous_blocks_completed_survey():
    memory, roster = scenario(seen=True)
    bot = strategy(memory)
    bot.decide(make_turn(objects=[core(position=HOME), *roster]))
    target = memory.core_home_position
    assert target is not None and block_at(target) != block_at(HOME)
    for actor in roster:
        actor["position"] = [
            actor["position"][0] + target[0] - HOME[0],
            actor["position"][1] + target[1] - HOME[1],
        ]
    arrived = make_turn(tick=104, objects=[core(position=target), *roster])
    bot.decide(arrived)
    assert memory.core_home_position == target
    assert memory.campaign_scout_target is None
    assert memory.campaign_resting
    assert not bot._campaign_block_cleared(arrived)
    assert len(memory.block_clearance.blocks[block_at(target)].seen) < 1024
