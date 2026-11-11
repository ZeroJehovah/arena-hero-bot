"""Movement contention, nearby mining, and actual hostile fighter pressure."""

from uuid import UUID

import pytest

from arena_hero_bot.memory import UnitGoal, WorldMemory
from arena_hero_bot.strategy import (
    CORE_LAYOUT_VERSION,
    RESOURCE_CLAIM_PURPOSE,
    AggressiveStrategy,
    StrategyConfig,
)

from .factories import core, make_turn, object_id, unit
from .test_core_traffic import settled_positions


@pytest.mark.parametrize("restart", [False, True])
def test_two_workers_finish_nearby_mines_after_a_noncombat_move_contest(
    tmp_path, restart
):
    contested = (21, 26)
    mines = {(19, 26), (22, 25)}
    rocks = {
        (15, 23),
        (16, 27),
        (17, 22),
        (18, 20),
        (18, 21),
        (20, 21),
        (20, 23),
        (22, 18),
        (22, 20),
    }
    memory = WorldMemory(
        core_home_position=(16, 16),
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_last_core_id=object_id(1),
        campaign_phase=4,
        pending_move_targets={object_id(2): contested},
    )
    memory.observe(
        make_turn(
            tick=90,
            objects=[
                core(position=(16, 16)),
                unit(2, "WORKER", position=(20, 26)),
                unit(900, "WORKER", controlled=False, position=contested),
            ],
            events=[
                {
                    "event_id": object_id(1000),
                    "tick": 89,
                    "event_type": "UNIT_MOVE_FAILED",
                    "reason_code": "MOVE_CONTESTED",
                    "actor_id": object_id(2),
                    "position": [20, 26],
                }
            ],
        )
    )
    assert memory.contested_positions == {contested: 90}
    config = StrategyConfig(
        target_workers=16, max_population=None, expedition_mode=True, campaign_mode=True
    )
    strategy = AggressiveStrategy(memory, config)
    workers = [
        unit(2, "WORKER", position=(19, 22)),
        unit(3, "WORKER", position=(18, 22)),
    ]
    harvested = set()
    for tick in range(100, 116):
        if restart and tick == 102:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), config)
        turn = make_turn(
            tick=tick,
            objects=[core(position=(16, 16)), *workers],
            resource_cells=mines - harvested,
            obstacles=rocks,
        )
        report = strategy.decide(turn)
        assert not any(
            "retreat" in d.reason for d in report.decisions if d.actor_kind == "WORKER"
        )
        positions = settled_positions(turn)
        assert contested not in positions.values()
        assert contested in strategy.memory.contested_positions
        for worker in workers:
            uid = UUID(worker["id"])
            action = turn.plan.unit_actions.get(uid)
            if action is not None and action.type == "HARVEST":
                position = tuple(worker["position"])
                assert position in mines - harvested
                harvested.add(position)
                worker["cargo"] = 1
            worker["position"] = list(positions[uid])
        if harvested == mines:
            break
    assert harvested == mines


@pytest.mark.parametrize("chunk", [False, True])
def test_contested_mine_does_not_steal_a_reachable_claim(chunk):
    contested, mine, position = (5, 6), (6, 7), (5, 5)
    memory = WorldMemory(
        core_home_position=(16, 16) if chunk else None,
        core_layout_version=CORE_LAYOUT_VERSION,
        contested_positions={contested: 99},
        resource_cells={contested: 99, mine: 99},
    )
    memory.set_goal(object_id(2), UnitGoal(contested, 99, RESOURCE_CLAIM_PURPOSE))
    strategy = AggressiveStrategy(memory, StrategyConfig(expedition_mode=chunk))
    for tick in range(100, 104):
        turn = make_turn(
            tick=tick,
            objects=[core(position=(16, 16)), unit(2, "WORKER", position=position)],
            resource_cells=[contested, mine],
        )
        report = strategy.decide(turn)
        decision = next(d for d in report.decisions if d.actor_kind == "WORKER")
        assert decision.target == mine
        if tick == 103:
            assert turn.plan.unit_actions[turn.workers[0].id].type == "HARVEST"
        else:
            position = settled_positions(turn)[turn.workers[0].id]
            assert position != contested


@pytest.mark.parametrize("kind", ["RANGER", "VANGUARD"])
@pytest.mark.parametrize("visible", [False, True])
def test_contested_cell_does_not_disable_retreat_from_real_fighters(kind, visible):
    contested = (21, 26)
    enemy = unit(900, kind, controlled=False, position=contested)
    memory = WorldMemory(contested_positions={contested: 99})
    memory.observe(make_turn(tick=99, objects=[core(position=(16, 16)), enemy]))
    turn = make_turn(
        tick=100,
        objects=[
            core(position=(16, 16)),
            unit(2, "WORKER", position=(19, 22)),
            *([enemy] if visible else []),
        ],
        resource_cells=[(19, 26)],
    )
    report = AggressiveStrategy(memory).decide(turn)
    decision = next(d for d in report.decisions if d.actor_kind == "WORKER")
    assert decision.reason == "retreat from visible enemy pressure"
    assert settled_positions(turn)[turn.workers[0].id] != contested
