"""Persistent world-memory tests."""

import json

import pytest

from arena_hero_bot.memory import UnitGoal, WorldMemory

from .factories import core, make_turn, unit


def test_observations_round_trip_without_hidden_data(tmp_path) -> None:
    turn = make_turn(
        tick=50,
        objects=[
            core(),
            unit(2, "WORKER", position=(1, 0)),
            core(
                3,
                controlled=False,
                owner_username="rival_one",
                position=(4, 0),
                hp=3,
                shield=1,
            ),
            unit(4, "RANGER", controlled=False, position=(3, 0), hp=1),
        ],
        obstacles=[(2, 2)],
    )
    memory = WorldMemory()
    memory.observe(turn)
    memory.set_goal("worker", UnitGoal((9, 9), 50, "explore"))
    path = tmp_path / "memory.json"
    memory.save(path)

    loaded = WorldMemory.load(path)
    assert loaded.last_tick == 50
    assert loaded.obstacles == {(2, 2)}
    assert loaded.enemies[next(iter(loaded.enemies))].owner_username in {
        None,
        "rival_one",
    }
    assert loaded.goal_for("worker") == UnitGoal((9, 9), 50, "explore")
    assert loaded.recent_positions(str(turn.workers[0].id)) == ()
    assert turn.core is not None
    assert loaded.recent_positions(str(turn.core.id)) == ()


def test_enemy_ttl_sorting_and_goal_lifecycle() -> None:
    memory = WorldMemory()
    memory.observe(
        make_turn(
            tick=20,
            objects=[
                core(),
                core(5, controlled=False, owner_username="rival", position=(5, 0)),
                unit(6, "WORKER", controlled=False, position=(4, 0)),
            ],
        )
    )
    recent = memory.recent_enemies(25, ttl=10)
    assert [enemy.kind for enemy in recent] == ["CORE", "UNIT"]
    assert memory.recent_enemies(31, ttl=10) == ()

    memory.set_goal("unit", UnitGoal((1, 2), 20, "explore"))
    assert memory.goal_for("unit") is not None
    memory.clear_goal("unit")
    assert memory.goal_for("unit") is None


def test_missing_file_and_unknown_schema(tmp_path) -> None:
    assert WorldMemory.load(tmp_path / "missing.json") == WorldMemory()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported memory schema"):
        WorldMemory.load(path)


def test_old_friendly_histories_are_pruned() -> None:
    memory = WorldMemory()
    first = make_turn(objects=[core(), unit(2, "WORKER", position=(1, 0))])
    memory.observe(first)
    worker_id = str(first.workers[0].id)
    memory.set_goal(worker_id, UnitGoal((5, 5), 100, "explore"))

    memory.observe(make_turn(tick=101, objects=[core()]))
    assert worker_id not in memory.position_history
    assert memory.goal_for(worker_id) is None
