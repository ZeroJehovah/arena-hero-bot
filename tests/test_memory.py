"""Persistent world-memory tests."""

import json

import pytest

from arena_hero_bot.memory import UnitGoal, WorldMemory

from .factories import core, make_turn, object_id, unit


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


def test_enemy_track_predicts_one_cardinal_step() -> None:
    memory = WorldMemory()
    memory.observe(
        make_turn(
            tick=20,
            objects=[core(), unit(6, "RANGER", controlled=False, position=(0, 4))],
        )
    )
    memory.observe(
        make_turn(
            tick=21,
            objects=[core(), unit(6, "RANGER", controlled=False, position=(0, 3))],
        )
    )
    memory.observe(
        make_turn(
            tick=22,
            objects=[core(), unit(6, "RANGER", controlled=False, position=(0, 2))],
        )
    )

    assert memory.predicted_enemy_position(object_id(6), 22) == (0, 1)


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


def test_memory_marks_contested_move_target_and_expires_it() -> None:
    memory = WorldMemory(pending_move_targets={object_id(2): (1, 0)})
    turn = make_turn(
        tick=100,
        objects=[core(), unit(2, "WORKER")],
        events=[
            {
                "event_id": object_id(99),
                "tick": 100,
                "event_type": "UNIT_MOVE_FAILED",
                "reason_code": "MOVE_CONTESTED",
                "actor_id": object_id(2),
                "position": [0, 0],
            }
        ],
    )

    memory.observe(turn)

    assert memory.contested_positions == {(1, 0): 100}
    assert memory.pending_move_targets == {}

    memory.observe(make_turn(tick=181, objects=[core(), unit(2, "WORKER")]))
    assert memory.contested_positions == {}


def test_destruction_event_forgets_enemy_and_motion_history() -> None:
    memory = WorldMemory()
    enemy_id = object_id(6)
    memory.observe(
        make_turn(
            tick=20,
            objects=[core(), unit(6, "WORKER", controlled=False, position=(1, 0))],
        )
    )
    memory.observe(
        make_turn(
            tick=21,
            objects=[core(), unit(6, "WORKER", controlled=False, position=(0, 1))],
        )
    )
    assert enemy_id in memory.enemies
    assert enemy_id in memory.enemy_position_history

    memory.observe(
        make_turn(
            tick=22,
            objects=[core()],
            events=[
                {
                    "event_id": object_id(99),
                    "tick": 22,
                    "event_type": "DESTRUCTION_PARTICIPATION",
                    "reason_code": "UNIT",
                    "target_id": enemy_id,
                    "position": [0, 1],
                }
            ],
        )
    )

    assert enemy_id not in memory.enemies
    assert enemy_id not in memory.enemy_position_history
    assert memory.recent_enemies(22, ttl=100) == ()


def test_enemy_drift_leads_a_track_seen_across_a_vision_gap() -> None:
    memory = WorldMemory()
    for tick, position in ((20, (0, 6)), (22, (0, 4))):
        memory.observe(
            make_turn(
                tick=tick,
                objects=[
                    core(),
                    unit(6, "WORKER", controlled=False, position=position),
                ],
            )
        )

    # The strict predictor needs three consecutive same-delta observations,
    # which a flickering intruder track never supplies.
    assert memory.predicted_enemy_position(object_id(6), 22) is None
    assert memory.enemy_drift_position(object_id(6), 22, 1) == (0, 3)
    assert memory.enemy_drift_position(object_id(6), 22, 2) == (0, 2)


def test_enemy_drift_ignores_stationary_and_stale_tracks() -> None:
    memory = WorldMemory()
    for tick in (20, 21):
        memory.observe(
            make_turn(
                tick=tick,
                objects=[
                    core(),
                    unit(6, "WORKER", controlled=False, position=(0, 4)),
                ],
            )
        )

    assert memory.enemy_drift_position(object_id(6), 21, 1) is None
    # Extrapolating from a sighting that is not from this Tick would send the
    # guard to a cell the target left several Ticks ago.
    assert memory.enemy_drift_position(object_id(6), 22, 1) is None


def test_resource_cells_survive_leaving_vision_and_round_trip(tmp_path) -> None:
    # A cell thirty out is the whole reason this memory exists: vision reaches
    # four cells, so the visible pool empties as soon as the cells beside the
    # Core are spent, and the Core could never walk back to what it had found.
    memory = WorldMemory()
    memory.observe(
        make_turn(
            tick=50,
            objects=[core(), unit(2, "WORKER", position=(1, 0))],
            resource_cells=[(30, 0), (0, 20)],
        )
    )
    assert memory.remembered_resource_cells(999) == frozenset({(30, 0), (0, 20)})

    # The Worker walked away; neither cell is visible and neither is disproven.
    # The Core counts as an observer too, so both cells are kept clear of
    # everything's reach: a resource memory that trusted absence from further
    # out would rest cells that are merely unobserved.
    memory.observe(
        make_turn(tick=51, objects=[core(), unit(2, "WORKER", position=(9, 9))])
    )
    assert memory.remembered_resource_cells(999) == frozenset({(30, 0), (0, 20)})

    path = tmp_path / "memory.json"
    memory.save(path)
    assert WorldMemory.load(path).remembered_resource_cells(999) == frozenset(
        {(30, 0), (0, 20)}
    )


def test_an_emptied_cell_rests_instead_of_being_forgotten() -> None:
    from arena_hero_bot.memory import RESOURCE_ABSENCE_COOLDOWN

    memory = WorldMemory()
    memory.observe(make_turn(tick=50, objects=[core()], resource_cells=[(20, 0)]))

    # Two cells away is already too far to believe an absence: in the recording
    # a cell reported empty from one cell away held a resource again 5.5% of the
    # time, so trusting a wider radius would rest live cells.
    memory.observe(
        make_turn(tick=51, objects=[core(), unit(2, "WORKER", position=(22, 0))])
    )
    assert memory.remembered_resource_cells(51) == frozenset({(20, 0)})

    # Adjacent and empty.  The site is rested, not deleted - an empty cell is
    # not a dead cell, and roughly a third of them regrow after 500 Ticks.
    memory.observe(
        make_turn(tick=52, objects=[core(), unit(2, "WORKER", position=(21, 0))])
    )
    assert memory.resource_cells == {(20, 0): 50}
    assert memory.remembered_resource_cells(52) == frozenset()
    assert (
        memory.remembered_resource_cells(52 + RESOURCE_ABSENCE_COOLDOWN) == frozenset()
    )
    assert memory.remembered_resource_cells(
        53 + RESOURCE_ABSENCE_COOLDOWN
    ) == frozenset({(20, 0)})


def test_seeing_a_resource_again_clears_its_rest_immediately() -> None:
    memory = WorldMemory()
    memory.observe(make_turn(tick=50, objects=[core()], resource_cells=[(20, 0)]))
    memory.observe(
        make_turn(tick=51, objects=[core(), unit(2, "WORKER", position=(21, 0))])
    )
    assert memory.remembered_resource_cells(51) == frozenset()

    # Reporting flickers, so a cell can vanish for a Tick and come straight
    # back.  A live sighting is authoritative and outranks the rest period.
    memory.observe(
        make_turn(
            tick=52,
            objects=[core(), unit(2, "WORKER", position=(21, 0))],
            resource_cells=[(20, 0)],
        )
    )
    assert memory.remembered_resource_cells(52) == frozenset({(20, 0)})


def test_stale_resource_cells_expire_and_memory_stays_bounded() -> None:
    from arena_hero_bot.memory import RESOURCE_MEMORY_LIMIT, RESOURCE_MEMORY_TTL

    memory = WorldMemory()
    memory.observe(make_turn(tick=10, objects=[core()], resource_cells=[(40, 0)]))
    memory.observe(make_turn(tick=10 + RESOURCE_MEMORY_TTL, objects=[core()]))
    assert memory.remembered_resource_cells(999) == frozenset({(40, 0)})
    memory.observe(make_turn(tick=11 + RESOURCE_MEMORY_TTL, objects=[core()]))
    assert memory.remembered_resource_cells(999) == frozenset()

    crowd = [(x, 500) for x in range(RESOURCE_MEMORY_LIMIT + 40)]
    memory.observe(make_turn(tick=9000, objects=[core()], resource_cells=crowd))
    assert len(memory.remembered_resource_cells(999)) == RESOURCE_MEMORY_LIMIT
