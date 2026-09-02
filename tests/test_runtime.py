"""Runtime, telemetry, and CLI behavior without a live credential."""

from __future__ import annotations

import json

import pytest
from arena_hero import APIError, ArenaHeroError, Turn

from arena_hero_bot import cli, runtime
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.models import DecisionReport
from arena_hero_bot.runtime import RuntimeConfig
from arena_hero_bot.strategy import AggressiveStrategy
from arena_hero_bot.telemetry import JsonlTelemetry

from .factories import core, make_turn, unit


def test_jsonl_telemetry_appends_records(tmp_path) -> None:
    path = tmp_path / "nested" / "turns.jsonl"
    telemetry = JsonlTelemetry(path)
    telemetry.append({"tick": 1})
    telemetry.append({"tick": 2})
    assert [json.loads(line)["tick"] for line in path.read_text().splitlines()] == [
        1,
        2,
    ]


def test_submit_turn_modes_and_errors() -> None:
    accepted_turn = make_turn(objects=[core()])
    assert runtime._submit_turn(accepted_turn, observe_only=True) == {
        "status": "observed"
    }
    assert runtime._submit_turn(accepted_turn, observe_only=False)["status"] == (
        "accepted"
    )

    rejected = Turn(
        tick=accepted_turn.tick,
        state=accepted_turn.state,
        submitter=lambda _plan, _key: _raise(
            APIError(status_code=409, error="COMMAND_WINDOW_CLOSED")
        ),
    )
    rejection = runtime._submit_turn(rejected, observe_only=False)
    assert rejection["status"] == "rejected"
    assert rejection["error"] == "COMMAND_WINDOW_CLOSED"

    failed = Turn(
        tick=accepted_turn.tick,
        state=accepted_turn.state,
        submitter=lambda _plan, _key: _raise(ArenaHeroError("network")),
    )
    sdk_error = runtime._submit_turn(failed, observe_only=False)
    assert sdk_error["status"] == "sdk_error"
    assert sdk_error["error_type"] == "ArenaHeroError"


def test_plan_failure_falls_back_to_empty_plan() -> None:
    turn = make_turn(objects=[core(), unit(2, "WORKER", position=(1, 0))])

    class BrokenStrategy:
        def decide(self, turn):
            raise RuntimeError("broken strategy")

    report, error = runtime._plan_turn(BrokenStrategy(), turn)
    assert report == DecisionReport(tick=turn.tick)
    assert error == "RuntimeError: broken strategy"
    assert turn.plan.unit_actions == {}


def test_turn_record_contains_state_plan_and_no_credential() -> None:
    turn = make_turn(objects=[core(), unit(2, "WORKER", position=(1, 0))])
    report = AggressiveStrategy(WorldMemory()).decide(turn)
    record = runtime._turn_record(
        turn=turn,
        report=report,
        planning_seconds=0.001,
        planning_error=None,
        submission={"status": "observed"},
        observe_only=True,
    )
    assert record["tick"] == turn.tick
    assert record["state"]["status"] == "ACTIVE"
    assert record["plan"]["tick"] == turn.tick
    assert "api_key" not in json.dumps(record).lower()


def test_runtime_tick_log_includes_resource_progress(
    tmp_path, monkeypatch, caplog
) -> None:
    turn = make_turn(
        tick=100,
        resources=7,
        objects=[core(), unit(2, "WORKER", position=(1, 0))],
    )

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def turns(self):
            yield turn

    monkeypatch.setattr(runtime, "ArenaHeroClient", FakeClient)
    with caplog.at_level("INFO", logger="arena_hero_bot.runtime"):
        runtime.run_bot(
            RuntimeConfig(
                api_key="test-key",
                data_dir=tmp_path,
                observe_only=True,
                max_turns=1,
            )
        )

    assert "resources=7/10 population=1" in caplog.text


def test_run_bot_persists_memory_and_turn_telemetry(tmp_path, monkeypatch) -> None:
    turns = [
        make_turn(
            tick=100,
            objects=[core(), unit(2, "WORKER", position=(1, 0))],
            obstacles=[(3, 3)],
        ),
        make_turn(tick=101, objects=[core(), unit(2, "WORKER", position=(2, 0))]),
    ]

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def turns(self):
            yield from turns

    monkeypatch.setattr(runtime, "ArenaHeroClient", FakeClient)
    count = runtime.run_bot(
        RuntimeConfig(
            api_key="test-key",
            data_dir=tmp_path,
            observe_only=True,
            max_turns=2,
        )
    )
    assert count == 2
    assert (tmp_path / "memory.json").exists()
    records = [
        json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()
    ]
    assert [record["tick"] for record in records] == [100, 101]
    assert all(record["submission"]["status"] == "observed" for record in records)
    assert "test-key" not in (tmp_path / "turns.jsonl").read_text()


def test_run_bot_skips_replayed_tick_after_restart(tmp_path, monkeypatch) -> None:
    WorldMemory(last_tick=100).save(tmp_path / "memory.json")
    turns = [
        make_turn(tick=100, objects=[core(), unit(2, "WORKER", position=(1, 0))]),
        make_turn(tick=101, objects=[core(), unit(2, "WORKER", position=(2, 0))]),
    ]

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def turns(self):
            yield from turns

    monkeypatch.setattr(runtime, "ArenaHeroClient", FakeClient)
    count = runtime.run_bot(
        RuntimeConfig(
            api_key="test-key",
            data_dir=tmp_path,
            observe_only=True,
            max_turns=2,
        )
    )

    assert count == 2
    records = [
        json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()
    ]
    assert [record["tick"] for record in records] == [101]
    assert WorldMemory.load(tmp_path / "memory.json").last_tick == 101


def test_run_bot_reconnects_after_tick_mismatch(tmp_path, monkeypatch) -> None:
    stale_source = make_turn(tick=100, objects=[core()])
    stale_turn = Turn(
        tick=stale_source.tick,
        state=stale_source.state,
        submitter=lambda _plan, _key: _raise(
            APIError(status_code=409, error="TICK_MISMATCH")
        ),
    )
    current_turn = make_turn(tick=101, objects=[core()])
    clients = []

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def turns(self):
            if len(clients) == 1:
                yield stale_turn
            else:
                yield current_turn

    monkeypatch.setattr(runtime, "ArenaHeroClient", FakeClient)
    count = runtime.run_bot(
        RuntimeConfig(
            api_key="test-key",
            data_dir=tmp_path,
            max_turns=2,
        )
    )

    assert count == 2
    assert len(clients) == 2
    records = [
        json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()
    ]
    assert [record["tick"] for record in records] == [100, 101]
    assert [record["submission"]["status"] for record in records] == [
        "rejected",
        "accepted",
    ]


def test_cli_requires_key_and_builds_runtime(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ARENA_HERO_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ARENA_HERO_API_KEY is required"):
        cli.main(["--env-file", str(tmp_path / "missing.env")])

    captured = {}

    def fake_run(config, *, strategy_config):
        captured["runtime"] = config
        captured["strategy"] = strategy_config
        return 0

    monkeypatch.setenv("ARENA_HERO_API_KEY", "secret-value")
    monkeypatch.setattr(cli, "run_bot", fake_run)
    cli.main(
        [
            "--observe-only",
            "--max-turns",
            "3",
            "--target-workers",
            "1",
            "--max-population",
            "9",
            "--resource-target",
            "40",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert captured["runtime"].api_key == "secret-value"
    assert captured["runtime"].observe_only is True
    assert captured["strategy"].target_workers == 1
    assert captured["strategy"].max_population == 9
    assert captured["strategy"].resource_target == 40

    cli.main(
        [
            "--observe-only",
            "--no-max-population",
            "--max-turns",
            "1",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert captured["strategy"].max_population is None
    assert captured["strategy"].resource_target == 0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--max-turns", "0"], "must be greater than zero"),
        (["--target-workers", "-1"], "must be zero or greater"),
        (["--resource-target", "-1"], "must be zero or greater"),
    ],
)
def test_cli_rejects_invalid_numbers(arguments, message, capsys) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments)
    assert message in capsys.readouterr().err


def test_cli_rejects_unreachable_resource_target(monkeypatch, capsys) -> None:
    monkeypatch.setenv("ARENA_HERO_API_KEY", "secret-value")
    with pytest.raises(SystemExit):
        cli.main(["--max-population", "18", "--resource-target", "95"])
    assert "cannot exceed the maximum Core capacity of 90" in capsys.readouterr().err


def _raise(error: Exception):
    raise error
