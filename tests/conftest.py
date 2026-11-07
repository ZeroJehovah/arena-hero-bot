"""Keep all tests isolated from the live client and default runtime directory."""

import pytest

from arena_hero_bot import cli, runtime


@pytest.fixture(autouse=True)
def forbid_live_runtime(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("tests must replace the runtime/client before invoking it")

    monkeypatch.setenv("ARENA_HERO_API_KEY", "test-key")
    # Block the CLI before run_bot can load or prune a real data directory.
    # Runtime tests replace the client explicitly and use tmp_path storage.
    monkeypatch.setattr(cli, "run_bot", forbidden)
    monkeypatch.setattr(runtime, "ArenaHeroClient", forbidden)
