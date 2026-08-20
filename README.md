# arena-hero-bot

An aggressive, telemetry-first tactic bot for the persistent Arena Hero world.

The first strategy keeps a small Worker economy, produces combat units early,
actively pursues visible or recently observed enemies, and records every Turn,
decision, plan, and resolution event for later analysis.

## Requirements

- Python 3.11+
- An Arena Hero API key for live observation or play
- [`uv`](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

Put the API key in an untracked `.env` file:

```dotenv
ARENA_HERO_API_KEY=replace-with-your-key
```

Observe and collect data without submitting plans:

```bash
uv run arena-hero-bot --observe-only
```

Run the tactic:

```bash
uv run arena-hero-bot
```

Run safety-first growth without a fixed population cap or Core resource target:

```bash
uv run arena-hero-bot --target-workers 12 --no-max-population
```

The unbounded live posture keeps a dynamic emergency reserve: at least the
configured base reserve and enough for two of the most expensive combat units
at the current population. It establishes an initial combat guard and only
expands when the dynamic unit price can be paid without spending that reserve.
Workers patrol the local resource grid; once the combat roster is large enough,
a stable minority of Rangers/Vanguards performs a bounded outward patrol and
pursues recently seen enemy units or Cores. That roaming squad is recalled when
the Core is damaged or resources approach the reserve.

For a long-running local instance that restarts after an unexpected process exit:

```bash
./scripts/run-live.sh
```

The supervisor uses the live defaults `--target-workers 12 --no-max-population`
when no arguments are supplied. Pass normal CLI arguments to override them.

For a persistent systemd deployment, install the supplied service after creating
the virtual environment and `.env` file:

```bash
sudo install -m 0644 deploy/arena-hero-bot.service \
  /etc/systemd/system/arena-hero-bot.service
sudo systemctl daemon-reload
sudo systemctl enable --now arena-hero-bot.service
sudo systemctl status arena-hero-bot.service
journalctl -u arena-hero-bot.service -f
```

The supplied unit targets the canonical deployment path
`/opt/dev/projects/personal/arena-hero-bot/src` and runs as the `ubuntu` user.

Runtime memory and JSONL telemetry are written under `var/` by default.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```
