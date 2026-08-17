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

For a long-running local instance that restarts after an unexpected process exit:

```bash
./scripts/run-live.sh
```

The supervisor uses the live defaults `--target-workers 2 --max-population 20`
when no arguments are supplied. Pass normal CLI arguments to override them.

Runtime memory and JSONL telemetry are written under `var/` by default.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```
