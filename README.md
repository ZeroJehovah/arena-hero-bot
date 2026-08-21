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

The unbounded live posture uses capacity-based stockpile tiers. With Core
capacity below 50 it spends healthy surplus resources immediately to expand;
it retains resources needed to repair a damaged Core and enters the normal
emergency reserve during active combat. At capacities 50–94 and 95–100 it
preserves 50 and 95 resources, while capacities above 100 preserve 100
resources before paying the next dynamic unit price. It establishes an
initial combat guard, while a stable minority of Rangers/Vanguards performs a
bounded outward patrol once the combat roster is large enough. That roaming
squad is suspended when the Core is damaged or the roster is too small; Core
stockpile level controls production and does not turn the patrol off. The
remaining combat units hold a single obstacle-aware
Manhattan perimeter: its slots (including traversable upper and lower
cardinal anchors) stay fixed across Turns, and a guard queues `WAIT` after
reaching its slot until an enemy or a roster/Core change requires redeployment.

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
