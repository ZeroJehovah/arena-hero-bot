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
initial combat guard and, once four Workers are online, adds a second combat
unit (preferring a Ranger) before resuming unrestricted Worker growth. A
stable minority of Rangers/Vanguards performs a
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

The supervisor uses the live defaults `--target-workers 16 --no-max-population
--expedition-mode --campaign-mode` when no arguments are supplied. Pass normal CLI
arguments to override them. Campaign mode follows one cumulative Ranger-only
sequence: `8W -> 16W -> 8R(defense) -> 20R(defense) -> 8R(patrol) -> *R(patrol)`.
While a visible hostile combat unit is inside the Core assault ring, the next
production slot temporarily becomes a Ranger; the normal sequence resumes as
soon as that local threat is gone.
It spends every affordable resource below the max formation and keeps a stockpile
only after that formation is complete. Every later Ranger remains a patrol unit.
Legacy expedition squads are recalled, fight normally while returning, and join
the patrol roster within eight cells of the Core; campaign mode creates no new
expeditions and no Vanguards.

Fixed defenders do not pursue enemies. During a Core assault they shoot at an
enemy's current cell when a real ray exists; without one, up to two defenders
may rotate onto a safe adjacent firing lane before speculative cover fire is
used. Overlapping defenders prefer distinct firing cells. If an aim cell and all
of its neighbours are visible and contain no enemy, the Ranger holds fire instead
of rotating shots through known-empty ground. Repositioning requires a
threatened ally or Core, stays within the defensive layers, and avoids new
firing sources. A necessary withdrawal is planned before rear-rank shots and
prefers an equally safe cell that preserves supporting fire.

Patrol Rangers may pursue a visible fighter only with a healthy local force
advantage and a supporting shooter that can actually cover the contact. Each
engagement is limited to four cells from its original contact position and
six Ticks; the budget survives restarts and continuous sightings do not renew
it. A threatened Core takes priority and recalls forward patrols. Deliberate
empty-cell coverage may produce misses; shot counts alone are not a measure of
combat success.

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

## Long Core migration

The bot can relocate the Core between resource chunks without stopping the
local economy. Long campaigns publish one current-or-cardinal-neighbor chunk
center at a time; after reaching that roughly 32-cell hop they publish the next
one. The Core moves for four Ticks, reserves one Tick for a deposit or heal, and
harvests across the source and destination chunks while defensive posts and
patrols follow it. Formation tiers cap movement at quota 2, quota 3, and then
the max formation's quota-8 goal. Before every hop into or within quota 4+,
patrols must observe the destination block, confirm it is clear, and allow an
eight-Tick enemy-clear settling window. The quota goal, current hop, scout
coverage, and pause schedule are durable across restarts.

Plan the cleanest nearby block at the next natural-resource tier:

```bash
uv run arena-core-migration --data-dir var plan
```

Publish an atomic hot request to the running bot:

```bash
uv run arena-core-migration --data-dir var request-next-tier
```

An explicit chunk-center cell can be supplied with `request X Y`; it must stay
in the current chunk or a cardinal-neighbor chunk. `status` shows the persisted
home and pending request. Requests are validated against chunk-center geometry
and remembered obstacles before the running bot picks them up on its next Turn.
`CORE_RESPAWNED`, Core UUID changes, non-cardinal position jumps, and per-Turn
block adjacency checks prevent a new Core from inheriting an old destination.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```
