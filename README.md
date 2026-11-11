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
sequence: `8W -> 16W -> 8R -> 20R -> *R`.
Ranger production does not choose a role. After births or losses, the local
Ranger roster splits evenly between defense and patrol, assigning an odd extra
Ranger to defense. Recruits fill vacancies first; only the minimum necessary
survivors change roles, and existing patrol team identities remain stable.
While a visible hostile combat unit is inside the Core assault ring, the next
production slot temporarily becomes a Ranger; the normal sequence resumes as
soon as that local threat is gone.
It spends every affordable resource below the `16W+20R` max formation, then
builds further Rangers only at a full bank. Losses below that base roster
restore immediate affordable replenishment, even if the durable phase is max.
Legacy expedition squads are recalled, fight normally while returning, and join
the local roster within eight cells of the Core; campaign mode creates no new
expeditions and no Vanguards.

`--campaign-mode` enables the required formation management and uncapped growth
on its own. It rejects `--max-population` and a positive `--resource-target`,
which would prevent the campaign from following its formation sequence.
When a hostile fighter enters the Core assault ring, even a single contact
temporarily selects a Ranger. If that Ranger is unaffordable, delivery remains
available until it can be funded. Once the contact leaves, normal production
resumes. Expedition telemetry includes the persisted recall flag.

Fixed defenders do not pursue enemies. During a Core assault they shoot at an
enemy's current cell when a real ray exists; without one, up to two defenders
may rotate onto a safe adjacent firing lane before speculative cover fire is
used. Overlapping defenders prefer distinct firing cells. If an aim cell and all
of its neighbours are visible and contain no enemy, the Ranger holds fire instead
of rotating shots through known-empty ground. Repositioning requires a
threatened ally or Core, stays within the defensive layers, and avoids new
firing sources. A necessary withdrawal is planned before rear-rank shots and
prefers an equally safe cell that preserves supporting fire.

In a one- or two-fighter campaign contact, a stationary wounded fighter
with one or two HP may pull
one or two healthy non-garrison Rangers onto a real firing line when the whole
move stays within three cells of its assigned post and completes in at most
three steps. The task is fixed for at most eight Ticks. The first shooter waits
at the lane until the pair can volley only if it can survive the worst current
fire until that volley; otherwise any existing shot proceeds normally. Target
movement, Core pressure, extra fighters or timeout cancels the task without
following the last seen cell. Stale remembered
contacts keep all-hands defence active only for a sixteen-Tick flicker window;
three visible or still-current fighters are still required to trigger it.

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

## Traffic around a stationary Core

Local Worker movements are collected before capacity is resolved. A cell can
hold two entities, including the Core, so cargo may share a guard's cell or
exchange places with departing traffic. Overlap alone never forces cargo to
backtrack. Complete swaps, rotations and outward chains are committed together;
an incomplete chain leaves no partial moves behind.

Production reserves the Core cell: an outward chain frees it, then spawning
follows once the departure is observed. Chains need spare capacity at their end;
swaps cannot create space for an additional entity. Real attacks, harvesting,
deposits, healing, terrain and enemy danger retain priority. The local search is
bounded to eight cells around the Core, twelve dependent moves and 1,024 search
attempts per Tick. Migration lanes still keep one slot available for the moving
Core.

Idle Worker coverage and routine Ranger patrols keep their entire routes inside
the assigned resource area. They prefer positions and paths three cells inside
the boundary; rocks may require a detour through the edge strip. Workers and
forward scouts approach the edge only to observe cells hidden from the interior.
A unit already outside returns safely. Migration retains its authorized source
and destination blocks, while resource service and combat retain their own rules.

## Long Core migration

The bot can relocate the Core between resource chunks without stopping the
local economy. Long campaigns publish one current-or-cardinal-neighbor chunk
center at a time; after reaching that roughly 32-cell hop they publish the next
one. The Core moves for four Ticks; an arrival service Tick may spawn a unit,
heal before depositing, or accept a delivery, keeping the hop to five Ticks. A
new hop, including its forward scout, waits until every live Worker and Ranger
is in the Core's current 32x32 block and affordable formation slots are built.
Nearby cargo must also be delivered and the current quota gate satisfied.
Worker mining and idle patrols stay near the actual Core: during travel their
ordinary area is the source/destination chunks intersected with a moving 32x32
window. At rest the current block remains available. Rangers cover the source
inside that window plus a four-cell margin, and may use the entire destination,
including its preparatory scout. Patrol paths remain within the permitted area.
An empty Worker may temporarily mine a live visible resource outside it when
the resource is within six Manhattan steps, a safe route takes at most eight
steps, and both stay inside the Core window plus four cells. Mining may also
take bounded terrain detours inside that margin. An existing live near-mine
claim may finish its short rock detour within the same eight-step route budget.
Deliveries, healing, necessary combat, and the bounded enemy Core hunt retain
their own rules.
With sixteen Workers, eight Rangers permit quota 2 travel and twenty permit
travel through quota 3 to the quota 4 boundary. Thirty Rangers permit
entry into quota 4, stopping at the first block center. Forty permit travel
through quota 4 and entry into quota 5; fifty permit travel through quota 5 and
entry into quota 6, and so on. There is no fixed quota-8 destination. Casualties
immediately revoke unauthorized hops and scouts. After a major defensive battle
and the eight-Tick calm window, an insufficient roster cancels the old journey
and refits at a reachable center in the Core's current block, including the old
destination if the Core has already crossed into it.
Before every hop into or within quota 4+,
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
