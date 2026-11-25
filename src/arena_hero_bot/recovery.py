"""Bounded campaign attrition evidence and one-hop recovery state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arena_hero import Position, Turn, UnitType, UnitView, unit_cost

from .geometry import manhattan

WINDOW = 256
MIN_OBSERVATIONS = 128
MIN_RANGER_LOSSES = 4
RETRY_TICKS = 16
MAX_ROUTE = 64


@dataclass(slots=True)
class Recovery:
    core_id: str | None = None
    last_tick: int = -1
    roster: dict[str, str] = field(default_factory=dict)
    # Tick, delivered resources, estimated replacement cost, Ranger deaths.
    samples: list[tuple[int, int, int, int]] = field(default_factory=list)
    danger: Position | None = None
    danger_tick: int = -1
    holding: bool = False
    target: Position | None = None
    route: tuple[Position, ...] = ()
    retry_after: int = -1
    reserve: int = 20
    underfunded: bool = False

    def observe(self, turn: Turn) -> None:
        core_id = str(turn.core.id) if turn.core else None
        respawn = any(
            event.event_type == "CORE_RESPAWNED" and str(event.target_id) == core_id
            for event in turn.events
        )
        if core_id != self.core_id or respawn:
            self.core_id = core_id
            self.last_tick = -1
            self.roster.clear()
            self.samples.clear()
            self.danger = None
            self.danger_tick = -1
            self.holding = False
            self.target = None
            self.route = ()
            self.retry_after = -1
            self.reserve = 20
            self.underfunded = False
        core = turn.core
        if core is None or turn.tick <= self.last_tick:
            return
        income = cost = losses = 0
        if turn.tick != self.last_tick + 1:
            self.samples.clear()
        else:
            destroyed: set[str] = set()
            seen: set[str] = set()
            for event in turn.events:
                if event.tick != self.last_tick or str(event.event_id) in seen:
                    continue
                seen.add(str(event.event_id))
                values = event.values or {}
                if (
                    event.event_type == "DEPOSIT_SUCCEEDED"
                    and str(event.actor_id) in self.roster
                ):
                    income += max(0, int(values.get("amount", 0)))
                key = str(event.target_id)
                if key not in self.roster or key in destroyed:
                    continue
                if event.event_type == "UNIT_DESTROYED" or (
                    event.event_type == "UNIT_DAMAGED" and values.get("hp") == 0
                ):
                    destroyed.add(key)
                    kind = UnitType(self.roster[key])
                    cost += unit_cost(kind, len(self.roster))
                    losses += kind is UnitType.RANGER
        self.samples.append((turn.tick, income, cost, losses))
        self.samples = self.samples[-WINDOW:]
        self.roster = {str(unit.id): unit.view.unit_type.value for unit in turn.units}
        self.last_tick = turn.tick
        enemies = [
            enemy
            for enemy in turn.visible_enemies
            if isinstance(enemy, UnitView)
            and enemy.unit_type in {UnitType.RANGER, UnitType.VANGUARD}
        ]
        if enemies:
            self.danger = min(
                enemies,
                key=lambda enemy: (
                    manhattan(core.position, enemy.position),
                    enemy.id,
                ),
            ).position
            self.danger_tick = turn.tick

    def deficit(self, resources: int, rangers: int, target: int) -> bool:
        return (
            len(self.samples) >= MIN_OBSERVATIONS
            and rangers * 4 <= target * 3
            and sum(row[3] for row in self.samples) >= MIN_RANGER_LOSSES
            and sum(row[2] - row[1] for row in self.samples) > resources
        )

    def summary(self) -> dict[str, Any]:
        return {
            "observations": len(self.samples),
            "deposits": sum(row[1] for row in self.samples),
            "replacement_cost": sum(row[2] for row in self.samples),
            "ranger_losses": sum(row[3] for row in self.samples),
            "holding": self.holding,
            "target": self.target,
            "reserve": self.reserve,
            "underfunded": self.underfunded,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_id": self.core_id,
            "last_tick": self.last_tick,
            "roster": self.roster,
            "samples": self.samples,
            "danger": self.danger,
            "danger_tick": self.danger_tick,
            "holding": self.holding,
            "target": self.target,
            "route": self.route,
            "retry_after": self.retry_after,
            "reserve": self.reserve,
            "underfunded": self.underfunded,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Recovery:
        return cls(
            core_id=raw.get("core_id"),
            last_tick=int(raw.get("last_tick", -1)),
            roster=dict(raw.get("roster", {})),
            samples=[
                (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                for row in raw.get("samples", [])[-WINDOW:]
            ],
            danger=tuple(raw["danger"]) if raw.get("danger") else None,
            danger_tick=int(raw.get("danger_tick", -1)),
            holding=bool(raw.get("holding", False)),
            target=tuple(raw["target"]) if raw.get("target") else None,
            route=tuple(tuple(cell) for cell in raw.get("route", [])[: MAX_ROUTE + 1]),
            retry_after=int(raw.get("retry_after", -1)),
            reserve=max(20, min(60, int(raw.get("reserve", 20)))),
            underfunded=bool(raw.get("underfunded", False)),
        )
