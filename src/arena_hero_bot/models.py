"""Internal decision and telemetry models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from arena_hero import Position


@dataclass(frozen=True, slots=True)
class Decision:
    """One queued action and the tactical reason behind it."""

    actor_id: str
    actor_kind: str
    action: str
    reason: str
    target: Position | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable decision."""

        return asdict(self)


@dataclass(slots=True)
class DecisionReport:
    """All decisions and summary counters for one Turn."""

    tick: int
    decisions: list[Decision] = field(default_factory=list)
    visible_enemies: int = 0
    remembered_enemies: int = 0
    threat_level: str = "NORMAL"
    threat_reason: str = "NONE"
    minimum_ticks_to_range: int | None = None
    projected_core_damage: int = 0
    planned_damage: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        *,
        actor_id: str,
        actor_kind: str,
        action: str,
        reason: str,
        target: Position | None = None,
    ) -> None:
        """Append one action explanation."""

        self.decisions.append(
            Decision(
                actor_id=actor_id,
                actor_kind=actor_kind,
                action=action,
                reason=reason,
                target=target,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe telemetry data."""

        return {
            "tick": self.tick,
            "visible_enemies": self.visible_enemies,
            "remembered_enemies": self.remembered_enemies,
            "threat_level": self.threat_level,
            "threat_reason": self.threat_reason,
            "minimum_ticks_to_range": self.minimum_ticks_to_range,
            "projected_core_damage": self.projected_core_damage,
            "planned_damage": dict(sorted(self.planned_damage.items())),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }
