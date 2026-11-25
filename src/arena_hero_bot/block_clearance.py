"""Durable local surveys and confirmed Core clearance before campaign hops."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from arena_hero import CoreView, Position, Turn

SIZE = 32


def block_at(position: Position) -> Position:
    return position[0] // SIZE, position[1] // SIZE


@lru_cache(maxsize=16)
def cells_in(block: Position) -> frozenset[Position]:
    return frozenset(
        (block[0] * SIZE + x, block[1] * SIZE + y)
        for x in range(SIZE)
        for y in range(SIZE)
    )


@dataclass(slots=True)
class Survey:
    seen: dict[Position, int] = field(default_factory=dict)
    # Moving or losing sight of an enemy never discharges its original block.
    cores: dict[str, Position] = field(default_factory=dict)


@dataclass(slots=True)
class BlockClearance:
    core_id: str | None = None
    current: Position | None = None
    blocks: dict[Position, Survey] = field(default_factory=dict)
    # Terrain exclusions are recomputed after restart and new rock discoveries.
    excluded: set[Position] = field(default_factory=set)
    terrain_key: tuple[Position, int] | None = None
    unseen_cells: int | None = None

    def observe(
        self,
        turn: Turn,
        anchors: Iterable[Position | None],
        seen: Iterable[Position],
    ) -> None:
        core_id = str(turn.core.id) if turn.core is not None else None
        respawn = any(
            event.event_type == "CORE_RESPAWNED" and str(event.target_id) == core_id
            for event in turn.events
        )
        if core_id != self.core_id or respawn:
            self.core_id = core_id
            self.blocks.clear()
            self.terrain_key = None
            self.excluded.clear()
        current = block_at(turn.core.position) if turn.core else None
        if current != self.current:
            self.terrain_key = None
            self.excluded.clear()
        self.current = current
        self.unseen_cells = None
        if self.current is None:
            return
        # Only the active journey's blocks are retained, never a world atlas.
        active = {self.current} | {
            block_at(position)
            for position in anchors
            if position is not None
            and max(abs(block_at(position)[i] - self.current[i]) for i in (0, 1)) <= 1
        }
        self.blocks = {block: self.blocks.get(block, Survey()) for block in active}
        for position in seen:
            survey = self.blocks.get(block_at(position))
            if survey is not None:
                survey.seen[position] = turn.tick
        destroyed = {
            str(event.target_id)
            for event in turn.events
            if event.target_id is not None
            and event.event_type in {"CORE_DESTROYED", "DESTRUCTION_PARTICIPATION"}
        }
        for survey in self.blocks.values():
            for enemy_id in destroyed:
                survey.cores.pop(enemy_id, None)
        for enemy in turn.visible_enemies:
            if not isinstance(enemy, CoreView) or str(enemy.id) in destroyed:
                continue
            enemy_id = str(enemy.id)
            survey = self.blocks.get(block_at(enemy.position))
            if survey is not None:
                survey.cores[enemy_id] = enemy.position
            for survey in self.blocks.values():
                if enemy_id in survey.cores:
                    survey.cores[enemy_id] = enemy.position

    def summary(self) -> dict[str, Any]:
        survey = self.blocks.get(self.current) if self.current is not None else None
        return {
            "block": self.current,
            "observed_cells": len(survey.seen) if survey else 0,
            "unobservable_cells": len(self.excluded),
            "unseen_cells": self.unseen_cells,
            "pending_cores": sorted(survey.cores) if survey else [],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_id": self.core_id,
            "current": self.current,
            "blocks": [
                {
                    "block": block,
                    "seen": [
                        [*cell, tick] for cell, tick in sorted(survey.seen.items())
                    ],
                    "cores": dict(sorted(survey.cores.items())),
                }
                for block, survey in sorted(self.blocks.items())
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BlockClearance:
        current = raw.get("current")
        return cls(
            core_id=raw.get("core_id"),
            current=(int(current[0]), int(current[1])) if current is not None else None,
            blocks={
                (int(row["block"][0]), int(row["block"][1])): Survey(
                    seen={(int(x), int(y)): int(tick) for x, y, tick in row["seen"]},
                    cores={
                        key: (int(pos[0]), int(pos[1]))
                        for key, pos in row["cores"].items()
                    },
                )
                for row in raw.get("blocks", [])
            },
        )
