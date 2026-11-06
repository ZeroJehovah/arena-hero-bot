"""Bounded simultaneous movement through friendly two-entity cells."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID

from arena_hero import Position


def resolve_moves(
    occupancy: Mapping[Position, int],
    origins: Mapping[UUID, Position],
    requests: Mapping[UUID, tuple[Position, ...]],
    alternatives: Mapping[UUID, tuple[Position, ...]],
    *,
    max_depth: int = 12,
    max_search: int = 1024,
) -> dict[UUID, Position]:
    """Commit complete swaps/chains; failed searches leave no partial moves.

    Occupancy includes immovable entities and already committed movements.
    Only actors in ``alternatives`` may be moved to complete a dependency.
    Request order chooses between competing flows, never whether a full-cell
    swap is legal. Every accepted group is checked at its simultaneous end.
    """

    residents: dict[Position, list[UUID]] = {}
    for actor, origin in origins.items():
        residents.setdefault(origin, []).append(actor)
    for actors in residents.values():
        actors.sort(key=lambda actor: (actor not in requests, actor.bytes))
    searched = 0

    def count(position: Position, moves: dict[UUID, Position]) -> int:
        return occupancy.get(position, 0) + sum(
            (destination == position) - (origins[actor] == position)
            for actor, destination in moves.items()
        )

    def place(
        actor: UUID,
        destination: Position,
        committed: dict[UUID, Position],
        depth: int,
    ) -> dict[UUID, Position] | None:
        nonlocal searched
        searched += 1
        if (
            searched > max_search
            or depth > max_depth
            or actor in committed
            or destination == origins[actor]
        ):
            return None
        trial = {**committed, actor: destination}
        if count(destination, trial) <= 2:
            return trial
        for occupant in residents.get(destination, ()):
            if occupant in trial:
                continue
            for escape in alternatives.get(occupant, ()):
                result = place(occupant, escape, trial, depth + 1)
                if result is not None and count(destination, result) <= 2:
                    return result
        return None

    moves: dict[UUID, Position] = {}
    for actor, destinations in requests.items():
        if actor in moves:
            continue
        for destination in destinations:
            result = place(actor, destination, moves, 0)
            if result is not None:
                moves = result
                break
    return moves
