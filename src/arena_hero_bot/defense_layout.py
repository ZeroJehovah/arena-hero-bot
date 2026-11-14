"""Stationary Ranger posts with verified terrain-aware shared vision."""

from collections import deque
from functools import lru_cache

from arena_hero import Position

from .exploration import visible_cells
from .geometry import adjacent_positions, manhattan

LAYOUT_VERSION = 1
MAX_RADIUS = 12
VISION = 5
SPACING = 3


def covered_radius(
    center: Position,
    observers: tuple[tuple[Position, int], ...],
    rocks: set[Position] | frozenset[Position],
    maximum: int,
) -> int:
    """Measure a complete disk from real observers, excluding only rocks."""

    seen = set(visible_cells(center, VISION, rocks))
    for position, vision in observers:
        seen.update(visible_cells(position, vision, rocks))
    return min(
        (
            abs(dx) + abs(dy) - 1
            for dx in range(-maximum, maximum + 1)
            for dy in range(-maximum, maximum + 1)
            if abs(dx) + abs(dy) <= maximum
            and (cell := (center[0] + dx, center[1] + dy)) not in rocks
            and cell not in seen
        ),
        default=maximum,
    )


@lru_cache(maxsize=64)
def plan_posts(
    center: Position,
    count: int,
    local_rocks: tuple[Position, ...],
    bounds: tuple[int, int, int, int],
    retained: tuple[Position, ...] = (),
) -> tuple[tuple[Position, ...], int]:
    """Keep usable owners, cover the interior, then spread remaining posts.

    The search is bounded by a radius-12 disk and two coordinate passes per
    radius. Only reachable posts inside the current chunk are candidates.
    A radius is accepted only after every non-rock cell is actually covered;
    neither unfilled template slots nor Workers/patrols stand in for guards.
    """

    rocks = frozenset(local_rocks)
    min_x, max_x, min_y, max_y = bounds
    reachable = {center}
    frontier = deque([center])
    while frontier:
        for cell in adjacent_positions(frontier.popleft()):
            if (
                cell not in reachable
                and cell not in rocks
                and min_x <= cell[0] <= max_x
                and min_y <= cell[1] <= max_y
            ):
                reachable.add(cell)
                frontier.append(cell)
    maximum = min(MAX_RADIUS, 4 + count // 2)
    # Existing owners may continue using the depth earned by a larger roster.
    maximum = min(MAX_RADIUS, max((maximum, *(manhattan(center, p) for p in retained))))
    cells = tuple(
        (center[0] + dx, center[1] + dy)
        for dx in range(-maximum - VISION, maximum + VISION + 1)
        for dy in range(-maximum - VISION, maximum + VISION + 1)
        if abs(dx) + abs(dy) <= maximum + VISION
        and (center[0] + dx, center[1] + dy) not in rocks
    )
    indexes = {cell: index for index, cell in enumerate(cells)}

    def view(position: Position) -> int:
        return sum(
            1 << indexes[cell]
            for cell in visible_cells(position, VISION, rocks)
            if cell in indexes
        )

    core_view = view(center)
    masks = {
        cell: view(cell)
        for cell in cells
        if 2 <= manhattan(center, cell) <= maximum and cell in reachable
    }
    nearby = {
        p: tuple(
            sum(
                1 << indexes[q]
                for dx in range(-distance, distance + 1)
                for dy in range(-distance, distance + 1)
                if abs(dx) + abs(dy) == distance
                and (q := (p[0] + dx, p[1] + dy)) in masks
            )
            for distance in range(1, SPACING + 1)
        )
        for p in masks
    }
    for radius in range(maximum, 0, -1):
        target = sum(
            1 << indexes[cell] for cell in cells if manhattan(center, cell) <= radius
        )
        candidates = tuple(p for p in masks if manhattan(center, p) <= radius)
        if len(candidates) < count:
            continue
        chosen = list(dict.fromkeys(p for p in retained if p in candidates))[:count]
        union = core_view
        for p in chosen:
            union |= masks[p]

        def score(
            position: Position, base: int, occupied: int, target: int = target
        ) -> tuple[int, int, int, int, int]:
            trial = base | masks[position]
            near_one, near_two, near_three = nearby[position]
            crowding = (near_one & occupied).bit_count() * 4 + (
                near_two & occupied
            ).bit_count()
            return (
                (trial & target).bit_count(),
                -crowding,
                trial.bit_count(),
                int(not (near_three & occupied)),
                -manhattan(center, position),
            )

        occupied = sum(1 << indexes[p] for p in chosen)
        while len(chosen) < count:
            best = max(
                (p for p in candidates if p not in chosen),
                key=lambda p: score(p, union, occupied),
            )
            chosen.append(best)
            occupied |= 1 << indexes[best]
            union |= masks[best]
        # Preserve already valid owners on ordinary reinforcement/casualties.
        # Repair obstructed coverage before considering any outward benefit.
        if union & target != target or not retained:
            for _pass in range(2):
                changed = False
                for slot, old in enumerate(chosen):
                    others = chosen[:slot] + chosen[slot + 1 :]
                    base = core_view
                    for p in others:
                        base |= masks[p]
                    occupied = sum(1 << indexes[p] for p in others)
                    best = max(
                        (p for p in candidates if p not in others),
                        key=lambda p: score(p, base, occupied),
                    )
                    if score(best, base, occupied) > score(old, base, occupied):
                        chosen[slot] = best
                        changed = True
                if not changed:
                    break
        posts = tuple(chosen)
        if (
            covered_radius(center, tuple((p, VISION) for p in posts), rocks, radius)
            == radius
        ):
            return posts, radius
    # An enclosed non-rock cell is still a blind spot. Never label a failed
    # layout gap-free, and never send a guard onto a rock as a fallback.
    return (), 0
