"""Convergence on a lone fighter that walks into the defensive zone."""

from __future__ import annotations

from arena_hero import Turn, UnitType, core_resource_capacity, unit_cost

from arena_hero_bot.geometry import add, manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig

from .factories import core, make_turn, object_id, unit

PERIMETER_REASON = "hold a defensive perimeter around the resource Core"
CONVERGE_REASON = "converge on the intruder inside the defensive perimeter"
# ``_offensive_patrol_enabled`` needs the roaming squad affordable, so the bank
# must clear the previous population tier's patrol floor.
GUARD_COUNT = 12
PATROL_FLOOR = core_resource_capacity(GUARD_COUNT - 1) - unit_cost(
    UnitType.RANGER,
    GUARD_COUNT - 1,
)


def _guards() -> list[dict[str, object]]:
    """A ring of Vanguards spread over the four bearings around the Core."""

    places = [
        (0, 6),
        (0, 7),
        (0, 8),
        (6, 0),
        (7, 0),
        (8, 0),
        (0, -6),
        (0, -7),
        (0, -8),
        (-6, 0),
        (-7, 0),
        (-8, 0),
    ]
    return [
        unit(number, "VANGUARD", position=place)
        for number, place in enumerate(places, start=2)
    ]


def _intruder_turn(tick: int, position: tuple[int, int]) -> Turn:
    return make_turn(
        tick=tick,
        resources=PATROL_FLOOR,
        objects=[
            core(),
            *_guards(),
            unit(40, "VANGUARD", controlled=False, position=position),
        ],
    )


def _reasons(report) -> dict[int, str]:
    lookup = {item.actor_id: item.reason for item in report.decisions}
    return {
        number: lookup[object_id(number)]
        for number in range(1, 64)
        if object_id(number) in lookup
    }


def test_nearby_guards_converge_on_a_lone_fighter_inside_the_zone() -> None:
    # Replay of the live shape: an enemy Vanguard walks the ground just below
    # the ring at Core distance 19-21.  Its first step raises the threat to
    # ALERT, which used to demote the roaming third and, together with the
    # twelve-cell memory clip, left the whole army standing still while it
    # strolled past.
    strategy = AggressiveStrategy(
        WorldMemory(),
        StrategyConfig(target_workers=0, max_population=None),
    )

    strategy.decide(_intruder_turn(100, (0, -20)))
    turn = _intruder_turn(101, (1, -19))
    report = strategy.decide(turn)

    assert report.threat_level == "ALERT"
    converging = [
        item
        for item in report.decisions
        if "VANGUARD" in item.reason or item.reason == CONVERGE_REASON
    ]
    assert len(converging) >= 3
    # Every mover stays inside the defensive leash: this is a bounded pincer,
    # not the open-ended pursuit the leash exists to stop.
    for item in converging:
        assert item.action == "MOVE"
        assert item.target is not None
        assert manhattan((0, 0), item.target) <= 20
    # The ring itself is never emptied for one fighter.
    reasons = _reasons(report)
    assert sum(text == PERIMETER_REASON for text in reasons.values()) >= 3


def test_convergence_holds_the_last_cell_through_a_dark_tick() -> None:
    # Vision over a mover flickers, so the squad must not disband and restart
    # every other Tick.  The remembered cell keeps the same units committed.
    strategy = AggressiveStrategy(
        WorldMemory(),
        StrategyConfig(target_workers=0, max_population=None),
    )

    strategy.decide(_intruder_turn(100, (0, -20)))
    strategy.decide(_intruder_turn(101, (1, -19)))
    dark = make_turn(
        tick=102,
        resources=PATROL_FLOOR,
        objects=[core(), *_guards()],
    )
    report = strategy.decide(dark)

    assert strategy._perimeter_intercept is not None
    _, anchor, _ = strategy._perimeter_intercept
    # Whether the squad reaches the cell through the remembered sighting or
    # through the idle fallback, it has to keep closing on it.
    committed = [
        item
        for item in report.decisions
        if item.reason in {CONVERGE_REASON, "hunt last seen unit"}
    ]
    assert len(committed) >= 3
    for item in committed:
        mover = next(
            unit_view for unit_view in dark.units if str(unit_view.id) == item.actor_id
        )
        step = dark.plan.unit_actions[mover.id]
        assert step.type == "MOVE"
        assert manhattan(anchor, add(mover.position, step.direction)) < manhattan(
            anchor,
            mover.position,
        )


def test_convergence_ignores_a_fighter_beyond_the_defensive_leash() -> None:
    # Only Core-local evidence may launch this.  A fighter outside the leash
    # is the raid detachment's business, not the whole army's.
    strategy = AggressiveStrategy(
        WorldMemory(),
        StrategyConfig(target_workers=0, max_population=None),
    )

    strategy.decide(_intruder_turn(100, (0, -26)))
    report = strategy.decide(_intruder_turn(101, (1, -25)))

    assert strategy._perimeter_intercept is None
    assert all(item.reason != CONVERGE_REASON for item in report.decisions)


def test_convergence_yields_to_a_real_attack() -> None:
    # Two or more fighters inside the ring is a defence, not a chase, and the
    # emergency path already owns that formation.
    strategy = AggressiveStrategy(
        WorldMemory(),
        StrategyConfig(target_workers=0, max_population=None),
    )

    attack = make_turn(
        tick=100,
        resources=PATROL_FLOOR,
        objects=[
            core(),
            *_guards(),
            unit(40, "VANGUARD", controlled=False, position=(0, -12)),
            unit(41, "VANGUARD", controlled=False, position=(1, -12)),
            unit(42, "VANGUARD", controlled=False, position=(2, -12)),
        ],
    )
    report = strategy.decide(attack)

    assert strategy._perimeter_intercept is None
    assert all(item.reason != CONVERGE_REASON for item in report.decisions)


def test_convergence_stops_once_the_squad_stands_on_an_empty_cell() -> None:
    # Arriving with nothing in sight ends the hunt.  Walking at an empty cell
    # for the rest of the memory window is the pursuit the leash forbids.
    strategy = AggressiveStrategy(
        WorldMemory(),
        StrategyConfig(target_workers=0, max_population=None),
    )

    strategy.decide(_intruder_turn(100, (0, -7)))
    strategy.decide(_intruder_turn(101, (0, -6)))
    dark = make_turn(
        tick=102,
        resources=PATROL_FLOOR,
        objects=[core(), *_guards()],
    )
    report = strategy.decide(dark)

    assert strategy._perimeter_intercept is None
    assert all(item.reason != CONVERGE_REASON for item in report.decisions)
