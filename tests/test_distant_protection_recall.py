"""A distant pursuit needs real intercepting fire before recall may wait."""

from dataclasses import replace

import pytest
from arena_hero import Direction

from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig

from .factories import core, make_turn, unit


@pytest.mark.parametrize(
    "gate",
    [
        "covered",
        "single",
        "wounded",
        "rock",
        "moving",
        "recent",
        "projected",
        "near",
        "hidden",
    ],
)
def test_pursuit_recall_uses_arrival_time_and_distinct_current_safe_shooters(gate):
    strategy = AggressiveStrategy(WorldMemory(), StrategyConfig(campaign_mode=True))
    for tick, distance in ((99, 29), (100, 28), (101, 27), (102, 26)):
        if gate == "near":
            distance -= 14
        height = distance - 3
        objects = [
            core(),
            unit(2, "RANGER", position=(-3, height)),
            unit(3, "RANGER", position=(3, height), hp=1 if gate == "wounded" else 2),
        ]
        if gate == "single":
            objects.pop()
        if gate != "hidden" or tick < 102:
            objects.append(
                unit(50, "VANGUARD", controlled=False, position=(0, distance))
            )
        turn = make_turn(
            tick=tick,
            objects=objects,
            obstacles=[(-2, height + 1)] if gate == "rock" else [],
        )
        strategy.memory.observe(turn)
        threat = strategy._combat_policy.assess(turn, strategy._rock_map(turn))
    assert threat.pursuing_enemy_ids
    if gate == "recent":
        threat = replace(threat, recent_core_attack=True)
    if gate == "projected":
        threat = replace(threat, projected_core_damage=1)
    if gate == "moving":
        turn.rangers[0].move(Direction.LEFT)
    assert strategy._clearance_recall_needed(turn, threat) is (gate != "covered")


def test_two_pursuers_cannot_both_borrow_the_same_two_shooters():
    strategy = AggressiveStrategy(WorldMemory(), StrategyConfig(campaign_mode=True))
    for tick, distance in ((100, 28), (101, 27), (102, 26)):
        turn = make_turn(
            tick=tick,
            objects=[
                core(),
                unit(2, "RANGER", position=(-3, distance - 3)),
                unit(3, "RANGER", position=(3, distance - 3)),
                unit(50, "VANGUARD", controlled=False, position=(0, distance)),
                unit(51, "VANGUARD", controlled=False, position=(0, distance)),
            ],
        )
        strategy.memory.observe(turn)
        threat = strategy._combat_policy.assess(turn, set())
    assert strategy._clearance_recall_needed(turn, threat)
