"""A sheltered patient must not orbit a distant ally instead of healing."""

from uuid import UUID

import pytest

from arena_hero_bot.geometry import add, manhattan
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig

from .factories import object_id, unit
from .test_layered_defense import Encounter
from .test_skirmish import _context


def _returning(*, hp=1):
    friends = [
        unit(2, "RANGER", position=(-4, 1), hp=hp),
        unit(3, "RANGER", position=(-9, -3)),
    ]
    enemies = [
        unit(50 + i, "RANGER", position=(-10, 0), controlled=False) for i in range(2)
    ]
    encounter = Encounter(friends, enemies, resources=1, rocks=[(-4, 0), (-5, 0)])
    memory = WorldMemory(
        unit_roles={friend["id"]: "defense" for friend in friends},
        unit_roles_initialized=True,
        core_home_position=(0, 0),
    )
    return AggressiveStrategy(memory, StrategyConfig(campaign_mode=True)), encounter


@pytest.mark.parametrize("restart", [False, True])
def test_safe_wounded_ranger_reaches_core_and_heals_without_cover_orbit(
    tmp_path, restart
):
    strategy, encounter = _returning()
    distances = [5]
    for offset in range(7):
        turn = encounter.turn()
        context = _context(strategy, turn)
        context.remaining_resources = turn.resources
        ranger = next(r for r in turn.rangers if r.id.int == 2)
        assert strategy._recover_if_critical(ranger, maximum_hp=2, context=context)
        encounter.settle(turn.plan, {})
        distances.append(
            manhattan(tuple(encounter.friends[object_id(2)]["position"]), (0, 0))
        )
        if restart and offset == 0:
            path = tmp_path / "memory.json"
            strategy.memory.save(path)
            strategy = AggressiveStrategy(WorldMemory.load(path), strategy.config)
        if encounter.heals:
            break
    assert distances == [5, 4, 3, 2, 1, 0, 0]
    assert encounter.heals == 1 and encounter.losses == 0
    assert encounter.friends[object_id(2)]["hp"] == 2


def test_healthy_defender_still_keeps_retreat_cover():
    strategy, encounter = _returning(hp=2)
    turn = encounter.turn()
    context = _context(strategy, turn)
    ranger = next(r for r in turn.rangers if r.id.int == 2)
    assert strategy._move_ranger_toward_core(ranger, context, reason="withdraw")
    action = turn.plan.unit_actions[UUID(int=2)]
    assert action.type == "MOVE"
    assert add(ranger.position, action.direction) == (-5, 1)


@pytest.mark.parametrize("danger", ["current", "future", "known_hit"])
def test_healing_progress_never_enters_new_fire_or_proven_lethal_cell(danger):
    strategy, encounter = _returning()
    if danger == "known_hit":
        strategy.memory.incoming_damage[(-3, 1)] = (99, 1)
    else:
        enemy = unit(
            52,
            "RANGER",
            position=(-3, -2) if danger == "current" else (-3, -3),
            controlled=False,
        )
        encounter.foes[enemy["id"]] = enemy
        if danger == "future":
            for tick, y in ((98, -5), (99, -4)):
                encounter.tick = tick
                enemy["position"] = [-3, y]
                strategy.memory.observe(encounter.turn())
            encounter.tick = 100
            enemy["position"] = [-3, -3]
    turn = encounter.turn()
    context = _context(strategy, turn)
    context.remaining_resources = turn.resources
    ranger = next(r for r in turn.rangers if r.id.int == 2)
    if danger == "future":
        assert strategy._future_attackers_at((-3, 1), turn, strategy._rock_map(turn))
    assert strategy._recover_if_critical(ranger, maximum_hp=2, context=context)
    action = turn.plan.unit_actions[UUID(int=2)]
    if action.type == "MOVE":
        destination = add(ranger.position, action.direction)
        assert destination != (-3, 1)
        assert not strategy._ranger_attackers_at(
            destination, turn, strategy._rock_map(turn)
        )
        assert not strategy._future_attackers_at(
            destination, turn, strategy._rock_map(turn)
        )
