"""A hunt inside ordinary patrol bounds keeps its hunter in the local roster."""

from uuid import UUID

import pytest

from arena_hero_bot.geometry import line_of_fire
from arena_hero_bot.memory import CoreHunt, WorldMemory

from .factories import core, make_turn, object_id, unit
from .test_core_hunt import HOME, TARGET, _advance, _memory, _rival, _roster, _strategy
from .test_core_traffic import settled_positions


@pytest.mark.parametrize("restart", [False, True])
def test_two_local_rangers_complete_a_core_hunt_without_detaching(tmp_path, restart):
    memory = _memory()
    roster = _roster()[:2]
    strategy = _strategy(memory)
    hp, shield = 5, 5
    hits = 0
    for tick in range(100, 120):
        if restart and tick == 105:
            path = tmp_path / "memory.json"
            memory.save(path)
            memory = WorldMemory.load(path)
            strategy = _strategy(memory)
        rival = _rival()
        rival.update(hp=hp, shield=shield)
        turn = make_turn(tick=tick, objects=[core(position=HOME), *roster, rival])

        strategy.decide(turn)

        hunt = memory.core_hunt
        assert hunt is not None
        assert hunt.ranger_id == object_id(2)
        assert (hunt.started, hunt.deadline) == (100, 240)
        assert memory.unit_roles[object_id(2)] == "patrol-1"
        hunter = turn.unit(object_id(2))
        action = turn.plan.unit_actions[hunter.id]
        if action.type == "SHOOT":
            assert action.target_id == UUID(object_id(900))
            assert action.expected_cell == TARGET
            assert line_of_fire(hunter.position, TARGET, turn.obstacle_cells)
            if shield:
                shield -= 1
            else:
                hp -= 1
            hits += 1
        else:
            assert action.type == "MOVE"
        _advance(roster, turn)
        assert all(0 <= coordinate <= 31 for coordinate in roster[0]["position"])
        if hp == 0:
            break
    else:
        pytest.fail("local hunter did not finish the stationary Core")
    assert hits == 10
    finished = make_turn(
        tick=tick + 1,
        objects=[core(position=HOME), *roster],
        events=[
            {
                "event_id": object_id(990),
                "tick": tick + 1,
                "event_type": "CORE_DESTROYED",
                "target_id": object_id(900),
                "position": list(TARGET),
            }
        ],
    )
    strategy.decide(finished)
    assert memory.core_hunt is None
    assert memory.core_hunt_cooldown_until == finished.tick + 60


@pytest.mark.parametrize("reserve", ["missing", "wounded", "outside", "returning"])
def test_local_hunter_still_needs_another_actual_healthy_local_unit(reserve):
    memory = _memory()
    roster = _roster()[:2]
    if reserve == "missing":
        roster = roster[:1]
    elif reserve == "wounded":
        roster[1]["hp"] = 1
    elif reserve == "outside":
        roster[1]["position"] = [100, 100]
    else:
        roster[1]["position"] = [31, 31]
        memory.core_hunt_returns[object_id(3)] = 99
    turn = make_turn(objects=[core(position=HOME), *roster, _rival()])

    _strategy(memory).decide(turn)

    assert memory.core_hunt is None


def test_local_hunter_uses_patrol_bounds_even_beyond_the_old_twenty_cell_ring():
    memory = _memory()
    roster = _roster()[:2]
    roster[0]["position"] = [2, 3]
    target = (2, 0)
    turn = make_turn(objects=[core(position=HOME), *roster, _rival(target)])

    _strategy(memory).decide(turn)

    assert memory.core_hunt is not None
    assert memory.core_hunt.ranger_id == object_id(2)
    shot = turn.plan.unit_actions[UUID(object_id(2))]
    assert shot.type == "SHOOT"
    assert shot.target_id == UUID(object_id(900))


@pytest.mark.parametrize("count", [2, 3])
def test_crossing_the_patrol_edge_still_needs_two_other_local_reserves(count):
    memory = _memory()
    roster = _roster()[:count]
    roster[0]["position"] = [16, 0]
    turn = make_turn(objects=[core(position=HOME), *roster, _rival((16, -4))])

    _strategy(memory).decide(turn)

    assert (memory.core_hunt is not None) is (count == 3)
    if count == 3:
        assert settled_positions(turn)[UUID(object_id(2))] == (16, -1)


@pytest.mark.parametrize("restart", [False, True])
def test_local_hunt_takes_the_longer_in_bounds_route_around_a_wall(tmp_path, restart):
    memory = _memory()
    roster = _roster()[:2]
    roster[0]["position"] = [0, 7]
    roster[1]["position"] = [28, 28]
    rocks = {(x, 8) for x in range(9)}
    target = (0, 12)
    strategy = _strategy(memory)
    visited = set()
    for tick in range(100, 135):
        if restart and tick == 105:
            path = tmp_path / "memory.json"
            memory.save(path)
            memory = WorldMemory.load(path)
            strategy = _strategy(memory)
        turn = make_turn(
            tick=tick,
            objects=[core(position=HOME), *roster, _rival(target)],
            obstacles=rocks,
        )

        strategy.decide(turn)

        assert memory.core_hunt is not None
        assert memory.core_hunt.ranger_id == object_id(2)
        hunter = turn.unit(object_id(2))
        action = turn.plan.unit_actions[hunter.id]
        if action.type == "SHOOT":
            assert action.target_id == UUID(object_id(900))
            assert line_of_fire(hunter.position, target, rocks)
            break
        assert action.type == "MOVE"
        next_position = settled_positions(turn)[hunter.id]
        assert all(0 <= coordinate <= 31 for coordinate in next_position)
        assert next_position not in rocks
        visited.add(next_position)
        _advance(roster, turn)
    else:
        pytest.fail("hunter did not reach the firing position inside patrol bounds")
    assert any(x >= 9 for x, _y in visited)


@pytest.mark.parametrize("restart", [False, True])
def test_fleeing_target_cannot_pull_local_hunter_out_until_reserves_arrive(
    tmp_path, restart
):
    memory = _memory()
    roster = _roster()[:2]
    roster[0]["position"] = [16, 4]
    strategy = _strategy(memory)
    strategy.decide(
        make_turn(tick=100, objects=[core(position=HOME), *roster, _rival((16, 1))])
    )
    assert memory.core_hunt is not None
    if restart:
        path = tmp_path / "memory.json"
        memory.save(path)
        memory = WorldMemory.load(path)
        strategy = _strategy(memory)
    roster[0]["position"] = [16, 0]
    blocked = make_turn(
        tick=120, objects=[core(position=HOME), *roster, _rival((16, -4))]
    )

    strategy.decide(blocked)

    assert memory.core_hunt is not None
    assert (memory.core_hunt.started, memory.core_hunt.deadline) == (100, 240)
    assert blocked.plan.unit_actions[UUID(object_id(2))].type == "WAIT"
    roster.append(_roster()[2])
    permitted = make_turn(
        tick=121, objects=[core(position=HOME), *roster, _rival((16, -4))]
    )
    strategy.decide(permitted)
    assert memory.core_hunt is not None
    assert (memory.core_hunt.started, memory.core_hunt.deadline) == (100, 240)
    assert settled_positions(permitted)[UUID(object_id(2))] == (16, -1)


def test_search_for_a_hidden_core_also_stays_inside_patrol_bounds():
    memory = _memory()
    roster = _roster()[:2]
    roster[0]["position"] = [16, 0]
    memory.core_hunt = CoreHunt(
        object_id(1), object_id(900), object_id(2), 90, 230, (16, -4), 100
    )
    turn = make_turn(tick=108, objects=[core(position=HOME), *roster])

    _strategy(memory).decide(turn)

    assert memory.core_hunt is not None
    assert memory.core_hunt.deadline == 230
    assert turn.plan.unit_actions[UUID(object_id(2))].type == "WAIT"


def test_local_handoff_does_not_reuse_the_outbound_hunters_unrestricted_route():
    memory = _memory()
    roster = _roster()[:3]
    roster[0]["position"] = [16, 2]
    roster[1]["position"] = [16, 0]
    target = (16, -4)
    memory.core_hunt = CoreHunt(
        object_id(1), object_id(900), object_id(2), 90, 230, target, 99
    )
    turn = make_turn(objects=[core(position=HOME), *roster, _rival(target)])

    _strategy(memory).decide(turn)

    assert memory.core_hunt is not None
    # The nearer candidate has only one other reserve once the old hunter
    # returns, so its route must stay in bounds despite the shared target.
    assert memory.core_hunt.ranger_id == object_id(2)
    assert memory.core_hunt.deadline == 230
    assert settled_positions(turn)[UUID(object_id(2))] == (16, 1)


@pytest.mark.parametrize("danger", ["core_hp", "core_shield", "escort", "remembered"])
def test_local_hunt_keeps_core_pressure_and_escort_safety_checks(danger):
    memory = _memory()
    roster = _roster()[:2]
    home = core(position=HOME)
    objects = [home, *roster, _rival()]
    if danger == "core_hp":
        home["hp"] = 4
    elif danger == "core_shield":
        home["shield"] = 4
    elif danger == "escort":
        objects.append(unit(901, "VANGUARD", controlled=False, position=(7, 1)))
    else:
        memory.observe(
            make_turn(
                tick=99,
                objects=[
                    *objects,
                    unit(901, "RANGER", controlled=False, position=(7, -3)),
                ],
            )
        )
    turn = make_turn(tick=100, objects=objects)

    _strategy(memory).decide(turn)

    assert memory.core_hunt is None
