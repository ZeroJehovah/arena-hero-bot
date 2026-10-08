"""Legacy squad identity and production must agree with available recruits."""

from dataclasses import replace

import pytest
from arena_hero import UnitType

from arena_hero_bot.exploration import ExplorationRoute
from arena_hero_bot.memory import ExpeditionSquad, WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig

from .factories import core, make_turn, object_id, unit

CONFIG = StrategyConfig(target_workers=16, max_population=None, expedition_mode=True)


def test_duplicate_serial_migration_preserves_state_and_survives_restart(tmp_path):
    route = ExplorationRoute(path=((30, 0), (31, 0)), assigned_tick=90)
    original = [
        ExpeditionSquad(4, (object_id(2),), (1, 0)),
        ExpeditionSquad(
            4,
            (object_id(3), object_id(4)),
            (0, 1),
            pursuit_target_id=object_id(99),
            pursuit_position=(40, 0),
            pursuit_direction=(1, 0),
            pursuit_distance=5,
            exploration_route=route,
            regroup_order=(object_id(4), object_id(3)),
        ),
        ExpeditionSquad(30, (object_id(5),), (0, -1)),
    ]
    memory = WorldMemory(
        expedition_squads=original.copy(),
        next_expedition_serial=29,
        unit_roles={object_id(3): "expedition-4", object_id(4): "expedition-4"},
    )
    strategy = AggressiveStrategy(memory, CONFIG)
    assert memory.expedition_squads == [
        original[0],
        replace(original[1], serial=31),
        original[2],
    ]
    assert memory.next_expedition_serial == 31
    assert memory.unit_roles[object_id(3)] == "expedition-31"
    path = tmp_path / "memory.json"
    memory.save(path)
    resumed = AggressiveStrategy(WorldMemory.load(path), CONFIG)
    assert resumed.memory.expedition_squads == memory.expedition_squads
    assert resumed._expedition_serial == strategy._expedition_serial == 31
    resumed._update_expeditions(
        make_turn(
            objects=[core(), unit(2, "VANGUARD"), unit(4, "RANGER"), unit(5, "RANGER")]
        )
    )
    survivor = resumed.memory.expedition_squads[1]
    assert survivor.serial == 31
    assert survivor.members == (object_id(4),)
    assert survivor.exploration_route == route
    assert survivor.pursuit_target_id == object_id(99)


def _production_case(staged_v, staged_r, away_v=13, away_r=11):
    objects = [core()]
    roles = {}
    next_id = 2

    def add(kind, count, role):
        nonlocal next_id
        members = []
        for _ in range(count):
            objects.append(unit(next_id, kind, position=(next_id, 20)))
            roles[object_id(next_id)] = role
            members.append(object_id(next_id))
            next_id += 1
        return members

    add("WORKER", 16, "worker")
    add("VANGUARD", 4, "defense")
    add("RANGER", 16, "defense")
    for team in range(1, 5):
        add("RANGER", 2, f"patrol-{team}")
    squads = []
    # Independent remnants deliberately skew the departed force's ratio.
    for kind, count in (("VANGUARD", away_v), ("RANGER", away_r)):
        for _ in range(count):
            serial = len(squads) + 1
            members = add(kind, 1, f"expedition-{serial}")
            squads.append(ExpeditionSquad(serial, tuple(members), (1, 0)))
    add("VANGUARD", staged_v, "staged")
    add("RANGER", staged_r, "staged")
    memory = WorldMemory(
        unit_roles=roles, unit_roles_initialized=True, expedition_squads=squads
    )
    return AggressiveStrategy(memory, CONFIG), make_turn(
        resources=10000, objects=objects
    )


@pytest.mark.parametrize(
    "staged_v,staged_r,expected",
    [
        (1, 2, UnitType.VANGUARD),
        (1, 3, UnitType.VANGUARD),
        (0, 3, UnitType.VANGUARD),
        (3, 1, UnitType.RANGER),
        (9, 1, UnitType.RANGER),
        (7, 0, UnitType.RANGER),
    ],
)
def test_production_completes_staging_instead_of_balancing_remnants(
    staged_v, staged_r, expected
):
    strategy, turn = _production_case(staged_v, staged_r)
    assert strategy._choose_spawn(turn, turn.resources) is expected


def test_missing_local_defender_is_not_hidden_by_departed_units():
    strategy, turn = _production_case(0, 3)
    missing = next(
        u for u in turn.vanguards if strategy.memory.unit_roles[str(u.id)] == "defense"
    )
    turn = make_turn(
        resources=10000,
        objects=[
            obj
            for obj in turn.state.model_dump(mode="json")["objects"]
            if obj.get("id") != str(missing.id)
        ],
    )
    assert strategy._choose_spawn(turn, turn.resources) is UnitType.VANGUARD


def test_recruit_completes_new_squad_without_absorbing_remnants():
    strategy, turn = _production_case(1, 3)
    original = {frozenset(s.members) for s in strategy.memory.expedition_squads}
    turn = make_turn(
        resources=10000,
        objects=[*turn.state.model_dump(mode="json")["objects"], unit(999, "VANGUARD")],
    )
    strategy._update_expeditions(turn)
    squads = strategy.memory.expedition_squads
    assert {frozenset(s.members) for s in squads[:-1]} == original
    assert len(squads[-1].members) == 4
    assert len({s.serial for s in squads}) == len(squads)
    assert squads[-1].serial > max(s.serial for s in squads[:-1])
    assert len(strategy._staged_ids) == 1


def test_released_vanguards_form_a_new_squad_when_the_missing_ranger_arrives():
    strategy, turn = _production_case(9, 1)
    original = {frozenset(s.members) for s in strategy.memory.expedition_squads}
    local_roles = {
        uid: role
        for uid, role in strategy.memory.unit_roles.items()
        if role == "defense" or role.startswith("patrol-")
    }
    assert strategy._choose_spawn(turn, turn.resources) is UnitType.RANGER
    strategy._update_expeditions(turn)
    assert {frozenset(s.members) for s in strategy.memory.expedition_squads} == original
    assert len(strategy._staged_ids) == 10
    recruited = make_turn(
        tick=101,
        resources=10000,
        objects=[*turn.state.model_dump(mode="json")["objects"], unit(999, "RANGER")],
    )

    strategy._update_expeditions(recruited)

    squads = strategy.memory.expedition_squads
    assert {frozenset(s.members) for s in squads[:-1]} == original
    new_members = set(squads[-1].members)
    assert object_id(999) in new_members
    assert sum(str(u.id) in new_members for u in recruited.vanguards) == 2
    assert sum(str(u.id) in new_members for u in recruited.rangers) == 2
    assert len(strategy._staged_ids) == 7
    assert strategy._staged_ids <= {u.id for u in recruited.vanguards}
    assert all(
        strategy.memory.unit_roles[uid] == role for uid, role in local_roles.items()
    )
    assert strategy._choose_spawn(recruited, recruited.resources) is UnitType.RANGER


def test_missing_patrol_ranger_is_replaced_without_reintroducing_a_vanguard():
    strategy, turn = _production_case(9, 0)
    original_roles = {
        uid: role
        for uid, role in strategy.memory.unit_roles.items()
        if role != "worker"
    }
    missing = next(uid for uid, role in original_roles.items() if role == "patrol-2")
    survivors = [
        obj
        for obj in turn.state.model_dump(mode="json")["objects"]
        if obj.get("id") != missing
    ]
    depleted = make_turn(tick=101, resources=10000, objects=survivors)

    assert strategy._choose_spawn(depleted, depleted.resources) is UnitType.RANGER
    recruited = make_turn(
        tick=102, resources=10000, objects=[*survivors, unit(999, "RANGER")]
    )
    strategy._reconcile_unit_roles(recruited)

    assert strategy.memory.unit_roles == {
        **{uid: role for uid, role in original_roles.items() if uid != missing},
        object_id(999): "patrol-2",
    }
    for team in range(1, 5):
        members = strategy._patrol_team_members(team, recruited)
        assert len(members) == 2
        assert all(member.unit_type is UnitType.RANGER for member in members)
