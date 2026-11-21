"""Unreachable scout viewpoints must not consume the economy's planning time."""

from arena_hero_bot import strategy as strategy_module
from arena_hero_bot.exploration import visible_cells
from arena_hero_bot.geometry import add, next_step
from arena_hero_bot.memory import UnitGoal, WorldMemory
from arena_hero_bot.strategy import (
    CAMPAIGN_PHASE_STAGE4,
    CAMPAIGN_SCOUT_PURPOSE,
    CORE_LAYOUT_VERSION,
    AggressiveStrategy,
    StrategyConfig,
    _OutsidePatrolArea,
)

from .factories import core, make_turn, object_id, unit
from .test_shared_movement import context_for

HOME = (815, 15)
TARGET = (783, 15)
ORIGIN = (780, 15)
PURPOSE = f"{CAMPAIGN_SCOUT_PURPOSE}-783-15"


def scout_memory():
    rocks = {(784, y) for y in range(32)} | {
        (x, y) for x in (788, 796) for y in (4, 12, 20, 28)
    }
    unseen = {(789, y) for y in range(10, 22)}
    return WorldMemory(
        core_home_position=HOME,
        core_layout_version=CORE_LAYOUT_VERSION,
        campaign_last_core_id=object_id(1),
        campaign_phase=CAMPAIGN_PHASE_STAGE4,
        campaign_target_quota=8,
        campaign_scout_target=TARGET,
        campaign_scout_started_tick=90,
        campaign_scout_seen_ticks={
            (x, y): 100
            for x in range(768, 800)
            for y in range(32)
            if (x, y) not in unseen
        },
        obstacles=rocks,
    )


def test_scout_routes_around_a_wall_only_when_in_block_views_are_unreachable(
    monkeypatch,
):
    memory = scout_memory()
    strategy = AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))
    turn = make_turn(objects=[core(position=HOME), unit(2, "RANGER", position=ORIGIN)])
    searches = 0

    def counted_search(*args, **kwargs):
        nonlocal searches
        searches += 1
        return next_step(*args, **kwargs)

    monkeypatch.setattr(strategy_module, "next_step", counted_search)

    ranger = turn.rangers[0]
    context = context_for(turn)
    goal = strategy._campaign_scout_goal(ranger, turn, context)

    assert goal != ORIGIN
    assert 768 <= goal[0] <= 799 and 0 <= goal[1] <= 31
    assert any(
        (789, y) in visible_cells(goal, 5, memory.obstacles) for y in range(10, 22)
    )
    assert memory.goal_for(str(ranger.id)).purpose == f"{PURPOSE}-detour"
    inner_area = strategy._campaign_scout_area(ranger, TARGET, turn)
    detour_area = strategy._campaign_scout_detour_area(ranger, TARGET, turn)
    assert not strategy._has_static_route(
        ranger,
        goal,
        context,
        extra_blockers=_OutsidePatrolArea(inner_area, ranger.position),
    )
    assert strategy._has_static_route(
        ranger,
        goal,
        context,
        extra_blockers=_OutsidePatrolArea(detour_area, ranger.position),
    )
    assert searches <= 2
    assert memory.campaign_scout_target == TARGET
    assert (789, 15) not in memory.campaign_scout_seen_ticks


def test_opening_a_scout_passage_refreshes_reachability_in_the_same_tick():
    memory = scout_memory()
    strategy = AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))
    turn = make_turn(objects=[core(position=HOME), unit(2, "RANGER", position=ORIGIN)])
    ranger = turn.rangers[0]
    first_goal = strategy._campaign_scout_goal(ranger, turn, context_for(turn))
    assert first_goal != ORIGIN
    assert memory.goal_for(str(ranger.id)).purpose == f"{PURPOSE}-detour"

    memory.obstacles.remove((784, 15))
    goal = strategy._campaign_scout_goal(ranger, turn, context_for(turn))

    assert goal == first_goal
    assert memory.goal_for(str(ranger.id)).purpose == PURPOSE
    assert 768 <= goal[0] <= 799 and 0 <= goal[1] <= 31
    assert memory.goal_for(str(ranger.id)).purpose == PURPOSE
    assert any(
        (789, y) in visible_cells(goal, 5, memory.obstacles) for y in range(10, 22)
    )
    assert (
        next_step(ORIGIN, goal, blocked=memory.obstacles, require_path=True) is not None
    )


def test_scout_reachability_follows_actual_origin_after_restart(tmp_path):
    memory = scout_memory()
    strategy = AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))
    first = make_turn(objects=[core(position=HOME), unit(2, "RANGER", position=ORIGIN)])
    first_goal = strategy._campaign_scout_goal(
        first.rangers[0], first, context_for(first)
    )
    assert first_goal != ORIGIN
    path = tmp_path / "memory.json"
    memory.save(path)
    strategy = AggressiveStrategy(
        WorldMemory.load(path), StrategyConfig(campaign_mode=True)
    )
    second = make_turn(
        objects=[core(position=HOME), unit(2, "RANGER", position=(794, 25))]
    )

    goal = strategy._campaign_scout_goal(second.rangers[0], second, context_for(second))

    assert goal != ORIGIN
    assert goal[0] > 784
    assert 0 <= goal[1] <= 31
    assert (
        next_step((794, 25), goal, blocked=memory.obstacles, require_path=True)
        is not None
    )


def test_scout_uses_detour_when_retained_route_becomes_contested():
    memory = scout_memory()
    memory.obstacles.remove((784, 15))
    memory.set_goal(object_id(2), UnitGoal((787, 15), 99, PURPOSE))
    strategy = AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))
    turn = make_turn(objects=[core(position=HOME), unit(2, "RANGER", position=ORIGIN)])
    assert strategy._campaign_scout_goal(turn.rangers[0], turn, context_for(turn)) == (
        787,
        15,
    )

    memory.contested_positions[(784, 15)] = turn.tick
    goal = strategy._campaign_scout_goal(turn.rangers[0], turn, context_for(turn))

    assert goal != ORIGIN
    assert goal != (787, 15)
    assert memory.goal_for(str(turn.rangers[0].id)).purpose == f"{PURPOSE}-detour"


def test_distant_scout_can_return_to_its_reachable_goal():
    memory = scout_memory()
    memory.set_goal(object_id(2), UnitGoal((787, 15), 99, PURPOSE))
    strategy = AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))
    turn = make_turn(
        objects=[core(position=HOME), unit(2, "RANGER", position=(875, 75))]
    )

    goal = strategy._campaign_scout_goal(turn.rangers[0], turn, context_for(turn))

    assert goal == (787, 15)


def test_scout_observation_planning_preserves_worker_harvest_and_deposit(monkeypatch):
    memory = scout_memory()
    workers = [
        unit(100 + index, "WORKER", position=(808 + index, 14)) for index in range(16)
    ]
    workers[0]["position"] = list(HOME)
    workers[0]["cargo"] = 1
    workers[1]["position"] = [810, 16]
    defenders = [
        unit(200 + index, "RANGER", position=(810 + index % 5, 18 + index // 5))
        for index in range(20)
    ]
    scouts = [
        unit(300 + index, "RANGER", position=(776 + index % 4, 8 + index // 4))
        for index in range(20)
    ]
    memory.unit_roles = {
        **{obj["id"]: "defense" for obj in defenders},
        **{obj["id"]: f"patrol-{index // 2 + 1}" for index, obj in enumerate(scouts)},
    }
    memory.unit_roles_initialized = True
    strategy = AggressiveStrategy(memory, StrategyConfig(campaign_mode=True))
    search_work = 0.0

    def costed_search(*args, **kwargs):
        nonlocal search_work
        # Model repeated full searches as planning work without a timing-sensitive
        # wall-clock assertion on this shared production/test host.
        search_work += 0.05
        return next_step(*args, **kwargs)

    monkeypatch.setattr(strategy_module, "next_step", costed_search)
    monkeypatch.setattr(strategy_module, "monotonic", lambda: search_work)
    positions = {
        obj["id"]: tuple(obj["position"]) for obj in [*workers, *defenders, *scouts]
    }
    actions = []
    for tick in range(100, 103):
        for obj in [*workers, *defenders, *scouts]:
            obj["position"] = list(positions[obj["id"]])
        turn = make_turn(
            tick=tick,
            resources=100,
            objects=[core(position=HOME), *workers, *defenders, *scouts],
            resource_cells=[(810, 16)],
        )
        report = strategy.decide(turn)
        assert not any(
            "planning time limit" in decision.reason for decision in report.decisions
        )
        assert memory.campaign_scout_target == TARGET
        worker_actions = {
            decision.actor_id: decision.action
            for decision in report.decisions
            if decision.actor_kind == "WORKER"
        }
        assert len(worker_actions) == 16
        actions.append(worker_actions)
        for uid, action in turn.plan.unit_actions.items():
            if action.type == "MOVE":
                positions[str(uid)] = add(positions[str(uid)], action.direction)
        if tick == 100:
            assert worker_actions[object_id(100)] == "DEPOSIT"
            assert worker_actions[object_id(101)] == "HARVEST"
            workers[0]["cargo"] = 0
            workers[1]["cargo"] = 1
    assert any(
        action == "MOVE" for tick_actions in actions for action in tick_actions.values()
    )
