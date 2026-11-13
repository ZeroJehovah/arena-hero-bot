"""Independent player fog, deterministic responses and settlement fixtures."""

from collections import Counter
from copy import deepcopy

from arena_hero import Direction, UnitView, unit_cost

from arena_hero_bot.exploration import visible_cells
from arena_hero_bot.geometry import add, adjacent_positions, line_of_fire
from arena_hero_bot.memory import WorldMemory
from arena_hero_bot.strategy import AggressiveStrategy, StrategyConfig

from .factories import core, make_turn, object_id, unit
from .test_layered_defense import Encounter


class FogEncounter(Encounter):
    """The opponent receives only its own view, unless explicitly scripted."""

    def __init__(self, friends, foes, *, resources=0, rocks=(), home=(0, 0)):
        super().__init__(friends, foes, resources=resources, rocks=rocks)
        self.core = core(position=home)
        self.events = []
        self.event_serial = 90000
        self.opening_tick = None
        self.trace = []
        self.friendly_ids = set(self.friends)
        self.hostile_ids = set(self.foes)
        self.spawns = 0
        self.healing_cost = 0
        self.friendly_damage = 0
        self.enemy_damage = 0

    def view(self, *, enemy=False):
        own = self.foes if enemy else self.friends
        other = self.friends if enemy else self.foes
        observers = [
            (
                tuple(actor["position"]),
                {"RANGER": 5, "VANGUARD": 4, "WORKER": 3}[actor["unit_type"]],
            )
            for actor in own.values()
        ]
        if not enemy:
            observers.append((tuple(self.core["position"]), 5))
        seen = frozenset().union(
            *(
                visible_cells(position, radius, self.rocks)
                for position, radius in observers
            )
        )
        objects = [dict(deepcopy(actor), controlled=True) for actor in own.values()]
        objects.extend(
            dict(deepcopy(actor), controlled=False)
            for actor in other.values()
            if tuple(actor["position"]) in seen
        )
        if not enemy or tuple(self.core["position"]) in seen:
            objects.append(dict(deepcopy(self.core), controlled=not enemy))
        events = [
            event
            for event in self.events
            if (
                event["event_type"] != "SHOT_HIT"
                and str(event.get("target_id"))
                in (self.hostile_ids if enemy else self.friendly_ids)
            )
            or str(event.get("actor_id"))
            in (self.hostile_ids if enemy else self.friendly_ids)
            or (
                event["event_type"] == "UNIT_DESTROYED"
                and tuple(event["position"]) in seen
            )
        ]
        return make_turn(
            tick=self.tick,
            resources=self.resources if not enemy else 0,
            objects=objects,
            obstacles=self.rocks & seen,
            events=events,
        )

    def turn(self):
        return self.view()

    def response(self, enemy_view, *, mode="infer_after_hit", member_ids=()):
        """Pressure scripts are explicit; fair policies never inspect hidden actors."""
        orders = {}
        if mode == "precise_pressure":
            # This intentionally stronger response is NOT a fair-fog policy.
            victims = [
                actor for key, actor in self.friends.items() if key in member_ids
            ]
            victims.sort(key=lambda actor: (actor["hp"], actor["id"]))
            for ranger in enemy_view.rangers:
                target = next(
                    (
                        actor
                        for actor in victims
                        if line_of_fire(
                            ranger.position, tuple(actor["position"]), self.rocks
                        )
                    ),
                    None,
                )
                if target is not None:
                    orders[str(ranger.id)] = {
                        "type": "SHOOT",
                        "target_id": target["id"],
                        "expected_cell": target["position"],
                    }
            return orders
        received_hit = any(
            event.event_type == "UNIT_DAMAGED" for event in enemy_view.events
        )
        for ranger in enemy_view.rangers:
            visible = [
                enemy
                for enemy in enemy_view.visible_enemies
                if isinstance(enemy, UnitView)
            ]
            shootable = [
                enemy
                for enemy in visible
                if line_of_fire(ranger.position, enemy.position, self.rocks)
            ]
            if shootable:
                target = min(shootable, key=lambda enemy: (enemy.hp, enemy.id))
                orders[str(ranger.id)] = {
                    "type": "SHOOT",
                    "target_id": str(target.id),
                    "expected_cell": list(target.position),
                }
            elif received_hit and mode == "infer_after_hit":
                # One legal guess based on the attacked Ranger's own cell,
                # not the hidden actor's coordinates or health.
                cell = (ranger.position[0] + 3, ranger.position[1] + 3)
                orders[str(ranger.id)] = {"type": "SHOOT", "expected_cell": list(cell)}
            elif visible:
                front = min(visible, key=lambda enemy: enemy.id)
                cells = [
                    cell
                    for cell in adjacent_positions(front.position)
                    if line_of_fire(ranger.position, cell, self.rocks)
                ]
                if cells:
                    orders[str(ranger.id)] = {
                        "type": "SHOOT",
                        "expected_cell": list(min(cells)),
                    }
        return orders

    def settle(self, plan, enemy_actions):
        before = {**deepcopy(self.friends), **deepcopy(self.foes)}
        old_resources = self.resources
        old_heals = self.heals
        commands = {
            str(key): action.model_dump() for key, action in plan.unit_actions.items()
        } | enemy_actions
        final_positions = {
            key: tuple(actor["position"]) for key, actor in before.items()
        }
        for key, action in commands.items():
            if key in final_positions and action["type"] == "MOVE":
                final_positions[key] = add(
                    final_positions[key], Direction(action["direction"])
                )
        self.trace.append(
            {
                "tick": self.tick,
                "plan": plan.model_dump(mode="json"),
                "enemy": deepcopy(enemy_actions),
            }
        )
        super().settle(plan, enemy_actions)
        self.healing_cost += self.heals - old_heals
        spawned = None
        if plan.core_action is not None and plan.core_action.type == "SPAWN":
            kind = plan.core_action.unit_type
            cost = unit_cost(kind, len(self.friends))
            if self.resources >= cost:
                spawned = unit(
                    200000 + self.spawns,
                    kind.value,
                    position=tuple(self.core["position"]),
                )
                self.friends[spawned["id"]] = spawned
                self.friendly_ids.add(spawned["id"])
                self.resources -= cost
                self.spawns += 1
        after = self.friends | self.foes
        self.events = []

        def event(event_type, **fields):
            self.event_serial += 1
            self.events.append(
                {
                    "event_id": object_id(self.event_serial),
                    "tick": self.tick - 1,
                    "event_type": event_type,
                    **fields,
                }
            )

        for key, actor in before.items():
            action = commands.get(key, {})
            if (
                key in self.friends
                and action.get("type") == "DEPOSIT"
                and actor.get("cargo", 0) > 0
                and self.friends[key].get("cargo") == 0
            ):
                event(
                    "DEPOSIT_SUCCEEDED",
                    actor_id=key,
                    target_id=self.core["id"],
                    position=self.core["position"],
                    values={"amount": actor["cargo"]},
                )
            if (
                key in self.friends
                and action.get("type") == "HEAL"
                and self.friends[key]["hp"] > actor["hp"]
            ):
                amount = self.friends[key]["hp"] - actor["hp"]
                event(
                    "UNIT_HEAL_SUCCEEDED",
                    actor_id=key,
                    target_id=key,
                    position=self.friends[key]["position"],
                    values={
                        "amount": amount,
                        "cost": amount,
                        "hp": self.friends[key]["hp"],
                    },
                )
            if final_positions[key] != tuple(actor["position"]):
                event(
                    "UNIT_MOVE_SUCCEEDED",
                    actor_id=key,
                    position=list(final_positions[key]),
                )
            hp = after[key]["hp"] if key in after else 0
            if hp < actor["hp"]:
                sources = [
                    source
                    for source, action in commands.items()
                    if source in before
                    and action["type"] == "SHOOT"
                    and (not action.get("target_id") or str(action["target_id"]) == key)
                    and tuple(action["expected_cell"]) == final_positions[key]
                    and line_of_fire(
                        final_positions[source], final_positions[key], self.rocks
                    )
                ]
                damage = len(sources) or actor["hp"] - hp
                if key in self.friendly_ids:
                    self.friendly_damage += damage
                else:
                    self.enemy_damage += damage
                # Production emits per-arrow hits and a separate aggregate
                # damage event whose actor is unknown. Preserve that split.
                for source in sources:
                    event(
                        "SHOT_HIT",
                        actor_id=source,
                        target_id=key,
                        position=list(final_positions[key]),
                        values={"damage": 1},
                    )
                event(
                    "UNIT_DAMAGED",
                    actor_id=None,
                    target_id=key,
                    position=list(final_positions[key]),
                    reason_code="ATTACK",
                    values={"damage": damage, "hp": hp},
                )
            if key not in after:
                event(
                    "UNIT_DESTROYED", target_id=key, position=list(final_positions[key])
                )
        if spawned is not None:
            event(
                "CORE_SPAWN_SUCCEEDED",
                actor_id=self.core["id"],
                target_id=spawned["id"],
                position=self.core["position"],
                values={
                    "unit_type": spawned["unit_type"],
                    "cost": old_resources - self.resources,
                },
            )
        counts = Counter(tuple(actor["position"]) for actor in after.values())
        counts[tuple(self.core["position"])] += 1
        assert max(counts.values(), default=0) <= 2


def standoff(*, side_count=2, home=(0, 0), opponents=2, target_hp=2):
    offsets = [(3, -3), (4, -4), (-3, -3), (-4, -4)]
    friends = [
        unit(index + 2, "RANGER", position=(home[0] + dx, home[1] + dy))
        for index, (dx, dy) in enumerate(offsets[:side_count])
    ]
    friends.extend(
        unit(20 + index, "RANGER", position=(home[0] + dx, home[1] + dy))
        for index, (dx, dy) in enumerate([(0, -3), (-3, 2), (3, 2), (0, 3)])
    )
    foes = [
        unit(
            50 + index,
            "RANGER",
            controlled=False,
            position=(home[0], home[1] - 7),
            hp=target_hp,
        )
        for index in range(opponents)
    ]
    roles = {
        actor["id"]: "defense" if index < side_count + 1 else "patrol-1"
        for index, actor in enumerate(friends)
    }
    memory = WorldMemory(
        core_home_position=home,
        core_layout_version=3,
        unit_roles=roles,
        unit_roles_initialized=True,
    )
    return AggressiveStrategy(memory, StrategyConfig(campaign_mode=True)), FogEncounter(
        friends, foes, home=home
    )


def seed_contact(strategy, encounter, *, count=7):
    original_tick = encounter.tick
    for tick in range(original_tick - count, original_tick):
        encounter.tick = tick
        turn = encounter.turn()
        strategy.memory.observe(turn)
        strategy._observe_flank_contacts(turn)
    encounter.tick = original_tick
