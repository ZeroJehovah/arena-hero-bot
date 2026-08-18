"""Authoritative SDK model factories for tactic tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import UUID

from arena_hero import Accepted, PlayerState, Turn


def object_id(number: int) -> str:
    return str(UUID(int=number))


def core(
    number: int = 1,
    *,
    controlled: bool = True,
    position: tuple[int, int] = (0, 0),
    hp: int = 5,
    shield: int = 5,
    state: str = "NORMAL",
    owner_username: str = "player_one",
    **movement: Any,
) -> dict[str, Any]:
    return {
        "kind": "CORE",
        "id": object_id(number),
        "controlled": controlled,
        "owner_username": owner_username,
        "position": list(position),
        "hp": hp,
        "shield": shield,
        "state": state,
        **movement,
    }


def unit(
    number: int,
    unit_type: str,
    *,
    controlled: bool = True,
    position: tuple[int, int] = (0, 0),
    hp: int | None = None,
    cargo: int = 0,
) -> dict[str, Any]:
    maximum_hp = {"WORKER": 2, "VANGUARD": 4, "RANGER": 2}[unit_type]
    result: dict[str, Any] = {
        "kind": "UNIT",
        "id": object_id(number),
        "controlled": controlled,
        "position": list(position),
        "hp": maximum_hp if hp is None else hp,
        "unit_type": unit_type,
    }
    if controlled and unit_type == "WORKER":
        result["cargo"] = cargo
    return result


def make_turn(
    *,
    tick: int = 100,
    resources: int = 0,
    objects: Iterable[dict[str, Any]] = (),
    resource_cells: Iterable[tuple[int, int]] = (),
    obstacles: Iterable[tuple[int, int]] = (),
    beacon: dict[str, Any] | None = None,
    events: Iterable[dict[str, Any]] = (),
    status: str = "ACTIVE",
    respawn_at_tick: int | None = None,
) -> Turn:
    object_list = list(objects)
    resource_list = [list(position) for position in resource_cells]
    obstacle_list = [list(position) for position in obstacles]
    if resource_list:
        object_list.append({"kind": "RESOURCE", "positions": resource_list})
    if obstacle_list:
        object_list.append({"kind": "OBSTACLE", "positions": obstacle_list})
    population = sum(
        item["kind"] == "UNIT" and item["controlled"] for item in object_list
    )
    payload: dict[str, Any] = {
        "status": status,
        "resources": resources,
        "population": population,
        "champion_beacon": beacon or {"position": [100, 100]},
        "objects": object_list,
        "events": list(events),
    }
    if respawn_at_tick is not None:
        payload["respawn_at_tick"] = respawn_at_tick
    state = PlayerState.model_validate(payload)

    def submitter(plan, _key):
        return Accepted(
            accepted=True,
            tick=plan.tick,
            source="AGENT",
            received_at="2026-08-17T00:00:00Z",
        )

    return Turn(tick=tick, state=state, submitter=submitter)
