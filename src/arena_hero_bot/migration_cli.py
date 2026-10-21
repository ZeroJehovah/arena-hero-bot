"""Operator CLI for planning and requesting long Core migrations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .memory import WorldMemory
from .migration import (
    load_migration_request,
    plan_next_resource_tier,
    resource_block_index,
    resource_block_quota,
    resource_block_ring,
    write_migration_request,
)

MIGRATION_REQUEST_NAME = "core-migration.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, inspect, or request a durable Core migration."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("var"),
        help="runtime data directory containing memory.json (default: var)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="recommend the next natural-resource tier")
    next_tier = commands.add_parser(
        "request-next-tier",
        help="request the cleanest nearby block at the next quota tier",
    )
    next_tier.add_argument("--note", default="next natural-resource tier")
    request = commands.add_parser("request", help="request an explicit center cell")
    request.add_argument("x", type=int)
    request.add_argument("y", type=int)
    request.add_argument("--note", default="operator migration request")
    commands.add_parser("status", help="show the persisted and requested targets")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    memory = WorldMemory.load(args.data_dir / "memory.json")
    request_path = args.data_dir / MIGRATION_REQUEST_NAME

    if args.command == "plan":
        plan = plan_next_resource_tier(memory)
        if plan is None:
            raise SystemExit("no inner resource tier is available")
        print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "request-next-tier":
        plan = plan_next_resource_tier(memory)
        if plan is None:
            raise SystemExit("no inner resource tier is available")
        target = plan.target
        note = args.note
    elif args.command == "request":
        target = args.x, args.y
        note = args.note
        if target in memory.obstacles or target in memory.resource_cells:
            raise SystemExit("requested target is a remembered obstacle or resource")
        plan = None
    else:
        request = load_migration_request(request_path)
        print(
            json.dumps(
                {
                    "persisted_home": (
                        list(memory.core_home_position)
                        if memory.core_home_position is not None
                        else None
                    ),
                    "request_file": str(request_path),
                    "requested_target": (
                        list(request.target) if request is not None else None
                    ),
                    "requested_at": request.requested_at
                    if request is not None
                    else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    block = resource_block_index(target)
    request = write_migration_request(request_path, target, note=note)
    result: dict[str, object] = {
        "request_file": str(request_path),
        "target": list(request.target),
        "block": list(block),
        "ring": resource_block_ring(block),
        "quota": resource_block_quota(block),
        "requested_at": request.requested_at,
    }
    if plan is not None:
        result["estimated_cells"] = plan.estimated_cells
        result["block_rocks"] = plan.block_rocks
        result["blocked_defense_posts"] = plan.blocked_defense_posts
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
