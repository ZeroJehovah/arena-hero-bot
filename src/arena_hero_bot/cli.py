"""Command-line entry point for continuous Arena Hero play."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from arena_hero import core_resource_capacity
from dotenv import load_dotenv

from .runtime import RuntimeConfig, run_bot
from .strategy import StrategyConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run the aggressive Arena Hero telemetry-first tactic."
    )
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="record authoritative Turns without submitting Agent plans",
    )
    parser.add_argument(
        "--max-turns",
        type=_positive_integer,
        help="stop after this many authoritative Turns",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("var"),
        help="memory and JSONL telemetry directory (default: var)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="dotenv file containing ARENA_HERO_API_KEY (default: .env)",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.arenahero.io",
        help="Arena Hero HTTP API base URL",
    )
    parser.add_argument("--websocket-url", help="override the game WebSocket URL")
    parser.add_argument(
        "--target-workers",
        type=_non_negative_integer,
        default=2,
        help="minimum Worker economy before combat production (default: 2)",
    )
    population_group = parser.add_mutually_exclusive_group()
    population_group.add_argument(
        "--max-population",
        type=_positive_integer,
        default=12,
        help="stop producing Units at this population (default: 12)",
    )
    population_group.add_argument(
        "--no-max-population",
        action="store_true",
        help="remove the explicit population cap for safety-first growth",
    )
    parser.add_argument(
        "--resource-target",
        type=_non_negative_integer,
        default=0,
        help=(
            "legacy fixed Core reserve target "
            "(0 disables it; live mode leaves it unset)"
        ),
    )
    parser.add_argument(
        "--expedition-mode",
        action="store_true",
        help=(
            "active-offense posture: saturate a fixed defensive formation, then "
            "send surplus units out on independent non-returning expeditions"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Load configuration and run until interrupted or ``--max-turns``."""

    parser = build_parser()
    args = parser.parse_args(argv)
    max_population = None if args.no_max_population else args.max_population
    maximum_capacity = (
        core_resource_capacity(max_population) if max_population is not None else None
    )
    if maximum_capacity is not None and args.resource_target > maximum_capacity:
        parser.error(
            f"--resource-target cannot exceed the maximum Core capacity "
            f"of {maximum_capacity}"
        )
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    load_dotenv(dotenv_path=args.env_file, override=False)
    api_key = os.environ.get("ARENA_HERO_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            f"ARENA_HERO_API_KEY is required in the environment or {args.env_file}"
        )

    runtime = RuntimeConfig(
        api_key=api_key,
        data_dir=args.data_dir,
        observe_only=args.observe_only,
        max_turns=args.max_turns,
        base_url=args.base_url,
        websocket_url=args.websocket_url,
    )
    strategy = StrategyConfig(
        target_workers=args.target_workers,
        max_population=max_population,
        resource_target=args.resource_target,
        expedition_mode=args.expedition_mode,
    )
    try:
        run_bot(runtime, strategy_config=strategy)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("stopped by user")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed
