"""Durable synchronous Arena Hero command loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Protocol

from arena_hero import APIError, ArenaHeroClient, ArenaHeroError, Turn, __version__

from .memory import WorldMemory
from .models import DecisionReport
from .strategy import AggressiveStrategy, StrategyConfig
from .telemetry import JsonlTelemetry

LOGGER = logging.getLogger(__name__)


class Tactic(Protocol):
    """A testable strategy that queues one complete Turn plan."""

    def decide(self, turn: Turn) -> DecisionReport:
        """Queue actions and explain them."""

        ...


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Connection, persistence, and one-session run options."""

    api_key: str
    data_dir: Path
    observe_only: bool = False
    max_turns: int | None = None
    base_url: str = "https://api.arenahero.io"
    websocket_url: str | None = None


def run_bot(
    config: RuntimeConfig,
    *,
    strategy_config: StrategyConfig | None = None,
) -> int:
    """Observe Turns, submit aggressive plans, and return the Turn count."""

    memory_path = config.data_dir / "memory.json"
    telemetry = JsonlTelemetry(config.data_dir / "turns.jsonl")
    memory = WorldMemory.load(memory_path)
    strategy = AggressiveStrategy(memory, strategy_config)
    turns_seen = 0
    mode = "observe-only" if config.observe_only else "aggressive-pvp"
    LOGGER.info(
        "starting mode=%s sdk=%s endpoint=%s",
        mode,
        __version__,
        config.base_url,
    )

    with ArenaHeroClient(
        api_key=config.api_key,
        base_url=config.base_url,
        websocket_url=config.websocket_url,
    ) as game:
        for turn in game.turns():
            turns_seen += 1
            started = monotonic()
            report, planning_error = _plan_turn(strategy, turn)
            planning_seconds = monotonic() - started
            submission = _submit_turn(turn, observe_only=config.observe_only)

            memory.save(memory_path)
            telemetry.append(
                _turn_record(
                    turn=turn,
                    report=report,
                    planning_seconds=planning_seconds,
                    planning_error=planning_error,
                    submission=submission,
                    observe_only=config.observe_only,
                )
            )
            LOGGER.info(
                "tick=%d enemies=%d actions=%d planning_ms=%.1f submit=%s",
                turn.tick,
                len(turn.visible_enemies),
                len(report.decisions),
                planning_seconds * 1000,
                submission["status"],
            )
            if config.max_turns is not None and turns_seen >= config.max_turns:
                break

    return turns_seen


def _plan_turn(strategy: Tactic, turn: Turn) -> tuple[DecisionReport, str | None]:
    try:
        return strategy.decide(turn), None
    except Exception as exc:
        LOGGER.exception(
            "strategy failed at tick=%d; falling back to an empty plan", turn.tick
        )
        turn.clear()
        return DecisionReport(tick=turn.tick), f"{type(exc).__name__}: {exc}"


def _submit_turn(turn: Turn, *, observe_only: bool) -> dict[str, Any]:
    if observe_only:
        return {"status": "observed"}
    try:
        accepted = turn.submit()
        return {
            "status": "accepted",
            "accepted": accepted.accepted,
            "tick": accepted.tick,
            "source": accepted.source.value,
            "received_at": accepted.received_at.isoformat(),
        }
    except APIError as exc:
        LOGGER.warning(
            "command rejected tick=%d status=%d error=%s",
            turn.tick,
            exc.status_code,
            exc.error,
        )
        return {
            "status": "rejected",
            "status_code": exc.status_code,
            "error": exc.error,
            "details": exc.details,
        }
    except ArenaHeroError as exc:
        LOGGER.warning("command failed tick=%d error=%s", turn.tick, type(exc).__name__)
        return {
            "status": "sdk_error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _turn_record(
    *,
    turn: Turn,
    report: DecisionReport,
    planning_seconds: float,
    planning_error: str | None,
    submission: dict[str, Any],
    observe_only: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tick": turn.tick,
        "mode": "observe-only" if observe_only else "aggressive-pvp",
        "planning_ms": round(planning_seconds * 1000, 3),
        "planning_error": planning_error,
        "state": turn.state.model_dump(mode="json"),
        "decision": report.to_dict(),
        "plan": turn.plan.model_dump(mode="json"),
        "submission": submission,
    }
