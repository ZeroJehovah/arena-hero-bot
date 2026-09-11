"""Append-only telemetry for replaying tactical decisions."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

BEIJING_TZ = timezone(timedelta(hours=8))
RETAIN_TELEMETRY_DAYS = 4  # 今日 + 前 3 个北京自然日
_TIMESTAMP_LEN = 19  # second-resolution UTC received_at prefix length


class JsonlTelemetry:
    """Write one complete, secret-free record per observed Turn."""

    def __init__(self, path: Path, *, rotate_daily: bool = False) -> None:
        self.path = path
        self.rotate_daily = rotate_daily

    def append(self, record: dict[str, Any]) -> None:
        """Append and flush one JSON object."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        target_path = (
            self._daily_path(datetime.now(BEIJING_TZ).date())
            if self.rotate_daily
            else self.path
        )
        with target_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()

    def retained_since(
        self, *, days: int = RETAIN_TELEMETRY_DAYS, tz: timezone = BEIJING_TZ
    ) -> datetime:
        """Return the UTC instant opening the retained ``days``-day window.

        The window runs from the start of ``days - 1`` natural days agot
        (in ``tz``) through now; ``days=4`` keeps today plus the previous
        three natural days."""

        if not 1 <= days <= 60:
            raise ValueError("retention days must be in 1..60")
        today = datetime.now(tz=tz)
        midnight = today.replace(hour=0, minute=0, second=0, microsecond=0)
        return (midnight - timedelta(days=days - 1)).astimezone(UTC)

    def prune_recent(
        self, *, days: int = RETAIN_TELEMETRY_DAYS, tz: timezone = BEIJING_TZ
    ) -> bool:
        """Rewrite the telemetry file, keeping only the most recent Beijing
        natural-day records (see ``retained_since``).

        Records that lack a submission timestamp cannot be dated and are kept.
        When every retained line already falls inside the window the file is left
        untouched.  Returns True iff the file was rewritten."""

        if self.rotate_daily:
            return self._prune_daily_files(days=days, tz=tz)
        if not self.path.exists():
            return False
        boundary_prefix = self._boundary_prefix(days=days, tz=tz)
        earliest = self._earliest_received_prefix()
        if earliest is not None and earliest >= boundary_prefix:
            return False
        temporary: str | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=".turns-prune-",
                delete=False,
            ) as target:
                temporary = target.name
                with self.path.open("rb") as source:
                    for line in source:
                        prefix = self._received_prefix(line)
                        if prefix is None or prefix >= boundary_prefix:
                            target.write(line)
                target.flush()
                os.fsync(target.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)
        return True

    def _daily_path(self, day: date) -> Path:
        filename = f"{self.path.stem}-{day.isoformat()}{self.path.suffix}"
        return self.path.with_name(filename)

    def _prune_daily_files(self, *, days: int, tz: timezone) -> bool:
        boundary = self.retained_since(days=days, tz=tz).astimezone(tz).date()
        changed = False
        prefix = f"{self.path.stem}-"
        pattern = f"{self.path.stem}-????????{self.path.suffix}"
        for candidate in self.path.parent.glob(pattern):
            name = candidate.name
            if not name.startswith(prefix):
                continue
            try:
                day = date.fromisoformat(name[len(prefix) : -len(self.path.suffix)])
            except ValueError:
                continue
            if day < boundary:
                candidate.unlink()
                changed = True
        return changed

    def _boundary_prefix(
        self, *, days: int = RETAIN_TELEMETRY_DAYS, tz: timezone = BEIJING_TZ
    ) -> str:
        """Render the retention boundary as a UTC second-resolution ISO prefix."""

        return self.retained_since(days=days, tz=tz).strftime("%Y-%m-%dT%H:%M:%S")

    def _received_prefix(self, line: bytes) -> str | None:
        """Extract the record's received_at UTC second-resolution prefix, or None."""

        marker = b'"received_at":"'
        at = line.find(marker)
        if at < 0:
            return None
        start = at + len(marker)
        prefix = line[start : start + _TIMESTAMP_LEN].decode("ascii", errors="ignore")
        if len(prefix) != _TIMESTAMP_LEN or not all(
            ch.isdigit() or ch in "-T:" for ch in prefix
        ):
            return None
        return prefix

    def _earliest_received_prefix(self, *, limit: int = 256) -> str | None:
        """Return the first datable record near the chronological head without
        scanning the whole file."""

        with self.path.open("rb") as source:
            for index, line in enumerate(source):
                if index >= limit:
                    break
                prefix = self._received_prefix(line)
                if prefix is not None:
                    return prefix
        return None
