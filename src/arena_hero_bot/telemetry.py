"""Append-only telemetry for replaying tactical decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlTelemetry:
    """Write one complete, secret-free record per observed Turn."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: dict[str, Any]) -> None:
        """Append and flush one JSON object."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
