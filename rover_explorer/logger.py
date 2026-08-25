from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class SessionLogger:
    def __init__(self, session_dir: str | Path):
        self.root = Path(session_dir)
        self.raw_dir = self.root / "raw"
        self.annotated_dir = self.root / "annotated"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "cycles.jsonl"

    def log_cycle(
        self,
        cycle: int,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        **record: Any,
    ) -> dict[str, Any]:
        raw_name = f"{cycle:06d}.jpg"
        annotated_name = f"{cycle:06d}.jpg"
        cv2.imwrite(str(self.raw_dir / raw_name), raw_frame)
        cv2.imwrite(str(self.annotated_dir / annotated_name), annotated_frame)
        complete = {
            "timestamp": time.time(),
            "cycle": cycle,
            "raw_frame": f"raw/{raw_name}",
            "annotated_frame": f"annotated/{annotated_name}",
            **record,
        }
        complete = _jsonable(complete)
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(complete, separators=(",", ":")) + "\n")
        (self.root / "latest.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
        return complete

