"""Minimal training logger: console + CSV, no external dependency required."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path


class CSVLogger:
    def __init__(self, log_dir: str | os.PathLike, filename: str = "metrics.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / filename
        self._fields: list[str] | None = None
        self._start = time.time()

    def log(self, row: dict) -> None:
        row = {"elapsed_sec": round(time.time() - self._start, 2), **row}
        write_header = not self.path.exists()
        if self._fields is None:
            self._fields = list(row.keys())
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def __repr__(self) -> str:
        return f"CSVLogger({self.path})"


def print_metrics(prefix: str, metrics: dict) -> None:
    parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    print(f"[{prefix}] " + "  ".join(parts))
