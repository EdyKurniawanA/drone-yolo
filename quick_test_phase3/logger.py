# quick_test_phase3/logger.py
import csv
import os
import json
import datetime
import time
from typing import List, Optional


class Logger:
    """
    Flexible CSV logger:
    - Accepts ANY kwargs in .log(**kwargs)
    - Writes a stable set of columns; anything unknown goes into 'extra' as JSON
    - Auto-fills timestamp and system_uptime if not provided
    - Serializes dicts (e.g., counts) to JSON strings
    """

    def __init__(self, filepath: str, fieldnames: Optional[List[str]] = None):
        # Default/stable columns for easy analysis
        if fieldnames is None:
            fieldnames = [
                "timestamp",
                "system_uptime",
                "frame_id",
                "fps",
                "counts",
                "event",
                "confidence",
                "note",
                "extra",  # JSON blob for any additional fields you pass later
            ]

        self.filepath = filepath
        self.fieldnames = fieldnames

        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        # Open once and reuse (flush on every write)
        self.file = open(self.filepath, mode="a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        # Write header if file is empty
        if os.stat(self.filepath).st_size == 0:
            self.writer.writeheader()

    def log(self, **kwargs) -> None:
        """
        Log a row with flexible fields.

        Known fields go to their columns.
        Unknown fields are packed into 'extra' (JSON).
        """
        # Auto-fill timestamp/system_uptime if missing
        kwargs.setdefault("timestamp", datetime.datetime.now().isoformat())
        kwargs.setdefault("system_uptime", time.perf_counter())

        # Serialize dict-like fields to JSON (e.g., counts)
        if "counts" in kwargs and isinstance(kwargs["counts"], dict):
            kwargs["counts"] = json.dumps(kwargs["counts"], ensure_ascii=False)

        # Build row for known columns
        row = {k: kwargs.get(k, "") for k in self.fieldnames}

        # Anything not in header → 'extra' JSON
        extras = {k: v for k, v in kwargs.items() if k not in self.fieldnames}
        if extras:
            # Merge with existing 'extra' if present
            try:
                existing = json.loads(row.get("extra") or "{}")
            except Exception:
                existing = {}
            existing.update(extras)
            row["extra"] = json.dumps(existing, ensure_ascii=False)

        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        try:
            if not self.file.closed:
                self.file.close()
        except Exception:
            pass
