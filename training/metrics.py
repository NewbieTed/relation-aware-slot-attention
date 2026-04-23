from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(self, output_dir: Path, fieldnames: list[str]) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self.fieldnames = fieldnames

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, payload: dict[str, Any]) -> None:
        with self.jsonl_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        with self.csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow({key: payload.get(key, "") for key in self.fieldnames})


def write_split_manifest(
    output_dir: Path,
    *,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    eval_fraction: float,
    test_fraction: float,
) -> None:
    payload = {
        "seed": seed,
        "eval_fraction": eval_fraction,
        "test_fraction": test_fraction,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "test_count": len(test_rows),
        "train_files": [row["file_name"] for row in train_rows],
        "eval_files": [row["file_name"] for row in eval_rows],
        "test_files": [row["file_name"] for row in test_rows],
    }
    (output_dir / "data_split.json").write_text(json.dumps(payload, indent=2))
