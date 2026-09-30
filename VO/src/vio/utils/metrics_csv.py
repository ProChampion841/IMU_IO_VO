"""One-row-per-epoch CSV of training metrics, for plotting and comparison."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence


class EpochMetricsCsv:
    """Append one row per epoch with a fixed column set.

    The header is written when the file is created; on resume the existing
    file is appended to. A value missing from ``row`` is written empty rather
    than raising, so a partial epoch never corrupts the file.
    """

    def __init__(self, path: "str | Path", columns: Sequence[str]) -> None:
        self.path = Path(path)
        self.columns = tuple(columns)
        if not self.columns:
            raise ValueError("EpochMetricsCsv needs at least one column")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(self.columns)

    def append(self, row: Mapping[str, object]) -> None:
        values = []
        for column in self.columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(value)
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(values)
            handle.flush()


def open_metrics_csv(path: "str | Path", columns: Sequence[str]) -> EpochMetricsCsv:
    """Open metrics.csv, retiring one that was written with other columns.

    :class:`EpochMetricsCsv` writes its header only when the file is new and
    thereafter appends by position, so a run resumed after the column set
    changed would keep the old header above rows in the new order: every
    column silently mislabelled, with nothing in the file to say so. The old
    rows are real measurements, so they are moved aside rather than
    overwritten, and the move is announced.
    """

    path = Path(path)
    if path.is_file() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if tuple(header) != tuple(columns):
            retired = path.with_name(f"{path.stem}.{len(header)}col{path.suffix}")
            index = 1
            while retired.exists():
                retired = path.with_name(
                    f"{path.stem}.{len(header)}col.{index}{path.suffix}"
                )
                index += 1
            path.rename(retired)
            print(
                f"{path.name} was written with a different column set; kept it "
                f"as {retired.name} and started a new one"
            )
    return EpochMetricsCsv(path, columns)


__all__ = ["EpochMetricsCsv", "open_metrics_csv"]
