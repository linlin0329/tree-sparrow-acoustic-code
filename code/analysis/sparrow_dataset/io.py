"""Shared UTF-8 CSV, identity and package-path operations."""

from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    require(
        bool(relative)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in relative,
        f"Unsafe relative path: {relative!r}",
    )
    target = root / path
    require(
        target.resolve().is_relative_to(root.resolve()),
        f"Path escapes package: {relative}",
    )
    return target


def indexed(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows) and "" not in result, f"Duplicate/empty {key}")
    return result


def write_rows(path: Path, rows: list[dict[str, str]], *, delimiter: str = ",") -> None:
    require(bool(rows), "Cannot infer columns for an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter=delimiter, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
