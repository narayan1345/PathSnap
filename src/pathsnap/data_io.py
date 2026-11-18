"""Dataset discovery and conservative readers for PathSnap.

The exact source-file schema should be confirmed from the assignment data before
using these readers in model code. This module deliberately avoids assuming
column names or silently coercing malformed rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

EXPECTED_SUFFIXES = (".track", ".nodes", ".arcs", ".route")


def discover_trips(split_dir: Path) -> list[Path]:
    """Return trip directories containing at least one expected data file."""
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split directory does not exist: {split_dir}")
    if not split_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {split_dir}")
    return sorted(
        p for p in split_dir.iterdir()
        if p.is_dir() and any((p / f"{p.name}{suffix}").is_file()
                              for suffix in EXPECTED_SUFFIXES)
    )


def trip_file_inventory(trip_dir: Path) -> Dict[str, Path | None]:
    """Map expected file kinds to paths, preserving missing files as None."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    return {
        suffix.lstrip("."): (
            trip_dir / f"{trip_id}{suffix}"
            if (trip_dir / f"{trip_id}{suffix}").is_file()
            else None
        )
        for suffix in EXPECTED_SUFFIXES
    }
