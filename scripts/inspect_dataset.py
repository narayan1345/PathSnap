"""Print a read-only inventory of the PathSnap dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathsnap.data_io import discover_trips, trip_file_inventory


def inspect_split(split_dir: Path) -> dict:
    trips = discover_trips(split_dir)
    missing_by_kind = {kind: 0 for kind in ("track", "nodes", "arcs", "route")}
    incomplete = []
    for trip in trips:
        inventory = trip_file_inventory(trip)
        missing = [kind for kind, path in inventory.items() if path is None]
        for kind in missing:
            missing_by_kind[kind] += 1
        if missing:
            incomplete.append({"trip_id": trip.name, "missing": missing})
    return {
        "directory": str(split_dir),
        "trip_count": len(trips),
        "file_count_by_kind": {
            kind: len(trips) - missing_by_kind[kind]
            for kind in missing_by_kind
        },
        "missing_file_count_by_kind": missing_by_kind,
        "incomplete_trips": incomplete,
        "sample_trip_ids": [trip.name for trip in trips[:5]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    report = {}
    for split in ("train", "student_test"):
        split_dir = args.data_root / split
        if split_dir.exists():
            report[split] = inspect_split(split_dir)
        else:
            report[split] = {"directory": str(split_dir), "error": "directory not found"}

    print(json.dumps(report, indent=2))
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_inventory.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved inventory to {output_path}")


if __name__ == "__main__":
    main()
