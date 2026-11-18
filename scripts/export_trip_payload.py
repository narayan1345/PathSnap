"""Export a trip's track and road graph as a JSON payload for the local API.

Example:
python scripts/export_trip_payload.py --trip-dir data/raw/train/00000001 --output data/processed/trip_00000001_api.json

The .route reference labels are intentionally not exported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    folder = args.trip_dir
    trip_id = folder.name
    track = pd.read_csv(
        folder / f"{trip_id}.track", sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"],
    )
    nodes = pd.read_csv(
        folder / f"{trip_id}.nodes", sep=r"\s+", header=None,
        names=["longitude", "latitude"],
    )
    arcs = pd.read_csv(
        folder / f"{trip_id}.arcs", sep=r"\s+", header=None,
        names=["from_node_id", "to_node_id"],
    )
    if not track.timestamp.is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted before API export.")
    if (arcs.from_node_id.max() >= len(nodes) or
            arcs.to_node_id.max() >= len(nodes)):
        raise ValueError("An arc endpoint is outside the node table.")

    payload = {
        "trip_id": trip_id,
        "track": track.to_dict(orient="records"),
        "nodes": nodes.to_dict(orient="records"),
        "arcs": arcs.to_dict(orient="records"),
        "model_name": "extratrees",
        "candidate_k": 8,
        "apply_hmm": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(track)} GPS points, {len(nodes)} nodes and "
          f"{len(arcs)} arcs to {args.output}")
    print("Reference .route was not included in the API request.")


if __name__ == "__main__":
    main()
