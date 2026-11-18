"""Build per-observation road-candidate feature tables for both development splits.

Example:
python scripts/build_candidate_dataset.py --data-root data/raw --output-dir data/processed

The training and student-test outputs are written separately. The student-test
labels are for development evaluation only and must never be appended to training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pathsnap.candidates import generate_candidates, load_trip_folder


def trip_folders(split_dir: Path):
    return sorted(p for p in split_dir.iterdir() if p.is_dir())


def build_split(split_dir: Path, split_name: str, candidate_k: int,
                observation_stride: int, max_trips: int | None):
    candidate_parts, observation_parts = [], []
    trips = trip_folders(split_dir)
    if max_trips is not None:
        trips = trips[:max_trips]
    for index, folder in enumerate(trips, start=1):
        track, nodes, arcs, route = load_trip_folder(folder)
        candidates, observations = generate_candidates(
            track, nodes, arcs, route, trip_id=folder.name,
            candidate_k=candidate_k, observation_stride=observation_stride,
        )
        candidates["partition"] = split_name
        observations["partition"] = split_name
        candidate_parts.append(candidates)
        observation_parts.append(observations)
        miss_rate = float(observations.candidate_miss.mean()) if len(observations) else float("nan")
        print(
            f"[{split_name}] {index}/{len(trips)} trip {folder.name}: "
            f"{len(observations)} observations, {len(candidates)} candidates, "
            f"candidate miss rate={miss_rate:.3f}"
        )
    if not candidate_parts:
        raise ValueError(f"No trip folders found in {split_dir}")
    return pd.concat(candidate_parts, ignore_index=True), pd.concat(observation_parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--student-test-stride", type=int, default=5)
    parser.add_argument("--max-trips", type=int, default=None,
                        help="Optional smoke-test limit per split.")
    args = parser.parse_args()

    if args.candidate_k < 1 or args.train_stride < 1 or args.student_test_stride < 1:
        parser.error("candidate-k and both strides must be positive integers.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_candidates, train_observations = build_split(
        args.data_root / "train", "train", args.candidate_k,
        args.train_stride, args.max_trips,
    )
    test_candidates, test_observations = build_split(
        args.data_root / "student_test", "student_test", args.candidate_k,
        args.student_test_stride, args.max_trips,
    )

    outputs = {
        "train_candidates": "train_candidates.csv.gz",
        "train_observations": "train_observations.csv.gz",
        "student_test_candidates": "student_test_candidates.csv.gz",
        "student_test_observations": "student_test_observations.csv.gz",
    }
    tables = {
        "train_candidates": train_candidates,
        "train_observations": train_observations,
        "student_test_candidates": test_candidates,
        "student_test_observations": test_observations,
    }
    for key, frame in tables.items():
        path = args.output_dir / outputs[key]
        frame.to_csv(path, index=False, compression="gzip")
        print(f"Saved {len(frame)} rows to {path}")

    metadata = {
        "candidate_k": args.candidate_k,
        "train_observation_stride": args.train_stride,
        "student_test_observation_stride": args.student_test_stride,
        "label_definition": (
            "Nearest road arc among unique .route arc IDs by projected point-to-segment "
            "distance; this is a spatial proxy, not timestamp-aligned ground truth."
        ),
        "feature_columns": [
            "distance_to_segment_m", "midpoint_distance_m", "fraction_along_segment",
            "segment_length_m", "heading_difference_rad", "gps_speed_mps",
            "time_gap_s", "from_node_out_degree", "to_node_in_degree",
        ],
        "rows": {key: len(value) for key, value in tables.items()},
        "candidate_miss_rate_train": float(train_observations.candidate_miss.mean()),
        "candidate_miss_rate_student_test": float(test_observations.candidate_miss.mean()),
    }
    meta_path = args.output_dir / "candidate_generation_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
