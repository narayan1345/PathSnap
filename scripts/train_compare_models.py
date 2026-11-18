"""Train five candidate classifiers and compare them with trip-group CV.

Run after scripts/build_candidate_dataset.py:
python scripts/train_compare_models.py --processed-dir data/processed --artifacts-dir artifacts --results-dir results

Labels are spatial nearest-reference-arc proxies as documented in candidates.py.
This is an experimental baseline; report the proxy-label limitation explicitly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from pathsnap.candidates import FEATURE_COLUMNS, endpoints_for_trip, load_trip_folder


SEED = 42


def make_models():
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED),
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=16, min_samples_leaf=5, class_weight="balanced", random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=80, max_depth=16, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=80, max_depth=16, min_samples_leaf=3,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEED,
        ),
    }


def fit_model(model, x, y):
    # Balanced sample weights are applied to tree/boosting estimators that do
    # not expose class_weight in a consistent way. Logistic regression has it
    # directly in the estimator configuration.
    if hasattr(model, "named_steps"):  # sklearn Pipeline (Logistic Regression)
        model.fit(x, y)
    else:
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(x, y, sample_weight=weights)


def evaluate_model(model, candidates: pd.DataFrame, observations: pd.DataFrame,
                   endpoints_by_trip: dict[str, dict[int, tuple[int, int]]]):
    x = candidates[FEATURE_COLUMNS].to_numpy(dtype=float)
    start = time.perf_counter()
    probabilities = model.predict_proba(x)[:, 1]
    inference_seconds = time.perf_counter() - start

    scored = candidates[[
        "observation_key", "trip_id", "observation_index", "arc_id", "target"
    ]].copy()
    scored["positive_probability"] = probabilities
    top_indices = scored.groupby("observation_key", sort=False)[
        "positive_probability"
    ].idxmax()
    top = scored.loc[top_indices].merge(
        observations[[
            "observation_key", "target_arc_id", "candidate_miss"
        ]],
        on="observation_key", how="left", validate="one_to_one",
    )
    top["correct"] = top.arc_id == top.target_arc_id
    top1_accuracy = float(top.correct.mean()) if len(top) else float("nan")

    per_trip_macro_f1 = []
    for _, part in top.groupby("trip_id"):
        labels = np.unique(np.concatenate([
            part.target_arc_id.to_numpy(), part.arc_id.to_numpy()
        ]))
        per_trip_macro_f1.append(float(f1_score(
            part.target_arc_id, part.arc_id, labels=labels,
            average="macro", zero_division=0,
        )))
    macro_f1 = float(np.mean(per_trip_macro_f1)) if per_trip_macro_f1 else float("nan")

    candidate_truth = candidates.target.to_numpy(dtype=int)
    candidate_prediction = (probabilities >= 0.5).astype(int)
    candidate_log_loss = float(log_loss(candidate_truth, probabilities, labels=[0, 1]))
    candidate_precision = float(precision_score(candidate_truth, candidate_prediction, zero_division=0))
    candidate_recall = float(recall_score(candidate_truth, candidate_prediction, zero_division=0))
    candidate_f1 = float(f1_score(candidate_truth, candidate_prediction, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(candidate_truth, candidate_prediction, labels=[0, 1]).ravel()
    candidate_miss_rate = float(observations.candidate_miss.mean())

    continuity_values = []
    for trip_id, part in top.groupby("trip_id"):
        part = part.sort_values("observation_index")
        endpoint_arrays = endpoints_by_trip.get(str(trip_id))
        arc_sequence = part.arc_id.astype(int).to_numpy()
        if len(arc_sequence) < 2 or endpoint_arrays is None:
            continue
        from_nodes, to_nodes = endpoint_arrays
        valid = (
            (arc_sequence[:-1] >= 0) & (arc_sequence[:-1] < len(to_nodes))
            & (arc_sequence[1:] >= 0) & (arc_sequence[1:] < len(from_nodes))
        )
        connected = np.zeros(len(arc_sequence) - 1, dtype=bool)
        prev_ids, next_ids = arc_sequence[:-1], arc_sequence[1:]
        connected[valid] = (
            (prev_ids[valid] == next_ids[valid])
            | (to_nodes[prev_ids[valid]] == from_nodes[next_ids[valid]])
        )
        continuity_values.append(float(np.mean(connected)))
    route_continuity = float(np.mean(continuity_values)) if continuity_values else float("nan")

    return {
        "top1_accuracy": top1_accuracy,
        "macro_f1_mean_per_trip": macro_f1,
        "candidate_binary_log_loss": candidate_log_loss,
        "candidate_binary_precision": candidate_precision,
        "candidate_binary_recall": candidate_recall,
        "candidate_binary_f1": candidate_f1,
        "candidate_tn": int(tn),
        "candidate_fp": int(fp),
        "candidate_fn": int(fn),
        "candidate_tp": int(tp),
        "candidate_miss_rate": candidate_miss_rate,
        "direct_route_continuity": route_continuity,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_observation": (
            inference_seconds / max(1, len(observations))
        ),
        "predictions": top,
    }


def load_endpoints(data_root: Path, trip_ids: set[str]):
    """Load compact NumPy endpoint arrays; avoid millions of Python dict entries."""
    lookup = {}
    for split in ("train", "student_test"):
        split_dir = data_root / split
        for trip_id in trip_ids:
            folder = split_dir / trip_id
            if folder.is_dir():
                _, _, arcs, _ = load_trip_folder(folder)
                lookup[trip_id] = (
                    arcs.from_node_id.to_numpy(dtype=np.int32),
                    arcs.to_node_id.to_numpy(dtype=np.int32),
                )
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    train_candidates = pd.read_csv(
        args.processed_dir / "train_candidates.csv.gz", compression="gzip"
    )
    train_observations = pd.read_csv(
        args.processed_dir / "train_observations.csv.gz", compression="gzip"
    )
    test_candidates = pd.read_csv(
        args.processed_dir / "student_test_candidates.csv.gz", compression="gzip"
    )
    test_observations = pd.read_csv(
        args.processed_dir / "student_test_observations.csv.gz", compression="gzip"
    )
    if train_candidates.empty or test_candidates.empty:
        raise ValueError("Candidate datasets are empty. Run build_candidate_dataset.py first.")

    # Missed target arcs cannot contribute a positive training example. They
    # remain in validation/student-test evaluation and count toward miss rate.
    hit_keys = set(train_observations.loc[
        ~train_observations.candidate_miss, "observation_key"
    ])
    fit_candidates = train_candidates[
        train_candidates.observation_key.isin(hit_keys)
    ].copy()
    if fit_candidates.target.nunique() < 2:
        raise ValueError("Training candidate data must contain positive and negative labels.")

    trip_ids = set(train_candidates.trip_id.astype(str)) | set(test_candidates.trip_id.astype(str))
    endpoints = load_endpoints(args.data_root, trip_ids)
    groups = fit_candidates.trip_id.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=args.folds)
    cv_rows, final_rows = [], []
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for model_name, prototype in make_models().items():
        fold_metrics = []
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(fit_candidates, fit_candidates.target, groups=groups), start=1
        ):
            fold_train = fit_candidates.iloc[train_idx]
            valid_trip_ids = set(fit_candidates.iloc[valid_idx].trip_id.astype(str))
            fold_valid = train_candidates[
                train_candidates.trip_id.astype(str).isin(valid_trip_ids)
            ]
            valid_obs = train_observations[
                train_observations.trip_id.astype(str).isin(valid_trip_ids)
            ]
            model = clone(prototype)
            x_train = fold_train[FEATURE_COLUMNS].to_numpy(dtype=float)
            y_train = fold_train.target.to_numpy(dtype=int)
            fit_model(model, x_train, y_train)
            metrics = evaluate_model(model, fold_valid, valid_obs, endpoints)
            fold_metrics.append(metrics)
            cv_rows.append({
                "model": model_name, "fold": fold,
                "top1_accuracy": metrics["top1_accuracy"],
                "macro_f1_mean_per_trip": metrics["macro_f1_mean_per_trip"],
                "candidate_binary_log_loss": metrics["candidate_binary_log_loss"],
                "candidate_binary_precision": metrics["candidate_binary_precision"],
                "candidate_binary_recall": metrics["candidate_binary_recall"],
                "candidate_binary_f1": metrics["candidate_binary_f1"],
                "candidate_miss_rate": metrics["candidate_miss_rate"],
            })
            print(
                f"{model_name} fold {fold}: "
                f"Top-1={metrics['top1_accuracy']:.4f}, "
                f"Macro-F1={metrics['macro_f1_mean_per_trip']:.4f}"
            )

        # Refit on all candidate-hit training observations, then evaluate once
        # on the separate student-testing partition.
        final_model = clone(prototype)
        started = time.perf_counter()
        fit_model(
            final_model,
            fit_candidates[FEATURE_COLUMNS].to_numpy(dtype=float),
            fit_candidates.target.to_numpy(dtype=int),
        )
        training_seconds = time.perf_counter() - started
        final_metrics = evaluate_model(
            final_model, test_candidates, test_observations, endpoints
        )
        model_path = args.artifacts_dir / f"{model_name.lower()}.joblib"
        joblib.dump(final_model, model_path)
        prediction_path = args.results_dir / f"student_test_predictions_{model_name.lower()}.csv.gz"
        final_metrics["predictions"].to_csv(
            prediction_path, index=False, compression="gzip"
        )
        final_rows.append({
            "model": model_name,
            "cv_macro_f1_mean": float(np.mean([
                item["macro_f1_mean_per_trip"] for item in fold_metrics
            ])),
            "cv_macro_f1_std": float(np.std([
                item["macro_f1_mean_per_trip"] for item in fold_metrics
            ])),
            "student_test_top1_accuracy": final_metrics["top1_accuracy"],
            "student_test_macro_f1": final_metrics["macro_f1_mean_per_trip"],
            "student_test_candidate_binary_log_loss": final_metrics["candidate_binary_log_loss"],
            "student_test_candidate_binary_precision": final_metrics["candidate_binary_precision"],
            "student_test_candidate_binary_recall": final_metrics["candidate_binary_recall"],
            "student_test_candidate_binary_f1": final_metrics["candidate_binary_f1"],
            "student_test_candidate_tn": final_metrics["candidate_tn"],
            "student_test_candidate_fp": final_metrics["candidate_fp"],
            "student_test_candidate_fn": final_metrics["candidate_fn"],
            "student_test_candidate_tp": final_metrics["candidate_tp"],
            "student_test_candidate_miss_rate": final_metrics["candidate_miss_rate"],
            "student_test_direct_route_continuity": final_metrics["direct_route_continuity"],
            "training_seconds": training_seconds,
            "inference_seconds": final_metrics["inference_seconds"],
            "inference_seconds_per_observation": final_metrics["inference_seconds_per_observation"],
            "model_path": str(model_path),
            "prediction_path": str(prediction_path),
        })
        print(
            f"{model_name} student test: "
            f"Top-1={final_metrics['top1_accuracy']:.4f}, "
            f"Macro-F1={final_metrics['macro_f1_mean_per_trip']:.4f}, "
            f"Miss={final_metrics['candidate_miss_rate']:.4f}"
        )

    cv_df = pd.DataFrame(cv_rows)
    comparison = pd.DataFrame(final_rows)
    cv_df.to_csv(args.results_dir / "cross_validation_results.csv", index=False)
    comparison.to_csv(args.results_dir / "model_comparison.csv", index=False)
    for metric, title, output in [
        ("cv_macro_f1_mean", "Group cross-validation Macro F1", "cv_macro_f1.png"),
        ("student_test_top1_accuracy", "Student-test observation Top-1 accuracy", "student_test_top1.png"),
        ("student_test_macro_f1", "Student-test Macro F1 (mean per trip)", "student_test_macro_f1.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(comparison.model, comparison[metric])
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(args.results_dir / output, dpi=160)
        plt.close(fig)

    metadata = {
        "seed": SEED,
        "validation": f"{args.folds}-fold GroupKFold grouped by trip_id",
        "candidate_label": "nearest supplied reference-route arc by spatial point-to-segment distance",
        "training_rows_after_candidate-hit_filter": int(len(fit_candidates)),
        "models": list(make_models()),
    }
    (args.results_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Saved comparison to", args.results_dir / "model_comparison.csv")


if __name__ == "__main__":
    main()
