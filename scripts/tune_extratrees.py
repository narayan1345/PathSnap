"""Bayesian hyperparameter optimisation for the ExtraTrees candidate classifier.

Run after candidate generation and baseline model comparison:
python scripts/tune_extratrees.py --trials 15 --folds 3

The objective is mean per-trip Macro F1 under GroupKFold by trip_id. Labels are
nearest reference-route arc within the 12-midpoint-neighbour shortlist, refined by projected point-to-segment distance (spatial proxy), not timestamp-aligned route labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from pathsnap.candidates import FEATURE_COLUMNS


SEED = 42


def load_data(directory: Path):
    train_x = pd.read_csv(directory / "train_candidates.csv.gz")
    train_obs = pd.read_csv(directory / "train_observations.csv.gz")
    test_x = pd.read_csv(directory / "student_test_candidates.csv.gz")
    test_obs = pd.read_csv(directory / "student_test_observations.csv.gz")
    hit_keys = set(train_obs.loc[~train_obs.candidate_miss, "observation_key"])
    fit_x = train_x[train_x.observation_key.isin(hit_keys)].copy()
    if fit_x.target.nunique() != 2:
        raise ValueError("Training data must contain positive and negative candidate labels.")
    return train_x, train_obs, fit_x, test_x, test_obs


def per_trip_macro_f1(predictions: pd.DataFrame) -> float:
    values = []
    for _, trip in predictions.groupby("trip_id"):
        labels = np.unique(np.concatenate([
            trip.target_arc_id.to_numpy(), trip.predicted_arc_id.to_numpy()
        ]))
        values.append(f1_score(
            trip.target_arc_id, trip.predicted_arc_id, labels=labels,
            average="macro", zero_division=0,
        ))
    return float(np.mean(values)) if values else 0.0


def predict_top_arc(model, candidates: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(
        candidates[FEATURE_COLUMNS].to_numpy(dtype=float)
    )[:, 1]
    frame = candidates[[
        "observation_key", "trip_id", "observation_index", "arc_id"
    ]].copy()
    frame["probability"] = probabilities
    index = frame.groupby("observation_key", sort=False).probability.idxmax()
    return frame.loc[index].rename(columns={
        "arc_id": "predicted_arc_id", "probability": "selected_probability"
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1 or args.folds < 2:
        parser.error("trials must be >=1 and folds must be >=2.")

    train_x, train_obs, fit_x, test_x, test_obs = load_data(args.processed_dir)
    groups = fit_x.trip_id.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=args.folds)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 180, step=30),
            "max_depth": trial.suggest_int("max_depth", 8, 28, step=4),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": SEED,
        }
        fold_scores = []
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(fit_x, fit_x.target, groups=groups), start=1
        ):
            fold_train = fit_x.iloc[train_idx]
            valid_trip_ids = set(fit_x.iloc[valid_idx].trip_id.astype(str))
            fold_valid = train_x[train_x.trip_id.astype(str).isin(valid_trip_ids)]
            valid_obs = train_obs[train_obs.trip_id.astype(str).isin(valid_trip_ids)]

            model = ExtraTreesClassifier(**params)
            model.fit(
                fold_train[FEATURE_COLUMNS].to_numpy(dtype=float),
                fold_train.target.to_numpy(dtype=int),
            )
            prediction = predict_top_arc(model, fold_valid)
            prediction = prediction.merge(
                valid_obs[["observation_key", "target_arc_id"]],
                on="observation_key", how="left", validate="one_to_one",
            )
            prediction = prediction.rename(columns={"target_arc_id": "target_arc_id"})
            fold_scores.append(per_trip_macro_f1(prediction))
            trial.report(float(np.mean(fold_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.trials)

    best_params = {
        **study.best_params,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
    }
    tuned = ExtraTreesClassifier(**best_params)
    tuned.fit(
        fit_x[FEATURE_COLUMNS].to_numpy(dtype=float),
        fit_x.target.to_numpy(dtype=int),
    )

    test_prediction = predict_top_arc(tuned, test_x).merge(
        test_obs[["observation_key", "target_arc_id", "candidate_miss"]],
        on="observation_key", how="left", validate="one_to_one",
    )
    test_metrics = {
        "best_cv_macro_f1_mean_per_trip": float(study.best_value),
        "student_test_top1_accuracy": float(
            (test_prediction.predicted_arc_id == test_prediction.target_arc_id).mean()
        ),
        "student_test_macro_f1_mean_per_trip": per_trip_macro_f1(
            test_prediction.rename(columns={"predicted_arc_id": "predicted_arc_id"})
        ),
        "student_test_candidate_miss_rate": float(test_obs.candidate_miss.mean()),
        "candidate_label_definition": (
            "Nearest point-to-segment distance among up to 12 midpoint-nearest route arcs; spatial proxy, "
            "not time-aligned ground truth."
        ),
    }

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "tuned_extratrees.joblib"
    joblib.dump(tuned, model_path)
    study.trials_dataframe().to_csv(args.results_dir / "optuna_trials.csv", index=False)
    (args.results_dir / "optuna_best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    (args.results_dir / "tuned_model_student_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    test_prediction.to_csv(
        args.results_dir / "student_test_predictions_tuned_extratrees.csv.gz",
        index=False, compression="gzip",
    )

    # Compare with the untuned ExtraTrees baseline when it has been trained.
    baseline_path = args.artifacts_dir / "extratrees.joblib"
    comparison_rows = [{
        "model": "Optuna-tuned ExtraTrees",
        "student_test_top1_accuracy": test_metrics["student_test_top1_accuracy"],
        "student_test_macro_f1_mean_per_trip": test_metrics["student_test_macro_f1_mean_per_trip"],
        "student_test_candidate_miss_rate": test_metrics["student_test_candidate_miss_rate"],
    }]
    if baseline_path.is_file():
        baseline = joblib.load(baseline_path)
        baseline_prediction = predict_top_arc(baseline, test_x).merge(
            test_obs[["observation_key", "target_arc_id", "candidate_miss"]],
            on="observation_key", how="left", validate="one_to_one",
        )
        comparison_rows.insert(0, {
            "model": "Untuned ExtraTrees baseline",
            "student_test_top1_accuracy": float(
                (baseline_prediction.predicted_arc_id == baseline_prediction.target_arc_id).mean()
            ),
            "student_test_macro_f1_mean_per_trip": per_trip_macro_f1(baseline_prediction),
            "student_test_candidate_miss_rate": float(test_obs.candidate_miss.mean()),
        })

    pd.DataFrame(comparison_rows).to_csv(
        args.results_dir / "tuned_vs_untuned_comparison.csv", index=False
    )
    print("Best CV Macro F1:", study.best_value)
    print("Best parameters:", json.dumps(best_params, indent=2))
    print("Student-test metrics:", json.dumps(test_metrics, indent=2))
    print("Saved tuned model:", model_path)


if __name__ == "__main__":
    main()
