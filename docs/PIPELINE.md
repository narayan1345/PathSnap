# End-to-end development pipeline

Run commands from the repository root after installing requirements and the editable package.

## 1. Dataset audit

~~~powershell
python src/data_exploration.py --data data/raw --output results --map visualizations/sample_trip_map.html
~~~

The audit checks file availability, GPS coordinate validity, timestamps, graph IDs, graph connectivity and reference-route arc adjacency.

## 2. Candidate generation and features

~~~powershell
python scripts/build_candidate_dataset.py --data-root data/raw --output-dir data/processed
~~~

Defaults: 8 candidate segments per GPS observation, sample every 10th training observation and every 5th student-testing observation. To smoke-test one trip from each partition, add --max-trips 1.

Features are calculated after projecting the trip to a local metre-based azimuthal-equidistant CRS. Candidate segments are nearest road-arc midpoints, refined with point-to-segment distance features. The label is the nearest arc among unique reference-route arcs by spatial distance. This label is a **proxy** and is not guaranteed to be the road segment at that exact timestamp, especially on loops or routes that revisit a road. Candidate miss rate is reported separately.

## 3. Baseline model experiments

~~~powershell
python scripts/train_compare_models.py --folds 3
~~~

Five classifiers are compared: Logistic Regression, Decision Tree, Random Forest, Extra Trees and HistGradientBoosting. Cross-validation is grouped by trip ID so observations from one trip do not appear in both training and validation folds. After cross-validation, each model is refit on candidate-hit training observations and evaluated against the separate student-testing partition.

Reported measures include observation Top-1 accuracy, per-trip Macro F1, candidate binary precision/recall/F1, candidate binary log loss, candidate miss rate, candidate confusion-matrix counts, direct arc continuity, training and inference time. Interpret these in light of the spatial proxy target and candidate-generation assumptions.

## 4. Bayesian optimisation

~~~powershell
python scripts/tune_extratrees.py --trials 15 --folds 3
~~~

Optuna's TPE sampler searches ExtraTrees parameters using trip-grouped GroupKFold. The objective is mean per-trip Macro F1. The best estimator is fitted using training data and evaluated on student_test. If artifacts/extratrees.joblib exists, the script also writes a tuned-versus-untuned student-test comparison.

## 5. Unit tests

~~~powershell
python -m unittest discover -s tests -v
~~~

The current tests cover synthetic candidate extraction and the Viterbi utility's connected-path preference, missing-candidate sentinel and long-gap relaxation.

## 6. Local inference API and UI

After fitting a model (for example ExtraTrees):

~~~powershell
python scripts/export_trip_payload.py --trip-dir data/raw/train/00000001 --output data/processed/trip_00000001_api.json
uvicorn api.app:app --reload
~~~

Open http://127.0.0.1:8000/ and upload the JSON payload. The exporter does not include the .route label file, and the inference API does not accept a reference route. Choose classifier-only or classifier + HMM/Viterbi. The API is intended for local evaluation and should not be exposed publicly without authentication, request-size limits and deployment hardening.

## Reproducibility and split discipline

- Do not commit raw trip folders, evaluator-held data, model binaries or credentials.
- Candidate generation outputs and trained model files are local and ignored by Git.
- Commit actual summary tables, figures, logs and comparison CSVs only after running the commands and verifying that the outputs are genuine.
- Keep evaluator-held data untouched until final grading.
- The target-generation method is a prototype spatial proxy; establish alignment validity before presenting its model metrics as definitive map-matching quality.
