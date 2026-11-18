# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap is a student project for reconstructing road-network routes from noisy GPS trajectories.

## Implemented in the repository

- Dataset inventory and data-quality audit for the supplied training and student-testing trips.
- GPS-vs-reference-route Folium map generation.
- Geospatial road-segment candidate generation in local metre-based projected coordinates.
- Observation/segment feature extraction (point-to-segment distance, segment midpoint distance, projection fraction, segment length, heading difference, GPS speed, time gap and node degrees).
- Five baseline classifier definitions and a trip-grouped model comparison script.
- Optuna/TPE hyperparameter tuning script for ExtraTrees.
- Log-space Viterbi road-arc sequence decoder.
- FastAPI inference backend and browser-based evaluator UI.
- Unit tests and a GitHub Actions CI workflow.

**Important:** the scripts for candidate building, five-model training and Optuna tuning are supplied but have not yet been run end-to-end by the student on the local machine. No model scores are prefilled or fabricated. The target used by the current prototype is the spatially nearest arc in the supplied reference route, not a timestamp-aligned ground-truth label. This assumption must be validated and reported.

## Dataset layout

Place the supplied dataset locally as:

~~~
data/raw/
├── train/<trip-id>/<trip-id>.{track,nodes,arcs,route}
└── student_test/<trip-id>/<trip-id>.{track,nodes,arcs,route}
~~~

The raw dataset is intentionally excluded from Git. Keep evaluator-held data out of development and training.

## Setup (Windows PowerShell)

~~~
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
~~~

## Run sequence

### 1. Dataset exploration

~~~
python src/data_exploration.py --data data/raw --output results --map visualizations/sample_trip_map.html
~~~

This writes per-trip and aggregate audit files under results/ and an interactive Folium map under visualizations/.

### 2. Candidate and feature dataset

~~~
python scripts/build_candidate_dataset.py --data-root data/raw --output-dir data/processed
~~~

For a smoke test with one trip from each split, add --max-trips 1. Defaults are 8 candidates, every 10th training observation and every 5th student-test observation. Processed candidate tables are written separately for train and student_test and are git-ignored.

### 3. Train and compare five baselines

~~~
python scripts/train_compare_models.py --folds 3
~~~

Models: Logistic Regression, Decision Tree, Random Forest, Extra Trees and HistGradientBoosting. Validation uses GroupKFold grouped by trip ID. The script saves model files locally under artifacts/ and metrics, predictions and plots under results/.

### 4. Bayesian optimisation

~~~
python scripts/tune_extratrees.py --trials 15 --folds 3
~~~

Optuna's TPE sampler tunes ExtraTrees on group-aware training folds. It evaluates the best setting on the separate student_test split and, when the baseline artifact exists, saves a tuned-versus-untuned comparison.

### 5. Tests

~~~
python -m unittest discover -s tests -v
~~~

### 6. Local API and UI

Train at least one model first. For a sample trip, export an inference payload (the reference .route file is deliberately excluded):

~~~
python scripts/export_trip_payload.py --trip-dir data/raw/train/00000001 --output data/processed/trip_00000001_api.json
uvicorn api.app:app --reload
~~~

Open http://127.0.0.1:8000/ and upload the JSON payload. API documentation is at http://127.0.0.1:8000/docs. The API compares classifier-only and optional classifier+HMM/Viterbi predictions.

## Measured dataset audit

The supplied development data contains 80 training trips and 10 student-testing trips (90 total), 227,577 GPS observations, 2,024,174 road nodes, 4,061,945 road arcs, and 34,813 reference-route entries. All 90 trips contained the four expected files. The audit found zero invalid GPS coordinate rows, zero non-monotonic timestamp trips, zero route arc IDs out of range and zero arc endpoint IDs out of range.

There are 7,550 timestamp gaps greater than one second; 42 trips have a maximum gap over 10 seconds and 23 trips over 60 seconds. The largest gap is 153,064 seconds in training trip 00000090. The audit found 14 adjacent reference-route arc connectivity breaks. These are data-quality findings to investigate, not errors to silently repair.

See docs/DATASET_AUDIT.md, results/dataset_summary.json and results/trip_inventory.csv. These are data-exploration results, not model performance results.

## Key design and evaluation notes

- .track is whitespace-separated longitude, latitude and timestamp (seconds).
- .nodes is longitude and latitude; row index is the 0-based node ID.
- .arcs is from-node ID and to-node ID; row index is the 0-based arc ID.
- .route is the ordered sequence of reference arc IDs.
- GPS observations are not assumed to correspond directly to road-node or road-arc rows.
- Distance and motion features are computed in projected metres, not raw longitude/latitude degrees.
- Candidate miss rate must be reported separately; a classifier cannot select the correct arc when candidate generation omits it.
- Candidate target labels are currently nearest-reference-arc spatial proxies. Loops/repeated streets may create ambiguous labels.
- Keep evaluator-held data untouched until the final evaluation; no evaluator labels are used by the development pipeline.
- Commit only real metrics, outputs and model artefacts after verifying they were generated by the submitted code.

## Development status

| Component | Status |
|---|---|
| Public repository and data audit | Implemented; data audit outputs committed |
| Folium map generation code | Implemented |
| Candidate generation and feature builder | Implemented; local end-to-end run pending |
| Five model experiments and group-CV script | Implemented; local training run pending |
| Optuna/TPE tuning script | Implemented; local tuning run pending |
| HMM/Viterbi utility | Implemented; unit tests included |
| Classifier + HMM API inference | Implemented; integration run pending |
| Browser UI | Implemented; local demo run pending |
| Full evaluated results and final report | Pending |
