# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap is a student project for reconstructing likely road-network routes from noisy GPS trajectories.

## Current implementation status

Implemented and run against the supplied development data:

- Read-only per-trip inventory for the four supplied file types.
- GPS coordinate and timestamp quality checks, including irregular and long sampling gaps.
- Arc/node identifier range checks.
- Sparse road-graph weak-component and degree statistics.
- Reference-route arc-to-arc connectivity checks.
- Folium map generation comparing observed GPS points with the supplied reference route.
- Reproducible dataset summary and per-trip inventory outputs.
- Log-space Viterbi decoding utility with direct-connectivity transitions, disconnected-transition fallback, and long-gap/missing-candidate handling.
- Unit tests for the Viterbi utility.

Not yet implemented: candidate generation, supervised feature engineering, five trained models and comparison, Bayesian optimisation, integration of Viterbi with a trained classifier, complete prediction API, evaluator UI, and final report. Do not present these as complete until implemented and tested.

## Dataset layout

Place the supplied dataset locally as:

~~~
data/raw/
├── train/<trip-id>/<trip-id>.{track,nodes,arcs,route}
└── student_test/<trip-id>/<trip-id>.{track,nodes,arcs,route}
~~~

The raw dataset is intentionally excluded from Git. Keep the evaluator-held partition out of development and training.

## Setup (Windows PowerShell)

~~~
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
~~~

## Run dataset exploration

From the repository root:

~~~
python src/data_exploration.py --data data/raw --output results --map visualizations/sample_trip_map.html
~~~

The script creates:

- results/trip_inventory.csv: per-trip counts, timestamp checks, coordinate bounds, graph connectivity and route continuity.
- results/missing_files.csv: trips with missing expected files.
- results/dataset_summary.json: aggregate dataset audit.
- visualizations/sample_trip_map.html: interactive observed-GPS and reference-route map for the first training trip.

The notebook notebooks/01_dataset_exploration.ipynb runs the same audit and provides summary plots and a map view.

## File semantics

The supplied Dataset Exploration guide specifies:

- .track: longitude, latitude, timestamp in seconds; whitespace separated, no header.
- .nodes: longitude, latitude; row index is the 0-based node ID.
- .arcs: from-node ID, to-node ID; row index is the 0-based arc ID.
- .route: chronological sequence of arc IDs.

A GPS row is not assumed to correspond directly to a node or arc row. Use geodesic or projected metre-based distances for spatial calculations, not raw Euclidean distances in longitude/latitude degrees.

## Measured dataset audit

The supplied development data audit found 80 training trips and 10 student-testing trips (90 total), 227,577 GPS observations, 2,024,174 road nodes, 4,061,945 road arcs, and 34,813 reference-route entries. All 90 trips contained the four expected files; the audit found zero invalid GPS coordinate rows, zero non-monotonic timestamp trips, zero route arc IDs out of range, and zero arc endpoint IDs out of range. The audit identified 14 adjacent-arc connectivity breaks across the supplied route sequences; these should be investigated rather than silently repaired.

There are 7,550 timestamp gaps greater than one second; 42 trips have a maximum gap over 10 seconds and 23 trips over 60 seconds. The largest gap is 153,064 seconds in training trip 00000090.

See docs/DATASET_AUDIT.md, results/dataset_summary.json and results/trip_inventory.csv. These are dataset inspection results, not model performance results.

## Viterbi utility

The decoder implementation is src/pathsnap/hmm.py, with design notes in docs/HMM_DECODER.md and unit tests in tests/test_hmm.py. Run tests with:

~~~
python -m unittest discover -s tests -v
~~~

This utility has not yet been integrated with trained classifier emissions or evaluated on the trip dataset.

## Data partition and evaluation integrity

- Use the 80 training trips for fitting, development and group-aware cross-validation.
- Use the 10 student-testing trips for development-stage comparisons.
- Keep the 10 evaluator-testing trips untouched until final evaluation.
- Never use evaluator labels for training, feature selection, tuning or UI predictions.
- Retain seeds, configurations, logs and actual metrics. Do not fabricate model scores or training curves.

## Development status

| Component | Status |
|---|---|
| Public repository and project scaffold | Created |
| Dataset inventory and quality audit | Implemented and run |
| Folium GPS/reference-route map code | Implemented and generated locally |
| Viterbi decoder utility | Implemented; unit tests pass |
| Candidate generation / supervised features | Pending |
| Five model experiments and comparison | Pending |
| Bayesian hyperparameter optimisation | Pending |
| Classifier plus HMM route evaluation | Pending |
| FastAPI backend and evaluator UI | Pending |
| Technical report | Pending |
