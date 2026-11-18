# Development status

This file distinguishes implemented and executed work from planned work.

## Implemented and tested

- Public GitHub repository and project scaffold.
- Dependency manifest and ignore rules for raw data, virtual environments, local secrets and generated artefacts.
- Parsing of all four documented trip file formats: .track, .nodes, .arcs, .route.
- Per-trip data inventory for the supplied 80 training and 10 student-testing trips.
- GPS coordinate validity, missing-cell, timestamp-order, duplicate timestamp and sampling-gap checks.
- Arc endpoint and reference-route arc ID range checks.
- Sparse graph analysis: weakly connected component and node-degree statistics.
- Consecutive reference-route arc connectivity audit.
- Folium map-generation code for observed GPS and supplied reference-route geometry.
- Recorded dataset summary and written audit findings in docs/DATASET_AUDIT.md.

## Remaining implementation work

- Candidate road-segment generation for each GPS observation.
- Geospatial feature engineering using metre-based distances and motion features.
- At least five trained and evaluated model families, with trip-group-aware validation.
- Candidate miss-rate, observation-level accuracy, Macro F1, log loss and route-continuity evaluation.
- Bayesian hyperparameter optimisation for the selected model, including retained trial results.
- HMM emission and transition modelling plus Viterbi decoding.
- Classifier-only versus classifier-plus-HMM route comparison.
- FastAPI inference backend with server-side reference data and evaluation.
- Evaluator-facing UI that calls the API.
- End-to-end integration and technical report.

## Important

The repository currently contains genuine EDA and data-integrity results, not trained-model
performance results. Do not claim that pending components are complete. Update this document
as each component is implemented, executed and verified.
