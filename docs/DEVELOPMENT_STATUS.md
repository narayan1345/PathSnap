# Development status

This file distinguishes code that has been implemented and checked from work that still
requires a full local dataset run and evaluation.

## Implemented and checked

- Public GitHub repository, Python project package, dependency manifest and .gitignore.
- Read-only exploration of the supplied 80 training and 10 student-testing trips.
- Per-trip inventory, file checks, coordinate validation, timestamp quality checks,
  graph component/degree statistics and reference-route adjacency diagnostics.
- Saved dataset inventory, aggregate summary and written audit.
- Folium code comparing a sample trip's observed GPS trajectory with its supplied
  reference route; generated locally during dataset exploration.
- Local projected-metre road candidate retrieval and feature-generation code.
- Candidate dataset builder that keeps train and student-test files separate.
- Five baseline model definitions and a trip-grouped GroupKFold comparison script.
- Optuna/TPE ExtraTrees tuning script with tuned-versus-untuned comparison output.
- Log-space Viterbi decoder utility, with explicit handling for disconnected candidates,
  long GPS gaps and observations with no candidates.
- FastAPI inference endpoint that accepts track/nodes/arcs only and does not accept
  reference-route labels.
- Browser-based evaluator UI and JSON payload export script.
- CI workflow: syntax compilation and six unit tests have passed on GitHub Actions.

## Still requires local execution and final validation

- Run candidate generation end-to-end on all 80 training and 10 student-testing trips.
- Validate the spatial nearest-reference-arc proxy label against the assignment's intended
  label semantics; it may be ambiguous for loops and repeated streets.
- Run all five models and retain real CV/student-test metrics, plots, confusion-matrix
  counts, logs and model artefacts.
- Run Optuna and record actual trials, best parameters and tuned-versus-untuned results.
- Run the API/UI against a trained artifact and demonstrate classifier-only versus
  classifier-plus-HMM outputs on a sample trip.
- Review results and produce the final technical report.

## Important limitations

The current target-generation approach is a spatial proxy: for each sampled observation,
it selects the nearest arc among a shortlist of arcs appearing in the reference route.
It is not guaranteed to be the true time-aligned arc for that observation. Model scores
based on this proxy must be clearly qualified and should not be represented as definitive
map-matching performance without validating the label alignment.

The Viterbi utility and API path are implemented, but end-to-end classifier-to-HMM
evaluation has not yet been run. No model performance scores are claimed in this repo.

## Evaluation integrity

- The 80 training trips are used for fitting and trip-grouped validation.
- The 10 student-testing trips are reserved for development-stage comparisons.
- Any separate evaluator-testing partition remains untouched until final evaluation.
- Raw data, generated processed feature tables, and model binaries are excluded from Git
  by default. Commit only genuine reports, metrics, plots and logs after execution.
