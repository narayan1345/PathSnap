# Development status

This file distinguishes implemented work from planned work.

## Implemented in this commit

- Repository scaffold and Python package layout.
- Git ignore rules for local raw data, virtual environments, and generated files.
- Conservative trip discovery and expected-file inventory.
- Dataset inventory CLI that writes a JSON report.

## Not yet implemented

- Parsing and validation of all trip file schemas.
- GPS-to-road candidate generation and feature extraction.
- Five trained and evaluated models.
- Bayesian hyperparameter optimisation.
- HMM emission/transition probabilities and Viterbi decoding.
- Route-level evaluation and Folium output.
- FastAPI endpoints and evaluator UI.

Update this document as each component is implemented and tested. Do not mark a
component complete merely because its folder or placeholder documentation exists.
