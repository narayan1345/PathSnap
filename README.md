# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap is a student project for reconstructing likely road-network routes from noisy GPS trajectories. The repository is organised to support data exploration, candidate generation, feature engineering, supervised experiments, HMM/Viterbi sequence decoding, evaluation, and a small FastAPI/UI layer.

## Project status

This repository currently contains the project scaffold and a runnable dataset-inspection pipeline. Model training, Bayesian optimisation, HMM/Viterbi decoding, and the API/UI are planned components and must be implemented and evaluated before they can be described as complete.

## Dataset layout

Place the supplied dataset locally as:

```
data/raw/
├── train/
│   └── <trip-id>/
│       ├── <trip-id>.track
│       ├── <trip-id>.nodes
│       ├── <trip-id>.arcs
│       └── <trip-id>.route
└── student_test/
    └── <trip-id>/
        ├── <trip-id>.track
        ├── <trip-id>.nodes
        ├── <trip-id>.arcs
        └── <trip-id>.route
```

The raw dataset is intentionally excluded from Git. Do not commit evaluator-held data, credentials, virtual environments, or generated model binaries.

## Quick start (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
python scripts/inspect_dataset.py --data-root data/raw
```

## Repository structure

- `src/pathsnap/data_io.py`: safe trip discovery and basic file loading.
- `scripts/inspect_dataset.py`: dataset inventory and basic integrity checks.
- `notebooks/`: exploratory analysis and experiments.
- `src/pathsnap/`: implementation modules.
- `api/`: FastAPI application (planned).
- `ui/`: evaluator interface (planned).
- `artifacts/`: locally generated model artefacts (not committed).
- `results/`: locally generated experiment outputs (not committed).

## Important evaluation notes

1. Keep the student test split separate from training and model fitting.
2. Fit preprocessing and learned transformations on training data only.
3. Record seeds, package versions, configurations, and evaluation metrics.
4. Never report fabricated or placeholder model scores.
5. Document dataset-specific column formats after inspecting the supplied files; do not assume delimiters or ID conventions.

## Development status

| Component | Status |
|---|---|
| Repository scaffold | Created |
| Dataset inventory script | Runnable |
| Dataset exploration notebook | To be completed |
| Candidate generation / features | Pending |
| Five model experiments | Pending |
| Bayesian optimisation | Pending |
| HMM / Viterbi | Pending |
| Route evaluation and maps | Pending |
| FastAPI backend and UI | Pending |
| Technical report | Pending |
