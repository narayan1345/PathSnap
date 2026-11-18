# FastAPI backend and evaluator UI

## Start

From the repository root, after installing requirements and training at least one model:

~~~powershell
uvicorn api.app:app --reload
~~~

Open http://127.0.0.1:8000/ for the evaluator UI, or http://127.0.0.1:8000/docs for the interactive API documentation. Check /health for available model artifacts.

## API request

POST /predict accepts:

- trip_id
- track: list of objects containing longitude, latitude and timestamp
- nodes: list of objects containing longitude and latitude
- arcs: list of objects containing from_node_id and to_node_id
- model_name (defaults to extratrees)
- candidate_k (defaults to 8)
- apply_hmm (defaults to true)
- long_gap_seconds (defaults to 30)

The reference .route file is deliberately not accepted: inference must not receive ground-truth labels. Export an example payload with:

~~~powershell
python scripts/export_trip_payload.py --trip-dir data/raw/train/00000001 --output data/processed/trip_00000001_api.json
~~~

Upload that JSON in the browser UI. The API returns one predicted road arc per GPS observation and includes the classifier-only arc for comparison when HMM decoding is enabled.

The service loads model files from artifacts/. It returns HTTP 503 when the requested model artifact is not present. Train models before running the prediction interface.

## Local-use warning

This is an evaluation prototype, not a production-secure service. It accepts trip geometry through JSON and has no authentication. Run on localhost only; do not expose it publicly without authentication, strict request-size limits, rate limiting, and deployment hardening.
