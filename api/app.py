"""PathSnap FastAPI inference service.

Start after running the model-training script:
    uvicorn api.app:app --reload

POST /predict with a JSON trip containing track, nodes and arcs. The route file is
not accepted at inference time: it is ground truth and must never be used to
generate features or predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pathsnap.candidates import FEATURE_COLUMNS, generate_candidates
from pathsnap.hmm import viterbi_decode


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
UI_FILE = ROOT / "ui" / "index.html"
MODEL_NAMES = Literal[
    "logisticregression", "decisiontree", "randomforest",
    "extratrees", "histgradientboosting", "tuned_extratrees",
]
MODEL_CACHE = {}


class GPSPoint(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    timestamp: float


class RoadNode(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)


class RoadArc(BaseModel):
    from_node_id: int = Field(ge=0)
    to_node_id: int = Field(ge=0)


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=128)
    track: list[GPSPoint] = Field(min_length=1, max_length=30000)
    nodes: list[RoadNode] = Field(min_length=2, max_length=1000000)
    arcs: list[RoadArc] = Field(min_length=1, max_length=1500000)
    model_name: MODEL_NAMES = "extratrees"
    candidate_k: int = Field(default=8, ge=1, le=32)
    apply_hmm: bool = True
    long_gap_seconds: float = Field(default=30.0, ge=0)


app = FastAPI(
    title="PathSnap Map-Matching API",
    version="0.1.0",
    description=(
        "Predict road-segment arc IDs from GPS observations and a road graph. "
        "Reference route labels are deliberately excluded from inference inputs."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def load_model(name: str):
    if name not in MODEL_CACHE:
        path = ARTIFACTS / f"{name}.joblib"
        if not path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact not found: {path.name}. Train models first.",
            )
        try:
            MODEL_CACHE[name] = joblib.load(path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load {path.name}: {exc}") from exc
    return MODEL_CACHE[name]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_models": [
            name for name in MODEL_NAMES.__args__
            if (ARTIFACTS / f"{name}.joblib").is_file()
        ],
        "note": "Train model artifacts before calling /predict.",
    }


@app.get("/")
def evaluator_ui():
    if not UI_FILE.is_file():
        raise HTTPException(status_code=404, detail="UI file is not installed.")
    return FileResponse(UI_FILE)


@app.post("/predict")
def predict(request: PredictionRequest):
    model = load_model(request.model_name)
    track = pd.DataFrame([point.model_dump() for point in request.track])
    nodes = pd.DataFrame([point.model_dump() for point in request.nodes])
    arcs = pd.DataFrame([arc.model_dump() for arc in request.arcs])
    route = pd.DataFrame({"arc_id": pd.Series(dtype="int64")})

    try:
        candidate_frame, _ = generate_candidates(
            track, nodes, arcs, route,
            trip_id=request.trip_id,
            candidate_k=request.candidate_k,
            observation_stride=1,
        )
        if candidate_frame.empty:
            raise ValueError("Candidate generation produced no road candidates.")
        probabilities = model.predict_proba(
            candidate_frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        )[:, 1]
    except (ValueError, KeyError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    candidate_frame = candidate_frame.copy()
    candidate_frame["probability"] = probabilities
    ordered_groups = []
    for observation_index, group in candidate_frame.groupby(
        "observation_index", sort=True
    ):
        group = group.sort_values("probability", ascending=False)
        ordered_groups.append({
            "observation_index": int(observation_index),
            "arc_ids": group.arc_id.astype(int).tolist(),
            "probabilities": group.probability.astype(float).tolist(),
        })

    classifier_sequence = [group["arc_ids"][0] for group in ordered_groups]
    classifier_confidence = [group["probabilities"][0] for group in ordered_groups]
    predicted_sequence = classifier_sequence
    mode = "classifier_only"

    if request.apply_hmm and ordered_groups:
        # The Viterbi decoder only needs endpoints for candidate arcs, not a
        # Python dictionary for every arc in potentially large road networks.
        used_arc_ids = {
            int(arc_id)
            for group in ordered_groups
            for arc_id in group["arc_ids"]
        }
        endpoints = {
            arc_id: (
                int(arcs.iloc[arc_id].from_node_id),
                int(arcs.iloc[arc_id].to_node_id),
            )
            for arc_id in used_arc_ids
            if 0 <= arc_id < len(arcs)
        }
        predicted_sequence = viterbi_decode(
            [group["arc_ids"] for group in ordered_groups],
            [group["probabilities"] for group in ordered_groups],
            endpoints,
            long_gap_seconds=request.long_gap_seconds,
            time_gaps_seconds=track.timestamp.diff().fillna(0).tolist()[1:],
        )
        mode = "classifier_plus_hmm"

    points = []
    for index, group in enumerate(ordered_groups):
        predicted_arc = int(predicted_sequence[index])
        confidence = (
            float(group["probabilities"][group["arc_ids"].index(predicted_arc)])
            if predicted_arc in group["arc_ids"] else None
        )
        points.append({
            "observation_index": group["observation_index"],
            "candidate_arc_ids": group["arc_ids"],
            "classifier_arc_id": int(classifier_sequence[index]),
            "predicted_arc_id": predicted_arc,
            "selected_probability": confidence,
        })

    return {
        "trip_id": request.trip_id,
        "model_name": request.model_name,
        "mode": mode,
        "observation_count": len(points),
        "points": points,
    }
