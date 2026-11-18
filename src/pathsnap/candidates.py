"""Road-segment candidate generation and observation-level feature extraction.

Candidates are generated from the road graph only; reference-route information is
used solely to create an auditable nearest-reference-arc target for experiments.
That target is a spatial proxy (nearest reference segment), not a timestamp-aligned
ground-truth label. Validate this assumption, especially for loops and revisited roads.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


FEATURE_COLUMNS = [
    "distance_to_segment_m",
    "midpoint_distance_m",
    "fraction_along_segment",
    "segment_length_m",
    "heading_difference_rad",
    "gps_speed_mps",
    "time_gap_s",
    "from_node_out_degree",
    "to_node_in_degree",
]


@dataclass
class ProjectedRoadGraph:
    start_xy: np.ndarray
    end_xy: np.ndarray
    midpoint_xy: np.ndarray
    vectors: np.ndarray
    lengths: np.ndarray
    bearings: np.ndarray
    from_node: np.ndarray
    to_node: np.ndarray
    from_out_degree: np.ndarray
    to_in_degree: np.ndarray
    tree: cKDTree


def _project_trip(track: pd.DataFrame, nodes: pd.DataFrame):
    """Project coordinates to a trip-centred azimuthal-equidistant CRS in metres."""
    lon0 = float(track.longitude.mean())
    lat0 = float(track.latitude.mean())
    local_crs = (
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    tx, ty = transformer.transform(
        nodes.longitude.to_numpy(dtype=float),
        nodes.latitude.to_numpy(dtype=float),
    )
    px, py = transformer.transform(
        track.longitude.to_numpy(dtype=float),
        track.latitude.to_numpy(dtype=float),
    )
    return np.column_stack([tx, ty]), np.column_stack([px, py])


def _build_graph(nodes: pd.DataFrame, arcs: pd.DataFrame, node_xy: np.ndarray):
    src = arcs.from_node_id.to_numpy(dtype=np.int64)
    dst = arcs.to_node_id.to_numpy(dtype=np.int64)
    if ((src < 0) | (src >= len(nodes)) | (dst < 0) | (dst >= len(nodes))).any():
        raise ValueError("Arc endpoint is outside the 0-based node index range.")
    start, end = node_xy[src], node_xy[dst]
    vectors = end - start
    lengths = np.linalg.norm(vectors, axis=1)
    midpoints = (start + end) / 2.0
    bearings = np.arctan2(vectors[:, 1], vectors[:, 0])
    out_degree = np.bincount(src, minlength=len(nodes))
    in_degree = np.bincount(dst, minlength=len(nodes))
    return ProjectedRoadGraph(
        start_xy=start, end_xy=end, midpoint_xy=midpoints, vectors=vectors,
        lengths=lengths, bearings=bearings, from_node=src, to_node=dst,
        from_out_degree=out_degree[src], to_in_degree=in_degree[dst],
        tree=cKDTree(midpoints),
    )


def _point_segment_features(point_xy: np.ndarray, graph: ProjectedRoadGraph,
                             arc_ids: np.ndarray):
    starts, vectors = graph.start_xy[arc_ids], graph.vectors[arc_ids]
    denom = np.einsum("ij,ij->i", vectors, vectors)
    numer = np.einsum("ij,ij->i", point_xy - starts, vectors)
    fraction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
    fraction = np.clip(fraction, 0.0, 1.0)
    projected = starts + fraction[:, None] * vectors
    distances = np.linalg.norm(point_xy - projected, axis=1)
    midpoint_distances = np.linalg.norm(point_xy - graph.midpoint_xy[arc_ids], axis=1)
    return distances, midpoint_distances, fraction


def _nearest_reference_arc(point_xy: np.ndarray, route_ids: np.ndarray,
                           graph: ProjectedRoadGraph, k: int = 12) -> tuple[int, float]:
    """Find the closest supplied reference-route arc by point-to-segment distance."""
    if len(route_ids) == 0:
        return -1, float("nan")
    unique_ids = np.unique(route_ids.astype(np.int64))
    if ((unique_ids < 0) | (unique_ids >= len(graph.start_xy))).any():
        raise ValueError("Reference route contains an arc ID outside the arc table.")
    tree = cKDTree(graph.midpoint_xy[unique_ids])
    count = min(k, len(unique_ids))
    _, local_indices = tree.query(point_xy, k=count)
    candidate_ids = unique_ids[np.atleast_1d(local_indices).astype(int)]
    distances, _, _ = _point_segment_features(point_xy, graph, candidate_ids)
    best = int(np.argmin(distances))
    return int(candidate_ids[best]), float(distances[best])


def generate_candidates(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    route: pd.DataFrame,
    *,
    trip_id: str,
    candidate_k: int = 8,
    observation_stride: int = 5,
    reference_search_k: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate nearest road candidates and features for sampled GPS observations.

    Returns (candidate_rows, observation_rows). Candidate rows contain one row
    per observation/road-arc pair and a binary target flag. Observation rows
    contain a spatially nearest reference-arc target and candidate-miss flag.
    The reference arc is only used to define the target, never as an input feature.
    """
    if candidate_k < 1 or observation_stride < 1:
        raise ValueError("candidate_k and observation_stride must both be >= 1.")
    if track.empty or nodes.empty or arcs.empty:
        raise ValueError(f"Trip {trip_id} has an empty track, node table or arc table.")

    node_xy, point_xy_all = _project_trip(track, nodes)
    graph = _build_graph(nodes, arcs, node_xy)
    indices = np.arange(0, len(track), observation_stride, dtype=int)
    points = point_xy_all[indices]
    count = min(candidate_k, len(arcs))
    _, candidate_indices = graph.tree.query(points, k=count)
    if count == 1:
        candidate_indices = candidate_indices.reshape(-1, 1)
    candidate_indices = candidate_indices.astype(np.int64)

    # GPS movement direction/speed are computed from consecutive observations
    # using projected metre coordinates. Time gaps are retained as a separate feature.
    dx = np.zeros(len(track), dtype=float)
    dy = np.zeros(len(track), dtype=float)
    displacement = np.zeros(len(track), dtype=float)
    timestamp = track.timestamp.to_numpy(dtype=float)
    backward_gap = np.diff(timestamp, prepend=timestamp[0])
    forward_gap = np.diff(timestamp, append=timestamp[-1])
    if len(track) > 1:
        delta = np.diff(point_xy_all, axis=0)
        dx[:-1], dy[:-1] = delta[:, 0], delta[:, 1]
        displacement[:-1] = np.linalg.norm(delta, axis=1)
        dx[-1], dy[-1] = delta[-1, 0], delta[-1, 1]
        displacement[-1] = displacement[-2]
        forward_gap[-1] = forward_gap[-2]
    gps_heading = np.arctan2(dy, dx)
    time_gap = np.nan_to_num(backward_gap, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.divide(
        displacement, forward_gap, out=np.zeros_like(displacement),
        where=np.isfinite(forward_gap) & (forward_gap > 0),
    )
    speed = np.clip(speed, 0.0, 80.0)

    route_ids = route.arc_id.to_numpy(dtype=np.int64)
    feature_rows, observation_rows = [], []
    for sampled_pos, original_idx in enumerate(indices):
        point = points[sampled_pos]
        candidates = candidate_indices[sampled_pos]
        seg_dist, mid_dist, fraction = _point_segment_features(point, graph, candidates)
        target_arc, target_distance = _nearest_reference_arc(
            point, route_ids, graph, k=reference_search_k
        )
        candidate_hit = bool(np.any(candidates == target_arc)) if target_arc >= 0 else False
        observation_key = f"{trip_id}:{int(original_idx)}"
        observation_rows.append({
            "observation_key": observation_key,
            "trip_id": str(trip_id),
            "observation_index": int(original_idx),
            "target_arc_id": int(target_arc),
            "target_distance_m": target_distance,
            "candidate_miss": not candidate_hit,
            "time_gap_s": float(time_gap[original_idx]),
        })

        angle_delta = graph.bearings[candidates] - gps_heading[original_idx]
        angle_delta = np.arctan2(np.sin(angle_delta), np.cos(angle_delta))
        for j, arc_id in enumerate(candidates):
            feature_rows.append({
                "observation_key": observation_key,
                "trip_id": str(trip_id),
                "observation_index": int(original_idx),
                "arc_id": int(arc_id),
                "target": int(int(arc_id) == target_arc),
                "distance_to_segment_m": float(seg_dist[j]),
                "midpoint_distance_m": float(mid_dist[j]),
                "fraction_along_segment": float(fraction[j]),
                "segment_length_m": float(graph.lengths[arc_id]),
                "heading_difference_rad": float(angle_delta[j]),
                "gps_speed_mps": float(speed[original_idx]),
                "time_gap_s": float(time_gap[original_idx]),
                "from_node_out_degree": int(graph.from_out_degree[arc_id]),
                "to_node_in_degree": int(graph.to_in_degree[arc_id]),
            })

    return pd.DataFrame(feature_rows), pd.DataFrame(observation_rows)


def load_trip_folder(folder: Path):
    trip_id = folder.name
    track = pd.read_csv(folder / f"{trip_id}.track", sep=r"\s+", header=None,
                        names=["longitude", "latitude", "timestamp"])
    nodes = pd.read_csv(folder / f"{trip_id}.nodes", sep=r"\s+", header=None,
                        names=["longitude", "latitude"])
    arcs = pd.read_csv(folder / f"{trip_id}.arcs", sep=r"\s+", header=None,
                       names=["from_node_id", "to_node_id"])
    route = pd.read_csv(folder / f"{trip_id}.route", sep=r"\s+", header=None,
                        names=["arc_id"])
    return track, nodes, arcs, route


def endpoints_for_trip(arcs: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """Return the trip-local arc_id -> (from_node_id, to_node_id) lookup."""
    return {
        int(arc_id): (int(row.from_node_id), int(row.to_node_id))
        for arc_id, row in arcs.iterrows()
    }
