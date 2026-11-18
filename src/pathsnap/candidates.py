"""Road-segment candidates and observation-level geospatial features.

Candidate generation uses only the road graph. The reference route is used only
to create a nearest-reference-arc target for experiments. This target is a spatial
proxy, not a timestamp-aligned ground-truth label; loops and revisited roads may
make nearest-geometry alignment ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

FEATURE_COLUMNS = [
    "distance_to_segment_m", "midpoint_distance_m", "fraction_along_segment",
    "segment_length_m", "heading_difference_rad", "gps_speed_mps",
    "time_gap_s", "from_node_out_degree", "to_node_in_degree",
]


@dataclass
class ProjectedRoadGraph:
    start_xy: np.ndarray
    midpoint_xy: np.ndarray
    vectors: np.ndarray
    lengths: np.ndarray
    bearings: np.ndarray
    from_node_out_degree: np.ndarray
    to_node_in_degree: np.ndarray
    tree: cKDTree


def project_trip(track: pd.DataFrame, nodes: pd.DataFrame):
    """Project one trip to a trip-centred azimuthal-equidistant CRS in metres."""
    lon0, lat0 = float(track.longitude.mean()), float(track.latitude.mean())
    local_crs = (
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    nx, ny = transformer.transform(nodes.longitude.to_numpy(float), nodes.latitude.to_numpy(float))
    px, py = transformer.transform(track.longitude.to_numpy(float), track.latitude.to_numpy(float))
    return np.column_stack((nx, ny)), np.column_stack((px, py))


def build_projected_graph(nodes: pd.DataFrame, arcs: pd.DataFrame, node_xy: np.ndarray):
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
        start_xy=start, midpoint_xy=midpoints, vectors=vectors, lengths=lengths,
        bearings=bearings, from_node_out_degree=out_degree[src],
        to_node_in_degree=in_degree[dst], tree=cKDTree(midpoints),
    )


def point_segment_features(point_xy: np.ndarray, graph: ProjectedRoadGraph,
                           arc_ids: np.ndarray):
    starts, vectors = graph.start_xy[arc_ids], graph.vectors[arc_ids]
    denom = np.einsum("ij,ij->i", vectors, vectors)
    numer = np.einsum("ij,ij->i", point_xy - starts, vectors)
    fraction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = starts + fraction[:, None] * vectors
    distance = np.linalg.norm(point_xy - closest, axis=1)
    midpoint_distance = np.linalg.norm(point_xy - graph.midpoint_xy[arc_ids], axis=1)
    return distance, midpoint_distance, fraction


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
    return {int(i): (int(r.from_node_id), int(r.to_node_id)) for i, r in arcs.iterrows()}


def generate_candidates(
    track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame, route: pd.DataFrame,
    *, trip_id: str, candidate_k: int = 8, observation_stride: int = 5,
    reference_search_k: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return candidate-level features and per-observation proxy targets.

    Candidate features never include the reference route. Each GPS observation's
    target is its nearest segment among the unique arcs appearing in the supplied
    route. Candidate miss is reported if that target is absent from the graph-only
    candidate set. Use this as a declared spatial-alignment assumption in reports.
    """
    if candidate_k < 1 or observation_stride < 1 or reference_search_k < 1:
        raise ValueError("candidate/search k and observation_stride must be >= 1.")
    if track.empty or nodes.empty or arcs.empty:
        raise ValueError(f"Trip {trip_id} has an empty track, node table or arc table.")

    node_xy, point_xy_all = project_trip(track, nodes)
    graph = build_projected_graph(nodes, arcs, node_xy)
    indices = np.arange(0, len(track), observation_stride, dtype=int)
    points = point_xy_all[indices]

    candidate_count = min(candidate_k, len(arcs))
    _, candidate_idx = graph.tree.query(points, k=candidate_count)
    if candidate_count == 1:
        candidate_idx = candidate_idx.reshape(-1, 1)
    candidate_idx = candidate_idx.astype(np.int64)

    route_ids = route.arc_id.to_numpy(dtype=np.int64)
    unique_route_ids = np.unique(route_ids)
    if len(unique_route_ids) and (
        (unique_route_ids < 0).any() or (unique_route_ids >= len(arcs)).any()
    ):
        raise ValueError("Reference route contains an arc ID outside the arc table.")
    if len(unique_route_ids):
        route_tree = cKDTree(graph.midpoint_xy[unique_route_ids])
        route_count = min(reference_search_k, len(unique_route_ids))
        _, route_local_idx = route_tree.query(points, k=route_count)
        if route_count == 1:
            route_local_idx = route_local_idx.reshape(-1, 1)
        route_candidate_idx = unique_route_ids[np.asarray(route_local_idx, dtype=int)]
    else:
        route_candidate_idx = np.empty((len(points), 0), dtype=int)

    # Motion features are calculated in projected metres, never raw degrees.
    n_track = len(track)
    dx = np.zeros(n_track)
    dy = np.zeros(n_track)
    displacement = np.zeros(n_track)
    timestamps = track.timestamp.to_numpy(dtype=float)
    backward_gap = np.diff(timestamps, prepend=timestamps[0])
    forward_gap = np.diff(timestamps, append=timestamps[-1])
    if n_track > 1:
        delta = np.diff(point_xy_all, axis=0)
        dx[:-1], dy[:-1] = delta[:, 0], delta[:, 1]
        displacement[:-1] = np.linalg.norm(delta, axis=1)
        dx[-1], dy[-1] = delta[-1, 0], delta[-1, 1]
        displacement[-1] = displacement[-2]
        forward_gap[-1] = forward_gap[-2]
    heading = np.arctan2(dy, dx)
    speed = np.divide(displacement, forward_gap, out=np.zeros_like(displacement),
                      where=np.isfinite(forward_gap) & (forward_gap > 0))
    speed = np.clip(speed, 0.0, 80.0)
    backward_gap = np.nan_to_num(backward_gap, nan=0.0, posinf=0.0, neginf=0.0)

    candidate_rows, observation_rows = [], []
    for pos, original_idx in enumerate(indices):
        candidates = candidate_idx[pos]
        seg_dist, midpoint_dist, fraction = point_segment_features(points[pos], graph, candidates)
        if route_candidate_idx.shape[1]:
            ref_candidates = route_candidate_idx[pos]
            ref_dist, _, _ = point_segment_features(points[pos], graph, ref_candidates)
            best = int(np.argmin(ref_dist))
            target_arc, target_dist = int(ref_candidates[best]), float(ref_dist[best])
        else:
            target_arc, target_dist = -1, float("nan")
        hit = bool(np.any(candidates == target_arc)) if target_arc >= 0 else False
        key = f"{trip_id}:{int(original_idx)}"
        gap = float(backward_gap[original_idx])
        observation_rows.append({
            "observation_key": key, "trip_id": str(trip_id),
            "observation_index": int(original_idx), "target_arc_id": target_arc,
            "target_distance_m": target_dist, "candidate_miss": not hit,
            "time_gap_s": gap,
        })

        angle = graph.bearings[candidates] - heading[original_idx]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        for j, arc_id in enumerate(candidates):
            candidate_rows.append({
                "observation_key": key, "trip_id": str(trip_id),
                "observation_index": int(original_idx), "arc_id": int(arc_id),
                "target": int(int(arc_id) == target_arc),
                "distance_to_segment_m": float(seg_dist[j]),
                "midpoint_distance_m": float(midpoint_dist[j]),
                "fraction_along_segment": float(fraction[j]),
                "segment_length_m": float(graph.lengths[arc_id]),
                "heading_difference_rad": float(angle[j]),
                "gps_speed_mps": float(speed[original_idx]),
                "time_gap_s": gap,
                "from_node_out_degree": int(graph.from_node_out_degree[arc_id]),
                "to_node_in_degree": int(graph.to_node_in_degree[arc_id]),
            })
    return pd.DataFrame(candidate_rows), pd.DataFrame(observation_rows)
