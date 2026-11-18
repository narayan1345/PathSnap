"""PathSnap dataset inventory, quality checks and Folium sample map.
Run from the project root:
python src/data_exploration.py --data data/raw --output results --map visualizations/sample_trip_map.html
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

FILES = {"track": ".track", "nodes": ".nodes", "arcs": ".arcs", "route": ".route"}

def find_partition(root: Path, name: str) -> Path:
    for path in (root / name, root / "Student folder" / name, root / "student folder" / name):
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Cannot find {name!r} under {root}")

def trip_dirs(partition: Path) -> list[Path]:
    return sorted(p for p in partition.iterdir() if p.is_dir())

def read_trip(folder: Path):
    tid = folder.name
    track = pd.read_csv(folder / f"{tid}.track", sep=r"\s+", header=None,
                        names=["longitude", "latitude", "timestamp"])
    nodes = pd.read_csv(folder / f"{tid}.nodes", sep=r"\s+", header=None,
                        names=["longitude", "latitude"])
    arcs = pd.read_csv(folder / f"{tid}.arcs", sep=r"\s+", header=None,
                       names=["from_node_id", "to_node_id"])
    route = pd.read_csv(folder / f"{tid}.route", sep=r"\s+", header=None,
                        names=["arc_id"])
    return track, nodes, arcs, route

def graph_statistics(nodes: pd.DataFrame, arcs: pd.DataFrame) -> dict:
    n = len(nodes)
    if n == 0 or arcs.empty:
        return {"weak_components": n, "isolated_nodes": n, "mean_in_degree": 0.0,
                "mean_out_degree": 0.0}
    src = arcs.from_node_id.to_numpy(dtype=np.int64)
    dst = arcs.to_node_id.to_numpy(dtype=np.int64)
    valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
    src, dst = src[valid], dst[valid]
    matrix = coo_matrix((np.ones(len(src), dtype=np.uint8), (src, dst)),
                        shape=(n, n)).tocsr()
    undirected = matrix.maximum(matrix.T).tocsr()
    components, _ = connected_components(undirected, directed=False)
    out_degree = np.bincount(src, minlength=n)
    in_degree = np.bincount(dst, minlength=n)
    return {"weak_components": int(components),
            "isolated_nodes": int(((out_degree + in_degree) == 0).sum()),
            "mean_in_degree": float(in_degree.mean()),
            "mean_out_degree": float(out_degree.mean())}

def inspect_partition(partition: Path, label: str):
    rows, missing = [], []
    for folder in trip_dirs(partition):
        tid = folder.name
        expected = [folder / f"{tid}{ext}" for ext in FILES.values()]
        absent = [p.name for p in expected if not p.is_file()]
        if absent:
            missing.append({"partition": label, "trip_id": tid, "missing": ";".join(absent)})
            continue
        track, nodes, arcs, route = read_trip(folder)
        dt = track.timestamp.diff().dropna()
        valid_coords = (np.isfinite(track.longitude) & np.isfinite(track.latitude)
                        & track.longitude.between(-180, 180)
                        & track.latitude.between(-90, 90))
        valid_route = route.arc_id.between(0, len(arcs) - 1)
        valid_nodes = (arcs.from_node_id.between(0, len(nodes) - 1)
                       & arcs.to_node_id.between(0, len(nodes) - 1))
        route_breaks = np.nan
        if valid_route.all() and len(route) > 1:
            selected = arcs.iloc[route.arc_id.astype(int).to_numpy()]
            route_breaks = int((selected.to_node_id.to_numpy()[:-1]
                                != selected.from_node_id.to_numpy()[1:]).sum())
        graph = graph_statistics(nodes, arcs) if valid_nodes.all() else {
            "weak_components": np.nan, "isolated_nodes": np.nan,
            "mean_in_degree": np.nan, "mean_out_degree": np.nan}
        rows.append({
            "partition": label, "trip_id": tid, "track_rows": len(track),
            "nodes": len(nodes), "arcs": len(arcs), "route_rows": len(route),
            "track_missing_cells": int(track.isna().sum().sum()),
            "invalid_coordinate_rows": int((~valid_coords).sum()),
            "timestamp_monotonic": bool(track.timestamp.is_monotonic_increasing),
            "duplicate_timestamps": int(track.timestamp.duplicated().sum()),
            "median_time_step_s": float(dt.median()) if len(dt) else np.nan,
            "max_time_gap_s": float(dt.max()) if len(dt) else np.nan,
            "time_gaps_over_1s": int((dt > 1).sum()),
            "route_arc_ids_in_range": bool(valid_route.all()),
            "arcs_node_ids_in_range": bool(valid_nodes.all()),
            "route_connectivity_breaks": route_breaks,
            "longitude_min": float(track.longitude.min()),
            "longitude_max": float(track.longitude.max()),
            "latitude_min": float(track.latitude.min()),
            "latitude_max": float(track.latitude.max()),
            **graph,
        })
    return rows, missing

def create_sample_map(folder: Path, output: Path):
    import folium
    track, nodes, arcs, route = read_trip(folder)
    m = folium.Map(location=[float(track.latitude.mean()), float(track.longitude.mean())],
                   zoom_start=14, control_scale=True)
    gps_group = folium.FeatureGroup(name="Observed GPS trajectory")
    gps = track[["latitude", "longitude"]].dropna().values.tolist()
    if gps:
        folium.PolyLine(gps, color="blue", weight=3, tooltip="Observed GPS").add_to(gps_group)
        stride = max(1, len(gps) // 100)
        for i, (lat, lon) in enumerate(gps[::stride]):
            folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True,
                                tooltip=f"GPS sample {i * stride}").add_to(gps_group)
    gps_group.add_to(m)
    route_group = folium.FeatureGroup(name="Reference route")
    route_coords = []
    for aid in route.arc_id.astype(int):
        if 0 <= aid < len(arcs):
            u, v = map(int, arcs.iloc[aid][["from_node_id", "to_node_id"]])
            if 0 <= u < len(nodes) and 0 <= v < len(nodes):
                route_coords.extend([
                    [float(nodes.iloc[u].latitude), float(nodes.iloc[u].longitude)],
                    [float(nodes.iloc[v].latitude), float(nodes.iloc[v].longitude)]])
    if route_coords:
        folium.PolyLine(route_coords, color="green", weight=4,
                        tooltip="Reference route (.route → .arcs → .nodes)").add_to(route_group)
    route_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    return {"trip_id": folder.name, "gps_points": len(track),
            "route_arcs": len(route), "map_file": str(output)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--output", default="results")
    parser.add_argument("--map", default="visualizations/sample_trip_map.html")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows, all_missing = [], []
    for part in ("train", "student_test"):
        folder = find_partition(Path(args.data), part)
        rows, missing = inspect_partition(folder, part)
        all_rows.extend(rows)
        all_missing.extend(missing)
        print(f"{part}: {len(trip_dirs(folder))} folders; {len(rows)} complete")
    df = pd.DataFrame(all_rows)
    df.to_csv(output / "trip_inventory.csv", index=False)
    pd.DataFrame(all_missing, columns=["partition", "trip_id", "missing"]).to_csv(
        output / "missing_files.csv", index=False)
    summary = {
        "trip_counts": df.groupby("partition").size().to_dict(),
        "total_trips_inspected": int(len(df)),
        "missing_file_trip_count": len(all_missing),
        "track_observations_total": int(df.track_rows.sum()),
        "nodes_total": int(df.nodes.sum()), "arcs_total": int(df.arcs.sum()),
        "route_entries_total": int(df.route_rows.sum()),
        "invalid_coordinate_rows_total": int(df.invalid_coordinate_rows.sum()),
        "non_monotonic_timestamp_trips": int((~df.timestamp_monotonic).sum()),
        "route_ids_out_of_range_trips": int((~df.route_arc_ids_in_range).sum()),
        "arc_node_ids_out_of_range_trips": int((~df.arcs_node_ids_in_range).sum()),
        "total_route_connectivity_breaks": int(df.route_connectivity_breaks.fillna(0).sum())}
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    train = find_partition(Path(args.data), "train")
    map_info = create_sample_map(trip_dirs(train)[0], Path(args.map))
    print("Sample map:", json.dumps(map_info, indent=2))

if __name__ == "__main__":
    main()
