"""
Drone Path Generator — Etosha National Park
============================================
Generates cardinal loop patrol paths up to 7 cells long and scores them
by how many distinct danger hotspot clusters they connect.

Core idea
---------
A patrol drone's unique value over a stationary drone is that it can
surveil multiple separated hotspots in a single flight. We find N danger
clusters in the park, then score each path by:

    cluster_score = danger_removed * clusters_hit / path_km

Paths that thread through multiple clusters beat tight loops that intensively
cover just one, even if raw danger_removed is similar.

Outputs
-------
  drone_paths_raw.csv       — all generated loop paths + metrics
  drone_paths_filtered.csv  — top N by cluster_score, deduplicated
  drone_paths_summary.txt   — distribution + top 20

Usage
-----
  python drone_path_generator.py \
      --fire   fire_risk_5km.csv \
      --animal animal_value_5km.csv \
      --n-clusters 8 \
      --top-n  200
"""

import argparse
import time
from collections import deque

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ── Constants ─────────────────────────────────────────────────────────────────
DRONE_EFFICIENCY  = 0.70
SENSOR_RANGE_KM   = 7.0
MAX_INFLUENCE_KM  = 3 * SENSOR_RANGE_KM
CELL_SIZE_KM      = 5.0
MAX_PATH_LEN      = 7
MINUTES_PER_CELL  = 30 / 7

# Cluster reach: how close a path must come to a cluster centre to "hit" it
CLUSTER_REACH_KM  = 15.0   # ~3 cells — generous to reward near-misses too

# Cardinal moves
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_grid(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index   = df.index.astype(float)
    df.columns = df.columns.astype(float)
    return df


def reindex_nearest(df_src, ref_index, ref_columns, tolerance_km=3.0):
    src_rows = df_src.index.values
    src_cols = df_src.columns.values

    def snap(ref_vals, src_vals):
        snapped = []
        for v in ref_vals:
            diffs = np.abs(src_vals - v)
            i = diffs.argmin()
            snapped.append(src_vals[i] if diffs[i] <= tolerance_km else np.nan)
        return snapped

    snapped_rows = snap(ref_index, src_rows)
    snapped_cols = snap(ref_columns, src_cols)

    valid_src_rows = [r for r in snapped_rows if not np.isnan(r)]
    valid_src_cols = [c for c in snapped_cols if not np.isnan(c)]
    resampled = df_src.reindex(index=valid_src_rows, columns=valid_src_cols)

    valid_ref_rows = [ref_index[i]   for i, r in enumerate(snapped_rows) if not np.isnan(r)]
    valid_ref_cols = [ref_columns[i] for i, c in enumerate(snapped_cols) if not np.isnan(c)]
    resampled.index   = valid_ref_rows
    resampled.columns = valid_ref_cols

    return resampled.reindex(index=ref_index, columns=ref_columns)


def build_danger(fire_csv: str, animal_csv: str):
    fire_df   = load_grid(fire_csv)
    animal_df = load_grid(animal_csv)

    ref_index   = fire_df.index.values
    ref_columns = fire_df.columns.values
    animal_aligned = reindex_nearest(animal_df, ref_index, ref_columns)

    fire_arr   = fire_df.values.astype(float)
    animal_arr = animal_aligned.values.astype(float)

    def norm(arr):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

    danger  = norm(fire_arr) + norm(animal_arr)
    in_park = ~(np.isnan(fire_arr) | np.isnan(animal_arr))

    return danger, in_park, fire_df.index.values, fire_df.columns.values


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def find_clusters(danger, in_park, rows_km, cols_km, n_clusters: int,
                  danger_threshold_pct: float = 0.5):
    """
    K-means cluster the top danger cells (above danger_threshold_pct percentile).
    Returns cluster centres as (northing_km, easting_km) array, shape (n_clusters, 2).
    Weighted by danger value so centres land in the densest hotspots.
    """
    r_idx, c_idx = np.where(in_park)
    danger_vals  = danger[r_idx, c_idx]

    # Only cluster the high-danger cells
    threshold = np.percentile(danger_vals, danger_threshold_pct * 100)
    mask      = danger_vals >= threshold
    r_hot     = r_idx[mask]
    c_hot     = c_idx[mask]
    d_hot     = danger_vals[mask]

    coords  = np.column_stack([rows_km[r_hot], cols_km[c_hot]])
    weights = d_hot / d_hot.sum()

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(coords, sample_weight=weights)

    centres = km.cluster_centers_   # shape (n_clusters, 2)
    labels  = km.labels_

    # Danger mass per cluster (for reporting)
    cluster_danger = np.array([d_hot[labels == k].sum() for k in range(n_clusters)])

    return centres, cluster_danger


# ─────────────────────────────────────────────────────────────────────────────
# 3. PATH GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_loop_paths(in_park: np.ndarray, max_len: int = MAX_PATH_LEN):
    """
    Yield all simple cardinal loop paths of length 3..max_len staying in park.
    Loop closes when last cell is cardinally adjacent to first cell.
    """
    valid = set(zip(*np.where(in_park)))
    stack = deque()
    for start in valid:
        stack.append(([start], {start}))

    while stack:
        path, visited = stack.pop()
        current = path[-1]

        if len(path) >= 3:
            sr, sc = path[0]
            cr, cc = current
            dr, dc = abs(cr - sr), abs(cc - sc)
            if (dr == 1 and dc == 0) or (dr == 0 and dc == 1):
                yield path

        if len(path) < max_len:
            r, c = current
            for dr, dc in DIRECTIONS:
                nb = (r + dr, c + dc)
                if nb in valid and nb not in visited:
                    stack.append((path + [nb], visited | {nb}))


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def path_length_km(path, rows_km, cols_km) -> float:
    pts   = path + [path[0]]
    total = 0.0
    for i in range(1, len(pts)):
        r0, c0 = pts[i - 1]
        r1, c1 = pts[i]
        dy = rows_km[r1] - rows_km[r0]
        dx = cols_km[c1] - cols_km[c0]
        total += np.sqrt(dy**2 + dx**2)
    return total


def clusters_hit(path, rows_km, cols_km, cluster_centres,
                 reach_km: float = CLUSTER_REACH_KM) -> int:
    """
    Count how many cluster centres have at least one path cell within reach_km.
    """
    path_rows = np.array([rows_km[r] for r, c in path])
    path_cols = np.array([cols_km[c] for r, c in path])

    hit = 0
    for cy, cx in cluster_centres:
        dy = path_rows - cy
        dx = path_cols - cx
        if np.sqrt(dy**2 + dx**2).min() <= reach_km:
            hit += 1
    return hit


def compute_metrics(path, danger, in_park, rows_km, cols_km,
                    cluster_centres) -> dict:
    n_cells   = len(path)
    length_km = path_length_km(path, rows_km, cols_km)
    duration  = n_cells * MINUTES_PER_CELL

    r_idx, c_idx = np.where(in_park)
    cell_rows    = rows_km[r_idx]
    cell_cols    = cols_km[c_idx]

    path_rows = np.array([rows_km[r] for r, c in path])
    path_cols = np.array([cols_km[c] for r, c in path])

    dr       = cell_rows[:, None] - path_rows[None, :]
    dc       = cell_cols[:, None] - path_cols[None, :]
    min_dist = np.sqrt(dr**2 + dc**2).min(axis=1)

    raw_cov = DRONE_EFFICIENCY * np.exp(-min_dist / SENSOR_RANGE_KM)
    raw_cov[min_dist > MAX_INFLUENCE_KM] = 0.0

    danger_vals    = danger[r_idx, c_idx]
    cov            = np.minimum(raw_cov, danger_vals)
    danger_removed = float(cov.sum())

    meaningful   = (cov / np.where(danger_vals > 0, danger_vals, 1)) > 0.05
    coverage_km2 = float(meaningful.sum() * CELL_SIZE_KM**2)

    n_clusters_hit = clusters_hit(path, rows_km, cols_km, cluster_centres)

    # cluster_score rewards paths that:
    #   - remove more danger (danger_removed)
    #   - connect more distinct hotspots (n_clusters_hit, min 1 so score != 0)
    #   - do so efficiently per km flown (/path_km)
    score = (danger_removed * max(n_clusters_hit, 1)) / length_km if length_km > 0 else 0.0

    return {
        "n_cells":         n_cells,
        "path_km":         round(length_km, 3),
        "duration_min":    round(duration, 2),
        "danger_removed":  round(danger_removed, 6),
        "coverage_km2":    round(coverage_km2, 1),
        "clusters_hit":    n_clusters_hit,
        "cluster_score":   round(score, 4),
        "path_str":        path_to_str(path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def path_to_str(path) -> str:
    return ";".join(f"{r},{c}" for r, c in path)

def str_to_path(s: str):
    return [tuple(int(x) for x in p.split(",")) for p in s.split(";")]

def canonical(path_str: str) -> str:
    cells = sorted(path_str.split(";"))
    return ";".join(cells)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fire",         default="fire_risk_5km.csv")
    parser.add_argument("--animal",       default="animal_value_5km.csv")
    parser.add_argument("--max-len",      type=int,   default=MAX_PATH_LEN)
    parser.add_argument("--n-clusters",   type=int,   default=8,
                        help="Number of danger hotspot clusters to identify")
    parser.add_argument("--top-n",        type=int,   default=200)
    parser.add_argument("--out-raw",      default="drone_paths_raw.csv")
    parser.add_argument("--out-filtered", default="drone_paths_filtered.csv")
    parser.add_argument("--out-summary",  default="drone_paths_summary.txt")
    args = parser.parse_args()

    print("=" * 60)
    print("  Drone Loop Path Generator -- Etosha (cluster-scoring)")
    print("=" * 60)

    # Load
    print("\n[1/5] Loading risk grids ...")
    danger, in_park, rows_km, cols_km = build_danger(args.fire, args.animal)
    n_valid      = int(in_park.sum())
    total_danger = float(np.nansum(danger))
    print(f"      Grid        : {danger.shape[0]} x {danger.shape[1]}")
    print(f"      Park cells  : {n_valid}")
    print(f"      Total danger: {total_danger:.3f}")

    # Cluster
    print(f"\n[2/5] Finding {args.n_clusters} danger hotspot clusters ...")
    cluster_centres, cluster_danger = find_clusters(
        danger, in_park, rows_km, cols_km, n_clusters=args.n_clusters)
    for k, (centre, dmass) in enumerate(zip(cluster_centres, cluster_danger)):
        print(f"      Cluster {k+1:>2}: centre ({centre[0]:+.1f}, {centre[1]:+.1f}) km  "
              f"danger mass {dmass:.2f}")

    # Generate
    print(f"\n[3/5] Generating cardinal loop paths (max {args.max_len} cells) ...")
    t0 = time.time()
    records, n_generated = [], 0

    for path in generate_loop_paths(in_park, max_len=args.max_len):
        records.append(compute_metrics(
            path, danger, in_park, rows_km, cols_km, cluster_centres))
        n_generated += 1
        if n_generated % 5_000 == 0:
            print(f"      {n_generated:>8,} loops  ({time.time()-t0:.1f}s) ...")

    elapsed = time.time() - t0
    print(f"\n      Done -- {n_generated:,} loops in {elapsed:.1f}s")

    if n_generated == 0:
        print("\n  No loops found. Try --max-len 5 or check your CSVs.")
        return

    # Save raw
    print(f"\n[4/5] Saving raw -> {args.out_raw} ...")
    col_order = ["n_cells", "path_km", "duration_min",
                 "danger_removed", "coverage_km2", "clusters_hit",
                 "cluster_score", "path_str"]
    df_raw = pd.DataFrame(records)[col_order]
    df_raw.to_csv(args.out_raw, index=False)
    print(f"      {len(df_raw):,} rows saved")

    # Filter
    print(f"\n[5/5] Deduplicating, keeping top {args.top_n} by cluster_score ...")
    df_raw["_canon"] = df_raw["path_str"].apply(canonical)
    df_filt = (df_raw
               .sort_values("cluster_score", ascending=False)
               .drop_duplicates(subset="_canon")
               .drop(columns="_canon")
               .head(args.top_n)
               .reset_index(drop=True))

    df_filt.to_csv(args.out_filtered, index=False)
    print(f"      {len(df_filt)} unique routes -> {args.out_filtered}")

    # Summary
    lines = []
    lines.append("=" * 60)
    lines.append("  Drone Loop Path Generator -- Summary")
    lines.append("=" * 60)
    lines.append(f"\nGrid              : {danger.shape[0]} x {danger.shape[1]}")
    lines.append(f"Park cells        : {n_valid}")
    lines.append(f"Total danger      : {total_danger:.3f}")
    lines.append(f"Clusters          : {args.n_clusters}  (reach {CLUSTER_REACH_KM} km)")
    lines.append(f"Max path length   : {args.max_len} cells")
    lines.append(f"Loops generated   : {n_generated:,}")
    lines.append(f"After dedup+filter: {len(df_filt)}")
    lines.append(f"Runtime           : {elapsed:.1f}s")
    lines.append(f"\nCluster centres and danger mass:")
    for k, (centre, dmass) in enumerate(zip(cluster_centres, cluster_danger)):
        lines.append(f"  {k+1}: ({centre[0]:+.1f}, {centre[1]:+.1f}) km  mass={dmass:.2f}")
    lines.append(f"\nPath length distribution (all loops):")
    lines.append(df_raw["n_cells"].value_counts().sort_index().to_string())
    lines.append(f"\nclusters_hit distribution (filtered top {args.top_n}):")
    lines.append(df_filt["clusters_hit"].value_counts().sort_index().to_string())
    lines.append(f"\nTop 20 by cluster_score:")
    lines.append(df_filt.head(20).drop(columns="path_str").to_string())

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(args.out_summary, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary -> {args.out_summary}")
    print("Done.\n")


if __name__ == "__main__":
    main()