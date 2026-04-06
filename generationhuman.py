"""
Human Patrol Path Generator — Etosha National Park
====================================================
Generates loop patrol paths for human patrol officers. Uses 8-directional
movement (cardinal + diagonal) to allow more natural patrol routes.
Path length is shorter than the drone (default 5 cells) since humans
move slower and cover less ground per patrol.

Ranked by avg_path_danger — rewards patrols that stay in the hottest zones.

Outputs
-------
  human_paths_raw.csv       — all generated loop paths + metrics
  human_paths_filtered.csv  — top N by avg_path_danger, deduplicated
  human_paths_summary.txt   — distribution + top 20

Usage
-----
  python human_path_generator.py \
      --fire   fire_risk_5km.csv \
      --animal animal_value_5km.csv \
      --max-len 5 \
      --top-n  200
"""

import argparse
import time
from collections import deque

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
PATROL_EFFICIENCY = 0.448      # same as patrol_person in optimizer
SENSOR_RANGE_KM   = 0.5        # human sensor range — shorter than drone
MAX_INFLUENCE_KM  = 100 * SENSOR_RANGE_KM
CELL_SIZE_KM      = 5.0
MAX_PATH_LEN      = 3          # humans cover less ground
MINUTES_PER_CELL  = 30/3         # rough estimate: 30 min per 5km cell on foot
coeff = 1.5

# 8-directional movement
DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]


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

    danger  = (2- coeff)*(1-norm(fire_arr)) + (coeff) * norm(animal_arr)
    in_park = ~(np.isnan(fire_arr) | np.isnan(animal_arr))

    return danger, in_park, fire_df.index.values, fire_df.columns.values


# ─────────────────────────────────────────────────────────────────────────────
# 2. PATH GENERATION — 8-directional, loops only
# ─────────────────────────────────────────────────────────────────────────────

def generate_loop_paths(in_park: np.ndarray, max_len: int = MAX_PATH_LEN):
    """
    Yield all simple loop paths of length 3..max_len that:
      - stay entirely within the park
      - use 8-directional moves (cardinal + diagonal)
      - end on a cell 8-directionally adjacent to the start
      - have no repeated cells
    """
    valid = set(zip(*np.where(in_park)))
    stack = deque()
    for start in valid:
        stack.append(([start], {start}))

    while stack:
        path, visited = stack.pop()
        current = path[-1]

        # Close the loop — last cell must be 8-adjacent to first
        if len(path) >= 3:
            sr, sc = path[0]
            cr, cc = current
            dr = abs(cr - sr)
            dc = abs(cc - sc)
            if dr <= 1 and dc <= 1 and (dr + dc) > 0:
                yield path

        # Keep extending
        if len(path) < max_len:
            r, c = current
            for dr, dc in DIRECTIONS:
                nb = (r + dr, c + dc)
                if nb in valid and nb not in visited:
                    stack.append((path + [nb], visited | {nb}))


# ─────────────────────────────────────────────────────────────────────────────
# 3. METRICS
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


def compute_metrics(path, danger, in_park, rows_km, cols_km) -> dict:
    n_cells   = len(path)
    length_km = path_length_km(path, rows_km, cols_km)
    duration  = n_cells * MINUTES_PER_CELL

    path_danger_vals = [danger[r, c] for r, c in path]
    avg_path_danger  = float(np.mean(path_danger_vals))
    max_path_danger  = float(np.max(path_danger_vals))

    r_idx, c_idx = np.where(in_park)
    cell_rows    = rows_km[r_idx]
    cell_cols    = cols_km[c_idx]

    path_rows = np.array([rows_km[r] for r, c in path])
    path_cols = np.array([cols_km[c] for r, c in path])

    dr       = cell_rows[:, None] - path_rows[None, :]
    dc       = cell_cols[:, None] - path_cols[None, :]
    min_dist = np.sqrt(dr**2 + dc**2).min(axis=1)

    raw_cov = PATROL_EFFICIENCY * np.exp(-min_dist / SENSOR_RANGE_KM)
    raw_cov[min_dist > MAX_INFLUENCE_KM] = 0.0

    danger_vals    = danger[r_idx, c_idx]
    cov            = np.minimum(raw_cov, danger_vals)
    danger_removed = float(cov.sum())

    meaningful   = (cov / np.where(danger_vals > 0, danger_vals, 1)) > 0.05
    coverage_km2 = float(meaningful.sum() * CELL_SIZE_KM**2)

    return {
        "n_cells":          n_cells,
        "path_km":          round(length_km, 3),
        "duration_min":     round(duration, 2),
        "avg_path_danger":  round(avg_path_danger, 6),
        "max_path_danger":  round(max_path_danger, 6),
        "danger_removed":   round(danger_removed, 6),
        "coverage_km2":     round(coverage_km2, 1),
        "path_str":         path_to_str(path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def path_to_str(path) -> str:
    return ";".join(f"{r},{c}" for r, c in path)

def str_to_path(s: str):
    return [tuple(int(x) for x in p.split(",")) for p in s.split(";")]

def canonical(path_str: str) -> str:
    cells = sorted(path_str.split(";"))
    return ";".join(cells)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fire",         default="fire_risk_5km.csv")
    parser.add_argument("--animal",       default="animal_value_5km.csv")
    parser.add_argument("--max-len",      type=int, default=MAX_PATH_LEN)
    parser.add_argument("--top-n",        type=int, default=200)
    parser.add_argument("--out-raw",      default="human_paths_raw.csv")
    parser.add_argument("--out-filtered", default="human_paths_filtered_1.5.csv")
    parser.add_argument("--out-summary",  default="human_paths_summary.txt")
    args = parser.parse_args()

    print("=" * 60)
    print("  Human Patrol Path Generator — Etosha")
    print("=" * 60)

    # Load
    print("\n[1/4] Loading risk grids ...")
    danger, in_park, rows_km, cols_km = build_danger(args.fire, args.animal)
    n_valid      = int(in_park.sum())
    total_danger = float(np.nansum(danger))
    print(f"      Grid        : {danger.shape[0]} x {danger.shape[1]}")
    print(f"      Park cells  : {n_valid}")
    print(f"      Total danger: {total_danger:.3f}")
    print(f"      Danger max  : {np.nanmax(danger):.3f}")
    print(f"      Danger mean : {np.nanmean(danger[in_park]):.3f}")

    # Generate
    print(f"\n[2/4] Generating loop paths (max {args.max_len} cells, 8-directional) ...")
    t0 = time.time()
    records, n_generated = [], 0

    for path in generate_loop_paths(in_park, max_len=args.max_len):
        records.append(compute_metrics(path, danger, in_park, rows_km, cols_km))
        n_generated += 1
        if n_generated % 10_000 == 0:
            print(f"      {n_generated:>8,} loops  ({time.time()-t0:.1f}s) ...")

    elapsed = time.time() - t0
    print(f"\n      Done — {n_generated:,} loops in {elapsed:.1f}s")

    if n_generated == 0:
        print("\n  No loops found. Try --max-len 4 or check your CSVs.")
        return

    # Save raw
    print(f"\n[3/4] Saving raw -> {args.out_raw} ...")
    col_order = ["n_cells", "path_km", "duration_min",
                 "avg_path_danger", "max_path_danger",
                 "danger_removed", "coverage_km2", "path_str"]
    df_raw = pd.DataFrame(records)[col_order]
    df_raw.to_csv(args.out_raw, index=False)
    print(f"      {len(df_raw):,} rows saved")

    # Filter
    print(f"\n[4/4] Deduplicating, keeping top {args.top_n} by avg_path_danger ...")
    df_raw["_canon"] = df_raw["path_str"].apply(canonical)
    df_filt = (df_raw
               .sort_values("avg_path_danger", ascending=False)
               .drop_duplicates(subset="_canon")
               .drop(columns="_canon")
               .head(args.top_n)
               .reset_index(drop=True))

    df_filt.to_csv(args.out_filtered, index=False)
    print(f"      {len(df_filt)} unique routes -> {args.out_filtered}")

    # Summary
    lines = []
    lines.append("=" * 60)
    lines.append("  Human Patrol Path Generator — Summary")
    lines.append("=" * 60)
    lines.append(f"\nGrid              : {danger.shape[0]} x {danger.shape[1]}")
    lines.append(f"Park cells        : {n_valid}")
    lines.append(f"Total danger      : {total_danger:.3f}")
    lines.append(f"Max path length   : {args.max_len} cells")
    lines.append(f"Movement          : 8-directional (cardinal + diagonal)")
    lines.append(f"Sensor range      : {SENSOR_RANGE_KM} km")
    lines.append(f"Ranking metric    : avg_path_danger")
    lines.append(f"Loops generated   : {n_generated:,}")
    lines.append(f"After dedup+filter: {len(df_filt)}")
    lines.append(f"Runtime           : {elapsed:.1f}s")
    lines.append(f"\nPath length distribution (all loops):")
    lines.append(df_raw["n_cells"].value_counts().sort_index().to_string())
    lines.append(f"\nTop 20 by avg_path_danger:")
    lines.append(df_filt.head(20).drop(columns="path_str").to_string())

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(args.out_summary, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary -> {args.out_summary}")
    print("Done.\n")


if __name__ == "__main__":
    main()