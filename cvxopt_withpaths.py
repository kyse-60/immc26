"""
Etosha National Park — Hybrid Patrol Path Optimizer
====================================================
Solver: cvxopt.glpk.ilp
 
HYBRID MODEL
------------
  Fixed-cell placement:
    - stationary_person  : binary x_sp[k], placed at grid cell k
    - camera             : binary x_cam[k], placed at grid cell k
 
  Path selection:
    - drone              : binary x_dp[p], loop from drone_paths_filtered.csv
    - patrol_person      : binary x_hp[p], loop from human_paths_filtered.csv
 
CRITICAL GRID ALIGNMENT NOTE
-----------------------------
The generation scripts (generation.py, generationhuman.py) use fire_df
(31x67) as the reference grid and align animal onto it. The path (r,c)
indices therefore index into the fire_df grid. This optimizer must use
the same reference grid so that path waypoints map to the correct km
coordinates. We align animal onto fire_df, matching the generation scripts.
 
ASSET VALUES (from immc_protect_vals.csv)
-----------------------------------------
  Asset              Efficiency  Eff.range(km)  Cost($/week)
  stationary_person  0.64        3.0            750
  camera             0.50        2.5            20
  drone (path)       0.35        22.0           325
  patrol (path)      0.448       14.0           1190
 
NUMERICAL SCALING
-----------------
Costs are normalised to units of $100 internally to reduce the
condition number of the constraint matrix (was ~10^12, now ~10^4).
Budget is scaled accordingly.
 
VARIABLE LAYOUT
---------------
  [OFF_SP  .. OFF_CAM-1]   n_cells       binary  stationary person per cell
  [OFF_CAM .. OFF_DP-1]    n_cells       binary  camera per cell
  [OFF_DP  .. OFF_HP-1]    n_drone_paths binary  drone path selected
  [OFF_HP  .. OFF_COV-1]   n_human_paths binary  human path selected
  [OFF_COV .. end]         n_cells       continuous total_cov[i]
"""
 
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import cvxopt
from cvxopt import matrix, spmatrix
from cvxopt import glpk
 
# ================================================================
# 1.  ASSET PARAMETERS
# ================================================================
COST_UNIT = 100.0   # normalise all costs to units of $100
 
FIXED_ASSETS = {
    "stationary_person": {
        "efficiency": 1,
        "range_km":   0.5,
        "cost_raw":   160,
        "max_total":  100,
    },
    "camera": {
        "efficiency": 0.50,
        "range_km":   3,
        "cost_raw":   20,
        "max_total":  100,
    },
}
for p in FIXED_ASSETS.values():
    p["cost"] = p["cost_raw"] / COST_UNIT
 
DRONE_PARAMS = {"efficiency": 0.35,  "eff_range_km": 4, "cost_raw": 325,  "cost": 325  / COST_UNIT}
HUMAN_PARAMS = {"efficiency": 1, "eff_range_km": 0.5, "cost_raw": 400, "cost": 400 / COST_UNIT}
 
MAX_DRONE_PATHS  = 100
MAX_HUMAN_PATHS  = 100
MAX_INFLUENCE_KM = 55.0
 
# ================================================================
# 2.  PROBLEM SETTINGS
# ================================================================
BUDGET_RAW    = 10000          # $ per week
BUDGET        = BUDGET_RAW / COST_UNIT   # in cost units
TIME_LIMIT_MS = 2000_000
MIP_GAP       = 0.05             # slightly relaxed for faster solve
 
# ================================================================
# 3.  LOAD GRIDS — fire_df is the REFERENCE (matches generation scripts)
# ================================================================
fire_df   = pd.read_csv("fire_risk_5km.csv",    index_col=0)
animal_df = pd.read_csv("animal_value_5km.csv", index_col=0)
 
fire_df.index     = fire_df.index.astype(float).round(4)
fire_df.columns   = fire_df.columns.astype(float).round(4)
animal_df.index   = animal_df.index.astype(float).round(4)
animal_df.columns = animal_df.columns.astype(float).round(4)
 
print(f"fire_risk_5km    shape : {fire_df.shape}  <- reference grid (matches generation scripts)")
print(f"animal_value_5km shape : {animal_df.shape}")
 
 
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
    valid_ref_rows = [ref_index[i] for i, r in enumerate(snapped_rows) if not np.isnan(r)]
    valid_ref_cols = [ref_columns[i] for i, c in enumerate(snapped_cols) if not np.isnan(c)]
    resampled.index   = valid_ref_rows
    resampled.columns = valid_ref_cols
    return resampled.reindex(index=ref_index, columns=ref_columns)
 
 
# Align animal onto fire grid — same as generation.py
animal_aligned = reindex_nearest(
    animal_df,
    ref_index=fire_df.index.values,
    ref_columns=fire_df.columns.values,
)
print(f"animal aligned to fire grid: {animal_aligned.shape}")
 
# ================================================================
# 4.  BUILD DANGER MAP  (identical logic to generation.py)
# ================================================================
fire_arr   = fire_df.values.astype(float)
animal_arr = animal_aligned.values.astype(float)
 
inside_park = ~(np.isnan(fire_arr) | np.isnan(animal_arr))
 
GRID_ROWS, GRID_COLS = fire_arr.shape
 
 
def norm(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
 
 
fire_norm_2d   = norm(fire_arr)
animal_norm_2d = norm(animal_arr)
 
danger = fire_norm_2d + animal_norm_2d   # [0, 2], NaN outside park
danger[~inside_park] = 0.0
 
# rows_km / cols_km are from fire_df — same as generation scripts
rows_km = fire_df.index.values.astype(float)
cols_km = fire_df.columns.values.astype(float)
 
valid_cells  = [(r, c) for r in range(GRID_ROWS)
                        for c in range(GRID_COLS) if inside_park[r, c]]
valid_coords = np.array([(rows_km[r], cols_km[c]) for r, c in valid_cells])
n_cells      = len(valid_cells)
danger_flat  = np.array([danger[r, c] for r, c in valid_cells])
 
print(f"\nGrid          : {GRID_ROWS} x {GRID_COLS}  (matches generation scripts)")
print(f"Cells in park : {n_cells}")
print(f"Total danger  : {danger_flat.sum():.3f}")
print(f"Budget        : ${BUDGET_RAW:,}  ({BUDGET:.1f} cost-units of ${COST_UNIT:.0f})")
 
# ================================================================
# 5.  LOAD CANDIDATE PATHS
# ================================================================
 
def str_to_path(s: str):
    return [tuple(int(x) for x in p.split(",")) for p in s.split(";")]
 
 
print("\nLoading candidate paths ...")
drone_df = pd.read_csv("drone_paths_filtered.csv")
human_df = pd.read_csv("human_paths_filtered.csv")
 
drone_paths   = [str_to_path(s) for s in drone_df["path_str"]]
human_paths   = [str_to_path(s) for s in human_df["path_str"]]
n_drone_paths = len(drone_paths)
n_human_paths = len(human_paths)
 
print(f"  Drone paths : {n_drone_paths}")
print(f"  Human paths : {n_human_paths}")
 
# Verify grid alignment
all_paths_combined = drone_paths + human_paths
all_rs = [r for path in all_paths_combined for r, c in path]
all_cs = [c for path in all_paths_combined for r, c in path]
print(f"\n  Path index ranges:")
print(f"    row indices : {min(all_rs)} .. {max(all_rs)}  (grid has 0..{GRID_ROWS-1})")
print(f"    col indices : {min(all_cs)} .. {max(all_cs)}  (grid has 0..{GRID_COLS-1})")
oob = sum(1 for r in all_rs if r < 0 or r >= GRID_ROWS) + \
      sum(1 for c in all_cs if c < 0 or c >= GRID_COLS)
if oob:
    print(f"  WARNING: {oob} out-of-bounds indices — check grid alignment!")
else:
    print(f"  Grid alignment OK.")
 
# ================================================================
# 6.  PRECOMPUTE REMOVAL COEFFICIENTS
# ================================================================
# REMOVAL_THRESHOLD: values below this are zeroed out.
# Physically: 0.01 means <1% of a cell's danger removed — negligible.
# Numerically: eliminates tiny exp(-x) values that cause the 10^10
# matrix condition ratio which makes GLPK's branch-and-bound stall.
REMOVAL_THRESHOLD = 0.01
 
print(f"\nPrecomputing removal coefficients (threshold={REMOVAL_THRESHOLD}) ...")
 
dist_cc = cdist(valid_coords, valid_coords)
 
fixed_removal = {}
for aname, params in FIXED_ASSETS.items():
    mat = params["efficiency"] * np.exp(-dist_cc / params["range_km"])
    mat[dist_cc > MAX_INFLUENCE_KM] = 0.0
    mat[mat < REMOVAL_THRESHOLD] = 0.0
    fixed_removal[aname] = mat
    nz = (mat > 0).sum()
    mnz = mat[mat > 0].min() if nz > 0 else 0
    print(f"  {aname:22s}: {nz:,} nonzero  max={mat.max():.4f}  min_nonzero={mnz:.4f}")
 
 
def compute_path_removal(paths, efficiency, eff_range_km, label):
    removal = np.zeros((len(paths), n_cells), dtype=np.float64)
    for p_idx, path in enumerate(paths):
        path_coords = np.array([
            (rows_km[r], cols_km[c])
            for r, c in path
            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS
        ])
        if len(path_coords) == 0:
            continue
        dists = cdist(valid_coords, path_coords).min(axis=1)
        raw = efficiency * np.exp(-dists / eff_range_km)
        raw[dists > MAX_INFLUENCE_KM] = 0.0
        raw[raw < REMOVAL_THRESHOLD] = 0.0
        removal[p_idx] = raw
    nz = (removal > 0).sum()
    mnz = removal[removal > 0].min() if nz > 0 else 0
    print(f"  {label}: {len(paths)} paths  nonzero={nz:,}  max={removal.max():.4f}  min_nonzero={mnz:.4f}")
    return removal
 
 
drone_removal = compute_path_removal(
    drone_paths, DRONE_PARAMS["efficiency"], DRONE_PARAMS["eff_range_km"], "Drone paths"
)
human_removal = compute_path_removal(
    human_paths, HUMAN_PARAMS["efficiency"], HUMAN_PARAMS["eff_range_km"], "Human paths"
)
 
# ================================================================
# 7.  VARIABLE INDEX OFFSETS
# ================================================================
OFF_SP  = 0
OFF_CAM = n_cells
OFF_DP  = 2 * n_cells
OFF_HP  = OFF_DP + n_drone_paths
OFF_COV = OFF_HP + n_human_paths
n_vars  = OFF_COV + n_cells
n_x_bin = OFF_COV
 
print(f"\nVariable layout:")
print(f"  stationary_person : [{OFF_SP} .. {OFF_CAM-1}]  ({n_cells})")
print(f"  camera            : [{OFF_CAM} .. {OFF_DP-1}]  ({n_cells})")
print(f"  drone paths       : [{OFF_DP} .. {OFF_HP-1}]  ({n_drone_paths})")
print(f"  human paths       : [{OFF_HP} .. {OFF_COV-1}]  ({n_human_paths})")
print(f"  total_cov         : [{OFF_COV} .. {n_vars-1}]  ({n_cells})")
print(f"  Total             : {n_vars}  ({n_x_bin} binary + {n_cells} continuous)")
 
# ================================================================
# 8.  BUILD CONSTRAINT MATRIX
# ================================================================
print("\nBuilding constraint matrix ...")
 
row_vals, col_vals, data_vals = [], [], []
h_vals = []
current_row = 0
 
 
def add_constraint(row_i, col_j_list, coeff_list, rhs):
    for j, v in zip(col_j_list, coeff_list):
        row_vals.append(int(row_i))
        col_vals.append(int(j))
        data_vals.append(float(v))
    h_vals.append(float(rhs))
 
 
# 1. Coverage linking: total_cov[i] - sum_all removal*x <= 0
print("  Coverage-link ...")
for i in range(n_cells):
    cols_i = [OFF_COV + i]
    vals_i = [1.0]
 
    sp_col = fixed_removal["stationary_person"][:, i]
    for k in np.where(sp_col > 1e-9)[0]:
        cols_i.append(OFF_SP + int(k)); vals_i.append(-float(sp_col[k]))
 
    cam_col = fixed_removal["camera"][:, i]
    for k in np.where(cam_col > 1e-9)[0]:
        cols_i.append(OFF_CAM + int(k)); vals_i.append(-float(cam_col[k]))
 
    for p in range(n_drone_paths):
        v = float(drone_removal[p, i])
        if v > 1e-9:
            cols_i.append(OFF_DP + p); vals_i.append(-v)
 
    for p in range(n_human_paths):
        v = float(human_removal[p, i])
        if v > 1e-9:
            cols_i.append(OFF_HP + p); vals_i.append(-v)
 
    add_constraint(current_row, cols_i, vals_i, 0.0)
    current_row += 1
 
# 2. Budget (in cost-units)
print("  Budget ...")
bc, bv = [], []
for k in range(n_cells):
    bc.append(OFF_SP  + k); bv.append(FIXED_ASSETS["stationary_person"]["cost"])
    bc.append(OFF_CAM + k); bv.append(FIXED_ASSETS["camera"]["cost"])
for p in range(n_drone_paths):
    bc.append(OFF_DP + p); bv.append(DRONE_PARAMS["cost"])
for p in range(n_human_paths):
    bc.append(OFF_HP + p); bv.append(HUMAN_PARAMS["cost"])
add_constraint(current_row, bc, bv, float(BUDGET))
current_row += 1
 
# 3-6. Fleet limits
print("  Fleet limits ...")
add_constraint(current_row,
    [OFF_SP + k for k in range(n_cells)], [1.0]*n_cells,
    float(FIXED_ASSETS["stationary_person"]["max_total"]))
current_row += 1
 
add_constraint(current_row,
    [OFF_CAM + k for k in range(n_cells)], [1.0]*n_cells,
    float(FIXED_ASSETS["camera"]["max_total"]))
current_row += 1
 
add_constraint(current_row,
    [OFF_DP + p for p in range(n_drone_paths)], [1.0]*n_drone_paths,
    float(MAX_DRONE_PATHS))
current_row += 1
 
add_constraint(current_row,
    [OFF_HP + p for p in range(n_human_paths)], [1.0]*n_human_paths,
    float(MAX_HUMAN_PATHS))
current_row += 1
 
# 7-8. Coverage caps
print("  Coverage caps ...")
for i in range(n_cells):
    add_constraint(current_row, [OFF_COV + i], [1.0],  float(danger_flat[i]))
    current_row += 1
for i in range(n_cells):
    add_constraint(current_row, [OFF_COV + i], [-1.0], 0.0)
    current_row += 1
 
n_constraints = current_row
print(f"\n  Constraints : {n_constraints}")
 
G_cvx = spmatrix(data_vals, row_vals, col_vals, (n_constraints, n_vars), 'd')
h_cvx = matrix(h_vals, tc='d')
c_cvx = matrix([0.0] * n_x_bin + [-1.0] * n_cells, tc='d')
B_set = set(range(n_x_bin))
 
# ================================================================
# 9.  SOLVE
# ================================================================
glpk.options['msg_lev'] = 'GLP_MSG_ALL'
glpk.options['tm_lim']  = TIME_LIMIT_MS
glpk.options['mip_gap'] = MIP_GAP
 
print("\nSolving with cvxopt.glpk.ilp ...\n")
status, x_sol = glpk.ilp(c_cvx, G_cvx, h_cvx, B=B_set)
 
print(f"\nSolver status: {status}")
 
# Accept both 'optimal' and 'undefined' (timeout with incumbent)
if x_sol is None:
    print("No incumbent solution found — solver returned nothing.")
    print("Try increasing TIME_LIMIT_MS or raising MIP_GAP.")
    raise SystemExit
 
# ================================================================
# 10.  EXTRACT & REPORT RESULTS
# ================================================================
x_arr   = np.array(x_sol).flatten()
cov_arr = x_arr[OFF_COV:]
 
total_danger   = danger_flat.sum()
danger_removed = cov_arr.sum()
residual       = total_danger - danger_removed
 
print(f"\n{'='*60}")
print(f"Solver status    : {status}")
print(f"Total danger     : {total_danger:.3f}")
print(f"Danger removed   : {danger_removed:.3f}  ({danger_removed/total_danger*100:.1f}%)")
print(f"Residual danger  : {residual:.3f}  ({residual/total_danger*100:.1f}%)")
print(f"{'='*60}\n")
 
sp_placed  = [(k, valid_cells[k]) for k in range(n_cells) if x_arr[OFF_SP  + k] > 0.5]
cam_placed = [(k, valid_cells[k]) for k in range(n_cells) if x_arr[OFF_CAM + k] > 0.5]
sel_drone  = [p for p in range(n_drone_paths) if x_arr[OFF_DP + p] > 0.5]
sel_human  = [p for p in range(n_human_paths) if x_arr[OFF_HP + p] > 0.5]
 
used_cost_raw = (
    len(sp_placed)  * FIXED_ASSETS["stationary_person"]["cost_raw"] +
    len(cam_placed) * FIXED_ASSETS["camera"]["cost_raw"] +
    len(sel_drone)  * DRONE_PARAMS["cost_raw"] +
    len(sel_human)  * HUMAN_PARAMS["cost_raw"]
)
 
print(f"Stationary people placed    : {len(sp_placed)} / {FIXED_ASSETS['stationary_person']['max_total']}")
for k, (r, c) in sp_placed[:5]:
    print(f"  cell ({r:2d},{c:2d})  {rows_km[r]:.1f} km N  {cols_km[c]:.1f} km E")
if len(sp_placed) > 5:
    print(f"  ... and {len(sp_placed)-5} more")
 
print(f"\nCameras placed              : {len(cam_placed)} / {FIXED_ASSETS['camera']['max_total']}")
for k, (r, c) in cam_placed[:5]:
    print(f"  cell ({r:2d},{c:2d})  {rows_km[r]:.1f} km N  {cols_km[c]:.1f} km E")
if len(cam_placed) > 5:
    print(f"  ... and {len(cam_placed)-5} more")
 
print(f"\nDrone paths selected        : {len(sel_drone)} / {MAX_DRONE_PATHS}")
for p in sel_drone:
    row = drone_df.iloc[p]
    print(f"  Path {p:3d}  avg_danger={row['avg_path_danger']:.4f}"
          f"  cells={row['n_cells']}  km={row['path_km']:.1f}"
          f"  removal={drone_removal[p].sum():.3f}")
 
print(f"\nHuman patrol paths selected : {len(sel_human)} / {MAX_HUMAN_PATHS}")
for p in sel_human:
    row = human_df.iloc[p]
    print(f"  Path {p:3d}  avg_danger={row['avg_path_danger']:.4f}"
          f"  cells={row['n_cells']}  km={row['path_km']:.1f}"
          f"  removal={human_removal[p].sum():.3f}")
 
print(f"\nBudget used : ${used_cost_raw:,} / ${BUDGET_RAW:,}")
 
assert np.all(cov_arr <= danger_flat + 1e-6), "Coverage exceeded danger cap!"
print("Sanity check passed.\n")
 
# ================================================================
# 11.  EXPORT
# ================================================================
rows_out = []
for i, (r, c) in enumerate(valid_cells):
    rows_out.append({
        "row":              r,
        "col":              c,
        "northing_km":      rows_km[r],
        "easting_km":       cols_km[c],
        "fire_norm":        fire_norm_2d[r, c],
        "animal_norm":      animal_norm_2d[r, c] if not np.isnan(animal_norm_2d[r, c]) else 0.0,
        "danger":           danger_flat[i],
        "removed":          cov_arr[i],
        "residual":         danger_flat[i] - cov_arr[i],
        "has_stationary":   int(x_arr[OFF_SP  + i] > 0.5),
        "has_camera":       int(x_arr[OFF_CAM + i] > 0.5),
        "covered_by_drone": int(any(drone_removal[p, i] > 1e-6 for p in sel_drone)),
        "covered_by_human": int(any(human_removal[p, i] > 1e-6 for p in sel_human)),
    })
pd.DataFrame(rows_out).to_csv("etosha_path_results.csv", index=False)
print("Cell-level results   -> etosha_path_results.csv")
 
drone_out = [
    {**drone_df.iloc[p].to_dict(),
     "optimizer_danger_removed": round(float(drone_removal[p].sum()), 6)}
    for p in sel_drone
]
pd.DataFrame(drone_out).to_csv("selected_drone_paths.csv", index=False)
print("Selected drone paths -> selected_drone_paths.csv")
 
human_out = [
    {**human_df.iloc[p].to_dict(),
     "optimizer_danger_removed": round(float(human_removal[p].sum()), 6)}
    for p in sel_human
]
pd.DataFrame(human_out).to_csv("selected_human_paths.csv", index=False)
print("Selected human paths -> selected_human_paths.csv")
 
print("\nDone.\n")