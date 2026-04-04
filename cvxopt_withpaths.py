"""
Etosha National Park — Hybrid Patrol Path Optimizer
====================================================
Solver: cvxopt.glpk.ilp  (open-source GLPK, no Gurobi licence needed)
Install: pip install cvxopt
 
HYBRID MODEL
------------
  Fixed-cell placement (original style):
    - stationary_person  : placed at a grid cell, covers nearby area
    - camera             : placed at a grid cell, covers nearby area
 
  Path selection (new style):
    - drone              : picks a loop path from drone_paths_filtered.csv
    - patrol_person      : picks a loop path from human_paths_filtered.csv
 
All four asset types contribute to total_cov[i] and share one budget
and one set of danger-cap constraints.
 
ASSET VALUES (from immc_protect_vals.csv)
-----------------------------------------
  Asset              Efficiency  Range(km)  Cost($/week)
  stationary_person  0.64        0.5        750
  patrol_person      0.448       0.5        1190  (path-based)
  camera             0.50        0.1         20
  drone              0.35        1.5        325   (path-based)
 
VARIABLE LAYOUT (flat vector passed to glpk)
---------------------------------------------
  [OFF_SP  .. OFF_CAM-1]   binary  x_sp[k]   stationary_person at cell k
  [OFF_CAM .. OFF_DP-1]    binary  x_cam[k]  camera at cell k
  [OFF_DP  .. OFF_HP-1]    binary  x_dp[p]   drone path p selected
  [OFF_HP  .. OFF_COV-1]   binary  x_hp[p]   human patrol path p selected
  [OFF_COV .. end]         continuous total_cov[i]
 
CONSTRAINTS (G*x <= h):
  1. Coverage link : total_cov[i] - sum_all removal*x      <= 0
  2. Budget        : sum_all cost*x                         <= BUDGET
  3. Fleet sp      : sum_k x_sp[k]                         <= MAX_SP
  4. Fleet cam     : sum_k x_cam[k]                        <= MAX_CAM
  5. Fleet drone   : sum_p x_dp[p]                         <= MAX_DRONE_PATHS
  6. Fleet human   : sum_p x_hp[p]                         <= MAX_HUMAN_PATHS
  7. cov upper cap : total_cov[i]                          <= danger[i]
  8. cov lower cap : -total_cov[i]                         <= 0
"""
 
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import cvxopt
from cvxopt import matrix, spmatrix
from cvxopt import glpk
 
# ================================================================
# 1.  ASSET PARAMETERS  (from immc_protect_vals.csv)
# ================================================================
 
FIXED_ASSETS = {
    "stationary_person": {
        "efficiency": 0.64,
        "range_km":   0.5,
        "cost":       750,
        "max_total":  20,
    },
    "camera": {
        "efficiency": 0.50,
        "range_km":   0.1,
        "cost":       20,
        "max_total":  60,
    },
}
 
DRONE_PARAMS = {"efficiency": 0.35,  "range_km": 1.5, "cost": 325}
HUMAN_PARAMS = {"efficiency": 0.448, "range_km": 0.5, "cost": 1190}
 
MAX_DRONE_PATHS  = 10
MAX_HUMAN_PATHS  = 12
MAX_INFLUENCE_KM = 55.0
 
# ================================================================
# 2.  PROBLEM SETTINGS
# ================================================================
BUDGET        = 600_000   # adjust to your actual weekly budget ($)
TIME_LIMIT_MS = 300_000
MIP_GAP       = 0.02
 
# ================================================================
# 3.  LOAD & ALIGN RISK GRIDS
# ================================================================
fire_df   = pd.read_csv("fire_risk_5km.csv",    index_col=0)
animal_df = pd.read_csv("animal_value_5km.csv", index_col=0)
 
fire_df.index     = fire_df.index.astype(float).round(4)
fire_df.columns   = fire_df.columns.astype(float).round(4)
animal_df.index   = animal_df.index.astype(float).round(4)
animal_df.columns = animal_df.columns.astype(float).round(4)
 
print(f"fire_risk_5km    shape : {fire_df.shape}")
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
 
 
fire_aligned = reindex_nearest(
    fire_df,
    ref_index=animal_df.index.values,
    ref_columns=animal_df.columns.values,
)
 
# ================================================================
# 4.  BUILD DANGER MAP
# ================================================================
fire_raw   = fire_aligned.values.astype(float)
animal_raw = animal_df.values.astype(float)
 
inside_park = ~np.isnan(fire_raw) & ~np.isnan(animal_raw)
fire_raw[~inside_park]   = 0.0
animal_raw[~inside_park] = 0.0
 
GRID_ROWS, GRID_COLS = fire_raw.shape
 
 
def normalize_01(arr, mask):
    vals = arr[mask]
    lo, hi = vals.min(), vals.max()
    out = np.zeros_like(arr)
    out[mask] = (vals - lo) / (hi - lo + 1e-12)
    return out
 
 
fire_norm   = normalize_01(fire_raw,   inside_park)
animal_norm = normalize_01(animal_raw, inside_park)
 
danger = fire_norm + animal_norm   # [0, 2]
danger[~inside_park] = 0.0
 
rows_km = animal_df.index.values.astype(float)
cols_km = animal_df.columns.values.astype(float)
 
valid_cells  = [(r, c) for r in range(GRID_ROWS)
                        for c in range(GRID_COLS) if inside_park[r, c]]
valid_coords = np.array([(rows_km[r], cols_km[c]) for r, c in valid_cells])
n_cells      = len(valid_cells)
danger_flat  = np.array([danger[r, c] for r, c in valid_cells])
cell_index   = {rc: i for i, rc in enumerate(valid_cells)}
 
print(f"\nGrid          : {GRID_ROWS} x {GRID_COLS}")
print(f"Cells in park : {n_cells}")
print(f"Total danger  : {danger.sum():.3f}")
print(f"Budget        : ${BUDGET:,}")
 
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
 
# ================================================================
# 6.  PRECOMPUTE REMOVAL COEFFICIENTS
# ================================================================
print("\nPrecomputing removal coefficients ...")
 
dist_cc = cdist(valid_coords, valid_coords)   # (n_cells, n_cells)
 
# Fixed-cell assets: removal[k, i] = eff * exp(-dist(k,i) / range)
fixed_removal = {}
for aname, params in FIXED_ASSETS.items():
    mat = params["efficiency"] * np.exp(-dist_cc / params["range_km"])
    mat[dist_cc > MAX_INFLUENCE_KM] = 0.0
    mat = np.minimum(mat, danger_flat[np.newaxis, :])   # cap at local danger
    fixed_removal[aname] = mat
    print(f"  {aname:22s}: {(mat > 1e-6).sum():,} nonzero pairs")
 
 
# Path-based assets: removal[p, i] = eff * exp(-min_dist_to_path / range)
def compute_path_removal(paths, efficiency, range_km, max_inf_km, label):
    removal = np.zeros((len(paths), n_cells), dtype=np.float64)
    for p_idx, path in enumerate(paths):
        path_coords = np.array([
            (rows_km[r], cols_km[c])
            for r, c in path if (r, c) in cell_index
        ])
        if len(path_coords) == 0:
            continue
        dists = cdist(valid_coords, path_coords).min(axis=1)
        raw = efficiency * np.exp(-dists / range_km)
        raw[dists > max_inf_km] = 0.0
        removal[p_idx] = np.minimum(raw, danger_flat)
        if (p_idx + 1) % 50 == 0 or (p_idx + 1) == len(paths):
            print(f"  {label}: {p_idx+1}/{len(paths)} done")
    return removal
 
 
drone_removal = compute_path_removal(
    drone_paths,
    DRONE_PARAMS["efficiency"], DRONE_PARAMS["range_km"],
    100 * DRONE_PARAMS["range_km"], "Drone paths"
)
human_removal = compute_path_removal(
    human_paths,
    HUMAN_PARAMS["efficiency"], HUMAN_PARAMS["range_km"],
    100 * HUMAN_PARAMS["range_km"], "Human paths"
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
print(f"  stationary_person : [{OFF_SP} .. {OFF_CAM-1}]  ({n_cells} vars)")
print(f"  camera            : [{OFF_CAM} .. {OFF_DP-1}]  ({n_cells} vars)")
print(f"  drone paths       : [{OFF_DP} .. {OFF_HP-1}]  ({n_drone_paths} vars)")
print(f"  human paths       : [{OFF_HP} .. {OFF_COV-1}]  ({n_human_paths} vars)")
print(f"  total_cov         : [{OFF_COV} .. {n_vars-1}]  ({n_cells} vars)")
print(f"  Total variables   : {n_vars}")
 
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
 
 
# 1. Coverage linking
print("  Adding coverage-link constraints ...")
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
        v = drone_removal[p, i]
        if v > 1e-9:
            cols_i.append(OFF_DP + p); vals_i.append(-v)
 
    for p in range(n_human_paths):
        v = human_removal[p, i]
        if v > 1e-9:
            cols_i.append(OFF_HP + p); vals_i.append(-v)
 
    add_constraint(current_row, cols_i, vals_i, 0.0)
    current_row += 1
 
# 2. Budget
print("  Adding budget constraint ...")
bc, bv = [], []
for k in range(n_cells):
    bc.append(OFF_SP  + k); bv.append(float(FIXED_ASSETS["stationary_person"]["cost"]))
    bc.append(OFF_CAM + k); bv.append(float(FIXED_ASSETS["camera"]["cost"]))
for p in range(n_drone_paths):
    bc.append(OFF_DP + p); bv.append(float(DRONE_PARAMS["cost"]))
for p in range(n_human_paths):
    bc.append(OFF_HP + p); bv.append(float(HUMAN_PARAMS["cost"]))
add_constraint(current_row, bc, bv, float(BUDGET))
current_row += 1
 
# 3-6. Fleet limits
print("  Adding fleet constraints ...")
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
print("  Adding coverage cap constraints ...")
for i in range(n_cells):
    add_constraint(current_row, [OFF_COV + i], [1.0],  float(danger_flat[i]))
    current_row += 1
for i in range(n_cells):
    add_constraint(current_row, [OFF_COV + i], [-1.0], 0.0)
    current_row += 1
 
n_constraints = current_row
print(f"\n  Binary vars   : {n_x_bin}")
print(f"  Continuous    : {n_cells}")
print(f"  Constraints   : {n_constraints}")
 
G_cvx = spmatrix(data_vals, row_vals, col_vals, (n_constraints, n_vars), 'd')
h_cvx = matrix(h_vals, tc='d')
c_cvx = matrix([0.0]*n_x_bin + [-1.0]*n_cells, tc='d')
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
if x_sol is None:
    print("No solution found.")
    raise SystemExit
 
# ================================================================
# 10.  EXTRACT RESULTS
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
 
used_cost = (
    len(sp_placed)  * FIXED_ASSETS["stationary_person"]["cost"] +
    len(cam_placed) * FIXED_ASSETS["camera"]["cost"] +
    len(sel_drone)  * DRONE_PARAMS["cost"] +
    len(sel_human)  * HUMAN_PARAMS["cost"]
)
 
print(f"Stationary people placed : {len(sp_placed)} / {FIXED_ASSETS['stationary_person']['max_total']}")
for k, (r, c) in sp_placed[:5]:
    print(f"  cell ({r:2d},{c:2d})  {rows_km[r]:.1f} km N  {cols_km[c]:.1f} km E")
if len(sp_placed) > 5:
    print(f"  ... and {len(sp_placed)-5} more")
 
print(f"\nCameras placed           : {len(cam_placed)} / {FIXED_ASSETS['camera']['max_total']}")
for k, (r, c) in cam_placed[:5]:
    print(f"  cell ({r:2d},{c:2d})  {rows_km[r]:.1f} km N  {cols_km[c]:.1f} km E")
if len(cam_placed) > 5:
    print(f"  ... and {len(cam_placed)-5} more")
 
print(f"\nDrone paths selected     : {len(sel_drone)} / {MAX_DRONE_PATHS}")
for p in sel_drone:
    row = drone_df.iloc[p]
    print(f"  Path {p:3d}  avg_danger={row['avg_path_danger']:.4f}"
          f"  cells={row['n_cells']}  km={row['path_km']:.1f}")
 
print(f"\nHuman patrol paths selected : {len(sel_human)} / {MAX_HUMAN_PATHS}")
for p in sel_human:
    row = human_df.iloc[p]
    print(f"  Path {p:3d}  avg_danger={row['avg_path_danger']:.4f}"
          f"  cells={row['n_cells']}  km={row['path_km']:.1f}")
 
print(f"\nBudget used : ${used_cost:,} / ${BUDGET:,}")
 
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
        "fire_norm":        fire_norm[r, c],
        "animal_norm":      animal_norm[r, c],
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