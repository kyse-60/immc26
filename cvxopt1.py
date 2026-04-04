"""
Etosha National Park — Security Asset Placement Optimizer
==========================================================
Solver: cvxopt.glpk.ilp  (open-source GLPK, no Gurobi licence needed)
Install: pip install cvxopt
 
INPUTS:
  - fire_risk_5km.csv        (from ndvi_map.py)
  - animal_value_5km.csv     (from AnimalValue_5x5.py)
  - BUDGET                   (maximum cost units to spend)
 
VARIABLE LAYOUT (one flat vector passed to glpk):
  [0 .. n_assets*n_cells - 1]           binary x[atype, k]
  [n_assets*n_cells .. +n_cells - 1]    continuous total_cov[i]
 
OBJECTIVE (minimise c'x):
  c = 0 for all x variables, -1 for each total_cov[i]
  => maximises total danger removed
 
CONSTRAINTS (all as G*x <= h):
  1. Coverage link : total_cov[i] - sum_{t,k} removal[t,k,i]*x[t,k]  <= 0
  2. Budget        : sum_{t,k} cost_t * x[t,k]                        <= BUDGET
  3. Fleet         : sum_k x[atype,k]                                  <= max_total_t
  4. cov upper cap : total_cov[i]                                       <= danger[i]
  5. cov lower cap : -total_cov[i]                                      <= 0
"""
 
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import sparse as sp_sparse
import cvxopt
from cvxopt import matrix, spmatrix
from cvxopt import glpk
 
# ================================================================
# 1.  ASSET PARAMETERS
# ================================================================
ASSET_PARAMS = {
    "stationary_person": {
        "efficiency": 0.90,
        "range_km":   3.0,
        "cost":       10,
        "max_total":  20,
    },
    "patrol_person": {
        "efficiency": 0.55,
        "range_km":  14.0,
        "cost":       14,
        "max_total":  12,
    },
    "camera": {
        "efficiency": 0.40,
        "range_km":   2.5,
        "cost":        4,
        "max_total":  60,
    },
    "drone": {
        "efficiency": 0.70,
        "range_km":  22.0,
        "cost":       28,
        "max_total":  10,
    },
}
ASSET_LIST = list(ASSET_PARAMS.keys())
N_ASSETS   = len(ASSET_LIST)
 
# ================================================================
# 2.  PROBLEM SETTINGS
# ================================================================
BUDGET           = 600    # <- change this to your actual budget
MAX_INFLUENCE_KM = 55.0
TIME_LIMIT_MS    = 300_000   # 5 minutes in milliseconds (GLPK uses ms)
MIP_GAP          = 0.02      # stop within 2% of optimal
 
# ================================================================
# 3.  LOAD & ALIGN THE TWO GRIDS
# ================================================================
fire_df    = pd.read_csv("fire_risk_5km.csv",    index_col=0)
poacher_df = pd.read_csv("animal_value_5km.csv", index_col=0)
 
fire_df.index      = fire_df.index.astype(float).round(4)
fire_df.columns    = fire_df.columns.astype(float).round(4)
poacher_df.index   = poacher_df.index.astype(float).round(4)
poacher_df.columns = poacher_df.columns.astype(float).round(4)
 
print(f"fire_risk_5km    shape : {fire_df.shape}")
print(f"  rows : {fire_df.index.min():.2f} -> {fire_df.index.max():.2f} km")
print(f"  cols : {fire_df.columns.min():.2f} -> {fire_df.columns.max():.2f} km")
print(f"animal_value_5km shape : {poacher_df.shape}")
print(f"  rows : {poacher_df.index.min():.2f} -> {poacher_df.index.max():.2f} km")
print(f"  cols : {poacher_df.columns.min():.2f} -> {poacher_df.columns.max():.2f} km")
 
def reindex_nearest(df_src, ref_index, ref_columns, tolerance_km=3.0):
    """
    Snap each reference coordinate to the nearest source coordinate
    within tolerance_km.  Returns a DataFrame aligned to ref shape.
    """
    src_rows = df_src.index.values
    src_cols = df_src.columns.values
 
    def snap(ref_vals, src_vals):
        snapped = []
        for v in ref_vals:
            diffs = np.abs(src_vals - v)
            i     = diffs.argmin()
            snapped.append(src_vals[i] if diffs[i] <= tolerance_km else np.nan)
        return snapped
 
    snapped_rows = snap(ref_index,   src_rows)
    snapped_cols = snap(ref_columns, src_cols)
 
    valid_src_rows = [r for r in snapped_rows if not np.isnan(r)]
    valid_src_cols = [c for c in snapped_cols if not np.isnan(c)]
    resampled = df_src.reindex(index=valid_src_rows, columns=valid_src_cols)
 
    valid_ref_rows = [ref_index[i]   for i, r in enumerate(snapped_rows) if not np.isnan(r)]
    valid_ref_cols = [ref_columns[i] for i, c in enumerate(snapped_cols) if not np.isnan(c)]
    resampled.index   = valid_ref_rows
    resampled.columns = valid_ref_cols
 
    return resampled.reindex(index=ref_index, columns=ref_columns)
 
fire_df_aligned = reindex_nearest(
    fire_df,
    ref_index=poacher_df.index.values,
    ref_columns=poacher_df.columns.values,
)
print(f"\nAfter alignment — fire: {fire_df_aligned.shape}  animal: {poacher_df.shape}")
 
# ================================================================
# 4.  BUILD THE DANGER MAP
# ================================================================
fire_raw    = fire_df_aligned.values.astype(float)
poacher_raw = poacher_df.values.astype(float)
 
inside_park = ~np.isnan(fire_raw) & ~np.isnan(poacher_raw)
fire_raw[~inside_park]    = 0.0
poacher_raw[~inside_park] = 0.0
 
GRID_ROWS, GRID_COLS = fire_raw.shape
 
def normalize_01(arr, mask):
    vals = arr[mask]
    lo, hi = vals.min(), vals.max()
    out = np.zeros_like(arr)
    out[mask] = (vals - lo) / (hi - lo + 1e-12)
    return out
 
fire_norm    = normalize_01(fire_raw,    inside_park)
poacher_norm = normalize_01(poacher_raw, inside_park)
 
danger = fire_norm + poacher_norm   # values in [0, 2]
danger[~inside_park] = 0.0
 
rows_km = poacher_df.index.values.astype(float)
cols_km = poacher_df.columns.values.astype(float)
 
print(f"\nGrid          : {GRID_ROWS} x {GRID_COLS}")
print(f"Cells in park : {inside_park.sum()}")
print(f"Total danger  : {danger.sum():.3f}")
print(f"Budget        : {BUDGET}")
 
# ================================================================
# 5.  GRID GEOMETRY & VALID CELLS
# ================================================================
valid_cells  = [(r, c) for r in range(GRID_ROWS)
                        for c in range(GRID_COLS) if inside_park[r, c]]
valid_coords = np.array([(rows_km[r], cols_km[c]) for r, c in valid_cells])
n_cells      = len(valid_cells)
danger_flat  = np.array([danger[r, c] for r, c in valid_cells])
 
print(f"Valid cells   : {n_cells}")
 
# ================================================================
# 6.  PRECOMPUTE REMOVAL COEFFICIENTS
# ================================================================
print("\nPrecomputing removal coefficients ...")
dist_km = cdist(valid_coords, valid_coords)
 
removal_coeff: dict[str, np.ndarray] = {}
for atype, params in ASSET_PARAMS.items():
    mat = params["efficiency"] * np.exp(-dist_km / params["range_km"])
    mat[dist_km > MAX_INFLUENCE_KM] = 0.0
    removal_coeff[atype] = mat
    print(f"  {atype:22s}: {(mat > 1e-6).sum():,} nonzero pairs")
 
# ================================================================
# 7.  BUILD GLPK PROBLEM MATRICES
# ================================================================
# Variable layout:
#   x[atype_idx, k]  ->  atype_idx * n_cells + k          (binary)
#   total_cov[i]     ->  N_ASSETS * n_cells + i            (continuous, [0, danger[i]])
 
n_x    = N_ASSETS * n_cells
n_vars = n_x + n_cells
 
print("\nBuilding constraint matrix ...")
 
# ── Objective: minimise -sum(total_cov) ─────────────────────────
c_obj = [0.0] * n_x + [-1.0] * n_cells
c_cvx = matrix(c_obj, tc='d')
 
# ── Build G (inequality constraints) as COO then convert ─────────
# Row groups:
#   [0          .. n_cells-1]              coverage link   (n_cells rows)
#   [n_cells]                              budget          (1 row)
#   [n_cells+1  .. n_cells+N_ASSETS]       fleet           (N_ASSETS rows)
#   [n_cells+N_ASSETS+1 .. 2*n_cells+...]  cov upper cap   (n_cells rows)
#   [2*n_cells+.. .. 3*n_cells+...]        cov lower cap   (n_cells rows)
 
row_vals, col_vals, data_vals = [], [], []
h_vals = []
row_ptr = [0]   # current row counter
 
def add_constraint(row_i, col_j_list, coeff_list, rhs):
    """Append one row to the COO lists — all values must be plain Python types."""
    for j, v in zip(col_j_list, coeff_list):
        row_vals.append(int(row_i))
        col_vals.append(int(j))
        data_vals.append(float(v))
    h_vals.append(float(rhs))
 
current_row = 0
 
# 1. Coverage linking: total_cov[i] - sum_{t,k} removal[t,k,i]*x[t,k] <= 0
print("  Adding coverage-link constraints ...")
for i in range(n_cells):
    cols_i = [n_x + i]
    vals_i = [1.0]
    for atype_idx, atype in enumerate(ASSET_LIST):
        col = removal_coeff[atype][:, i]
        nz  = np.where(col > 1e-6)[0]
        for k in nz:
            cols_i.append(atype_idx * n_cells + k)
            vals_i.append(-col[k])
    add_constraint(current_row, cols_i, vals_i, 0.0)
    current_row += 1
 
# 2. Budget: sum_{t,k} cost_t * x[t,k] <= BUDGET
print("  Adding budget constraint ...")
cols_b, vals_b = [], []
for atype_idx, atype in enumerate(ASSET_LIST):
    cost = ASSET_PARAMS[atype]["cost"]
    for k in range(n_cells):
        cols_b.append(atype_idx * n_cells + k)
        vals_b.append(float(cost))
add_constraint(current_row, cols_b, vals_b, float(BUDGET))
current_row += 1
 
# 3. Fleet limits: sum_k x[atype,k] <= max_total
print("  Adding fleet constraints ...")
for atype_idx, atype in enumerate(ASSET_LIST):
    cols_f = [atype_idx * n_cells + k for k in range(n_cells)]
    vals_f = [1.0] * n_cells
    add_constraint(current_row, cols_f, vals_f, float(ASSET_PARAMS[atype]["max_total"]))
    current_row += 1
 
# 4. Upper cap on total_cov[i]: total_cov[i] <= danger_flat[i]
print("  Adding coverage upper-cap constraints ...")
for i in range(n_cells):
    add_constraint(current_row, [n_x + i], [1.0], float(danger_flat[i]))
    current_row += 1
 
# 5. Lower cap on total_cov[i]: -total_cov[i] <= 0  (i.e. cov >= 0)
print("  Adding coverage lower-cap constraints ...")
for i in range(n_cells):
    add_constraint(current_row, [n_x + i], [-1.0], 0.0)
    current_row += 1
 
n_constraints = current_row
print(f"\n  Variables   : {n_vars}  ({n_x} binary + {n_cells} continuous)")
print(f"  Constraints : {n_constraints}")
 
# Convert COO to cvxopt spmatrix (column-sparse, but spmatrix takes row,col,val)
G_cvx = spmatrix(data_vals, row_vals, col_vals, (n_constraints, n_vars), 'd')
h_cvx = matrix(h_vals, tc='d')
 
# Binary variable set — all x variables
B_set = set(range(n_x))
 
# ================================================================
# 8.  SOLVE WITH GLPK
# ================================================================
# GLPK options
glpk.options['msg_lev'] = 'GLP_MSG_ALL'   # verbose output
glpk.options['tm_lim']  = TIME_LIMIT_MS
glpk.options['mip_gap'] = MIP_GAP
 
print("\nSolving with cvxopt.glpk.ilp ...\n")
status, x_sol = glpk.ilp(c_cvx, G_cvx, h_cvx, B=B_set)
 
print(f"\nSolver status: {status}")
if x_sol is None:
    print("No solution found.")
    raise SystemExit
 
# ================================================================
# 9.  EXTRACT RESULTS
# ================================================================
x_arr   = np.array(x_sol).flatten()
x_bin   = x_arr[:n_x]               # binary placement decisions
cov_arr = x_arr[n_x:]               # total_cov values
 
total_danger   = danger_flat.sum()
danger_removed = cov_arr.sum()
residual       = total_danger - danger_removed
 
print(f"\n{'='*60}")
print(f"Solver status   : {status}")
print(f"Total danger    : {total_danger:.3f}")
print(f"Danger removed  : {danger_removed:.3f}  ({danger_removed/total_danger*100:.1f}%)")
print(f"Residual danger : {residual:.3f}  ({residual/total_danger*100:.1f}%)")
print(f"{'='*60}\n")
 
# Collect placement locations
placements: dict[str, list] = {a: [] for a in ASSET_LIST}
used_cost = 0
for atype_idx, atype in enumerate(ASSET_LIST):
    for k in range(n_cells):
        if x_bin[atype_idx * n_cells + k] > 0.5:
            r, c = valid_cells[k]
            placements[atype].append((r, c, rows_km[r], cols_km[c]))
            used_cost += ASSET_PARAMS[atype]["cost"]
 
print("PLACEMENTS:")
for atype, locs in placements.items():
    print(f"  {atype:22s}: {len(locs):3d} units")
    for r, c, y_km, x_km in locs[:4]:
        print(f"      cell ({r:2d},{c:2d})  {y_km:.1f} km N  {x_km:.1f} km E")
    if len(locs) > 4:
        print(f"      ... and {len(locs)-4} more")
 
print(f"\nBudget used : {used_cost} / {BUDGET}")
 
# Sanity check — no cell should go negative
assert np.all(cov_arr <= danger_flat + 1e-6), "A cell went negative — check caps!"
print("Sanity check passed — no cell has negative residual danger")
 
# ================================================================
# 10.  EXPORT
# ================================================================
rows_out = []
for i, (r, c) in enumerate(valid_cells):
    rows_out.append({
        "row":          r,
        "col":          c,
        "northing_km":  rows_km[r],
        "easting_km":   cols_km[c],
        "fire_norm":    fire_norm[r, c],
        "poacher_norm": poacher_norm[r, c],
        "danger":       danger_flat[i],
        "removed":      cov_arr[i],
        "residual":     danger_flat[i] - cov_arr[i],
        **{f"has_{atype}": int(any(loc[0]==r and loc[1]==c
                                   for loc in placements[atype]))
           for atype in ASSET_LIST}
    })
 
pd.DataFrame(rows_out).to_csv("etosha_results.csv", index=False)
print("Results saved to etosha_results.csv")
