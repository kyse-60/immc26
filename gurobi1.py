"""
Etosha National Park — Security Asset Placement Optimizer
==========================================================
 
INPUTS:
  - fire_risk_5km.csv        (from ndvi_map.py)
  - animal_value_5km.csv     (from AnimalValue_5x5.py)
  - BUDGET                   (maximum cost units to spend)
 
DANGER MAP:
  danger[i] = normalize(fire[i]) + normalize(animal_value[i])
  Values range [0, 2].
 
REMOVAL MODEL:
  Placing asset of type t at cell k removes from cell i:
      removal(k->i, t) = efficiency_t x exp(-dist(k,i) / range_t)
 
  Residual danger at cell i:
      residual[i] = max(0, danger[i] - removed[i])
 
OBJECTIVE:
  Minimize sum_i residual[i]
  = Maximize sum_i total_cov[i]
      where total_cov[i] in [0, danger[i]]   <- per-cell cap
 
Dependencies: gurobipy, numpy, pandas, scipy
"""
 
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
 
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
 
# ================================================================
# 2.  PROBLEM SETTINGS
# ================================================================
CELL_SIZE_KM          = 5.0
BUDGET                = 600    # <- change this to your actual budget
MAX_INFLUENCE_KM      = 55.0
GUROBI_TIME_LIMIT_SEC = 300
GUROBI_MIP_GAP        = 0.02
 
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
 
# ── Resample fire grid onto animal grid by nearest-neighbour snap ─
# The two scripts used different bounding boxes as their grid origin,
# so cell centres land at slightly different km values.
# We use the animal (poacher) grid as reference since it was built
# from the park shapefile bounds, then snap each fire cell to the
# nearest matching animal cell within tolerance.
def reindex_nearest(df_src, ref_index, ref_columns, tolerance_km=3.0):
    """
    Snap each coordinate in ref_index / ref_columns to the nearest
    coordinate in df_src, within tolerance_km.  Returns a DataFrame
    with the same shape as (ref_index x ref_columns).
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
 
    # Pull the matching rows/cols from source
    valid_src_rows = [r for r in snapped_rows if not np.isnan(r)]
    valid_src_cols = [c for c in snapped_cols if not np.isnan(c)]
    resampled = df_src.reindex(index=valid_src_rows, columns=valid_src_cols)
 
    # Re-label with reference coordinates
    valid_ref_rows = [ref_index[i]   for i, r in enumerate(snapped_rows) if not np.isnan(r)]
    valid_ref_cols = [ref_columns[i] for i, c in enumerate(snapped_cols) if not np.isnan(c)]
    resampled.index   = valid_ref_rows
    resampled.columns = valid_ref_cols
 
    # Expand back to full reference shape (missing coords become NaN)
    resampled = resampled.reindex(index=ref_index, columns=ref_columns)
    return resampled
 
fire_df_aligned = reindex_nearest(
    fire_df,
    ref_index=poacher_df.index.values,
    ref_columns=poacher_df.columns.values,
)
 
print(f"\nAfter alignment:")
print(f"  fire   : {fire_df_aligned.shape}")
print(f"  animal : {poacher_df.shape}")
 
# ================================================================
# 4.  BUILD THE DANGER MAP
# ================================================================
fire_raw    = fire_df_aligned.values.astype(float)
poacher_raw = poacher_df.values.astype(float)
 
# A cell is inside the park only if BOTH grids have real data there
inside_park = ~np.isnan(fire_raw) & ~np.isnan(poacher_raw)
fire_raw[~inside_park]    = 0.0
poacher_raw[~inside_park] = 0.0
 
GRID_ROWS, GRID_COLS = fire_raw.shape
 
def normalize_01(arr, mask):
    """Scale only the in-park values to [0, 1]."""
    vals = arr[mask]
    lo, hi = vals.min(), vals.max()
    out = np.zeros_like(arr)
    out[mask] = (vals - lo) / (hi - lo + 1e-12)
    return out
 
fire_norm    = normalize_01(fire_raw,    inside_park)
poacher_norm = normalize_01(poacher_raw, inside_park)
 
# Danger map: plain sum, values in [0, 2]
danger = fire_norm + poacher_norm
danger[~inside_park] = 0.0
 
print(f"\nGrid          : {GRID_ROWS} x {GRID_COLS}")
print(f"Cells in park : {inside_park.sum()}")
print(f"Total danger  : {danger.sum():.3f}")
print(f"Budget        : {BUDGET}")
 
# ================================================================
# 5.  GRID GEOMETRY & VALID CELLS
# ================================================================
# Use the animal grid coordinates as the reference (already aligned)
rows_km = poacher_df.index.values.astype(float)
cols_km = poacher_df.columns.values.astype(float)
 
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
# 7.  BUILD GUROBI MODEL
# ================================================================
print("\nBuilding model ...")
m = gp.Model("etosha_security", env=env)
m.setParam("TimeLimit",  GUROBI_TIME_LIMIT_SEC)
m.setParam("MIPGap",     GUROBI_MIP_GAP)
m.setParam("OutputFlag", 1)
 
# Binary placement variables
x = m.addVars(
    [(atype, k) for atype in ASSET_PARAMS for k in range(n_cells)],
    vtype=GRB.BINARY,
    name="x"
)
 
# Coverage variables capped at each cell's own danger value.
# Once a cell is fully neutralised its marginal value = 0,
# so the solver spreads assets to remaining dangerous cells.
total_cov = [
    m.addVar(lb=0.0, ub=float(danger_flat[i]), name=f"cov_{i}")
    for i in range(n_cells)
]
 
# Coverage-linking constraints:
# total_cov[i] <= sum of all removal contributions at cell i
print("Adding coverage constraints ...")
for i in range(n_cells):
    rhs = gp.LinExpr()
    for atype in ASSET_PARAMS:
        col = removal_coeff[atype][:, i]
        for k in np.where(col > 1e-6)[0]:
            rhs += col[k] * x[atype, k]
    m.addConstr(total_cov[i] <= rhs, name=f"cov_{i}")
 
# Budget constraint
m.addConstr(
    gp.quicksum(ASSET_PARAMS[atype]["cost"] * x[atype, k]
                for atype in ASSET_PARAMS
                for k in range(n_cells)) <= BUDGET,
    name="budget"
)
 
# Fleet-size constraints
for atype, params in ASSET_PARAMS.items():
    m.addConstr(
        gp.quicksum(x[atype, k] for k in range(n_cells)) <= params["max_total"],
        name=f"fleet_{atype}"
    )
 
# Objective: maximize total danger removed (unweighted)
# = minimize sum of residual danger across the whole park
m.setObjective(
    gp.quicksum(total_cov[i] for i in range(n_cells)),
    GRB.MAXIMIZE
)
 
# ================================================================
# 8.  SOLVE
# ================================================================
print("\nSolving ...\n")
m.optimize()
 
if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
    print(f"No feasible solution. Status: {m.status}")
    raise SystemExit
 
# ================================================================
# 9.  RESULTS
# ================================================================
total_danger   = danger_flat.sum()
danger_removed = m.ObjVal
residual       = total_danger - danger_removed
 
print(f"\n{'='*60}")
print(f"Status          : {'OPTIMAL' if m.status == GRB.OPTIMAL else 'BEST FOUND'}")
print(f"Total danger    : {total_danger:.3f}")
print(f"Danger removed  : {danger_removed:.3f}  ({danger_removed/total_danger*100:.1f}%)")
print(f"Residual danger : {residual:.3f}  ({residual/total_danger*100:.1f}%)")
print(f"{'='*60}\n")
 
placements: dict[str, list] = {a: [] for a in ASSET_PARAMS}
used_cost = 0
for atype, params in ASSET_PARAMS.items():
    for k in range(n_cells):
        if x[atype, k].X > 0.5:
            r, c = valid_cells[k]
            placements[atype].append((r, c, rows_km[r], cols_km[c]))
            used_cost += params["cost"]
 
print("PLACEMENTS:")
for atype, locs in placements.items():
    print(f"  {atype:22s}: {len(locs):3d} units")
    for r, c, y_km, x_km in locs[:4]:
        print(f"      cell ({r:2d},{c:2d})  {y_km:.1f} km N  {x_km:.1f} km E")
    if len(locs) > 4:
        print(f"      ... and {len(locs)-4} more")
 
print(f"\nBudget used : {used_cost} / {BUDGET}")
 
# Sanity check — no cell should go negative
for i in range(n_cells):
    assert total_cov[i].X <= danger_flat[i] + 1e-6, f"Cell {i} went negative!"
print("Sanity check passed — no cell has negative residual danger")
 
# ================================================================
# 10.  EXPORT
# ================================================================
rows_out = []
for i, (r, c) in enumerate(valid_cells):
    cov_val = total_cov[i].X
    rows_out.append({
        "row":          r,
        "col":          c,
        "northing_km":  rows_km[r],
        "easting_km":   cols_km[c],
        "fire_norm":    fire_norm[r, c],
        "poacher_norm": poacher_norm[r, c],
        "danger":       danger_flat[i],
        "removed":      cov_val,
        "residual":     danger_flat[i] - cov_val,
        **{f"has_{atype}": int(any(loc[0]==r and loc[1]==c
                                   for loc in placements[atype]))
           for atype in ASSET_PARAMS}
    })
 
pd.DataFrame(rows_out).to_csv("etosha_results.csv", index=False)
print("Results saved to etosha_results.csv")
