"""
Etosha Security Placement — Results Visualizer
===============================================
Reads etosha_results.csv and plots:
  1. Original danger map
  2. Residual danger map (after optimization)
  3. Asset placement map
  4. Side-by-side before/after comparison
 
Dependencies: pandas, numpy, matplotlib, geopandas, pyproj
"""
 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import geopandas as gpd
from pyproj import Transformer
 
# ================================================================
# CONFIG
# ================================================================
RESULTS_CSV = "etosha_results.csv"
SHP_PATH    = "WDPA_WDOECM_Apr2026_Public_884_shp-polygons.shp"
CELL_SIZE   = 5.0   # km
 
ASSET_STYLES = {
    "stationary_person": {"marker": "o", "color": "#00ffff", "size": 80,  "label": "Stationary person"},
    "patrol_person":     {"marker": "^", "color": "#ffff00", "size": 80,  "label": "Patrol person"},
    "camera":            {"marker": "s", "color": "#ff69b4", "size": 60,  "label": "Camera"},
    "drone":             {"marker": "D", "color": "#ff8c00", "size": 80,  "label": "Drone"},
}
 
# ================================================================
# 1. LOAD RESULTS
# ================================================================
df = pd.read_csv(RESULTS_CSV)
 
northing = sorted(df["northing_km"].unique())
easting  = sorted(df["easting_km"].unique())
row_to_idx = {v: i for i, v in enumerate(northing)}
col_to_idx = {v: i for i, v in enumerate(easting)}
 
NROWS = len(northing)
NCOLS = len(easting)
 
# Reconstruct 2D grids (NaN outside park)
def make_grid(col):
    g = np.full((NROWS, NCOLS), np.nan)
    for _, row in df.iterrows():
        r = row_to_idx[row["northing_km"]]
        c = col_to_idx[row["easting_km"]]
        g[r, c] = row[col]
    return g
 
danger_grid   = make_grid("danger")
residual_grid = make_grid("residual")
removed_grid  = make_grid("removed")
fire_grid     = make_grid("fire_norm")
poacher_grid  = make_grid("poacher_norm")
 
# Coordinate meshgrid (cell centres)
xi = np.array(easting)
yi = np.array(northing)
XI, YI = np.meshgrid(xi, yi)
 
# Asset placement grids
asset_grids = {}
for atype in ASSET_STYLES:
    col = f"has_{atype}"
    if col in df.columns:
        asset_grids[atype] = make_grid(col)
 
# Placed asset locations
placements = {}
for atype in ASSET_STYLES:
    col = f"has_{atype}"
    if col in df.columns:
        sub = df[df[col] == 1]
        placements[atype] = sub[["easting_km", "northing_km"]].values
 
# ================================================================
# 2. LOAD PARK BOUNDARY
# ================================================================
os.environ["SHAPE_RESTORE_SHX"] = "YES"
park = gpd.read_file(SHP_PATH)
park = park.set_crs("EPSG:4326") if park.crs is None else park.to_crs("EPSG:4326")
 
centroid       = park.geometry.union_all().centroid
center_lon, center_lat = centroid.x, centroid.y
 
proj_str = (
    f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
    "+datum=WGS84 +units=m +no_defs"
)
park_proj = park.to_crs(proj_str)
 
def draw_boundary(ax, lw=1.8, color="white"):
    for geom in park_proj.geometry:
        if geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            ax.plot([v/1000 for v in xs], [v/1000 for v in ys],
                    color=color, linewidth=lw, zorder=4)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                xs, ys = part.exterior.xy
                ax.plot([v/1000 for v in xs], [v/1000 for v in ys],
                        color=color, linewidth=lw, zorder=4)
 
def style_ax(ax):
    ax.set_facecolor("#0d1117")
    ax.grid(True, linestyle="--", alpha=0.12, color="white", zorder=0)
    ax.set_xlabel("Easting from centroid (km)", fontsize=9, color="#cccccc")
    ax.set_ylabel("Northing from centroid (km)", fontsize=9, color="#cccccc")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.set_aspect("equal")
 
def add_cbar(fig, ax, sm, label):
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03, shrink=0.85)
    cbar.set_label(label, fontsize=8, color="#cccccc")
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa", fontsize=7)
 
# ================================================================
# 3. SUMMARY STATS
# ================================================================
total_danger   = df["danger"].sum()
total_removed  = df["removed"].sum()
total_residual = df["residual"].sum()
pct_removed    = total_removed / total_danger * 100
 
n_placed = {atype: int(df[f"has_{atype}"].sum())
            for atype in ASSET_STYLES if f"has_{atype}" in df.columns}
 
print(f"Total danger    : {total_danger:.3f}")
print(f"Danger removed  : {total_removed:.3f}  ({pct_removed:.1f}%)")
print(f"Residual danger : {total_residual:.3f}  ({100-pct_removed:.1f}%)")
print(f"Assets placed   : {n_placed}")
 
# ================================================================
# 4. FIGURE 1 — BEFORE / AFTER / ASSETS  (3-panel)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor="#0d1117")
plt.subplots_adjust(wspace=0.3)
 
vmax_danger = np.nanmax(danger_grid)
 
# ── Panel A: original danger ──────────────────────────────────────
ax = axes[0]
style_ax(ax)
cmap_d = plt.cm.YlOrRd
norm_d = mcolors.Normalize(vmin=0, vmax=vmax_danger)
ax.pcolormesh(XI, YI, danger_grid, cmap=cmap_d, norm=norm_d,
              shading="nearest", zorder=1)
draw_boundary(ax)
ax.set_title("Original danger map\n(fire + poaching risk)",
             color="white", fontsize=11, fontweight="bold", pad=8)
add_cbar(fig, ax, plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_d),
         "Danger [0–2]")
 
# ── Panel B: residual danger ──────────────────────────────────────
ax = axes[1]
style_ax(ax)
norm_r = mcolors.Normalize(vmin=0, vmax=vmax_danger)
ax.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_r,
              shading="nearest", zorder=1)
draw_boundary(ax)
ax.set_title(f"Residual danger after deployment\n({pct_removed:.1f}% of danger neutralised)",
             color="white", fontsize=11, fontweight="bold", pad=8)
add_cbar(fig, ax, plt.cm.ScalarMappable(norm=norm_r, cmap=cmap_d),
         "Residual danger [0–2]")
 
# ── Panel C: asset placements on residual map ─────────────────────
ax = axes[2]
style_ax(ax)
ax.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_r,
              shading="nearest", zorder=1, alpha=0.75)
draw_boundary(ax)
 
for atype, style in ASSET_STYLES.items():
    locs = placements.get(atype, np.empty((0, 2)))
    if len(locs):
        ax.scatter(locs[:, 0], locs[:, 1],
                   marker=style["marker"],
                   c=style["color"],
                   s=style["size"],
                   zorder=6,
                   edgecolors="black",
                   linewidths=0.5,
                   label=f"{style['label']} (×{len(locs)})")
 
legend = ax.legend(loc="upper left", fontsize=8,
                   facecolor="#1e2433", edgecolor="#555555",
                   labelcolor="white", framealpha=0.9,
                   markerscale=1.1)
ax.set_title("Optimal asset placement\n(on residual danger map)",
             color="white", fontsize=11, fontweight="bold", pad=8)
add_cbar(fig, ax, plt.cm.ScalarMappable(norm=norm_r, cmap=cmap_d),
         "Residual danger [0–2]")
 
fig.suptitle("Etosha National Park — Security Optimisation Results",
             color="white", fontsize=14, fontweight="bold", y=1.01)
plt.savefig("etosha_results_3panel.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → etosha_results_3panel.png")
 
# ================================================================
# 5. FIGURE 2 — DANGER REDUCTION MAP (how much each cell improved)
# ================================================================
reduction_grid = removed_grid / np.where(danger_grid > 0, danger_grid, np.nan)
 
fig2, ax2 = plt.subplots(figsize=(14, 7), facecolor="#0d1117")
style_ax(ax2)
 
cmap_g = plt.cm.RdYlGn
norm_g = mcolors.Normalize(vmin=0, vmax=1)
ax2.pcolormesh(XI, YI, reduction_grid, cmap=cmap_g, norm=norm_g,
               shading="nearest", zorder=1)
draw_boundary(ax2)
 
# Overlay assets
for atype, style in ASSET_STYLES.items():
    locs = placements.get(atype, np.empty((0, 2)))
    if len(locs):
        ax2.scatter(locs[:, 0], locs[:, 1],
                    marker=style["marker"],
                    c=style["color"],
                    s=style["size"],
                    zorder=6,
                    edgecolors="black",
                    linewidths=0.5,
                    label=f"{style['label']} (×{len(locs)})")
 
ax2.legend(loc="upper left", fontsize=9,
           facecolor="#1e2433", edgecolor="#555555",
           labelcolor="white", framealpha=0.9)
add_cbar(fig2, ax2,
         plt.cm.ScalarMappable(norm=norm_g, cmap=cmap_g),
         "Fraction of danger neutralised  (0 = none, 1 = fully)")
 
ax2.set_title(
    "Danger Reduction per Cell — Etosha National Park\n"
    "Green = fully neutralised  |  Red = untouched",
    color="white", fontsize=13, fontweight="bold", pad=12,
)
plt.tight_layout()
plt.savefig("etosha_results_reduction.png", dpi=160, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
print("Saved → etosha_results_reduction.png")
 
# ================================================================
# 6. FIGURE 3 — INDIVIDUAL ASSET TYPE MAPS
# ================================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(18, 10), facecolor="#0d1117")
plt.subplots_adjust(wspace=0.25, hspace=0.35)
axes3_flat = axes3.flatten()
 
for idx, (atype, style) in enumerate(ASSET_STYLES.items()):
    ax = axes3_flat[idx]
    style_ax(ax)
 
    # Residual map as background
    ax.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_r,
                  shading="nearest", zorder=1, alpha=0.6)
    draw_boundary(ax, lw=1.2)
 
    locs = placements.get(atype, np.empty((0, 2)))
    if len(locs):
        ax.scatter(locs[:, 0], locs[:, 1],
                   marker=style["marker"],
                   c=style["color"],
                   s=100,
                   zorder=6,
                   edgecolors="black",
                   linewidths=0.6)
 
    ax.set_title(f"{style['label']}  —  {len(locs)} placed",
                 color=style["color"], fontsize=11, fontweight="bold", pad=6)
 
fig3.suptitle("Per-Asset-Type Placement Maps — Etosha",
              color="white", fontsize=13, fontweight="bold", y=1.01)
plt.savefig("etosha_results_per_asset.png", dpi=160, bbox_inches="tight",
            facecolor=fig3.get_facecolor())
print("Saved → etosha_results_per_asset.png")
 
plt.show()