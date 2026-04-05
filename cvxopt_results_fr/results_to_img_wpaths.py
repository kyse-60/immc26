"""
Etosha Security Placement — Results Visualizer
===============================================
- Fixed assets (stationary, camera): scatter markers at placed cells
- Path assets (drone, human): actual loop lines drawn from path_str coords
"""
 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import geopandas as gpd
 
# ================================================================
# CONFIG
# ================================================================
RESULTS_CSV      = "cvxopt_results_fr/etosha_path_results_30k.csv"
DRONE_PATHS_CSV  = "selected_drone_paths.csv"
HUMAN_PATHS_CSV  = "selected_human_paths.csv"
SHP_PATH         = "WDPA_WDOECM_Apr2026_Public_884_shp-polygons.shp"
 
# Grid that was used by the optimizer (fire_df = reference)
FIRE_CSV   = "fire_risk_5km.csv"
 
DRONE_COLOR = "#ff8c00"
HUMAN_COLOR = "#ffff00"
SP_COLOR    = "#00ffff"
CAM_COLOR   = "#ff69b4"
 
# ================================================================
# 1. LOAD RESULT GRID
# ================================================================
print("Loading results ...")
df = pd.read_csv(RESULTS_CSV)
 
northing = sorted(df["northing_km"].unique())
easting  = sorted(df["easting_km"].unique())
row_idx  = {v: i for i, v in enumerate(northing)}
col_idx  = {v: i for i, v in enumerate(easting)}
NROWS, NCOLS = len(northing), len(easting)
 
def make_grid(col):
    g = np.full((NROWS, NCOLS), np.nan)
    if col not in df.columns:
        return g
    for _, row in df.iterrows():
        r = row_idx[row["northing_km"]]
        c = col_idx[row["easting_km"]]
        g[r, c] = row[col]
    return g
 
danger_grid   = make_grid("danger")
residual_grid = make_grid("residual")
removed_grid  = make_grid("removed")
fire_grid     = make_grid("fire_norm")
animal_grid   = make_grid("animal_norm")
XI, YI = np.meshgrid(np.array(easting), np.array(northing))
 
# Fixed asset scatter locations
sp_locs  = df[df["has_stationary"] == 1][["easting_km","northing_km"]].values \
            if "has_stationary" in df.columns else np.empty((0,2))
cam_locs = df[df["has_camera"] == 1][["easting_km","northing_km"]].values \
            if "has_camera" in df.columns else np.empty((0,2))
 
# ================================================================
# 2. LOAD PATH COORDS FROM SELECTED PATH CSVs
# ================================================================
# path_str format: "r,c;r,c;r,c" — grid indices into fire_df
# We need fire_df's rows_km/cols_km to convert to km coordinates
 
fire_df = pd.read_csv(FIRE_CSV, index_col=0)
fire_df.index   = fire_df.index.astype(float)
fire_df.columns = fire_df.columns.astype(float)
rows_km = fire_df.index.values
cols_km = fire_df.columns.values
 
def str_to_path(s):
    return [tuple(int(x) for x in p.split(",")) for p in s.split(";")]
 
def path_to_km(path):
    """Convert list of (r,c) grid indices to (easting_km, northing_km) arrays,
    closing the loop by repeating the first point at the end."""
    xs = [cols_km[c] for r, c in path] + [cols_km[path[0][1]]]
    ys = [rows_km[r] for r, c in path] + [rows_km[path[0][0]]]
    return xs, ys
 
drone_paths_km = []   # list of (xs, ys) for each selected drone path
human_paths_km = []
 
def load_paths(csv_path, label):
    paths = []
    if not os.path.exists(csv_path):
        print(f"  {csv_path} not found — {label} won't be drawn")
        return paths
    try:
        pdf = pd.read_csv(csv_path)
        if pdf.empty or "path_str" not in pdf.columns:
            print(f"  {label}: 0 paths selected (empty file)")
            return paths
        for s in pdf["path_str"]:
            xs, ys = path_to_km(str_to_path(s))
            paths.append((xs, ys))
        print(f"  {label} loaded : {len(paths)}")
    except Exception as e:
        print(f"  {label}: could not load ({e})")
    return paths
 
drone_paths_km = load_paths(DRONE_PATHS_CSV, "Drone paths")
human_paths_km = load_paths(HUMAN_PATHS_CSV, "Human paths")
 
# ================================================================
# 3. LOAD PARK BOUNDARY
# ================================================================
boundary_available = False
park_proj = None
 
if os.path.exists(SHP_PATH):
    try:
        os.environ["SHAPE_RESTORE_SHX"] = "YES"
        park = gpd.read_file(SHP_PATH)
        park = park.set_crs("EPSG:4326") if park.crs is None else park.to_crs("EPSG:4326")
        try:
            centroid = park.geometry.union_all().centroid
        except AttributeError:
            centroid = park.geometry.unary_union.centroid
        clon, clat = centroid.x, centroid.y
        proj_str = (f"+proj=aeqd +lat_0={clat} +lon_0={clon} "
                    "+datum=WGS84 +units=m +no_defs")
        park_proj = park.to_crs(proj_str)
        boundary_available = True
        print("Park boundary loaded.")
    except Exception as e:
        print(f"  Boundary skipped: {e}")
 
def draw_boundary(ax, lw=1.8, color="white"):
    if not boundary_available:
        return
    for geom in park_proj.geometry:
        parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for part in parts:
            xs, ys = part.exterior.xy
            ax.plot([v/1000 for v in xs], [v/1000 for v in ys],
                    color=color, linewidth=lw, zorder=4)
 
def style_ax(ax):
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", alpha=0.2, color="#cccccc", zorder=0)
    ax.set_xlabel("East from centroid (km)", fontsize=9, color="black")
    ax.set_ylabel("North from centroid (km)", fontsize=9, color="black")
    ax.tick_params(colors="black", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.set_aspect("equal")
 
def add_cbar(fig, ax, sm, label):
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03, shrink=0.85)
    cbar.set_label(label, fontsize=8, color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black", fontsize=7)
 
def draw_paths(ax, paths_km, color, lw=1.8, alpha=0.9, zorder=7):
    """Draw each path as a closed loop line."""
    for i, (xs, ys) in enumerate(paths_km):
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                zorder=zorder, solid_capstyle="round", solid_joinstyle="round",
                label="_nolegend_" if i > 0 else None)
        # dot at each waypoint
        ax.scatter(xs[:-1], ys[:-1], color=color, s=18,
                   zorder=zorder+1, edgecolors="black", linewidths=0.4)
 
# ================================================================
# 4. SUMMARY STATS
# ================================================================
total_danger   = df["danger"].sum()
total_removed  = df["removed"].sum()
pct_removed    = total_removed / total_danger * 100 if total_danger > 0 else 0
 
print(f"\nTotal danger    : {total_danger:.3f}")
print(f"Danger removed  : {total_removed:.3f}  ({pct_removed:.1f}%)")
print(f"Residual danger : {total_danger-total_removed:.3f}  ({100-pct_removed:.1f}%)")
print(f"Stationary: {len(sp_locs)}  Cameras: {len(cam_locs)}  "
      f"Drone paths: {len(drone_paths_km)}  Human paths: {len(human_paths_km)}")
 
# ================================================================
# 5. FIGURE 1 — THREE-PANEL
# ================================================================
print("\nGenerating Figure 1 — 3-panel ...")
vmax_d = np.nanmax(danger_grid)
cmap_d = plt.cm.YlOrRd
norm_d = mcolors.Normalize(vmin=0, vmax=vmax_d)
 
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor="white")
plt.subplots_adjust(wspace=0.3)
 
# Panel A — original danger
ax = axes[0]
style_ax(ax)
ax.pcolormesh(XI, YI, danger_grid, cmap=cmap_d, norm=norm_d, shading="nearest", zorder=1)
draw_boundary(ax, color="black")
ax.set_title("Danger Map (No Security Measures)", color="black",
             fontsize=11, fontweight="bold")
add_cbar(fig, ax, plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_d), "Danger [0–2]")
 
# Panel B — residual danger
ax = axes[1]
style_ax(ax)
ax.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_d, shading="nearest", zorder=1)
draw_boundary(ax, color="black")
ax.set_title(f"Residual danger after deployment\n({pct_removed:.1f}% neutralised)",
             color="black", fontsize=11, fontweight="bold")
add_cbar(fig, ax, plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_d), "Residual danger [0–2]")
 
# Panel C — asset placements on residual
ax = axes[2]
style_ax(ax)
ax.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_d,
              shading="nearest", zorder=1, alpha=0.75)
draw_boundary(ax, color="black")
 
# Fixed assets as scatter
if len(sp_locs):
    ax.scatter(sp_locs[:,0], sp_locs[:,1], marker="o", c=SP_COLOR,
               s=80, zorder=6, edgecolors="black", linewidths=0.5)
if len(cam_locs):
    ax.scatter(cam_locs[:,0], cam_locs[:,1], marker="s", c=CAM_COLOR,
               s=60, zorder=6, edgecolors="black", linewidths=0.5)
 
# Paths as lines
draw_paths(ax, drone_paths_km, DRONE_COLOR, lw=2.0)
draw_paths(ax, human_paths_km, HUMAN_COLOR, lw=1.8)
 
# Manual legend
legend_handles = [
    mlines.Line2D([],[],marker="o", color="w", markerfacecolor=SP_COLOR,
                  markersize=8, label=f"Stationary (×{len(sp_locs)})"),
    mlines.Line2D([],[],marker="s", color="w", markerfacecolor=CAM_COLOR,
                  markersize=8, label=f"Camera (×{len(cam_locs)})"),
    mlines.Line2D([],[],color=DRONE_COLOR, linewidth=2,
                  label=f"Drone path (×{len(drone_paths_km)})"),
    mlines.Line2D([],[],color=HUMAN_COLOR, linewidth=2,
                  label=f"Patrol path (×{len(human_paths_km)})"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
          facecolor="white", edgecolor="#aaaaaa", labelcolor="black", framealpha=0.9)
ax.set_title("Optimal asset placement\n(on residual danger map)",
             color="black", fontsize=11, fontweight="bold")
add_cbar(fig, ax, plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_d), "Residual danger [0–2]")
 
fig.suptitle("Etosha National Park — Security Optimisation Results",
             color="black", fontsize=14, fontweight="bold", y=0.8)
fig.tight_layout()
plt.savefig("cvxopt_results_fr/etosha_results_3panel.png", dpi=160, bbox_inches="tight",
            facecolor="white")
print("  Saved → etosha_results_3panel.png")
plt.close(fig)
 
# ================================================================
# 6. FIGURE 2 — DANGER REDUCTION MAP
# ================================================================
print("Generating Figure 2 — reduction map ...")
reduction_grid = removed_grid / np.where(danger_grid > 0, danger_grid, np.nan)
 
fig2, ax2 = plt.subplots(figsize=(14, 7), facecolor="white")
style_ax(ax2)
cmap_g = plt.cm.RdYlGn
norm_g = mcolors.Normalize(vmin=0, vmax=1)
ax2.pcolormesh(XI, YI, reduction_grid, cmap=cmap_g, norm=norm_g, shading="nearest", zorder=1)
draw_boundary(ax2, color="black")
 
if len(sp_locs):
    ax2.scatter(sp_locs[:,0], sp_locs[:,1], marker="o", c=SP_COLOR,
                s=80, zorder=6, edgecolors="black", linewidths=0.5)
if len(cam_locs):
    ax2.scatter(cam_locs[:,0], cam_locs[:,1], marker="s", c=CAM_COLOR,
                s=60, zorder=6, edgecolors="black", linewidths=0.5)
draw_paths(ax2, drone_paths_km, DRONE_COLOR, lw=2.0)
draw_paths(ax2, human_paths_km, HUMAN_COLOR, lw=1.8)
 
ax2.legend(handles=legend_handles, loc="upper left", fontsize=9,
           facecolor="white", edgecolor="#aaaaaa", labelcolor="black", framealpha=0.9)
add_cbar(fig2, ax2, plt.cm.ScalarMappable(norm=norm_g, cmap=cmap_g),
         "Fraction of danger neutralised  (0 = none, 1 = fully)")
ax2.set_title("Danger Reduction per Cell — Etosha National Park\n"
              "Green = fully neutralised  |  Red = untouched",
              color="black", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("cvxopt_results_fr/etosha_results_reduction.png", dpi=160, bbox_inches="tight",
            facecolor="white")
print("  Saved → etosha_results_reduction.png")
plt.close(fig2)
 
# ================================================================
# 7. FIGURE 3 — PER-ASSET-TYPE MAPS (2x2)
# ================================================================
print("Generating Figure 3 — per-asset maps ...")
fig3, axes3 = plt.subplots(2, 2, figsize=(18, 10), facecolor="white")
plt.subplots_adjust(wspace=0.25, hspace=0.35)
ax_sp, ax_cam, ax_drone, ax_human = axes3.flatten()
 
# Stationary
style_ax(ax_sp)
ax_sp.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_d,
                 shading="nearest", zorder=1, alpha=0.6)
draw_boundary(ax_sp, lw=1.2, color="black")
if len(sp_locs):
    ax_sp.scatter(sp_locs[:,0], sp_locs[:,1], marker="o", c=SP_COLOR,
                  s=100, zorder=6, edgecolors="black", linewidths=0.6)
ax_sp.set_title(f"Stationary person  —  {len(sp_locs)} placed",
                color=SP_COLOR, fontsize=11, fontweight="bold")
 
# Camera
style_ax(ax_cam)
ax_cam.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_d,
                  shading="nearest", zorder=1, alpha=0.6)
draw_boundary(ax_cam, lw=1.2, color="black")
if len(cam_locs):
    ax_cam.scatter(cam_locs[:,0], cam_locs[:,1], marker="s", c=CAM_COLOR,
                   s=100, zorder=6, edgecolors="black", linewidths=0.6)
ax_cam.set_title(f"Camera  —  {len(cam_locs)} placed",
                 color=CAM_COLOR, fontsize=11, fontweight="bold")
 
# Drone paths
style_ax(ax_drone)
ax_drone.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_d,
                    shading="nearest", zorder=1, alpha=0.6)
draw_boundary(ax_drone, lw=1.2, color="black")
draw_paths(ax_drone, drone_paths_km, DRONE_COLOR, lw=2.2, alpha=0.95)
ax_drone.set_title(f"Drone paths  —  {len(drone_paths_km)} selected",
                   color=DRONE_COLOR, fontsize=11, fontweight="bold")
 
# Human patrol paths
style_ax(ax_human)
ax_human.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_d,
                    shading="nearest", zorder=1, alpha=0.6)
draw_boundary(ax_human, lw=1.2, color="black")
draw_paths(ax_human, human_paths_km, HUMAN_COLOR, lw=2.2, alpha=0.95)
ax_human.set_title(f"Patrol paths  —  {len(human_paths_km)} selected",
                   color=HUMAN_COLOR, fontsize=11, fontweight="bold")
 
fig3.suptitle("Per-Asset-Type Maps — Etosha",
              color="black", fontsize=13, fontweight="bold", y=0.8)
plt.savefig("cvxopt_results_fr/etosha_results_per_asset.png", dpi=160, bbox_inches="tight",
            facecolor="white")
print("  Saved → etosha_results_per_asset.png")
plt.close(fig3)
 
# ================================================================
# 8. FIGURE 4 — FIRE vs ANIMAL RISK COMPONENTS
# ================================================================
print("Generating Figure 4 — risk components ...")
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
plt.subplots_adjust(wspace=0.3)
 
for ax, grid, title, cmap in [
    (axes4[0], fire_grid,   "Fire risk (normalised)",    plt.cm.hot),
    (axes4[1], animal_grid, "Animal value (normalised)", plt.cm.YlGn),
]:
    style_ax(ax)
    vmax = np.nanmax(grid) if not np.all(np.isnan(grid)) else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    ax.pcolormesh(XI, YI, grid, cmap=cmap, norm=norm, shading="nearest", zorder=1)
    draw_boundary(ax, color="black")
    ax.set_title(title, color="black", fontsize=11, fontweight="bold")
    add_cbar(fig4, ax, plt.cm.ScalarMappable(norm=norm, cmap=cmap), title)
 
fig4.suptitle("Etosha — Input Risk Components", color="black",
              fontsize=13, fontweight="bold", y=0.8)
plt.savefig("etosha_results_components.png", dpi=160, bbox_inches="tight",
            facecolor="white")
print("  Saved → etosha_results_components.png")
plt.close(fig4)
 
print("\nAll done. Four PNG files saved.")