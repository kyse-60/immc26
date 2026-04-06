"""
Etosha Security Placement — Results Visualizer
===============================================
Panels B & C only (residual danger + asset placements)
for budgets: 20k, 40k, 60k.
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
BUDGETS = [
    {
        "label":           "70k",
        "results_csv":     "etosha_path_results_0.5change.csv",
        "drone_paths_csv": "selected_drone_paths.csv",
        "human_paths_csv": "selected_human_paths.csv",
        "fire_csv":        "fire_risk_5km.csv",
        "out_png":         "etosha_results_0.5.png",
    },
]
 
SHP_PATH    = "WDPA_WDOECM_Apr2026_Public_884_shp-polygons.shp"
DRONE_COLOR = "#0026ff"
HUMAN_COLOR = "#00fff2"
SP_COLOR    = "#a600ff"
CAM_COLOR   = "#00ff00"
 
# ================================================================
# PARK BOUNDARY — load once
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
 
def draw_boundary(ax, lw=1.8, color="black"):
    if not boundary_available:
        return
    for geom in park_proj.geometry:
        parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for part in parts:
            xs, ys = part.exterior.xy
            ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
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
    for xs, ys in paths_km:
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                zorder=zorder, solid_capstyle="round", solid_joinstyle="round")
        #ax.scatter(xs[:-1], ys[:-1], color=color, s=2,
                   #zorder=zorder + 1, edgecolors="black", linewidths=0.4)
 
# ================================================================
# MAIN LOOP
# ================================================================
for cfg in BUDGETS:
    label = cfg["label"]
    print(f"\n{'='*50}\n  Budget: NAD {label}\n{'='*50}")
 
    if not os.path.exists(cfg["results_csv"]):
        print(f"  Not found: {cfg['results_csv']} — skipping.")
        continue
 
    # Load results
    df = pd.read_csv(cfg["results_csv"])
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
    XI, YI = np.meshgrid(np.array(easting), np.array(northing))
 
    sp_locs  = df[df["has_stationary"] == 1][["easting_km", "northing_km"]].values \
               if "has_stationary" in df.columns else np.empty((0, 2))
    cam_locs = df[df["has_camera"] == 1][["easting_km", "northing_km"]].values \
               if "has_camera"    in df.columns else np.empty((0, 2))
 
    # Load fire grid for path index → km conversion
    fire_df = pd.read_csv(cfg["fire_csv"], index_col=0)
    fire_df.index   = fire_df.index.astype(float)
    fire_df.columns = fire_df.columns.astype(float)
    rows_km = fire_df.index.values
    cols_km = fire_df.columns.values
 
    def str_to_path(s):
        return [tuple(int(x) for x in p.split(",")) for p in s.split(";")]
 
    def path_to_km(path):
        xs = [cols_km[c] for r, c in path] + [cols_km[path[0][1]]]
        ys = [rows_km[r] for r, c in path] + [rows_km[path[0][0]]]
        return xs, ys
 
    def load_paths(csv_path, lbl):
        paths = []
        if not os.path.exists(csv_path):
            print(f"  {csv_path} not found — {lbl} won't be drawn")
            return paths
        try:
            pdf = pd.read_csv(csv_path)
            if pdf.empty or "path_str" not in pdf.columns:
                print(f"  {lbl}: 0 paths (empty file)")
                return paths
            for s in pdf["path_str"]:
                paths.append(path_to_km(str_to_path(s)))
            print(f"  {lbl}: {len(paths)} loaded")
        except Exception as e:
            print(f"  {lbl}: could not load ({e})")
        return paths
 
    drone_paths_km = load_paths(cfg["drone_paths_csv"], "Drone paths")
    human_paths_km = load_paths(cfg["human_paths_csv"], "Human paths")
 
    total_danger  = df["danger"].sum()
    total_removed = df["removed"].sum() if "removed" in df.columns else 0
    pct_removed   = total_removed / total_danger * 100 if total_danger > 0 else 0
    print(f"  {pct_removed:.1f}% neutralised | "
          f"SP:{len(sp_locs)} CAM:{len(cam_locs)} "
          f"DR:{len(drone_paths_km)} HU:{len(human_paths_km)}")
 
    # Colour scale
    vmax_d = np.nanmax(danger_grid)
    cmap_d = plt.cm.YlOrRd
    norm_d = mcolors.Normalize(vmin=0, vmax=vmax_d)
 
    # Legend handles
    legend_handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=SP_COLOR,
                      markeredgecolor="black", markeredgewidth=0.5,
                      markersize=8, label=f"Stationary (×{len(sp_locs)})"),
        mlines.Line2D([], [], marker="s", color="w", markerfacecolor=CAM_COLOR,
                      markeredgecolor="black", markeredgewidth=0.5,
                      markersize=8, label=f"Camera (×{len(cam_locs)})"),
        mlines.Line2D([], [], color=DRONE_COLOR, linewidth=2,
                      label=f"Drone path (×{len(drone_paths_km)})"),
        mlines.Line2D([], [], color=HUMAN_COLOR, linewidth=2,
                      label=f"Patrol path (×{len(human_paths_km)})"),
    ]
 
    # Figure: 2 panels, right margin reserved for legend
    fig, axes = plt.subplots(1, 2, figsize=(16, 3), facecolor="white")
    fig.subplots_adjust(top =0.88,left=0.06, right=0.78, wspace=0.28)
 
    # # Panel B — residual danger
    # ax = axes[0]
    # style_ax(ax)
    # ax.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_d,
    #               shading="nearest", zorder=1)
    # draw_boundary(ax)
    # ax.set_title(f"Residual Danger After Optimal Safety Measures\n({pct_removed:.1f}% neutralised)",
    #              color="black", fontsize=11, fontweight="bold")
    # add_cbar(fig, ax, plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_d), "Residual danger [0–2]")

    # Panel B — raw danger
    ax = axes[0]
    style_ax(ax)
    ax.pcolormesh(XI, YI, danger_grid, cmap=cmap_d, norm=norm_d,
                shading="nearest", zorder=1)
    draw_boundary(ax)
    ax.set_title("Danger Map (No Safety Measures)",
                color="black", fontsize=11, fontweight="bold")
    add_cbar(fig, ax, plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_d), "Danger [0–2]")
 
    # Panel C — asset placements on residual
    ax = axes[1]
    style_ax(ax)
    ax.pcolormesh(XI, YI, residual_grid, cmap=cmap_d, norm=norm_d,
                  shading="nearest", zorder=1, alpha=0.75)
    draw_boundary(ax)
    if len(sp_locs):
        ax.scatter(sp_locs[:, 0], sp_locs[:, 1], marker="o", c=SP_COLOR,
                   s=15, zorder=6, edgecolors="black", linewidths=0.5)
    if len(cam_locs):
        ax.scatter(cam_locs[:, 0], cam_locs[:, 1], marker="s", c=CAM_COLOR,
                   s=15, zorder=6, edgecolors="black", linewidths=0.5)
    draw_paths(ax, drone_paths_km, DRONE_COLOR, lw=2.0)
    draw_paths(ax, human_paths_km, HUMAN_COLOR, lw=1.8)
    ax.set_title("Optimal Safety Measure Placement",
                 color="black", fontsize=11, fontweight="bold")
    add_cbar(fig, ax, plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_d), "Residual danger [0–2]")
 
    # Legend outside the axes, clear of the colorbar
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.18, 1.0),
        fontsize=9,
        facecolor="white",
        edgecolor="#aaaaaa",
        labelcolor="black",
        framealpha=0.95,
        borderpad=0.8,
        handlelength=2.2,
    )
 
    fig.suptitle(
        f"Security Optimization (λ = 0.5)",
        color="black", fontsize=14, fontweight="bold", y=0.98, x=0.37
    )
    fig.tight_layout(rect=[0,0,0.79,1])
 
    os.makedirs(os.path.dirname(cfg["out_png"]) or ".", exist_ok=True)
    plt.savefig(cfg["out_png"], dpi=160, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {cfg['out_png']}")
    plt.close(fig)
 
print("\nAll done.")