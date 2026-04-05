"""
Animal Value Density Map — Etosha National Park
================================================
Pipeline:
  1. Parse the animal × lodge probability matrix
  2. For each animal, interpolate sighting probability to a 1 km² grid
     (Ordinary Kriging or IDW — selectable)
  3. Multiply by Price × Population → per-animal value density
  4. Sum all animals → total value density map
  5. Clip to park shapefile, discretize on 1 km AEQD grid centred on park centroid
  6. Plot and save

Dependencies:
    pip install pandas numpy matplotlib pyproj geopandas scipy pykrige shapely
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import shapely
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging


# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH        = "NEWERanimalmap.csv"
SHP_PATH        = "WDPA_WDOECM_Apr2026_Public_884_shp-polygons.shp"   # ← update to your .shp path
OUTPUT_PNG      = "etosha_animal_value_map.png"
GRID_RES_KM     = 1              # 1 km × 1 km grid cells

INTERP_METHOD   = "kriging"      # "kriging"  or  "idw"

# Kriging variogram settings — if you have only 16 points, DO NOT blindly fit
# these from data. Instead, set them based on domain knowledge:
#   range_km  ~ the distance beyond which sighting probabilities are uncorrelated
#              (a reasonable first guess for Etosha: ~80–150 km)
#   sill      ~ variance of the data (leave as None to auto-estimate from data)
#   nugget    ~ measurement/small-scale noise (leave as 0 if you trust the data)
VARIOGRAM_MODEL  = "spherical"   # 'linear', 'power', 'gaussian', 'spherical', 'exponential'
VARIOGRAM_RANGE_KM = 100.0       # ← tune this; controls smoothness of the map
FIT_VARIOGRAM    = False         # False = use fixed range above (safer with 16 pts)
                                 # True  = let pykrige fit from data (risky with 16 pts)
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. Parse CSV ──────────────────────────────────────────────────────────────
df_raw = pd.read_csv(CSV_PATH, header=0, index_col=0)
df_raw.index = df_raw.index.str.strip()        # ← strips leading/trailing whitespace
df_raw.columns = df_raw.columns.str.strip()    # ← same fix for column names just in case

# The last two rows are coordinates; the last two columns are Price & Population
location_cols = [c for c in df_raw.columns if c not in ["Price", "Population"]]

lats = df_raw.loc["Latitude",  location_cols].values.astype(float)
lons = df_raw.loc["Longitude", location_cols].values.astype(float)

animal_df    = df_raw.drop(["Latitude", "Longitude"])
prices       = animal_df["Price"].values.astype(float)
populations  = animal_df["Population"].values.astype(float)
prob_matrix  = animal_df[location_cols].values.astype(float)  # (n_animals, n_locs)
animal_names = animal_df.index.tolist()

print(f"Animals loaded  : {len(animal_names)}")
print(f"Locations loaded: {len(lats)}")


# ── 2. AEQD projection centred on shapefile centroid ─────────────────────────
os.environ["SHAPE_RESTORE_SHX"] = "YES"
park = gpd.read_file(SHP_PATH)
park = park.set_crs("EPSG:4326") if park.crs is None else park.to_crs("EPSG:4326")

centroid         = park.geometry.union_all().centroid
center_lon, center_lat = centroid.x, centroid.y
print(f"Park centroid: lon={center_lon:.4f}  lat={center_lat:.4f}")

proj_str = (
    f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
    "+datum=WGS84 +units=m +no_defs"
)

transformer = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)

# Lodge coordinates in projected km
x_m, y_m = transformer.transform(lons, lats)
x_km = x_m / 1000.0
y_km = y_m / 1000.0

# Reproject park polygon
park_proj  = park.to_crs(proj_str)
park_union = park_proj.geometry.union_all()

bounds = park_proj.total_bounds          # [xmin_m, ymin_m, xmax_m, ymax_m]
xmin_km = bounds[0] / 1000
xmax_km = bounds[2] / 1000
ymin_km = bounds[1] / 1000
ymax_km = bounds[3] / 1000


# ── 3. Build 1 km grid ───────────────────────────────────────────────────────
xi = np.arange(xmin_km, xmax_km + GRID_RES_KM, GRID_RES_KM)  # 1-D x coords
yi = np.arange(ymin_km, ymax_km + GRID_RES_KM, GRID_RES_KM)  # 1-D y coords
XI, YI = np.meshgrid(xi, yi)
grid_shape = XI.shape
print(f"Grid: {grid_shape[1]} cols × {grid_shape[0]} rows  ({grid_shape[0]*grid_shape[1]:,} cells)")


# ── 4. Park mask — True where grid cell centre is inside the park ─────────────
# shapely 2.x vectorised contains_xy (works in metres, park_union is in metres)
pts_x_m = (XI * 1000).ravel()
pts_y_m = (YI * 1000).ravel()
inside_mask = shapely.contains_xy(park_union, pts_x_m, pts_y_m).reshape(grid_shape)
print(f"Cells inside park: {inside_mask.sum():,}")


# ── 5. Interpolation functions ────────────────────────────────────────────────

def interpolate_kriging(x_km, y_km, z, xi, yi):
    """
    Ordinary Kriging using pykrige.

    With only 16 points, FIT_VARIOGRAM=False is strongly recommended:
      - We fix the variogram range to VARIOGRAM_RANGE_KM
      - sill is estimated as the variance of the data (reasonable)
      - nugget is set to 0 (no measurement error assumed)
    
    Always clips output to [0, 1] since kriging can extrapolate beyond
    probability bounds.
    """
    if FIT_VARIOGRAM:
        # Let pykrige fit everything — unreliable with 16 points
        ok = OrdinaryKriging(
            x_km, y_km, z,
            variogram_model=VARIOGRAM_MODEL,
            nlags=4,        # fewer lags = more stable with sparse data
            weight=True,    # weight fit by number of pairs per lag
            verbose=False,
            enable_plotting=False,
        )
    else:
        # Fix parameters based on domain knowledge — much safer
        data_sill = float(np.var(z))
        ok = OrdinaryKriging(
            x_km, y_km, z,
            variogram_model=VARIOGRAM_MODEL,
            variogram_parameters={
                "sill":   data_sill,
                "range":  VARIOGRAM_RANGE_KM,
                "nugget": 0.0,
            },
            verbose=False,
            enable_plotting=False,
        )

    # pykrige.execute("grid", ...) takes 1-D xi and yi and grids internally
    z_grid, _ = ok.execute("grid", xi, yi)
    return np.clip(z_grid.data, 0.0, 1.0)   # enforce [0, 1]


def interpolate_idw(x_km, y_km, z, XI, YI, power=2):
    """
    Inverse Distance Weighting.
    Simple, no distributional assumptions, always stays within data range.
    power=2 is standard; increase for more local influence.
    """
    pts   = np.column_stack([x_km, y_km])                          # (n_locs, 2)
    query = np.column_stack([XI.ravel(), YI.ravel()])               # (n_cells, 2)
    dists = np.linalg.norm(query[:, None, :] - pts[None, :, :], axis=2)  # (n_cells, n_locs)

    # Avoid division by zero at exact lodge locations
    exact   = dists == 0
    weights = np.where(exact, 1e12, 1.0 / np.maximum(dists, 1e-12) ** power)
    weights /= weights.sum(axis=1, keepdims=True)

    z_flat = (weights @ z).reshape(XI.shape)
    return np.clip(z_flat, 0.0, 1.0)


# ── 6. Per-animal interpolation → value density → accumulate ─────────────────
total_value = np.full(grid_shape, np.nan)
animal_maps = {}

for i, name in enumerate(animal_names):
    z   = prob_matrix[i]
    price = prices[i]
    pop   = populations[i]

    if INTERP_METHOD == "kriging":
        prob_grid = interpolate_kriging(x_km, y_km, z, xi, yi)
    else:
        prob_grid = interpolate_idw(x_km, y_km, z, XI, YI)

    # Mask outside park
    prob_grid_masked = np.where(inside_mask, prob_grid, np.nan)

    # ── KEY CHANGE ────────────────────────────────────────────────────────────
    # Distribute population proportionally across cells.
    # Each cell gets: (its prob / sum of all probs in park) × total population
    # So summing animal_density over all park cells = Population exactly.
    prob_sum = np.nansum(prob_grid_masked)   # total probability mass inside park

    if prob_sum > 0:
        animal_density = prob_grid_masked / prob_sum * pop  # animals per cell
    else:
        animal_density = np.zeros(grid_shape)

    # Value density = animals per cell × price
    value_grid = np.where(inside_mask, animal_density * price, np.nan)
    # ─────────────────────────────────────────────────────────────────────────

    animal_maps[name] = {"prob": prob_grid_masked, 
                         "density": animal_density,
                         "value": value_grid}

    # Accumulate
    if i == 0:
        total_value = value_grid.copy()
    else:
        total_value = np.where(
            inside_mask,
            np.nansum(np.stack([total_value, value_grid]), axis=0),
            np.nan
        )

    print(f"  {name:<22s}  total animals distributed: {np.nansum(animal_density):,.1f} "
          f"(should = {pop:,})   total value: ${np.nansum(value_grid):,.0f}")


# ── 7. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9), facecolor="#0d1117")
ax.set_facecolor("#0d1117")

cmap = plt.cm.YlOrRd
vmin = np.nanpercentile(total_value, 2)    # avoid outlier stretching
vmax = np.nanpercentile(total_value, 98)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

ax.pcolormesh(XI, YI, total_value, cmap=cmap, norm=norm,
              shading="auto", zorder=1)

# Park boundary
for geom in park_proj.geometry:
    if geom.geom_type == "Polygon":
        xs, ys = geom.exterior.xy
        ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
                color="white", linewidth=1.8, zorder=4)
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            xs, ys = part.exterior.xy
            ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
                    color="white", linewidth=1.8, zorder=4)

# Lodge markers & labels
ax.scatter(x_km, y_km, c="cyan", s=50, zorder=5,
           edgecolors="black", linewidths=0.5, label="Lodges / camps")
for j, loc in enumerate(location_cols):
    ax.annotate(loc, (x_km[j], y_km[j]), fontsize=6.5,
                color="white", xytext=(5, 4), textcoords="offset points",
                zorder=6)

# Centroid crosshair
ax.plot(0, 0, "+", color="white", markersize=14,
        markeredgewidth=2.5, zorder=6, label="Park centroid (0, 0)")

# Formatting
ax.grid(True, linestyle="--", alpha=0.15, color="white", zorder=0)
ax.set_xlabel("Easting from centroid (km)",  fontsize=11, color="#cccccc")
ax.set_ylabel("Northing from centroid (km)", fontsize=11, color="#cccccc")
ax.tick_params(colors="#aaaaaa", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")

method_label = "Ordinary Kriging" if INTERP_METHOD == "kriging" else "IDW (power=2)"
ax.set_title(
    f"Animal Sighting Value Density — Etosha National Park\n"
    f"Σ  P(sighting) × Price × Population  |  1 km² grid  |  {method_label}",
    fontsize=13, fontweight="bold", color="white", pad=16,
)

sm   = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
cbar.set_label("Σ  P(sighting) × Price × Population",
               fontsize=10, color="#cccccc")
cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa")

handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h
ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=9,
          facecolor="#1e2433", edgecolor="#555555", labelcolor="white")

ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
# print(f"\nSaved → {OUTPUT_PNG}")
# plt.show()


# ── 8. Plot all per-animal probability maps in a grid ────────────────────────
n_animals = len(animal_names)
n_cols    = 4
n_rows    = (n_animals + n_cols - 1) // n_cols   # ceiling division

fig2, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * 5, n_rows * 4),
                          facecolor="#0d1117")
axes_flat = axes.flatten()

cmap_prob = plt.cm.YlGn
norm_prob  = mcolors.Normalize(vmin=0, vmax=1)

for i, name in enumerate(animal_names):
    ax = axes_flat[i]
    ax.set_facecolor("#0d1117")

    prob_grid = animal_maps[name]["prob"]

    ax.pcolormesh(XI, YI, prob_grid, cmap=cmap_prob, norm=norm_prob,
                  shading="auto", zorder=1)

    # Park boundary
    for geom in park_proj.geometry:
        if geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
                    color="white", linewidth=0.8, zorder=3)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                xs, ys = part.exterior.xy
                ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
                        color="white", linewidth=0.8, zorder=3)

    # Lodge dots
    ax.scatter(x_km, y_km, c="cyan", s=10, zorder=4,
               edgecolors="none")

    # Title with price × population weight
    w = prices[animal_names.index(name)] * populations[animal_names.index(name)]
    ax.set_title(f"{name}\n$P \\times$ pop: {w:,.0f}",
                 fontsize=8, color="white", pad=4)

    ax.tick_params(colors="#aaaaaa", labelsize=6)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

# Hide any unused subplot panels
for j in range(n_animals, len(axes_flat)):
    axes_flat[j].set_visible(False)

# Shared colorbar
sm2   = plt.cm.ScalarMappable(norm=norm_prob, cmap=cmap_prob)
cbar2 = fig2.colorbar(sm2, ax=axes_flat[:n_animals],
                      pad=0.02, fraction=0.015, location="right")
cbar2.set_label("P(sighting)  [0 = never → 1 = always]",
                fontsize=9, color="#cccccc")
cbar2.ax.yaxis.set_tick_params(color="#aaaaaa")
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="#aaaaaa")

fig2.suptitle(
    f"Per-Animal Sighting Probability Maps — Etosha  ({method_label})",
    fontsize=13, fontweight="bold", color="white", y=1.01
)

plt.tight_layout()
fig2.savefig("etosha_per_animal_maps.png", dpi=150,
             bbox_inches="tight", facecolor=fig2.get_facecolor())
print("Saved → etosha_per_animal_maps.png")
plt.show()