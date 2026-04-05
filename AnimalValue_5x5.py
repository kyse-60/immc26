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

# ── 5b. Variogram diagnostic plot ────────────────────────────────────────────

def plot_variogram_diagnostic(x_km, y_km, z, animal_name="[first animal]"):
    """
    Fits (or uses fixed params for) an OrdinaryKriging object on z,
    then extracts the experimental variogram cloud + lags and overlays
    the fitted/fixed model curve.
    """
    from pykrige.ok import OrdinaryKriging
    from pykrige.variogram_models import (
        spherical_variogram_model,
        gaussian_variogram_model,
        exponential_variogram_model,
        linear_variogram_model,
        power_variogram_model,
    )

    VARIOGRAM_FUNCS = {
        "spherical":   spherical_variogram_model,
        "gaussian":    gaussian_variogram_model,
        "exponential": exponential_variogram_model,
        "linear":      linear_variogram_model,
        "power":       power_variogram_model,
    }

    # Build the same OK object used in interpolation
    if FIT_VARIOGRAM:
        ok = OrdinaryKriging(
            x_km, y_km, z,
            variogram_model=VARIOGRAM_MODEL,
            nlags=4, weight=True,
            verbose=False, enable_plotting=False,
        )
    else:
        data_sill = float(np.var(z))
        ok = OrdinaryKriging(
            x_km, y_km, z,
            variogram_model=VARIOGRAM_MODEL,
            variogram_parameters={
                "sill":   data_sill,
                "range":  VARIOGRAM_RANGE_KM,
                "nugget": 0.0,
            },
            verbose=False, enable_plotting=False,
        )

    # ── Experimental variogram points ────────────────────────────────────────
    # pykrige stores these after fitting:
    #   ok.lags         → lag bin centres (km)
    #   ok.semivariance → experimental semivariance per bin
    #   ok.variogram_model_parameters → [sill, range, nugget] (spherical/gaussian/exp)
    lags_exp  = ok.lags          # shape (nlags,)
    gamma_exp = ok.semivariance  # shape (nlags,)

    # ── Fitted / fixed model curve ────────────────────────────────────────────
    h_fine = np.linspace(0, lags_exp.max() * 1.1, 300)
    params = ok.variogram_model_parameters   # list: [psill, range, nugget]
    vfunc  = VARIOGRAM_FUNCS[VARIOGRAM_MODEL]
    gamma_model = vfunc(params, h_fine)

    # ── Raw variogram cloud (all point-pairs) ─────────────────────────────────
    pts  = np.column_stack([x_km, y_km])
    n    = len(pts)
    cloud_h, cloud_g = [], []
    for a in range(n):
        for b in range(a + 1, n):
            dist = np.linalg.norm(pts[a] - pts[b])
            sv   = 0.5 * (z[a] - z[b]) ** 2
            cloud_h.append(dist)
            cloud_g.append(sv)
    cloud_h = np.array(cloud_h)
    cloud_g = np.array(cloud_g)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    # Raw cloud
    ax.scatter(cloud_h, cloud_g, s=18, alpha=0.45, color="#4fc3f7",
               zorder=2, label="Variogram cloud\n(all point-pairs)")

    # Binned experimental variogram
    ax.scatter(lags_exp, gamma_exp, s=80, color="#ffd54f", zorder=4,
               edgecolors="white", linewidths=0.8,
               label=f"Experimental variogram\n({len(lags_exp)} lag bins)")

    # Fitted model curve
    fit_label = (
        f"{VARIOGRAM_MODEL.capitalize()} model\n"
        f"sill={params[0]:.4f}  range={params[1]:.1f} km  nugget={params[2]:.4f}"
    )
    ax.plot(h_fine, gamma_model, color="#ef5350", linewidth=2.2,
            zorder=5, label=fit_label)

    # Sill & nugget reference lines
    total_sill = params[0] + params[2]   # psill + nugget
    ax.axhline(total_sill, color="#ef5350", linewidth=0.8,
               linestyle="--", alpha=0.5, zorder=1)
    ax.axhline(params[2],  color="#aaaaaa", linewidth=0.8,
               linestyle=":",  alpha=0.5, zorder=1)
    ax.axvline(params[1],  color="#ef5350", linewidth=0.8,
               linestyle="--", alpha=0.4, zorder=1,
               label=f"Range = {params[1]:.1f} km")

    ax.set_xlabel("Lag distance  (km)",    fontsize=11, color="#cccccc")
    ax.set_ylabel("Semivariance  γ(h)",    fontsize=11, color="#cccccc")
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    fit_mode = "fitted from data" if FIT_VARIOGRAM else "fixed (domain knowledge)"
    ax.set_title(
        f"Variogram Diagnostic  —  '{animal_name}'  probabilities\n"
        f"Model: {VARIOGRAM_MODEL}  |  Parameters: {fit_mode}",
        fontsize=12, fontweight="bold", color="white", pad=12,
    )

    ax.legend(fontsize=8.5, facecolor="#1e2433",
              edgecolor="#555555", labelcolor="white",
              loc="upper left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("etosha_variogram_diagnostic.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Saved → etosha_variogram_diagnostic.png")
    plt.show()


# Call once using the first animal's data as representative
if INTERP_METHOD == "kriging":
    from pykrige.variogram_models import spherical_variogram_model

    targets = ["Spotted Hyena", "Cheetah", "Leopard"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")

    for ax, name in zip(axes, targets):
        ax.set_facecolor("white")
        i = animal_names.index(name)
        z = prob_matrix[i]

        ok = OrdinaryKriging(x_km, y_km, z, variogram_model="spherical",
                             variogram_parameters={"sill": float(np.var(z)),
                                                   "range": VARIOGRAM_RANGE_KM,
                                                   "nugget": 0.0},
                             verbose=False, enable_plotting=False)

        pts     = np.column_stack([x_km, y_km])
        cloud_h = [np.linalg.norm(pts[a] - pts[b])
                   for a in range(len(pts)) for b in range(a+1, len(pts))]
        cloud_g = [0.5*(z[a]-z[b])**2
                   for a in range(len(z))  for b in range(a+1, len(z))]

        h_fine      = np.linspace(0, max(cloud_h) * 1.1, 300)
        gamma_model = spherical_variogram_model(ok.variogram_model_parameters, h_fine)

        sc1 = ax.scatter(cloud_h, cloud_g, s=12, alpha=0.4, color="steelblue", zorder=2)
        sc2 = ax.scatter(ok.lags, ok.semivariance, s=60, color="gold",
                         edgecolors="black", linewidths=0.6, zorder=4)
        ln, = ax.plot(h_fine, gamma_model, color="red", linewidth=2, zorder=5)

        ax.set_title(name, fontsize=11, fontweight="bold", color="black")
        ax.set_xlabel("Lag (km)", fontsize=9, color="black")
        ax.set_ylabel("γ(h)",     fontsize=9, color="black")
        ax.tick_params(colors="black", labelsize=8)
        ax.spines[:].set_edgecolor("black")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        ax.legend([sc1, sc2, ln],
          ["Variogram cloud", "Binned experimental", "Spherical model"],
          fontsize=8, framealpha=1, edgecolor="black", loc="upper right")

    plt.tight_layout()
    plt.savefig("etosha_variogram_selected.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.show()

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


# ================================================================
# AGGREGATE 1 km → 5 km grid
# ================================================================

def aggregate_1km_to_5km(grid_1km, block=5):
    rows, cols = grid_1km.shape
    rows_trim = (rows // block) * block
    cols_trim = (cols // block) * block
    trimmed   = grid_1km[:rows_trim, :cols_trim]

    reshaped  = trimmed.reshape(rows_trim // block, block,
                                cols_trim // block, block)

    all_nan   = np.all(np.isnan(reshaped), axis=(1, 3))

    block_sum = np.nanmean(reshaped, axis=(1, 3)) * block * block
    block_sum[all_nan] = np.nan

    return block_sum

total_value_5km = aggregate_1km_to_5km(total_value)

# Cell centers = left edge + 2.5 km
# Use centers for BOTH the containment check and the plot — no offset mismatch
xi_5km_centers = xi[:total_value_5km.shape[1] * 5 : 5] + 2.5
yi_5km_centers = yi[:total_value_5km.shape[0] * 5 : 5] + 2.5
XI_5km, YI_5km = np.meshgrid(xi_5km_centers, yi_5km_centers)

center_inside = shapely.contains_xy(
    park_union,
    (XI_5km * 1000).ravel(),
    (YI_5km * 1000).ravel()
).reshape(total_value_5km.shape)

total_value_5km[~center_inside] = np.nan

# ================================================================
# SAVE PER-ANIMAL 5km PROBABILITY MAPS TO CSV
# ================================================================
import os
os.makedirs("animal_maps", exist_ok=True)

for name, maps in animal_maps.items():
    prob_grid_1km = maps["prob"]
    prob_grid_5km = aggregate_1km_to_5km(prob_grid_1km) / 25  # mean probability
    prob_grid_5km[~center_inside] = np.nan

    safe_name = name.replace(" ", "_").replace("/", "-")
    df_animal = pd.DataFrame(
        prob_grid_5km,
        index=np.round(yi_5km_centers, 2),
        columns=np.round(xi_5km_centers, 2)
    )
    df_animal.to_csv(f"animal_maps/{safe_name}_5km.csv")

print(f"Saved {len(animal_maps)} per-animal CSVs → animal_maps/")



print(f"\n1 km grid : {total_value.shape}")
print(f"5 km grid : {total_value_5km.shape}")
print(f"Cells inside park (5km, centre-point rule): {center_inside.sum()}")

# Save — index/columns are true cell centers in km
df_5km = pd.DataFrame(
    total_value_5km,
    index=np.round(yi_5km_centers, 2),
    columns=np.round(xi_5km_centers, 2)
)
df_5km.to_csv("animal_value_5km.csv")
print("Saved → animal_value_5km.csv")



# ── 7. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
ax.set_facecolor("white")

cmap = plt.cm.YlOrRd
vmin = np.nanpercentile(total_value, 2)
vmax = np.nanpercentile(total_value, 98)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

ax.pcolormesh(XI, YI, total_value, cmap=cmap, norm=norm,
              shading="auto", zorder=1)

# Park boundary
for geom in park_proj.geometry:
    if geom.geom_type == "Polygon":
        xs, ys = geom.exterior.xy
        ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
                color="black", linewidth=1.8, zorder=4)
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            xs, ys = part.exterior.xy
            ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
                    color="black", linewidth=1.8, zorder=4)

# Lodge markers & labels
ax.scatter(x_km, y_km, c="steelblue", s=50, zorder=5,
           edgecolors="black", linewidths=0.5, label="Lodges / camps")
for j, loc in enumerate(location_cols):
    ax.annotate(loc, (x_km[j], y_km[j]), fontsize=6.5,
                color="black", xytext=(5, 4), textcoords="offset points",
                zorder=6)

# Centroid crosshair
ax.plot(0, 0, "+", color="black", markersize=14,
        markeredgewidth=2.5, zorder=6, label="Park centroid (0, 0)")

# Formatting
ax.grid(True, linestyle="--", alpha=0.15, color="black", zorder=0)
ax.set_xlabel("East from centroid (km)",  fontsize=11, color="black")
ax.set_ylabel("North from centroid (km)", fontsize=11, color="black")
ax.tick_params(colors="black", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("black")

method_label = "Ordinary Kriging" if INTERP_METHOD == "kriging" else "IDW (power=2)"
ax.set_title(
    f"Animal Sighting Value Density — Etosha National Park\n"
    f"Σ  P(sighting) × Price × Population",
    fontsize=13, fontweight="bold", color="black", pad=16,
)

sm   = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
cbar.set_label("Σ  P(sighting) × Price × Population",
               fontsize=10, color="black")
cbar.ax.yaxis.set_tick_params(color="black")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h
ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=9,
          facecolor="white", edgecolor="black", labelcolor="black")

ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())

# ── 8. Plot all per-animal probability maps in a grid ────────────────────────
n_animals = len(animal_names)
n_cols    = 4
n_rows    = (n_animals + n_cols - 1) // n_cols

fig2, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * 5, n_rows * 2.2),
                          facecolor="white")

axes_flat = axes.flatten()

cmap_prob = plt.cm.YlGnBu
norm_prob  = mcolors.Normalize(vmin=0, vmax=1)

for i, name in enumerate(animal_names):
    ax = axes_flat[i]
    ax.set_facecolor("white")


    # ── Aggregate 1 km prob grid → 5 km ──────────────────────────────────
    prob_grid_1km = animal_maps[name]["prob"]
    prob_grid_5km = aggregate_1km_to_5km(prob_grid_1km) / 25  # mean, not sum
    prob_grid_5km[~center_inside] = np.nan

    ax.pcolormesh(XI_5km, YI_5km, prob_grid_5km,
                  cmap=cmap_prob, norm=norm_prob,
                  shading="nearest", zorder=1)

    # Park boundary
    for geom in park_proj.geometry:
        if geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
                    color="black", linewidth=0.8, zorder=3)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                xs, ys = part.exterior.xy
                ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
                        color="black", linewidth=0.8, zorder=3)
    ax.scatter(x_km, y_km, c="cyan", s=10, zorder=4, edgecolors="none")
    ax.set_title(f"{name}\npopulation: {populations[i]:,.0f}",
                 fontsize=8, color="black", pad=4)

    ax.tick_params(colors="black", labelsize=6)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")



for j in range(n_animals, len(axes_flat)):
    axes_flat[j].set_visible(False)

# Colorbar in its own axis to the right of the whole figure — no overlap
fig2.subplots_adjust(right=0.88)
cbar_ax = fig2.add_axes([0.90, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
sm2 = plt.cm.ScalarMappable(norm=norm_prob, cmap=cmap_prob)
cbar2 = fig2.colorbar(sm2, cax=cbar_ax)
cbar2.set_label("Probability of sighting)",
                fontsize=9, color="black")
cbar2.ax.yaxis.set_tick_params(color="black")
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="black")

fig2.suptitle(
    f"Per-Animal Sighting Probability Maps — Etosha National Park",
    fontsize=13, fontweight="bold", color="black", y=0.98
)
plt.tight_layout(rect=[0,0,0.88, 0.97])
plt.savefig("etosha_per_animal_maps.png", dpi=150,
            bbox_inches="tight", facecolor=fig2.get_facecolor())
print("Saved → etosha_per_animal_maps.png")
plt.show()

# ================================================================
# PLOT 5 km PIXELATED MAP
# ================================================================
fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
ax.set_facecolor("white")

cmap = plt.cm.YlOrRd
vmin = np.nanpercentile(total_value_5km, 2)
vmax = np.nanpercentile(total_value_5km, 98)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

ax.pcolormesh(XI_5km, YI_5km, total_value_5km,
              cmap=cmap, norm=norm, shading="nearest", zorder=1)

# Park boundary
for geom in park_proj.geometry:
    if geom.geom_type == "Polygon":
        xs, ys = geom.exterior.xy
        ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
                color="black", linewidth=1.8, zorder=4)
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            xs, ys = part.exterior.xy
            ax.plot([v / 1000 for v in xs], [v / 1000 for v in ys],
                    color="black", linewidth=1.8, zorder=4)

# Lodge markers
ax.scatter(x_km, y_km, c="steelblue", s=50, zorder=5,
           edgecolors="black", linewidths=0.5, label="Lodges / camps")
for j, loc in enumerate(location_cols):
    ax.annotate(loc, (x_km[j], y_km[j]), fontsize=6.5,
                color="black", xytext=(5, 4), textcoords="offset points", zorder=6)

ax.plot(0, 0, "+", color="black", markersize=14,
        markeredgewidth=2.5, zorder=6, label="Park centroid (0, 0)")

ax.grid(True, linestyle="--", alpha=0.15, color="black", zorder=0)
ax.set_xlabel("East from centroid (km)",  fontsize=11, color="black")
ax.set_ylabel("North from centroid (km)", fontsize=11, color="black")
ax.tick_params(colors="black", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("black")

ax.set_title(
    "Animal Sighting Value — Etosha National Park",
    fontsize=13, fontweight="bold", color="black", pad=16,
)

sm   = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
cbar.set_label("Poacher's Percieved Value ($)",
               fontsize=10, color="black")
cbar.ax.yaxis.set_tick_params(color="black")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

ax.legend(loc="upper right", fontsize=9,
          facecolor="white", edgecolor="black", labelcolor="black")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("etosha_animal_value_5km.png", dpi=160,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved → etosha_animal_value_5km.png")
plt.show()