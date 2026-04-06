"""
Plot Drone Patrol Paths — Etosha National Park
===============================================
Reads drone_paths_filtered.csv and the two risk CSVs, then plots all
paths overlaid on the danger map.

Usage
-----
  python plot_drone_paths.py \
      --filtered drone_paths_filtered.csv \
      --fire     fire_risk_5km.csv \
      --animal   animal_value_5km.csv \
      --out      drone_paths_plot.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# ── Helpers (mirror generator) ────────────────────────────────────────────────

def load_grid(path):
    df = pd.read_csv(path, index_col=0)
    df.index   = df.index.astype(float)
    df.columns = df.columns.astype(float)
    return df

def reindex_nearest(df_src, ref_index, ref_columns, tolerance_km=3.0):
    src_rows = df_src.index.values
    src_cols = df_src.columns.values
    def snap(ref_vals, src_vals):
        out = []
        for v in ref_vals:
            d = np.abs(src_vals - v)
            i = d.argmin()
            out.append(src_vals[i] if d[i] <= tolerance_km else np.nan)
        return out
    sr = snap(ref_index,   src_rows)
    sc = snap(ref_columns, src_cols)
    vsr = [r for r in sr if not np.isnan(r)]
    vsc = [c for c in sc if not np.isnan(c)]
    res = df_src.reindex(index=vsr, columns=vsc)
    res.index   = [ref_index[i]   for i,r in enumerate(sr) if not np.isnan(r)]
    res.columns = [ref_columns[i] for i,c in enumerate(sc) if not np.isnan(c)]
    return res.reindex(index=ref_index, columns=ref_columns)

def build_danger(fire_csv, animal_csv):
    fire_df   = load_grid(fire_csv)
    animal_df = load_grid(animal_csv)
    ref_index, ref_columns = fire_df.index.values, fire_df.columns.values
    animal_aligned = reindex_nearest(animal_df, ref_index, ref_columns)
    fire_arr   = fire_df.values.astype(float)
    animal_arr = animal_aligned.values.astype(float)
    def norm(a):
        mn, mx = np.nanmin(a), np.nanmax(a)
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)
    in_park = ~(np.isnan(fire_arr) | np.isnan(animal_arr))
    fire_norm   = norm(fire_arr)
    fire_norm   = 1.0 - fire_norm        # FIX: low NDVI → high fire risk
    animal_norm = norm(animal_arr)
    danger = fire_norm + animal_norm
    danger[~in_park] = np.nan            # keep NaN outside park for display
    return danger, in_park, ref_index, ref_columns

def str_to_path(s):
    return [tuple(int(x) for x in p.split(",")) for p in s.split(";")]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--filtered", default="drone_paths_by_danger.csv")
    #parser.add_argument("--filtered", default="human_paths_filtered.csv")
    parser.add_argument("--filtered", default="drone_paths_filtered_1.5.csv")
    parser.add_argument("--fire",     default="fire_risk_5km.csv")
    parser.add_argument("--animal",   default="animal_value_5km.csv")
    parser.add_argument("--out",      default="drone_PATHS_plot.png")
    #parser.add_argument("--out",      default="human_PATHS_plot.png")
    args = parser.parse_args()

    # Load danger map
    danger, in_park, rows_km, cols_km = build_danger(args.fire, args.animal)

    # Build display grid (NaN outside park)
    display = danger.copy().astype(float)
    display[~in_park] = np.nan

    # Meshgrid for pcolormesh
    XI, YI = np.meshgrid(cols_km, rows_km)

    # Load filtered paths
    df = pd.read_csv(args.filtered)
    n_paths = len(df)
    print(f"Loaded {n_paths} paths")

    # Colour paths by rank (best = brightest)
    path_cmap  = LinearSegmentedColormap.from_list('blue_green', ['#1a0a6b', '#00aaff', '#00ff99'])
    path_norm  = mcolors.Normalize(vmin=0, vmax=n_paths - 1)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 9), facecolor="black")
    plt.subplots_adjust(wspace=0.25)

    def style_ax(ax, title):
        ax.set_facecolor("white")
        ax.grid(True, linestyle="--", alpha=0.08, color="black", zorder=0)
        ax.set_xlabel("East of Centroid",  fontsize=9,  color="black")
        ax.set_ylabel("North of Centroid", fontsize=9,  color="black")
        ax.tick_params(colors="#888888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
        ax.set_aspect("equal")
        ax.set_title(title, color="black", fontsize=12, fontweight="bold", pad=10)

    danger_norm = mcolors.Normalize(vmin=0, vmax=np.nanmax(danger))

    # ── Panel A: danger map only ──────────────────────────────────────────────
    ax0 = axes[0]
    style_ax(ax0, "Danger Map")
    ax0.pcolormesh(XI, YI, display,
                   cmap="YlOrRd", norm=danger_norm,
                   shading="nearest", zorder=1)

    # ── Panel B: paths overlaid on danger ─────────────────────────────────────
    ax1 = axes[1]
    style_ax(ax1, f"Top {n_paths} Human Patrol Loops")
    ax1.pcolormesh(XI, YI, display,
                   cmap="YlOrRd", norm=danger_norm,
                   shading="nearest", alpha=0.4, zorder=1)

    for rank, row in df.iterrows():
        path  = str_to_path(row["path_str"])
        loop  = path + [path[0]]          # close the loop visually
        xs    = [cols_km[c] for _, c in loop]
        ys    = [rows_km[r] for r, _ in loop]
        color = path_cmap(1.0 - path_norm(rank))   # rank 0 = green (best)
        alpha = max(0.15, 1.0 - rank / n_paths)    # top paths more opaque
        lw    = max(0.4,  1.5  - rank / n_paths)

        ax1.plot(xs, ys, color=color, alpha=alpha, linewidth=lw, zorder=3)

        # Mark start cell with a small dot for top 20
        if rank < 20:
            ax1.scatter(cols_km[path[0][1]], rows_km[path[0][0]],
                        color=color, s=18, zorder=5,
                        edgecolors="white", linewidths=0.4)

    # Colourbar for paths
    sm = cm.ScalarMappable(cmap=path_cmap.reversed(),
                           norm=mcolors.Normalize(vmin=1, vmax=n_paths))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("Danger Removed Rank (1 is the best!)", color="black", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="black", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

    # Stats annotation
    top = df.iloc[0]
    note = (f"Best path:  {top['n_cells']} cells, "
            f"{top['path_km']:.1f} km,  "
            f"danger removed = {top['danger_removed']:.2f}")
    fig.text(0.5, 0.01, note, ha="center", color="black", fontsize=8)

    plt.savefig(args.out, dpi=160, bbox_inches="tight",
                facecolor="white")
    print(f"Saved -> {args.out}")
    plt.show()


if __name__ == "__main__":
    main()