"""
Etosha Security Validation — Coupled LV + Kriged Density + MILP
================================================================
 
Reads:
    *_5km.csv                       kriged species density maps (one per species)
    immc_-_species_final__3_.csv    LV species list
    immc_-_foodweb_final__5_.csv    food web
    immc_-_animal_maps.csv          species black-market prices
    etosha_path_results_30k.csv     MILP security placement output
 
Produces:
    etosha_rhino_populations.png
 
Model
-----
Poaching is incorporated as a continuous mortality term directly in the ODE:
 
    dN_i/dt = r_i * N_i * (1 - Pi/K_i)  -  mu_i * N_i
 
    Pi    = N_i + sum_j alpha_ij * N_j   (competitive LV pressure)
    mu_i  = gamma * price_weight_i * danger_weight_i
 
    danger_weight_i = sum_cells [ D_i(c) * R(c) ] / sum_cells [ D_i(c) ]
    D_i(c) = kriged sighting probability at cell c   (*_5km.csv)
    R(c)   = residual danger after MILP placement    (etosha_path_results_30k.csv)
 
Stable equilibrium: N*_i = K_i * (1 - mu_i / r_i)
  mu_i > r_i  ->  extinction  (no security)
  mu_i < r_i  ->  stable pop  (MILP security)
 
Constants and sources
---------------------
gamma = 0.12 yr^-1
    Historical poaching rate. Southern Africa lost ~90% of black rhino
    1970-1993. Source: NACSO Namibia Wildlife Report 2022;
    Emslie & Brooks (1999) African Rhino Status Survey, IUCN.
 
r_white_rhino = 0.069 yr^-1
    Source: Owen-Smith (1988) Megaherbivores. Cambridge University Press.
 
r_black_rhino = 0.054 yr^-1
    Source: Emslie & Brooks (1999) African Rhino Status Survey, IUCN.
 
price_weight_i = Price_i / max(Price)
    Source: Price column, immc_-_animal_maps.csv.
 
r_i = r0 * M_i^(-1/4), r0 = 0.8 yr^-1  (all other species)
    Source: Brown et al. (2004) Metabolic Theory of Ecology.
 
c_i = r_i / (n_plant_prey * K_P/2)
    Source: Tilman (1982) Resource Competition and Community Structure.
"""
 
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
 
 
# =============================================================================
# 1.  LOAD DATA
# =============================================================================
 
milp_df    = pd.read_csv('cvxopt_results/etosha_path_results_30k.csv')
species_df = pd.read_csv('immc - species_final (3).csv')
foodweb_df = pd.read_csv('immc - foodweb_final (5).csv')
maps_df    = pd.read_csv('immc - animal_maps (2).csv', index_col=0)
 
grid_n   = milp_df['northing_km'].values
grid_e   = milp_df['easting_km'].values
residual = milp_df['residual'].values   # danger remaining after MILP placement
full_dan = milp_df['danger'].values     # original danger with no security
 
print(f"MILP grid: {len(milp_df)} cells")
 
 
# =============================================================================
# 2.  KRIGED SPECIES DENSITY MAPS
# =============================================================================
 
SPECIES_FILES = {
    'Aardvark':       ('Aardvark_5km.csv',       'AARDVA'),
    'Black Rhino':    ('Black_Rhino_5km.csv',    'BRHINO'),
    'Brown Hyena':    ('Brown_Hyena_5km.csv',    'CROCRO'),
    'Cheetah':        ('Cheetah_5km.csv',        'ACIJUB'),
    'Eland':          ('Eland_5km.csv',          'TAUORY'),
    'Elephant':       ('Elephant_5km.csv',       'LOXAFR'),
    'Giraffe':        ('Giraffe_5km.csv',        'GIRCAM'),
    'Leopard':        ('Leopard_5km.csv',        'PANPAR'),
    'Lion':           ('Lion_5km.csv',           'PANLEO'),
    'Oryx':           ('Oryx_5km.csv',           'ORYXOR'),
    'Pangolin':       ('Pangolin_5km.csv',       'PANGOL'),
    'Roan Antelope':  ('Roan_Antelope_5km.csv',  'ANTELO'),
    'Sable Antelope': ('Sable_Antelope_5km.csv', 'ANTELO'),
    'Spotted Hyena':  ('Spotted_Hyena_5km.csv',  'CROCRO'),
    'White Rhino':    ('White_Rhino_5km.csv',    'WRHINO'),
    'Wildebeest':     ('Wildebeest_5km.csv',     'CONTAU'),
    'Zebra':          ('Zebra_5km.csv',          'EQUBUR'),
}
 
 
def load_density_on_grid(fpath, grid_n, grid_e):
    df   = pd.read_csv(fpath, index_col=0)
    rows = df.index.astype(float).values
    cols = df.columns.astype(float).values
    vals = df.values.astype(float)
    density = np.zeros(len(grid_n))
    for c in range(len(grid_n)):
        ri = np.argmin(np.abs(rows - grid_n[c]))
        ci = np.argmin(np.abs(cols - grid_e[c]))
        v  = vals[ri, ci]
        density[c] = 0.0 if np.isnan(v) else v
    return density
 
 
print("Loading kriged density maps...")
density_maps = {}
for name, (fname, _) in SPECIES_FILES.items():
    density_maps[name] = load_density_on_grid("animal_maps/"+ fname, grid_n, grid_e)
print(f"  Loaded {len(density_maps)} species")
 
 
# =============================================================================
# 3.  PRICES AND POACHING RATE
# =============================================================================
 
animal_rows = [r for r in maps_df.index
               if r.strip() not in ('Latitude ', 'Longitude ')]
prices    = maps_df.loc[animal_rows, 'Price'].astype(float)
max_price = prices.max()
 
price_weight = {}
for name in SPECIES_FILES:
    if name in prices.index:
        price_weight[name] = prices[name] / max_price
    else:
        matches = [r for r in prices.index
                   if name.lower() in r.lower() or r.lower() in name.lower()]
        price_weight[name] = prices[matches[0]] / max_price if matches else 0.1
 
# Baseline poaching mortality for most-targeted species (zero security coverage)
# Source: NACSO 2022; Emslie & Brooks 1999
GAMMA = 0.12   # yr^-1
 
 
# =============================================================================
# 4.  POACHING RATE PER LV SPECIES (mu_i)
# =============================================================================
 
def compute_mu(threat_per_cell, lv_idx, n_lv):
    """
    mu_i = gamma * price_weight_i * danger_weight_i
 
    danger_weight_i = sum_cells [ density_i(c) * threat(c) ]
                      / sum_cells [ density_i(c) ]
    """
    dw = {}
    for name, dens in density_maps.items():
        total = dens.sum()
        dw[name] = 0.0 if total < 1e-9 else (dens * threat_per_cell).sum() / total
 
    lv_danger = defaultdict(float)
    lv_price  = defaultdict(float)
    for name, (_, lv_code) in SPECIES_FILES.items():
        if lv_code in lv_idx:
            lv_danger[lv_code] = max(lv_danger[lv_code], dw[name])
            lv_price[lv_code]  = max(lv_price[lv_code],  price_weight[name])
 
    mu = np.zeros(n_lv)
    for lv_code, i in lv_idx.items():
        if lv_code in lv_danger:
            mu[i] = GAMMA * lv_price[lv_code] * lv_danger[lv_code]
    return mu
 
 
# =============================================================================
# 5.  LV MODEL SETUP
# =============================================================================
 
CODES  = list(species_df['code'])
NAMES  = list(species_df['species'])
TYPES  = list(species_df['type'])
MASSES = np.array(species_df['body_mass_kg'], dtype=float)
 
N0 = species_df['initial_pop'].fillna(0).to_numpy(dtype=float).copy()
N0[CODES.index('PLANTS')] = 800_000.0
N0[CODES.index('BUGS')]   = 400_000.0
# Start rhinos near K — reflects Etosha pre-poaching baseline
# Source: Emslie & Brooks (1999)
N0[CODES.index('WRHINO')] = 2400.0
N0[CODES.index('BRHINO')] = 1400.0
 
n_lv   = len(CODES)
lv_idx = {c: i for i, c in enumerate(CODES)}
 
pp = defaultdict(list)
for _, row in foodweb_df.iterrows():
    pp[row['pred_code']].append(row['prey_code'])
 
K_dict = {
    'PLANTS': 2_000_000, 'BUGS':   5_000_000,
    'AEPMEL':     8_000, 'ALCBUS':     1_000, 'CONTAU':    10_000,
    'EQUBUR':    25_000, 'GIRCAM':     5_000, 'LOXAFR':     5_000,
    'MADKIR':     8_000, 'PHAACT':     8_000, 'PROCAP':    30_000,
    'TAUORY':     5_000, 'ORYXOR':     8_000, 'BRHINO':     1_500,
    'WRHINO':     2_500, 'ANTELO':    25_000, 'PANGOL':    15_000,
    'AARDVA':    25_000, 'PANLEO':       800, 'PANPAR':       700,
    'CROCRO':       800, 'ACIJUB':       400, 'CARCAR':       300,
    'LEPSER':       600,
}
K_arr = np.array([K_dict.get(c, 5_000) for c in CODES], dtype=float)
 
IP = lv_idx['PLANTS']; IB = lv_idx['BUGS']
PRODUCER_IDX = {IP, IB}
 
# Allometric growth rates: r_i = r0 * M_i^(-1/4)  [Brown et al. 2004]
r0 = 0.8
r  = r0 * MASSES**(-0.25)
r[IP] = 3.5; r[IB] = 10.0
# Empirical rhino rates (allometric overestimates slow-reproducing megaherbivores)
r[lv_idx['WRHINO']] = 0.069   # Owen-Smith 1988
r[lv_idx['BRHINO']] = 0.054   # Emslie & Brooks 1999
 
# Alpha matrix from food web
C_SUP = 1.2; BENEFIT_FRAC = 0.30
alpha = np.zeros((n_lv, n_lv))
for i in range(n_lv): alpha[i, i] = 1.0
 
n_predators_on = defaultdict(int)
for pred, prey_list in pp.items():
    for prey in prey_list:
        if lv_idx[prey] not in PRODUCER_IDX:
            n_predators_on[lv_idx[prey]] += 1
 
for pred, prey_list in pp.items():
    i = lv_idx[pred]; n_prey = len(prey_list)
    animal_prey = [p for p in prey_list if lv_idx[p] not in PRODUCER_IDX]
    for prey in prey_list:
        j = lv_idx[prey]
        if j in PRODUCER_IDX: continue
        alpha[j, i] = (C_SUP / max(n_predators_on[j], 1)) / max(n_prey, 1)
    if animal_prey:
        total_N = sum(N0[lv_idx[p]] for p in animal_prey)
        for prey in animal_prey:
            j = lv_idx[prey]; w = N0[j] / max(total_N, 1.)
            alpha[i, j] = -w * BENEFIT_FRAC * K_arr[i] / max(N0[j], 1.)
 
# Herbivory rates [Tilman 1982]: c_i = r_i / (n_plant_prey * K_P/2)
herb_rate = {}
n_plant_prey_of = {
    lv_idx[pred]: sum(1 for pr in prey_list
                      if TYPES[lv_idx[pr]] == 'producer' or pr == 'BUGS')
    for pred, prey_list in pp.items()
}
for prod_code, prod_idx in [('PLANTS', IP), ('BUGS', IB)]:
    K_eq    = K_arr[prod_idx] / 2.0
    grazers = [(lv_idx[pred], prod_idx) for pred, prey_list in pp.items()
               for prey in prey_list if prey == prod_code]
    for i, j in grazers:
        n_pp = max(n_plant_prey_of.get(i, 1), 1)
        herb_rate[(i, j)] = r[i] / (n_pp * K_eq)
 
 
# =============================================================================
# 6.  ODE WITH CONTINUOUS POACHING
# =============================================================================
 
def make_ode(mu):
    """
    dN_i/dt = r_i * N_i * (1 - Pi/K_i)  -  mu_i * N_i
    """
    def ode(t, N):
        N  = np.maximum(N, 0.)
        dN = np.zeros(n_lv)
        for pi in PRODUCER_IDX:
            eaten  = sum(herb_rate.get((i, pi), 0.) * N[i] * N[pi]
                         for i in range(n_lv))
            dN[pi] = r[pi] * N[pi] * (1. - N[pi] / K_arr[pi]) - eaten
        for i in range(n_lv):
            if i in PRODUCER_IDX: continue
            pressure = N[i] + sum(alpha[i, j] * N[j] for j in range(n_lv)
                                  if j != i and alpha[i, j] != 0.)
            dN[i] = r[i] * N[i] * (1. - pressure / K_arr[i]) - mu[i] * N[i]
        return dN
    return ode
 
 
# =============================================================================
# 7.  RUN SIMULATIONS
# =============================================================================
 
T_SIM    = 30
T_MONTHS = T_SIM * 12
 
 
def run_simulation(threat_per_cell, label):
    mu  = compute_mu(threat_per_cell, lv_idx, n_lv)
    ode = make_ode(mu)
    sol = solve_ivp(
        ode, (0, T_MONTHS), N0.copy(),
        method='RK45',
        t_eval=np.linspace(0, T_MONTHS, T_MONTHS * 4),
        rtol=1e-6, atol=1e-8, max_step=0.5
    )
    Y = np.maximum(sol.y, 0.)
    print(f"  [{label}]  "
          f"White Rhino = {Y[lv_idx['WRHINO'], -1]:.0f}  "
          f"Black Rhino = {Y[lv_idx['BRHINO'], -1]:.0f}")
    return sol.t / 12., Y
 
 
print("\nRunning: no security (baseline)...")
t_none, Y_none = run_simulation(full_dan, "No security")
 
print("Running: MILP placement...")
t_milp, Y_milp = run_simulation(residual, "MILP security")
 
 
# =============================================================================
# 8.  RESULTS TABLE
# =============================================================================
 
print(f"\n{'Species':22s}  {'Year 0':>8s}  {'No Security':>12s}  "
      f"{'MILP':>8s}  {'Saved':>8s}")
print("-" * 68)
for name, code in [('White Rhino', 'WRHINO'), ('Black Rhino', 'BRHINO'),
                   ('Lion', 'PANLEO'), ('Cheetah', 'ACIJUB'),
                   ('Elephant', 'LOXAFR'), ('Zebra', 'EQUBUR')]:
    i    = lv_idx[code]
    n0   = Y_none[i, 0]
    none = Y_none[i, -1]
    milp = Y_milp[i, -1]
    diff = milp - none
    pct  = 100 * diff / max(none, 1)
    print(f"  {name:20s}  {n0:>8,.0f}  {none:>12,.0f}  "
          f"{milp:>8,.0f}  {diff:>+8.0f}  ({pct:+.1f}%)")
 
 
# =============================================================================
# 9.  PLOT
# =============================================================================
 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')
fig.suptitle(
    'Rhino Population Dynamics Over 30 Years\n'
    'With vs Without MILP Security Resource Allocation',
    fontsize=15, fontweight='bold', color='black'
)
 
for ax, code, name, c_sec, c_non in [
    (axes[0], 'WRHINO', 'White Rhino', '#1565C0', '#C62828'),
    (axes[1], 'BRHINO', 'Black Rhino', '#00838F', '#BF360C'),
]:
    i    = lv_idx[code]
    ysec = Y_milp[i]
    ynon = Y_none[i]
 
    ax.plot(t_milp, ysec, color=c_sec, lw=2.5, label='With MILP security')
    ax.plot(t_none, ynon, color=c_non, lw=2.5, ls='--', label='Without security')
    ax.fill_between(t_milp, ynon, ysec, where=(ysec >= ynon),
                    color=c_sec, alpha=0.15)
    ax.axhline(50, color='gray', lw=1, ls=':', label='Extinction threshold')
 
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Population', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_SIM)
    ax.set_ylim(0, K_arr[lv_idx[code]] * 1.1)
 
    ax.annotate(f'{ysec[-1]:.0f}', xy=(T_SIM, ysec[-1]),
                xytext=(-5, 5), textcoords='offset points',
                color=c_sec, fontsize=10, fontweight='bold', ha='right')
    ax.annotate(f'{ynon[-1]:.0f}', xy=(T_SIM, max(ynon[-1], 10)),
                xytext=(-5, -15), textcoords='offset points',
                color=c_non, fontsize=10, fontweight='bold', ha='right')
 
plt.tight_layout()
plt.savefig('etosha_rhino_populations.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("\nSaved: etosha_rhino_populations.png")