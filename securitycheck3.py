"""
Etosha — Black Rhino Population vs Protection Budget
=====================================================
Runs the coupled LV + poaching + fire ODE for each budget level,
loading the corresponding MILP results CSV, and plots all rhino
population trajectories on a single figure.
"""
 
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')
 
 
# =============================================================================
# CONFIG — add/remove budgets here
# =============================================================================
BUDGETS = [10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000]
RESULTS_DIR = 'cvxopt_results_fr'
RESULTS_TEMPLATE = RESULTS_DIR + '/etosha_path_results_{k}k.csv'
 
GAMMA_POACH = 0.18
GAMMA_FIRE  = 0.03
T_SIM       = 50
T_MONTHS    = T_SIM * 12
 
# Manually chosen colors — distinct, no yellow, red→blue/purple progression
BUDGET_COLORS = [
    '#d62728',   # $10k  — brick red
    '#e8711a',   # $20k  — burnt orange
    '#c8a800',   # $30k  — dark gold (visible on white)
    '#7db010',   # $40k  — olive green
    '#2ca02c',   # $50k  — mid green
    '#1a7bbf',   # $60k  — steel blue
    '#6a3d9a',   # $70k  — purple
]
 
# =============================================================================
# 1. LOAD STATIC DATA
# =============================================================================
species_df = pd.read_csv('immc - species_final (3).csv')
foodweb_df = pd.read_csv('immc - foodweb_final (5).csv')
maps_df    = pd.read_csv('immc - animal_maps (2).csv', index_col=0)
 
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
 
CODES  = list(species_df['code'])
NAMES  = list(species_df['species'])
TYPES  = list(species_df['type'])
MASSES = np.array(species_df['body_mass_kg'], dtype=float)
n_lv   = len(CODES)
lv_idx = {c: i for i, c in enumerate(CODES)}
 
N0 = species_df['initial_pop'].fillna(0).to_numpy(dtype=float).copy()
N0[CODES.index('PLANTS')] = 800_000.0
N0[CODES.index('BUGS')]   = 400_000.0
 
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
 
r0 = 0.8
r  = r0 * MASSES**(-0.25)
r[IP] = 3.5; r[IB] = 10.0
r[lv_idx['BRHINO']] = 0.054
 
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
 
# Prices
animal_rows = [row for row in maps_df.index
               if row.strip() not in ('Latitude ', 'Longitude ')]
prices    = maps_df.loc[animal_rows, 'Price'].astype(float)
max_price = prices.max()
price_weight = {}
for name in SPECIES_FILES:
    if name in prices.index:
        price_weight[name] = prices[name] / max_price
    else:
        matches = [r2 for r2 in prices.index
                   if name.lower() in r2.lower() or r2.lower() in name.lower()]
        price_weight[name] = prices[matches[0]] / max_price if matches else 0.1
 
 
# =============================================================================
# 2. DENSITY MAPS
# =============================================================================
 
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
 
 
# =============================================================================
# 3. MORTALITY + ODE
# =============================================================================
 
def compute_mu(density_maps, animal_threat, fire_threat):
    dw_animal = {}
    dw_fire   = {}
    for name, dens in density_maps.items():
        total = dens.sum()
        if total < 1e-9:
            dw_animal[name] = 0.0
            dw_fire[name]   = 0.0
        else:
            dw_animal[name] = (dens * animal_threat).sum() / total
            dw_fire[name]   = (dens * fire_threat).sum()   / total
 
    lv_dw_animal = defaultdict(float)
    lv_dw_fire   = defaultdict(float)
    lv_price     = defaultdict(float)
    for name, (_, lv_code) in SPECIES_FILES.items():
        if lv_code in lv_idx:
            lv_dw_animal[lv_code] = max(lv_dw_animal[lv_code], dw_animal[name])
            lv_dw_fire[lv_code]   = max(lv_dw_fire[lv_code],   dw_fire[name])
            lv_price[lv_code]     = max(lv_price[lv_code],      price_weight[name])
 
    mu = np.zeros(n_lv)
    for lv_code, i in lv_idx.items():
        if lv_code in lv_dw_animal:
            mu[i] = (GAMMA_POACH * lv_price[lv_code] * lv_dw_animal[lv_code]
                     + GAMMA_FIRE * lv_dw_fire[lv_code])
    return mu
 
 
def make_ode(mu):
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
 
 
def run_simulation(density_maps, animal_threat, fire_threat, label):
    mu  = compute_mu(density_maps, animal_threat, fire_threat)
    ode = make_ode(mu)
    ib  = lv_idx['BRHINO']
    print(f"  [{label}] Black rhino mu={mu[ib]:.4f}  r={r[ib]:.4f}  net={r[ib]-mu[ib]:+.4f}")
    sol = solve_ivp(
        ode, (0, T_MONTHS), N0.copy(),
        method='RK45',
        t_eval=np.linspace(0, T_MONTHS, T_MONTHS * 4),
        rtol=1e-6, atol=1e-8, max_step=0.5
    )
    Y = np.maximum(sol.y, 0.)
    print(f"  [{label}] Black Rhino yr{T_SIM} = {Y[ib, -1]:.0f}")
    return sol.t / 12., Y
 
 
# =============================================================================
# 4. RUN ONE SCENARIO PER BUDGET
# =============================================================================
 
results = {}
 
for budget in BUDGETS:
    k = budget // 1000
    fpath = RESULTS_TEMPLATE.format(k=k)
    print(f"\n── Budget ${budget:,} ({fpath}) ──")
 
    try:
        milp_df = pd.read_csv(fpath)
    except FileNotFoundError:
        print(f"  File not found — skipping.")
        continue
 
    grid_n = milp_df['northing_km'].values
    grid_e = milp_df['easting_km'].values
 
    density_maps = {}
    for name, (fname, _) in SPECIES_FILES.items():
        density_maps[name] = load_density_on_grid(
            "animal_maps/" + fname, grid_n, grid_e)
 
    danger   = milp_df['danger'].values
    fire_n   = milp_df['fire_norm'].values
    animal_n = milp_df['animal_norm'].values
    residual = milp_df['residual'].values
    safe_dan = np.where(danger > 0, danger, 1.0)
 
    residual_fire   = residual * fire_n   / safe_dan
    residual_animal = residual * animal_n / safe_dan
 
    label = f"${k}k/week"
    t, Y = run_simulation(density_maps, residual_animal, residual_fire, label)
    results[budget] = (t, Y[lv_idx['BRHINO']])
 
# No-security baseline
print("\n── No security baseline ──")
milp_df_ref = pd.read_csv(RESULTS_TEMPLATE.format(k=BUDGETS[-1] // 1000))
grid_n_ref  = milp_df_ref['northing_km'].values
grid_e_ref  = milp_df_ref['easting_km'].values
density_maps_ref = {}
for name, (fname, _) in SPECIES_FILES.items():
    density_maps_ref[name] = load_density_on_grid(
        "animal_maps/" + fname, grid_n_ref, grid_e_ref)
nosec_fire   = milp_df_ref['fire_norm'].values
nosec_animal = milp_df_ref['animal_norm'].values
t_none, Y_none = run_simulation(density_maps_ref, nosec_animal, nosec_fire, "No security")
rhino_none = Y_none[lv_idx['BRHINO']]
 
 
# =============================================================================
# 5. PLOT
# =============================================================================
 
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white')
 
for idx, (budget, (t, rhino)) in enumerate(sorted(results.items())):
    k     = budget // 1000
    color = BUDGET_COLORS[idx % len(BUDGET_COLORS)]
    ax.plot(t, rhino, color=color, lw=2.4,
            label=f'${k}k/week  →  {rhino[-1]:.0f} animals at yr{T_SIM}')
 
ax.plot(t_none, rhino_none, color='black', lw=2.0, ls='--',
        label=f'No security  →  {rhino_none[-1]:.0f} animals at yr{T_SIM}')
 
# ── Threshold lines ───────────────────────────────────────────────────────────
# Minimum viable population
ax.axhline(500, color="#7a7a7a", lw=1.4, ls=':', zorder=1)

# Functional extinction
ax.axhline(50, color='#888888', lw=1.2, ls=':', zorder=1)

 
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Black Rhino Population', fontsize=12)
ax.set_xlim(0, T_SIM)
ax.set_ylim(0)
ax.set_xticks(range(0, T_SIM + 1, 5))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='#cccccc')
 
fig.suptitle(
    'Black Rhino Population Stability vs. Protection Budget',
    fontsize=13, fontweight='bold', color='black'
)
 
plt.tight_layout()
plt.savefig('etosha_rhino_budget_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("\nSaved: etosha_rhino_budget_comparison.png")
 