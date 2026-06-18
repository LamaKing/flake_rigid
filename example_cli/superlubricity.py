"""
Superlubricity demo: commensurate vs incommensurate depinning.

Run from example_cli/:
    python superlubricity.py

Tune parameters below to explore.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from flake.substrate import substrate_from_params, get_ks
from flake.cluster import make_cluster, rotate
from flake.sweep import sweep_md, drift_velocity

# ── parameters to tune ──────────────────────────────────────────────────────
N1, N2      = 4, 4          # cluster grid size (N=19 for circle)
SHAPE       = 'circle'
THETA_COMM  = 0.0           # commensurate rotation (deg)
THETA_INC   = 30.0          # incommensurate rotation (deg) -- try 1.5, 5, 15, 30

EPSILON     = 1.0           # substrate corrugation
N_F         = 20            # number of force values in sweep
N_STEPS     = 200_000       # steps per run (200 time units at dt=1e-3)
KBT         = 0.0           # 0 = deterministic; try 1e-3 for thermal noise
N_JOBS      = -1             # parallel workers
# ────────────────────────────────────────────────────────────────────────────

A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, -np.sqrt(3) / 2.0])

sub_params = {
    'well_shape': 'sin',
    'epsilon':    EPSILON,
    'sub_basis':  [[0., 0.]],
    'ks':         get_ks(1.0, 3, 4.0 / 3.0, 0.0).tolist(),
}
_, en_func, _ = substrate_from_params(sub_params)

pos_comm  = make_cluster(A1, A2, N1, N2, shape=SHAPE)
pos_incomm = rotate(pos_comm, THETA_INC)
N = len(pos_comm)

F1s_analytic = 2.793  # max |dV/dx| for epsilon=1, spacing=1
Fc_comm_est  = N * F1s_analytic

print(f"Cluster: N={N}, shape={SHAPE}({N1},{N2})")
print(f"Estimated Fc (commensurate) = {Fc_comm_est:.1f}  (N x F1s)")
print(f"Commensurate rotation: {THETA_COMM} deg")
print(f"Incommensurate rotation: {THETA_INC} deg")
print()

md_kwargs = dict(eta=1.0, kBT=KBT, dt=1e-3, n_steps=N_STEPS, print_every=N_STEPS // 2)
post_fn   = drift_velocity()

# sweep from 0 to 1.2 * Fc_est for commensurate, 0 to 0.5 * Fc_est for incommensurate
F_comm   = np.linspace(0.0, 1.2 * Fc_comm_est, N_F)
F_incomm = np.linspace(0.0, 0.5 * Fc_comm_est, N_F)

print("Running commensurate sweep ...")
res_comm = sweep_md(pos_comm, en_func,
                    [{'Fx': float(f)} for f in F_comm],
                    base_md_kwargs=md_kwargs, post_fn=post_fn,
                    n_jobs=N_JOBS, save=False, verbose=False)

print("Running incommensurate sweep ...")
res_incomm = sweep_md(pos_incomm, en_func,
                      [{'Fx': float(f)} for f in F_incomm],
                      base_md_kwargs=md_kwargs, post_fn=post_fn,
                      n_jobs=N_JOBS, save=False, verbose=False)

vx_comm   = np.array([r['result'][0] for r in res_comm])
vx_incomm = np.array([r['result'][0] for r in res_incomm])

# find depinning brackets
def depinning_F(F_vals, vx, threshold=0.05):
    sliding = vx > threshold
    if not sliding.any():
        return None, None
    return float(F_vals[~sliding].max()) if (~sliding).any() else 0.0, \
           float(F_vals[sliding].min())

Fc_lo, Fc_hi   = depinning_F(F_comm, vx_comm)
Fs_lo, Fs_hi   = depinning_F(F_incomm, vx_incomm)

print()
print("── commensurate ────────────────────────────────")
print(f"  F values:  {np.round(F_comm, 1)}")
print(f"  vx:        {np.round(vx_comm, 3)}")
if Fc_lo is not None:
    print(f"  depinning: Fc in [{Fc_lo:.2f}, {Fc_hi:.2f}]  (analytic N*F1s = {Fc_comm_est:.2f})")
else:
    print("  no depinning found -- widen F range")

print()
print("── incommensurate ──────────────────────────────")
print(f"  F values:  {np.round(F_incomm, 1)}")
print(f"  vx:        {np.round(vx_incomm, 3)}")
if Fs_lo is not None:
    ratio = Fc_hi / Fs_hi if Fs_hi else float('inf')
    print(f"  depinning: Fs in [{Fs_lo:.2f}, {Fs_hi:.2f}]")
    print(f"  Fc / Fs ~ {ratio:.1f}x  friction reduction")
else:
    print("  no depinning found in scanned range -- widen F range")
print()

# ── plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(F_comm,   vx_comm,   'o-', color='steelblue',  label=f'commensurate  θ={THETA_COMM}°')
ax.plot(F_incomm, vx_incomm, 's-', color='tomato',     label=f'incommensurate θ={THETA_INC}°')

if Fc_hi:
    ax.axvline(Fc_hi, color='steelblue', ls='--', lw=0.8, label=f'Fc≈{Fc_hi:.1f}')
if Fs_hi:
    ax.axvline(Fs_hi, color='tomato',    ls='--', lw=0.8, label=f'Fs≈{Fs_hi:.1f}')

ax.set_xlabel('Applied force $F_x$')
ax.set_ylabel('Drift velocity $v_x$')
ax.set_title(f'Superlubricity  N={N}  ε={EPSILON}  kBT={KBT}')
ax.legend()
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('superlubricity.png', dpi=150)
print("\nPlot saved to superlubricity.png")
plt.show()
