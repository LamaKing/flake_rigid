#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
import os
import sys
import warnings
import tempfile

import numpy as np
from numpy import sqrt

from flake.substrate import substrate_from_params, get_ks
from flake.cluster import cluster_from_params, calc_cluster_langevin
from flake.sweep import (
    grid_sweep, line_sweep, force_sweep, concat_sweeps,
    last_state, mean_velocity, drift_velocity,
    sweep_md, load_sweep,
)


_KS = get_ks(1.0, 3, 4.0 / 3.0, 0.0)

_PARAMS = {
    'sub_basis':     [[0, 0]],
    'epsilon':       1.0,
    'well_shape':    'sin',
    'ks':            _KS,
    'a1':            np.array([1.0, 0.0]),
    'a2':            np.array([0.5, -sqrt(3.0) / 2.0]),
    'cl_basis':      [[0, 0]],
    'cluster_shape': 'circle',
    'N1': 15, 'N2': 15,
    'theta': 0.0, 'pos_cm': [0, 0],
}


cluster = cluster_from_params(_PARAMS)


"""loky n_jobs=2 must be faster than an explicit for-loop over run_md.

The serial baseline is a plain Python loop calling run_md directly,
exactly as a user would write it by hand.  This is the fairest comparison:
it avoids sweep_md dispatch overhead and matches the pattern the user
already has in their notebook.

Parameters match the depinning notebook: n_steps=200000, dt=5e-4,
kBT=1e-5.  Four force values are used so the test takes ~70 s total.

Run explicitly with:
    pytest tests/test_sweep_md.py -m slow -v
"""
from time import time
from flake.dynamics import run_md as _run_md

_, en_func, _ = substrate_from_params(_PARAMS)
eta      = 1.0
kBT      = 1e-5
dt       = 5e-4
n_steps  = 200000
pos_cm0  = np.array([0.0, 0.0])
F_values = [0, 100, 250., 255., 260., 265., 300, 500]

# --- serial baseline: explicit loop, no sweep_md ---
t0 = time()
results_loop = []
for Fx in F_values:
    traj = _run_md(
        cluster, en_func,
        eta=eta, Fx=Fx, kBT=kBT,
        dt=dt, n_steps=n_steps,
        theta0=0.0, pos_cm0=pos_cm0.copy(),
        print_every=500,
    )
    x_final     = traj['pos_cm'][-1, 0] - pos_cm0[0]
    theta_final = traj['theta'][-1]
    slid        = x_final > 1.0
    v_avg       = x_final / traj['t'][-1]
    results_loop.append({
        'Fx': Fx, 'x_final': x_final,
        'theta_final': theta_final, 'v_avg': v_avg, 'slid': slid,
    })
    print("Fx=%.4f  x_final=%.3f  v_avg=%.3f  slid=%s"
          % (Fx, x_final, v_avg, slid))
t_loop = time() - t0

# --- parallel: sweep_md with loky n_jobs=2 ---
spec = [{'Fx': float(f)} for f in F_values]
base = {'eta': eta, 'kBT': kBT, 'dt': dt,
        'n_steps': n_steps, 'print_every': 500,
        'pos_cm0': pos_cm0.copy()}
t0     = time()
r_loky = sweep_md(cluster, en_func, spec,
                      base_md_kwargs=base,
                      post_fn=drift_velocity(),
                      n_jobs=4, backend='loky',
                      save=False, verbose=True)
t_loky = time() - t0

speedup = t_loop /t_loky
print("\nexplicit loop %.1fs  |  loky n_jobs=4 %.1fs  |  speedup %.2fx"
      % (t_loop, t_loky, speedup), file=sys.stderr, flush=True)


