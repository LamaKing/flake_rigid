#!/usr/bin/env python3
"""
MD loop profiling: measure where time goes per step.

Run from src/ directory:
    python3 profile_md.py

Measures three things:
1. Time for calc_en_f alone (Python closure -> JIT core)
2. Time for rotate alone
3. Time for the full step (overhead = total - calc_en_f - rotate)

Then estimates full-JIT speedup if both became @njit.
"""

import numpy as np
from numpy import sqrt, pi
import time

from flake.substrate import substrate_from_params, get_ks
from flake.cluster import cluster_from_params, rotate

# ---- System: commensurate triangular, N=85 ----
ks = get_ks(1, 3, 4./3., 0.)
params = {
    'sub_basis': [[0, 0]], 'epsilon': 1.0, 'well_shape': 'sin', 'ks': ks,
    'a1': [1.0, 0.0], 'a2': [0.5, -sqrt(3.)/2.],
    'cl_basis': [[0, 0]], 'cluster_shape': 'circle', 'N1': 20, 'N2': 20,
}
_, en_func, _ = substrate_from_params(params)
pos = cluster_from_params(params)
N = pos.shape[0]
print('N = %d' % N)

pos_cm = np.zeros(2, dtype=np.float64)
theta  = 0.0
pos_rot = rotate(pos, theta)

# Warm up JIT
en_func(pos_rot + pos_cm, pos_cm)
rotate(pos, 1.0)

# ---- Benchmark calc_en_f ----
n_calls = 20000
t0 = time.perf_counter()
for _ in range(n_calls):
    e, f, tau = en_func(pos_rot + pos_cm, pos_cm)
t_en = (time.perf_counter() - t0) / n_calls * 1e6
print('calc_en_f:  %.2f us/call' % t_en)

# ---- Benchmark rotate ----
t0 = time.perf_counter()
for _ in range(n_calls):
    pr = rotate(pos, theta)
t_rot = (time.perf_counter() - t0) / n_calls * 1e6
print('rotate:     %.2f us/call' % t_rot)

# ---- Benchmark array ops in step (delta, x+=, etc.) ----
x     = np.zeros(3, dtype=np.float64)
delta = np.zeros(3, dtype=np.float64)
d_buf = np.zeros(3, dtype=np.float64)
B_buf = np.zeros((3, 3), dtype=np.float64)
rng   = np.random.default_rng(0)

t0 = time.perf_counter()
for _ in range(n_calls):
    xi    = rng.standard_normal(3)
    delta = (d_buf + B_buf @ xi) * 1e-3
    x    += delta
t_misc = (time.perf_counter() - t0) / n_calls * 1e6
print('step misc:  %.2f us/call (rng + matmul + add)' % t_misc)

# ---- Full step (simulate what run_md does) ----
x   = np.array([0., 0., 0.], dtype=np.float64)
t   = 0.
dt  = 1e-3
eta = 1.0
from flake.cluster import calc_cluster_langevin
eta_t, eta_r = calc_cluster_langevin(eta, pos)

t0 = time.perf_counter()
for _ in range(n_calls):
    pos_cm_   = x[:2]
    theta_deg = np.rad2deg(x[2])
    pos_rot_  = rotate(pos, theta_deg)
    e0, f0, tau0 = en_func(pos_rot_ + pos_cm_, pos_cm_)
    # drift
    vx = (f0[0]) / eta_t
    vy = (f0[1]) / eta_t
    vr = float(tau0) / eta_r
    xi = rng.standard_normal(3)
    x += np.array([vx, vy, vr]) * dt
    t += dt
t_step = (time.perf_counter() - t0) / n_calls * 1e6
print('full step:  %.2f us/step' % t_step)

# ---- Summary ----
print()
print('Breakdown:')
print('  calc_en_f : %.2f us  (%.0f%%)' % (t_en,   100*t_en/t_step))
print('  rotate    : %.2f us  (%.0f%%)' % (t_rot,  100*t_rot/t_step))
print('  misc      : %.2f us  (%.0f%%)' % (t_misc, 100*t_misc/t_step))
print('  overhead  : %.2f us  (%.0f%%)' % (
    t_step - t_en - t_rot - t_misc,
    100*(t_step - t_en - t_rot - t_misc)/t_step))

# ---- Estimate full-JIT speedup ----
# If calc_en_f and rotate become @njit and are called from a @njit loop:
# - JIT call overhead: ~0.1-0.5 us instead of Python call overhead
# - rotate as @njit: should be < 0.5 us (vs current)
# - calc_en_f core is already @njit; what we save is the Python wrapper cost
#   (np.asarray conversions etc.) = roughly 2-5 us
t_jit_estimate = 0.3 + 0.3 + t_misc  # rotate + en call + misc, all in JIT
print()
print('Estimated full-JIT step: %.2f us' % t_jit_estimate)
print('Estimated speedup: %.1fx' % (t_step / t_jit_estimate))
print()
print('For 200k steps, N=%d:' % N)
print('  Current:  %.1f s' % (t_step * 200000 / 1e6))
print('  Full JIT: %.1f s' % (t_jit_estimate * 200000 / 1e6))
