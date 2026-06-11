#!/usr/bin/env python3
"""
Generic Euler-Maruyama integrator for Ito SDEs.

Convention
----------
The SDE is written in Ito form:

    dX = drift(X, t) dt + diffusion(X, t) dW

where dW = sqrt(dt) * xi, xi ~ N(0,1)^noise_dim.  In discrete time the
single-step update is:

    X_{n+1} = X_n + drift(X_n, t_n) * dt + diffusion(X_n, t_n) @ (sqrt_dt * xi)

drift and diffusion return velocity-like quantities; diffusion is a
(state_dim, noise_dim) matrix.  The diagonal entry for component i is:

    diffusion[i, i] = sqrt(2 * D_i)

so that  diffusion[i,i]^2 * dt = 2 * D_i * dt  matches <dX_i^2> = 2*D_i*dt.

For overdamped Langevin with D = kBT/eta:  diffusion[i, i] = sqrt(2*kBT/eta).

Note on the manual run_md loop
--------------------------------
src/dynamics.py cannot use these functions directly because the substrate
force must be recomputed at every step from Python.  The run_md loop follows
the equivalent convention:

    delta = (drift + diffusion_v2 @ xi) * dt

where diffusion_v2[i,i] = sqrt(2*kBT/(eta*dt)).  Both conventions give
identical MSD per step; they differ only in how sqrt_dt is absorbed.

Constraints on drift_f and diffusion_f
---------------------------------------
Both must be @njit.  Signature:

    func(x, t, params, out) -> None

where out is a pre-zeroed array filled in-place.
"""

import numpy as np
from numba import njit


# ============================================================
# Generic Euler-Maruyama Integrator
# ============================================================
 
@njit
def euler_maruyama_traj(
    drift,
    diffusion,
    x0,
    t0,
    dt,
    nsteps,
    params,
    noise_dim,
):
    """Return full trajectory"""
    
    dim = x0.shape[0]
    x = x0.copy()
    t = t0
 
    sqrt_dt = np.sqrt(dt)
 
    drift_buf = np.zeros(dim)
    diff_buf = np.zeros((dim, noise_dim))
 
    traj = np.zeros((nsteps+1, dim))
    traj[0] = x
 
    for i in range(nsteps):
 
        drift(x, t, params, drift_buf)
        diffusion(x, t, params, diff_buf)
 
        dW = np.random.normal(0.0, 1.0, noise_dim)
 
        x += drift_buf*dt + diff_buf @ (sqrt_dt*dW)
 
        t += dt
        traj[i+1] = x
 
    return traj
 
@njit
def euler_maruyama_final(
    drift,
    diffusion,
    x0,
    t0,
    dt,
    nsteps,
    params,
    noise_dim,
):
    """Return only final point"""
    dim = x0.shape[0]
    x = x0.copy()
    t = t0
 
    sqrt_dt = np.sqrt(dt)
 
    drift_buf = np.zeros(dim)
    diff_buf = np.zeros((dim, noise_dim))
 
    for _ in range(nsteps):
 
        drift(x, t, params, drift_buf)
        diffusion(x, t, params, diff_buf)
 
        dW = np.random.normal(0.0, 1.0, noise_dim)
 
        x += drift_buf*dt + diff_buf @ (sqrt_dt*dW)
 
        t += dt
 
    return x
