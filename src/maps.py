"""
Static energy maps for rigid-cluster simulations.

Three maps are provided:

    translational_map  -- E(x, y) at fixed orientation theta
    rotational_map     -- E(theta) at fixed CM position
    rototrasl_map      -- E(theta, x, y) full 3-D scan

All maps return plain dicts of numpy arrays.  No file I/O, no logging,
no argparse.  The caller (io.py or user code) decides what to do with
the results.

Grid coordinate convention
--------------------------
Fractional coordinates (dda1, dda2) in [0, 1) x [0, 1) are converted to
real-space CM positions via:

    pos_cm = u_inv @ np.array([dda1, dda2])

where u_inv is the 2x2 matrix from calc_matrices_bvect (columns are the
primitive vectors).  For sinusoidal substrates without an explicit unit
cell, pass np.eye(2) to obtain a Cartesian grid.

Parallelism
-----------
Set n_jobs > 1 to parallelise over grid points with joblib (loky backend).
n_jobs = -1 uses all available cores.  For small grids, n_jobs = 1 is
faster due to process-spawn overhead.
"""

import numpy as np
from tool_create_cluster import rotate


def _eval_point(args):
    """Single grid-point evaluation.  Unpacked by joblib workers."""
    pos_rot, pos_cm, calc_en_f, en_params = args
    e, f, tau = calc_en_f(pos_rot + pos_cm, pos_cm, *en_params)
    return e, f[0], f[1], tau


# ============================================================
# Translational map
# ============================================================

def translational_map(pos, calc_en_f, en_params, u_inv,
                      n_x, n_y,
                      frac_x=(0., 1.), frac_y=(0., 1.),
                      n_jobs=1):
    """Energy landscape as a function of CM position at fixed orientation.

    The cluster orientation is taken as-is from pos (caller rotates first
    if a non-zero theta is desired).

    Args:
        pos:       (N, 2) ndarray  -- cluster positions in cluster frame.
        calc_en_f: callable        -- total energy function from substrate_from_params.
        en_params: list            -- extra arguments for calc_en_f.
        u_inv:     (2, 2) ndarray  -- metric matrix; use np.eye(2) for sin substrates.
        n_x:       int             -- grid points along first fractional axis.
        n_y:       int             -- grid points along second fractional axis.
        frac_x:    (float, float)  -- fractional range along a1 (default 0 to 1).
        frac_y:    (float, float)  -- fractional range along a2 (default 0 to 1).
        n_jobs:    int             -- joblib parallel workers (1 = serial).

    Returns:
        dict with keys:
            'pos_cm'  : (n_x*n_y, 2) -- CM positions in real space.
            'energy'  : (n_x*n_y,)
            'force'   : (n_x*n_y, 2)
            'torque'  : (n_x*n_y,)
    """
    u_inv = np.asarray(u_inv, dtype=np.float64)
    pos   = np.asarray(pos,   dtype=np.float64)

    da1_vals = np.linspace(frac_x[0], frac_x[1], n_x, endpoint=True)
    da2_vals = np.linspace(frac_y[0], frac_y[1], n_y, endpoint=True)

    grid_pos_cm = np.array([
        u_inv @ np.array([da1, da2])
        for da1 in da1_vals
        for da2 in da2_vals
    ])

    tasks = [(pos, cm, calc_en_f, en_params) for cm in grid_pos_cm]
    results = _run_tasks(tasks, n_jobs)

    energy = np.array([r[0] for r in results])
    force  = np.array([[r[1], r[2]] for r in results])
    torque = np.array([r[3] for r in results])

    return {'pos_cm': grid_pos_cm, 'energy': energy,
            'force': force, 'torque': torque}


# ============================================================
# Rotational map
# ============================================================

def rotational_map(pos, calc_en_f, en_params,
                   theta_deg, pos_cm=(0., 0.),
                   n_jobs=1):
    """Energy as a function of cluster orientation at fixed CM position.

    Args:
        pos:       (N, 2) ndarray       -- cluster positions at theta=0.
        calc_en_f: callable             -- total energy function.
        en_params: list                 -- extra arguments for calc_en_f.
        theta_deg: (n_theta,) array     -- angles in degrees.
        pos_cm:    (2,) array-like      -- fixed CM position (default origin).
        n_jobs:    int                  -- parallel workers.

    Returns:
        dict with keys:
            'theta'  : (n_theta,)    -- angles in degrees (copy of input).
            'energy' : (n_theta,)
            'force'  : (n_theta, 2)
            'torque' : (n_theta,)
    """
    pos     = np.asarray(pos,   dtype=np.float64)
    pos_cm  = np.asarray(pos_cm, dtype=np.float64)
    theta_deg = np.asarray(theta_deg, dtype=np.float64)

    # Pre-rotate cluster for every angle before launching tasks.
    rotated = [rotate(pos, th) for th in theta_deg]
    tasks   = [(r, pos_cm, calc_en_f, en_params) for r in rotated]
    results = _run_tasks(tasks, n_jobs)

    energy = np.array([r[0] for r in results])
    force  = np.array([[r[1], r[2]] for r in results])
    torque = np.array([r[3] for r in results])

    return {'theta': theta_deg.copy(), 'energy': energy,
            'force': force, 'torque': torque}


# ============================================================
# Roto-translational map
# ============================================================

def rototrasl_map(pos, calc_en_f, en_params, u_inv,
                  theta_deg, n_x, n_y,
                  frac_x=(0., 1.), frac_y=(0., 1.),
                  n_jobs=1):
    """Full energy landscape over (theta, x, y).

    For each angle, scans the same translational grid. The parallelism
    target is the flattened (theta, grid) index.

    Args:
        pos:       (N, 2) ndarray       -- cluster positions at theta=0.
        calc_en_f: callable             -- total energy function.
        en_params: list                 -- extra arguments for calc_en_f.
        u_inv:     (2, 2) ndarray       -- metric matrix.
        theta_deg: (n_theta,) array     -- angles in degrees.
        n_x:       int                  -- grid points along a1.
        n_y:       int                  -- grid points along a2.
        frac_x:    (float, float)       -- fractional range along a1.
        frac_y:    (float, float)       -- fractional range along a2.
        n_jobs:    int                  -- parallel workers.

    Returns:
        dict with keys:
            'theta'  : (n_theta,)
            'pos_cm' : (n_theta, n_x*n_y, 2)
            'energy' : (n_theta, n_x*n_y)
            'force'  : (n_theta, n_x*n_y, 2)
            'torque' : (n_theta, n_x*n_y)
    """
    u_inv     = np.asarray(u_inv,     dtype=np.float64)
    pos       = np.asarray(pos,       dtype=np.float64)
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    n_theta   = len(theta_deg)
    n_grid    = n_x * n_y

    da1_vals = np.linspace(frac_x[0], frac_x[1], n_x, endpoint=True)
    da2_vals = np.linspace(frac_y[0], frac_y[1], n_y, endpoint=True)
    grid_pos_cm = np.array([
        u_inv @ np.array([da1, da2])
        for da1 in da1_vals
        for da2 in da2_vals
    ])

    # Build all tasks: outer index = theta, inner index = grid point.
    tasks = []
    for th in theta_deg:
        pos_rot = rotate(pos, th)
        for cm in grid_pos_cm:
            tasks.append((pos_rot, cm, calc_en_f, en_params))

    results = _run_tasks(tasks, n_jobs)

    energy = np.array([r[0] for r in results]).reshape(n_theta, n_grid)
    force  = np.array([[r[1], r[2]] for r in results]).reshape(n_theta, n_grid, 2)
    torque = np.array([r[3] for r in results]).reshape(n_theta, n_grid)
    pos_cm_out = np.tile(grid_pos_cm, (n_theta, 1, 1))

    return {'theta': theta_deg.copy(), 'pos_cm': pos_cm_out,
            'energy': energy, 'force': force, 'torque': torque}


# ============================================================
# Internal helpers
# ============================================================

def _run_tasks(tasks, n_jobs):
    """Run _eval_point on tasks, serially or in parallel."""
    if n_jobs == 1:
        return [_eval_point(t) for t in tasks]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError(
            "joblib is required for n_jobs != 1. Install with: pip install joblib"
        )
    return Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_eval_point)(t) for t in tasks
    )
