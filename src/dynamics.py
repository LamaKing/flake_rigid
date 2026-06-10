"""
Overdamped Langevin dynamics for a rigid cluster on a substrate.

Physics
-------
Equations of motion:

    eta_t * dx/dt     = Fx_sub(x, y, theta) + Fx_ext + xi_x(t)
    eta_t * dy/dt     = Fy_sub(x, y, theta) + Fy_ext + xi_y(t)
    eta_r * dtheta/dt = tau_sub(x, y, theta) + tau_ext + xi_r(t)

where  eta_t = eta * N  and  eta_r = eta * sum_i r_i^2.

Fluctuation-dissipation theorem:
    <xi_i(t)^2> = 2 * kBT * eta_i  (variance per unit time)

Integration
-----------
Euler-Maruyama step for all kBT >= 0:

    delta_i = (F_total_i / eta_i) * dt
              + sqrt(2 * kBT / (eta_i * dt)) * xi_i * dt

where xi_i ~ N(0, 1).  The displacement variance is

    <delta_i^2> = 2 * kBT / eta_i * dt = 2 * D_i * dt   (correct FDT)

The factor sqrt(1/dt) in the noise amplitude compensates for the dt^2 that
comes from squaring the full step: without it the variance would scale as
dt^2 rather than dt.

At kBT=0 the noise term vanishes and this reduces to explicit
Euler gradient descent.  The step size dt is fixed; use small dt
(check convergence by halving it).  Avoid kBT=0 exactly: a
deterministic trajectory at a saddle point is sensitive to
floating-point rounding.  Use kBT=1e-8 (for epsilon=1) instead.

Public API
----------
    run_md             -- Euler-Maruyama integrator for all kBT >= 0.
    make_params_array  -- convenience constructor for the @njit params array.

JIT note
--------
_overdamped_drift and _overdamped_diffusion are @njit cores called from the
Euler-Maruyama loop.  The substrate force is recomputed in Python at every
step because calc_en_f is a Python-level function.
"""

import logging
import warnings
import numpy as np
from numba import njit

from tool_create_cluster import rotate, calc_cluster_langevin

_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())


# ============================================================
# JIT cores
# ============================================================

@njit
def _overdamped_drift(x, t, params, out):
    """Fill out with the drift velocity d[i] = F_total_i / eta_i.

    params array layout (length 10, float64):
         0: eta_t   -- translational damping
         1: eta_r   -- rotational damping
         2: Fx_ext  -- external force x
         3: Fy_ext  -- external force y
         4: Tau_ext -- external torque
         5: kBT     -- thermal energy
         6: dt      -- timestep
         7: Fx_sub  -- substrate force x (updated each step)
         8: Fy_sub  -- substrate force y (updated each step)
         9: tau_sub -- substrate torque  (updated each step)

    Args:
        x:      (3,)  float64 -- state [x_cm, y_cm, theta_rad] (unused).
        t:      float         -- time (unused).
        params: (10,) float64 -- physics parameters (see layout above).
        out:    (3,)  float64 -- filled in-place.
    """
    eta_t   = params[0]
    eta_r   = params[1]
    Fx_ext  = params[2]
    Fy_ext  = params[3]
    Tau_ext = params[4]
    Fx_sub  = params[7]
    Fy_sub  = params[8]
    tau_sub = params[9]

    out[0] = (Fx_sub + Fx_ext) / eta_t
    out[1] = (Fy_sub + Fy_ext) / eta_t
    out[2] = (tau_sub + Tau_ext) / eta_r


@njit
def _overdamped_diffusion(x, t, params, out):
    """Fill out with the diagonal noise amplitude matrix B.

    B[i,i] = sqrt(2 * kBT / (eta_i * dt)) so that
    B[i,i]^2 * dt = 2 * D_i = 2 * kBT / eta_i  (correct FDT).
    When kBT=0 the sqrt evaluates to 0 and the noise term vanishes.

    Args:
        x:      (3,)    float64 -- state (unused).
        t:      float           -- time (unused).
        params: (10,)   float64 -- physics parameters (see _overdamped_drift).
        out:    (3, 3)  float64 -- filled in-place; diagonal entries only.
    """
    eta_t = params[0]
    eta_r = params[1]
    kBT   = params[5]
    dt    = params[6]

    out[0, 0] = (2.0 * kBT / (eta_t * dt)) ** 0.5
    out[1, 1] = (2.0 * kBT / (eta_t * dt)) ** 0.5
    out[2, 2] = (2.0 * kBT / (eta_r * dt)) ** 0.5


# ============================================================
# Convenience constructor
# ============================================================

def make_params_array(eta_t, eta_r, Fx, Fy, Tau, kBT, dt):
    """Build the (10,) params array expected by the @njit drift/diffusion.

    Args:
        eta_t: float -- translational damping  (eta * N).
        eta_r: float -- rotational damping      (eta * sum r_i^2).
        Fx:    float -- external force x.
        Fy:    float -- external force y.
        Tau:   float -- external torque.
        kBT:   float -- thermal energy.
        dt:    float -- timestep.

    Returns:
        (10,) float64 ndarray.  Substrate-force slots 7-9 are zero;
        run_md overwrites them each step.
    """
    p = np.zeros(10, dtype=np.float64)
    p[0] = float(eta_t)
    p[1] = float(eta_r)
    p[2] = float(Fx)
    p[3] = float(Fy)
    p[4] = float(Tau)
    p[5] = float(kBT)
    p[6] = float(dt)
    return p


# ============================================================
# MD driver
# ============================================================

def run_md(pos, calc_en_f, en_params,
           eta, Fx=0., Fy=0., Tau=0., kBT=0.,
           dt=1e-4, n_steps=10000,
           theta0=0., pos_cm0=None,
           print_every=100,
           stop_fn=None,
           output_fn=None,
           seed=12345):
    """Run overdamped Langevin MD for a rigid cluster on a substrate.

    Uses the Euler-Maruyama integrator for all kBT >= 0.  At kBT=0 the
    noise term vanishes and the step reduces to explicit Euler gradient
    descent.

    Every print_every steps a state snapshot is emitted.  If output_fn is
    provided it is called with (step, t, state_dict) and run_md returns None;
    otherwise snapshots accumulate in memory and are returned as a dict of
    arrays.

    Args:
        pos:         (N, 2) ndarray   -- cluster positions in the cluster frame
                                         (CM at origin).
        calc_en_f:   callable         -- substrate energy function from
                                         substrate_from_params.  Signature:
                                         (abs_pos, pos_cm, *en_params) ->
                                         (energy, force_cm, torque).
        en_params:   list             -- extra arguments for calc_en_f.
        eta:         float            -- single-particle drag coefficient.
        Fx:          float            -- constant external force x (default 0).
        Fy:          float            -- constant external force y (default 0).
        Tau:         float            -- constant external torque  (default 0).
        kBT:         float            -- thermal energy; 0 for a T=0 run.
        dt:          float            -- timestep (snapshot cadence = print_every*dt).
        n_steps:     int              -- number of MD steps.
        theta0:      float            -- initial orientation in degrees.
        pos_cm0:     (2,) array-like  -- initial CM position; default (0, 0).
        print_every: int              -- emit a snapshot every this many steps.
        stop_fn:     callable or None -- stop_fn(step, state_dict) -> bool;
                                         called at every snapshot;
                                         return True to halt early.
        output_fn:   callable or None -- output_fn(step, t, state_dict) -> None;
                                         if given, run_md returns None.
                                         state_dict keys: 'pos_cm', 'theta',
                                         'energy', 'force', 'torque',
                                         'vel_cm', 'omega'.
        seed:        int              -- numpy RNG seed.

    Returns:
        None if output_fn is provided.
        Otherwise a dict with keys:
            't'      : (n_rec,)    -- time at each snapshot.
            'pos_cm' : (n_rec, 2)  -- CM position.
            'theta'  : (n_rec,)    -- orientation in degrees.
            'energy' : (n_rec,)    -- substrate energy.
            'force'  : (n_rec, 2)  -- substrate force on CM.
            'torque' : (n_rec,)    -- substrate torque.
            'vel_cm' : (n_rec, 2)  -- CM velocity (drift) at snapshot.
            'omega'  : (n_rec,)    -- angular velocity at snapshot.

    Raises:
        ValueError: if eta_r == 0 and (Tau != 0 or kBT > 0).
    """
    pos = np.asarray(pos, dtype=np.float64)
    if pos_cm0 is None:
        pos_cm0 = np.zeros(2, dtype=np.float64)
    pos_cm0 = np.asarray(pos_cm0, dtype=np.float64)

    eta_t, eta_r = calc_cluster_langevin(eta, pos)

    if eta_r == 0.0 and (Tau != 0.0 or kBT > 0.0):
        raise ValueError(
            "eta_r=0 (point-like cluster): orientation is undefined "
            "when Tau!=0 or kBT>0. Use a finite-size cluster."
        )

    if eta_r == 0.0:
        # Single-particle cluster: no moment of inertia, no rotational dynamics.
        # Set eta_r=1 so the JIT cores (which divide by eta_r) don't blow up.
        # With kBT=0 and Tau=0 (guaranteed by the ValueError above), the JIT
        # produces d_buf[2]=0/1=0 and B_buf[2,2]=sqrt(0)=0 automatically, so
        # theta never drifts.  The value 1 is otherwise arbitrary.
        warnings.warn(
            "eta_r=0 (single-particle cluster at origin): rotational drag "
            "is zero. Setting eta_r=1 internally to avoid division by zero "
            "in the integrator. This is safe only when kBT=0 and Tau=0, "
            "which is enforced above. Theta will not evolve.",
            UserWarning, stacklevel=2
        )
        eta_r = 1.0

    if kBT == 0.0:
        warnings.warn(
            "kBT=0 is not recommended: the Euler-Maruyama integrator "
            "uses finite dt and a deterministic saddle point may be "
            "crossed or missed depending on floating-point rounding. "
            "Use a small kBT (e.g. 1e-8 for epsilon=1) to regularise "
            "saddle-point crossings. Continuing with kBT=0.",
            UserWarning, stacklevel=2
        )

    x0 = np.array([pos_cm0[0], pos_cm0[1], np.deg2rad(theta0)], dtype=np.float64)

    params = make_params_array(eta_t, eta_r, Fx, Fy, Tau, kBT, dt)
    rng    = np.random.default_rng(seed)

    d_buf = np.zeros(3,      dtype=np.float64)
    B_buf = np.zeros((3, 3), dtype=np.float64)

    x         = x0.copy()
    t         = 0.0
    pos_cm    = x[:2].copy()
    theta_deg = np.rad2deg(x[2])
    pos_rot   = rotate(pos, theta_deg)

    e0, f0, tau0 = calc_en_f(pos_rot + pos_cm, pos_cm, *en_params)

    records = [] if output_fn is None else None

    for i_step in range(1, n_steps + 1):
        params[7] = f0[0]
        params[8] = f0[1]
        params[9] = float(tau0)

        d_buf[:] = 0.
        B_buf[:] = 0.
        _overdamped_drift(x, t, params, d_buf)
        _overdamped_diffusion(x, t, params, B_buf)

        xi    = rng.standard_normal(3)
        delta = (d_buf + B_buf @ xi) * dt

        x += delta
        t += dt

        pos_cm    = x[:2].copy()
        theta_deg = np.rad2deg(x[2])
        pos_rot   = rotate(pos, theta_deg)

        e0, f0, tau0 = calc_en_f(pos_rot + pos_cm, pos_cm, *en_params)

        if i_step % print_every == 0:
            state = {
                'pos_cm': pos_cm.copy(),
                'theta':  theta_deg,
                'energy': float(e0),
                'force':  np.asarray(f0, dtype=np.float64).copy(),
                'torque': float(tau0),
                'vel_cm': (delta[:2] / dt).copy(),
                'omega':  float(delta[2] / dt),
            }

            if output_fn is not None:
                output_fn(i_step, t, state)
            else:
                state['t'] = t
                records.append(state)

            _log.debug("step %6d  t=%.4e  E=%.6f", i_step, t, float(e0))

            if stop_fn is not None and stop_fn(i_step, state):
                _log.debug("stop_fn triggered at step %d", i_step)
                break

    if output_fn is not None:
        return None

    return {
        't':      np.array([r['t']      for r in records]),
        'pos_cm': np.array([r['pos_cm'] for r in records]),
        'theta':  np.array([r['theta']  for r in records]),
        'energy': np.array([r['energy'] for r in records]),
        'force':  np.array([r['force']  for r in records]),
        'torque': np.array([r['torque'] for r in records]),
        'vel_cm': np.array([r['vel_cm'] for r in records]),
        'omega':  np.array([r['omega']  for r in records]),
    }
