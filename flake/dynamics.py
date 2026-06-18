r"""
Overdamped Langevin dynamics for a rigid cluster on a substrate.

Physics
-------
Equations of motion:

.. math::

    \eta_t \frac{dx_\mathrm{cm}}{dt} &= F_x^\mathrm{sub}(x,y,\theta) + F_x^\mathrm{ext} + \xi_x(t) \\
    \eta_t \frac{dy_\mathrm{cm}}{dt} &= F_y^\mathrm{sub}(x,y,\theta) + F_y^\mathrm{ext} + \xi_y(t) \\
    \eta_r \frac{d\theta}{dt}        &= \tau^\mathrm{sub}(x,y,\theta) + \tau^\mathrm{ext}  + \xi_r(t)

where :math:`\eta_t = \eta N` and :math:`\eta_r = \eta \sum_i r_i^2`.

Fluctuation-dissipation theorem:

.. math::

    \langle \xi_i(t)^2 \rangle = 2 k_B T \, \eta_i \quad \text{(variance per unit time)}

Integration
-----------
Euler-Maruyama step:

.. math::

    \delta_i = \frac{F_i^\mathrm{total}}{\eta_i} \, dt
               + \sqrt{\frac{2 k_B T}{\eta_i \, dt}} \; \xi_i \, dt, \quad \xi_i \sim \mathcal{N}(0,1)

The displacement variance is :math:`\langle \delta_i^2 \rangle = 2 k_B T / \eta_i \cdot dt = 2 D_i \, dt`.

At :math:`k_B T = 0` the noise term vanishes and this reduces to explicit Euler gradient descent.
The step size ``dt`` is fixed; check convergence by halving it.
Avoid :math:`k_B T = 0` exactly: a deterministic trajectory at a saddle point is sensitive to
floating-point rounding. Use ``kBT=1e-8`` (for ``epsilon=1``) instead.

JIT path vs Python path
-----------------------
When ``stop_fn`` and ``output_fn`` are both ``None``, the inner EM loop runs entirely
in Numba JIT via ``_md_loop_njit`` — no Python/JIT boundary per step.
This requires ``calc_en_f`` to expose ``_jit_core`` and ``_jit_params`` attributes,
which ``substrate_from_params`` attaches automatically.  If those attributes are
absent and no callbacks are provided, ``run_md`` raises ``NotImplementedError``.

When ``stop_fn`` or ``output_fn`` is provided, the Python loop path is used instead.
This is ~6–11× slower but supports arbitrary Python callbacks, and ``calc_en_f``
does not need ``_jit_core``/``_jit_params``.

RNG and reproducibility
-----------------------
The two paths use different RNG backends.  The JIT path calls
``np.random.seed(seed)`` before the loop, which seeds Numba's internal xorshift
generator; ``np.random.normal`` inside ``@njit`` draws from that generator.
The Python path uses ``np.random.default_rng(seed)`` (PCG64), which is a
completely separate RNG.  **The same integer ``seed`` will produce different
trajectories on the JIT and Python paths.** Do not compare results across paths
expecting identical noise sequences.

Public API
----------
    run_md  -- Euler-Maruyama integrator for all :math:`k_B T \geq 0`.
"""

import logging
import warnings
import numpy as np
from numba import njit

from flake.cluster import calc_cluster_langevin

_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())


# ============================================================
# JIT cores
# ============================================================

@njit
def _rotate_core(pos, angle_deg):
    """Rotate (N,2) array by angle_deg about origin.

    Duplicate of rotate() in tool_create_cluster, made @njit so the
    MD step loop can call it without crossing the Python/JIT boundary.
    """
    theta = angle_deg * 3.141592653589793 / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    out = np.empty_like(pos)
    for i in range(pos.shape[0]):
        out[i, 0] = c * pos[i, 0] - s * pos[i, 1]
        out[i, 1] = s * pos[i, 0] + c * pos[i, 1]
    return out


@njit
def _md_loop_njit(pos, calc_en_core, substrate_params,
                  eta_t, eta_r,
                  Fx_ext, Fy_ext, Tau_ext,
                  kBT, dt, x0, n_steps, print_every):
    """Euler-Maruyama loop, fully in Numba JIT.

    calc_en_core is a @njit function with signature:
        (abs_pos, pos_cm, *substrate_params) -> (float, (2,) float64, float)
    substrate_params is a tuple of pre-converted float64 arrays captured in
    the en_func closure and exposed via en_func._jit_params.

    Numba specialises this function for each concrete (calc_en_core, param
    tuple type) pair at first call; subsequent calls use the compiled version.

    RNG: np.random.seed() before calling from Python seeds Numba's internal
    xorshift generator.  np.random.normal(0.0, 1.0) inside @njit uses that
    generator -- no Python/JIT boundary crossing.

    Returns flat arrays (no Python dicts inside @njit):
        (t, xcm, ycm, theta_deg, energy, Fx, Fy, torque, vcmx, vcmy, omega)
    each shape (n_steps // print_every,).
    """
    n_rec = n_steps // print_every
    t_arr    = np.empty(n_rec)
    xcm_arr  = np.empty(n_rec)
    ycm_arr  = np.empty(n_rec)
    th_arr   = np.empty(n_rec)
    en_arr   = np.empty(n_rec)
    Fx_arr   = np.empty(n_rec)
    Fy_arr   = np.empty(n_rec)
    tau_arr  = np.empty(n_rec)
    vcmx_arr = np.empty(n_rec)
    vcmy_arr = np.empty(n_rec)
    om_arr   = np.empty(n_rec)

    x = x0.copy()
    t = 0.0
    i_rec = 0

    B_t = (2.0 * kBT / (eta_t * dt)) ** 0.5
    B_r = (2.0 * kBT / (eta_r * dt)) ** 0.5

    for i_step in range(1, n_steps + 1):
        theta_deg = x[2] * 180.0 / 3.141592653589793
        pos_rot   = _rotate_core(pos, theta_deg)

        abs_pos = np.empty_like(pos_rot)
        for i in range(pos_rot.shape[0]):
            abs_pos[i, 0] = pos_rot[i, 0] + x[0]
            abs_pos[i, 1] = pos_rot[i, 1] + x[1]
        pos_cm = x[:2].copy()

        e0, f0, tau0 = calc_en_core(abs_pos, pos_cm, *substrate_params)

        xi0 = np.random.normal(0.0, 1.0)
        xi1 = np.random.normal(0.0, 1.0)
        xi2 = np.random.normal(0.0, 1.0)

        dx     = ((f0[0] + Fx_ext) / eta_t + B_t * xi0) * dt
        dy     = ((f0[1] + Fy_ext) / eta_t + B_t * xi1) * dt
        dtheta = ((tau0  + Tau_ext) / eta_r + B_r * xi2) * dt

        x[0] += dx
        x[1] += dy
        x[2] += dtheta
        t    += dt

        if i_step % print_every == 0 and i_rec < n_rec:
            t_arr[i_rec]    = t
            xcm_arr[i_rec]  = x[0]
            ycm_arr[i_rec]  = x[1]
            th_arr[i_rec]   = x[2] * 180.0 / 3.141592653589793
            en_arr[i_rec]   = e0
            Fx_arr[i_rec]   = f0[0]
            Fy_arr[i_rec]   = f0[1]
            tau_arr[i_rec]  = tau0
            vcmx_arr[i_rec] = dx / dt
            vcmy_arr[i_rec] = dy / dt
            om_arr[i_rec]   = dtheta / dt
            i_rec += 1

    return (t_arr, xcm_arr, ycm_arr, th_arr, en_arr,
            Fx_arr, Fy_arr, tau_arr, vcmx_arr, vcmy_arr, om_arr)


# ============================================================
# Python-loop path (used when callbacks are present)
# ============================================================

def _run_python_loop(pos, calc_en_f, en_params,
                     eta_t, eta_r, Fx, Fy, Tau, kBT, dt,
                     x0, n_steps, print_every,
                     stop_fn, output_fn, seed):
    """Euler-Maruyama loop in Python; supports stop_fn and output_fn callbacks.

    Called by run_md when stop_fn or output_fn is not None.
    Uses np.random.default_rng for reproducibility on the Python path.

    Returns dict of arrays, or None if output_fn is provided.
    """
    from flake.cluster import rotate

    rng = np.random.default_rng(seed)

    x         = x0.copy()
    t         = 0.0
    pos_cm    = x[:2].copy()
    theta_deg = np.rad2deg(x[2])
    pos_rot   = rotate(pos, theta_deg)

    e0, f0, tau0 = calc_en_f(pos_rot + pos_cm, pos_cm, *en_params)

    records = [] if output_fn is None else None

    # Pre-compute noise amplitudes (constant through the run).
    B_t = (2.0 * kBT / (eta_t * dt)) ** 0.5
    B_r = (2.0 * kBT / (eta_r * dt)) ** 0.5

    for i_step in range(1, n_steps + 1):
        xi    = rng.standard_normal(3)
        dx    = ((f0[0] + Fx) / eta_t + B_t * xi[0]) * dt
        dy    = ((f0[1] + Fy) / eta_t + B_t * xi[1]) * dt
        dth   = ((tau0  + Tau) / eta_r + B_r * xi[2]) * dt

        x[0] += dx
        x[1] += dy
        x[2] += dth
        t    += dt

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
                'vel_cm': np.array([dx / dt, dy / dt]),
                'omega':  float(dth / dt),
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

    Performance note:
        When stop_fn and output_fn are both None, the inner loop runs in
        Numba JIT (_md_loop_njit) with no Python overhead per step.
        When either callback is provided, the slower Python loop runs instead.
        For maximum throughput, avoid callbacks and post-process the returned
        trajectory dict.

    Args:
        pos: (N, 2) ndarray -- cluster positions in the cluster frame (CM at origin).
        calc_en_f: callable -- substrate energy function. Must expose
            ``_jit_core`` and ``_jit_params`` attributes when no callbacks are
            provided (JIT path); not required when stop_fn or output_fn is set.
            Use ``substrate_from_params`` to build a compliant function.
        en_params: list -- extra arguments for ``calc_en_f``. Always ``[]``
            for functions produced by ``substrate_from_params`` (closures capture
            all parameters internally).
        eta: float -- single-particle drag coefficient.
        Fx: float -- constant external force along x (default 0).
        Fy: float -- constant external force along y (default 0).
        Tau: float -- constant external torque (default 0).
        kBT: float -- thermal energy; 0 for a T=0 run.
        dt: float -- timestep.
        n_steps: int -- number of MD steps.
        theta0: float -- initial orientation in degrees.
        pos_cm0: (2,) array-like -- initial CM position (default ``(0, 0)``).
        print_every: int -- emit a snapshot every this many steps.
        stop_fn: callable or None -- ``stop_fn(step, state_dict) -> bool``;
            forces Python loop path.
        output_fn: callable or None -- ``output_fn(step, t, state_dict) -> None``;
            forces Python loop; ``run_md`` returns ``None``.
        seed: int -- RNG seed. Note: the JIT path (no callbacks) seeds Numba's
            internal xorshift via ``np.random.seed``; the Python path (callbacks
            present) uses ``np.random.default_rng``. The same seed gives
            different trajectories on the two paths.

    Returns:
        ``None`` if ``output_fn`` is provided. Otherwise a dict with keys
        ``'t'`` (time), ``'pos_cm'`` (CM position), ``'theta'`` (degrees),
        ``'energy'``, ``'force'``, ``'torque'``, ``'vel_cm'``, ``'omega'``.

    Raises:
        ValueError: if ``eta_r == 0`` and (``Tau != 0`` or ``kBT > 0``).
        NotImplementedError: if ``calc_en_f`` lacks ``_jit_core``/``_jit_params``
            and no callbacks are provided.

    Warns:
        UserWarning: if ``eta_r == 0`` even at T=0 (safe but theta will not evolve).
        UserWarning: if ``kBT == 0`` (deterministic integrator at saddle points).
        UserWarning: if ``|theta0| > 720`` (likely radians passed by mistake).
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

    if abs(theta0) > 720.0:
        warnings.warn(
            "theta0=%.4g looks suspiciously large. "
            "run_md expects degrees, not radians." % theta0,
            UserWarning, stacklevel=2
        )

    x0 = np.array([pos_cm0[0], pos_cm0[1], np.deg2rad(theta0)], dtype=np.float64)

    # Python loop path: required when callbacks are present.
    if stop_fn is not None or output_fn is not None:
        return _run_python_loop(
            pos, calc_en_f, en_params,
            eta_t, eta_r, Fx, Fy, Tau, kBT, dt,
            x0, n_steps, print_every,
            stop_fn, output_fn, seed
        )

    # JIT path: requires _jit_core and _jit_params on calc_en_f.
    jit_core   = getattr(calc_en_f, '_jit_core',   None)
    jit_params = getattr(calc_en_f, '_jit_params', None)
    if jit_core is None or jit_params is None:
        raise NotImplementedError(
            "run_md requires calc_en_f to expose _jit_core and _jit_params. "
            "Use substrate_from_params to build the energy function. "
            "Custom energy functions must attach these attributes manually "
            "to use run_md without callbacks."
        )

    # np.random.seed sets Numba's internal xorshift state.
    # np.random.default_rng() does NOT affect Numba RNG -- do not use it here.
    np.random.seed(seed)

    arrs = _md_loop_njit(
        pos, jit_core, jit_params,
        eta_t, eta_r, Fx, Fy, Tau, kBT, dt,
        x0, n_steps, print_every
    )
    t_a, xcm_a, ycm_a, th_a, en_a, fx_a, fy_a, tau_a, vx_a, vy_a, om_a = arrs

    return {
        't':      t_a,
        'pos_cm': np.column_stack([xcm_a, ycm_a]),
        'theta':  th_a,
        'energy': en_a,
        'force':  np.column_stack([fx_a, fy_a]),
        'torque': tau_a,
        'vel_cm': np.column_stack([vx_a, vy_a]),
        'omega':  om_a,
    }
