#!/usr/bin/env python3
"""
Substrate potential functions for rigid-cluster simulations.

Each substrate type exposes two public functions:
    particle_en_<type>  --  per-particle energy, force, torque contribution
    calc_en_<type>      --  total (summed) energy, force, torque on CM

Both return (energy, force, torque).  The particle-level functions are useful
for diagnostics and testing; production code typically calls calc_en_<type>
through the substrate_from_params factory.

Coordinate convention
---------------------
All positions are 2-D Cartesian (x, y).  The cluster CM is at pos_cm; the
rigid-body orientation is handled externally.  Torque is computed about
pos_torque (normally the CM).

Units: consistent with the calling code -- distances in micron, forces in fN,
energies in zJ, angles in radians internally (degrees at the MD level).

JIT notes
---------
The inner loops are compiled with Numba @njit.  This means:
  - No NumPy fancy indexing or boolean masks inside jitted functions.
  - Explicit for-loops over particles -- Numba handles these efficiently.
  - Functions passed to @njit callers must themselves be @njit.
The non-jitted wrappers (calc_en_*) handle array allocation and call the
jitted core.
"""

import numpy as np
from numpy import pi, sqrt
from numba import njit


# ============================================================
# Lattice metric matrices
# ============================================================

def calc_matrices_square(R):
    """Metric matrices for a square lattice of spacing R.

    Returns (u, u_inv): 2x2 matrices that map real-space displacements to
    fractional coordinates and back.
    """
    area = R * R
    u     = np.array([[1., 0.], [0., 1.]]) * (R / area)
    u_inv = np.array([[1., 0.], [0., 1.]]) * R
    return u, u_inv


def calc_matrices_triangle(R):
    """Metric matrices for a triangular lattice of spacing R.

    Nearest-neighbour direction is along x (consistent with the cluster
    creation conventions in tool_create_cluster).
    """
    area = R * R * sqrt(3.) / 2.
    u     = np.array([[sqrt(3.) / 2., 0.5 ], [0.,  1.       ]]) * (R / area)
    u_inv = np.array([[1.,           -0.5 ], [0.,  sqrt(3.) / 2.]]) * R
    return u, u_inv


def calc_matrices_bvect(b1, b2):
    """Metric matrices from two arbitrary primitive lattice vectors.

    Args:
        b1, b2: array-like, shape (2,).

    Returns (u, u_inv) such that u @ r gives fractional coordinates and
    u_inv @ frac gives real-space coordinates.
    """
    St = np.array([b1, b2], dtype=float)
    u     = np.linalg.inv(St).T
    u_inv = St.T
    return u, u_inv


# ============================================================
# Gaussian substrate
# ============================================================

def gaussian(x, mu, sigma):
    """Unnormalised Gaussian: exp(-(x-mu)^2 / (2*sigma^2)).

    Kept public because tool_reciprocal_space.py uses it directly.
    Not normalised by design -- it represents a well *shape*, not a PDF.
    """
    return np.exp(-np.square(x - mu) / (2. * np.square(sigma)))


@njit
def _particle_en_gaussian_core(pos, pos_torque, basis,
                                a, b, sigma, epsilon,
                                u, u_inv):
    """JIT core for Gaussian-well substrate.

    The well is:
        V(r) = -epsilon * exp(-r^2 / (2*sigma^2))          for r <= a  (bulk)
        V(r) = -epsilon * exp(-r^2 / (2*sigma^2)) * f(rho) for a < r < b  (tail)
        V(r) = 0                                            for r >= b  (outside)

    where rho = (r-a)/(b-a) in [0,1] and f is the C^2 smoothstep:
        f(rho) = 1 - 10*rho^3 + 15*rho^4 - 6*rho^5

    The tail ensures the potential and its first two derivatives go to zero
    smoothly at r=b, avoiding discontinuous forces.

    Args:
        pos:        (N, 2) particle positions.
        pos_torque: (2,)   reference point for torque (usually CM).
        basis:      (M, 2) positions of the crystal basis sites.
        a, b:       inner radius (no damping) and cutoff radius.
        sigma:      width of the Gaussian well.
        epsilon:    depth of the well (positive = attractive).
        u, u_inv:   (2, 2) metric matrices from calc_matrices_bvect.

    Returns:
        en:  (N,) potential energy per particle.
        F:   (N, 2) force on each particle.
        tau: (N,) torque contribution per particle about pos_torque.
    """
    N = pos.shape[0]
    en  = np.zeros(N)
    F   = np.zeros((N, 2))
    tau = np.zeros(N)

    for r in basis:
        # F_site accumulates forces from this basis site only.
        # We need this separate from F to compute the torque correctly:
        # tau += r_from_app x F_site, NOT r_from_app x F_total.
        F_site = np.zeros((N, 2))

        for i in range(N):
            # Map particle into the substrate unit cell (fractional coords).
            dx = pos[i, 0] - r[0]
            dy = pos[i, 1] - r[1]
            fx = u[0, 0] * dx + u[0, 1] * dy
            fy = u[1, 0] * dx + u[1, 1] * dy
            # Fold back to [-0.5, 0.5) -- nearest image in substrate cell.
            fx -= np.floor(fx + 0.5)
            fy -= np.floor(fy + 0.5)
            # Back to real space (shortest vector to substrate site).
            rx = u_inv[0, 0] * fx + u_inv[0, 1] * fy
            ry = u_inv[1, 0] * fx + u_inv[1, 1] * fy
            rr = sqrt(rx * rx + ry * ry)

            if rr <= a:
                # Bulk region: full Gaussian, no damping.
                g = np.exp(-rr * rr / (2. * sigma * sigma))
                en[i] += -epsilon * g
                if rr > 0.:
                    # F = -dV/dr: V = -epsilon*g, so dV/dr = epsilon*g*r/sigma^2 > 0,
                    # meaning the force is inward (toward the well centre).
                    scale = epsilon * g / (sigma * sigma)
                    F_site[i, 0] -= scale * rx
                    F_site[i, 1] -= scale * ry
                # At rr == 0 force is exactly zero by symmetry; skip.

            elif rr < b:
                # Tail region: Gaussian * smoothstep.
                rho  = (rr - a) / (b - a)
                f    =  1. - 10.*rho**3 + 15.*rho**4 -  6.*rho**5
                df   = (-30.*rho**2 + 60.*rho**3 - 30.*rho**4) / (b - a)
                g    = np.exp(-rr * rr / (2. * sigma * sigma))
                en[i] += -epsilon * g * f
                # F = -dV/dr = epsilon * (g' * f + g * f') projected onto x,y
                # g' = g * r / sigma^2  (with sign: d/dr of -epsilon*g*f)
                gp = g * rr / (sigma * sigma)   # |dg/dr|
                force_r = -epsilon * (gp * f - g * df)  # radial magnitude; minus because V=-epsilon*g*f
                if rr > 0.:
                    F_site[i, 0] += force_r * rx / rr
                    F_site[i, 1] += force_r * ry / rr

            # rr >= b: zero energy and force, nothing to add.

        # Torque about pos_torque from this basis site's forces.
        # tau_i = (pos_i - pos_torque) x F_site_i  (2-D cross product -> scalar)
        for i in range(N):
            arm_x = pos[i, 0] - pos_torque[0]
            arm_y = pos[i, 1] - pos_torque[1]
            tau[i] += arm_x * F_site[i, 1] - arm_y * F_site[i, 0]

        # Accumulate site forces into total force array.
        for i in range(N):
            F[i, 0] += F_site[i, 0]
            F[i, 1] += F_site[i, 1]

    return en, F, tau


# One JIT boundary crossing per MD step: the closure in substrate_from_params
# calls this directly with pre-converted float64 arrays, so no Python-level
# np.sum or np.array(basis,...) overhead accumulates over 200k steps.
@njit
def _calc_en_gaussian_core(pos, pos_cm,
                            basis, a, b, sigma, epsilon, u, u_inv):
    """Sum particle energies, forces, and torque over all cluster particles.

    Called from the en_func closure returned by substrate_from_params.
    All arguments are pre-converted float64 arrays; no conversion at call time.

    Returns:
        (float, (2,) float64, float) -- total energy, force on CM, total torque.
    """
    en, F, tau = _particle_en_gaussian_core(
        pos, pos_cm, basis, a, b, sigma, epsilon, u, u_inv)
    E_tot   = 0.0
    Fx_tot  = 0.0
    Fy_tot  = 0.0
    tau_tot = 0.0
    for i in range(en.shape[0]):
        E_tot   += en[i]
        Fx_tot  += F[i, 0]
        Fy_tot  += F[i, 1]
        tau_tot += tau[i]
    return E_tot, np.array([Fx_tot, Fy_tot]), tau_tot


def particle_en_gaussian(pos, pos_torque, basis, a, b, sigma, epsilon, u, u_inv):
    """Per-particle energy, force, and torque for a Gaussian-well substrate.

    See _particle_en_gaussian_core for the potential definition.

    Args:
        pos:        (N, 2) ndarray -- particle positions.
        pos_torque: (2,)   ndarray -- torque reference point (usually CM).
        basis:      list of (2,) arrays -- substrate basis site positions.
        a:          float -- inner cutoff (bulk region ends here).
        b:          float -- outer cutoff (tail region ends here).
        sigma:      float -- Gaussian width.
        epsilon:    float -- well depth (positive = attractive).
        u, u_inv:   (2,2) ndarrays -- metric matrices from calc_matrices_bvect.

    Returns:
        en:  (N,) ndarray -- potential energy per particle.
        F:   (N, 2) ndarray -- force on each particle.
        tau: (N,) ndarray -- torque contribution per particle.
    """
    basis_arr = np.array(basis, dtype=np.float64)
    return _particle_en_gaussian_core(
        np.asarray(pos, dtype=np.float64),
        np.asarray(pos_torque, dtype=np.float64),
        basis_arr, a, b, sigma, epsilon,
        np.asarray(u, dtype=np.float64),
        np.asarray(u_inv, dtype=np.float64),
    )


def calc_en_gaussian(pos, pos_torque, basis, a, b, sigma, epsilon, u, u_inv):
    """Total energy, force, and torque on the CM for a Gaussian-well substrate.

    Sums particle_en_gaussian over all particles.

    Returns:
        (float, (2,) ndarray, float) -- total energy, total force, total torque.
    """
    en, F, tau = particle_en_gaussian(pos, pos_torque, basis,
                                       a, b, sigma, epsilon, u, u_inv)
    return np.sum(en), np.sum(F, axis=0), np.sum(tau)


# ============================================================
# Tanh substrate
# ============================================================

@njit
def _particle_en_tanh_core(pos, pos_torque, basis,
                            a, b, ww, epsilon,
                            u, u_inv):
    """JIT core for tanh-shaped substrate wells.

    The well profile (following Cao, Phys. Rev. E 103, 2021):
        V = -epsilon                                      for r <= a  (flat bottom)
        V = epsilon/2 * (tanh((rho-ww)/(rho*(1-rho))) - 1)  for a < r < b
        V = 0                                             for r >= b

    where rho = (r-a)/(b-a) in (0,1).  The parameter ww controls the
    steepness of the wall: smaller ww -> steeper outer wall.

    Args:
        pos, pos_torque, basis: same as Gaussian core.
        a:       inner flat-bottom radius.
        b:       outer cutoff radius.
        ww:      shape parameter (wall steepness).
        epsilon: well depth.
        u, u_inv: metric matrices.

    Returns:
        en, F, tau: same shapes as Gaussian core.
    """
    N = pos.shape[0]
    en  = np.zeros(N)
    F   = np.zeros((N, 2))
    tau = np.zeros(N)

    for r in basis:
        F_site = np.zeros((N, 2))

        for i in range(N):
            dx = pos[i, 0] - r[0]
            dy = pos[i, 1] - r[1]
            fx = u[0, 0] * dx + u[0, 1] * dy
            fy = u[1, 0] * dx + u[1, 1] * dy
            fx -= np.floor(fx + 0.5)
            fy -= np.floor(fy + 0.5)
            rx = u_inv[0, 0] * fx + u_inv[0, 1] * fy
            ry = u_inv[1, 0] * fx + u_inv[1, 1] * fy
            rr = sqrt(rx * rx + ry * ry)

            if rr <= a:
                en[i] += -epsilon
                # Flat bottom: force is zero.

            elif rr < b:
                rho = (rr - a) / (b - a)
                arg = (rho - ww) / (rho * (1. - rho))
                en[i] += epsilon / 2. * (np.tanh(arg) - 1.)
                # Derivative of tanh term with respect to rho:
                # d/drho [ (rho-ww)/(rho*(1-rho)) ] = (rho^2 + ww - 2*ww*rho) / (rho*(1-rho))^2
                # Force = -dV/dr = -(dV/drho)*(1/(b-a))
                cosh_arg = np.cosh(arg)
                darg_drho = (rho * rho + ww - 2. * ww * rho) / (rho * (1. - rho)) ** 2
                dV_drho   = epsilon / 2. / (cosh_arg * cosh_arg) * darg_drho
                force_r   = -dV_drho / (b - a)  # radial force component
                if rr > 0.:
                    F_site[i, 0] += force_r * rx / rr
                    F_site[i, 1] += force_r * ry / rr

        for i in range(N):
            arm_x = pos[i, 0] - pos_torque[0]
            arm_y = pos[i, 1] - pos_torque[1]
            tau[i] += arm_x * F_site[i, 1] - arm_y * F_site[i, 0]

        for i in range(N):
            F[i, 0] += F_site[i, 0]
            F[i, 1] += F_site[i, 1]

    return en, F, tau


# Same rationale as _calc_en_gaussian_core: sum inside Numba, called once
# per MD step with zero Python array-conversion overhead.
@njit
def _calc_en_tanh_core(pos, pos_cm,
                       basis, a, b, ww, epsilon, u, u_inv):
    """Sum particle energies, forces, and torque for tanh-well substrate.

    See _calc_en_gaussian_core for the rationale.

    Returns:
        (float, (2,) float64, float) -- total energy, force on CM, total torque.
    """
    en, F, tau = _particle_en_tanh_core(
        pos, pos_cm, basis, a, b, ww, epsilon, u, u_inv)
    E_tot   = 0.0
    Fx_tot  = 0.0
    Fy_tot  = 0.0
    tau_tot = 0.0
    for i in range(en.shape[0]):
        E_tot   += en[i]
        Fx_tot  += F[i, 0]
        Fy_tot  += F[i, 1]
        tau_tot += tau[i]
    return E_tot, np.array([Fx_tot, Fy_tot]), tau_tot


def particle_en_tanh(pos, pos_torque, basis, a, b, ww, epsilon, u, u_inv):
    """Per-particle energy, force, and torque for a tanh-well substrate.

    Args:
        pos:        (N, 2) ndarray -- particle positions.
        pos_torque: (2,)   ndarray -- torque reference point.
        basis:      list of (2,) arrays -- substrate basis site positions.
        a:          float -- flat-bottom radius.
        b:          float -- outer cutoff radius.
        ww:         float -- wall shape parameter.
        epsilon:    float -- well depth.
        u, u_inv:   (2, 2) ndarrays -- metric matrices.

    Returns:
        en, F, tau: same as particle_en_gaussian.
    """
    basis_arr = np.array(basis, dtype=np.float64)
    return _particle_en_tanh_core(
        np.asarray(pos, dtype=np.float64),
        np.asarray(pos_torque, dtype=np.float64),
        basis_arr, a, b, ww, epsilon,
        np.asarray(u, dtype=np.float64),
        np.asarray(u_inv, dtype=np.float64),
    )


def calc_en_tanh(pos, pos_torque, basis, a, b, ww, epsilon, u, u_inv):
    """Total energy, force, and torque on the CM for a tanh-well substrate.

    Returns:
        (float, (2,) ndarray, float) -- total energy, total force, total torque.
    """
    en, F, tau = particle_en_tanh(pos, pos_torque, basis,
                                   a, b, ww, epsilon, u, u_inv)
    return np.sum(en), np.sum(F, axis=0), np.sum(tau)


# ============================================================
# Sinusoidal (plane-wave) substrate
# ============================================================

def get_ks(R, n, c_n, alpha_n):
    """Wave vectors for an n-fold symmetric sinusoidal substrate.

    Constructs n plane waves whose interference gives a potential with the
    desired lattice symmetry.  Coefficients from:
        Vanossi, Manini, Tosatti, PNAS 109, 16429 (2012).

    Preset recipes:
        Lines        : n=2, c_n=1,            alpha_n=0
        Triangular   : n=3, c_n=4/3,          alpha_n=0
        Square       : n=4, c_n=sqrt(2),      alpha_n=pi/4
        Quasicrystal5: n=5, c_n=2,            alpha_n=0
        Quasicrystal6: n=6, c_n=4/sqrt(3),    alpha_n=-pi/6

    Args:
        R:       float -- lattice spacing.
        n:       int   -- number of plane waves (fold symmetry).
        c_n:     float -- amplitude pre-factor for |k|.
        alpha_n: float -- overall rotation of the wave-vector star [radians].

    Returns:
        ks: (n, 2) ndarray of wave vectors.
    """
    return np.array([
        c_n * pi / R * np.array([np.cos(2. * pi / n * l + alpha_n),
                                  np.sin(2. * pi / n * l + alpha_n)])
        for l in range(n)
    ])


@njit
def _particle_en_sin_core(pos, pos_torque, basis, ks, epsilon):
    """JIT core for sinusoidal (plane-wave interference) substrate.

    The potential is:
        V(r) = -epsilon/n^2 * |sum_l exp(i k_l . r)|^2

    which expands to a sum of cosines.  The pre-factor ensures V ranges
    from -epsilon (at a potential minimum) to 0 (at a saddle/maximum)
    regardless of n.

    Forces follow from F = -grad V.

    Args:
        pos:        (N, 2) particle positions.
        pos_torque: (2,)   torque reference point.
        basis:      (M, 2) substrate basis positions.
        ks:         (n, 2) wave vectors from get_ks.
        epsilon:    float  well depth.

    Returns:
        en, F, tau: shapes (N,), (N,2), (N,).
    """
    N    = pos.shape[0]
    nk   = ks.shape[0]
    en   = np.zeros(N)
    F    = np.zeros((N, 2))
    tau  = np.zeros(N)
    inv_n2 = 1. / (nk * nk)

    for r in basis:
        F_site = np.zeros((N, 2))

        for i in range(N):
            x = pos[i, 0] - r[0]
            y = pos[i, 1] - r[1]

            # Energy: -epsilon/n^2 * |sum exp(i k.r)|^2
            # = -epsilon/n^2 * (sum cos(k.r))^2 + (sum sin(k.r))^2)
            sum_cos = 0.
            sum_sin = 0.
            for l in range(nk):
                phase = ks[l, 0] * x + ks[l, 1] * y
                sum_cos += np.cos(phase)
                sum_sin += np.sin(phase)
            en[i] += -epsilon * inv_n2 * (sum_cos * sum_cos + sum_sin * sum_sin)

            # Force: F = -dV/dr
            # dV/dx = -epsilon/n^2 * 2*(sum_cos * sum(k_x sin) - sum_sin * sum(k_x cos))
            sum_kx_sin = 0.
            sum_ky_sin = 0.
            sum_kx_cos = 0.
            sum_ky_cos = 0.
            for l in range(nk):
                phase = ks[l, 0] * x + ks[l, 1] * y
                s = np.sin(phase)
                c = np.cos(phase)
                sum_kx_sin += ks[l, 0] * s
                sum_ky_sin += ks[l, 1] * s
                sum_kx_cos += ks[l, 0] * c
                sum_ky_cos += ks[l, 1] * c

            # F = -dV/dr: note sign and factor of 2
            F_site[i, 0] += -epsilon * 2. * inv_n2 * (
                sum_cos * sum_kx_sin - sum_sin * sum_kx_cos
            )
            F_site[i, 1] += -epsilon * 2. * inv_n2 * (
                sum_cos * sum_ky_sin - sum_sin * sum_ky_cos
            )

        for i in range(N):
            arm_x = pos[i, 0] - pos_torque[0]
            arm_y = pos[i, 1] - pos_torque[1]
            tau[i] += arm_x * F_site[i, 1] - arm_y * F_site[i, 0]

        for i in range(N):
            F[i, 0] += F_site[i, 0]
            F[i, 1] += F_site[i, 1]

    return en, F, tau


# Same rationale as _calc_en_gaussian_core.  ks is pre-converted once in
# substrate_from_params; basis and epsilon likewise.
@njit
def _calc_en_sin_core(pos, pos_cm, basis, ks, epsilon):
    """Sum particle energies, forces, and torque for plane-wave substrate.

    See _calc_en_gaussian_core for the rationale.

    Returns:
        (float, (2,) float64, float) -- total energy, force on CM, total torque.
    """
    en, F, tau = _particle_en_sin_core(
        pos, pos_cm, basis, ks, epsilon)
    E_tot   = 0.0
    Fx_tot  = 0.0
    Fy_tot  = 0.0
    tau_tot = 0.0
    for i in range(en.shape[0]):
        E_tot   += en[i]
        Fx_tot  += F[i, 0]
        Fy_tot  += F[i, 1]
        tau_tot += tau[i]
    return E_tot, np.array([Fx_tot, Fy_tot]), tau_tot


def particle_en_sin(pos, pos_torque, basis, ks, epsilon):
    """Per-particle energy, force, and torque for a plane-wave substrate.

    Args:
        pos:        (N, 2) ndarray -- particle positions.
        pos_torque: (2,)   ndarray -- torque reference point.
        basis:      list of (2,) arrays -- substrate basis site positions.
        ks:         (n, 2) ndarray  -- wave vectors (from get_ks).
        epsilon:    float           -- well depth.

    Returns:
        en, F, tau: same shapes as particle_en_gaussian.
    """
    basis_arr = np.array(basis, dtype=np.float64)
    return _particle_en_sin_core(
        np.asarray(pos, dtype=np.float64),
        np.asarray(pos_torque, dtype=np.float64),
        basis_arr,
        np.asarray(ks, dtype=np.float64),
        float(epsilon),
    )


def calc_en_sin(pos, pos_torque, basis, ks, epsilon):
    """Total energy, force, and torque on the CM for a plane-wave substrate.

    Returns:
        (float, (2,) ndarray, float) -- total energy, total force, total torque.
    """
    en, F, tau = particle_en_sin(pos, pos_torque, basis, ks, epsilon)
    return np.sum(en), np.sum(F, axis=0), np.sum(tau)


# ============================================================
# Flat (zero) substrate
# ============================================================

@njit
def _calc_en_flat_core(pos, pos_cm):
    """Zero energy and force for all positions.

    Used as a trivial substrate for testing the integrator in isolation:
    with no substrate force, analytic predictions for drift velocity and
    diffusion are exact, giving a clean check of the EM integrator.

    Returns:
        (0.0, zeros(2), 0.0)
    """
    return 0.0, np.zeros(2), 0.0


# ============================================================
# Factory: build substrate from parameter dictionary
# ============================================================

def substrate_from_params(params):
    """Build substrate energy functions from a parameter dictionary.

    This is the standard entry point for production use.  The dictionary
    is typically loaded from a JSON input file.

    Supported well shapes and required keys:

        'gaussian':  epsilon, sigma, a, b, b1, b2, sub_basis
        'tanh':      epsilon, wd (=ww), a, b, b1, b2, sub_basis
        'sin':       epsilon, ks, sub_basis
        'flat':      no keys required (not even epsilon or sub_basis)

    For 'gaussian' and 'tanh', b1 and b2 are the primitive lattice vectors
    used to build the metric matrices (via calc_matrices_bvect).

    For 'sin', ks should be a pre-computed (n, 2) array (e.g. from get_ks).
    No metric matrices are needed because the plane-wave potential is
    already periodic by construction.

    For 'flat', the substrate is identically zero everywhere -- useful for
    testing the integrator against analytic predictions (drift velocity,
    diffusion coefficient, FDT) without substrate interference.

    All array parameters (basis, u, u_inv, ks) are converted to float64 ONCE
    here and captured in the returned closures.  Callers pass en_inputs=[]
    and call en_func(pos, pos_cm) -- the closure holds everything else.
    This eliminates per-step np.array conversions in run_md.

    Args:
        params: dict with at least 'well_shape' and shape-specific keys
                listed above.  'epsilon' and 'sub_basis' are not required
                for 'flat'.

    Returns:
        pen_func:  particle-level closure (pos, pos_cm) ->
                   (en, F, tau) arrays (per particle).
        en_func:   total-energy closure  (pos, pos_cm) ->
                   (float, (2,) float64, float).
        en_inputs: always [] -- all parameters live in the closures.

    Raises:
        NotImplementedError: if well_shape is not recognised.
    """
    well_shape = params['well_shape']

    if well_shape == 'flat':
        def pen_func(pos, pos_cm):
            N = pos.shape[0]
            return np.zeros(N), np.zeros((N, 2)), np.zeros(N)

        def en_func(pos, pos_cm):
            return _calc_en_flat_core(
                np.asarray(pos,    dtype=np.float64),
                np.asarray(pos_cm, dtype=np.float64))

        en_func._jit_core   = _calc_en_flat_core
        en_func._jit_params = ()
        en_inputs = []
        return pen_func, en_func, en_inputs

    epsilon = params['epsilon']
    basis   = params['sub_basis']

    if well_shape == 'gaussian':
        sigma    = params['sigma']
        a, b     = params['a'], params['b']
        u, u_inv = calc_matrices_bvect(params['b1'], params['b2'])

        _basis   = np.array(basis,  dtype=np.float64).reshape(-1, 2)
        _u       = np.asarray(u,    dtype=np.float64)
        _u_inv   = np.asarray(u_inv, dtype=np.float64)
        _a, _b   = float(a), float(b)
        _sigma   = float(sigma)
        _epsilon = float(epsilon)

        def pen_func(pos, pos_cm):
            return particle_en_gaussian(
                pos, pos_cm, _basis, _a, _b, _sigma, _epsilon, _u, _u_inv)

        def en_func(pos, pos_cm):
            return _calc_en_gaussian_core(
                np.asarray(pos,    dtype=np.float64),
                np.asarray(pos_cm, dtype=np.float64),
                _basis, _a, _b, _sigma, _epsilon, _u, _u_inv)

        en_func._jit_core   = _calc_en_gaussian_core
        en_func._jit_params = (_basis, _a, _b, _sigma, _epsilon, _u, _u_inv)
        en_inputs = []

    elif well_shape == 'tanh':
        ww   = params['wd']   # 'wd' is the historical key; ww in the code
        a, b = params['a'], params['b']
        u, u_inv = calc_matrices_bvect(params['b1'], params['b2'])

        _basis   = np.array(basis,  dtype=np.float64).reshape(-1, 2)
        _u       = np.asarray(u,    dtype=np.float64)
        _u_inv   = np.asarray(u_inv, dtype=np.float64)
        _a, _b   = float(a), float(b)
        _ww      = float(ww)
        _epsilon = float(epsilon)

        def pen_func(pos, pos_cm):
            return particle_en_tanh(
                pos, pos_cm, _basis, _a, _b, _ww, _epsilon, _u, _u_inv)

        def en_func(pos, pos_cm):
            return _calc_en_tanh_core(
                np.asarray(pos,    dtype=np.float64),
                np.asarray(pos_cm, dtype=np.float64),
                _basis, _a, _b, _ww, _epsilon, _u, _u_inv)

        en_func._jit_core   = _calc_en_tanh_core
        en_func._jit_params = (_basis, _a, _b, _ww, _epsilon, _u, _u_inv)
        en_inputs = []

    elif well_shape == 'sin':
        _ks      = np.asarray(params['ks'], dtype=np.float64)
        _basis   = np.array(basis, dtype=np.float64).reshape(-1, 2)
        _epsilon = float(epsilon)

        def pen_func(pos, pos_cm):
            return particle_en_sin(pos, pos_cm, _basis, _ks, _epsilon)

        def en_func(pos, pos_cm):
            return _calc_en_sin_core(
                np.asarray(pos,    dtype=np.float64),
                np.asarray(pos_cm, dtype=np.float64),
                _basis, _ks, _epsilon)

        en_func._jit_core   = _calc_en_sin_core
        en_func._jit_params = (_basis, _ks, _epsilon)
        en_inputs = []

    else:
        raise NotImplementedError(
            "Unknown well shape '%s'.  Supported: 'gaussian', 'tanh', 'sin'."
            % well_shape
        )

    return pen_func, en_func, en_inputs
