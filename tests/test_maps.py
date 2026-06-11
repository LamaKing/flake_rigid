"""
Tests for maps.py and string_method.py.

Substrate: sinusoidal triangular (n=3, c_n=4/3, alpha_n=0, spacing=1, epsilon=1).
The minimum of this potential is at the origin and at every triangular
lattice site.

Clusters are built explicitly here without importing tool_create_cluster,
so this file remains independent of the cluster module.

Triangular lattice (spacing 1):
    a1 = [1, 0]
    a2 = [-0.5, sqrt(3)/2]

7-particle commensurate cluster (center + 6 nearest neighbours):
    [0,0], ±a1, ±a2, ±(a1+a2)
    CM is at origin by symmetry.
"""

import numpy as np
import pytest

from flake.substrate import substrate_from_params, get_ks, calc_matrices_bvect
from flake.cluster import rotate
from flake.maps import translational_map, rotational_map, rototrasl_map
from flake.string_method import find_mep

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EPSILON = 1.0
R_SUB   = 1.0

A1 = np.array([1.0, 0.0])
A2 = np.array([-0.5, np.sqrt(3.0) / 2.0])

# 7-particle cluster: center + 6 nearest neighbours on the same lattice.
_COMM_POS = np.array([
    [0.0, 0.0],
     A1,   -A1,
     A2,   -A2,
     A1 + A2, -(A1 + A2),
], dtype=np.float64)

# Incommensurate: stretch all positions by 1.1 (CM stays at origin).
_INCOMM_POS = _COMM_POS * 1.1

SIN_PARAMS = {
    'well_shape': 'sin',
    'epsilon':    EPSILON,
    'sub_basis':  [[0.0, 0.0]],
    'ks':         get_ks(R_SUB, 3, 4.0 / 3.0, 0.0).tolist(),
}

# Single-particle cluster at origin.
SINGLE = np.array([[0.0, 0.0]])


@pytest.fixture(scope='module')
def substrate():
    """Return (calc_en_f, en_params) for the triangular sin substrate."""
    _, calc_en_f, en_params = substrate_from_params(SIN_PARAMS)
    return calc_en_f, en_params


@pytest.fixture(scope='module')
def sub_u_inv():
    """u_inv for the triangular lattice (a1=[1,0], a2=[-0.5, sqrt(3)/2])."""
    _, u_inv = calc_matrices_bvect(A1, A2)
    return u_inv


# ---------------------------------------------------------------------------
# 1. Single particle at substrate minimum
# ---------------------------------------------------------------------------

def test_single_particle_at_minimum(substrate):
    calc_en_f, en_params = substrate
    r = translational_map(SINGLE, calc_en_f, en_params, np.eye(2),
                          n_x=1, n_y=1,
                          frac_x=(0., 0.), frac_y=(0., 0.))
    assert abs(r['energy'][0] - (-EPSILON)) < 1e-12


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

def test_single_particle_trasl_map_shape(substrate):
    calc_en_f, en_params = substrate
    r = translational_map(SINGLE, calc_en_f, en_params, np.eye(2),
                          n_x=5, n_y=5)
    assert r['pos_cm'].shape == (25, 2)
    assert r['energy'].shape == (25,)
    assert r['force'].shape  == (25, 2)
    assert r['torque'].shape == (25,)


# ---------------------------------------------------------------------------
# 3. Force direction from map
# ---------------------------------------------------------------------------

def test_single_particle_force_from_map(substrate):
    calc_en_f, en_params = substrate
    # At origin: force must vanish.
    r0 = translational_map(SINGLE, calc_en_f, en_params, np.eye(2),
                           n_x=1, n_y=1,
                           frac_x=(0., 0.), frac_y=(0., 0.))
    assert abs(r0['force'][0, 0]) < 1e-12
    assert abs(r0['force'][0, 1]) < 1e-12

    # At (0.1, 0): force must point toward origin (-x direction).
    dx = 0.1
    r1 = translational_map(SINGLE, calc_en_f, en_params, np.eye(2),
                           n_x=1, n_y=1,
                           frac_x=(dx, dx), frac_y=(0., 0.))
    assert r1['force'][0, 0] < 0.0, (
        "F_x should be negative at (%.2f,0), got %.6f" % (dx, r1['force'][0, 0])
    )


# ---------------------------------------------------------------------------
# 4. Translational map periodicity
# ---------------------------------------------------------------------------

def test_translational_map_periodicity(substrate, sub_u_inv):
    """E(0,0) == E(a1) == E(a2) for the triangular substrate unit cell."""
    calc_en_f, en_params = substrate

    def energy_at_frac(f1, f2):
        r = translational_map(SINGLE, calc_en_f, en_params, sub_u_inv,
                              n_x=1, n_y=1,
                              frac_x=(f1, f1), frac_y=(f2, f2))
        return r['energy'][0]

    e00 = energy_at_frac(0., 0.)
    e10 = energy_at_frac(1., 0.)
    e01 = energy_at_frac(0., 1.)

    assert abs(e00 - e10) < 1e-10
    assert abs(e00 - e01) < 1e-10


# ---------------------------------------------------------------------------
# 5. Commensurate cluster energy at minimum
# ---------------------------------------------------------------------------

def test_commensurate_cluster_matches_single_particle_energy(substrate):
    """7 commensurate particles at pos_cm=0 must give E = 7 * (-epsilon)."""
    calc_en_f, en_params = substrate
    r = translational_map(_COMM_POS, calc_en_f, en_params, np.eye(2),
                          n_x=1, n_y=1,
                          frac_x=(0., 0.), frac_y=(0., 0.))
    expected = 7 * (-EPSILON)
    assert abs(r['energy'][0] - expected) < 1e-10, (
        "E = %.6f, expected %.6f" % (r['energy'][0], expected)
    )


# ---------------------------------------------------------------------------
# 6. Commensurate vs incommensurate barrier (translational)
# ---------------------------------------------------------------------------

def test_commensurate_cluster_barrier_vs_incommensurate(substrate):
    """Incommensurate cluster barrier must be strictly less than commensurate."""
    calc_en_f, en_params = substrate
    common_kwargs = dict(n_x=12, n_y=12,
                         frac_x=(0., 1.), frac_y=(0., 1.))

    r_c = translational_map(_COMM_POS,   calc_en_f, en_params, np.eye(2), **common_kwargs)
    r_i = translational_map(_INCOMM_POS, calc_en_f, en_params, np.eye(2), **common_kwargs)

    barrier_c = r_c['energy'].max() - r_c['energy'].min()
    barrier_i = r_i['energy'].max() - r_i['energy'].min()

    assert barrier_i < barrier_c, (
        "Incommensurate barrier %.4f should be < commensurate %.4f"
        % (barrier_i, barrier_c)
    )


# ---------------------------------------------------------------------------
# 7. Commensurate vs rotated barrier
# ---------------------------------------------------------------------------

def test_commensurate_cluster_barrier_vs_rotated(substrate):
    """Cluster rotated by 15 degrees should have smaller translational barrier."""
    calc_en_f, en_params = substrate
    common_kwargs = dict(n_x=12, n_y=12,
                         frac_x=(0., 1.), frac_y=(0., 1.))

    pos_rot = rotate(_COMM_POS, 15.0)

    r_aligned = translational_map(_COMM_POS, calc_en_f, en_params, np.eye(2), **common_kwargs)
    r_rotated = translational_map(pos_rot,   calc_en_f, en_params, np.eye(2), **common_kwargs)

    barrier_aligned = r_aligned['energy'].max() - r_aligned['energy'].min()
    barrier_rotated = r_rotated['energy'].max() - r_rotated['energy'].min()

    assert barrier_rotated < barrier_aligned, (
        "Rotated barrier %.4f should be < aligned %.4f"
        % (barrier_rotated, barrier_aligned)
    )


# ---------------------------------------------------------------------------
# 8. Rotational map shape
# ---------------------------------------------------------------------------

def test_rotational_map_shape(substrate):
    calc_en_f, en_params = substrate
    n_theta = 7
    r = rotational_map(SINGLE, calc_en_f, en_params,
                       np.linspace(0., 60., n_theta))
    assert r['theta'].shape  == (n_theta,)
    assert r['energy'].shape == (n_theta,)
    assert r['force'].shape  == (n_theta, 2)
    assert r['torque'].shape == (n_theta,)


# ---------------------------------------------------------------------------
# 9. Rotational map periodicity (6-fold cluster on 6-fold substrate)
# ---------------------------------------------------------------------------

def test_rotational_map_periodicity(substrate):
    """7-particle cluster: E(0 deg) == E(60 deg) from 6-fold symmetry."""
    calc_en_f, en_params = substrate
    # Include 0 and 60 as the first and last points.
    theta = np.linspace(0., 60., 13)
    r = rotational_map(_COMM_POS, calc_en_f, en_params, theta)

    e_0deg  = r['energy'][0]
    e_60deg = r['energy'][-1]

    assert abs(e_0deg - e_60deg) < 1e-8, (
        "E(0 deg)=%.8f, E(60 deg)=%.8f, diff=%.2e"
        % (e_0deg, e_60deg, abs(e_0deg - e_60deg))
    )


# ---------------------------------------------------------------------------
# 10. Roto-translational map shape
# ---------------------------------------------------------------------------

def test_rototrasl_map_shape(substrate):
    calc_en_f, en_params = substrate
    r = rototrasl_map(SINGLE, calc_en_f, en_params, np.eye(2),
                      theta_deg=np.linspace(0., 30., 3),
                      n_x=4, n_y=4)
    assert r['energy'].shape == (3, 16)
    assert r['force'].shape  == (3, 16, 2)
    assert r['torque'].shape == (3, 16)
    assert r['pos_cm'].shape == (3, 16, 2)


# ---------------------------------------------------------------------------
# 11. MEP barrier is positive
# ---------------------------------------------------------------------------

def test_find_mep_barrier_positive(substrate):
    """MEP for single particle between two adjacent minima: barrier > 0."""
    calc_en_f, en_params = substrate
    p0 = np.array([0.0, 0.0])
    p1 = A1.copy()  # next minimum at [1, 0]
    result = find_mep(SINGLE, calc_en_f, en_params, p0, p1,
                      n_pt=30, max_steps=1000, dt=1e-4)
    assert result['barrier'] > 0.0


# ---------------------------------------------------------------------------
# 12. MEP converges with generous settings
# ---------------------------------------------------------------------------

def test_find_mep_converged(substrate):
    """With n_pt=50 and max_steps=5000 the MEP should converge."""
    calc_en_f, en_params = substrate
    p0 = np.array([0.0, 0.0])
    p1 = A1.copy()
    result = find_mep(SINGLE, calc_en_f, en_params, p0, p1,
                      n_pt=50, max_steps=5000, dt=1e-4)
    assert result['converged'], (
        "find_mep did not converge in %d steps" % result['n_steps']
    )


# ---------------------------------------------------------------------------
# 13. MEP barrier: incommensurate cluster smaller than commensurate
# ---------------------------------------------------------------------------

def test_find_mep_barrier_incommensurate_smaller(substrate):
    """7-particle incommensurate cluster (R=1.1) must have smaller MEP barrier."""
    calc_en_f, en_params = substrate
    p0 = np.array([0.0, 0.0])
    p1 = A1.copy()

    mep_c = find_mep(_COMM_POS,   calc_en_f, en_params, p0, p1,
                     n_pt=30, max_steps=1000, dt=1e-4)
    mep_i = find_mep(_INCOMM_POS, calc_en_f, en_params, p0, p1,
                     n_pt=30, max_steps=1000, dt=1e-4)

    assert mep_i['barrier'] < mep_c['barrier'], (
        "Incommensurate barrier %.4f should be < commensurate %.4f"
        % (mep_i['barrier'], mep_c['barrier'])
    )
