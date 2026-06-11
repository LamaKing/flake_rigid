"""
Tests for tool_create_substrate.py.

All tests use a square unit cell (b1=[1,0], b2=[0,1]) and a single-site
basis ([[0,0]]) unless the test specifically requires triangular geometry.
The metric matrices for a unit square are trivially the identity, so nearest-
image positions equal the raw displacement modulo 1.

Convention reminder: force = -grad V.  Any test that checks force direction
implicitly verifies this sign convention.
"""

import numpy as np
import pytest

from flake.substrate import (
    calc_matrices_bvect,
    particle_en_gaussian,
    calc_en_gaussian,
    particle_en_tanh,
    particle_en_sin,
    get_ks,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B1 = np.array([1.0, 0.0])
B2 = np.array([0.0, 1.0])
BASIS = [[0.0, 0.0]]

EPSILON = 1.5
SIGMA   = 0.2
# a > sigma so that the test point sigma/2 lies in the bulk region.
A_GAUSS = 0.4
B_GAUSS = 0.5


@pytest.fixture
def sq_metric():
    return calc_matrices_bvect(B1, B2)


# ---------------------------------------------------------------------------
# Gaussian tests
# ---------------------------------------------------------------------------

def test_gaussian_single_particle_minimum(sq_metric):
    """Particle sitting exactly at a potential minimum: E=-epsilon, F=(0,0)."""
    u, u_inv = sq_metric
    pos        = np.array([[0.0, 0.0]])
    pos_torque = np.array([0.0, 0.0])

    en, F, tau = particle_en_gaussian(
        pos, pos_torque, BASIS, A_GAUSS, B_GAUSS, SIGMA, EPSILON, u, u_inv
    )

    assert abs(en[0] - (-EPSILON)) < 1e-12
    assert abs(F[0, 0]) < 1e-12
    assert abs(F[0, 1]) < 1e-12


def test_gaussian_force_direction(sq_metric):
    """Particle displaced along +x should feel a force toward the origin (-x)."""
    u, u_inv = sq_metric
    # a > sigma ensures sigma/2 is in the bulk (no smoothstep damping).
    assert A_GAUSS > SIGMA
    x_test = SIGMA / 2.0
    pos        = np.array([[x_test, 0.0]])
    pos_torque = np.array([0.0, 0.0])

    _, F, _ = particle_en_gaussian(
        pos, pos_torque, BASIS, A_GAUSS, B_GAUSS, SIGMA, EPSILON, u, u_inv
    )

    # Force must be in the -x direction and y component must be zero.
    assert F[0, 0] < 0.0, (
        "F_x should be negative (toward origin) but got %g" % F[0, 0]
    )
    assert abs(F[0, 1]) < 1e-12


def test_gaussian_force_equals_minus_grad_energy(sq_metric):
    """Analytical force must equal -dE/dr up to finite-difference accuracy."""
    u, u_inv = sq_metric
    pos0       = np.array([[0.3, 0.1]])
    pos_torque = np.array([0.0, 0.0])
    h          = 1e-6

    def energy_at(pos):
        en, _, _ = particle_en_gaussian(
            pos, pos_torque, BASIS, A_GAUSS, B_GAUSS, SIGMA, EPSILON, u, u_inv
        )
        return en[0]

    _, F, _ = particle_en_gaussian(
        pos0, pos_torque, BASIS, A_GAUSS, B_GAUSS, SIGMA, EPSILON, u, u_inv
    )

    # Central difference for x and y.
    dEdx = (energy_at(pos0 + [[h, 0]]) - energy_at(pos0 - [[h, 0]])) / (2 * h)
    dEdy = (energy_at(pos0 + [[0, h]]) - energy_at(pos0 - [[0, h]])) / (2 * h)

    assert abs(F[0, 0] - (-dEdx)) < 1e-5
    assert abs(F[0, 1] - (-dEdy)) < 1e-5


def test_gaussian_N_particles_at_minimum(sq_metric):
    """N particles all at the potential minimum: E_total = N*(-epsilon), F_total = 0."""
    u, u_inv   = sq_metric
    N          = 7
    pos        = np.zeros((N, 2))
    pos_torque = np.array([0.0, 0.0])

    en_total, F_total, _ = calc_en_gaussian(
        pos, pos_torque, BASIS, A_GAUSS, B_GAUSS, SIGMA, EPSILON, u, u_inv
    )

    assert abs(en_total - N * (-EPSILON)) < 1e-10
    assert abs(F_total[0]) < 1e-12
    assert abs(F_total[1]) < 1e-12


# ---------------------------------------------------------------------------
# Tanh tests
# ---------------------------------------------------------------------------

def test_tanh_flat_bottom(sq_metric):
    """Particle inside the flat-bottom region: E=-epsilon, F=(0,0)."""
    u, u_inv = sq_metric
    a_tanh  = 0.3
    b_tanh  = 0.5
    ww      = 0.3
    # Place particle well inside the flat bottom.
    r_inner = a_tanh * 0.5
    pos        = np.array([[r_inner, 0.0]])
    pos_torque = np.array([0.0, 0.0])

    en, F, _ = particle_en_tanh(
        pos, pos_torque, BASIS, a_tanh, b_tanh, ww, EPSILON, u, u_inv
    )

    assert abs(en[0] - (-EPSILON)) < 1e-12
    assert abs(F[0, 0]) < 1e-12
    assert abs(F[0, 1]) < 1e-12


# ---------------------------------------------------------------------------
# Sinusoidal tests
# ---------------------------------------------------------------------------

def test_sin_triangular_energy_minimum():
    """Triangular sin substrate: energy at lattice site (0,0) = -epsilon."""
    R       = 1.0
    n       = 3
    c_n     = 4.0 / 3.0
    alpha_n = 0.0
    ks = get_ks(R, n, c_n, alpha_n)

    pos        = np.array([[0.0, 0.0]])
    pos_torque = np.array([0.0, 0.0])

    en, _, _ = particle_en_sin(pos, pos_torque, BASIS, ks, EPSILON)

    assert abs(en[0] - (-EPSILON)) < 1e-12


# ---------------------------------------------------------------------------
# Torque test (substrate-type agnostic)
# ---------------------------------------------------------------------------

def test_torque_zero_single_particle_at_cm(sq_metric):
    """Torque about a point is zero when the single particle sits on that point."""
    u, u_inv   = sq_metric
    # Displace from the well centre to get a non-zero force; the torque arm
    # (pos - pos_torque) is still (0,0) because pos == pos_torque.
    pos_torque = np.array([0.1, 0.2])
    pos        = pos_torque.reshape(1, 2).copy()

    _, _, tau = particle_en_gaussian(
        pos, pos_torque, BASIS, A_GAUSS, B_GAUSS, SIGMA, EPSILON, u, u_inv
    )

    assert abs(tau[0]) < 1e-12


# ---------------------------------------------------------------------------
# Consistency test: total == sum of particle arrays
# ---------------------------------------------------------------------------

def test_total_equals_sum_of_particles(sq_metric):
    """calc_en_gaussian must equal the sums of the particle_en_gaussian arrays."""
    rng      = np.random.default_rng(42)
    u, u_inv = sq_metric
    N        = 5
    pos      = rng.uniform(-0.4, 0.4, size=(N, 2))
    pos_torque = np.array([0.0, 0.0])

    en_p, F_p, tau_p = particle_en_gaussian(
        pos, pos_torque, BASIS, A_GAUSS, B_GAUSS, SIGMA, EPSILON, u, u_inv
    )
    en_t, F_t, tau_t = calc_en_gaussian(
        pos, pos_torque, BASIS, A_GAUSS, B_GAUSS, SIGMA, EPSILON, u, u_inv
    )

    assert abs(en_t - np.sum(en_p)) < 1e-12
    assert abs(F_t[0] - np.sum(F_p[:, 0])) < 1e-12
    assert abs(F_t[1] - np.sum(F_p[:, 1])) < 1e-12
    assert abs(tau_t - np.sum(tau_p)) < 1e-12
