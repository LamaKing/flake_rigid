"""
Tests for the extended string_method.py (n-dimensional + scale support).

Substrate: sinusoidal triangular (n=3, epsilon=1, spacing=1) throughout.
Shared geometry mirrors test_maps.py so this file is self-contained.

2D regression tests (1-7) verify that the extended code produces the
same results as the original 2D implementation.

New tests (8-12) cover:
  8.  scaling does not change the barrier (only reparametrization)
  9.  collapsed path raises RuntimeError
  10. 3D MEP returns the correct shapes and 'dim' key
  11. 3D barrier <= 2D barrier for commensurate cluster
  12. fix_ends holds endpoints fixed through manual stepping
"""

import numpy as np
import pytest

from flake.substrate import substrate_from_params, get_ks
from flake.cluster import rotate
from flake.string_method import StringPath, StringPotential, find_mep


# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------

A1 = np.array([1.0, 0.0])
A2 = np.array([-0.5, np.sqrt(3.0) / 2.0])

SIN_PARAMS = {
    'well_shape': 'sin',
    'epsilon':    1.0,
    'sub_basis':  [[0., 0.]],
    'ks':         get_ks(1.0, 3, 4.0 / 3.0, 0.0).tolist(),
}

# 7-particle commensurate cluster.
_COMM_POS = np.array([
    [0., 0.], A1, -A1, A2, -A2, A1 + A2, -(A1 + A2)
], dtype=np.float64)

# Single particle at origin.
SINGLE = np.array([[0., 0.]])


@pytest.fixture(scope='module')
def substrate():
    """Return (calc_en_f, en_params) for the triangular sin substrate."""
    _, calc_en_f, en_params = substrate_from_params(SIN_PARAMS)
    return calc_en_f, en_params


# ---------------------------------------------------------------------------
# 2D regression tests (1-7)
# ---------------------------------------------------------------------------

def test_2d_barrier_positive(substrate):
    """Single particle, 2D MEP: barrier must be positive."""
    calc_en_f, en_params = substrate
    r = find_mep(SINGLE, calc_en_f, en_params,
                 p0=[0., 0.], p1=A1.tolist(),
                 n_pt=30, max_steps=1000, dt=1e-4)
    assert r['barrier'] > 0.0


def test_2d_converges(substrate):
    """Single particle, 2D MEP: must converge with generous settings."""
    calc_en_f, en_params = substrate
    r = find_mep(SINGLE, calc_en_f, en_params,
                 p0=[0., 0.], p1=A1.tolist(),
                 n_pt=50, max_steps=5000, dt=1e-4)
    assert r['converged'], "find_mep did not converge in %d steps" % r['n_steps']


def test_2d_output_keys(substrate):
    """Return dict has all required keys and dim=2."""
    calc_en_f, en_params = substrate
    r = find_mep(SINGLE, calc_en_f, en_params,
                 p0=[0., 0.], p1=A1.tolist(),
                 n_pt=20, max_steps=100)
    expected = {'points', 'energy', 'gradient', 'converged', 'n_steps',
                'barrier', 'dim'}
    assert expected == set(r.keys())
    assert r['dim'] == 2


def test_2d_output_shapes(substrate):
    """Output arrays have shapes consistent with n_pt and dim=2."""
    calc_en_f, en_params = substrate
    n_pt = 17
    r = find_mep(SINGLE, calc_en_f, en_params,
                 p0=[0., 0.], p1=A1.tolist(),
                 n_pt=n_pt, max_steps=100)
    assert r['points'].shape   == (n_pt, 2)
    assert r['energy'].shape   == (n_pt,)
    assert r['gradient'].shape == (n_pt, 2)


def test_2d_barrier_incommensurate_smaller(substrate):
    """Incommensurate cluster (x1.1) has smaller barrier than commensurate."""
    calc_en_f, en_params = substrate
    r_c = find_mep(_COMM_POS, calc_en_f, en_params,
                   p0=[0., 0.], p1=A1.tolist(),
                   n_pt=30, max_steps=1000, dt=1e-4)
    r_i = find_mep(_COMM_POS * 1.1, calc_en_f, en_params,
                   p0=[0., 0.], p1=A1.tolist(),
                   n_pt=30, max_steps=1000, dt=1e-4)
    assert r_i['barrier'] < r_c['barrier'], (
        "Incommensurate %.4f should be < commensurate %.4f"
        % (r_i['barrier'], r_c['barrier'])
    )


def test_2d_string_path_init():
    """StringPath initialises as a straight line; endpoints match p0, p1."""
    p0 = np.array([0., 0.])
    p1 = np.array([1., 0.5])
    path = StringPath(p0, p1, n_pt=11)
    assert np.allclose(path.points[0],  p0, atol=1e-14)
    assert np.allclose(path.points[-1], p1, atol=1e-14)
    assert path.points.shape == (11, 2)


def test_2d_string_path_reparametrize_equal_spacing():
    """After reparametrize on a non-uniformly spaced straight line, spacing is equal.

    Linear interpolation is exact for collinear points, so one call suffices.
    """
    p0 = np.array([0., 0.])
    p1 = np.array([3., 0.])
    path = StringPath(p0, p1, n_pt=5, scale=None)
    # Skew the interior points to non-uniform spacing along the same line.
    path.points = np.array([[0., 0.], [0.1, 0.], [0.5, 0.], [2.5, 0.], [3., 0.]])
    path.reparametrize()
    diffs   = np.diff(path.points, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    assert np.allclose(seg_len, seg_len[0], atol=1e-10)


# ---------------------------------------------------------------------------
# 8. Scaling affects reparametrization but not the barrier
# ---------------------------------------------------------------------------

def test_scaling_affects_path_not_barrier(substrate):
    """barrier with scale=[2.,2.] equals barrier with scale=None within 1e-4."""
    calc_en_f, en_params = substrate
    common = dict(n_pt=30, max_steps=1000, dt=1e-4,
                  p0=[0., 0.], p1=A1.tolist())

    r_no_scale = find_mep(SINGLE, calc_en_f, en_params, scale=None,   **common)
    r_scaled   = find_mep(SINGLE, calc_en_f, en_params, scale=[2., 2.], **common)

    assert abs(r_no_scale['barrier'] - r_scaled['barrier']) < 1e-4, (
        "barrier without scale=%.6f, with scale=[2,2]=%.6f"
        % (r_no_scale['barrier'], r_scaled['barrier'])
    )


# ---------------------------------------------------------------------------
# 9. Collapsed path raises RuntimeError
# ---------------------------------------------------------------------------

def test_string_path_collapse_raises():
    """reparametrize() raises RuntimeError when all points are identical."""
    p0   = np.array([0., 0.])
    p1   = np.array([1., 0.])
    path = StringPath(p0, p1, n_pt=10)
    # Force all points to the same position.
    path.points[:] = path.points[0]
    with pytest.raises(RuntimeError, match="collapsed"):
        path.reparametrize()


# ---------------------------------------------------------------------------
# 10. 3D MEP returns correct shape and dim key
# ---------------------------------------------------------------------------

def test_find_mep_3d_correct_shape(substrate):
    """3D find_mep: points.shape=(n_pt,3), dim=3, gradient.shape=(n_pt,3)."""
    calc_en_f, en_params = substrate
    n_pt = 15
    r = find_mep(SINGLE, calc_en_f, en_params,
                 p0=[0., 0., 0.], p1=[1., 0., 0.],
                 n_pt=n_pt, max_steps=100, dt=1e-4,
                 scale=[1., 1., 60.])

    assert r['points'].shape   == (n_pt, 3)
    assert r['dim']            == 3
    assert r['gradient'].shape == (n_pt, 3)
    assert 'barrier' in r and 'converged' in r


# ---------------------------------------------------------------------------
# 11. 3D barrier <= 2D barrier for commensurate cluster
# ---------------------------------------------------------------------------

def test_find_mep_3d_barrier_leq_2d(substrate):
    """3D MEP barrier is at most the 2D barrier for the commensurate cluster.

    The 3D search space subsumes the 2D search (theta=0 is a feasible path
    in 3D).  For the triangular lattice the torque is zero by symmetry along
    y=0 at theta=0, so the path stays at theta=0 and the barriers are equal.
    """
    calc_en_f, en_params = substrate
    common = dict(n_pt=30, max_steps=500, dt=1e-4)

    r_2d = find_mep(_COMM_POS, calc_en_f, en_params,
                    p0=[0., 0.],       p1=[1., 0.], **common)

    r_3d = find_mep(_COMM_POS, calc_en_f, en_params,
                    p0=[0., 0., 0.],   p1=[1., 0., 0.],
                    scale=[1., 1., 60.], **common)

    assert r_3d['barrier'] <= r_2d['barrier'] + 1e-6, (
        "3D barrier %.6f should be <= 2D barrier %.6f"
        % (r_3d['barrier'], r_2d['barrier'])
    )


# ---------------------------------------------------------------------------
# 12. fix_ends keeps endpoints fixed through manual steps
# ---------------------------------------------------------------------------

def test_fix_ends_respected_after_steps(substrate):
    """After 10 manual string steps, p0 and p1 remain unchanged."""
    calc_en_f, en_params = substrate
    p0 = np.array([0., 0.])
    p1 = A1.copy()

    path      = StringPath(p0, p1, n_pt=20, fix_ends=True)
    potential = StringPotential(SINGLE, calc_en_f, en_params)

    for _ in range(10):
        _, gradients = potential.evaluate(path.points)
        path.step(gradients, dt=1e-4)

    assert np.allclose(path.points[0],  p0, atol=1e-10), (
        "First point moved: %s" % path.points[0]
    )
    assert np.allclose(path.points[-1], p1, atol=1e-10), (
        "Last point moved: %s" % path.points[-1]
    )
