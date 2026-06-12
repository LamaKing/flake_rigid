"""
Tests for tool_create_cluster.py.

All tests use a1=[1,0], a2=[-0.5, sqrt(3)/2] (triangular lattice, spacing 1)
unless the test specifically needs a different lattice.  N1=N2=5 is used for
shapes that accept general N; parallelogram tests use N1=N2=9 (sqrt(81)=9,
odd -- valid) or N1=N2=4 (sqrt(16)=4, even -- invalid).
"""

import tempfile
import os

import numpy as np
import pytest

from flake.cluster import (
    make_cluster,
    rotate,
    save_cluster,
    load_cluster,
    add_basis,
    calc_cluster_langevin,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

A1 = np.array([1.0, 0.0])
A2 = np.array([-0.5, np.sqrt(3.0) / 2.0])

SHAPES_GENERAL = ['circle', 'hexagon', 'rectangle', 'triangle', 'parallelogram']


@pytest.fixture
def tri_lat():
    """Triangular lattice vectors, spacing 1."""
    return A1.copy(), A2.copy()


# ---------------------------------------------------------------------------
# 1. CM at origin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('shape,n1,n2', [
    ('circle',        5, 5),
    ('hexagon',       5, 5),
    ('rectangle',     5, 5),
    ('triangle',      5, 5),
    ('parallelogram', 5, 5),   # sqrt(25) = 5 (odd): valid
])
def test_cm_at_origin(tri_lat, shape, n1, n2):
    """Every shape must return a cluster with CM at the origin."""
    a1, a2 = tri_lat
    pos = make_cluster(a1, a2, n1, n2, shape=shape)
    cm  = np.mean(pos, axis=0)
    assert abs(cm[0]) < 1e-10, "CM_x = %.4e for shape '%s'" % (cm[0], shape)
    assert abs(cm[1]) < 1e-10, "CM_y = %.4e for shape '%s'" % (cm[1], shape)


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('shape,n1,n2', [
    ('circle',        5, 5),
    ('hexagon',       5, 5),
    ('rectangle',     5, 5),
    ('triangle',      5, 5),
    ('parallelogram', 5, 5),
    ('ellipse',       5, 5),
])
def test_output_shape(tri_lat, shape, n1, n2):
    """make_cluster must return a 2-D array with shape (N, 2) for some N > 0."""
    a1, a2 = tri_lat
    pos = make_cluster(a1, a2, n1, n2, shape=shape)
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    assert pos.shape[0] > 0


# ---------------------------------------------------------------------------
# 3. rotate -- identity
# ---------------------------------------------------------------------------

def test_rotate_identity(tri_lat):
    """Rotating by 360 degrees must recover the original positions."""
    a1, a2 = tri_lat
    pos = make_cluster(a1, a2, 5, 5, shape='circle')
    assert np.allclose(rotate(pos, 360.0), pos, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. rotate -- 90 degrees
# ---------------------------------------------------------------------------

def test_rotate_90():
    """rotate([[1,0]], 90) should give [[0,1]]."""
    result = rotate([[1.0, 0.0]], 90.0)
    assert result.shape == (1, 2)
    assert abs(result[0, 0] - 0.0) < 1e-10
    assert abs(result[0, 1] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 5. parallelogram -- validation
# ---------------------------------------------------------------------------

def test_parallelogram_valid_N(tri_lat):
    """sqrt(N1*N2) = 9 (odd) should succeed; sqrt=4 (even) should raise."""
    a1, a2 = tri_lat
    # Valid: 9*9 = 81, sqrt = 9 (odd)
    pos = make_cluster(a1, a2, 9, 9, shape='parallelogram')
    assert pos.shape == (81, 2)

    # Invalid: 4*4 = 16, sqrt = 4 (even)
    with pytest.raises(ValueError):
        make_cluster(a1, a2, 4, 4, shape='parallelogram')


# ---------------------------------------------------------------------------
# 6. parallelogram -- particle at origin
# ---------------------------------------------------------------------------

def test_parallelogram_particle_at_origin(tri_lat):
    """The j1=j2=0 lattice site must survive and be at (0,0) after centering."""
    a1, a2 = tri_lat
    pos    = make_cluster(a1, a2, 9, 9, shape='parallelogram')
    dists  = np.linalg.norm(pos, axis=1)
    # Exactly one particle should be at the origin.
    assert np.sum(dists < 1e-12) == 1


# ---------------------------------------------------------------------------
# 7. circle -- all particles within expected radius
# ---------------------------------------------------------------------------

def test_circle_radius(tri_lat):
    """All particles must lie strictly inside the design radius."""
    a1, a2 = tri_lat
    N1, N2 = 5, 5
    cell_area = abs(a1[0]*a2[1] - a1[1]*a2[0])
    R         = np.sqrt(N1 * N2 * cell_area / np.pi)
    pos       = make_cluster(a1, a2, N1, N2, shape='circle')
    dists     = np.linalg.norm(pos, axis=1)
    # Allow a small tolerance for particles right on the boundary.
    assert np.all(dists < R * (1 + 1e-6)), (
        "Particle(s) outside design radius R=%.4f: max dist=%.4f" % (R, dists.max())
    )


# ---------------------------------------------------------------------------
# 8. ellipse -- all particles inside ellipse
# ---------------------------------------------------------------------------

def test_ellipse_bounds(tri_lat):
    """All particles must satisfy (x/rx)^2 + (y/ry)^2 < 1, with rx=N1*|a1|, ry=N2*|a2|."""
    a1, a2 = tri_lat
    N1, N2 = 6, 4
    pos    = make_cluster(a1, a2, N1, N2, shape='ellipse')
    rx     = N1 * np.linalg.norm(a1)
    ry     = N2 * np.linalg.norm(a2)
    xi     = (pos[:, 0] / rx)**2 + (pos[:, 1] / ry)**2
    assert np.all(xi < 1.0 + 1e-9), (
        "Particle(s) outside ellipse: max xi=%.6f" % xi.max()
    )


# ---------------------------------------------------------------------------
# 9. add_basis
# ---------------------------------------------------------------------------

def test_add_basis():
    """A lattice of 4 points with a 2-atom basis must give 8 points total."""
    # Simple 2x2 square lattice.
    lat = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    basis = np.array([[0.0, 0.0], [0.25, 0.25]])
    result = add_basis(lat, basis)
    assert result.shape == (8, 2)


# ---------------------------------------------------------------------------
# 10. save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tri_lat):
    """save_cluster then load_cluster must recover positions within 1e-14."""
    a1, a2 = tri_lat
    pos    = make_cluster(a1, a2, 5, 5, shape='circle')

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        fname = f.name
    try:
        save_cluster(pos, fname)
        pos2 = load_cluster(fname)
        assert np.allclose(pos, pos2, atol=1e-14)
    finally:
        # Clean up both the written file and any .npy extension variant.
        for path in (fname, fname.replace('.npy', '') + '.npy'):
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# 11. calc_cluster_langevin
# ---------------------------------------------------------------------------

def test_calc_cluster_langevin():
    """N particles on a ring of radius R: check etat and etar."""
    N   = 8
    R   = 3.0
    eta = 1.5
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos    = np.column_stack([R * np.cos(angles), R * np.sin(angles)])

    etat_eff, etar_eff = calc_cluster_langevin(eta, pos)

    assert abs(etat_eff - eta * N)        < 1e-12
    assert abs(etar_eff - eta * N * R**2) < 1e-10
