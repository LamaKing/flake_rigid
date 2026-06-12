#!/usr/bin/env python3
"""
Cluster creation utilities for rigid-cluster simulations.

A cluster is always an (N, 2) float64 ndarray of (x, y) positions with the
center of mass at the origin.  Nothing else -- no (N,6) legacy format, no
hidden state, no file intermediaries in the shape functions.

The cluster is built from a Bravais lattice defined by two primitive vectors
a1, a2 (each shape (2,)).  N1, N2 count unit cells along each direction and
control the cluster size.  The precise meaning of N1, N2 depends on shape:

    circle       : encloses N1*N2 particles at correct areal density
    hexagon      : N1 particles per edge of the hexagon
    rectangle    : N1 (N2) cells along the a1 (a2) direction
    triangle     : triangle spanned by N1*a1, N2*a2
    parallelogram: sqrt(N1*N2) x sqrt(N1*N2) lattice (sqrt must be odd int)
    ellipse      : semi-axes rx = N1*|a1|, ry = N2*|a2|

Units follow the calling code; no unit conversion is done here.

ASE and Shapely are optional dependencies -- their imports are guarded and
raise ImportError with installation hints if absent.
"""

import numpy as np
from numpy import pi, sqrt

# Small tolerance for boundary conditions in shape functions.
# Particles landing exactly on the boundary due to floating-point
# rounding are excluded consistently -- avoids asymmetric clusters
# in the commensurate case where particles can sit at exactly R.
_BOUNDARY_TOL = 1e-10

# ============================================================
# Rotation
# ============================================================

def rotate(pos, angle_deg, center=(0, 0)):
    """Rotate positions counter-clockwise by angle_deg degrees about center.

    Args:
        pos:       (N, 2) array-like -- input positions.
        angle_deg: float             -- rotation angle in degrees (ACW positive).
        center:    (2,) array-like   -- rotation center (default origin).

    Returns:
        (N, 2) float64 ndarray -- rotated positions.
    """
    pos    = np.asarray(pos, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    theta  = angle_deg * pi / 180.0
    c, s   = np.cos(theta), np.sin(theta)
    R      = np.array([[c, -s], [s, c]])
    # Translate to origin, rotate, translate back.
    return (R @ (pos - center).T).T + center


# ============================================================
# Shape-specific internal builders
# ============================================================

def _make_cluster_circle(a1, a2, N1, N2):
    """Build a circular cluster of approximately N1*N2 particles.

    Radius R = sqrt(N1*N2*|a1 x a2|/pi) so that the disk area equals the
    area of N1*N2 unit cells, giving the correct areal density regardless of
    lattice geometry.

    Args:
        a1, a2: (2,) float64 -- primitive lattice vectors.
        N1, N2: int          -- grid repetitions (control total particle count).

    Returns:
        (N, 2) float64 ndarray, CM at origin.
    """
    cell_area = abs(a1[0]*a2[1] - a1[1]*a2[0])
    R2        = N1 * N2 * cell_area / pi  # R^2; avoids the sqrt in the loop
    M         = 2 * (N1 + N2)

    pts = []
    for i in range(-M, M + 1):
        for j in range(-M, M + 1):
            x = j*a1[0] + i*a2[0]
            y = j*a1[1] + i*a2[1]
            if x*x + y*y < R2 - _BOUNDARY_TOL:
                pts.append([x, y])

    pos = np.array(pts, dtype=np.float64)
    pos -= np.mean(pos, axis=0)
    return pos


def _make_cluster_hexagon(a1, a2, N1, N2):
    """Build a hexagonal cluster with N1 particles per edge.

    The index-space loop follows the shape of a 2D hexagon aligned with the
    a1 direction.  Works for any lattice but is most naturally a regular
    hexagon for the triangular lattice.

    Args:
        a1, a2: (2,) float64 -- primitive lattice vectors.
        N1, N2: int          -- N1 controls the hexagon side length.

    Returns:
        (N, 2) float64 ndarray, CM at origin.
    """
    pts = []
    # Lower half (including equator): each row gains one extra particle.
    for i in range(N2):
        for j in range(N1 + i):
            pts.append([j*a1[0] + i*a2[0], j*a1[1] + i*a2[1]])
    # Upper half: each row loses one particle.
    for i in range(N2, N2 + N1 - 1):
        for j in range(N1 + N2 - 2, i - N2, -1):
            pts.append([j*a1[0] + i*a2[0], j*a1[1] + i*a2[1]])

    pos = np.array(pts, dtype=np.float64)
    pos -= np.mean(pos, axis=0)
    return pos


def _make_cluster_rectangle(a1, a2, N1, N2):
    """Build a rectangular cluster fitting N1 x N2 unit cells.

    Includes lattice points with |x| < N1/2 * |a1| and |y| < N2/2 * |a2|.
    This is an exact rectangle only for orthogonal lattices; for oblique
    lattices it is an approximation of a rectangular region.

    Args:
        a1, a2: (2,) float64 -- primitive lattice vectors.
        N1, N2: int          -- half-widths in units of |a1|, |a2|.

    Returns:
        (N, 2) float64 ndarray, CM at origin.
    """
    a1norm = np.linalg.norm(a1)
    a2norm = np.linalg.norm(a2)
    xlim   = N1 / 2.0 * a1norm
    ylim   = N2 / 2.0 * a2norm
    M      = 2 * (N1 + N2)

    pts = []
    for i in range(-M, M + 1):
        for j in range(-M, M + 1):
            x = j*a1[0] + i*a2[0]
            y = j*a1[1] + i*a2[1]
            if abs(x) < xlim - _BOUNDARY_TOL and abs(y) < ylim - _BOUNDARY_TOL:
                pts.append([x, y])

    pos = np.array(pts, dtype=np.float64)
    pos -= np.mean(pos, axis=0)
    return pos


def _make_cluster_triangle(a1, a2, N1, N2):
    """Build a triangular cluster using barycentric coordinates.

    The triangle has vertices N1*a1, N2*a2, and -(N1*a1 + N2*a2).
    Barycentric filtering ensures a clean triangular boundary for any lattice.

    Args:
        a1, a2: (2,) float64 -- primitive lattice vectors.
        N1, N2: int          -- scale the triangle vertices.

    Returns:
        (N, 2) float64 ndarray, CM at origin.
    """
    x1, y1 =  N1 * a1
    x2, y2 =  N2 * a2
    x3, y3 = -(N1*a1 + N2*a2)
    denom  = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    M      = 2 * (N1 + N2)

    pts = []
    for i in range(-M, M + 1):
        for j in range(-M, M + 1):
            x = j*a1[0] + i*a2[0]
            y = j*a1[1] + i*a2[1]
            # Barycentric coordinates of (x,y) w.r.t. the three vertices.
            ba = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
            bb = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
            bc = 1.0 - ba - bb
            if 0.0 < ba < 1.0 and 0.0 < bb < 1.0 and 0.0 < bc < 1.0:
                pts.append([x, y])

    pos = np.array(pts, dtype=np.float64)
    pos -= np.mean(pos, axis=0)
    return pos


def _make_cluster_parallelogram(a1, a2, N1, N2):
    """Build the special parallelogram from Nanoscale 2022 (Section V.A, eq. 16).

    Positions: R_j = j1*a1 + j2*a2  with
        j1, j2 in {-(n-1)/2, ..., 0, ..., (n-1)/2},  n = sqrt(N1*N2).

    n must be an odd integer; otherwise the lattice lacks the j=0 site that
    centres the CM at the origin.  The shape admits a closed-form weight
    function W that makes scaling-law analysis tractable.

    Args:
        a1, a2: (2,) float64 -- primitive lattice vectors.
        N1, N2: int          -- N1*N2 is the total number of particles.

    Returns:
        (N1*N2, 2) float64 ndarray, CM at origin.

    Raises:
        ValueError: if sqrt(N1*N2) is not an odd integer.
    """
    N    = N1 * N2
    sqN  = sqrt(float(N))
    n    = int(round(sqN))
    if abs(sqN - n) > 1e-9 or n % 2 == 0:
        raise ValueError(
            "Parallelogram requires sqrt(N1*N2) to be an odd integer; "
            "got N1=%d, N2=%d => sqrt(N1*N2) = %.6g." % (N1, N2, sqN)
        )

    half = (n - 1) // 2
    pts  = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            pts.append([j*a1[0] + i*a2[0], j*a1[1] + i*a2[1]])

    pos = np.array(pts, dtype=np.float64)
    # CM is exactly zero by symmetry (sum of j over symmetric range = 0),
    # but subtract mean anyway to absorb floating-point rounding.
    pos -= np.mean(pos, axis=0)
    return pos


def _make_cluster_ellipse(a1, a2, N1, N2):
    """Build an elliptical cluster with semi-axes rx = N1*|a1|, ry = N2*|a2|.

    Includes all lattice points satisfying (x/rx)^2 + (y/ry)^2 < 1.
    This is consistent with how N1, N2 scale the cluster for other shapes.

    Args:
        a1, a2: (2,) float64 -- primitive lattice vectors.
        N1, N2: int          -- semi-axes in units of |a1| and |a2|.

    Returns:
        (N, 2) float64 ndarray, CM at origin.
    """
    rx  = N1 * np.sqrt(a1[0]*a1[0] + a1[1]*a1[1])
    ry  = N2 * np.sqrt(a2[0]*a2[0] + a2[1]*a2[1])
    M   = 2 * (N1 + N2)
    rx2 = rx * rx
    ry2 = ry * ry

    pts = []
    for i in range(-M, M + 1):
        for j in range(-M, M + 1):
            x = j*a1[0] + i*a2[0]
            y = j*a1[1] + i*a2[1]
            if x*x / rx2 + y*y / ry2 < 1.0 - _BOUNDARY_TOL:
                pts.append([x, y])

    pos = np.array(pts, dtype=np.float64)
    pos -= np.mean(pos, axis=0)
    return pos


# ============================================================
# Public factory
# ============================================================

_SHAPE_FUNCS = {
    'circle':        _make_cluster_circle,
    'hexagon':       _make_cluster_hexagon,
    'rectangle':     _make_cluster_rectangle,
    'triangle':      _make_cluster_triangle,
    'parallelogram': _make_cluster_parallelogram,
    'ellipse':       _make_cluster_ellipse,
}


def make_cluster(a1, a2, N1, N2, shape='circle'):
    """Build a finite-size 2D lattice cluster of the requested shape.

    Args:
        a1, a2: array-like (2,) -- primitive lattice vectors.
        N1, N2: int             -- grid repetitions (shape-dependent meaning,
                                   see module docstring for per-shape semantics).
        shape:  str             -- one of 'circle', 'hexagon', 'rectangle',
                                   'triangle', 'parallelogram', 'ellipse'.

    Returns:
        (N, 2) float64 ndarray -- particle positions with CM at origin.

    Raises:
        ValueError: if shape is not recognised, or if shape constraints
                    are violated (e.g. parallelogram with even sqrt(N)).
    """
    if shape not in _SHAPE_FUNCS:
        raise ValueError(
            "Unknown shape '%s'.  Supported: %s."
            % (shape, ', '.join(sorted(_SHAPE_FUNCS)))
        )
    a1 = np.asarray(a1, dtype=np.float64)
    a2 = np.asarray(a2, dtype=np.float64)
    return _SHAPE_FUNCS[shape](a1, a2, int(N1), int(N2))


# ============================================================
# I/O
# ============================================================

def save_cluster(pos, filename):
    """Save cluster positions as a .npy file.

    Args:
        pos:      (N, 2) array-like -- particle positions.
        filename: str               -- output path (extension added by np.save).
    """
    np.save(filename, np.asarray(pos, dtype=np.float64))


def load_cluster(filename, angle_deg=0):
    """Load cluster from .npy file, optionally rotate, then re-centre CM.

    Args:
        filename:  str   -- path to .npy file (with or without '.npy').
        angle_deg: float -- ACW rotation applied before recentring (degrees).

    Returns:
        (N, 2) float64 ndarray, CM at origin.
    """
    pos = np.load(filename).astype(np.float64)
    if angle_deg != 0.0:
        pos = rotate(pos, angle_deg)
    pos -= np.mean(pos, axis=0)
    return pos


def save_xyz(pos, filename, elem='X'):
    """Write an XYZ file for visualisation (z coordinate set to zero).

    Args:
        pos:      (N, 2) array-like -- particle positions.
        filename: str               -- output path.
        elem:     str               -- element symbol written in each line.
    """
    pos = np.asarray(pos, dtype=np.float64)
    N   = pos.shape[0]
    with open(filename, 'w') as f:
        f.write('%d\n#\n' % N)
        for i in range(N):
            f.write('%s %20.15f %20.15f %20.15f\n' % (elem, pos[i, 0], pos[i, 1], 0.0))


# ============================================================
# Utilities
# ============================================================

def add_basis(lat_pos, basis):
    """Tile a Bravais lattice with a crystal basis.

    Each lattice point is displaced by every basis vector; the results are
    flattened into a single (N*M, 2) array.

    Args:
        lat_pos: (N, 2) array-like -- Bravais lattice positions.
        basis:   (M, 2) array-like -- basis site offsets relative to each
                                      lattice point.

    Returns:
        (N*M, 2) float64 ndarray.
    """
    lat_pos = np.asarray(lat_pos, dtype=np.float64)
    basis   = np.asarray(basis,   dtype=np.float64)
    # Broadcasting: (N, 1, 2) + (M, 2) -> (N, M, 2) -> (N*M, 2)
    expanded = lat_pos[:, np.newaxis, :] + basis
    return expanded.reshape(expanded.shape[0] * expanded.shape[1], 2)


def calc_cluster_langevin(eta, pos):
    """Effective translational and rotational damping for overdamped dynamics.

    For a cluster of N identical particles with friction coefficient eta:
        etat_eff = eta * N               (translational, scales as N)
        etar_eff = eta * sum_i |r_i|^2  (rotational, varies with shape)

    Args:
        eta: float        -- single-particle friction coefficient.
        pos: (N, 2) ndarray -- particle positions (CM at origin).

    Returns:
        (etat_eff, etar_eff): floats.
    """
    pos      = np.asarray(pos, dtype=np.float64)
    N        = pos.shape[0]
    etat_eff = eta * N
    etar_eff = eta * np.sum(pos * pos)
    return etat_eff, etar_eff


# ============================================================
# From parameter dictionary
# ============================================================

def cluster_from_params(params):
    """Build a cluster from a parameter dictionary (for JSON input files).

    Required keys:
        a1, a2:         primitive lattice vectors.
        N1, N2:         grid repetitions.
        cluster_shape:  shape string as in make_cluster, plus 'polygon'.

    Optional keys:
        cl_basis:  list of (2,) offsets for a multi-site basis
                   (default [[0,0]] -- single-site Bravais lattice).
        cl_poly:   vertex list for polygon masking (required if shape='polygon').
        direction: polygon masking direction -- 0 interior, 1 exterior.
        theta:     float, rotation in degrees applied after cluster creation.
        (ellipse semi-axes are N1*|a1| and N2*|a2|; no extra keys needed.)

    Returns:
        (N, 2) float64 ndarray, CM at origin.
    """
    a1    = np.array(params['a1'], dtype=np.float64)
    a2    = np.array(params['a2'], dtype=np.float64)
    N1    = params['N1']
    N2    = params['N2']
    shape = params['cluster_shape']

    if shape == 'polygon':
        poly      = get_poly(params['cl_poly'])
        direction = params.get('direction', 0)
        return cluster_poly(poly, params, direction)

    pos = make_cluster(a1, a2, N1, N2, shape=shape)

    basis = np.array(params.get('cl_basis', [[0.0, 0.0]]), dtype=np.float64)
    pos   = add_basis(pos, basis)
    # Re-centre: add_basis may shift CM if basis is asymmetric.
    pos  -= np.mean(pos, axis=0)

    # Leave outside, you risk applying multiple rotation
    #if 'theta' in params:
    #    pos = rotate(pos, float(params['theta']))

    return pos


# ============================================================
# Polygon masking (Shapely optional dependency)
# ============================================================

def get_poly(points, scale=1, tho=0, c=(0, 0), shift=0, cm=False):
    """Build a Shapely Polygon from a list of vertices.

    Optionally scale, rotate, and shift the polygon before returning.

    Args:
        points: array-like (M, 2) -- polygon vertices.
        scale:  float             -- uniform scale factor.
        tho:    float             -- rotation angle in degrees (ACW).
        c:      (2,)              -- rotation centre.
        shift:  float or (2,)    -- translation applied after rotation.
        cm:     bool              -- if True, translate vertices so their
                                    mean is at the origin before rotating.

    Returns:
        shapely.geometry.Polygon
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError(
            "Shapely is required for polygon masking. "
            "Install with: pip install shapely"
        )
    pts = scale * np.asarray(points, dtype=np.float64)
    if cm:
        pts -= np.mean(pts, axis=0)
    pts  = rotate(pts, tho, c)
    pts += np.asarray(shift, dtype=np.float64)
    return Polygon(pts)


def cluster_poly(polygon, params, direction=0):
    """Mask a lattice with a Shapely polygon.

    Args:
        polygon:   shapely.geometry.Polygon -- masking region.
        params:    dict -- cluster parameters (see cluster_from_params).
        direction: int  -- 0 = keep interior, 1 = keep exterior.

    Returns:
        (N, 2) float64 ndarray.
    """
    try:
        from shapely.geometry import MultiPoint
    except ImportError:
        raise ImportError(
            "Shapely is required for polygon masking. "
            "Install with: pip install shapely"
        )
    # Build the background grid.  If not specified, use rectangle large enough
    # to cover the polygon bounding box.
    working_params = dict(params)
    if 'masked_shape' not in working_params:
        working_params['cluster_shape'] = 'rectangle'
    else:
        working_params['cluster_shape'] = working_params['masked_shape']

    if direction == 0:
        a1 = np.array(params['a1'])
        a2 = np.array(params['a2'])
        l  = max(np.linalg.norm(a1), np.linalg.norm(a2))
        Nl = 3 * int(np.ceil(np.max(np.abs(polygon.bounds)) / l))
        working_params['N1'] = Nl
        working_params['N2'] = Nl

    pos = cluster_from_params(working_params)

    mpts = MultiPoint(pos)
    mask = np.array(
        [int(polygon.contains(pt)) - bool(direction) for pt in mpts.geoms],
        dtype=bool,
    )
    return pos[mask]


# ============================================================
# POSCAR / ASE loader (ASE optional dependency)
# ============================================================

def params_from_poscar(filename, cut_z=0, tol=0.9):
    """Extract lattice parameters from a POSCAR file via ASE.

    Reads the primitive vectors and atomic basis from a 2D slab POSCAR.
    The returned dict can be passed directly to cluster_from_params to
    create a cluster of any shape with the correct lattice geometry.

    The POSCAR must have the 2D periodicity in the first two lattice
    vectors (a, b); the third vector c must be along z. All z information
    is discarded -- the geometry is projected to the (x, y) plane.
    If the slab has multiple layers, use cut_z to select one.

    Args:
        filename: str   -- path to POSCAR file.
        cut_z:    float -- keep only atoms with z < cut_z. If 0 (default),
                          uses one standard deviation of z as threshold.
        tol:      float -- fractional tolerance on cut_z (default 0.9).

    Returns:
        dict with keys: 'a1', 'a2', 'cl_basis'
        pos_z: (M,) array of z coordinates of discarded atoms (for inspection).

    Raises:
        ImportError:  if ASE is not installed.
        RuntimeError: if no atoms pass the z filter.
    """
    try:
        import ase.io
    except ImportError:
        raise ImportError(
            "ASE is required for params_from_poscar. "
            "Install with: pip install ase"
        )

    geom  = ase.io.read(filename)
    cell  = geom.cell[:]

    # Primitive vectors from the POSCAR cell (rows of cell matrix).
    a1 = cell[0, :2].tolist()
    a2 = cell[1, :2].tolist()

    # Select atoms in the target layer by z threshold.
    pos_z = geom.positions[:, 2]
    if cut_z == 0:
        cut_z = float(np.mean(pos_z) - np.std(pos_z))
    mask  = pos_z < cut_z * tol
    pos2d = geom.positions[mask, :2]
    if len(pos2d) == 0:
        raise RuntimeError(
            "No atoms selected with cut_z=%.4g, tol=%.4g. "
            "Check the z coordinates of your slab." % (cut_z, tol)
        )

    # Express basis in fractional coordinates of a1, a2, then back to
    # Cartesian -- this ensures basis vectors are within the unit cell.
    L     = np.array([a1, a2], dtype=np.float64)
    frac  = np.linalg.solve(L.T, pos2d.T).T
    frac -= np.floor(frac)           # fold into [0, 1)
    basis = (frac @ L).tolist()

    params = {'a1': a1, 'a2': a2, 'cl_basis': basis}
    pos_z_ignored = geom.positions[~mask, 2]
    return params, pos_z_ignored
