"""
Plotting utilities for FLAKE.

Public API
----------
    plot_UC              -- draw the real-space unit cell on a matplotlib axis.
    get_brillouin_zone_2d -- compute BZ vertices via Voronoi decomposition.
    plot_BZ2d            -- add BZ polygon patch to a matplotlib axis.
    plt_cosmetic         -- set axis labels, zero lines, and equal aspect.
    plot_lattice_vectors -- draw primitive lattice vectors as arrows.
"""

import logging
import numpy as np

_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())


def plot_UC(ax, u, params=None):
    """Draw the real-space unit cell spanned by u[0], u[1] on ax.

    Args:
        ax:     matplotlib Axes.
        u:      (2, 2) array-like -- two primitive lattice vectors.
        params: dict -- line style kwargs forwarded to ax.plot.

    Returns:
        ax
    """
    if params is None:
        params = {'ls': ':', 'color': 'tab:gray', 'lw': 1}
    BZ_corner = np.array([(n*u[0] + m*u[1])
                          for n, m in [[0,0], [1,0], [1,1], [0,1], [0,0]]])
    for i in range(4):
        ax.plot([BZ_corner[i+1, 0], BZ_corner[i, 0]],
                [BZ_corner[i+1, 1], BZ_corner[i, 1]], **params)
    return ax


def get_brillouin_zone_2d(cell):
    """Compute the vertices of the 2D Brillouin Zone (Wigner-Seitz cell of reciprocal lattice).

    Uses Voronoi decomposition of the reciprocal lattice.

    Args:
        cell: (2, 2) array-like -- rows are the two reciprocal lattice vectors.

    Returns:
        (M, 2) float64 ndarray -- vertices of the BZ polygon.
    """
    from scipy.spatial import Voronoi

    cell = np.asarray(cell, dtype=float)
    if cell.shape != (2, 2):
        raise ValueError("cell must have shape (2, 2); got %s" % (cell.shape,))

    nn = np.array([i*cell[0] + j*cell[1]
                   for j in range(-1, 2) for i in range(-1, 2)])
    vor = Voronoi(nn)
    # origin is index 4 (i=0, j=0 in the 3x3 grid)
    orig_region = vor.regions[vor.point_region[4]]
    return vor.vertices[orig_region]


def plot_BZ2d(ax, ws_verts, params=None):
    """Add the Brillouin Zone polygon to a matplotlib axis.

    Args:
        ax:       matplotlib Axes.
        ws_verts: (M, 2) array-like -- BZ vertices (e.g. from get_brillouin_zone_2d).
        params:   dict -- Polygon kwargs (ls, color, lw, fill, ...).

    Returns:
        (ax, ws_cell) -- axis and the Polygon patch.
    """
    if params is None:
        params = {'ls': '--', 'color': 'tab:gray', 'lw': 1, 'fill': False}
    from matplotlib.patches import Polygon

    ws_cell = Polygon(ws_verts, **params)
    ax.add_patch(ws_cell)
    ax.set_aspect('equal')
    return ax, ws_cell


def plt_cosmetic(ax, xlabel='x', ylabel='y'):
    """Add zero lines, axis labels and equal aspect to ax.

    Args:
        ax:     matplotlib Axes.
        xlabel: str -- x-axis label.
        ylabel: str -- y-axis label.
    """
    ax.axhline(color='gray', ls=':', lw=1)
    ax.axvline(color='gray', ls=':', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')


def plot_lattice_vectors(ax, S, colors=('tab:red', 'tab:orange'), labels=('b1', 'b2')):
    """Draw the two primitive lattice vectors as arrows from the origin.

    Args:
        ax:     matplotlib Axes.
        S:      (2, 2) array-like -- rows are the two lattice vectors.
        colors: tuple of str -- arrow colours.
        labels: tuple of str -- legend labels.

    Returns:
        ax
    """
    for vec, color, label in zip(S, colors, labels):
        ax.quiver(0, 0, vec[0], vec[1],
                  angles='xy', scale_units='xy', scale=1,
                  zorder=5, color=color, label=label)
    return ax
